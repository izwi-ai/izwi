//! Compact E4M3FN projection. CUDA uses software FP8 decoding, warp-cooperative
//! decode and 16-bit Tensor Core prefill with F32 block accumulation. No expanded
//! persistent weight tensor or per-step full-matrix conversion is created.
use candle_core::{CpuStorage, CustomOp3, DType, Device, Layout, Result, Shape, Tensor};

pub const PROVIDER: &str = "block-e4m3fn-software-decode-f16-bf16-mma";

/// A capability test, not a performance recommendation.
pub fn provider_supported(device: &Device, dtype: DType, n: usize, k: usize) -> bool {
    if !matches!(dtype, DType::F16 | DType::BF16)
        || n == 0
        || k == 0
        || !n.is_multiple_of(64)
        || !k.is_multiple_of(128)
        || n.checked_mul(k).is_none_or(|v| v > i32::MAX as usize)
    {
        return false;
    }
    #[cfg(feature = "cuda")]
    if let Ok(device) = device.as_cuda_device() {
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
        // Immutable per-Candle-device capability cache. Projection calls and
        // graph capture must not issue a driver capability query on every layer.
        use std::{collections::VecDeque, sync::Mutex};
        static CAPABILITIES: Mutex<VecDeque<(candle_core::cuda_backend::DeviceId, bool)>> =
            Mutex::new(VecDeque::new());
        let mut cache = CAPABILITIES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, supported)) = cache.iter().find(|(id, _)| *id == device.id()) {
            return *supported;
        }
        let supported = device
            .cuda_stream()
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .is_ok_and(|v| v >= 8);
        if cache.len() >= 64 {
            cache.pop_front();
        }
        cache.push_back((device.id(), supported));
        return supported;
    }
    let _ = device;
    false
}

/// Until exact-device measurements establish a winning route, Auto retains Q8.
/// Explicit native-FP8 selection can use the structurally supported provider.
pub fn provider_auto_preferred(_device: &Device, _dtype: DType, _n: usize, _k: usize) -> bool {
    false
}

pub fn block_fp8_projection(input: &Tensor, weights: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let (n, k) = weights.dims2()?;
    if input.rank() == 0
        || input.dim(candle_core::D::Minus1)? != k
        || k == 0
        || n == 0
        || weights.dtype() != DType::U8
        || scales.dtype() != DType::F32
        || scales.dims() != [n.div_ceil(128), k.div_ceil(128)]
        || !input.device().same_device(weights.device())
        || !input.device().same_device(scales.device())
        || !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        candle_core::bail!("invalid compact E4M3FN projection contract")
    }
    if input.device().is_cuda() && !provider_supported(input.device(), input.dtype(), n, k) {
        candle_core::bail!("unsupported compact FP8 CUDA provider geometry or dtype")
    }
    input.contiguous()?.apply_op3_no_bwd(
        &weights.contiguous()?,
        &scales.contiguous()?,
        &Projection { n, k },
    )
}

pub(crate) fn decode_e4m3fn(b: u8) -> f32 {
    let e = (b >> 3) & 15;
    let m = b & 7;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if b & 128 != 0 {
        -v
    } else {
        v
    }
}
struct Projection {
    n: usize,
    k: usize,
}
impl CustomOp3 for Projection {
    fn name(&self) -> &'static str {
        "qwen38-block-fp8-projection"
    }
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        xl: &Layout,
        w: &CpuStorage,
        wl: &Layout,
        s: &CpuStorage,
        sl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::U8(w) = w else {
            candle_core::bail!("FP8 bytes required")
        };
        let CpuStorage::F32(s) = s else {
            candle_core::bail!("FP8 F32 scales required")
        };
        let dtype = match x {
            CpuStorage::F16(_) => DType::F16,
            CpuStorage::BF16(_) => DType::BF16,
            _ => DType::F32,
        };
        let x = match x {
            CpuStorage::F32(v) => v.clone(),
            CpuStorage::F16(v) => v.iter().map(|v| v.to_f32()).collect(),
            CpuStorage::BF16(v) => v.iter().map(|v| v.to_f32()).collect(),
            _ => candle_core::bail!("FP8 activation dtype"),
        };
        let (xs, xe) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("contiguous FP8 input required".into()))?;
        let m = (xe - xs) / self.k;
        let mut y = vec![0f32; m * self.n];
        for row in 0..m {
            for n in 0..self.n {
                let mut total = 0f32;
                for kb in (0..self.k).step_by(128) {
                    let mut part = 0f32;
                    for k in kb..(kb + 128).min(self.k) {
                        part = x[xs + row * self.k + k]
                            .mul_add(decode_e4m3fn(w[wl.start_offset() + n * self.k + k]), part);
                    }
                    total = part.mul_add(
                        s[sl.start_offset() + (n / 128) * self.k.div_ceil(128) + kb / 128],
                        total,
                    );
                }
                y[row * self.n + n] = total;
            }
        }
        let mut dims = xl.dims().to_vec();
        *dims.last_mut().unwrap() = self.n;
        let storage = match dtype {
            DType::F16 => CpuStorage::F16(y.into_iter().map(half::f16::from_f32).collect()),
            DType::BF16 => CpuStorage::BF16(y.into_iter().map(half::bf16::from_f32).collect()),
            _ => CpuStorage::F32(y),
        };
        Ok((storage, Shape::from(dims)))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        xl: &Layout,
        w: &candle_core::CudaStorage,
        wl: &Layout,
        s: &candle_core::CudaStorage,
        sl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
        let device = x.device();
        let m = xl.shape().elem_count() / self.k;
        if m == 0 || m.checked_mul(self.n).is_none_or(|v| v > i32::MAX as usize) {
            candle_core::bail!("FP8 output size overflow")
        }
        let CudaStorageSlice::U8(w) = &w.slice else {
            candle_core::bail!("FP8 bytes required")
        };
        let CudaStorageSlice::F32(s) = &s.slice else {
            candle_core::bail!("FP8 scales required")
        };
        let w = w.slice(wl.start_offset()..wl.start_offset() + self.n * self.k);
        let s = s.slice(
            sl.start_offset()..sl.start_offset() + self.n.div_ceil(128) * self.k.div_ceil(128),
        );
        macro_rules! run {
            ($v:ident,$ty:ty,$mv:literal,$mm:literal) => {{
                let CudaStorageSlice::$v(x) = &x.slice else {
                    unreachable!()
                };
                let x = x.slice(xl.start_offset()..xl.start_offset() + m * self.k);
                // SAFETY: every output element is written by the kernel.
                let mut out = unsafe { device.alloc::<$ty>(m * self.n)? };
                let small = m <= 4;
                let f = device.get_or_load_custom_func(
                    if small { $mv } else { $mm },
                    "izwi_qwen38_fp8",
                    super::cuda_ptx::FP8,
                )?;
                let mut b = f.builder();
                b.arg(&x);
                b.arg(&w);
                b.arg(&s);
                b.arg(&mut out);
                candle_core::builder_arg!(b, m as i32, self.n as i32, self.k as i32);
                let cfg = if small {
                    LaunchConfig {
                        grid_dim: (self.n.div_ceil(8) as u32, m.div_ceil(4) as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }
                } else {
                    LaunchConfig {
                        grid_dim: (self.n.div_ceil(64) as u32, m.div_ceil(16) as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    }
                };
                // SAFETY: compact bytes, block-scale grid and dimensions have been validated.
                unsafe { b.launch(cfg) }.w()?;
                CudaStorageSlice::$v(out)
            }};
        }
        let slice = match &x.slice {
            CudaStorageSlice::F16(_) => {
                run!(F16, half::f16, "qwen38_fp8_mv_f16", "qwen38_fp8_mm_f16")
            }
            CudaStorageSlice::BF16(_) => {
                run!(BF16, half::bf16, "qwen38_fp8_mv_bf16", "qwen38_fp8_mm_bf16")
            }
            _ => candle_core::bail!("FP8 CUDA activation requires F16/BF16"),
        };
        let mut dims = xl.dims().to_vec();
        *dims.last_mut().unwrap() = self.n;
        Ok((
            candle_core::CudaStorage {
                slice,
                device: device.clone(),
            },
            Shape::from(dims),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_e4m3fn_encodings_preserve_finite_range_subnormals_and_sign() {
        for b in 0u16..256 {
            let bits = b as u8;
            let exponent = (bits >> 3) & 15;
            let mantissa = bits & 7;
            let actual = decode_e4m3fn(bits);
            if bits & 127 == 127 {
                assert!(actual.is_nan());
                continue;
            }
            let magnitude = if exponent == 0 {
                mantissa as f32 / 512.
            } else {
                (8 + mantissa) as f32 * 2f32.powi(exponent as i32 - 10)
            };
            assert_eq!(actual.abs(), magnitude);
            assert_eq!(actual.is_sign_negative(), bits & 128 != 0);
        }
        assert_eq!(decode_e4m3fn(0x7e), 448.);
        assert_eq!(decode_e4m3fn(1), 1. / 512.);
    }
    #[test]
    fn projection_preserves_dtype_and_exact_scales_across_both_block_axes() {
        let device = Device::Cpu;
        let (n, k) = (129, 129);
        let w = Tensor::from_vec(vec![0x38u8; n * k], (n, k), &device).unwrap();
        let scales = Tensor::from_vec(vec![2f32, 3., 5., 7.], (2, 2), &device).unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for m in [1, 3, 17] {
                let mut x = vec![0f32; m * k];
                for row in 0..m {
                    x[row * k + 127] = 1.;
                    x[row * k + 128] = 2.;
                }
                let x = Tensor::from_vec(x, (m, k), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let out = block_fp8_projection(&x, &w, &scales).unwrap();
                assert_eq!(out.dtype(), dtype);
                assert_eq!(out.dims(), [m, n]);
                let values = out.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
                for row in values {
                    assert_eq!(row[0], 8.);
                    assert_eq!(row[127], 8.);
                    assert_eq!(row[128], 19.);
                }
            }
        }
    }
    #[test]
    fn projection_offsets_and_nontrivial_weights_match_dense_candle() {
        let (n, k) = (64, 256);
        let dev = Device::Cpu;
        let raw = (0..n * k)
            .map(|i| {
                if i % 3 == 0 {
                    0xb0u8
                } else if i % 3 == 1 {
                    0x38
                } else {
                    0x40
                }
            })
            .collect::<Vec<_>>();
        let scales = vec![0.5f32, 2.];
        let dense = raw
            .iter()
            .enumerate()
            .map(|(i, &b)| decode_e4m3fn(b) * scales[(i % k) / 128])
            .collect::<Vec<_>>();
        let w = Tensor::from_vec(raw, (n, k), &dev).unwrap();
        let s = Tensor::from_vec(scales, (1, 2), &dev).unwrap();
        let d = Tensor::from_vec(dense, (n, k), &dev).unwrap();
        let x = Tensor::from_vec(
            (0..4 * k)
                .map(|i| (i % 7) as f32 * 0.125 - 0.5)
                .collect::<Vec<_>>(),
            (4, k),
            &dev,
        )
        .unwrap()
        .narrow(0, 1, 3)
        .unwrap();
        let y = block_fp8_projection(&x, &w, &s)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let reference = x.matmul(&d.t().unwrap()).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(y, reference);
    }
    #[test]
    fn malformed_scales_fail_and_auto_never_substitutes_an_unmeasured_kernel() {
        assert!(!provider_supported(&Device::Cpu, DType::BF16, 512, 512));
        assert!(!provider_auto_preferred(
            &Device::Cpu,
            DType::BF16,
            512,
            512
        ));
        let x = Tensor::zeros((1, 128), DType::F32, &Device::Cpu).unwrap();
        let w = Tensor::zeros((128, 128), DType::U8, &Device::Cpu).unwrap();
        let s = Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap();
        assert!(block_fp8_projection(&x, &w, &s).is_err());
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_projection_decode_verify_prefill_match_portable_reference() {
        let Some(device) = super::super::cuda_test_device() else {
            return;
        };
        for dtype in [DType::F16, DType::BF16] {
            assert!(
                provider_supported(&device, dtype, 192, 256),
                "SM80 test geometry must be supported"
            );
            for m in [1, 2, 4, 17, 32] {
                let raw = (0..192 * 256)
                    .map(|i| {
                        if i % 3 == 0 {
                            0xb0u8
                        } else if i % 3 == 1 {
                            0x38
                        } else {
                            0x40
                        }
                    })
                    .collect::<Vec<_>>();
                let w = Tensor::from_vec(raw, (192, 256), &Device::Cpu).unwrap();
                let s =
                    Tensor::from_vec(vec![0.25f32, 1.5, 2., 0.125], (2, 2), &Device::Cpu).unwrap();
                let x = Tensor::from_vec(
                    (0..m * 256)
                        .map(|i| (i % 7) as f32 * 0.125 - 0.5)
                        .collect::<Vec<_>>(),
                    (m, 256),
                    &Device::Cpu,
                )
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
                let expected = block_fp8_projection(&x, &w, &s)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let actual = block_fp8_projection(
                    &x.to_device(&device).unwrap(),
                    &w.to_device(&device).unwrap(),
                    &s.to_device(&device).unwrap(),
                )
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                for (a, e) in actual.iter().zip(expected) {
                    assert!(
                        (a - e).abs() <= 0.03 + e.abs() * 0.008,
                        "{dtype:?} M={m}: {a} != {e}"
                    );
                }
            }
        }
    }
}
