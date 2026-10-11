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
    if device.is_metal() {
        // `kernels::metal_qwen36moe` decodes E4M3 in software on any Apple
        // GPU; the Metal plan runs F16 (or F32) activations.
        return cfg!(feature = "metal") && matches!(dtype, DType::F16);
    }
    device_is_sm80_or_newer(device)
}

/// Whether `device` is a CUDA device of compute capability 8.0 or newer (the
/// floor the repo's PTX targets). Cached per Candle device; false off CUDA.
pub(crate) fn device_is_sm80_or_newer(device: &Device) -> bool {
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

/// No device-independent recommendation: a family that defaults to the
/// compact provider gates it with its own device checks (Qwen3.8 does so in
/// its residency policy, with [`verify_decode_kernels_at`] and a real-weight
/// probe). Explicit native-FP8 selection can use the supported provider.
pub fn provider_auto_preferred(_device: &Device, _dtype: DType, _n: usize, _k: usize) -> bool {
    false
}

/// Process-wide switch for the vectorized decode GEMV (`qwen38_fp8_mv2_*`).
/// Off until [`verify_and_enable_vectorized_decode`] has compared it against
/// the reference decode kernel on a real device; projections keep the
/// reference kernel otherwise.
static VECTORIZED_DECODE_VERIFIED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn vectorized_decode_enabled() -> bool {
    VECTORIZED_DECODE_VERIFIED.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeKernel {
    /// Vectorized GEMV once verified, else the reference kernel.
    Auto,
    /// Always the reference byte-wise decode kernel.
    Reference,
    /// The vectorized GEMV (errors if the call is not eligible).
    Vectorized,
    /// The vectorized GEMV for an eligible call (at most four aligned rows),
    /// the reference kernels otherwise. The caller owns the verification, so
    /// the process-wide flag is neither read nor set.
    VectorizedWhenEligible,
}

/// Compare the vectorized decode GEMV against the reference kernel on
/// `device` (decode row counts 1..=4, trunk-like K) and enable it for every
/// compact-FP8 projection in the process on success. Idempotent.
pub fn verify_and_enable_vectorized_decode(device: &Device) -> Result<()> {
    if vectorized_decode_enabled() {
        return Ok(());
    }
    verify_vectorized_decode(device)?;
    VECTORIZED_DECODE_VERIFIED.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// The check behind [`verify_and_enable_vectorized_decode`], without touching
/// the process-wide flag: a family that selects the kernel per call (through
/// [`block_fp8_projection_selecting`]) keeps its own verdict.
pub fn verify_vectorized_decode(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        candle_core::bail!("the vectorized FP8 decode GEMV runs on CUDA only")
    }
    for (n, k) in [(256usize, 2048usize), (64, 512), (192, 4096)] {
        let raw = (0..n * k)
            .map(|i| {
                let mixed = (i.wrapping_mul(0x9E37_79B1) >> 17) as u8;
                if mixed & 0x7f == 0x7f {
                    0x38
                } else {
                    mixed
                }
            })
            .collect::<Vec<_>>();
        let scales = (0..n.div_ceil(128) * k.div_ceil(128))
            .map(|i| 0.001 + (i % 13) as f32 * 3e-4)
            .collect::<Vec<_>>();
        let w = Tensor::from_vec(raw, (n, k), &Device::Cpu)?.to_device(device)?;
        let s = Tensor::from_vec(scales, (n.div_ceil(128), k.div_ceil(128)), &Device::Cpu)?
            .to_device(device)?;
        for m in 1..=4usize {
            for dtype in [DType::BF16, DType::F16] {
                let x = Tensor::from_vec(
                    (0..m * k)
                        .map(|i| ((i as f32) * 0.754_877_7).sin() * 2.0)
                        .collect::<Vec<_>>(),
                    (m, k),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?
                .to_device(device)?;
                let host = |t: Tensor| -> Result<Vec<f32>> {
                    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
                };
                let reference = host(project(&x, &w, &s, DecodeKernel::Reference)?)?;
                let vectorized = host(project(&x, &w, &s, DecodeKernel::Vectorized)?)?;
                let scale = reference.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
                if let Some((index, (a, b))) = vectorized
                    .iter()
                    .zip(&reference)
                    .enumerate()
                    .find(|(_, (a, b))| !a.is_finite() || (*a - *b).abs() > 0.02 * scale)
                {
                    candle_core::bail!(
                        "vectorized FP8 decode GEMV diverges at n={n} k={k} m={m} {dtype:?} index {index}: {a} vs reference {b}"
                    )
                }
            }
        }
    }
    Ok(())
}

/// Outcome of [`verify_decode_kernels_at`]: the reference kernels (`mv` for
/// up to four rows, `mm` above) must pass for the call to succeed; the
/// vectorized GEMV's verdict is reported separately so a caller can keep the
/// reference decode when only the vectorized kernel disagrees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeKernelGate {
    pub vectorized: std::result::Result<(), String>,
}

/// Compare every compact-FP8 kernel on `device` with the portable reference
/// at a caller-chosen inner dimension `k` (the checkpoint's real K values) and
/// `n_rows` synthetic output channels: the vectorized GEMV and the reference
/// GEMV at one to four activation rows, the prefill kernel at 5, 8 and 17.
/// Weights mix every finite E4M3 byte and every 128x128 block has its own
/// scale; activations are rounded to `dtype` before the host reference runs,
/// so the comparison sees only kernel error. Tolerance: 1% relative L2 and 2%
/// of the largest reference magnitude per element (NaN-safe). Sets no flag.
pub fn verify_decode_kernels_at(
    device: &Device,
    dtype: DType,
    n_rows: usize,
    k: usize,
) -> Result<DecodeKernelGate> {
    if n_rows == 0 || k == 0 {
        candle_core::bail!("FP8 kernel gate needs a non-empty geometry")
    }
    let raw = (0..n_rows * k)
        .map(|i| {
            let mixed = (i.wrapping_mul(0x9E37_79B1) >> 17) as u8;
            if mixed & 0x7f == 0x7f {
                mixed ^ 0x41
            } else {
                mixed
            }
        })
        .collect::<Vec<_>>();
    let scale_shape = (n_rows.div_ceil(128), k.div_ceil(128));
    let scales = (0..scale_shape.0 * scale_shape.1)
        .map(|i| 1e-3 * (1.0 + (i % 17) as f32 * 0.37))
        .collect::<Vec<_>>();
    let host_w = Tensor::from_vec(raw, (n_rows, k), &Device::Cpu)?;
    let host_s = Tensor::from_vec(scales, scale_shape, &Device::Cpu)?;
    let w = host_w.to_device(device)?;
    let s = host_s.to_device(device)?;
    let host = |t: Tensor| -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    let compare = |label: &str,
                   actual: &[f32],
                   expected: &[f32]|
     -> std::result::Result<(), String> {
        let mut err = 0f64;
        let mut norm = 0f64;
        let mut max_ref = 0f32;
        let mut max_err = 0f32;
        for (a, e) in actual.iter().zip(expected) {
            let diff = (a - e).abs();
            let diff = if diff.is_finite() {
                diff
            } else {
                f32::INFINITY
            };
            err += f64::from(diff).powi(2);
            norm += f64::from(*e).powi(2);
            max_ref = max_ref.max(e.abs());
            max_err = max_err.max(diff);
        }
        let relative = (err / norm.max(f64::MIN_POSITIVE)).sqrt();
        if actual.len() != expected.len()
            || !relative.is_finite()
            || relative > 0.01
            || max_err > 0.02 * max_ref.max(f32::MIN_POSITIVE)
        {
            return Err(format!(
                "{label} at n={n_rows} k={k} {dtype:?}: relative L2 {relative:.4}, max error {max_err} vs max |ref| {max_ref}"
            ));
        }
        Ok(())
    };
    let mut vectorized = Ok(());
    for m in [1usize, 2, 3, 4, 5, 8, 17] {
        let x = Tensor::from_vec(
            (0..m * k)
                .map(|i| ((i as f32 + 0.5) * 0.754_877_7).sin() * 2.0)
                .collect::<Vec<_>>(),
            (m, k),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?;
        let expected = host(project(
            &x.to_dtype(DType::F32)?,
            &host_w,
            &host_s,
            DecodeKernel::Reference,
        )?)?;
        let x = x.to_device(device)?;
        let reference = host(project(&x, &w, &s, DecodeKernel::Reference)?)?;
        let kernel = if m <= 4 {
            "reference GEMV"
        } else {
            "prefill GEMM"
        };
        compare(&format!("FP8 {kernel} m={m}"), &reference, &expected)
            .map_err(candle_core::Error::Msg)?;
        if m <= 4 && vectorized.is_ok() {
            vectorized = match project(&x, &w, &s, DecodeKernel::Vectorized).and_then(host) {
                Ok(actual) => compare(&format!("FP8 vectorized GEMV m={m}"), &actual, &expected),
                Err(error) => Err(format!("FP8 vectorized GEMV m={m}: {error}")),
            };
        }
    }
    Ok(DecodeKernelGate { vectorized })
}

pub fn block_fp8_projection(input: &Tensor, weights: &Tensor, scales: &Tensor) -> Result<Tensor> {
    project(input, weights, scales, DecodeKernel::Auto)
}

/// [`block_fp8_projection`] with the decode kernel chosen per call instead of
/// by the process-wide flag: `vectorized` runs the vectorized GEMV on eligible
/// calls (at most four aligned rows) and the reference kernels otherwise;
/// `false` always runs the reference kernels.
pub fn block_fp8_projection_selecting(
    input: &Tensor,
    weights: &Tensor,
    scales: &Tensor,
    vectorized: bool,
) -> Result<Tensor> {
    project(
        input,
        weights,
        scales,
        if vectorized {
            DecodeKernel::VectorizedWhenEligible
        } else {
            DecodeKernel::Reference
        },
    )
}

fn project(
    input: &Tensor,
    weights: &Tensor,
    scales: &Tensor,
    kernel: DecodeKernel,
) -> Result<Tensor> {
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
        &Projection { n, k, kernel },
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

/// Largest finite E4M3FN magnitude.
pub(crate) const E4M3FN_MAX: f32 = 448.0;

/// The nearest E4M3FN byte to `value`, ties to even, saturating at ±448; NaN
/// maps to the NaN byte. Inverse of [`decode_e4m3fn`] on every finite byte.
pub(crate) fn encode_e4m3fn(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude >= E4M3FN_MAX {
        return sign | 0x7e;
    }
    if magnitude < 2f32.powi(-6) {
        // Subnormals are multiples of 2^-9; rounding up to 8 of them yields
        // the smallest normal, 0x08, which is the same bit pattern.
        return sign | (magnitude * 512.0).round_ties_even() as u8;
    }
    // A normal f32 here; scaling by a power of two is exact.
    let exponent = ((magnitude.to_bits() >> 23) & 0xff) as i32 - 127;
    let mut steps = (magnitude * 2f32.powi(3 - exponent)).round_ties_even() as u32;
    let mut biased = exponent + 7;
    if steps == 16 {
        steps = 8;
        biased += 1;
    }
    // `magnitude < 448` keeps the result at or below 0x7e.
    sign | ((biased as u8) << 3) | (steps - 8) as u8
}

/// Block-FP8 quantization in the checkpoints' own layout: row-major `[n, k]`
/// E4M3FN bytes and one F32 scale per 128×128 block, `[ceil(n/128),
/// ceil(k/128)]`, so `value ≈ decode(byte) * scale`. Each scale is
/// `amax / 448`, mapping the block's largest magnitude to ±448; an all-zero
/// block gets scale 1. Row blocks quantize in parallel.
pub(crate) fn quantize_block_fp8<T: Copy + Sync>(
    values: &[T],
    n: usize,
    k: usize,
    to_f32: impl Fn(T) -> f32 + Sync,
) -> Result<(Vec<u8>, Vec<f32>)> {
    use rayon::prelude::*;

    if n == 0 || k == 0 || n.checked_mul(k) != Some(values.len()) {
        candle_core::bail!(
            "block-FP8 quantization of {} values as [{n}, {k}]",
            values.len()
        )
    }
    let column_blocks = k.div_ceil(128);
    let mut bytes = vec![0u8; n * k];
    let mut scales = vec![0f32; n.div_ceil(128) * column_blocks];
    bytes
        .par_chunks_mut(128 * k)
        .zip(scales.par_chunks_mut(column_blocks))
        .enumerate()
        .for_each(|(row_block, (bytes, scales))| {
            let rows = bytes.len() / k;
            let source = &values[row_block * 128 * k..][..rows * k];
            for (column_block, scale) in scales.iter_mut().enumerate() {
                let columns = column_block * 128..((column_block + 1) * 128).min(k);
                let amax = (0..rows)
                    .flat_map(|row| source[row * k..][columns.clone()].iter())
                    .map(|&value| to_f32(value).abs())
                    .fold(0f32, f32::max);
                *scale = if amax > 0.0 && amax.is_finite() {
                    amax / E4M3FN_MAX
                } else {
                    1.0
                };
                for row in 0..rows {
                    for column in columns.clone() {
                        bytes[row * k + column] =
                            encode_e4m3fn(to_f32(source[row * k + column]) / *scale);
                    }
                }
            }
        });
    Ok((bytes, scales))
}

struct Projection {
    n: usize,
    k: usize,
    kernel: DecodeKernel,
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
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle_core::MetalStorage,
        xl: &Layout,
        w: &candle_core::MetalStorage,
        wl: &Layout,
        s: &candle_core::MetalStorage,
        sl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        crate::kernels::metal_qwen36moe::fp8_projection(x, xl, w, wl, s, sl, self.n, self.k)
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
            ($v:ident,$ty:ty,$mv:literal,$mv2:literal,$mm:literal) => {{
                let CudaStorageSlice::$v(x) = &x.slice else {
                    unreachable!()
                };
                let x = x.slice(xl.start_offset()..xl.start_offset() + m * self.k);
                // SAFETY: every output element is written by the kernel.
                let mut out = unsafe { device.alloc::<$ty>(m * self.n)? };
                let small = m <= 4;
                // 16-byte vector loads need 16-byte aligned activation and
                // weight rows; K is a multiple of 128 by contract.
                let aligned = (xl.start_offset() * 2).is_multiple_of(16)
                    && wl.start_offset().is_multiple_of(16);
                let vectorized = small
                    && aligned
                    && match self.kernel {
                        DecodeKernel::Auto => vectorized_decode_enabled(),
                        DecodeKernel::Reference => false,
                        DecodeKernel::Vectorized | DecodeKernel::VectorizedWhenEligible => true,
                    };
                if self.kernel == DecodeKernel::Vectorized && !vectorized {
                    candle_core::bail!(
                        "vectorized FP8 decode GEMV requested for an ineligible call"
                    )
                }
                let f = device.get_or_load_custom_func(
                    if vectorized {
                        $mv2
                    } else if small {
                        $mv
                    } else {
                        $mm
                    },
                    "izwi_qwen38_fp8",
                    super::cuda_ptx::FP8,
                )?;
                let mut b = f.builder();
                b.arg(&x);
                b.arg(&w);
                b.arg(&s);
                b.arg(&mut out);
                candle_core::builder_arg!(b, m as i32, self.n as i32, self.k as i32);
                let cfg = if vectorized {
                    LaunchConfig {
                        grid_dim: (self.n.div_ceil(8) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }
                } else if small {
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
                run!(
                    F16,
                    half::f16,
                    "qwen38_fp8_mv_f16",
                    "qwen38_fp8_mv2_f16",
                    "qwen38_fp8_mm_f16"
                )
            }
            CudaStorageSlice::BF16(_) => {
                run!(
                    BF16,
                    half::bf16,
                    "qwen38_fp8_mv_bf16",
                    "qwen38_fp8_mv2_bf16",
                    "qwen38_fp8_mm_bf16"
                )
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

    fn finite_bytes() -> impl Iterator<Item = u8> {
        (0u16..256).map(|b| b as u8).filter(|b| b & 127 != 127)
    }

    #[test]
    fn encoding_inverts_every_finite_byte() {
        for byte in finite_bytes() {
            assert_eq!(encode_e4m3fn(decode_e4m3fn(byte)), byte, "{byte:#04x}");
        }
        assert!(decode_e4m3fn(encode_e4m3fn(f32::NAN)).is_nan());
    }

    #[test]
    fn encoding_rounds_to_the_nearest_byte_with_ties_to_even_and_saturates() {
        let mut probes = vec![
            0.0,
            1e-9,
            2f32.powi(-10),
            3.0 * 2f32.powi(-10),
            0.0146,
            1.0625,
            1.1875,
            447.9,
            448.0,
            464.0,
            1e9,
            f32::INFINITY,
        ];
        // Every midpoint between neighbouring codes, and points just around it.
        let mut grid: Vec<f32> = finite_bytes()
            .filter(|b| b & 128 == 0)
            .map(decode_e4m3fn)
            .collect();
        grid.sort_by(f32::total_cmp);
        for pair in grid.windows(2) {
            let mid = (pair[0] + pair[1]) / 2.0;
            probes.extend([
                mid,
                f32::from_bits(mid.to_bits() - 1),
                f32::from_bits(mid.to_bits() + 1),
            ]);
        }
        probes.extend((0..4096).map(|i| ((i as f32) * 0.618_034).fract() * 500.0));
        for value in probes.iter().flat_map(|&v| [v, -v]) {
            let byte = encode_e4m3fn(value);
            let decoded = decode_e4m3fn(byte);
            if value.abs() >= E4M3FN_MAX {
                assert_eq!(decoded, E4M3FN_MAX.copysign(value), "{value} saturates");
                continue;
            }
            let distance = (decoded - value).abs();
            let best = finite_bytes()
                .map(|b| (decode_e4m3fn(b) - value).abs())
                .fold(f32::INFINITY, f32::min);
            assert_eq!(distance, best, "{value} encoded to {decoded}");
            let tied = finite_bytes()
                .filter(|&b| (decode_e4m3fn(b) - value).abs() == best)
                .map(decode_e4m3fn)
                .filter(|v| *v != decoded && !(*v == 0.0 && decoded == 0.0))
                .count()
                > 0;
            if tied {
                assert_eq!(byte & 1, 0, "{value} tie must round to an even mantissa");
            }
        }
    }

    #[test]
    fn block_quantization_maps_each_block_maximum_to_448_and_bounds_the_error() {
        let (n, k) = (130usize, 200usize);
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.754_877_7).sin() * (1.0 + (i / k) as f32 * 0.05))
            .collect();
        let (bytes, scales) = quantize_block_fp8(&values, n, k, |v| v).unwrap();
        assert_eq!(bytes.len(), n * k);
        assert_eq!(scales.len(), 2 * 2);
        for row_block in 0..2 {
            for column_block in 0..2 {
                let scale = scales[row_block * 2 + column_block];
                let rows = row_block * 128..((row_block + 1) * 128).min(n);
                let columns = column_block * 128..((column_block + 1) * 128).min(k);
                let mut amax = 0f32;
                let mut largest = 0f32;
                for row in rows.clone() {
                    for column in columns.clone() {
                        let value = values[row * k + column];
                        let restored = decode_e4m3fn(bytes[row * k + column]) * scale;
                        amax = amax.max(value.abs());
                        largest = largest.max(decode_e4m3fn(bytes[row * k + column]).abs());
                        // Three mantissa bits: half a step is at most 1/16 of
                        // the value, and subnormals are within half of 2^-9 * scale.
                        assert!(
                            (restored - value).abs()
                                <= value.abs() / 16.0 + scale * 2f32.powi(-10) + 1e-7,
                            "[{row}, {column}]: {value} -> {restored}"
                        );
                    }
                }
                assert_eq!(scale, amax / E4M3FN_MAX);
                assert_eq!(largest, E4M3FN_MAX);
            }
        }
        let (_, zero_scales) = quantize_block_fp8(&[0f32; 4 * 4], 4, 4, |v| v).unwrap();
        assert_eq!(zero_scales, vec![1.0]);
        assert!(quantize_block_fp8(&values, n, k + 1, |v| v).is_err());
    }
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
    #[test]
    fn selecting_projection_and_kernel_gate_run_on_the_portable_reference() {
        let (n, k) = (128usize, 256usize);
        let w = Tensor::from_vec(
            (0..n * k)
                .map(|i| [0x38u8, 0xb4, 0x41, 0x0c][i % 4])
                .collect::<Vec<_>>(),
            (n, k),
            &Device::Cpu,
        )
        .unwrap();
        let s = Tensor::from_vec(vec![0.5f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let x = Tensor::from_vec(
            (0..3 * k)
                .map(|i| (i % 9) as f32 * 0.1 - 0.4)
                .collect::<Vec<_>>(),
            (3, k),
            &Device::Cpu,
        )
        .unwrap();
        let reference = block_fp8_projection(&x, &w, &s)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for vectorized in [false, true] {
            assert_eq!(
                block_fp8_projection_selecting(&x, &w, &s, vectorized)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap(),
                reference
            );
        }
        // The gate compares the device kernels with the portable reference;
        // on the CPU both are the reference, so it must pass at any K.
        for k in [128usize, 384] {
            let gate = verify_decode_kernels_at(&Device::Cpu, DType::F32, 192, k).unwrap();
            assert_eq!(gate.vectorized, Ok(()));
        }
        assert!(verify_decode_kernels_at(&Device::Cpu, DType::F32, 0, 128).is_err());
        assert!(verify_vectorized_decode(&Device::Cpu).is_err());
    }

    /// The kernel gate Qwen3.8 runs before admitting native FP8, at the
    /// checkpoint's real inner dimensions, without touching the process flag.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fp8_kernel_gate_passes_at_qwen38_k() {
        let Some(device) = super::super::cuda_test_device() else {
            return;
        };
        for k in [5120usize, 6144, 17408] {
            let gate = verify_decode_kernels_at(&device, DType::BF16, 1024, k)
                .unwrap_or_else(|e| panic!("reference kernels at K={k}: {e}"));
            assert_eq!(gate.vectorized, Ok(()), "vectorized GEMV at K={k}");
        }
        verify_vectorized_decode(&device).unwrap();
    }

    /// Per-call kernel selection: the vectorized and the reference GEMV agree
    /// at decode row counts, and prefill rows take the GEMM either way.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_selecting_reference_vs_vectorized_match() {
        let Some(device) = super::super::cuda_test_device() else {
            return;
        };
        let (n, k) = (1024usize, 5120usize);
        let raw: Vec<u8> = (0..n * k)
            .map(|i| match (i.wrapping_mul(0x9E37_79B1) >> 17) as u8 {
                b if b & 0x7f == 0x7f => 0x38,
                b => b,
            })
            .collect();
        let w = Tensor::from_vec(raw, (n, k), &device).unwrap();
        let s = Tensor::from_vec(
            (0..8 * 40)
                .map(|i| 1e-3 * (1 + i % 11) as f32)
                .collect::<Vec<_>>(),
            (8, 40),
            &device,
        )
        .unwrap();
        for m in [1usize, 3, 4, 9] {
            let x = Tensor::from_vec(
                (0..m * k)
                    .map(|i| ((i as f32) * 0.37).sin())
                    .collect::<Vec<_>>(),
                (m, k),
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
            let host = |vectorized| {
                block_fp8_projection_selecting(&x, &w, &s, vectorized)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let (reference, vectorized) = (host(false), host(true));
            let scale = reference.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            for (a, b) in vectorized.iter().zip(&reference) {
                assert!((a - b).abs() <= 0.02 * scale, "m={m}: {a} vs {b}");
            }
        }
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

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_vectorized_decode_gemv_verifies_and_matches_the_reference() {
        let Some(device) = super::super::cuda_test_device() else {
            return;
        };
        verify_and_enable_vectorized_decode(&device).expect("vectorized GEMV self-check");
        assert!(vectorized_decode_enabled());
        // Unaligned activation rows (odd start offset) stay on the reference
        // kernel instead of faulting.
        let w = Tensor::full(0x38u8, (64, 128), &device).unwrap();
        let s = Tensor::ones((1, 1), DType::F32, &device).unwrap();
        let x = Tensor::ones((3, 129), DType::BF16, &device)
            .unwrap()
            .narrow(1, 1, 128)
            .unwrap()
            .narrow(0, 1, 1)
            .unwrap();
        let y = block_fp8_projection(&x, &w, &s).unwrap();
        let y = y
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(y.iter().all(|v| (*v - 128.0).abs() < 1.0), "{y:?}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_projection_matches_portable_reference_at_trunk_geometry() {
        let Some(device) = super::super::cuda_test_device() else {
            return;
        };
        // Real trunk projections: the fused q+gate q_proj is [8192, 2048] and
        // the expert down projection is [2048, 512]. Scales are distinct per
        // 128x128 block on BOTH axes so a scale-index swap cannot pass, and M
        // crosses the mv/mm dispatch boundary (4/5) plus partial m-tiles.
        // qwen36moe is this kernel's first production consumer and the toy
        // geometry below (192x256) cannot catch real-geometry defects, so the
        // trunk shapes are pinned here (see
        // tasks/qwen36moe-native-cuda-gibberish-research-2026-10-07.md).
        for (n, k, m_values) in [
            (8192usize, 2048usize, &[1usize, 4, 5, 17, 64][..]),
            (2048usize, 512usize, &[1usize, 4, 5, 33][..]),
        ] {
            let scale_rows = n.div_ceil(128);
            let scale_cols = k.div_ceil(128);
            let scales: Vec<f32> = (0..scale_rows * scale_cols)
                .map(|i| 0.25 * (1 + (i % 8)) as f32)
                .collect();
            let raw: Vec<u8> = (0..n * k)
                .map(|i| {
                    let mixed = (i.wrapping_mul(0x9E37_79B1) >> 17) as u8;
                    match mixed {
                        0x7F => 0x38,
                        0xFF => 0xB8,
                        other => other,
                    }
                })
                .collect();
            let w_cpu = Tensor::from_vec(raw, (n, k), &Device::Cpu).unwrap();
            let s_cpu =
                Tensor::from_vec(scales, (scale_rows, scale_cols), &Device::Cpu).unwrap();
            let w_gpu = w_cpu.to_device(&device).unwrap();
            let s_gpu = s_cpu.to_device(&device).unwrap();
            for m in m_values.iter().copied() {
                let x_cpu = Tensor::from_vec(
                    (0..m * k)
                        .map(|i| (i % 7) as f32 * 0.125 - 0.5)
                        .collect::<Vec<_>>(),
                    (m, k),
                    &Device::Cpu,
                )
                .unwrap();
                let expected = block_fp8_projection(&x_cpu, &w_cpu, &s_cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for dtype in [DType::F16, DType::BF16] {
                    let x_gpu = x_cpu
                        .to_dtype(dtype)
                        .unwrap()
                        .to_device(&device)
                        .unwrap();
                    let actual = block_fp8_projection(&x_gpu, &w_gpu, &s_gpu)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                        assert!(
                            (a - e).abs() <= 0.1 + e.abs() * 0.025,
                            "{dtype:?} n={n} k={k} M={m} index {index}: {a} != {e}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::kernels::cuda::graphs::{IslandOutput, TensorIsland};
    use candle_core::{DType, Tensor};

    /// Phase 5 capture-compatibility spike: the block-FP8 projection custom
    /// op must be capturable inside a CUDA graph island before any qwen36moe
    /// trunk island is ported. The kernel launches through the standard
    /// stream path with its output as the only allocation (no host reads,
    /// no growable scratch — the reason Q8/QMatMul stay excluded), and the
    /// identical kernel is already captured by qwen38's production MLP
    /// islands; this test makes that verdict executable at hardware handoff.
    /// A capture failure shows up as negative fallbacks with zero captures.
    #[test]
    fn compact_fp8_projection_is_capture_compatible() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let (n, k) = (64usize, 128usize);
        if !provider_supported(&device, DType::BF16, n, k) {
            return;
        }
        let weights = Tensor::full(0x38u8, (n, k), &device)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();
        let scales = Tensor::ones((n.div_ceil(128), k.div_ceil(128)), DType::F32, &device).unwrap();
        let island = TensorIsland::default();
        for step in 0..3 {
            let x = Tensor::full(step as f32 + 1.0, (2, k), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let eager = block_fp8_projection(&x, &weights, &scales).unwrap();
            let w = weights.clone();
            let s = scales.clone();
            let captured = island
                .run_multi(
                    &[&x],
                    &[weights.clone(), scales.clone()],
                    9,
                    "qwen36moe.compact-fp8-spike",
                    1 << 18,
                    1 << 15,
                    move |inputs| {
                        let out = block_fp8_projection(&inputs[0], &w, &s)?;
                        Ok(IslandOutput {
                            outputs: vec![out],
                            intermediates: vec![],
                        })
                    },
                )
                .unwrap()
                .expect("captured region must execute on CUDA");
            let expected = eager.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
            let actual = captured[0].to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
            for (row_a, row_e) in actual.iter().zip(expected.iter()) {
                for (a, e) in row_a.iter().zip(row_e.iter()) {
                    assert!((a - e).abs() <= 1e-2 + e.abs() * 1e-2, "{a} != {e}");
                }
            }
        }
        let d = island.diagnostics();
        assert_eq!(d["captures"], 1, "the region must capture, not fall back");
        assert_eq!(d["replays"], 2, "later steps must replay");
        assert_eq!(d["negative_fallbacks"], 0);
        island.invalidate();
    }
}
