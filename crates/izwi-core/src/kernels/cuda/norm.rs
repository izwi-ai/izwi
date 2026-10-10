//! RMSNorm with an F32 gain over 16-bit activations, optionally fused with the
//! preceding residual add.
//!
//! Under the CUDA BF16 plan, the Qwen3.6 trunk norms carry F32 `1 + w` gains,
//! so each norm runs as cast → rms_norm → cast. With the residual add before
//! it, that is four launches; the fused kernel does the work in one. The
//! residual sum is rounded to the activation dtype before normalization,
//! matching HF's BF16 residual stream and
//! `(norm(x.float()) * w.float()).type_as(x)`.
//!
//! The CPU path is the Candle composition the kernel replaces; tests and the
//! load-time self-check compare against it.
use candle_core::{DType, Device, Result, Tensor, D};

/// Whether the fused norm can serve `x`'s device and dtype with an F32 gain.
pub fn supported(device: &Device, dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
        && match device {
            Device::Cpu => true,
            Device::Cuda(_) => super::fp8::device_is_sm80_or_newer(device),
            _ => false,
        }
}

fn check(x: &Tensor, weight: &Tensor) -> Result<usize> {
    let hidden = x.dim(D::Minus1)?;
    if weight.dtype() != DType::F32
        || weight.elem_count() != hidden
        || !matches!(x.dtype(), DType::F16 | DType::BF16)
    {
        candle_core::bail!(
            "invalid fused RMSNorm contract: x {:?} {:?}, weight {:?} {:?}",
            x.dims(),
            x.dtype(),
            weight.dims(),
            weight.dtype()
        )
    }
    Ok(hidden)
}

fn reference(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.to_dtype(DType::F32)?, weight, eps)?.to_dtype(x.dtype())
}

/// `rms_norm(x) * weight` over the last dim, in `x`'s dtype.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden = check(x, weight)?;
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::launch(x, None, weight, eps, hidden).map(|(_, out)| out);
    }
    let _ = hidden;
    reference(x, weight, eps)
}

/// `(residual + delta, rms_norm(residual + delta) * weight)`, with the sum
/// rounded to the activation dtype before the norm.
pub fn add_rms_norm(
    residual: &Tensor,
    delta: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let hidden = check(residual, weight)?;
    if delta.dims() != residual.dims() || delta.dtype() != residual.dtype() {
        candle_core::bail!(
            "fused add+RMSNorm operands differ: {:?} {:?} vs {:?} {:?}",
            residual.dims(),
            residual.dtype(),
            delta.dims(),
            delta.dtype()
        )
    }
    #[cfg(feature = "cuda")]
    if residual.device().is_cuda() {
        let (sum, out) = cuda_impl::launch(delta, Some(residual), weight, eps, hidden)?;
        return Ok((sum.expect("fused add+RMSNorm returns the sum"), out));
    }
    let _ = hidden;
    let sum = (residual + delta)?;
    let out = reference(&sum, weight, eps)?;
    Ok((sum, out))
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaDevice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};

    const MODULE: &str = "izwi_qwen36moe";

    fn view<'a, T: CudaDType>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("fused RMSNorm inputs must be CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("fused RMSNorm inputs must be contiguous")
        };
        Ok(cuda.as_cuda_slice::<T>()?.slice(start..end))
    }

    fn wrap<T: CudaDType>(slice: CudaSlice<T>, device: &CudaDevice, shape: Shape) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    /// Launch the norm over `x` (plus `residual` when given). Returns the
    /// rounded sum (only with a residual) and the normalized output.
    pub(super) fn launch(
        x: &Tensor,
        residual: Option<&Tensor>,
        weight: &Tensor,
        eps: f32,
        hidden: usize,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let device = x.device().as_cuda_device()?;
        let shape = x.shape().clone();
        let elements = shape.elem_count();
        let rows = elements / hidden.max(1);
        if rows == 0 {
            candle_core::bail!("fused RMSNorm needs at least one row")
        }
        let x = x.contiguous()?;
        let weight = weight.contiguous()?;
        let residual = residual.map(Tensor::contiguous).transpose()?;
        let (x_storage, _) = x.storage_and_layout();
        let (w_storage, _) = weight.storage_and_layout();
        let w = view::<f32>(&w_storage, &weight)?;
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows).map_err(|_| candle_core::Error::Msg("RMSNorm grid".into()))?,
                1,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! run {
            ($ty:ty, $plain:literal, $fused:literal) => {{
                let xv = view::<$ty>(&x_storage, &x)?;
                // SAFETY: the kernel writes every element of both outputs.
                let out = unsafe { device.alloc::<$ty>(elements)? };
                match &residual {
                    None => {
                        let function = device.get_or_load_custom_func(
                            $plain,
                            MODULE,
                            super::super::cuda_ptx::QWEN36MOE,
                        )?;
                        let mut builder = function.builder();
                        builder.arg(&xv);
                        builder.arg(&w);
                        builder.arg(&out);
                        candle_core::builder_arg!(builder, hidden as i32, eps);
                        // SAFETY: argument order and types match the kernel.
                        unsafe { builder.launch(config) }.w()?;
                        (None, wrap(out, device, shape.clone()))
                    }
                    Some(residual) => {
                        let (r_storage, _) = residual.storage_and_layout();
                        let rv = view::<$ty>(&r_storage, residual)?;
                        // SAFETY: the kernel writes every element of the sum.
                        let sum = unsafe { device.alloc::<$ty>(elements)? };
                        let function = device.get_or_load_custom_func(
                            $fused,
                            MODULE,
                            super::super::cuda_ptx::QWEN36MOE,
                        )?;
                        let mut builder = function.builder();
                        builder.arg(&xv);
                        builder.arg(&rv);
                        builder.arg(&w);
                        builder.arg(&sum);
                        builder.arg(&out);
                        candle_core::builder_arg!(builder, hidden as i32, eps);
                        // SAFETY: argument order and types match the kernel.
                        unsafe { builder.launch(config) }.w()?;
                        (
                            Some(wrap(sum, device, shape.clone())),
                            wrap(out, device, shape.clone()),
                        )
                    }
                }
            }};
        }
        Ok(match x.dtype() {
            DType::BF16 => run!(
                half::bf16,
                "qwen36moe_rms_norm_bf16",
                "qwen36moe_add_rms_norm_bf16"
            ),
            DType::F16 => run!(
                half::f16,
                "qwen36moe_rms_norm_f16",
                "qwen36moe_add_rms_norm_f16"
            ),
            other => candle_core::bail!("fused RMSNorm does not support {other:?}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * scale)
            .collect()
    }

    #[test]
    fn cpu_composition_rounds_the_sum_before_the_norm() {
        let shape = (2, 1, 64);
        let r = Tensor::from_vec(wave(128, 1.0, 3.0), shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let d = Tensor::from_vec(wave(128, 2.0, 3.0), shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(
            wave(64, 3.0, 0.3)
                .iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            64,
            &Device::Cpu,
        )
        .unwrap();
        let (sum, out) = add_rms_norm(&r, &d, &w, 1e-6).unwrap();
        assert_eq!(sum.dtype(), DType::BF16);
        assert_eq!(out.dims(), [2, 1, 64]);
        let expected = rms_norm(&(&r + &d).unwrap(), &w, 1e-6).unwrap();
        assert_eq!(
            out.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert!(add_rms_norm(&r, &d.narrow(2, 0, 32).unwrap(), &w, 1e-6).is_err());
        assert!(rms_norm(&r.to_dtype(DType::F32).unwrap(), &w, 1e-6).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rms_norm_matches_the_candle_composition() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        for (rows, hidden) in [(1usize, 2048usize), (5, 2048), (32, 256)] {
            for dtype in [DType::BF16, DType::F16] {
                let r =
                    Tensor::from_vec(wave(rows * hidden, 1.0, 3.0), (rows, hidden), &Device::Cpu)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                let d =
                    Tensor::from_vec(wave(rows * hidden, 2.0, 3.0), (rows, hidden), &Device::Cpu)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                let w = Tensor::from_vec(
                    wave(hidden, 3.0, 0.3)
                        .iter()
                        .map(|v| 1.0 + v)
                        .collect::<Vec<_>>(),
                    hidden,
                    &Device::Cpu,
                )
                .unwrap();
                let (cpu_sum, cpu_out) = add_rms_norm(&r, &d, &w, 1e-6).unwrap();
                let g = |t: &Tensor| t.to_device(&device).unwrap();
                let (gpu_sum, gpu_out) = add_rms_norm(&g(&r), &g(&d), &g(&w), 1e-6).unwrap();
                let host = |t: &Tensor| {
                    t.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                };
                assert_eq!(host(&gpu_sum), host(&cpu_sum), "{dtype:?} sum");
                let plain = host(&rms_norm(&g(&r), &g(&w), 1e-6).unwrap());
                let plain_ref = host(&rms_norm(&r, &w, 1e-6).unwrap());
                for (label, actual, expected) in [
                    ("add+norm", host(&gpu_out), host(&cpu_out)),
                    ("norm", plain, plain_ref),
                ] {
                    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
                    for (a, e) in actual.iter().zip(&expected) {
                        assert!(
                            (a - e).abs() <= 0.01 * scale,
                            "{label} {dtype:?} rows={rows}: {a} vs {e}"
                        );
                    }
                }
            }
        }
    }
}
