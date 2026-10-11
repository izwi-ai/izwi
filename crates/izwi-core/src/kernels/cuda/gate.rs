//! Gated attention output for Qwen3.6 full attention: `attn * sigmoid(gate)`,
//! where the gate is the second half of each head's slice of the gated query
//! projection `[.., heads, 2 * head_dim]`.
//!
//! The composition it replaces narrows the gate out of `q_proj` (a strided
//! copy once reshaped), applies a sigmoid and multiplies: three launches per
//! attention layer. The kernel evaluates the sigmoid in F32, rounds it to the
//! activation dtype as the composition does, and rounds the product. The CPU
//! path is that composition; tests and the load-time self-check compare
//! against it.
use candle_core::{DType, Device, Result, Tensor, D};

/// Whether the fused gate can serve `dtype` activations on `device`.
pub fn supported(device: &Device, dtype: DType) -> bool {
    match device {
        Device::Cpu => matches!(dtype, DType::F16 | DType::BF16),
        Device::Cuda(_) => {
            matches!(dtype, DType::F16 | DType::BF16) && super::fp8::device_is_sm80_or_newer(device)
        }
        Device::Metal(_) => cfg!(feature = "metal") && dtype == DType::F16,
    }
}

fn check(attn: &Tensor, q_proj: &Tensor, heads: usize, head_dim: usize) -> Result<()> {
    let width = heads * head_dim;
    if width == 0
        || attn.dim(D::Minus1)? != width
        || q_proj.dims().len() < 2
        || q_proj.dim(D::Minus1)? != 2 * head_dim
        || q_proj.dim(D::Minus2)? != heads
        || q_proj.elem_count() != 2 * attn.elem_count()
        || attn.dtype() != q_proj.dtype()
    {
        candle_core::bail!(
            "invalid attention gate contract: attn {:?} {:?}, q_proj {:?} {:?}, heads {heads}, head_dim {head_dim}",
            attn.dims(),
            attn.dtype(),
            q_proj.dims(),
            q_proj.dtype()
        )
    }
    Ok(())
}

/// The Candle composition the kernel replaces.
pub fn reference(attn: &Tensor, q_proj: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    check(attn, q_proj, heads, head_dim)?;
    let gate = q_proj
        .narrow(D::Minus1, head_dim, head_dim)?
        .reshape(attn.shape())?;
    attn * candle_nn::ops::sigmoid(&gate)?
}

/// `attn * sigmoid(q_proj[.., h, head_dim..])`, shaped like `attn`.
pub fn attn_gate(attn: &Tensor, q_proj: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    check(attn, q_proj, heads, head_dim)?;
    #[cfg(feature = "cuda")]
    if attn.device().is_cuda() {
        return cuda_impl::launch(attn, q_proj, heads, head_dim);
    }
    #[cfg(feature = "metal")]
    if attn.device().is_metal() && attn.dtype() == DType::F16 {
        return crate::kernels::metal_qwen36moe::attn_gate(attn, q_proj, heads, head_dim);
    }
    reference(attn, q_proj, heads, head_dim)
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDType, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Result, Storage, Tensor};

    const MODULE: &str = "izwi_qwen36moe";

    fn view<'a, T: CudaDType>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("attention gate inputs must be CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("attention gate inputs must be contiguous")
        };
        Ok(cuda.as_cuda_slice::<T>()?.slice(start..end))
    }

    pub(super) fn launch(
        attn: &Tensor,
        q_proj: &Tensor,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let device = attn.device().as_cuda_device()?;
        let shape = attn.shape().clone();
        let total = shape.elem_count();
        let int = |value: usize, name: &str| {
            i32::try_from(value)
                .map_err(|_| candle_core::Error::Msg(format!("attention gate {name} {value}")))
        };
        let (attn, q_proj) = (attn.contiguous()?, q_proj.contiguous()?);
        let (a_storage, _) = attn.storage_and_layout();
        let (q_storage, _) = q_proj.storage_and_layout();
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(total.div_ceil(256))
                    .map_err(|_| candle_core::Error::Msg("attention gate grid".into()))?,
                1,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! run {
            ($ty:ty, $name:literal) => {{
                let a = view::<$ty>(&a_storage, &attn)?;
                let q = view::<$ty>(&q_storage, &q_proj)?;
                // SAFETY: the kernel writes every output element.
                let out = unsafe { device.alloc::<$ty>(total)? };
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = function.builder();
                builder.arg(&a);
                builder.arg(&q);
                builder.arg(&out);
                candle_core::builder_arg!(
                    builder,
                    int(heads, "heads")?,
                    int(head_dim, "head_dim")?,
                    int(total, "elements")?
                );
                // SAFETY: argument order and types match the kernel.
                unsafe { builder.launch(config) }.w()?;
                CudaStorage::wrap_cuda_slice(out, device.clone())
            }};
        }
        let storage = match attn.dtype() {
            DType::BF16 => run!(half::bf16, "qwen36moe_attn_gate_bf16"),
            DType::F16 => run!(half::f16, "qwen36moe_attn_gate_f16"),
            other => candle_core::bail!("attention gate does not support {other:?}"),
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(storage),
            shape,
            BackpropOp::none(),
            false,
        ))
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

    fn operands(rows: usize, heads: usize, head_dim: usize, dtype: DType) -> (Tensor, Tensor) {
        let width = heads * head_dim;
        let attn = Tensor::from_vec(wave(rows * width, 1.0, 4.0), (rows, 1, width), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let q_proj = Tensor::from_vec(
            wave(rows * 2 * width, 2.0, 8.0),
            (rows, 1, heads, 2 * head_dim),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        (attn, q_proj)
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    #[test]
    fn reference_gates_each_head_with_the_second_half_of_its_query_slice() {
        let (rows, heads, head_dim) = (2usize, 3usize, 4usize);
        let (attn, q_proj) = operands(rows, heads, head_dim, DType::F32);
        let out = attn_gate(&attn, &q_proj, heads, head_dim).unwrap();
        assert_eq!(out.dims(), attn.dims());
        let (a, q, o) = (host(&attn), host(&q_proj), host(&out));
        let width = heads * head_dim;
        for r in 0..rows {
            for h in 0..heads {
                for d in 0..head_dim {
                    let g = q[r * 2 * width + h * 2 * head_dim + head_dim + d];
                    let want = a[r * width + h * head_dim + d] / (1.0 + (-g).exp());
                    let got = o[r * width + h * head_dim + d];
                    assert!(
                        (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                        "{r} {h} {d}"
                    );
                }
            }
        }
    }

    #[test]
    fn contract_rejects_mismatched_layouts() {
        let (attn, q_proj) = operands(1, 2, 8, DType::F16);
        assert!(
            attn_gate(&attn, &q_proj, 4, 4).is_err(),
            "head split mismatch"
        );
        assert!(attn_gate(&attn, &q_proj.to_dtype(DType::BF16).unwrap(), 2, 8).is_err());
        let narrow = q_proj.narrow(3, 0, 8).unwrap();
        assert!(attn_gate(&attn, &narrow, 2, 8).is_err(), "ungated q_proj");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_attn_gate_matches_the_composition() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        for (rows, heads, head_dim) in [(1usize, 16usize, 256usize), (3, 2, 100)] {
            for dtype in [DType::BF16, DType::F16] {
                let (attn, q_proj) = operands(rows, heads, head_dim, dtype);
                let expected = host(&reference(&attn, &q_proj, heads, head_dim).unwrap());
                let actual = host(
                    &attn_gate(
                        &attn.to_device(&device).unwrap(),
                        &q_proj.to_device(&device).unwrap(),
                        heads,
                        head_dim,
                    )
                    .unwrap(),
                );
                let max_ref = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
                let max_err = actual
                    .iter()
                    .zip(&expected)
                    .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
                assert!(
                    max_err <= 0.02 * max_ref,
                    "{dtype:?} rows={rows}: max error {max_err} vs max |ref| {max_ref}"
                );
            }
        }
    }
}
