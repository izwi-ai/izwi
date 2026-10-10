//! Full-attention q/k head norm plus partial rotate-half RoPE for one token
//! (Qwen3.6 gated attention).
//!
//! One launch replaces the per-layer chain:
//!
//! - q/k contiguous copies;
//! - cast → rms_norm → cast;
//! - host-built cos/sin upload, then cos/sin/casts;
//! - the rotary op;
//! - the pass-through concatenation.
//!
//! Queries are read straight from the gated `q_proj` layout
//! `[num_heads, 2 * head_dim]` (query half first). Cos and sin are computed
//! on the device from scalar positions and F32 inverse frequencies.
//!
//! Rounding mirrors the reference chain: normalized values, cos and sin are
//! rounded to the activation dtype before the rotation, and the result is
//! rounded once. The CPU path implements the same arithmetic as the reference
//! for tests and the load-time self-check.
use candle_core::{DType, Device, Result, Tensor};

/// Head geometry for [`qk_norm_rope`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QkNormRopeSpec {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Leading dims of each head that rotate (rotate-half within them).
    pub rope_dim: usize,
    pub eps: f32,
}

/// One token's M-RoPE positions (temporal, height, width) and the interleaved
/// section lengths for the height and width components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MropePosition {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
    pub height_section: usize,
    pub width_section: usize,
}

impl MropePosition {
    fn component(&self, j: usize) -> usize {
        let sectioned = self.temporal != self.height || self.temporal != self.width;
        if sectioned && j % 3 == 1 && j < 3 * self.height_section {
            self.height
        } else if sectioned && j % 3 == 2 && j < 3 * self.width_section {
            self.width
        } else {
            self.temporal
        }
    }
}

const MAX_DYNAMIC_SHARED_BYTES: usize = 48 * 1024;

/// Whether the fused q/k norm + RoPE can serve this device, dtype and geometry.
pub fn supported(device: &Device, dtype: DType, head_dim: usize, rope_dim: usize) -> bool {
    let geometry = head_dim > 0
        && rope_dim.is_multiple_of(2)
        && rope_dim <= head_dim
        && head_dim * 4 <= MAX_DYNAMIC_SHARED_BYTES;
    geometry
        && match device {
            Device::Cpu => matches!(dtype, DType::F16 | DType::BF16),
            Device::Cuda(_) => {
                matches!(dtype, DType::F16 | DType::BF16)
                    && super::fp8::device_is_sm80_or_newer(device)
            }
            Device::Metal(_) => cfg!(feature = "metal") && dtype == DType::F16 && head_dim <= 1024,
        }
}

/// Normalize and rotate one token's query and key heads.
///
/// - `q_proj`: `num_heads * 2 * head_dim` values (gated layout).
/// - `k_proj`: `num_kv_heads * head_dim` values.
/// - `q_gain`, `k_gain`: `[head_dim]` F32.
/// - `inv_freq`: `[rope_dim / 2]` F32 on the activations' device.
///
/// Returns `(q [1, 1, num_heads, head_dim], k [1, 1, num_kv_heads, head_dim])`
/// in the activation dtype.
pub fn qk_norm_rope(
    q_proj: &Tensor,
    k_proj: &Tensor,
    q_gain: &Tensor,
    k_gain: &Tensor,
    inv_freq: &Tensor,
    spec: &QkNormRopeSpec,
    position: MropePosition,
) -> Result<(Tensor, Tensor)> {
    let dtype = q_proj.dtype();
    let d = spec.head_dim;
    if q_proj.elem_count() != spec.num_heads * 2 * d
        || k_proj.elem_count() != spec.num_kv_heads * d
        || k_proj.dtype() != dtype
        || !matches!(dtype, DType::F16 | DType::BF16)
        || q_gain.elem_count() != d
        || k_gain.elem_count() != d
        || q_gain.dtype() != DType::F32
        || k_gain.dtype() != DType::F32
        || inv_freq.elem_count() != spec.rope_dim / 2
        || inv_freq.dtype() != DType::F32
        || !spec.rope_dim.is_multiple_of(2)
        || spec.rope_dim > d
        || spec.num_heads == 0
        || spec.num_kv_heads == 0
    {
        candle_core::bail!("invalid fused q/k norm + RoPE contract for {spec:?}")
    }
    #[cfg(feature = "cuda")]
    if q_proj.device().is_cuda() {
        return cuda_impl::launch(q_proj, k_proj, q_gain, k_gain, inv_freq, spec, position);
    }
    #[cfg(feature = "metal")]
    if q_proj.device().is_metal() {
        return crate::kernels::metal_qwen36moe::qk_norm_rope(
            q_proj, k_proj, q_gain, k_gain, inv_freq, spec, position,
        );
    }
    if !q_proj.device().is_cpu() {
        candle_core::bail!("fused q/k norm + RoPE has no implementation for this device")
    }
    let round = |v: f32| -> f32 {
        match dtype {
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => half::f16::from_f32(v).to_f32(),
        }
    };
    let host = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
    let (q, k) = (host(q_proj)?, host(k_proj)?);
    let (qg, kg, inv) = (host(q_gain)?, host(k_gain)?, host(inv_freq)?);
    let head = |src: &[f32], gain: &[f32]| -> Vec<f32> {
        let mean = src.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let scale = 1.0 / (mean + spec.eps).sqrt();
        let n: Vec<f32> = src
            .iter()
            .zip(gain)
            .map(|(v, g)| round(v * scale * g))
            .collect();
        let half = spec.rope_dim / 2;
        (0..d)
            .map(|i| {
                if i >= spec.rope_dim {
                    return round(n[i]);
                }
                let j = if i < half { i } else { i - half };
                let angle = position.component(j) as f32 * inv[j];
                let (c, s) = (round(angle.cos()), round(angle.sin()));
                round(if i < half {
                    n[i] * c - n[i + half] * s
                } else {
                    n[i - half] * s + n[i] * c
                })
            })
            .collect()
    };
    let q_out: Vec<f32> = (0..spec.num_heads)
        .flat_map(|h| head(&q[h * 2 * d..h * 2 * d + d], &qg))
        .collect();
    let k_out: Vec<f32> = (0..spec.num_kv_heads)
        .flat_map(|h| head(&k[h * d..(h + 1) * d], &kg))
        .collect();
    let device = q_proj.device();
    Ok((
        Tensor::from_vec(q_out, (1, 1, spec.num_heads, d), device)?.to_dtype(dtype)?,
        Tensor::from_vec(k_out, (1, 1, spec.num_kv_heads, d), device)?.to_dtype(dtype)?,
    ))
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{MropePosition, QkNormRopeSpec};
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaDevice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};

    const MODULE: &str = "izwi_qwen36moe";

    fn view<'a, T: CudaDType>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("fused q/k norm + RoPE inputs must be CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("fused q/k norm + RoPE inputs must be contiguous")
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

    fn position_arg(value: usize) -> Result<i32> {
        i32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("RoPE position {value} exceeds i32")))
    }

    pub(super) fn launch(
        q_proj: &Tensor,
        k_proj: &Tensor,
        q_gain: &Tensor,
        k_gain: &Tensor,
        inv_freq: &Tensor,
        spec: &QkNormRopeSpec,
        position: MropePosition,
    ) -> Result<(Tensor, Tensor)> {
        let device = q_proj.device().as_cuda_device()?;
        let d = spec.head_dim;
        let tensors = [q_proj, k_proj, q_gain, k_gain, inv_freq].map(|t| t.contiguous());
        let [q, k, qg, kg, inv] = tensors;
        let (q, k, qg, kg, inv) = (q?, k?, qg?, kg?, inv?);
        let (q_s, _) = q.storage_and_layout();
        let (k_s, _) = k.storage_and_layout();
        let (qg_s, _) = qg.storage_and_layout();
        let (kg_s, _) = kg.storage_and_layout();
        let (inv_s, _) = inv.storage_and_layout();
        let qg_v = view::<f32>(&qg_s, &qg)?;
        let kg_v = view::<f32>(&kg_s, &kg)?;
        let inv_v = view::<f32>(&inv_s, &inv)?;
        let config = LaunchConfig {
            grid_dim: ((spec.num_heads + spec.num_kv_heads) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (d * 4) as u32,
        };
        macro_rules! run {
            ($ty:ty, $name:literal) => {{
                let qv = view::<$ty>(&q_s, &q)?;
                let kv = view::<$ty>(&k_s, &k)?;
                // SAFETY: one block writes every element of its head.
                let q_out = unsafe { device.alloc::<$ty>(spec.num_heads * d)? };
                let k_out = unsafe { device.alloc::<$ty>(spec.num_kv_heads * d)? };
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = function.builder();
                builder.arg(&qv);
                builder.arg(&kv);
                builder.arg(&qg_v);
                builder.arg(&kg_v);
                builder.arg(&inv_v);
                builder.arg(&q_out);
                builder.arg(&k_out);
                candle_core::builder_arg!(
                    builder,
                    spec.num_heads as i32,
                    d as i32,
                    spec.rope_dim as i32,
                    position_arg(position.temporal)?,
                    position_arg(position.height)?,
                    position_arg(position.width)?,
                    position.height_section as i32,
                    position.width_section as i32,
                    spec.eps
                );
                // SAFETY: validated geometry; argument order matches the kernel.
                unsafe { builder.launch(config) }.w()?;
                (
                    wrap(q_out, device, Shape::from((1, 1, spec.num_heads, d))),
                    wrap(k_out, device, Shape::from((1, 1, spec.num_kv_heads, d))),
                )
            }};
        }
        Ok(match q.dtype() {
            DType::BF16 => run!(half::bf16, "qwen36moe_qk_norm_rope_bf16"),
            DType::F16 => run!(half::f16, "qwen36moe_qk_norm_rope_f16"),
            other => candle_core::bail!("fused q/k norm + RoPE does not support {other:?}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrope_component_follows_the_interleaved_sections() {
        let text = MropePosition {
            temporal: 9,
            height: 9,
            width: 9,
            height_section: 2,
            width_section: 2,
        };
        assert!((0..16).all(|j| text.component(j) == 9));
        let image = MropePosition {
            temporal: 1,
            height: 2,
            width: 3,
            ..text
        };
        let components: Vec<usize> = (0..8).map(|j| image.component(j)).collect();
        // Height covers dims 1 and 4 (< 3*2), width 2 and 5; the rest temporal.
        assert_eq!(components, [1, 2, 3, 1, 2, 3, 1, 1]);
    }

    #[test]
    fn rope_rotates_only_the_leading_dims_and_preserves_norms() {
        let spec = QkNormRopeSpec {
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            rope_dim: 4,
            eps: 1e-6,
        };
        let q = Tensor::from_vec(
            (0..32).map(|i| (i as f32 * 0.37).sin()).collect::<Vec<_>>(),
            (1, 1, 32),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
        let k = Tensor::from_vec(
            (0..8).map(|i| (i as f32 * 0.61).cos()).collect::<Vec<_>>(),
            (1, 1, 8),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
        let gain = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let inv = Tensor::from_vec(vec![1.0f32, 0.01], 2, &Device::Cpu).unwrap();
        let at = |pos: usize| {
            qk_norm_rope(
                &q,
                &k,
                &gain,
                &gain,
                &inv,
                &spec,
                MropePosition {
                    temporal: pos,
                    height: pos,
                    width: pos,
                    height_section: 0,
                    width_section: 0,
                },
            )
            .unwrap()
        };
        let host = |t: &Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (q0, _) = at(0);
        let (q7, k7) = at(7);
        assert_eq!(q0.dims(), [1, 1, 2, 8]);
        assert_eq!(k7.dims(), [1, 1, 1, 8]);
        let (q0, q7) = (host(&q0), host(&q7));
        // Pass-through dims are untouched by position; rotated pairs keep their norm.
        assert_eq!(q0[4..8], q7[4..8]);
        let pair = |v: &[f32], i: usize| (v[i] * v[i] + v[i + 2] * v[i + 2]).sqrt();
        assert!((pair(&q0, 0) - pair(&q7, 0)).abs() < 1e-2);
        assert_ne!(q0[0], q7[0]);
    }
}
