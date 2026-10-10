//! Fused Gated DeltaNet single-token decode for Qwen3.6 linear-attention
//! layers.
//!
//! Two launches replace the per-layer Candle chain of about 49 ops:
//!
//! - [`conv_decode`]: the causal-conv step over the 3-slot history plus SiLU.
//!   It also returns the F32 copy of the input that becomes the ring's next
//!   slot.
//! - [`recurrent_decode`]: one block per value head. It runs the
//!   `softplus(alpha + dt_bias) * a` decay, `sigmoid(beta)`, q/k L2 norms, the
//!   delta-rule state update (state held in registers) and the gated RMSNorm,
//!   writing the out-projection input directly.
//!
//! Key heads are mapped by index (grouped: `v / repeats`, tiled:
//! `v % key_heads`), so no expanded q/k tensors are made. The previous state is
//! read-only and the next state is a fresh allocation, keeping state
//! publication transactional (snapshots may share the previous tensor).
//!
//! The CPU implementations are the portable reference for tests and the
//! load-time self-check.
use candle_core::{DType, Device, Result, Tensor};

/// Head width (key and value) the decode kernel is written for.
pub const HEAD_DIM: usize = 128;
/// Causal-conv taps the decode kernel is written for (3 history slots).
pub const CONV_TAPS: usize = 4;

/// Layout of one linear-attention layer's heads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnDecodeSpec {
    pub key_heads: usize,
    pub value_heads: usize,
    /// HF safetensors order (`value head v` reads key head `v / repeats`);
    /// false is llama.cpp's tiled order (`v % key_heads`).
    pub grouped: bool,
    pub norm_eps: f32,
}

impl GdnDecodeSpec {
    fn validate(&self) -> Result<()> {
        if self.key_heads == 0
            || self.value_heads == 0
            || !self.value_heads.is_multiple_of(self.key_heads)
        {
            candle_core::bail!("invalid DeltaNet head layout {self:?}")
        }
        Ok(())
    }

    pub fn conv_dim(&self) -> usize {
        (2 * self.key_heads + self.value_heads) * HEAD_DIM
    }

    fn key_head(&self, value_head: usize) -> usize {
        if self.grouped {
            value_head / (self.value_heads / self.key_heads)
        } else {
            value_head % self.key_heads
        }
    }
}

/// Whether the fused decode can serve this device and layer geometry. CUDA
/// needs SM80+; the CPU reference runs anywhere.
pub fn supported(device: &Device, head_k_dim: usize, head_v_dim: usize, conv_taps: usize) -> bool {
    if head_k_dim != HEAD_DIM || head_v_dim != HEAD_DIM || conv_taps != CONV_TAPS {
        return false;
    }
    match device {
        Device::Cpu => true,
        Device::Cuda(_) => super::fp8::device_is_sm80_or_newer(device),
        _ => false,
    }
}

/// Causal-conv decode step. `x` holds `conv_dim` activations (any shape),
/// `weight` is `[conv_dim, 4]` F32, and `history` is the three F32 slots oldest
/// first. Returns `(silu(conv) [conv_dim] F32, float(x) [conv_dim] F32)`.
pub fn conv_decode(x: &Tensor, weight: &Tensor, history: [&Tensor; 3]) -> Result<(Tensor, Tensor)> {
    let conv_dim = x.elem_count();
    if weight.dims() != [conv_dim, CONV_TAPS]
        || weight.dtype() != DType::F32
        || history
            .iter()
            .any(|slot| slot.elem_count() != conv_dim || slot.dtype() != DType::F32)
        || !matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        candle_core::bail!(
            "invalid DeltaNet conv decode contract: x {:?} {:?}, weight {:?}",
            x.dims(),
            x.dtype(),
            weight.dims()
        )
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::conv(x, weight, history, conv_dim);
    }
    if !x.device().is_cpu() {
        candle_core::bail!("fused DeltaNet conv decode has no implementation for this device")
    }
    let xs = host(x)?;
    let w = host(weight)?;
    let h = [host(history[0])?, host(history[1])?, host(history[2])?];
    let out = (0..conv_dim)
        .map(|c| {
            let wc = &w[c * CONV_TAPS..][..CONV_TAPS];
            let v = xs[c] * wc[3] + h[0][c] * wc[0] + h[1][c] * wc[1] + h[2][c] * wc[2];
            v / (1.0 + (-v).exp())
        })
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_vec(out, conv_dim, x.device())?,
        Tensor::from_vec(xs, conv_dim, x.device())?,
    ))
}

/// Recurrent decode step for one token.
///
/// - `conv`: `[conv_dim]` F32 from [`conv_decode`].
/// - `z`: `value_heads * 128` gate activations.
/// - `beta_raw`, `alpha`: `value_heads` pre-activation projections, in `z`'s
///   dtype.
/// - `dt_bias`, `a`: `value_heads` F32, with `a = -exp(A_log)`.
/// - `norm_weight`: `[128]` F32.
/// - `state`: `[1, value_heads, 128, 128]` F32.
///
/// Returns `(y [value_heads * 128] in z's dtype, next state [1, Hv, 128, 128] F32)`.
#[allow(clippy::too_many_arguments)]
pub fn recurrent_decode(
    conv: &Tensor,
    z: &Tensor,
    beta_raw: &Tensor,
    alpha: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    norm_weight: &Tensor,
    state: &Tensor,
    spec: &GdnDecodeSpec,
) -> Result<(Tensor, Tensor)> {
    spec.validate()?;
    let (hv, d) = (spec.value_heads, HEAD_DIM);
    let dtype = z.dtype();
    if conv.elem_count() != spec.conv_dim()
        || conv.dtype() != DType::F32
        || z.elem_count() != hv * d
        || beta_raw.elem_count() != hv
        || alpha.elem_count() != hv
        || beta_raw.dtype() != dtype
        || alpha.dtype() != dtype
        || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
        || dt_bias.elem_count() != hv
        || a.elem_count() != hv
        || dt_bias.dtype() != DType::F32
        || a.dtype() != DType::F32
        || norm_weight.dims() != [d]
        || norm_weight.dtype() != DType::F32
        || state.elem_count() != hv * d * d
        || state.dtype() != DType::F32
    {
        candle_core::bail!("invalid DeltaNet recurrent decode contract for {spec:?}")
    }
    #[cfg(feature = "cuda")]
    if conv.device().is_cuda() {
        return cuda_impl::recurrent(
            conv,
            z,
            beta_raw,
            alpha,
            dt_bias,
            a,
            norm_weight,
            state,
            spec,
        );
    }
    if !conv.device().is_cpu() {
        candle_core::bail!("fused DeltaNet decode has no implementation for this device")
    }
    let (y, next) = recurrent_reference(
        &host(conv)?,
        &host(z)?,
        &host(beta_raw)?,
        &host(alpha)?,
        &host(dt_bias)?,
        &host(a)?,
        &host(norm_weight)?,
        &host(state)?,
        spec,
    );
    Ok((
        Tensor::from_vec(y, hv * d, conv.device())?.to_dtype(dtype)?,
        Tensor::from_vec(next, (1, hv, d, d), conv.device())?,
    ))
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

#[allow(clippy::too_many_arguments)]
fn recurrent_reference(
    conv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    dt_bias: &[f32],
    a: &[f32],
    norm_weight: &[f32],
    state: &[f32],
    spec: &GdnDecodeSpec,
) -> (Vec<f32>, Vec<f32>) {
    let d = HEAD_DIM;
    let key_width = spec.key_heads * d;
    let mut y = vec![0f32; spec.value_heads * d];
    let mut next = vec![0f32; state.len()];
    for h in 0..spec.value_heads {
        let kh = spec.key_head(h);
        let q = &conv[kh * d..][..d];
        let k = &conv[key_width + kh * d..][..d];
        let v = &conv[2 * key_width + h * d..][..d];
        let qsum: f32 = q.iter().map(|x| x * x).sum();
        let ksum: f32 = k.iter().map(|x| x * x).sum();
        let qscale = 1.0 / ((qsum + 1e-6).sqrt() * (d as f32).sqrt());
        let knorm = 1.0 / (ksum + 1e-6).sqrt();
        let gate_in = alpha[h] + dt_bias[h];
        let softplus = gate_in.max(0.0) + (-gate_in.abs()).exp().ln_1p();
        let decay = (softplus * a[h]).exp();
        let beta = 1.0 / (1.0 + (-beta_raw[h]).exp());
        let s_in = &state[h * d * d..][..d * d];
        let s_out = &mut next[h * d * d..][..d * d];
        let mut o = vec![0f32; d];
        for col in 0..d {
            let recalled: f32 = (0..d)
                .map(|row| k[row] * knorm * (s_in[row * d + col] * decay))
                .sum();
            let delta = (v[col] - recalled) * beta;
            for row in 0..d {
                let updated = (k[row] * knorm).mul_add(delta, s_in[row * d + col] * decay);
                s_out[row * d + col] = updated;
                o[col] += q[row] * qscale * updated;
            }
        }
        let mean_sq = o.iter().map(|x| x * x).sum::<f32>() / d as f32;
        let inv = 1.0 / (mean_sq + spec.norm_eps).sqrt();
        for col in 0..d {
            let zz = z[h * d + col];
            let gate = zz / (1.0 + (-zz).exp());
            y[h * d + col] = o[col] * inv * norm_weight[col] * gate;
        }
    }
    (y, next)
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{GdnDecodeSpec, HEAD_DIM};
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaDevice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};

    const MODULE: &str = "izwi_qwen36moe";

    fn view<'a, T: CudaDType>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("fused DeltaNet inputs must be CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("fused DeltaNet inputs must be contiguous")
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

    pub(super) fn conv(
        x: &Tensor,
        weight: &Tensor,
        history: [&Tensor; 3],
        conv_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = x.device().as_cuda_device()?;
        let x = x.contiguous()?;
        let weight = weight.contiguous()?;
        let h = [
            history[0].contiguous()?,
            history[1].contiguous()?,
            history[2].contiguous()?,
        ];
        let (x_storage, _) = x.storage_and_layout();
        let (w_storage, _) = weight.storage_and_layout();
        let (h0_storage, _) = h[0].storage_and_layout();
        let (h1_storage, _) = h[1].storage_and_layout();
        let (h2_storage, _) = h[2].storage_and_layout();
        let w = view::<f32>(&w_storage, &weight)?;
        let h0 = view::<f32>(&h0_storage, &h[0])?;
        let h1 = view::<f32>(&h1_storage, &h[1])?;
        let h2 = view::<f32>(&h2_storage, &h[2])?;
        // SAFETY: one thread writes each channel of both outputs.
        let out = unsafe { device.alloc::<f32>(conv_dim)? };
        let cur = unsafe { device.alloc::<f32>(conv_dim)? };
        let config = LaunchConfig {
            grid_dim: ((conv_dim as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! launch {
            ($ty:ty, $name:literal) => {{
                let xv = view::<$ty>(&x_storage, &x)?;
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = function.builder();
                builder.arg(&xv);
                builder.arg(&w);
                builder.arg(&h0);
                builder.arg(&h1);
                builder.arg(&h2);
                builder.arg(&out);
                builder.arg(&cur);
                candle_core::builder_arg!(builder, conv_dim as i32);
                // SAFETY: argument order and types match the kernel signature.
                unsafe { builder.launch(config) }.w()?;
            }};
        }
        match x.dtype() {
            DType::BF16 => launch!(half::bf16, "qwen36moe_gdn_conv_bf16"),
            DType::F16 => launch!(half::f16, "qwen36moe_gdn_conv_f16"),
            DType::F32 => launch!(f32, "qwen36moe_gdn_conv_f32"),
            other => candle_core::bail!("fused DeltaNet conv does not support {other:?}"),
        }
        Ok((
            wrap(out, device, Shape::from(conv_dim)),
            wrap(cur, device, Shape::from(conv_dim)),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn recurrent(
        conv: &Tensor,
        z: &Tensor,
        beta_raw: &Tensor,
        alpha: &Tensor,
        dt_bias: &Tensor,
        a: &Tensor,
        norm_weight: &Tensor,
        state: &Tensor,
        spec: &GdnDecodeSpec,
    ) -> Result<(Tensor, Tensor)> {
        let device = conv.device().as_cuda_device()?;
        let (hv, d) = (spec.value_heads, HEAD_DIM);
        let tensors =
            [conv, z, beta_raw, alpha, dt_bias, a, norm_weight, state].map(|t| t.contiguous());
        let [conv, z, beta_raw, alpha, dt_bias, a, norm_weight, state] = tensors;
        let (conv, z, beta_raw, alpha) = (conv?, z?, beta_raw?, alpha?);
        let (dt_bias, a, norm_weight, state) = (dt_bias?, a?, norm_weight?, state?);
        let (conv_s, _) = conv.storage_and_layout();
        let (z_s, _) = z.storage_and_layout();
        let (b_s, _) = beta_raw.storage_and_layout();
        let (al_s, _) = alpha.storage_and_layout();
        let (dt_s, _) = dt_bias.storage_and_layout();
        let (a_s, _) = a.storage_and_layout();
        let (n_s, _) = norm_weight.storage_and_layout();
        let (st_s, _) = state.storage_and_layout();
        let conv_v = view::<f32>(&conv_s, &conv)?;
        let dt_v = view::<f32>(&dt_s, &dt_bias)?;
        let a_v = view::<f32>(&a_s, &a)?;
        let n_v = view::<f32>(&n_s, &norm_weight)?;
        let st_v = view::<f32>(&st_s, &state)?;
        // SAFETY: every state element and every output channel is written.
        let next = unsafe { device.alloc::<f32>(hv * d * d)? };
        let config = LaunchConfig {
            grid_dim: (hv as u32, 1, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! launch {
            ($ty:ty, $name:literal) => {{
                let zv = view::<$ty>(&z_s, &z)?;
                let bv = view::<$ty>(&b_s, &beta_raw)?;
                let alv = view::<$ty>(&al_s, &alpha)?;
                // SAFETY: the kernel writes every output channel.
                let y = unsafe { device.alloc::<$ty>(hv * d)? };
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = function.builder();
                builder.arg(&conv_v);
                builder.arg(&zv);
                builder.arg(&bv);
                builder.arg(&alv);
                builder.arg(&dt_v);
                builder.arg(&a_v);
                builder.arg(&n_v);
                builder.arg(&st_v);
                builder.arg(&next);
                builder.arg(&y);
                candle_core::builder_arg!(
                    builder,
                    spec.key_heads as i32,
                    spec.value_heads as i32,
                    i32::from(spec.grouped),
                    spec.norm_eps
                );
                // SAFETY: validated layout; argument order matches the kernel.
                unsafe { builder.launch(config) }.w()?;
                wrap(y, device, Shape::from(hv * d))
            }};
        }
        let y = match z.dtype() {
            DType::BF16 => launch!(half::bf16, "qwen36moe_gdn_decode_bf16"),
            DType::F16 => launch!(half::f16, "qwen36moe_gdn_decode_f16"),
            DType::F32 => launch!(f32, "qwen36moe_gdn_decode_f32"),
            other => candle_core::bail!("fused DeltaNet decode does not support {other:?}"),
        };
        Ok((y, wrap(next, device, Shape::from((1, hv, d, d)))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * scale)
            .collect()
    }

    fn tensor(v: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn conv_decode_matches_the_tap_formula_and_returns_the_ring_slot() {
        let conv_dim = 5;
        let x = tensor(vec![0.5, -1.0, 2.0, 0.0, 1.5], &[1, 1, conv_dim]);
        let w = tensor(values(conv_dim * 4, 1.0, 0.5), &[conv_dim, 4]);
        let h: Vec<Tensor> = (0..3)
            .map(|i| tensor(values(conv_dim, 10.0 * i as f32, 1.0), &[conv_dim, 1]))
            .collect();
        let (out, cur) = conv_decode(&x, &w, [&h[0], &h[1], &h[2]]).unwrap();
        let (xs, ws) = (host(&x).unwrap(), host(&w).unwrap());
        let hs: Vec<Vec<f32>> = h.iter().map(|t| host(t).unwrap()).collect();
        for (c, got) in out.to_vec1::<f32>().unwrap().iter().enumerate() {
            let v = xs[c] * ws[c * 4 + 3]
                + hs[0][c] * ws[c * 4]
                + hs[1][c] * ws[c * 4 + 1]
                + hs[2][c] * ws[c * 4 + 2];
            assert!((got - v / (1.0 + (-v).exp())).abs() < 1e-6);
        }
        assert_eq!(cur.to_vec1::<f32>().unwrap(), xs);
    }

    #[test]
    fn recurrent_decode_head_mapping_follows_the_declared_order() {
        // With two key heads and four value heads, grouped and tiled orders
        // pair different key heads with value heads 1 and 2.
        let spec = GdnDecodeSpec {
            key_heads: 2,
            value_heads: 4,
            grouped: true,
            norm_eps: 1e-6,
        };
        assert_eq!(
            (0..4).map(|v| spec.key_head(v)).collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
        let tiled = GdnDecodeSpec {
            grouped: false,
            ..spec
        };
        assert_eq!(
            (0..4).map(|v| tiled.key_head(v)).collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
        let d = HEAD_DIM;
        let conv = tensor(values(spec.conv_dim(), 3.0, 1.0), &[spec.conv_dim()]);
        let z = tensor(values(4 * d, 5.0, 1.0), &[4 * d]);
        let small = |seed| tensor(values(4, seed, 1.0), &[4]);
        let a = tensor(vec![-0.5, -1.0, -0.25, -2.0], &[4]);
        let state = tensor(values(4 * d * d, 7.0, 0.2), &[1, 4, d, d]);
        let norm = tensor(vec![1.0; d], &[d]);
        let run = |spec: &GdnDecodeSpec| {
            recurrent_decode(
                &conv,
                &z,
                &small(1.0),
                &small(2.0),
                &small(4.0),
                &a,
                &norm,
                &state,
                spec,
            )
            .unwrap()
            .0
            .to_vec1::<f32>()
            .unwrap()
        };
        let (grouped_y, tiled_y) = (run(&spec), run(&tiled));
        assert_eq!(
            grouped_y[..d],
            tiled_y[..d],
            "value head 0 reads key head 0 in both orders"
        );
        assert_ne!(
            grouped_y[d..2 * d],
            tiled_y[d..2 * d],
            "value head 1 must differ"
        );
        assert!(recurrent_decode(
            &conv,
            &z,
            &small(1.0),
            &small(2.0),
            &small(4.0),
            &a,
            &norm,
            &state,
            &GdnDecodeSpec {
                key_heads: 3,
                ..spec
            }
        )
        .is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gdn_decode_matches_cpu_reference_at_trunk_geometry() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let d = HEAD_DIM;
        for grouped in [true, false] {
            let spec = GdnDecodeSpec {
                key_heads: 16,
                value_heads: 32,
                grouped,
                norm_eps: 1e-6,
            };
            let conv_dim = spec.conv_dim();
            let x = tensor(values(conv_dim, 1.0, 2.0), &[1, 1, conv_dim])
                .to_dtype(DType::BF16)
                .unwrap();
            let w = tensor(values(conv_dim * 4, 2.0, 0.5), &[conv_dim, 4]);
            let h: Vec<Tensor> = (0..3)
                .map(|i| tensor(values(conv_dim, 9.0 + i as f32, 1.0), &[conv_dim, 1]))
                .collect();
            let z = tensor(values(32 * d, 5.0, 2.0), &[32 * d])
                .to_dtype(DType::BF16)
                .unwrap();
            let beta = tensor(values(32, 6.0, 2.0), &[32])
                .to_dtype(DType::BF16)
                .unwrap();
            let alpha = tensor(values(32, 7.0, 3.0), &[32])
                .to_dtype(DType::BF16)
                .unwrap();
            let dt = tensor(values(32, 8.0, 1.0), &[32]);
            let a = tensor(
                values(32, 9.0, 1.0).iter().map(|v| -v.exp()).collect(),
                &[32],
            );
            let norm = tensor(values(d, 10.0, 0.2).iter().map(|v| 1.0 + v).collect(), &[d]);
            let state = tensor(values(32 * d * d, 11.0, 0.3), &[1, 32, d, d]);
            let cpu = {
                let (conv, cur) = conv_decode(&x, &w, [&h[0], &h[1], &h[2]]).unwrap();
                let (y, next) =
                    recurrent_decode(&conv, &z, &beta, &alpha, &dt, &a, &norm, &state, &spec)
                        .unwrap();
                (
                    host(&conv).unwrap(),
                    host(&cur).unwrap(),
                    host(&y).unwrap(),
                    host(&next).unwrap(),
                )
            };
            let g = |t: &Tensor| t.to_device(&device).unwrap();
            let gpu = {
                let (conv, cur) =
                    conv_decode(&g(&x), &g(&w), [&g(&h[0]), &g(&h[1]), &g(&h[2])]).unwrap();
                let (y, next) = recurrent_decode(
                    &conv,
                    &g(&z),
                    &g(&beta),
                    &g(&alpha),
                    &g(&dt),
                    &g(&a),
                    &g(&norm),
                    &g(&state),
                    &spec,
                )
                .unwrap();
                (
                    host(&conv).unwrap(),
                    host(&cur).unwrap(),
                    host(&y).unwrap(),
                    host(&next).unwrap(),
                )
            };
            let close = |a: &[f32], b: &[f32], tol: f32, label: &str| {
                let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    assert!(
                        (x - y).abs() <= tol * scale,
                        "{label} grouped={grouped} index {i}: {x} vs {y}"
                    );
                }
            };
            close(&gpu.0, &cpu.0, 1e-5, "conv");
            assert_eq!(gpu.1, cpu.1, "ring slot must be the exact F32 input");
            close(&gpu.2, &cpu.2, 1e-2, "y");
            close(&gpu.3, &cpu.3, 1e-5, "state");
        }
    }
}
