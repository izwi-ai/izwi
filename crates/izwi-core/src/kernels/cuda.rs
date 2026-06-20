//! CUDA kernel dispatch.
//!
//! This module wires CUDA-only fused-operation entry points to Candle CUDA
//! tensor kernels where Candle provides the primitive. These paths stay guarded
//! by `Device::is_cuda()` and fall back to the caller's existing implementation
//! when a shape, dtype, or build does not support the operation.

use candle_core::{DType, Tensor, D};

use crate::kernels::FusedSiluMulResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelStatus {
    pub compiled: bool,
    pub available: bool,
    pub reason: &'static str,
}

pub fn cuda_kernels_compiled() -> bool {
    cfg!(feature = "cuda")
}

pub fn fused_kernels_available() -> bool {
    cuda_kernels_compiled()
}

pub fn use_block_fusion() -> bool {
    false
}

pub fn status() -> CudaKernelStatus {
    if !cuda_kernels_compiled() {
        return CudaKernelStatus {
            compiled: false,
            available: false,
            reason: "binary was not built with CUDA support",
        };
    }

    CudaKernelStatus {
        compiled: true,
        available: true,
        reason: "Candle CUDA kernel dispatch is enabled",
    }
}

pub fn try_fused_silu_mul(gate: &Tensor, up: &Tensor) -> Option<Tensor> {
    try_fused_silu_mul_with_status(gate, up).map(|result| result.tensor)
}

pub fn try_fused_silu_mul_with_status(gate: &Tensor, up: &Tensor) -> Option<FusedSiluMulResult> {
    if !cuda_tensor_pair_supported(gate, up) {
        return None;
    }

    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    let tensor = silu_gate.broadcast_mul(up).ok()?;
    Some(FusedSiluMulResult {
        tensor,
        used_custom_kernel: false,
    })
}

pub fn try_fused_l2_norm(input: &Tensor, eps: f64) -> Option<Tensor> {
    if !cuda_tensor_supported(input) || input.dtype() != DType::F32 {
        return None;
    }

    input
        .broadcast_div(
            &(input.sqr().ok()?.sum_keepdim(D::Minus1).ok()? + eps)
                .ok()?
                .sqrt()
                .ok()?,
        )
        .ok()
}

pub fn try_fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(input, weight) {
        return None;
    }

    candle_nn::ops::rms_norm(input, weight, eps as f32).ok()
}

pub fn try_fused_gated_rms_norm(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(hidden, gate) || !cuda_tensor_pair_supported(hidden, weight) {
        return None;
    }

    let normalized = try_fused_rms_norm(hidden, weight, eps)?;
    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    normalized.broadcast_mul(&silu_gate).ok()
}

pub fn try_fused_gated_delta_recurrent(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(query, key)
        || !cuda_tensor_pair_supported(query, value)
        || !cuda_tensor_pair_supported(query, g)
        || !cuda_tensor_pair_supported(query, beta)
        || !cuda_tensor_pair_supported(query, state)
        || query.dtype() != DType::F32
    {
        return None;
    }

    fused_gated_delta_candle(query, key, value, g, beta, state)
}

pub fn try_tiled_deltanet_recurrence(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
    tile_size: usize,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(queries, keys)
        || !cuda_tensor_pair_supported(queries, values)
        || !cuda_tensor_pair_supported(queries, g)
        || !cuda_tensor_pair_supported(queries, beta)
        || !cuda_tensor_pair_supported(queries, initial_state)
        || queries.dtype() != DType::F32
    {
        return None;
    }

    let (batch, seq_len, num_heads, head_k_dim) = queries.dims4().ok()?;
    let (k_batch, k_seq_len, k_num_heads, k_head_k_dim) = keys.dims4().ok()?;
    let (v_batch, v_seq_len, v_num_heads, _v_head_dim) = values.dims4().ok()?;
    let (g_batch, g_seq_len, g_heads) = g.dims3().ok()?;
    let (b_batch, b_seq_len, b_heads) = beta.dims3().ok()?;
    let (s_batch, s_heads, s_head_k_dim, _s_head_v_dim) = initial_state.dims4().ok()?;

    if batch != 1
        || k_batch != batch
        || v_batch != batch
        || g_batch != batch
        || b_batch != batch
        || s_batch != batch
    {
        return None;
    }
    if k_seq_len != seq_len || v_seq_len != seq_len || g_seq_len != seq_len || b_seq_len != seq_len
    {
        return None;
    }
    if k_num_heads != num_heads || v_num_heads != num_heads || g_heads != num_heads {
        return None;
    }
    if b_heads != num_heads || k_head_k_dim != head_k_dim || s_heads != num_heads {
        return None;
    }
    if s_head_k_dim != head_k_dim {
        return None;
    }

    let tile_size = tile_size.max(1).min(seq_len.max(1));
    let mut outputs = Vec::with_capacity(seq_len);
    let mut state = initial_state.clone();

    for tile_start in (0..seq_len).step_by(tile_size) {
        let tile_end = (tile_start + tile_size).min(seq_len);
        for token_idx in tile_start..tile_end {
            let query = queries.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let key = keys.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let value = values.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let g_t = g.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let beta_t = beta.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;

            let (output, next_state) =
                fused_gated_delta_candle(&query, &key, &value, &g_t, &beta_t, &state)?;
            outputs.push(output.unsqueeze(1).ok()?);
            state = next_state;
        }
    }

    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let outputs = Tensor::cat(&output_refs, 1).ok()?;
    Some((outputs, state))
}

fn cuda_tensor_supported(tensor: &Tensor) -> bool {
    cuda_kernels_compiled() && tensor.device().is_cuda()
}

fn cuda_tensor_pair_supported(lhs: &Tensor, rhs: &Tensor) -> bool {
    cuda_tensor_supported(lhs) && rhs.device().is_cuda() && lhs.dtype() == rhs.dtype()
}

fn fused_gated_delta_candle(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    let dim = query.dim(D::Minus1).ok()?;
    let scale = 1.0f64 / (dim as f64).sqrt();
    let scaled_query = (query * scale).ok()?;
    let g = g.exp().ok()?.reshape((1, g.dim(1).ok()?, 1, 1)).ok()?;
    let beta = beta.reshape((1, beta.dim(1).ok()?, 1)).ok()?;

    let gated_state = state.broadcast_mul(&g).ok()?;
    let kv_mem = key
        .unsqueeze(2)
        .ok()?
        .matmul(&gated_state)
        .ok()?
        .squeeze(2)
        .ok()?;
    let delta = (value - kv_mem).ok()?.broadcast_mul(&beta).ok()?;
    let new_state = (&gated_state
        + &key
            .unsqueeze(3)
            .ok()?
            .matmul(&delta.unsqueeze(2).ok()?)
            .ok()?)
        .ok()?;
    let output = scaled_query
        .unsqueeze(2)
        .ok()?
        .matmul(&new_state)
        .ok()?
        .squeeze(2)
        .ok()?;

    Some((output, new_state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_kernel_status_is_explicit() {
        let status = status();
        assert_eq!(status.compiled, cfg!(feature = "cuda"));
        assert_eq!(status.available, cfg!(feature = "cuda"));
        assert!(!status.reason.trim().is_empty());
    }

    #[test]
    fn cuda_candle_dispatch_rejects_cpu_tensors() {
        let device = candle_core::Device::Cpu;
        let lhs = Tensor::zeros((1, 2), DType::F32, &device).expect("lhs");
        let rhs = Tensor::zeros((1, 2), DType::F32, &device).expect("rhs");

        assert!(try_fused_silu_mul(&lhs, &rhs).is_none());
        assert!(try_fused_l2_norm(&lhs, 1e-6).is_none());
        assert!(try_fused_rms_norm(&lhs, &rhs, 1e-6).is_none());
        assert!(try_fused_gated_rms_norm(&lhs, &rhs, &rhs, 1e-6).is_none());
    }
}
