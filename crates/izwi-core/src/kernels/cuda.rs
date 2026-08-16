//! CUDA kernel dispatch.
//!
//! This module wires CUDA-only fused-operation entry points to Candle CUDA
//! tensor kernels where Candle provides the primitive. These paths stay guarded
//! by `Device::is_cuda()` and fall back to the caller's existing implementation
//! when a shape, dtype, or build does not support the operation.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Tensor, D};

#[cfg(feature = "cuda")]
use std::cell::RefCell;
#[cfg(feature = "cuda")]
use std::collections::VecDeque;

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

pub fn try_fused_qk_rms_norm(
    q: &Tensor,
    k: &Tensor,
    qk_weight: &Tensor,
    eps: f64,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(q, k) || !cuda_tensor_pair_supported(q, qk_weight) {
        return None;
    }
    candle_qk_rms_norm(q, k, qk_weight, eps)
}

fn candle_qk_rms_norm(
    q: &Tensor,
    k: &Tensor,
    qk_weight: &Tensor,
    eps: f64,
) -> Option<(Tensor, Tensor)> {
    let (q_batch, q_sequence, _q_heads, head_dim) = q.dims4().ok()?;
    let (k_batch, k_sequence, _k_heads, k_head_dim) = k.dims4().ok()?;
    if q_batch != k_batch
        || q_sequence != k_sequence
        || head_dim == 0
        || head_dim != k_head_dim
        || q.dtype() != k.dtype()
        || q.dtype() != qk_weight.dtype()
        || q.device().location() != k.device().location()
        || q.device().location() != qk_weight.device().location()
        || qk_weight.dims() != [head_dim.checked_mul(2)?]
    {
        return None;
    }
    let q_weight = qk_weight.narrow(0, 0, head_dim).ok()?;
    let k_weight = qk_weight.narrow(0, head_dim, head_dim).ok()?;
    Some((
        candle_nn::ops::rms_norm(q, &q_weight, eps as f32).ok()?,
        candle_nn::ops::rms_norm(k, &k_weight, eps as f32).ok()?,
    ))
}

pub fn try_fused_rope_pair_bshd(
    q: &Tensor,
    k: &Tensor,
    cos_sin: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(q, k) || !cuda_tensor_pair_supported(q, cos_sin) {
        return None;
    }
    candle_rope_pair_bshd(q, k, cos_sin)
}

fn candle_rope_pair_bshd(q: &Tensor, k: &Tensor, cos_sin: &Tensor) -> Option<(Tensor, Tensor)> {
    let (q_batch, sequence, _q_heads, head_dim) = q.dims4().ok()?;
    let (k_batch, k_sequence, _k_heads, k_head_dim) = k.dims4().ok()?;
    if q_batch != k_batch
        || sequence == 0
        || sequence != k_sequence
        || head_dim == 0
        || head_dim % 2 != 0
        || head_dim != k_head_dim
        || q.dtype() != k.dtype()
        || q.dtype() != cos_sin.dtype()
        || q.device().location() != k.device().location()
        || q.device().location() != cos_sin.device().location()
        || cos_sin.dims() != [sequence, head_dim]
    {
        return None;
    }
    let half_dim = head_dim / 2;
    let cos = cos_sin
        .narrow(1, 0, half_dim)
        .ok()?
        .reshape((1, sequence, 1, half_dim))
        .ok()?;
    let sin = cos_sin
        .narrow(1, half_dim, half_dim)
        .ok()?
        .reshape((1, sequence, 1, half_dim))
        .ok()?;
    let rotate = |input: &Tensor| -> Option<Tensor> {
        let first = input.narrow(D::Minus1, 0, half_dim).ok()?;
        let second = input.narrow(D::Minus1, half_dim, half_dim).ok()?;
        let rotated_first = first
            .broadcast_mul(&cos)
            .ok()?
            .broadcast_sub(&second.broadcast_mul(&sin).ok()?)
            .ok()?;
        let rotated_second = first
            .broadcast_mul(&sin)
            .ok()?
            .broadcast_add(&second.broadcast_mul(&cos).ok()?)
            .ok()?;
        Tensor::cat(&[&rotated_first, &rotated_second], D::Minus1).ok()
    };
    Some((rotate(q)?, rotate(k)?))
}

pub fn try_fused_decode_gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Option<Tensor> {
    let kv_len = k.dims4().ok()?.2;
    try_fused_decode_gqa_attention_with_kv_len(
        q,
        k,
        v,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn try_fused_decode_gqa_attention_with_kv_len(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(q, k) || !cuda_tensor_pair_supported(q, v) {
        return None;
    }
    candle_decode_gqa_attention(q, k, v, num_heads, num_kv_heads, head_dim, kv_len, scale)
}

#[allow(clippy::too_many_arguments)]
fn candle_decode_gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Option<Tensor> {
    let (q_batch, q_heads, q_sequence, q_dim) = q.dims4().ok()?;
    let (k_batch, k_heads, k_capacity, k_dim) = k.dims4().ok()?;
    let (v_batch, v_heads, v_capacity, v_dim) = v.dims4().ok()?;
    if q_batch != 1
        || k_batch != 1
        || v_batch != 1
        || q_sequence != 1
        || q_heads != num_heads
        || k_heads != num_kv_heads
        || v_heads != num_kv_heads
        || q_dim != head_dim
        || k_dim != head_dim
        || v_dim != head_dim
        || k_capacity != v_capacity
        || kv_len == 0
        || kv_len > k_capacity
        || num_heads == 0
        || num_kv_heads == 0
        || !num_heads.is_multiple_of(num_kv_heads)
        || !scale.is_finite()
        || scale <= 0.0
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
        || q.device().location() != k.device().location()
        || q.device().location() != v.device().location()
    {
        return None;
    }
    let groups = num_heads / num_kv_heads;
    let queries = q.reshape((num_kv_heads, groups, 1, head_dim)).ok()?;
    let keys = k
        .narrow(2, 0, kv_len)
        .ok()?
        .squeeze(0)
        .ok()?
        .unsqueeze(1)
        .ok()?
        .transpose(2, 3)
        .ok()?;
    let scores = (queries.broadcast_matmul(&keys).ok()? * scale as f64).ok()?;
    let probabilities = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32).ok()?)
        .ok()?
        .to_dtype(q.dtype())
        .ok()?;
    let values = v
        .narrow(2, 0, kv_len)
        .ok()?
        .squeeze(0)
        .ok()?
        .unsqueeze(1)
        .ok()?;
    probabilities
        .broadcast_matmul(&values)
        .ok()?
        .reshape((1, num_heads, 1, head_dim))
        .ok()
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

pub fn try_qwen35_causal_conv_sequence(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    try_qwen_hybrid_causal_conv_sequence(input, weight, history)
}

pub fn try_qwen38_causal_conv_sequence(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    try_qwen_hybrid_causal_conv_sequence(input, weight, history)
}

pub fn try_qwen38_causal_conv_decode(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(input, weight)
        || !cuda_tensor_pair_supported(input, history)
        || input.dtype() != DType::F32
    {
        return None;
    }
    let (batch, sequence, conv_dim) = input.dims3().ok()?;
    let (weight_channels, kernel_size) = weight.dims2().ok()?;
    let (history_channels, history_len) = history.dims2().ok()?;
    if batch != 1
        || sequence != 1
        || conv_dim == 0
        || kernel_size != 4
        || weight_channels != conv_dim
        || history_channels != conv_dim
        || history_len != 3
    {
        return None;
    }

    let input = input.contiguous().ok()?;
    let weight = weight.contiguous().ok()?;
    let history = history.contiguous().ok()?;
    let state_elements = conv_dim.checked_mul(history_len)?;
    let packed = input
        .apply_op3_no_bwd(
            &weight,
            &history,
            &CudaQwen38CausalConvDecodeOp { conv_dim },
        )
        .ok()?;
    let output = packed
        .narrow(0, 0, conv_dim)
        .ok()?
        .reshape((1, 1, conv_dim))
        .ok()?;
    let next_history = packed
        .narrow(0, conv_dim, state_elements)
        .ok()?
        .reshape((conv_dim, history_len))
        .ok()?;
    Some((output, next_history))
}

fn try_qwen_hybrid_causal_conv_sequence(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(input, weight)
        || !cuda_tensor_pair_supported(input, history)
        || input.dtype() != DType::F32
    {
        return None;
    }
    let (batch, sequence, conv_dim) = input.dims3().ok()?;
    let (weight_channels, kernel_size) = weight.dims2().ok()?;
    let (history_channels, history_len) = history.dims2().ok()?;
    if batch != 1
        || sequence == 0
        || conv_dim == 0
        || kernel_size < 2
        || weight_channels != conv_dim
        || history_channels != conv_dim
        || history_len != kernel_size - 1
    {
        return None;
    }

    let input = input.contiguous().ok()?;
    let weight = weight.contiguous().ok()?;
    let history = history.contiguous().ok()?;
    let output_elements = sequence.checked_mul(conv_dim)?;
    let state_elements = history_len.checked_mul(conv_dim)?;
    let packed = input
        .apply_op3_no_bwd(
            &weight,
            &history,
            &CudaCausalConvSequenceOp {
                conv_dim,
                sequence,
                kernel_size,
            },
        )
        .ok()?;
    let output = packed
        .narrow(0, 0, output_elements)
        .ok()?
        .reshape((1, sequence, conv_dim))
        .ok()?;
    let final_history = packed
        .narrow(0, output_elements, state_elements)
        .ok()?
        .reshape((conv_dim, history_len))
        .ok()?;
    Some((output, final_history))
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

    let queries = query.unsqueeze(1).ok()?;
    let keys = key.unsqueeze(1).ok()?;
    let values = value.unsqueeze(1).ok()?;
    let gates = g.unsqueeze(1).ok()?;
    let betas = beta.unsqueeze(1).ok()?;
    let (outputs, next_state) =
        cuda_gated_delta_sequence(&queries, &keys, &values, &gates, &betas, state)?;
    Some((outputs.squeeze(1).ok()?, next_state))
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
        || tile_size == 0
    {
        return None;
    }

    let (batch, seq_len, num_heads, head_k_dim) = queries.dims4().ok()?;
    let (k_batch, k_seq_len, k_num_heads, k_head_k_dim) = keys.dims4().ok()?;
    let (v_batch, v_seq_len, v_num_heads, v_head_dim) = values.dims4().ok()?;
    let (g_batch, g_seq_len, g_heads) = g.dims3().ok()?;
    let (b_batch, b_seq_len, b_heads) = beta.dims3().ok()?;
    let (s_batch, s_heads, s_head_k_dim, s_head_v_dim) = initial_state.dims4().ok()?;

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
    if s_head_k_dim != head_k_dim || s_head_v_dim != v_head_dim {
        return None;
    }

    let tile_size = tile_size.min(seq_len.max(1));
    if tile_size >= seq_len {
        return cuda_gated_delta_sequence(queries, keys, values, g, beta, initial_state);
    }

    let mut outputs = Vec::with_capacity(seq_len.div_ceil(tile_size));
    let mut state = initial_state.clone();
    for token_start in (0..seq_len).step_by(tile_size) {
        let token_count = tile_size.min(seq_len - token_start);
        let query_tile = queries.narrow(1, token_start, token_count).ok()?;
        let key_tile = keys.narrow(1, token_start, token_count).ok()?;
        let value_tile = values.narrow(1, token_start, token_count).ok()?;
        let g_tile = g.narrow(1, token_start, token_count).ok()?;
        let beta_tile = beta.narrow(1, token_start, token_count).ok()?;
        let (output, next_state) = cuda_gated_delta_sequence(
            &query_tile,
            &key_tile,
            &value_tile,
            &g_tile,
            &beta_tile,
            &state,
        )?;
        outputs.push(output);
        state = next_state;
    }
    let output_refs = outputs.iter().collect::<Vec<_>>();
    Some((Tensor::cat(&output_refs, 1).ok()?, state))
}

fn validate_cuda_paged_decode_metadata(
    metadata: &[u32],
    batch: usize,
    page_tokens: usize,
    max_blocks: usize,
    capacity_pages: usize,
) -> candle_core::Result<()> {
    let expected_metadata = batch
        .checked_mul(2_usize.checked_add(max_blocks).ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
        })?)
        .ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
        })?;
    if metadata.len() != expected_metadata {
        candle_core::bail!(
            "CUDA paged decode metadata has {} entries, expected {expected_metadata}",
            metadata.len()
        )
    }
    if batch == 0 || page_tokens == 0 || max_blocks == 0 || capacity_pages == 0 {
        candle_core::bail!("CUDA paged decode metadata has invalid empty geometry")
    }

    let table_start = batch.checked_mul(2).ok_or_else(|| {
        candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
    })?;
    for row in 0..batch {
        let context_len = metadata[row] as usize;
        let first_page_offset = metadata[batch + row] as usize;
        if context_len == 0 || first_page_offset >= page_tokens {
            candle_core::bail!("CUDA paged decode metadata row {row} has an invalid context")
        }
        let physical_tokens = context_len.checked_add(first_page_offset).ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged decode context overflow".to_string())
        })?;
        if physical_tokens > u32::MAX as usize {
            candle_core::bail!(
                "CUDA paged decode metadata row {row} exceeds the unsigned 32-bit token index ABI"
            )
        }
        let required_pages = physical_tokens.div_ceil(page_tokens);
        if required_pages == 0 || required_pages > max_blocks {
            candle_core::bail!("CUDA paged decode metadata row {row} has an incomplete block table")
        }
        let row_start = table_start
            .checked_add(row.checked_mul(max_blocks).ok_or_else(|| {
                candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
            })?)
            .ok_or_else(|| {
                candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
            })?;
        let row_end = row_start.checked_add(required_pages).ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
        })?;
        if metadata[row_start..row_end]
            .iter()
            .any(|&page| page as usize >= capacity_pages)
        {
            candle_core::bail!(
                "CUDA paged decode metadata row {row} contains an out-of-bounds physical page"
            )
        }
    }
    Ok(())
}

// This is deliberately a conservative, compile-time routing policy until a
// CUDA certification runner can establish model- and GPU-specific thresholds.
const CUDA_PAGED_DECODE_SPLIT_THRESHOLD_TOKENS: usize = 2_048;
const CUDA_PAGED_DECODE_PARTITION_TOKENS: usize = 1_024;
const CUDA_PAGED_DECODE_MAX_PARTITIONS: usize = u16::MAX as usize;
const CUDA_PAGED_DECODE_MAX_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaPagedDecodeStrategy {
    OnePass,
    Partitioned { partitions: usize },
}

fn cuda_paged_decode_strategy(
    max_context_len: usize,
    batch: usize,
    query_heads: usize,
    value_dim: usize,
    tuning: Option<(usize, usize)>,
) -> candle_core::Result<CudaPagedDecodeStrategy> {
    let Some((split_threshold, partition_tokens)) = tuning else {
        return Ok(CudaPagedDecodeStrategy::OnePass);
    };
    if split_threshold == 0 || partition_tokens == 0 {
        candle_core::bail!("CUDA paged decode tuning contains a zero threshold")
    }
    if max_context_len <= split_threshold {
        return Ok(CudaPagedDecodeStrategy::OnePass);
    }
    let partitions = max_context_len.div_ceil(partition_tokens);
    if partitions > CUDA_PAGED_DECODE_MAX_PARTITIONS {
        candle_core::bail!(
            "CUDA paged decode requires {partitions} partitions, exceeding the kernel grid limit"
        )
    }
    let workspace_bytes = batch
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(partitions))
        .and_then(|value| value.checked_mul(value_dim.checked_add(2)?))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .unwrap_or(usize::MAX);
    if workspace_bytes > CUDA_PAGED_DECODE_MAX_WORKSPACE_BYTES {
        return Ok(CudaPagedDecodeStrategy::OnePass);
    }
    Ok(CudaPagedDecodeStrategy::Partitioned { partitions })
}

fn cuda_paged_decode_page_tokens_supported(page_tokens: usize) -> bool {
    matches!(page_tokens, 16 | 32 | 64)
}

pub(crate) fn paged_prefill_attention(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    sequences: usize,
    total_queries: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    window_tokens: Option<u32>,
) -> candle_core::Result<Tensor> {
    let dense_kv = queries.dtype() == keys.dtype() && queries.dtype() == values.dtype();
    if keys.dtype() == DType::F8E4M3 || values.dtype() == DType::F8E4M3 {
        candle_core::bail!(
            "CUDA FP8 KV is disabled until scaled storage, scale accounting, and NVIDIA evidence are complete"
        )
    }
    if !queries.device().is_cuda()
        || queries.device().location() != keys.device().location()
        || queries.device().location() != values.device().location()
        || !dense_kv
        || !matches!(queries.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        candle_core::bail!(
            "CUDA paged prefill requires matching F32/F16/BF16 tensors on one CUDA device"
        )
    }
    let capacity_pages = keys.dims().first().copied().unwrap_or(0);
    let metadata_len = sequences
        .checked_mul(4_usize.checked_add(max_blocks).ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged prefill metadata overflow".to_string())
        })?)
        .ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged prefill metadata overflow".to_string())
        })?;
    if queries.dims() != [total_queries, query_heads, key_dim]
        || keys.dims().len() != 4
        || values.dims().len() != 4
        || keys.dims()[1..] != [page_tokens, kv_heads, key_dim]
        || values.dims()[0] != capacity_pages
        || values.dims()[1..] != [page_tokens, kv_heads, value_dim]
        || sequences == 0
        || total_queries == 0
        || query_heads == 0
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || page_tokens == 0
        || max_blocks == 0
        || key_dim == 0
        || key_dim > 512
        || value_dim == 0
        || value_dim > 512
        || capacity_pages == 0
        || metadata.device().location() != queries.device().location()
        || metadata.dtype() != DType::U32
        || metadata.dims() != [metadata_len]
        || !metadata.layout().is_contiguous()
        || !softmax_scale.is_finite()
        || softmax_scale <= 0.0
        || softcap.is_some_and(|value| !value.is_finite() || value <= 0.0)
        || window_tokens == Some(0)
    {
        candle_core::bail!("CUDA paged prefill received invalid tensor or attention geometry")
    }
    let geometry = [
        sequences,
        total_queries,
        query_heads,
        kv_heads,
        page_tokens,
        max_blocks,
        key_dim,
        value_dim,
        capacity_pages,
        metadata_len,
        window_tokens.unwrap_or(0) as usize,
        queries.elem_count(),
        keys.elem_count(),
        values.elem_count(),
    ];
    if geometry.iter().any(|&value| value > i32::MAX as usize) {
        candle_core::bail!("CUDA paged prefill exceeds the signed 32-bit kernel index ABI")
    }
    queries.contiguous()?.apply_op3_no_bwd(
        &keys.contiguous()?,
        &values.contiguous()?,
        &CudaPagedPrefillOp {
            metadata: metadata.clone(),
            sequences,
            total_queries,
            query_heads,
            kv_heads,
            page_tokens,
            max_blocks,
            key_dim,
            value_dim,
            capacity_pages,
            window_tokens: window_tokens.unwrap_or(0) as usize,
            softmax_scale,
            softcap,
        },
    )
}

pub(crate) fn paged_decode_attention(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    max_context_len: usize,
    partition_tuning: Option<(usize, usize)>,
) -> candle_core::Result<Tensor> {
    paged_decode_attention_with_graph(
        queries,
        keys,
        values,
        metadata,
        batch,
        query_heads,
        kv_heads,
        page_tokens,
        max_blocks,
        key_dim,
        value_dim,
        softmax_scale,
        softcap,
        max_context_len,
        partition_tuning,
        false,
        0,
    )
    .map(|(output, _)| output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaPagedDecodeGraphOutcome {
    Disabled,
    Warmed,
    WarmedAfterEviction,
    Captured,
    Replayed,
    Backoff,
    EagerFallback,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_decode_attention_with_graph(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    max_context_len: usize,
    partition_tuning: Option<(usize, usize)>,
    _allow_graph: bool,
    _backing_generation: u64,
) -> candle_core::Result<(Tensor, CudaPagedDecodeGraphOutcome)> {
    let dense_kv = queries.dtype() == keys.dtype() && queries.dtype() == values.dtype();
    if keys.dtype() == DType::F8E4M3 || values.dtype() == DType::F8E4M3 {
        candle_core::bail!(
            "CUDA FP8 KV is disabled until scaled storage, scale accounting, and NVIDIA evidence are complete"
        )
    }
    if !queries.device().is_cuda()
        || queries.device().location() != keys.device().location()
        || queries.device().location() != values.device().location()
        || !dense_kv
        || !matches!(queries.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        candle_core::bail!(
            "CUDA paged decode requires matching F32/F16/BF16 tensors on one CUDA device"
        )
    }
    if queries.dims() != [batch, query_heads, key_dim]
        || keys.dims().len() != 4
        || values.dims().len() != 4
        || keys.dims()[1..] != [page_tokens, kv_heads, key_dim]
        || values.dims()[0] != keys.dims()[0]
        || values.dims()[1..] != [page_tokens, kv_heads, value_dim]
        || batch == 0
        || query_heads == 0
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || key_dim == 0
        || key_dim > 512
        || value_dim == 0
        || value_dim > 512
        || !cuda_paged_decode_page_tokens_supported(page_tokens)
        || max_blocks == 0
        || !softmax_scale.is_finite()
        || softmax_scale <= 0.0
        || softcap.is_some_and(|softcap| !softcap.is_finite() || softcap <= 0.0)
    {
        candle_core::bail!("CUDA paged decode received invalid tensor or attention geometry")
    }
    let capacity_pages = keys.dims()[0];
    let metadata_len = batch
        .checked_mul(2_usize.checked_add(max_blocks).ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
        })?)
        .ok_or_else(|| {
            candle_core::Error::Msg("CUDA paged decode metadata overflow".to_string())
        })?;
    if metadata.device().location() != queries.device().location()
        || metadata.dtype() != DType::U32
        || metadata.dims() != [metadata_len]
        || !metadata.layout().is_contiguous()
    {
        candle_core::bail!("CUDA paged decode metadata must be contiguous U32 on the query device")
    }
    if max_context_len == 0 {
        candle_core::bail!("CUDA paged decode requires a non-empty validated context")
    }
    let strategy = cuda_paged_decode_strategy(
        max_context_len,
        batch,
        query_heads,
        value_dim,
        partition_tuning,
    )?;
    let kernel_geometry = [
        batch,
        query_heads,
        kv_heads,
        page_tokens,
        max_blocks,
        key_dim,
        value_dim,
        capacity_pages,
        metadata_len,
        queries.elem_count(),
        keys.elem_count(),
        values.elem_count(),
    ];
    if kernel_geometry
        .iter()
        .any(|&value| value > i32::MAX as usize)
    {
        candle_core::bail!("CUDA paged decode exceeds the signed 32-bit kernel index ABI")
    }
    let queries = queries.contiguous()?;
    let keys = keys.contiguous()?;
    let values = values.contiguous()?;
    #[cfg(feature = "cuda")]
    let graph_outcome = {
        if _allow_graph && dense_kv && strategy == CudaPagedDecodeStrategy::OnePass {
            let (output, outcome) = try_cuda_paged_decode_graph(
                &queries,
                &keys,
                &values,
                metadata,
                batch,
                query_heads,
                kv_heads,
                page_tokens,
                max_blocks,
                key_dim,
                value_dim,
                capacity_pages,
                softmax_scale,
                softcap,
                _backing_generation,
            )?;
            if let Some(output) = output {
                return Ok((output, outcome));
            }
            outcome
        } else {
            CudaPagedDecodeGraphOutcome::Disabled
        }
    };
    #[cfg(not(feature = "cuda"))]
    let graph_outcome = CudaPagedDecodeGraphOutcome::Disabled;
    let output = queries.apply_op3_no_bwd(
        &keys,
        &values,
        &CudaPagedDecodeOp {
            metadata: metadata.clone(),
            batch,
            query_heads,
            kv_heads,
            page_tokens,
            max_blocks,
            key_dim,
            value_dim,
            capacity_pages,
            softmax_scale,
            softcap,
            strategy,
            partition_tokens: partition_tuning.map(|(_, tokens)| tokens).unwrap_or(0),
        },
    )?;
    Ok((output, graph_outcome))
}

#[cfg(feature = "cuda")]
const CUDA_PAGED_DECODE_GRAPH_BUCKETS: usize = 64;

#[cfg(feature = "cuda")]
const CUDA_PAGED_DECODE_GRAPH_FAILURE_BACKOFF: u8 = 8;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CudaPagedDecodeGraphKey {
    queries_dtype: DType,
    keys_id: candle_core::TensorId,
    values_id: candle_core::TensorId,
    metadata_id: candle_core::TensorId,
    backing_generation: u64,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    capacity_pages: usize,
    softmax_scale_bits: u32,
    softcap_bits: Option<u32>,
}

#[cfg(feature = "cuda")]
enum CudaPagedDecodeGraphState {
    Warm {
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        metadata: Tensor,
        output: Tensor,
    },
    Captured {
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        metadata: Tensor,
        output: Tensor,
        graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    },
    Backoff {
        remaining_calls: u8,
    },
}

#[cfg(feature = "cuda")]
thread_local! {
    // cudarc graph objects are explicitly not thread safe. Keeping each graph
    // in the worker thread that captured it also makes capture mode and replay
    // ownership unambiguous.
    static CUDA_PAGED_DECODE_GRAPHS: RefCell<VecDeque<(CudaPagedDecodeGraphKey, CudaPagedDecodeGraphState)>> =
        const { RefCell::new(VecDeque::new()) };
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_paged_decode_graph_key(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    capacity_pages: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    backing_generation: u64,
) -> CudaPagedDecodeGraphKey {
    CudaPagedDecodeGraphKey {
        queries_dtype: queries.dtype(),
        keys_id: keys.id(),
        values_id: values.id(),
        metadata_id: metadata.id(),
        backing_generation,
        batch,
        query_heads,
        kv_heads,
        page_tokens,
        max_blocks,
        key_dim,
        value_dim,
        capacity_pages,
        softmax_scale_bits: softmax_scale.to_bits(),
        softcap_bits: softcap.map(f32::to_bits),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_cuda_paged_decode_graph(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    capacity_pages: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    backing_generation: u64,
) -> candle_core::Result<(Option<Tensor>, CudaPagedDecodeGraphOutcome)> {
    use candle_core::cuda_backend::cudarc::driver::sys::{
        CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    };

    let key = cuda_paged_decode_graph_key(
        queries,
        keys,
        values,
        metadata,
        batch,
        query_heads,
        kv_heads,
        page_tokens,
        max_blocks,
        key_dim,
        value_dim,
        capacity_pages,
        softmax_scale,
        softcap,
        backing_generation,
    );
    CUDA_PAGED_DECODE_GRAPHS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let Some(index) = cache.iter().position(|(candidate, _)| *candidate == key) else {
            let stable_queries = Tensor::zeros(queries.shape(), queries.dtype(), queries.device())?;
            let output = Tensor::zeros(
                (batch, query_heads, value_dim),
                queries.dtype(),
                queries.device(),
            )?;
            let evicted = cache.len() == CUDA_PAGED_DECODE_GRAPH_BUCKETS;
            if evicted {
                cache.pop_front();
            }
            cache.push_back((
                key,
                CudaPagedDecodeGraphState::Warm {
                    queries: stable_queries,
                    keys: keys.clone(),
                    values: values.clone(),
                    metadata: metadata.clone(),
                    output,
                },
            ));
            return Ok((
                None,
                if evicted {
                    CudaPagedDecodeGraphOutcome::WarmedAfterEviction
                } else {
                    CudaPagedDecodeGraphOutcome::Warmed
                },
            ));
        };

        let (_, state) = cache.remove(index).expect("located CUDA graph bucket");
        let result = match state {
            CudaPagedDecodeGraphState::Captured {
                queries: stable_queries,
                keys,
                values,
                metadata,
                output,
                graph,
            } => {
                if stable_queries.slice_set(queries, 0, 0).is_err() || graph.launch().is_err() {
                    cache.push_back((
                        key,
                        CudaPagedDecodeGraphState::Backoff {
                            remaining_calls: CUDA_PAGED_DECODE_GRAPH_FAILURE_BACKOFF,
                        },
                    ));
                    return Ok((None, CudaPagedDecodeGraphOutcome::EagerFallback));
                }
                let result = output.clone();
                cache.push_back((
                    key,
                    CudaPagedDecodeGraphState::Captured {
                        queries: stable_queries,
                        keys,
                        values,
                        metadata,
                        output,
                        graph,
                    },
                ));
                (Some(result), CudaPagedDecodeGraphOutcome::Replayed)
            }
            CudaPagedDecodeGraphState::Warm {
                queries: stable_queries,
                keys,
                values,
                metadata,
                output,
            } => {
                if stable_queries.slice_set(queries, 0, 0).is_err() {
                    cache.push_back((
                        key,
                        CudaPagedDecodeGraphState::Backoff {
                            remaining_calls: CUDA_PAGED_DECODE_GRAPH_FAILURE_BACKOFF,
                        },
                    ));
                    return Ok((None, CudaPagedDecodeGraphOutcome::EagerFallback));
                }
                let device = queries.device().as_cuda_device()?;
                let stream = device.cuda_stream();
                let _htod_cache = device.enable_cuda_graph_htod_cache();
                if stream
                    .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .is_err()
                {
                    cache.push_back((
                        key,
                        CudaPagedDecodeGraphState::Backoff {
                            remaining_calls: CUDA_PAGED_DECODE_GRAPH_FAILURE_BACKOFF,
                        },
                    ));
                    return Ok((None, CudaPagedDecodeGraphOutcome::EagerFallback));
                }
                let captured_launch = launch_cuda_paged_decode_one_pass_into(
                    &stable_queries,
                    &keys,
                    &values,
                    &metadata,
                    &output,
                    batch,
                    query_heads,
                    kv_heads,
                    page_tokens,
                    max_blocks,
                    key_dim,
                    value_dim,
                    capacity_pages,
                    softmax_scale,
                    softcap,
                );
                let captured_graph =
                    stream.end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                let Ok(()) = captured_launch else {
                    cache.push_back((
                        key,
                        CudaPagedDecodeGraphState::Backoff {
                            remaining_calls: CUDA_PAGED_DECODE_GRAPH_FAILURE_BACKOFF,
                        },
                    ));
                    return Ok((None, CudaPagedDecodeGraphOutcome::EagerFallback));
                };
                let Ok(Some(graph)) = captured_graph else {
                    cache.push_back((
                        key,
                        CudaPagedDecodeGraphState::Backoff {
                            remaining_calls: CUDA_PAGED_DECODE_GRAPH_FAILURE_BACKOFF,
                        },
                    ));
                    return Ok((None, CudaPagedDecodeGraphOutcome::EagerFallback));
                };
                if graph.launch().is_err() {
                    cache.push_back((
                        key,
                        CudaPagedDecodeGraphState::Backoff {
                            remaining_calls: CUDA_PAGED_DECODE_GRAPH_FAILURE_BACKOFF,
                        },
                    ));
                    return Ok((None, CudaPagedDecodeGraphOutcome::EagerFallback));
                }
                let result = output.clone();
                cache.push_back((
                    key,
                    CudaPagedDecodeGraphState::Captured {
                        queries: stable_queries,
                        keys,
                        values,
                        metadata,
                        output,
                        graph,
                    },
                ));
                (Some(result), CudaPagedDecodeGraphOutcome::Captured)
            }
            CudaPagedDecodeGraphState::Backoff { remaining_calls } => {
                if remaining_calls > 1 {
                    cache.push_back((
                        key,
                        CudaPagedDecodeGraphState::Backoff {
                            remaining_calls: remaining_calls - 1,
                        },
                    ));
                }
                (None, CudaPagedDecodeGraphOutcome::Backoff)
            }
        };
        Ok(result)
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn launch_cuda_paged_decode_one_pass_into(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    output: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    capacity_pages: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
) -> candle_core::Result<()> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

    let (query_storage, query_layout) = queries.storage_and_layout();
    let (key_storage, key_layout) = keys.storage_and_layout();
    let (value_storage, value_layout) = values.storage_and_layout();
    let (metadata_storage, metadata_layout) = metadata.storage_and_layout();
    let (output_storage, output_layout) = output.storage_and_layout();
    let candle_core::Storage::Cuda(query_storage) = &*query_storage else {
        candle_core::bail!("CUDA graph query storage is not CUDA")
    };
    let candle_core::Storage::Cuda(key_storage) = &*key_storage else {
        candle_core::bail!("CUDA graph key storage is not CUDA")
    };
    let candle_core::Storage::Cuda(value_storage) = &*value_storage else {
        candle_core::bail!("CUDA graph value storage is not CUDA")
    };
    let candle_core::Storage::Cuda(metadata_storage) = &*metadata_storage else {
        candle_core::bail!("CUDA graph metadata storage is not CUDA")
    };
    let candle_core::Storage::Cuda(output_storage) = &*output_storage else {
        candle_core::bail!("CUDA graph output storage is not CUDA")
    };
    let CudaStorageSlice::U32(metadata_slice) = &metadata_storage.slice else {
        candle_core::bail!("CUDA graph metadata storage is not U32")
    };
    let Some((metadata_start, metadata_end)) = metadata_layout.contiguous_offsets() else {
        candle_core::bail!("CUDA graph metadata must be contiguous")
    };
    let metadata_view = metadata_slice.slice(metadata_start..metadata_end);
    let blocks =
        u32::try_from(batch.checked_mul(query_heads).ok_or_else(|| {
            candle_core::Error::Msg("CUDA graph decode grid overflow".to_string())
        })?)
        .map_err(|_| candle_core::Error::Msg("CUDA graph decode grid overflow".to_string()))?;
    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
    };
    let device = query_storage.device();

    macro_rules! launch {
        ($variant:ident, $function_name:literal) => {{
            let CudaStorageSlice::$variant(query_slice) = &query_storage.slice else {
                candle_core::bail!("CUDA graph query dtype mismatch")
            };
            let CudaStorageSlice::$variant(key_slice) = &key_storage.slice else {
                candle_core::bail!("CUDA graph key dtype mismatch")
            };
            let CudaStorageSlice::$variant(value_slice) = &value_storage.slice else {
                candle_core::bail!("CUDA graph value dtype mismatch")
            };
            let CudaStorageSlice::$variant(output_slice) = &output_storage.slice else {
                candle_core::bail!("CUDA graph output dtype mismatch")
            };
            let Some((query_start, query_end)) = query_layout.contiguous_offsets() else {
                candle_core::bail!("CUDA graph queries must be contiguous")
            };
            let Some((key_start, key_end)) = key_layout.contiguous_offsets() else {
                candle_core::bail!("CUDA graph keys must be contiguous")
            };
            let Some((value_start, value_end)) = value_layout.contiguous_offsets() else {
                candle_core::bail!("CUDA graph values must be contiguous")
            };
            let Some((output_start, output_end)) = output_layout.contiguous_offsets() else {
                candle_core::bail!("CUDA graph output must be contiguous")
            };
            let query_view = query_slice.slice(query_start..query_end);
            let key_view = key_slice.slice(key_start..key_end);
            let value_view = value_slice.slice(value_start..value_end);
            let output_view = output_slice.slice(output_start..output_end);
            let function = device.get_or_load_custom_func(
                $function_name,
                "izwi_physical_state",
                cuda_ptx::PHYSICAL_STATE,
            )?;
            let mut builder = function.builder();
            builder.arg(&query_view);
            builder.arg(&key_view);
            builder.arg(&value_view);
            builder.arg(&metadata_view);
            builder.arg(&output_view);
            candle_core::builder_arg!(
                builder,
                batch as i32,
                query_heads as i32,
                kv_heads as i32,
                page_tokens as i32,
                max_blocks as i32,
                key_dim as i32,
                value_dim as i32,
                capacity_pages as i32,
                softmax_scale,
                softcap.unwrap_or(0.0)
            );
            // SAFETY: the same validated geometry used by the eager custom op
            // binds these stable buffers for the complete graph lifetime.
            unsafe { builder.launch(config) }.w()?;
        }};
    }

    match &query_storage.slice {
        CudaStorageSlice::F32(_) => launch!(F32, "physical_paged_decode_f32"),
        CudaStorageSlice::F16(_) => launch!(F16, "physical_paged_decode_f16"),
        CudaStorageSlice::BF16(_) => launch!(BF16, "physical_paged_decode_bf16"),
        _ => candle_core::bail!("CUDA graph decode requires F32/F16/BF16 storage"),
    }
    Ok(())
}

pub fn try_lfm_shortconv_ring_sequence(
    ring: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    expected_cursor: u64,
    valid_length: u64,
) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(ring, input)
        || !cuda_tensor_pair_supported(ring, weight)
        || ring.dtype() != DType::F32
        || !ring.layout().is_contiguous()
        || !input.layout().is_contiguous()
        || !weight.layout().is_contiguous()
    {
        return None;
    }
    let (capacity, batch, hidden) = ring.dims3().ok()?;
    let (input_batch, input_hidden, steps) = input.dims3().ok()?;
    let (weight_hidden, weight_capacity) = weight.dims2().ok()?;
    if capacity == 0
        || batch == 0
        || hidden == 0
        || steps == 0
        || input_batch != batch
        || input_hidden != hidden
        || weight_hidden != hidden
        || weight_capacity != capacity
        || valid_length > capacity as u64
        || valid_length > expected_cursor
    {
        return None;
    }
    ring.apply_op3_no_bwd(
        input,
        weight,
        &CudaPhysicalRingShortConvOp {
            batch,
            hidden,
            steps,
            capacity,
            expected_cursor,
            valid_length,
        },
    )
    .ok()
}

pub fn try_lfm_shortconv_decode3(cache: &Tensor, bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(cache, bx) || !cuda_tensor_pair_supported(cache, conv) {
        return None;
    }
    candle_lfm_shortconv_decode3(cache, bx, conv)
}

fn candle_lfm_shortconv_decode3(cache: &Tensor, bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    let (batch, hidden, cache_len) = cache.dims3().ok()?;
    let (input_batch, input_hidden, input_len) = bx.dims3().ok()?;
    if cache_len != 3
        || input_len != 1
        || batch != input_batch
        || hidden != input_hidden
        || conv.dims() != [hidden, 3]
        || cache.dtype() != bx.dtype()
        || cache.dtype() != conv.dtype()
        || cache.device().location() != bx.device().location()
        || cache.device().location() != conv.device().location()
    {
        return None;
    }
    let state = candle_lfm_shortconv_update3(cache, bx)?;
    state
        .broadcast_mul(&conv.unsqueeze(0).ok()?)
        .ok()?
        .sum_keepdim(2)
        .ok()
}

pub fn try_lfm_shortconv_update3(cache: &Tensor, bx: &Tensor) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(cache, bx) {
        return None;
    }
    candle_lfm_shortconv_update3(cache, bx)
}

fn candle_lfm_shortconv_update3(cache: &Tensor, bx: &Tensor) -> Option<Tensor> {
    let (batch, hidden, cache_len) = cache.dims3().ok()?;
    let (input_batch, input_hidden, input_len) = bx.dims3().ok()?;
    if cache_len != 3
        || input_len != 1
        || batch != input_batch
        || hidden != input_hidden
        || cache.dtype() != bx.dtype()
        || cache.device().location() != bx.device().location()
    {
        return None;
    }
    Tensor::cat(&[&cache.narrow(2, 1, 2).ok()?, bx], 2).ok()
}

pub fn try_lfm_shortconv_sequence3(bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(bx, conv) {
        return None;
    }
    candle_lfm_shortconv_sequence3(bx, conv)
}

fn candle_lfm_shortconv_sequence3(bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    let (batch, hidden, sequence) = bx.dims3().ok()?;
    if sequence == 0
        || conv.dims() != [hidden, 3]
        || bx.dtype() != conv.dtype()
        || bx.device().location() != conv.device().location()
    {
        return None;
    }
    let zeros = Tensor::zeros((batch, hidden, 2), bx.dtype(), bx.device()).ok()?;
    let padded = Tensor::cat(&[&zeros, bx], 2).ok()?;
    let weights = conv.reshape((1, hidden, 3, 1)).ok()?;
    let windows = Tensor::stack(
        &[
            &padded.narrow(2, 0, sequence).ok()?,
            &padded.narrow(2, 1, sequence).ok()?,
            &padded.narrow(2, 2, sequence).ok()?,
        ],
        2,
    )
    .ok()?;
    windows.broadcast_mul(&weights).ok()?.sum(2).ok()
}

fn cuda_tensor_supported(tensor: &Tensor) -> bool {
    cuda_kernels_compiled() && tensor.device().is_cuda()
}

fn cuda_tensor_pair_supported(lhs: &Tensor, rhs: &Tensor) -> bool {
    cuda_tensor_supported(lhs) && rhs.device().is_cuda() && lhs.dtype() == rhs.dtype()
}

fn cuda_gated_delta_sequence(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    let (batch, sequence, heads, key_dim) = queries.dims4().ok()?;
    let (key_batch, key_sequence, key_heads, key_width) = keys.dims4().ok()?;
    let (value_batch, value_sequence, value_heads, value_dim) = values.dims4().ok()?;
    let (g_batch, g_sequence, g_heads) = g.dims3().ok()?;
    let (beta_batch, beta_sequence, beta_heads) = beta.dims3().ok()?;
    let (state_batch, state_heads, state_key_dim, state_value_dim) = initial_state.dims4().ok()?;
    if sequence == 0 || heads == 0 || key_dim == 0 || value_dim == 0 {
        return None;
    }
    if (key_batch, key_sequence, key_heads, key_width) != (batch, sequence, heads, key_dim)
        || (value_batch, value_sequence, value_heads) != (batch, sequence, heads)
        || (g_batch, g_sequence, g_heads) != (batch, sequence, heads)
        || (beta_batch, beta_sequence, beta_heads) != (batch, sequence, heads)
        || (state_batch, state_heads, state_key_dim, state_value_dim)
            != (batch, heads, key_dim, value_dim)
    {
        return None;
    }

    let qkv = Tensor::cat(&[queries, keys, values], D::Minus1)
        .ok()?
        .contiguous()
        .ok()?;
    let gates = Tensor::cat(
        &[
            &g.unsqueeze(D::Minus1).ok()?,
            &beta.unsqueeze(D::Minus1).ok()?,
        ],
        D::Minus1,
    )
    .ok()?
    .contiguous()
    .ok()?;
    let initial_state = initial_state.contiguous().ok()?;
    let packed = qkv
        .apply_op3_no_bwd(
            &gates,
            &initial_state,
            &CudaGatedDeltaSequenceOp {
                batch,
                sequence,
                heads,
                key_dim,
                value_dim,
            },
        )
        .ok()?;

    let output_elements = batch * sequence * heads * value_dim;
    let state_elements = batch * heads * key_dim * value_dim;
    let outputs = packed
        .narrow(0, 0, output_elements)
        .ok()?
        .reshape((batch, sequence, heads, value_dim))
        .ok()?;
    let next_state = packed
        .narrow(0, output_elements, state_elements)
        .ok()?
        .reshape((batch, heads, key_dim, value_dim))
        .ok()?;
    Some((outputs, next_state))
}

struct CudaCausalConvSequenceOp {
    conv_dim: usize,
    sequence: usize,
    kernel_size: usize,
}

struct CudaQwen38CausalConvDecodeOp {
    conv_dim: usize,
}

impl CustomOp3 for CudaQwen38CausalConvDecodeOp {
    fn name(&self) -> &'static str {
        "qwen38-causal-conv-decode"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
        _history: &CpuStorage,
        _history_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("Qwen3.8 CUDA decode convolution has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
        history: &candle_core::CudaStorage,
        history_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        fn contiguous_slice<'a>(
            storage: &'a CudaStorageSlice,
            layout: &Layout,
            name: &str,
        ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>>
        {
            let CudaStorageSlice::F32(slice) = storage else {
                candle_core::bail!("{name} must use F32 storage")
            };
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("{name} must be contiguous")
            };
            Ok(slice.slice(start..end))
        }

        let input_slice = contiguous_slice(&input.slice, input_layout, "input")?;
        let weight_slice = contiguous_slice(&weight.slice, weight_layout, "weight")?;
        let history_slice = contiguous_slice(&history.slice, history_layout, "history")?;
        let total_elements = self.conv_dim.checked_mul(4).ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.8 CUDA decode convolution overflow".to_string())
        })?;
        if total_elements > i32::MAX as usize {
            candle_core::bail!("Qwen3.8 CUDA decode convolution tensor is too large")
        }
        let device = input.device();
        // SAFETY: the custom kernel writes every element before the storage is observed.
        let output = unsafe { device.alloc::<f32>(total_elements)? };
        let function = device.get_or_load_custom_func(
            "qwen38_causal_conv_decode_f32",
            "izwi_qwen38_causal_conv_decode",
            cuda_ptx::QWEN38,
        )?;
        let config = LaunchConfig::for_num_elems(total_elements as u32);
        let mut builder = function.builder();
        builder.arg(&input_slice);
        builder.arg(&weight_slice);
        builder.arg(&history_slice);
        builder.arg(&output);
        candle_core::builder_arg!(builder, self.conv_dim as i32);
        // SAFETY: argument types and element bounds match the CUDA kernel signature.
        unsafe { builder.launch(config) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from_dims(&[total_elements]),
        ))
    }
}

impl CustomOp3 for CudaCausalConvSequenceOp {
    fn name(&self) -> &'static str {
        "qwen35-causal-conv-sequence"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
        _history: &CpuStorage,
        _history_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("Qwen3.5 CUDA causal convolution has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
        history: &candle_core::CudaStorage,
        history_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        fn contiguous_slice<'a>(
            storage: &'a CudaStorageSlice,
            layout: &Layout,
            name: &str,
        ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>>
        {
            let CudaStorageSlice::F32(slice) = storage else {
                candle_core::bail!("{name} must use F32 storage")
            };
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("{name} must be contiguous")
            };
            Ok(slice.slice(start..end))
        }

        let input_slice = contiguous_slice(&input.slice, input_layout, "input")?;
        let weight_slice = contiguous_slice(&weight.slice, weight_layout, "weight")?;
        let history_slice = contiguous_slice(&history.slice, history_layout, "history")?;
        let output_elements = self.sequence.checked_mul(self.conv_dim).ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 CUDA convolution output overflow".to_string())
        })?;
        let state_elements = (self.kernel_size - 1)
            .checked_mul(self.conv_dim)
            .ok_or_else(|| {
                candle_core::Error::Msg("Qwen3.5 CUDA convolution state overflow".to_string())
            })?;
        let total_elements = output_elements.checked_add(state_elements).ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 CUDA convolution allocation overflow".to_string())
        })?;
        if total_elements > i32::MAX as usize {
            candle_core::bail!("Qwen3.5 CUDA convolution tensor is too large")
        }
        let device = input.device();
        // SAFETY: the custom kernel writes every element before the storage is observed.
        let output = unsafe { device.alloc::<f32>(total_elements)? };
        let function = device.get_or_load_custom_func(
            "qwen35_causal_conv_sequence_f32",
            "izwi_qwen35_causal_conv_sequence",
            cuda_ptx::QWEN35,
        )?;
        let config = LaunchConfig::for_num_elems(total_elements as u32);
        let mut builder = function.builder();
        builder.arg(&input_slice);
        builder.arg(&weight_slice);
        builder.arg(&history_slice);
        builder.arg(&output);
        candle_core::builder_arg!(
            builder,
            self.conv_dim as i32,
            self.sequence as i32,
            self.kernel_size as i32,
            output_elements as i32,
            total_elements as i32
        );
        // SAFETY: argument types and element bounds match the CUDA kernel signature.
        unsafe { builder.launch(config) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from_dims(&[total_elements]),
        ))
    }
}

struct CudaGatedDeltaSequenceOp {
    batch: usize,
    sequence: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
}

impl CustomOp3 for CudaGatedDeltaSequenceOp {
    fn name(&self) -> &'static str {
        "qwen35-gated-delta-sequence"
    }

    fn cpu_fwd(
        &self,
        _qkv: &CpuStorage,
        _qkv_layout: &Layout,
        _gates: &CpuStorage,
        _gates_layout: &Layout,
        _state: &CpuStorage,
        _state_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("Qwen3.5 CUDA recurrence has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        qkv: &candle_core::CudaStorage,
        qkv_layout: &Layout,
        gates: &candle_core::CudaStorage,
        gates_layout: &Layout,
        state: &candle_core::CudaStorage,
        state_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        fn contiguous_slice<'a>(
            storage: &'a CudaStorageSlice,
            layout: &Layout,
            name: &str,
        ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>>
        {
            let CudaStorageSlice::F32(slice) = storage else {
                candle_core::bail!("{name} must use F32 storage")
            };
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("{name} must be contiguous")
            };
            Ok(slice.slice(start..end))
        }

        let qkv_slice = contiguous_slice(&qkv.slice, qkv_layout, "qkv")?;
        let gates_slice = contiguous_slice(&gates.slice, gates_layout, "gates")?;
        let state_slice = contiguous_slice(&state.slice, state_layout, "initial_state")?;
        let device = qkv.device();
        let output_elements = self.batch * self.sequence * self.heads * self.value_dim;
        let state_elements = self.batch * self.heads * self.key_dim * self.value_dim;
        // SAFETY: the custom kernel writes every element before the storage is observed.
        let output = unsafe { device.alloc::<f32>(output_elements + state_elements)? };
        let function = device.get_or_load_custom_func(
            "qwen35_gated_delta_sequence_f32",
            "izwi_qwen35_gated_delta_sequence",
            cuda_ptx::QWEN35,
        )?;
        let block_size = self.value_dim.next_power_of_two().clamp(32, 256) as u32;
        let config = LaunchConfig {
            grid_dim: ((self.batch * self.heads) as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = function.builder();
        builder.arg(&qkv_slice);
        builder.arg(&gates_slice);
        builder.arg(&state_slice);
        builder.arg(&output);
        candle_core::builder_arg!(
            builder,
            self.batch as i32,
            self.sequence as i32,
            self.heads as i32,
            self.key_dim as i32,
            self.value_dim as i32
        );
        // SAFETY: argument types and launch dimensions match the CUDA kernel signature.
        unsafe { builder.launch(config) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from_dims(&[output_elements + state_elements]),
        ))
    }
}

struct CudaPagedDecodeOp {
    metadata: Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    capacity_pages: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    strategy: CudaPagedDecodeStrategy,
    partition_tokens: usize,
}

struct CudaPagedPrefillOp {
    metadata: Tensor,
    sequences: usize,
    total_queries: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_dim: usize,
    value_dim: usize,
    capacity_pages: usize,
    window_tokens: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
}

struct CudaPhysicalRingShortConvOp {
    batch: usize,
    hidden: usize,
    steps: usize,
    capacity: usize,
    expected_cursor: u64,
    valid_length: u64,
}

impl CustomOp3 for CudaPhysicalRingShortConvOp {
    fn name(&self) -> &'static str {
        "physical-ring-shortconv"
    }

    fn cpu_fwd(
        &self,
        _ring: &CpuStorage,
        _ring_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("physical CUDA ring ShortConv has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        ring: &candle_core::CudaStorage,
        ring_layout: &Layout,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        fn contiguous_f32<'a>(
            storage: &'a CudaStorageSlice,
            layout: &Layout,
            name: &str,
        ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>>
        {
            let CudaStorageSlice::F32(slice) = storage else {
                candle_core::bail!("{name} must use F32 storage")
            };
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("{name} must be contiguous")
            };
            Ok(slice.slice(start..end))
        }

        let device = ring.device();
        let ring = contiguous_f32(&ring.slice, ring_layout, "physical ShortConv ring")?;
        let input = contiguous_f32(&input.slice, input_layout, "physical ShortConv input")?;
        let weight = contiguous_f32(&weight.slice, weight_layout, "physical ShortConv weight")?;
        let output_elements = self
            .batch
            .checked_mul(self.hidden)
            .and_then(|value| value.checked_mul(self.steps))
            .ok_or_else(|| {
                candle_core::Error::Msg("physical CUDA ShortConv output overflow".to_string())
            })?;
        let output_elements_i32 = i32::try_from(output_elements).map_err(|_| {
            candle_core::Error::Msg("physical CUDA ShortConv output is too large".to_string())
        })?;
        // SAFETY: the custom kernel writes every output element before the
        // returned storage is observed.
        let output = unsafe { device.alloc::<f32>(output_elements)? };
        let function = device.get_or_load_custom_func(
            "physical_ring_shortconv_f32",
            "izwi_physical_state",
            cuda_ptx::PHYSICAL_STATE,
        )?;
        let config = LaunchConfig::for_num_elems(output_elements as u32);
        let mut builder = function.builder();
        builder.arg(&ring);
        builder.arg(&input);
        builder.arg(&weight);
        builder.arg(&output);
        candle_core::builder_arg!(
            builder,
            self.batch as i32,
            self.hidden as i32,
            self.steps as i32,
            self.capacity as i32,
            self.expected_cursor,
            self.valid_length,
            output_elements_i32
        );
        // SAFETY: argument types and element bounds match the CUDA kernel
        // signature and the validated physical-ring geometry.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from_dims(&[self.batch, self.hidden, self.steps]),
        ))
    }
}

impl CustomOp3 for CudaPagedPrefillOp {
    fn name(&self) -> &'static str {
        "physical-paged-prefill"
    }

    fn cpu_fwd(
        &self,
        _queries: &CpuStorage,
        _queries_layout: &Layout,
        _keys: &CpuStorage,
        _keys_layout: &Layout,
        _values: &CpuStorage,
        _values_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("physical CUDA paged prefill has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        queries: &candle_core::CudaStorage,
        queries_layout: &Layout,
        keys: &candle_core::CudaStorage,
        keys_layout: &Layout,
        values: &candle_core::CudaStorage,
        values_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        let device = queries.device();
        let (metadata_storage, metadata_layout) = self.metadata.storage_and_layout();
        let candle_core::Storage::Cuda(metadata_storage) = &*metadata_storage else {
            candle_core::bail!("CUDA paged prefill metadata storage is not CUDA")
        };
        let CudaStorageSlice::U32(metadata_slice) = &metadata_storage.slice else {
            candle_core::bail!("CUDA paged prefill metadata storage is not U32")
        };
        let Some((metadata_start, metadata_end)) = metadata_layout.contiguous_offsets() else {
            candle_core::bail!("CUDA paged prefill metadata must be contiguous")
        };
        let metadata = metadata_slice.slice(metadata_start..metadata_end);
        let output_elements = self
            .total_queries
            .checked_mul(self.query_heads)
            .and_then(|value| value.checked_mul(self.value_dim))
            .ok_or_else(|| {
                candle_core::Error::Msg("CUDA paged prefill output overflow".to_string())
            })?;
        let blocks = self
            .total_queries
            .checked_mul(self.query_heads)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                candle_core::Error::Msg("CUDA paged prefill grid overflow".to_string())
            })?;
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };

        macro_rules! launch {
            ($query_variant:ident, $kv_variant:ident, $ty:ty, $function_name:literal) => {{
                let CudaStorageSlice::$query_variant(query_slice) = &queries.slice else {
                    candle_core::bail!("CUDA paged prefill query storage dtype mismatch")
                };
                let CudaStorageSlice::$kv_variant(key_slice) = &keys.slice else {
                    candle_core::bail!("CUDA paged prefill key storage dtype mismatch")
                };
                let CudaStorageSlice::$kv_variant(value_slice) = &values.slice else {
                    candle_core::bail!("CUDA paged prefill value storage dtype mismatch")
                };
                let Some((query_start, query_end)) = queries_layout.contiguous_offsets() else {
                    candle_core::bail!("CUDA paged prefill queries must be contiguous")
                };
                let Some((key_start, key_end)) = keys_layout.contiguous_offsets() else {
                    candle_core::bail!("CUDA paged prefill keys must be contiguous")
                };
                let Some((value_start, value_end)) = values_layout.contiguous_offsets() else {
                    candle_core::bail!("CUDA paged prefill values must be contiguous")
                };
                let query_view = query_slice.slice(query_start..query_end);
                let key_view = key_slice.slice(key_start..key_end);
                let value_view = value_slice.slice(value_start..value_end);
                // SAFETY: the kernel initializes every output element for the
                // validated compact metadata and tensor geometry.
                let output = unsafe { device.alloc::<$ty>(output_elements)? };
                let function = device.get_or_load_custom_func(
                    $function_name,
                    "izwi_physical_state",
                    cuda_ptx::PHYSICAL_STATE,
                )?;
                let mut builder = function.builder();
                builder.arg(&query_view);
                builder.arg(&key_view);
                builder.arg(&value_view);
                builder.arg(&metadata);
                builder.arg(&output);
                candle_core::builder_arg!(
                    builder,
                    self.sequences as i32,
                    self.total_queries as i32,
                    self.query_heads as i32,
                    self.kv_heads as i32,
                    self.page_tokens as i32,
                    self.max_blocks as i32,
                    self.key_dim as i32,
                    self.value_dim as i32,
                    self.capacity_pages as i32,
                    self.window_tokens as i32,
                    self.softmax_scale,
                    self.softcap.unwrap_or(0.0)
                );
                // SAFETY: arguments and launch dimensions match the validated
                // native paged-prefill kernel ABI.
                unsafe { builder.launch(config) }.w()?;
                candle_core::CudaStorage {
                    slice: CudaStorageSlice::$query_variant(output),
                    device: device.clone(),
                }
            }};
        }

        let output = match (&queries.slice, &keys.slice, &values.slice) {
            (CudaStorageSlice::F32(_), CudaStorageSlice::F32(_), CudaStorageSlice::F32(_)) => {
                launch!(F32, F32, f32, "physical_paged_prefill_f32")
            }
            (CudaStorageSlice::F16(_), CudaStorageSlice::F16(_), CudaStorageSlice::F16(_)) => {
                launch!(F16, F16, half::f16, "physical_paged_prefill_f16")
            }
            (CudaStorageSlice::BF16(_), CudaStorageSlice::BF16(_), CudaStorageSlice::BF16(_)) => {
                launch!(BF16, BF16, half::bf16, "physical_paged_prefill_bf16")
            }
            (
                CudaStorageSlice::F16(_),
                CudaStorageSlice::F8E4M3(_),
                CudaStorageSlice::F8E4M3(_),
            ) => launch!(F16, F8E4M3, half::f16, "physical_paged_prefill_f16_fp8"),
            (
                CudaStorageSlice::BF16(_),
                CudaStorageSlice::F8E4M3(_),
                CudaStorageSlice::F8E4M3(_),
            ) => launch!(BF16, F8E4M3, half::bf16, "physical_paged_prefill_bf16_fp8"),
            _ => candle_core::bail!("CUDA paged prefill requires F32/F16/BF16 storage"),
        };
        Ok((
            output,
            Shape::from_dims(&[self.total_queries, self.query_heads, self.value_dim]),
        ))
    }
}

impl CustomOp3 for CudaPagedDecodeOp {
    fn name(&self) -> &'static str {
        "physical-paged-decode"
    }

    fn cpu_fwd(
        &self,
        _queries: &CpuStorage,
        _queries_layout: &Layout,
        _keys: &CpuStorage,
        _keys_layout: &Layout,
        _values: &CpuStorage,
        _values_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("physical CUDA paged decode has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        queries: &candle_core::CudaStorage,
        queries_layout: &Layout,
        keys: &candle_core::CudaStorage,
        keys_layout: &Layout,
        values: &candle_core::CudaStorage,
        values_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        let device = queries.device();
        let (metadata_storage, metadata_layout) = self.metadata.storage_and_layout();
        let candle_core::Storage::Cuda(metadata_storage) = &*metadata_storage else {
            candle_core::bail!("CUDA paged decode metadata storage is not CUDA")
        };
        let CudaStorageSlice::U32(metadata_slice) = &metadata_storage.slice else {
            candle_core::bail!("CUDA paged decode metadata storage is not U32")
        };
        let Some((metadata_start, metadata_end)) = metadata_layout.contiguous_offsets() else {
            candle_core::bail!("CUDA paged decode metadata must be contiguous")
        };
        let metadata = metadata_slice.slice(metadata_start..metadata_end);
        let output_elements = self
            .batch
            .checked_mul(self.query_heads)
            .and_then(|value| value.checked_mul(self.value_dim))
            .ok_or_else(|| {
                candle_core::Error::Msg("CUDA paged decode output overflow".to_string())
            })?;
        if output_elements > i32::MAX as usize {
            candle_core::bail!("CUDA paged decode output is too large")
        }
        let blocks = self
            .batch
            .checked_mul(self.query_heads)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                candle_core::Error::Msg("CUDA paged decode grid overflow".to_string())
            })?;
        let one_pass_config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };

        macro_rules! launch {
            ($query_variant:ident, $kv_variant:ident, $ty:ty, $function_name:literal, $partition_name:literal, $reduce_name:literal) => {{
                let CudaStorageSlice::$query_variant(query_slice) = &queries.slice else {
                    candle_core::bail!("CUDA paged decode query storage dtype mismatch")
                };
                let CudaStorageSlice::$kv_variant(key_slice) = &keys.slice else {
                    candle_core::bail!("CUDA paged decode key storage dtype mismatch")
                };
                let CudaStorageSlice::$kv_variant(value_slice) = &values.slice else {
                    candle_core::bail!("CUDA paged decode value storage dtype mismatch")
                };
                let Some((query_start, query_end)) = queries_layout.contiguous_offsets() else {
                    candle_core::bail!("CUDA paged decode queries must be contiguous")
                };
                let Some((key_start, key_end)) = keys_layout.contiguous_offsets() else {
                    candle_core::bail!("CUDA paged decode keys must be contiguous")
                };
                let Some((value_start, value_end)) = values_layout.contiguous_offsets() else {
                    candle_core::bail!("CUDA paged decode values must be contiguous")
                };
                let query_view = query_slice.slice(query_start..query_end);
                let key_view = key_slice.slice(key_start..key_end);
                let value_view = value_slice.slice(value_start..value_end);
                // SAFETY: the custom kernel writes every output element before
                // the returned storage is observed.
                let output = unsafe { device.alloc::<$ty>(output_elements)? };
                match self.strategy {
                    CudaPagedDecodeStrategy::OnePass => {
                        let function = device.get_or_load_custom_func(
                            $function_name,
                            "izwi_physical_state",
                            cuda_ptx::PHYSICAL_STATE,
                        )?;
                        let mut builder = function.builder();
                        builder.arg(&query_view);
                        builder.arg(&key_view);
                        builder.arg(&value_view);
                        builder.arg(&metadata);
                        builder.arg(&output);
                        candle_core::builder_arg!(
                            builder,
                            self.batch as i32,
                            self.query_heads as i32,
                            self.kv_heads as i32,
                            self.page_tokens as i32,
                            self.max_blocks as i32,
                            self.key_dim as i32,
                            self.value_dim as i32,
                            self.capacity_pages as i32,
                            self.softmax_scale,
                            self.softcap.unwrap_or(0.0)
                        );
                        // SAFETY: argument types, tensor bounds, and launch
                        // dimensions match the selected one-pass kernel.
                        unsafe { builder.launch(one_pass_config) }.w()?;
                    }
                    CudaPagedDecodeStrategy::Partitioned { partitions } => {
                        let partial_stride = self.value_dim.checked_add(2).ok_or_else(|| {
                            candle_core::Error::Msg(
                                "CUDA paged decode partial stride overflow".to_string(),
                            )
                        })?;
                        let partial_elements = (blocks as usize)
                            .checked_mul(partitions)
                            .and_then(|value| value.checked_mul(partial_stride))
                            .ok_or_else(|| {
                                candle_core::Error::Msg(
                                    "CUDA paged decode partial workspace overflow".to_string(),
                                )
                            })?;
                        // SAFETY: the partition kernel initializes every
                        // workspace element consumed by the reduction kernel.
                        let partials = unsafe { device.alloc::<f32>(partial_elements)? };
                        let partition_function = device.get_or_load_custom_func(
                            $partition_name,
                            "izwi_physical_state",
                            cuda_ptx::PHYSICAL_STATE,
                        )?;
                        let partition_config = LaunchConfig {
                            grid_dim: (blocks, partitions as u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
                        };
                        let mut builder = partition_function.builder();
                        builder.arg(&query_view);
                        builder.arg(&key_view);
                        builder.arg(&value_view);
                        builder.arg(&metadata);
                        builder.arg(&partials);
                        candle_core::builder_arg!(
                            builder,
                            self.batch as i32,
                            self.query_heads as i32,
                            self.kv_heads as i32,
                            self.page_tokens as i32,
                            self.max_blocks as i32,
                            self.key_dim as i32,
                            self.value_dim as i32,
                            self.capacity_pages as i32,
                            self.partition_tokens as i32,
                            partitions as i32,
                            self.softmax_scale,
                            self.softcap.unwrap_or(0.0)
                        );
                        // SAFETY: the validated metadata and geometry bound
                        // every input and workspace access.
                        unsafe { builder.launch(partition_config) }.w()?;

                        let reduce_function = device.get_or_load_custom_func(
                            $reduce_name,
                            "izwi_physical_state",
                            cuda_ptx::PHYSICAL_STATE,
                        )?;
                        let mut builder = reduce_function.builder();
                        builder.arg(&partials);
                        builder.arg(&output);
                        candle_core::builder_arg!(
                            builder,
                            blocks as i32,
                            self.value_dim as i32,
                            partitions as i32
                        );
                        // SAFETY: the first launch initializes the complete
                        // partial workspace on the same ordered CUDA stream.
                        unsafe { builder.launch(one_pass_config) }.w()?;
                    }
                }
                candle_core::CudaStorage {
                    slice: CudaStorageSlice::$query_variant(output),
                    device: device.clone(),
                }
            }};
        }

        let output = match (&queries.slice, &keys.slice, &values.slice) {
            (CudaStorageSlice::F32(_), CudaStorageSlice::F32(_), CudaStorageSlice::F32(_)) => {
                launch!(
                    F32,
                    F32,
                    f32,
                    "physical_paged_decode_f32",
                    "physical_paged_decode_partition_f32",
                    "physical_paged_decode_reduce_f32"
                )
            }
            (CudaStorageSlice::F16(_), CudaStorageSlice::F16(_), CudaStorageSlice::F16(_)) => {
                launch!(
                    F16,
                    F16,
                    half::f16,
                    "physical_paged_decode_f16",
                    "physical_paged_decode_partition_f16",
                    "physical_paged_decode_reduce_f16"
                )
            }
            (CudaStorageSlice::BF16(_), CudaStorageSlice::BF16(_), CudaStorageSlice::BF16(_)) => {
                launch!(
                    BF16,
                    BF16,
                    half::bf16,
                    "physical_paged_decode_bf16",
                    "physical_paged_decode_partition_bf16",
                    "physical_paged_decode_reduce_bf16"
                )
            }
            (
                CudaStorageSlice::F16(_),
                CudaStorageSlice::F8E4M3(_),
                CudaStorageSlice::F8E4M3(_),
            ) => {
                launch!(
                    F16,
                    F8E4M3,
                    half::f16,
                    "physical_paged_decode_f16_fp8",
                    "physical_paged_decode_partition_f16_fp8",
                    "physical_paged_decode_reduce_f16"
                )
            }
            (
                CudaStorageSlice::BF16(_),
                CudaStorageSlice::F8E4M3(_),
                CudaStorageSlice::F8E4M3(_),
            ) => {
                launch!(
                    BF16,
                    F8E4M3,
                    half::bf16,
                    "physical_paged_decode_bf16_fp8",
                    "physical_paged_decode_partition_bf16_fp8",
                    "physical_paged_decode_reduce_bf16"
                )
            }
            _ => candle_core::bail!("CUDA paged decode requires F32/F16/BF16 storage"),
        };
        Ok((
            output,
            Shape::from_dims(&[self.batch, self.query_heads, self.value_dim]),
        ))
    }
}

#[cfg(feature = "cuda")]
mod cuda_ptx {
    include!(concat!(env!("OUT_DIR"), "/qwen35_ptx.rs"));
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

        let conv_input = Tensor::zeros((1, 1, 8), DType::F32, &device).unwrap();
        let conv_weight = Tensor::zeros((8, 4), DType::F32, &device).unwrap();
        let conv_history = Tensor::zeros((8, 3), DType::F32, &device).unwrap();
        assert!(try_qwen38_causal_conv_decode(&conv_input, &conv_weight, &conv_history).is_none());

        let queries = Tensor::zeros((1, 1, 2), DType::F32, &device).unwrap();
        let keys = Tensor::zeros((1, 16, 1, 2), DType::F32, &device).unwrap();
        let values = Tensor::zeros((1, 16, 1, 2), DType::F32, &device).unwrap();
        let metadata = Tensor::from_vec(vec![0_u32, 1, 1, 0, 0], 5, &device).unwrap();
        assert!(paged_prefill_attention(
            &queries, &keys, &values, &metadata, 1, 1, 1, 1, 16, 1, 2, 2, 1.0, None, None,
        )
        .is_err());
        assert!(try_fused_rms_norm(&lhs, &rhs, 1e-6).is_none());
        assert!(try_fused_gated_rms_norm(&lhs, &rhs, &rhs, 1e-6).is_none());
    }

    #[test]
    fn qwen38_cuda_source_uses_independent_decode_symbol() {
        let source = include_str!("cuda/qwen38.cu");
        assert!(source.contains("qwen38_causal_conv_decode_f32"));
        assert!(!source.contains("qwen35"));
    }

    #[test]
    fn candle_cuda_fallback_primitives_match_cpu_references() {
        let device = candle_core::Device::Cpu;

        let q = Tensor::from_vec(vec![3.0f32, 4.0], (1, 1, 1, 2), &device).unwrap();
        let k = Tensor::from_vec(vec![0.0f32, 5.0], (1, 1, 1, 2), &device).unwrap();
        let weights = Tensor::ones(4, DType::F32, &device).unwrap();
        let (q_norm, k_norm) = candle_qk_rms_norm(&q, &k, &weights, 0.0).unwrap();
        let root_two = 2.0f32.sqrt();
        let q_actual = q_norm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let k_actual = k_norm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in q_actual
            .iter()
            .zip([3.0 * root_two / 5.0, 4.0 * root_two / 5.0])
        {
            assert!((actual - expected).abs() <= 1e-6);
        }
        for (actual, expected) in k_actual.iter().zip([0.0, root_two]) {
            assert!((actual - expected).abs() <= 1e-6);
        }

        let rope_input =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &device).unwrap();
        let cos_sin = Tensor::from_vec(vec![0.0f32, 0.0, 1.0, 1.0], (1, 4), &device).unwrap();
        let (q_rope, k_rope) = candle_rope_pair_bshd(&rope_input, &rope_input, &cos_sin).unwrap();
        let expected_rope = vec![-3.0f32, -4.0, 1.0, 2.0];
        assert_eq!(
            q_rope.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected_rope
        );
        assert_eq!(
            k_rope.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected_rope
        );

        let bx = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device).unwrap();
        let conv = Tensor::from_vec(vec![1.0f32, 10.0, 100.0], (1, 3), &device).unwrap();
        let sequence = candle_lfm_shortconv_sequence3(&bx, &conv).unwrap();
        assert_eq!(
            sequence.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![100.0, 210.0, 321.0]
        );
        let cache = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device).unwrap();
        let next = Tensor::from_vec(vec![4.0f32], (1, 1, 1), &device).unwrap();
        let decode = candle_lfm_shortconv_decode3(&cache, &next, &conv).unwrap();
        assert_eq!(
            decode.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![432.0]
        );
    }

    #[test]
    fn candle_cuda_gqa_fallback_preserves_grouped_query_semantics() {
        let device = candle_core::Device::Cpu;
        let q = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 2, 1, 2), &device).unwrap();
        let k = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(vec![2.0f32, 4.0, 6.0, 8.0], (1, 1, 2, 2), &device).unwrap();
        let output = candle_decode_gqa_attention(&q, &k, &v, 2, 1, 2, 2, 1.0).unwrap();
        let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let high = std::f32::consts::E / (std::f32::consts::E + 1.0);
        let low = 1.0 / (std::f32::consts::E + 1.0);
        let expected = [
            high * 2.0 + low * 6.0,
            high * 4.0 + low * 8.0,
            low * 2.0 + high * 6.0,
            low * 4.0 + high * 8.0,
        ];
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "GQA mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn cuda_paged_decode_metadata_validation_rejects_unsafe_tables() {
        // Layout: contexts, first-page offsets, then two padded table rows.
        let valid = vec![5, 3, 1, 0, 0, 1, 2, u32::MAX];
        validate_cuda_paged_decode_metadata(&valid, 2, 4, 2, 3).unwrap();

        let mut zero_context = valid.clone();
        zero_context[0] = 0;
        assert!(validate_cuda_paged_decode_metadata(&zero_context, 2, 4, 2, 3).is_err());

        let mut invalid_offset = valid.clone();
        invalid_offset[2] = 4;
        assert!(validate_cuda_paged_decode_metadata(&invalid_offset, 2, 4, 2, 3).is_err());

        let wrapping_context = vec![u32::MAX, 1, 0];
        assert!(
            validate_cuda_paged_decode_metadata(&wrapping_context, 1, usize::MAX, 1, 1,).is_err()
        );

        let mut incomplete_table = valid.clone();
        incomplete_table[0] = 8;
        assert!(validate_cuda_paged_decode_metadata(&incomplete_table, 2, 4, 2, 3).is_err());

        let mut out_of_bounds_page = valid.clone();
        out_of_bounds_page[5] = 3;
        assert!(validate_cuda_paged_decode_metadata(&out_of_bounds_page, 2, 4, 2, 3).is_err());

        assert!(validate_cuda_paged_decode_metadata(&valid[..7], 2, 4, 2, 3).is_err());
        assert!(validate_cuda_paged_decode_metadata(&[], usize::MAX, 1, usize::MAX, 1).is_err());
    }

    #[test]
    fn cuda_paged_decode_routes_only_long_contexts_to_partitions() {
        let tuned = Some((
            CUDA_PAGED_DECODE_SPLIT_THRESHOLD_TOKENS,
            CUDA_PAGED_DECODE_PARTITION_TOKENS,
        ));
        assert_eq!(
            cuda_paged_decode_strategy(16_384, 1, 1, 1, None).unwrap(),
            CudaPagedDecodeStrategy::OnePass,
            "an unobserved GPU must retain eager decode"
        );
        assert_eq!(
            cuda_paged_decode_strategy(CUDA_PAGED_DECODE_SPLIT_THRESHOLD_TOKENS, 1, 1, 1, tuned,)
                .unwrap(),
            CudaPagedDecodeStrategy::OnePass
        );
        assert_eq!(
            cuda_paged_decode_strategy(
                CUDA_PAGED_DECODE_SPLIT_THRESHOLD_TOKENS + 1,
                1,
                1,
                1,
                tuned,
            )
            .unwrap(),
            CudaPagedDecodeStrategy::Partitioned { partitions: 3 }
        );
        assert_eq!(
            cuda_paged_decode_strategy(CUDA_PAGED_DECODE_PARTITION_TOKENS * 9, 1, 1, 1, tuned,)
                .unwrap(),
            CudaPagedDecodeStrategy::Partitioned { partitions: 9 }
        );
        assert!(cuda_paged_decode_strategy(
            CUDA_PAGED_DECODE_PARTITION_TOKENS * CUDA_PAGED_DECODE_MAX_PARTITIONS + 1,
            1,
            1,
            1,
            tuned,
        )
        .is_err());
        assert_eq!(
            cuda_paged_decode_strategy(
                CUDA_PAGED_DECODE_SPLIT_THRESHOLD_TOKENS + 1,
                512,
                512,
                512,
                tuned,
            )
            .unwrap(),
            CudaPagedDecodeStrategy::OnePass,
            "an oversized split workspace must retain the bounded one-pass path"
        );
    }

    #[test]
    fn cuda_paged_decode_geometry_supports_certified_page_sizes_and_offsets() {
        for page_tokens in [16, 32, 64] {
            assert!(cuda_paged_decode_page_tokens_supported(page_tokens));
            let context_len = page_tokens * 2;
            let first_page_offset = page_tokens - 1;
            let metadata = vec![context_len as u32, first_page_offset as u32, 0, 1, 2];
            validate_cuda_paged_decode_metadata(&metadata, 1, page_tokens, 3, 3).unwrap();
        }
        for page_tokens in [0, 1, 8, 128] {
            assert!(!cuda_paged_decode_page_tokens_supported(page_tokens));
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_paged_decode_softcap_matches_reference_for_supported_dtypes() {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return;
        };
        let cpu = candle_core::Device::Cpu;
        let query_data = vec![2.0f32, -1.0];
        let mut key_data = vec![0.0f32; 16 * 2];
        key_data[..4].copy_from_slice(&[4.0, 0.0, 0.0, 2.0]);
        let mut value_data = vec![0.0f32; 16 * 2];
        value_data[..4].copy_from_slice(&[1.0, 3.0, 5.0, -2.0]);
        let metadata = vec![2, 0, 0];
        let softcap = 0.5f32;
        let raw_scores = [8.0f32, -2.0];
        let scores = raw_scores.map(|score| softcap * (score / softcap).tanh());
        let max_score = scores[0].max(scores[1]);
        let weights = scores.map(|score| (score - max_score).exp());
        let denominator = weights[0] + weights[1];
        let expected = [
            (weights[0] * 1.0 + weights[1] * 5.0) / denominator,
            (weights[0] * 3.0 + weights[1] * -2.0) / denominator,
        ];

        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let device_metadata =
                Tensor::from_vec(metadata.clone(), metadata.len(), &device).unwrap();
            let queries = Tensor::from_vec(query_data.clone(), (1, 1, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let keys = Tensor::from_vec(key_data.clone(), (1, 16, 1, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let values = Tensor::from_vec(value_data.clone(), (1, 16, 1, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let actual = paged_decode_attention(
                &queries,
                &keys,
                &values,
                &device_metadata,
                1,
                1,
                1,
                16,
                1,
                2,
                2,
                1.0,
                Some(softcap),
                2,
                Some((
                    CUDA_PAGED_DECODE_SPLIT_THRESHOLD_TOKENS,
                    CUDA_PAGED_DECODE_PARTITION_TOKENS,
                )),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let tolerance = if dtype == DType::F32 { 1e-5 } else { 5e-3 };
            for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} softcap mismatch at {index}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_lfm_shortconv_consumes_wrapped_physical_ring() {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return;
        };
        let ring = Tensor::from_vec(
            vec![
                12.0f32, 22.0, // physical slot 0 = absolute step 3
                10.0, 20.0, // physical slot 1 = absolute step 1
                11.0, 21.0, // physical slot 2 = absolute step 2
            ],
            (3, 1, 2),
            &device,
        )
        .unwrap();
        let input = Tensor::from_vec(vec![13.0f32, 14.0, 23.0, 24.0], (1, 2, 2), &device).unwrap();
        let weight =
            Tensor::from_vec(vec![1.0f32, 10.0, 100.0, -1.0, 0.5, 2.0], (2, 3), &device).unwrap();
        let output = try_lfm_shortconv_ring_sequence(&ring, &input, &weight, 4, 3)
            .expect("physical ShortConv ring kernel should run on CUDA")
            .to_device(&candle_core::Device::Cpu)
            .unwrap();
        let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = vec![
            11.0 + 12.0 * 10.0 + 13.0 * 100.0,
            12.0 + 13.0 * 10.0 + 14.0 * 100.0,
            -21.0 + 22.0 * 0.5 + 23.0 * 2.0,
            -22.0 + 23.0 * 0.5 + 24.0 * 2.0,
        ];
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "physical ShortConv mismatch at {index}: {actual} != {expected}"
            );
        }
    }
}
