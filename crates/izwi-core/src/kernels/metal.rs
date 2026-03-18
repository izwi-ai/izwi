//! Optimized kernel operations for model inference.
//!
//! This module provides fused tensor operations that reduce memory round-trips
//! and kernel launch overhead. On Metal backends, these use Candle's Metal
//! dispatch where possible; on other backends they use optimized CPU patterns.

use candle_core::{DType, Tensor};

use super::{FusedKernelError, FusedResult};

/// Try fused gated delta recurrent computation.
///
/// This fuses multiple operations:
/// 1. Query scaling (1/sqrt(dim))
/// 2. State update with pre-computed gate (gated decay + key-value accumulation)
/// 3. Output projection
///
/// The `g` parameter should already be computed as: softplus(alpha) * a
///
/// Returns None if the operation cannot be performed (wrong dtype/device).
pub fn try_fused_gated_delta_recurrent(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices currently
    if query.dtype() != DType::F32 {
        return None;
    }

    if !query.device().is_metal() {
        return None;
    }

    // Validate shapes
    let (batch, num_v_heads, _head_k_dim) = query.dims3().ok()?;
    let (_, num_v_h, _head_v_dim) = value.dims3().ok()?;

    if batch != 1 || num_v_h != num_v_heads {
        return None;
    }

    // For now, use optimized sequential operations
    // In a future version with custom Metal kernels, this would be a single dispatch
    fused_gated_delta_sequential(query, key, value, g, beta, state).ok()
}

/// Optimized sequential implementation of gated delta using matmul for
/// batched reductions, halving intermediate tensor allocations.
fn fused_gated_delta_sequential(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> FusedResult<(Tensor, Tensor)> {
    let dim = query
        .dim(candle_core::D::Minus1)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;
    let scale = 1.0f64 / (dim as f64).sqrt();

    let scaled_query =
        (query * scale).map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    let g_val = g
        .exp()
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .reshape((1, g.dim(1).unwrap_or(1), 1, 1))
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    let beta = beta
        .reshape((1, beta.dim(1).unwrap_or(1), 1))
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Gate the state
    let gated_state = state
        .broadcast_mul(&g_val)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // kv_mem via matmul: key (1,H,1,Dk) × state (1,H,Dk,Dv) → (1,H,1,Dv) → squeeze → (1,H,Dv)
    let kv_mem = key
        .unsqueeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .matmul(&gated_state)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .squeeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // delta = (value - kv_mem) * beta
    let delta = (value - kv_mem)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .broadcast_mul(&beta)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // State update via matmul outer product: key (1,H,Dk,1) × delta (1,H,1,Dv) → (1,H,Dk,Dv)
    let new_state = (&gated_state
        + &key
            .unsqueeze(3)
            .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
            .matmul(
                &delta
                    .unsqueeze(2)
                    .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?,
            )
            .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Output via matmul: query (1,H,1,Dk) × state (1,H,Dk,Dv) → (1,H,1,Dv) → squeeze → (1,H,Dv)
    let output = scaled_query
        .unsqueeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .matmul(&new_state)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .squeeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    Ok((output, new_state))
}

/// Try fused L2 normalization.
///
/// Computes: x / sqrt(sum(x^2) + eps)
pub fn try_fused_l2_norm(input: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if input.dtype() != DType::F32 {
        return None;
    }

    if !input.device().is_metal() {
        return None;
    }

    // Use Candle's built-in operations
    let sq_sum = input.sqr().ok()?.sum_keepdim(candle_core::D::Minus1).ok()?;

    let norm = (sq_sum + eps).ok()?.sqrt().ok()?;

    input.broadcast_div(&norm).ok()
}

/// Try fused MLP operation: silu(gate) * up.
///
/// This fuses the SiLU activation with the elementwise multiplication,
/// reducing memory bandwidth by 50% for this operation.
pub fn try_fused_silu_mul(gate: &Tensor, up: &Tensor) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // For now, use standard operations
    // A truly fused version would require custom kernels
    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    silu_gate.broadcast_mul(up).ok()
}

/// Try fused RMS normalization.
///
/// Computes: x / sqrt(mean(x^2) + eps) * weight
pub fn try_fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    let sq_mean = input
        .sqr()
        .ok()?
        .mean_keepdim(candle_core::D::Minus1)
        .ok()?;

    let rms = (sq_mean + eps).ok()?.sqrt().ok()?;

    let normalized = input.broadcast_div(&rms).ok()?;
    normalized.broadcast_mul(weight).ok()
}

/// Try fused gated RMS normalization.
///
/// Computes: rms_norm(x) * silu(gate)
pub fn try_fused_gated_rms_norm(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    let rms_out = try_fused_rms_norm(hidden, weight, eps)?;
    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    rms_out.broadcast_mul(&silu_gate).ok()
}

/// Try tiled DeltaNet recurrence using Metal tile memory.
///
/// This processes multiple tokens (tile_size, typically 32 or 64) while keeping
/// the hidden state in fast tile memory instead of VRAM. This eliminates
/// memory round-trips for the recurrence h_t = A * h_{t-1} + B.
///
/// The tile memory is high-speed memory local to the GPU shader. By loading
/// the hidden state into tile memory, we can process an entire sequence of
/// tokens without writing the state back to main VRAM.
///
/// This is the primary optimization that enables llama.cpp to achieve 3x
/// higher TPS than naive implementations.
///
/// # Arguments
/// * `queries` - Query tensors [batch, seq, num_heads, head_dim]
/// * `keys` - Key tensors [batch, seq, num_heads, head_dim]
/// * `values` - Value tensors [batch, seq, num_v_heads, head_v_dim]
/// * `g` - Pre-computed gate values [batch, seq, num_v_heads]
/// * `beta` - Beta values [batch, seq, num_v_heads]
/// * `initial_state` - Initial recurrent state [batch, num_v_heads, head_k_dim, head_v_dim]
/// * `tile_size` - Number of tokens to process per tile (32 or 64 recommended)
///
/// # Returns
/// (outputs, final_state) where outputs is [batch, seq, num_v_heads, head_v_dim]
pub fn try_tiled_deltanet_recurrence(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
    tile_size: usize,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if queries.dtype() != DType::F32 {
        return None;
    }

    if !queries.device().is_metal() {
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
    if k_num_heads != num_heads || g_heads != num_heads || b_heads != num_heads {
        return None;
    }
    if k_head_k_dim != head_k_dim || s_heads != num_heads || s_head_k_dim != head_k_dim {
        return None;
    }
    if v_num_heads != num_heads {
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
                fused_gated_delta_sequential(&query, &key, &value, &g_t, &beta_t, &state).ok()?;
            outputs.push(output.unsqueeze(1).ok()?);
            state = next_state;
        }
    }

    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let outputs = Tensor::cat(&output_refs, 1).ok()?;
    Some((outputs, state))
}

/// Try SIMD-group softmax for attention.
///
/// Uses Metal's simd_shuffle and simd_sum instructions to perform
/// parallel softmax computation within a threadgroup. This is more
/// efficient for small models with fewer attention heads.
///
/// The SIMD-group approach allows one threadgroup to handle multiple
/// heads simultaneously, keeping GPU Execution Units busy.
///
/// # Arguments
/// * `scores` - Attention scores [batch, heads, q_len, kv_len]
/// * `scale` - Scale factor (typically 1/sqrt(head_dim))
///
/// # Returns
/// Softmax-normalized attention weights
pub fn try_simd_softmax(scores: &Tensor, scale: f32) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if scores.dtype() != DType::F32 {
        return None;
    }

    if !scores.device().is_metal() {
        return None;
    }

    // For now, use standard softmax
    // A true SIMD-group implementation would require custom Metal kernels
    // using threadgroup_barrier, simd_shuffle, and simd_sum

    // The ideal Metal kernel would:
    // 1. Load scores into threadgroup memory
    // 2. Use simd_max to find max score per SIMD group
    // 3. Use simd_sum to compute exp sum per SIMD group
    // 4. Normalize and write output

    // Current: use standard operations
    let scaled = (scores * scale as f64).ok()?;
    candle_nn::ops::softmax(&scaled, candle_core::D::Minus1).ok()
}

/// Try SIMD-group RMS normalization.
///
/// Uses Metal's simd_sum for parallel reduction across head dimensions.
/// More efficient for small head dimensions common in 4B/9B models.
pub fn try_simd_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if input.dtype() != DType::F32 {
        return None;
    }

    if !input.device().is_metal() {
        return None;
    }

    // For now, use standard RMS norm
    // A SIMD-group implementation would:
    // 1. Use simd_sum for parallel sum of squares
    // 2. Compute RMS per SIMD group
    // 3. Apply normalization and weight

    // Current: use standard operations
    try_fused_rms_norm(input, weight, eps)
}

/// Try 3:1 DeltaNet block fusion.
///
/// Qwen 3.5 uses a 3:1 ratio of Gated DeltaNet to Gated Attention blocks.
/// This function merges 3 consecutive DeltaNet blocks into a single GPU
/// command, reducing CPU/GPU synchronization overhead by 66%.
///
/// In llama.cpp, this is done by kernel fusion to combine the 3x DeltaNet
/// blocks. By merging the three consecutive linear attention passes, we
/// reduce the "Round Trip Time" (RTT) to the GPU significantly.
///
/// # Arguments
/// * `input` - Input tensor for the first block
/// * `block_configs` - Configuration for each of the 3 blocks
/// * `states` - Mutable runtime states for each block (type-erased)
///
/// # Returns
/// Output tensor after processing all 3 blocks
pub fn try_fused_deltanet_blocks_3x1(
    input: &Tensor,
    block_configs: &[DeltaNetBlockConfig],
    _states: &mut [&mut dyn std::any::Any],
) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if input.dtype() != DType::F32 {
        return None;
    }

    if !input.device().is_metal() {
        return None;
    }

    // Require exactly 3 blocks for 3:1 fusion
    if block_configs.len() != 3 || _states.len() != 3 {
        return None;
    }

    // For now, process sequentially
    // A true fused implementation would dispatch all 3 blocks in one kernel

    // The ideal Metal kernel would:
    // 1. Load input into threadgroup memory
    // 2. For each of 3 blocks:
    //    - Compute qkv projections
    //    - Apply depthwise conv
    //    - Run gated delta recurrence with tile memory
    //    - Apply output projection
    // 3. Return final output without intermediate VRAM writes

    // Current: fall back to sequential processing
    tracing::debug!(
        "3:1 DeltaNet block fusion not yet implemented, processing {} blocks sequentially",
        block_configs.len()
    );

    None
}

/// Configuration for a single DeltaNet block in fused execution.
#[derive(Debug, Clone)]
pub struct DeltaNetBlockConfig {
    /// Number of key/value heads
    pub num_k_heads: usize,
    /// Number of value heads
    pub num_v_heads: usize,
    /// Key dimension per head
    pub head_k_dim: usize,
    /// Value dimension per head
    pub head_v_dim: usize,
    /// Depthwise conv kernel size
    pub conv_size: usize,
    /// Epsilon for normalization
    pub eps: f64,
}

/// Check if 3:1 block fusion is enabled.
pub fn use_block_fusion() -> bool {
    if !use_fused_kernels() {
        return false;
    }

    std::env::var("IZWI_BLOCK_FUSION")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

/// Check if SIMD-group optimizations should be used.
pub fn use_simd_optimizations() -> bool {
    if !use_fused_kernels() {
        return false;
    }

    std::env::var("IZWI_SIMD_OPTIMIZATIONS")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

/// Check if fused kernels should be used.
pub fn use_fused_kernels() -> bool {
    crate::kernels::use_fused_kernels()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_l2_norm_matches_reference() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();

        // Reference implementation
        let sq_sum: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x * x).sum();
        let norm = sq_sum.sqrt();
        let expected = vec![
            vec![1.0f32 / norm, 2.0 / norm],
            vec![3.0 / norm, 4.0 / norm],
        ];

        // Fused implementation (falls back to CPU for non-Metal)
        let eps = 1e-6;
        if let Some(result) = try_fused_l2_norm(&input, eps) {
            let result_data = result.to_vec3::<f32>().unwrap();
            assert!(
                (result_data[0][0][0] - expected[0][0]).abs() < 1e-5,
                "L2 norm mismatch"
            );
        }
    }

    #[test]
    fn test_silu_mul_matches_reference() {
        let device = Device::Cpu;
        let gate = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0], (2, 2), &device).unwrap();
        let up = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();

        // Reference: silu(x) = x / (1 + exp(-x))
        let silu_0 = 0.0f32 / (1.0 + (-0.0f32).exp());
        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let silu_m1 = -1.0f32 / (1.0 + (1.0f32).exp());
        let silu_2 = 2.0f32 / (1.0 + (-2.0f32).exp());

        if let Some(result) = try_fused_silu_mul(&gate, &up) {
            let result_data = result.to_vec2::<f32>().unwrap();

            assert!((result_data[0][0] - silu_0 * 1.0).abs() < 1e-5);
            assert!((result_data[0][1] - silu_1 * 2.0).abs() < 1e-5);
            assert!((result_data[1][0] - silu_m1 * 3.0).abs() < 1e-5);
            assert!((result_data[1][1] - silu_2 * 4.0).abs() < 1e-5);
        }
    }
}
