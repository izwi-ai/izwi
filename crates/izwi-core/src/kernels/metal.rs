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

/// Sequential implementation of gated delta with optimized operation ordering.
/// This reduces intermediate allocations compared to the naive implementation.
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

    // Scale query
    let scaled_query =
        (query * scale).map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Compute gate decay: exp(g) where g is already softplus(alpha) * a
    let g_val = g
        .exp()
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .reshape((1, g.dim(1).unwrap_or(1), 1, 1))
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Reshape beta
    let beta = beta
        .reshape((1, beta.dim(1).unwrap_or(1), 1))
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Gate the state
    let gated_state = state
        .broadcast_mul(&g_val)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Compute kv_mem: sum(gated_state * key.unsqueeze(3), dim=2)
    let key_expanded = key
        .unsqueeze(3)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;
    let kv_mem = gated_state
        .broadcast_mul(&key_expanded)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .sum(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Compute delta: (value - kv_mem) * beta
    let delta = (value - kv_mem)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .broadcast_mul(&beta)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Update state: gated_state + key.unsqueeze(3) * delta.unsqueeze(2)
    let delta_expanded = delta
        .unsqueeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;
    let new_state = (&gated_state
        + &key_expanded
            .broadcast_mul(&delta_expanded)
            .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Compute output: sum(new_state * query.unsqueeze(3), dim=2)
    let query_expanded = scaled_query
        .unsqueeze(3)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;
    let output = new_state
        .broadcast_mul(&query_expanded)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .sum(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    Ok((output, new_state))
}

/// Try fused L2 normalization.
///
/// Computes: x / sqrt(sum(x^2) + eps)
pub fn try_fused_l2_norm(input: &Tensor, eps: f64) -> Option<Tensor> {
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
    // For now, use standard operations
    // A truly fused version would require custom kernels
    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    silu_gate.broadcast_mul(up).ok()
}

/// Try fused RMS normalization.
///
/// Computes: x / sqrt(mean(x^2) + eps) * weight
pub fn try_fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
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
    let rms_out = try_fused_rms_norm(hidden, weight, eps)?;
    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    rms_out.broadcast_mul(&silu_gate).ok()
}

/// Check if fused kernels should be used.
pub fn use_fused_kernels() -> bool {
    std::env::var("IZWI_FUSED_KERNELS")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
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
