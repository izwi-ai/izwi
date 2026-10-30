//! GPU kernel implementations for model inference optimizations.
//!
//! This module provides fused kernels that combine multiple tensor operations
//! into single GPU dispatches, reducing memory bandwidth and kernel launch overhead.
//!
//! Currently supports:
//! - Metal (Apple Silicon) kernels for Qwen3.5 DeltaNet layers
//! - CUDA kernel dispatch scaffold with Candle fallbacks
//! - Fallback to Candle operations for unsupported backends and kernels

pub mod buffer_pool;
pub mod conv_tiling;
pub mod cpu_q8;
pub mod cuda;
pub mod depthwise_conv;
pub mod f16_weight_linear;
pub(crate) mod lfm_shortconv_step;
pub mod mamba2;
pub mod metal;
#[cfg(feature = "metal")]
pub(crate) mod metal_encoder;
pub mod metal_fp8;
#[cfg(feature = "metal")]
pub(crate) mod metal_kquant_mv;
#[cfg(feature = "metal")]
pub(crate) mod metal_lfm_norm;
#[cfg(feature = "metal")]
pub(crate) mod metal_paged_gqa;
#[cfg(feature = "metal")]
pub(crate) mod metal_qwen36moe;
pub mod moe_relu2;
pub mod numeric_check;

use crate::error::Error;
use candle_core::{Device, Tensor};

pub struct FusedSiluMulResult {
    pub tensor: Tensor,
    pub used_custom_kernel: bool,
}

/// Whether any fused kernel backend is available in this build.
pub fn fused_kernels_available() -> bool {
    metal_fused_kernels_available() || cuda::fused_kernels_available()
}

/// Whether fused kernels are available for a specific device backend.
pub fn fused_kernels_available_for_device(device: &Device) -> bool {
    if device.is_metal() {
        return metal_fused_kernels_available();
    }

    if device.is_cuda() {
        return cuda::fused_kernels_available();
    }

    false
}

fn metal_fused_kernels_available() -> bool {
    crate::backends::metal_runtime_supported()
}

/// Whether to use fused kernels (can be disabled via environment).
pub fn use_fused_kernels() -> bool {
    fused_kernels_enabled_by_env() && fused_kernels_available()
}

/// Whether fused kernels are enabled and available for a specific device.
pub fn use_fused_kernels_for_device(device: &Device) -> bool {
    fused_kernels_enabled_by_env() && fused_kernels_available_for_device(device)
}

/// `IZWI_FUSED_KERNELS` (default on), read once per process. Fused ops with
/// a plain-Rust CPU implementation honour it on CPU too.
pub fn fused_kernels_enabled_by_env() -> bool {
    static FUSED_KERNELS: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_FUSED_KERNELS", true));
    FUSED_KERNELS.get()
}

pub fn try_fused_silu_mul(gate: &Tensor, up: &Tensor) -> Option<Tensor> {
    try_fused_silu_mul_with_status(gate, up).map(|result| result.tensor)
}

pub fn try_fused_silu_mul_with_status(gate: &Tensor, up: &Tensor) -> Option<FusedSiluMulResult> {
    if !use_fused_kernels_for_device(gate.device()) {
        return None;
    }

    if gate.device().is_cuda() {
        return cuda::try_fused_silu_mul_with_status(gate, up);
    }

    if gate.device().is_metal() {
        return metal::try_fused_silu_mul_with_status(gate, up);
    }

    None
}

pub fn try_fused_qk_rms_norm(
    q: &Tensor,
    k: &Tensor,
    qk_weight: &Tensor,
    eps: f64,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels_for_device(q.device()) {
        return None;
    }

    if q.device().is_cuda() {
        return cuda::try_fused_qk_rms_norm(q, k, qk_weight, eps);
    }

    if q.device().is_metal() {
        return metal::try_fused_qk_rms_norm(q, k, qk_weight, eps);
    }

    None
}

pub fn try_fused_l2_norm(input: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels_for_device(input.device()) {
        return None;
    }

    if input.device().is_cuda() {
        return cuda::try_fused_l2_norm(input, eps);
    }

    if input.device().is_metal() {
        return metal::try_fused_l2_norm(input, eps);
    }

    None
}

/// Try the Qwen3.8-specific CUDA SiLU-mul decode epilogue.
pub fn try_qwen38_silu_mul_decode(gate: &Tensor, up: &Tensor) -> Option<Tensor> {
    if !use_fused_kernels_for_device(gate.device()) || !gate.device().is_cuda() {
        return None;
    }
    cuda::try_qwen38_silu_mul_decode(gate, up)
}

/// Try the Qwen3.8-specific CUDA L2-normalization decode epilogue.
pub fn try_qwen38_l2_norm_decode(input: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels_for_device(input.device()) || !input.device().is_cuda() {
        return None;
    }
    cuda::try_qwen38_l2_norm_decode(input, eps)
}

/// Try the Qwen3.8-specific CUDA gated-RMSNorm decode epilogue.
pub fn try_qwen38_gated_rms_norm_decode(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Option<Tensor> {
    if !use_fused_kernels_for_device(hidden.device()) || !hidden.device().is_cuda() {
        return None;
    }
    cuda::try_qwen38_gated_rms_norm_decode(hidden, gate, weight, eps)
}

pub fn try_fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels_for_device(input.device()) {
        return None;
    }

    if input.device().is_cuda() {
        return cuda::try_fused_rms_norm(input, weight, eps);
    }

    if input.device().is_metal() {
        return metal::try_fused_rms_norm(input, weight, eps);
    }

    None
}

pub fn try_fused_rope_pair_bshd(
    q: &Tensor,
    k: &Tensor,
    cos_sin: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels_for_device(q.device()) {
        return None;
    }

    if q.device().is_cuda() {
        return cuda::try_fused_rope_pair_bshd(q, k, cos_sin);
    }

    if q.device().is_metal() {
        return metal::try_fused_rope_pair_bshd(q, k, cos_sin);
    }

    None
}

pub fn try_lfm_shortconv_decode3(cache: &Tensor, bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    if !use_fused_kernels_for_device(cache.device()) {
        return None;
    }

    if cache.device().is_cuda() {
        return cuda::try_lfm_shortconv_decode3(cache, bx, conv);
    }

    if cache.device().is_metal() {
        return metal::try_lfm_shortconv_decode3(cache, bx, conv);
    }

    None
}

pub fn try_lfm_shortconv_update3(cache: &Tensor, bx: &Tensor) -> Option<Tensor> {
    if !use_fused_kernels_for_device(cache.device()) {
        return None;
    }

    if cache.device().is_cuda() {
        return cuda::try_lfm_shortconv_update3(cache, bx);
    }

    if cache.device().is_metal() {
        return metal::try_lfm_shortconv_update3(cache, bx);
    }

    None
}

pub fn try_lfm_shortconv_sequence3(bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    if !use_fused_kernels_for_device(bx.device()) {
        return None;
    }

    if bx.device().is_cuda() {
        return cuda::try_lfm_shortconv_sequence3(bx, conv);
    }

    if bx.device().is_metal() {
        return metal::try_lfm_shortconv_sequence3(bx, conv);
    }

    None
}

pub fn try_lfm_shortconv_ring_sequence(
    ring: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    expected_cursor: u64,
    valid_length: u64,
) -> Option<Tensor> {
    if !use_fused_kernels_for_device(ring.device()) {
        return None;
    }

    if ring.device().is_cuda() {
        return cuda::try_lfm_shortconv_ring_sequence(
            ring,
            input,
            weight,
            expected_cursor,
            valid_length,
        );
    }

    if ring.device().is_metal() {
        return metal::try_lfm_shortconv_ring_sequence(
            ring,
            input,
            weight,
            expected_cursor,
            valid_length,
        );
    }

    None
}

pub fn try_qwen35_causal_conv_sequence(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels_for_device(input.device()) {
        return None;
    }

    if input.device().is_cuda() {
        return cuda::try_qwen35_causal_conv_sequence(input, weight, history);
    }

    if input.device().is_metal() {
        return metal::try_qwen35_causal_conv_sequence(input, weight, history);
    }

    None
}

pub fn try_qwen38_causal_conv_sequence(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels_for_device(input.device()) {
        return None;
    }

    if input.device().is_cuda() {
        return cuda::try_qwen38_causal_conv_sequence(input, weight, history);
    }

    if input.device().is_metal() {
        return metal::try_qwen38_causal_conv_sequence(input, weight, history);
    }

    None
}

pub fn try_qwen38_causal_conv_decode(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels_for_device(input.device()) || !input.device().is_cuda() {
        return None;
    }

    cuda::try_qwen38_causal_conv_decode(input, weight, history)
}

pub fn try_fused_gated_rms_norm(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Option<Tensor> {
    if !use_fused_kernels_for_device(hidden.device()) {
        return None;
    }

    if hidden.device().is_cuda() {
        return cuda::try_fused_gated_rms_norm(hidden, gate, weight, eps);
    }

    if hidden.device().is_metal() {
        return metal::try_fused_gated_rms_norm(hidden, gate, weight, eps);
    }

    None
}

pub fn try_fused_gated_delta_recurrent(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels_for_device(query.device()) {
        return None;
    }

    if query.device().is_cuda() {
        return cuda::try_fused_gated_delta_recurrent(query, key, value, g, beta, state);
    }

    if query.device().is_metal() {
        return metal::try_fused_gated_delta_recurrent(query, key, value, g, beta, state);
    }

    None
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
    if !use_fused_kernels_for_device(queries.device()) {
        return None;
    }

    if queries.device().is_cuda() {
        return cuda::try_tiled_deltanet_recurrence(
            queries,
            keys,
            values,
            g,
            beta,
            initial_state,
            tile_size,
        );
    }

    if queries.device().is_metal() {
        return metal::try_tiled_deltanet_recurrence(
            queries,
            keys,
            values,
            g,
            beta,
            initial_state,
            tile_size,
        );
    }

    None
}

/// Qwen3.8-only CUDA single-token DeltaNet recurrence candidate.
///
/// Unsupported devices and geometries return `None`, keeping the model's
/// portable recurrence as the authoritative fallback.
pub fn try_qwen38_deltanet_decode(
    mixed_qkv: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels_for_device(mixed_qkv.device()) || !mixed_qkv.device().is_cuda() {
        return None;
    }

    cuda::try_qwen38_deltanet_decode(
        mixed_qkv,
        g,
        beta,
        initial_state,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )
}

/// Result type for fused kernel operations.
pub type FusedResult<T> = std::result::Result<T, FusedKernelError>;

#[derive(Debug, Clone)]
pub enum FusedKernelError {
    BackendNotSupported(String),
    KernelNotAvailable(String),
    ExecutionError(String),
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    DtypeNotSupported(String),
}

impl std::fmt::Display for FusedKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotSupported(s) => write!(f, "Backend not supported: {}", s),
            Self::KernelNotAvailable(s) => write!(f, "Kernel not available: {}", s),
            Self::ExecutionError(s) => write!(f, "Kernel execution error: {}", s),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::DtypeNotSupported(s) => write!(f, "Dtype not supported: {}", s),
        }
    }
}

impl std::error::Error for FusedKernelError {}

impl From<FusedKernelError> for Error {
    fn from(e: FusedKernelError) -> Self {
        Error::InferenceError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_never_reports_fused_kernel_availability() {
        assert!(!fused_kernels_available_for_device(&Device::Cpu));
        assert!(!use_fused_kernels_for_device(&Device::Cpu));
    }

    #[test]
    fn global_fused_kernel_availability_tracks_known_backends() {
        assert_eq!(
            fused_kernels_available(),
            crate::backends::metal_runtime_supported() || cuda::fused_kernels_available()
        );
    }
}
