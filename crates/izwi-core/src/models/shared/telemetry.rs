//! Model-agnostic kernel path telemetry counters.
//!
//! These counters are intentionally architecture-neutral so they can be reused
//! across model families while still surfacing hot-path behavior.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default)]
pub struct KernelPathTelemetrySnapshot {
    pub host_read_ops_total: u64,
    pub host_read_bytes_total: u64,
    pub dtype_cast_ops_total: u64,
    pub layout_copy_ops_total: u64,
    pub prefill_token_mode_steps_total: u64,
    pub prefill_sequence_spans_total: u64,
    pub prefill_sequence_tokens_total: u64,
    pub decode_attention_dense_total: u64,
    pub decode_attention_paged_total: u64,
    pub chunk_attention_sequence_calls_total: u64,
    pub chunk_attention_spans_total: u64,
    pub chunk_attention_tokens_total: u64,
    pub chunk_attention_fused_spans_total: u64,
    pub chunk_attention_unfused_spans_total: u64,
    pub chunk_attention_mask_fallback_total: u64,
    pub rope_kernel_total: u64,
    pub rope_manual_total: u64,
    pub fused_attention_attempts_total: u64,
    pub fused_attention_success_total: u64,
    pub fused_attention_fallback_total: u64,
    pub fused_attention_masked_attempts_total: u64,
    pub fused_attention_masked_success_total: u64,
    pub fused_attention_masked_fallback_total: u64,
    pub fused_attention_fallback_flash_not_requested_total: u64,
    pub fused_attention_fallback_flash_not_compiled_total: u64,
    pub fused_attention_fallback_flash_mask_unsupported_total: u64,
    pub fused_attention_fallback_flash_dtype_unsupported_total: u64,
    pub fused_attention_fallback_flash_dtype_mismatch_total: u64,
    pub fused_attention_fallback_flash_compute_capability_unsupported_total: u64,
    pub fused_attention_fallback_flash_runtime_error_total: u64,
    pub fused_attention_fallback_metal_sdpa_runtime_error_total: u64,
    pub fused_attention_fallback_metal_sdpa_mask_policy_disabled_total: u64,
    pub fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total: u64,
    pub fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total: u64,
    pub fused_attention_fallback_unsupported_backend_total: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum DecodeAttentionPath {
    Dense,
    Paged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionFallbackReason {
    FlashNotRequested,
    FlashNotCompiled,
    FlashMaskUnsupported,
    FlashDTypeUnsupported,
    FlashDTypeMismatch,
    FlashComputeCapabilityUnsupported,
    FlashRuntimeError,
    MetalSdpaRuntimeError,
    MetalSdpaMaskPolicyDisabled,
    MetalSdpaMaskShapeUnsupported,
    MetalSdpaMaskDTypeUnsupported,
    UnsupportedBackend,
}

impl AttentionFallbackReason {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::FlashNotRequested => "flash_not_requested",
            Self::FlashNotCompiled => "flash_not_compiled",
            Self::FlashMaskUnsupported => "flash_mask_unsupported",
            Self::FlashDTypeUnsupported => "flash_dtype_unsupported",
            Self::FlashDTypeMismatch => "flash_dtype_mismatch",
            Self::FlashComputeCapabilityUnsupported => "flash_compute_capability_unsupported",
            Self::FlashRuntimeError => "flash_runtime_error",
            Self::MetalSdpaRuntimeError => "metal_sdpa_runtime_error",
            Self::MetalSdpaMaskPolicyDisabled => "metal_sdpa_mask_policy_disabled",
            Self::MetalSdpaMaskShapeUnsupported => "metal_sdpa_mask_shape_unsupported",
            Self::MetalSdpaMaskDTypeUnsupported => "metal_sdpa_mask_dtype_unsupported",
            Self::UnsupportedBackend => "unsupported_backend",
        }
    }
}

static PREFILL_TOKEN_MODE_STEPS_TOTAL: AtomicU64 = AtomicU64::new(0);
static PREFILL_SEQUENCE_SPANS_TOTAL: AtomicU64 = AtomicU64::new(0);
static PREFILL_SEQUENCE_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);

static DECODE_ATTENTION_DENSE_TOTAL: AtomicU64 = AtomicU64::new(0);
static DECODE_ATTENTION_PAGED_TOTAL: AtomicU64 = AtomicU64::new(0);
static CHUNK_ATTENTION_SEQUENCE_CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CHUNK_ATTENTION_SPANS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CHUNK_ATTENTION_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CHUNK_ATTENTION_FUSED_SPANS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CHUNK_ATTENTION_UNFUSED_SPANS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CHUNK_ATTENTION_MASK_FALLBACK_TOTAL: AtomicU64 = AtomicU64::new(0);

static ROPE_KERNEL_TOTAL: AtomicU64 = AtomicU64::new(0);
static ROPE_MANUAL_TOTAL: AtomicU64 = AtomicU64::new(0);
static HOST_READ_OPS_TOTAL: AtomicU64 = AtomicU64::new(0);
static HOST_READ_BYTES_TOTAL: AtomicU64 = AtomicU64::new(0);
static DTYPE_CAST_OPS_TOTAL: AtomicU64 = AtomicU64::new(0);
static LAYOUT_COPY_OPS_TOTAL: AtomicU64 = AtomicU64::new(0);

static FUSED_ATTENTION_ATTEMPTS_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTENTION_SUCCESS_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTENTION_FALLBACK_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTENTION_MASKED_ATTEMPTS_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTENTION_MASKED_SUCCESS_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTENTION_MASKED_FALLBACK_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_FLASH_NOT_REQUESTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_FLASH_NOT_COMPILED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_FLASH_MASK_UNSUPPORTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_FLASH_DTYPE_UNSUPPORTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_FLASH_DTYPE_MISMATCH_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_FLASH_COMPUTE_CAPABILITY_UNSUPPORTED_TOTAL: AtomicU64 =
    AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_FLASH_RUNTIME_ERROR_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_RUNTIME_ERROR_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_POLICY_DISABLED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_SHAPE_UNSUPPORTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_DTYPE_UNSUPPORTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_UNSUPPORTED_BACKEND_TOTAL: AtomicU64 = AtomicU64::new(0);

pub fn record_prefill_token_mode_step() {
    PREFILL_TOKEN_MODE_STEPS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_prefill_sequence_span(token_count: usize) {
    PREFILL_SEQUENCE_SPANS_TOTAL.fetch_add(1, Ordering::Relaxed);
    PREFILL_SEQUENCE_TOKENS_TOTAL.fetch_add(token_count as u64, Ordering::Relaxed);
}

pub fn record_decode_attention_path(path: DecodeAttentionPath) {
    match path {
        DecodeAttentionPath::Dense => {
            DECODE_ATTENTION_DENSE_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        DecodeAttentionPath::Paged => {
            DECODE_ATTENTION_PAGED_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Record one successful tensor-to-Rust scalar or vector materialization.
///
/// Callers must invoke this only after the read succeeds and must report the
/// dtype and element count actually materialized, rather than the source
/// tensor's wider logical shape. This keeps bounded sampling readbacks and
/// full-vocabulary fallbacks distinguishable by their byte deltas.
pub fn record_host_read(dtype: candle_core::DType, elements: usize) {
    HOST_READ_OPS_TOTAL.fetch_add(1, Ordering::Relaxed);
    let bytes = elements.saturating_mul(dtype.size_in_bytes()) as u64;
    HOST_READ_BYTES_TOTAL.fetch_add(bytes, Ordering::Relaxed);
}

pub fn record_dtype_cast() {
    DTYPE_CAST_OPS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_layout_copy() {
    LAYOUT_COPY_OPS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_chunk_attention_sequence(spans: usize, tokens: usize) {
    CHUNK_ATTENTION_SEQUENCE_CALLS_TOTAL.fetch_add(1, Ordering::Relaxed);
    CHUNK_ATTENTION_SPANS_TOTAL.fetch_add(spans as u64, Ordering::Relaxed);
    CHUNK_ATTENTION_TOKENS_TOTAL.fetch_add(tokens as u64, Ordering::Relaxed);
}

pub fn record_chunk_attention_fused_span() {
    CHUNK_ATTENTION_FUSED_SPANS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_chunk_attention_unfused_span() {
    CHUNK_ATTENTION_UNFUSED_SPANS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_chunk_attention_mask_fallback() {
    CHUNK_ATTENTION_MASK_FALLBACK_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_rope_kernel() {
    ROPE_KERNEL_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_rope_manual() {
    ROPE_MANUAL_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fused_attention_attempt() {
    FUSED_ATTENTION_ATTEMPTS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fused_attention_success() {
    FUSED_ATTENTION_SUCCESS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fused_attention_masked_attempt() {
    FUSED_ATTENTION_MASKED_ATTEMPTS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fused_attention_masked_success() {
    FUSED_ATTENTION_MASKED_SUCCESS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fused_attention_masked_fallback() {
    FUSED_ATTENTION_MASKED_FALLBACK_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fused_attention_fallback(reason: AttentionFallbackReason) {
    FUSED_ATTENTION_FALLBACK_TOTAL.fetch_add(1, Ordering::Relaxed);
    match reason {
        AttentionFallbackReason::FlashNotRequested => {
            FUSED_ATTN_FALLBACK_FLASH_NOT_REQUESTED_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::FlashNotCompiled => {
            FUSED_ATTN_FALLBACK_FLASH_NOT_COMPILED_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::FlashMaskUnsupported => {
            FUSED_ATTN_FALLBACK_FLASH_MASK_UNSUPPORTED_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::FlashDTypeUnsupported => {
            FUSED_ATTN_FALLBACK_FLASH_DTYPE_UNSUPPORTED_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::FlashDTypeMismatch => {
            FUSED_ATTN_FALLBACK_FLASH_DTYPE_MISMATCH_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::FlashComputeCapabilityUnsupported => {
            FUSED_ATTN_FALLBACK_FLASH_COMPUTE_CAPABILITY_UNSUPPORTED_TOTAL
                .fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::FlashRuntimeError => {
            FUSED_ATTN_FALLBACK_FLASH_RUNTIME_ERROR_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::MetalSdpaRuntimeError => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_RUNTIME_ERROR_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::MetalSdpaMaskPolicyDisabled => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_POLICY_DISABLED_TOTAL
                .fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::MetalSdpaMaskShapeUnsupported => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_SHAPE_UNSUPPORTED_TOTAL
                .fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::MetalSdpaMaskDTypeUnsupported => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_DTYPE_UNSUPPORTED_TOTAL
                .fetch_add(1, Ordering::Relaxed);
        }
        AttentionFallbackReason::UnsupportedBackend => {
            FUSED_ATTN_FALLBACK_UNSUPPORTED_BACKEND_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn snapshot() -> KernelPathTelemetrySnapshot {
    KernelPathTelemetrySnapshot {
        host_read_ops_total: HOST_READ_OPS_TOTAL.load(Ordering::Relaxed),
        host_read_bytes_total: HOST_READ_BYTES_TOTAL.load(Ordering::Relaxed),
        dtype_cast_ops_total: DTYPE_CAST_OPS_TOTAL.load(Ordering::Relaxed),
        layout_copy_ops_total: LAYOUT_COPY_OPS_TOTAL.load(Ordering::Relaxed),
        prefill_token_mode_steps_total: PREFILL_TOKEN_MODE_STEPS_TOTAL.load(Ordering::Relaxed),
        prefill_sequence_spans_total: PREFILL_SEQUENCE_SPANS_TOTAL.load(Ordering::Relaxed),
        prefill_sequence_tokens_total: PREFILL_SEQUENCE_TOKENS_TOTAL.load(Ordering::Relaxed),
        decode_attention_dense_total: DECODE_ATTENTION_DENSE_TOTAL.load(Ordering::Relaxed),
        decode_attention_paged_total: DECODE_ATTENTION_PAGED_TOTAL.load(Ordering::Relaxed),
        chunk_attention_sequence_calls_total: CHUNK_ATTENTION_SEQUENCE_CALLS_TOTAL
            .load(Ordering::Relaxed),
        chunk_attention_spans_total: CHUNK_ATTENTION_SPANS_TOTAL.load(Ordering::Relaxed),
        chunk_attention_tokens_total: CHUNK_ATTENTION_TOKENS_TOTAL.load(Ordering::Relaxed),
        chunk_attention_fused_spans_total: CHUNK_ATTENTION_FUSED_SPANS_TOTAL
            .load(Ordering::Relaxed),
        chunk_attention_unfused_spans_total: CHUNK_ATTENTION_UNFUSED_SPANS_TOTAL
            .load(Ordering::Relaxed),
        chunk_attention_mask_fallback_total: CHUNK_ATTENTION_MASK_FALLBACK_TOTAL
            .load(Ordering::Relaxed),
        rope_kernel_total: ROPE_KERNEL_TOTAL.load(Ordering::Relaxed),
        rope_manual_total: ROPE_MANUAL_TOTAL.load(Ordering::Relaxed),
        fused_attention_attempts_total: FUSED_ATTENTION_ATTEMPTS_TOTAL.load(Ordering::Relaxed),
        fused_attention_success_total: FUSED_ATTENTION_SUCCESS_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_total: FUSED_ATTENTION_FALLBACK_TOTAL.load(Ordering::Relaxed),
        fused_attention_masked_attempts_total: FUSED_ATTENTION_MASKED_ATTEMPTS_TOTAL
            .load(Ordering::Relaxed),
        fused_attention_masked_success_total: FUSED_ATTENTION_MASKED_SUCCESS_TOTAL
            .load(Ordering::Relaxed),
        fused_attention_masked_fallback_total: FUSED_ATTENTION_MASKED_FALLBACK_TOTAL
            .load(Ordering::Relaxed),
        fused_attention_fallback_flash_not_requested_total:
            FUSED_ATTN_FALLBACK_FLASH_NOT_REQUESTED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_flash_not_compiled_total:
            FUSED_ATTN_FALLBACK_FLASH_NOT_COMPILED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_flash_mask_unsupported_total:
            FUSED_ATTN_FALLBACK_FLASH_MASK_UNSUPPORTED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_flash_dtype_unsupported_total:
            FUSED_ATTN_FALLBACK_FLASH_DTYPE_UNSUPPORTED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_flash_dtype_mismatch_total:
            FUSED_ATTN_FALLBACK_FLASH_DTYPE_MISMATCH_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_flash_compute_capability_unsupported_total:
            FUSED_ATTN_FALLBACK_FLASH_COMPUTE_CAPABILITY_UNSUPPORTED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_flash_runtime_error_total:
            FUSED_ATTN_FALLBACK_FLASH_RUNTIME_ERROR_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_metal_sdpa_runtime_error_total:
            FUSED_ATTN_FALLBACK_METAL_SDPA_RUNTIME_ERROR_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_metal_sdpa_mask_policy_disabled_total:
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_POLICY_DISABLED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total:
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_SHAPE_UNSUPPORTED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total:
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_DTYPE_UNSUPPORTED_TOTAL.load(Ordering::Relaxed),
        fused_attention_fallback_unsupported_backend_total:
            FUSED_ATTN_FALLBACK_UNSUPPORTED_BACKEND_TOTAL.load(Ordering::Relaxed),
    }
}

pub fn prometheus() -> String {
    let metrics = snapshot();
    let fallback_reasons = [
        AttentionFallbackReason::FlashNotRequested,
        AttentionFallbackReason::FlashNotCompiled,
        AttentionFallbackReason::FlashMaskUnsupported,
        AttentionFallbackReason::FlashDTypeUnsupported,
        AttentionFallbackReason::FlashDTypeMismatch,
        AttentionFallbackReason::FlashComputeCapabilityUnsupported,
        AttentionFallbackReason::FlashRuntimeError,
        AttentionFallbackReason::MetalSdpaRuntimeError,
        AttentionFallbackReason::MetalSdpaMaskPolicyDisabled,
        AttentionFallbackReason::MetalSdpaMaskShapeUnsupported,
        AttentionFallbackReason::MetalSdpaMaskDTypeUnsupported,
        AttentionFallbackReason::UnsupportedBackend,
    ];

    let mut output = format!(
        "# TYPE izwi_kernel_host_read_ops_total counter\nizwi_kernel_host_read_ops_total {}\n\
# TYPE izwi_kernel_host_read_bytes_total counter\nizwi_kernel_host_read_bytes_total {}\n\
# TYPE izwi_kernel_dtype_cast_ops_total counter\nizwi_kernel_dtype_cast_ops_total {}\n\
# TYPE izwi_kernel_layout_copy_ops_total counter\nizwi_kernel_layout_copy_ops_total {}\n\
# TYPE izwi_kernel_prefill_token_mode_steps_total counter\nizwi_kernel_prefill_token_mode_steps_total {}\n\
# TYPE izwi_kernel_prefill_sequence_spans_total counter\nizwi_kernel_prefill_sequence_spans_total {}\n\
# TYPE izwi_kernel_prefill_sequence_tokens_total counter\nizwi_kernel_prefill_sequence_tokens_total {}\n\
# TYPE izwi_kernel_decode_attention_dense_total counter\nizwi_kernel_decode_attention_dense_total {}\n\
# TYPE izwi_kernel_decode_attention_paged_total counter\nizwi_kernel_decode_attention_paged_total {}\n\
# TYPE izwi_kernel_chunk_attention_sequence_calls_total counter\nizwi_kernel_chunk_attention_sequence_calls_total {}\n\
# TYPE izwi_kernel_chunk_attention_spans_total counter\nizwi_kernel_chunk_attention_spans_total {}\n\
# TYPE izwi_kernel_chunk_attention_tokens_total counter\nizwi_kernel_chunk_attention_tokens_total {}\n\
# TYPE izwi_kernel_chunk_attention_fused_spans_total counter\nizwi_kernel_chunk_attention_fused_spans_total {}\n\
# TYPE izwi_kernel_chunk_attention_unfused_spans_total counter\nizwi_kernel_chunk_attention_unfused_spans_total {}\n\
# TYPE izwi_kernel_chunk_attention_mask_fallback_total counter\nizwi_kernel_chunk_attention_mask_fallback_total {}\n\
# TYPE izwi_kernel_rope_kernel_total counter\nizwi_kernel_rope_kernel_total {}\n\
# TYPE izwi_kernel_rope_manual_total counter\nizwi_kernel_rope_manual_total {}\n\
# TYPE izwi_kernel_fused_attention_attempts_total counter\nizwi_kernel_fused_attention_attempts_total {}\n\
# TYPE izwi_kernel_fused_attention_success_total counter\nizwi_kernel_fused_attention_success_total {}\n\
# TYPE izwi_kernel_fused_attention_fallback_total counter\nizwi_kernel_fused_attention_fallback_total {}\n\
# TYPE izwi_kernel_fused_attention_masked_attempts_total counter\nizwi_kernel_fused_attention_masked_attempts_total {}\n\
# TYPE izwi_kernel_fused_attention_masked_success_total counter\nizwi_kernel_fused_attention_masked_success_total {}\n\
# TYPE izwi_kernel_fused_attention_masked_fallback_total counter\nizwi_kernel_fused_attention_masked_fallback_total {}\n",
        metrics.host_read_ops_total,
        metrics.host_read_bytes_total,
        metrics.dtype_cast_ops_total,
        metrics.layout_copy_ops_total,
        metrics.prefill_token_mode_steps_total,
        metrics.prefill_sequence_spans_total,
        metrics.prefill_sequence_tokens_total,
        metrics.decode_attention_dense_total,
        metrics.decode_attention_paged_total,
        metrics.chunk_attention_sequence_calls_total,
        metrics.chunk_attention_spans_total,
        metrics.chunk_attention_tokens_total,
        metrics.chunk_attention_fused_spans_total,
        metrics.chunk_attention_unfused_spans_total,
        metrics.chunk_attention_mask_fallback_total,
        metrics.rope_kernel_total,
        metrics.rope_manual_total,
        metrics.fused_attention_attempts_total,
        metrics.fused_attention_success_total,
        metrics.fused_attention_fallback_total,
        metrics.fused_attention_masked_attempts_total,
        metrics.fused_attention_masked_success_total,
        metrics.fused_attention_masked_fallback_total,
    );

    output.push_str("# TYPE izwi_kernel_fused_attention_fallback_reason_total counter\n");
    for reason in fallback_reasons {
        output.push_str(&format!(
            "izwi_kernel_fused_attention_fallback_reason_total{{reason=\"{}\"}} {}\n",
            reason.as_label(),
            fallback_total_for_reason(reason)
        ));
    }

    output
}

fn fallback_total_for_reason(reason: AttentionFallbackReason) -> u64 {
    match reason {
        AttentionFallbackReason::FlashNotRequested => {
            FUSED_ATTN_FALLBACK_FLASH_NOT_REQUESTED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::FlashNotCompiled => {
            FUSED_ATTN_FALLBACK_FLASH_NOT_COMPILED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::FlashMaskUnsupported => {
            FUSED_ATTN_FALLBACK_FLASH_MASK_UNSUPPORTED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::FlashDTypeUnsupported => {
            FUSED_ATTN_FALLBACK_FLASH_DTYPE_UNSUPPORTED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::FlashDTypeMismatch => {
            FUSED_ATTN_FALLBACK_FLASH_DTYPE_MISMATCH_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::FlashComputeCapabilityUnsupported => {
            FUSED_ATTN_FALLBACK_FLASH_COMPUTE_CAPABILITY_UNSUPPORTED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::FlashRuntimeError => {
            FUSED_ATTN_FALLBACK_FLASH_RUNTIME_ERROR_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::MetalSdpaRuntimeError => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_RUNTIME_ERROR_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::MetalSdpaMaskPolicyDisabled => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_POLICY_DISABLED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::MetalSdpaMaskShapeUnsupported => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_SHAPE_UNSUPPORTED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::MetalSdpaMaskDTypeUnsupported => {
            FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_DTYPE_UNSUPPORTED_TOTAL.load(Ordering::Relaxed)
        }
        AttentionFallbackReason::UnsupportedBackend => {
            FUSED_ATTN_FALLBACK_UNSUPPORTED_BACKEND_TOTAL.load(Ordering::Relaxed)
        }
    }
}

#[cfg(test)]
pub fn reset_for_tests() {
    for counter in [
        &HOST_READ_OPS_TOTAL,
        &HOST_READ_BYTES_TOTAL,
        &DTYPE_CAST_OPS_TOTAL,
        &LAYOUT_COPY_OPS_TOTAL,
        &PREFILL_TOKEN_MODE_STEPS_TOTAL,
        &PREFILL_SEQUENCE_SPANS_TOTAL,
        &PREFILL_SEQUENCE_TOKENS_TOTAL,
        &DECODE_ATTENTION_DENSE_TOTAL,
        &DECODE_ATTENTION_PAGED_TOTAL,
        &CHUNK_ATTENTION_SEQUENCE_CALLS_TOTAL,
        &CHUNK_ATTENTION_SPANS_TOTAL,
        &CHUNK_ATTENTION_TOKENS_TOTAL,
        &CHUNK_ATTENTION_FUSED_SPANS_TOTAL,
        &CHUNK_ATTENTION_UNFUSED_SPANS_TOTAL,
        &CHUNK_ATTENTION_MASK_FALLBACK_TOTAL,
        &ROPE_KERNEL_TOTAL,
        &ROPE_MANUAL_TOTAL,
        &FUSED_ATTENTION_ATTEMPTS_TOTAL,
        &FUSED_ATTENTION_SUCCESS_TOTAL,
        &FUSED_ATTENTION_FALLBACK_TOTAL,
        &FUSED_ATTENTION_MASKED_ATTEMPTS_TOTAL,
        &FUSED_ATTENTION_MASKED_SUCCESS_TOTAL,
        &FUSED_ATTENTION_MASKED_FALLBACK_TOTAL,
        &FUSED_ATTN_FALLBACK_FLASH_NOT_REQUESTED_TOTAL,
        &FUSED_ATTN_FALLBACK_FLASH_NOT_COMPILED_TOTAL,
        &FUSED_ATTN_FALLBACK_FLASH_MASK_UNSUPPORTED_TOTAL,
        &FUSED_ATTN_FALLBACK_FLASH_DTYPE_UNSUPPORTED_TOTAL,
        &FUSED_ATTN_FALLBACK_FLASH_DTYPE_MISMATCH_TOTAL,
        &FUSED_ATTN_FALLBACK_FLASH_COMPUTE_CAPABILITY_UNSUPPORTED_TOTAL,
        &FUSED_ATTN_FALLBACK_FLASH_RUNTIME_ERROR_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_RUNTIME_ERROR_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_POLICY_DISABLED_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_SHAPE_UNSUPPORTED_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_DTYPE_UNSUPPORTED_TOTAL,
        &FUSED_ATTN_FALLBACK_UNSUPPORTED_BACKEND_TOTAL,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_and_transform_counters_are_exported() {
        let before = snapshot();
        record_host_read(candle_core::DType::F16, 2);
        record_dtype_cast();
        record_layout_copy();
        let after = snapshot();

        assert!(after.host_read_ops_total > before.host_read_ops_total);
        assert!(after.host_read_bytes_total >= before.host_read_bytes_total + 4);
        assert!(after.dtype_cast_ops_total > before.dtype_cast_ops_total);
        assert!(after.layout_copy_ops_total > before.layout_copy_ops_total);

        let prometheus = prometheus();
        for metric in [
            "izwi_kernel_host_read_ops_total",
            "izwi_kernel_host_read_bytes_total",
            "izwi_kernel_dtype_cast_ops_total",
            "izwi_kernel_layout_copy_ops_total",
        ] {
            assert!(prometheus.contains(metric), "missing {metric}");
        }
    }
}
