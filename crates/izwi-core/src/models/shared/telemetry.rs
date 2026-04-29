//! Model-agnostic kernel path telemetry counters.
//!
//! These counters are intentionally architecture-neutral so they can be reused
//! across model families while still surfacing hot-path behavior.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default)]
pub struct KernelPathTelemetrySnapshot {
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
    pub fused_attention_fallback_flash_runtime_error_total: u64,
    pub fused_attention_fallback_metal_sdpa_runtime_error_total: u64,
    pub fused_attention_fallback_metal_sdpa_mask_policy_disabled_total: u64,
    pub fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total: u64,
    pub fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total: u64,
    pub fused_attention_fallback_unsupported_backend_total: u64,
    pub cuda_kernel_attempts_total: u64,
    pub cuda_kernel_success_total: u64,
    pub cuda_kernel_fallback_total: u64,
    pub cuda_kernel_fallback_not_compiled_total: u64,
    pub cuda_kernel_fallback_not_cuda_device_total: u64,
    pub cuda_kernel_fallback_not_implemented_total: u64,
    pub cuda_kernel_fallback_unsupported_dtype_total: u64,
    pub cuda_kernel_fallback_unsupported_shape_total: u64,
    pub cuda_kernel_fallback_capability_missing_total: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum DecodeAttentionPath {
    Dense,
    Paged,
}

#[derive(Debug, Clone, Copy)]
pub enum AttentionFallbackReason {
    FlashNotRequested,
    FlashNotCompiled,
    FlashMaskUnsupported,
    FlashDTypeUnsupported,
    FlashDTypeMismatch,
    FlashRuntimeError,
    MetalSdpaRuntimeError,
    MetalSdpaMaskPolicyDisabled,
    MetalSdpaMaskShapeUnsupported,
    MetalSdpaMaskDTypeUnsupported,
    UnsupportedBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernelFallbackReason {
    NotCompiled,
    NotCudaDevice,
    NotImplemented,
    UnsupportedDType,
    UnsupportedShape,
    CapabilityMissing,
}

impl CudaKernelFallbackReason {
    fn as_label(self) -> &'static str {
        match self {
            Self::NotCompiled => "not_compiled",
            Self::NotCudaDevice => "not_cuda_device",
            Self::NotImplemented => "not_implemented",
            Self::UnsupportedDType => "unsupported_dtype",
            Self::UnsupportedShape => "unsupported_shape",
            Self::CapabilityMissing => "capability_missing",
        }
    }
}

impl AttentionFallbackReason {
    fn as_label(self) -> &'static str {
        match self {
            Self::FlashNotRequested => "flash_not_requested",
            Self::FlashNotCompiled => "flash_not_compiled",
            Self::FlashMaskUnsupported => "flash_mask_unsupported",
            Self::FlashDTypeUnsupported => "flash_dtype_unsupported",
            Self::FlashDTypeMismatch => "flash_dtype_mismatch",
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
static FUSED_ATTN_FALLBACK_FLASH_RUNTIME_ERROR_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_RUNTIME_ERROR_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_POLICY_DISABLED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_SHAPE_UNSUPPORTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_DTYPE_UNSUPPORTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static FUSED_ATTN_FALLBACK_UNSUPPORTED_BACKEND_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_ATTEMPTS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_SUCCESS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_FALLBACK_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_FALLBACK_NOT_COMPILED_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_FALLBACK_NOT_CUDA_DEVICE_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_FALLBACK_NOT_IMPLEMENTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_FALLBACK_UNSUPPORTED_DTYPE_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_FALLBACK_UNSUPPORTED_SHAPE_TOTAL: AtomicU64 = AtomicU64::new(0);
static CUDA_KERNEL_FALLBACK_CAPABILITY_MISSING_TOTAL: AtomicU64 = AtomicU64::new(0);

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

pub fn record_cuda_kernel_attempt() {
    CUDA_KERNEL_ATTEMPTS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_cuda_kernel_success() {
    CUDA_KERNEL_SUCCESS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_cuda_kernel_fallback(reason: CudaKernelFallbackReason) {
    CUDA_KERNEL_FALLBACK_TOTAL.fetch_add(1, Ordering::Relaxed);
    match reason {
        CudaKernelFallbackReason::NotCompiled => {
            CUDA_KERNEL_FALLBACK_NOT_COMPILED_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        CudaKernelFallbackReason::NotCudaDevice => {
            CUDA_KERNEL_FALLBACK_NOT_CUDA_DEVICE_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        CudaKernelFallbackReason::NotImplemented => {
            CUDA_KERNEL_FALLBACK_NOT_IMPLEMENTED_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        CudaKernelFallbackReason::UnsupportedDType => {
            CUDA_KERNEL_FALLBACK_UNSUPPORTED_DTYPE_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        CudaKernelFallbackReason::UnsupportedShape => {
            CUDA_KERNEL_FALLBACK_UNSUPPORTED_SHAPE_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        CudaKernelFallbackReason::CapabilityMissing => {
            CUDA_KERNEL_FALLBACK_CAPABILITY_MISSING_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn snapshot() -> KernelPathTelemetrySnapshot {
    KernelPathTelemetrySnapshot {
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
        cuda_kernel_attempts_total: CUDA_KERNEL_ATTEMPTS_TOTAL.load(Ordering::Relaxed),
        cuda_kernel_success_total: CUDA_KERNEL_SUCCESS_TOTAL.load(Ordering::Relaxed),
        cuda_kernel_fallback_total: CUDA_KERNEL_FALLBACK_TOTAL.load(Ordering::Relaxed),
        cuda_kernel_fallback_not_compiled_total: CUDA_KERNEL_FALLBACK_NOT_COMPILED_TOTAL
            .load(Ordering::Relaxed),
        cuda_kernel_fallback_not_cuda_device_total: CUDA_KERNEL_FALLBACK_NOT_CUDA_DEVICE_TOTAL
            .load(Ordering::Relaxed),
        cuda_kernel_fallback_not_implemented_total: CUDA_KERNEL_FALLBACK_NOT_IMPLEMENTED_TOTAL
            .load(Ordering::Relaxed),
        cuda_kernel_fallback_unsupported_dtype_total: CUDA_KERNEL_FALLBACK_UNSUPPORTED_DTYPE_TOTAL
            .load(Ordering::Relaxed),
        cuda_kernel_fallback_unsupported_shape_total: CUDA_KERNEL_FALLBACK_UNSUPPORTED_SHAPE_TOTAL
            .load(Ordering::Relaxed),
        cuda_kernel_fallback_capability_missing_total:
            CUDA_KERNEL_FALLBACK_CAPABILITY_MISSING_TOTAL.load(Ordering::Relaxed),
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
        AttentionFallbackReason::FlashRuntimeError,
        AttentionFallbackReason::MetalSdpaRuntimeError,
        AttentionFallbackReason::MetalSdpaMaskPolicyDisabled,
        AttentionFallbackReason::MetalSdpaMaskShapeUnsupported,
        AttentionFallbackReason::MetalSdpaMaskDTypeUnsupported,
        AttentionFallbackReason::UnsupportedBackend,
    ];
    let cuda_fallback_reasons = [
        CudaKernelFallbackReason::NotCompiled,
        CudaKernelFallbackReason::NotCudaDevice,
        CudaKernelFallbackReason::NotImplemented,
        CudaKernelFallbackReason::UnsupportedDType,
        CudaKernelFallbackReason::UnsupportedShape,
        CudaKernelFallbackReason::CapabilityMissing,
    ];

    let mut output = format!(
        "# TYPE izwi_kernel_prefill_token_mode_steps_total counter\nizwi_kernel_prefill_token_mode_steps_total {}\n\
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
# TYPE izwi_kernel_fused_attention_masked_fallback_total counter\nizwi_kernel_fused_attention_masked_fallback_total {}\n\
# TYPE izwi_cuda_kernel_attempts_total counter\nizwi_cuda_kernel_attempts_total {}\n\
# TYPE izwi_cuda_kernel_success_total counter\nizwi_cuda_kernel_success_total {}\n\
# TYPE izwi_cuda_kernel_fallback_total counter\nizwi_cuda_kernel_fallback_total {}\n",
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
        metrics.cuda_kernel_attempts_total,
        metrics.cuda_kernel_success_total,
        metrics.cuda_kernel_fallback_total,
    );

    output.push_str("# TYPE izwi_kernel_fused_attention_fallback_reason_total counter\n");
    for reason in fallback_reasons {
        output.push_str(&format!(
            "izwi_kernel_fused_attention_fallback_reason_total{{reason=\"{}\"}} {}\n",
            reason.as_label(),
            fallback_total_for_reason(reason)
        ));
    }
    output.push_str("# TYPE izwi_cuda_kernel_fallback_reason_total counter\n");
    for reason in cuda_fallback_reasons {
        output.push_str(&format!(
            "izwi_cuda_kernel_fallback_reason_total{{reason=\"{}\"}} {}\n",
            reason.as_label(),
            cuda_fallback_total_for_reason(reason)
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

fn cuda_fallback_total_for_reason(reason: CudaKernelFallbackReason) -> u64 {
    match reason {
        CudaKernelFallbackReason::NotCompiled => {
            CUDA_KERNEL_FALLBACK_NOT_COMPILED_TOTAL.load(Ordering::Relaxed)
        }
        CudaKernelFallbackReason::NotCudaDevice => {
            CUDA_KERNEL_FALLBACK_NOT_CUDA_DEVICE_TOTAL.load(Ordering::Relaxed)
        }
        CudaKernelFallbackReason::NotImplemented => {
            CUDA_KERNEL_FALLBACK_NOT_IMPLEMENTED_TOTAL.load(Ordering::Relaxed)
        }
        CudaKernelFallbackReason::UnsupportedDType => {
            CUDA_KERNEL_FALLBACK_UNSUPPORTED_DTYPE_TOTAL.load(Ordering::Relaxed)
        }
        CudaKernelFallbackReason::UnsupportedShape => {
            CUDA_KERNEL_FALLBACK_UNSUPPORTED_SHAPE_TOTAL.load(Ordering::Relaxed)
        }
        CudaKernelFallbackReason::CapabilityMissing => {
            CUDA_KERNEL_FALLBACK_CAPABILITY_MISSING_TOTAL.load(Ordering::Relaxed)
        }
    }
}

#[cfg(test)]
pub fn reset_for_tests() {
    for counter in [
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
        &FUSED_ATTN_FALLBACK_FLASH_RUNTIME_ERROR_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_RUNTIME_ERROR_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_POLICY_DISABLED_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_SHAPE_UNSUPPORTED_TOTAL,
        &FUSED_ATTN_FALLBACK_METAL_SDPA_MASK_DTYPE_UNSUPPORTED_TOTAL,
        &FUSED_ATTN_FALLBACK_UNSUPPORTED_BACKEND_TOTAL,
        &CUDA_KERNEL_ATTEMPTS_TOTAL,
        &CUDA_KERNEL_SUCCESS_TOTAL,
        &CUDA_KERNEL_FALLBACK_TOTAL,
        &CUDA_KERNEL_FALLBACK_NOT_COMPILED_TOTAL,
        &CUDA_KERNEL_FALLBACK_NOT_CUDA_DEVICE_TOTAL,
        &CUDA_KERNEL_FALLBACK_NOT_IMPLEMENTED_TOTAL,
        &CUDA_KERNEL_FALLBACK_UNSUPPORTED_DTYPE_TOTAL,
        &CUDA_KERNEL_FALLBACK_UNSUPPORTED_SHAPE_TOTAL,
        &CUDA_KERNEL_FALLBACK_CAPABILITY_MISSING_TOTAL,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}
