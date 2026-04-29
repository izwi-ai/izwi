//! Optional fused attention helpers.
//!
//! This module provides a single integration point for:
//! - CUDA FlashAttention2 via `candle-flash-attn` (when compiled with `flash-attn`)
//! - Metal fused SDPA via `candle_nn::ops::sdpa`
//!
//! Callers should treat these paths as opportunistic accelerations and always keep
//! a numerically equivalent fallback path.

use candle_core::{DType, Tensor};

use crate::error::Result;
use crate::models::shared::telemetry::{
    record_fused_attention_attempt, record_fused_attention_fallback,
    record_fused_attention_masked_attempt, record_fused_attention_masked_fallback,
    record_fused_attention_masked_success, record_fused_attention_success, AttentionFallbackReason,
};

/// Runtime opt-in for fused attention paths.
pub fn flash_attention_requested() -> bool {
    std::env::var("IZWI_USE_FLASH_ATTENTION")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Whether the build includes CUDA FlashAttention2 support.
pub const fn flash_attention_compiled() -> bool {
    cfg!(feature = "flash-attn")
}

/// Runtime check used by models that wire Candle's `use_flash_attn` flag.
pub fn should_enable_flash_attention_v2(device: &candle_core::Device) -> bool {
    flash_attention_requested() && device.is_cuda() && flash_attention_compiled()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaFlashAttentionDecision {
    Try,
    Skip(AttentionFallbackReason),
}

fn should_try_cuda_flash_attention(
    requested: bool,
    compiled: bool,
    masked: bool,
    q_dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
) -> CudaFlashAttentionDecision {
    if !requested {
        return CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashNotRequested);
    }
    if !compiled {
        return CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashNotCompiled);
    }
    if masked {
        return CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashMaskUnsupported);
    }
    if !dtype_supported_for_flash(q_dtype) {
        return CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashDTypeUnsupported);
    }
    if q_dtype != k_dtype || k_dtype != v_dtype {
        return CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashDTypeMismatch);
    }

    CudaFlashAttentionDecision::Try
}

/// Try a fused self-attention kernel and return `None` when unsupported.
///
/// Input/output layout: `[batch, heads, seq, head_dim]`.
pub fn try_fused_self_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
    causal: bool,
) -> Result<Option<Tensor>> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let masked = mask.is_some();
    record_fused_attention_attempt();
    if masked {
        record_fused_attention_masked_attempt();
    }
    let mut fallback_reason = AttentionFallbackReason::UnsupportedBackend;

    if q.device().is_cuda() {
        let cuda_decision = should_try_cuda_flash_attention(
            flash_attention_requested(),
            flash_attention_compiled(),
            masked,
            q.dtype(),
            k.dtype(),
            v.dtype(),
        );

        match cuda_decision {
            CudaFlashAttentionDecision::Try => {
                #[cfg(feature = "flash-attn")]
                {
                    let q = q.transpose(1, 2)?.contiguous()?;
                    let k = k.transpose(1, 2)?.contiguous()?;
                    let v = v.transpose(1, 2)?.contiguous()?;
                    let flash_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)
                        }));
                    match flash_result {
                        Ok(Ok(out)) => {
                            record_fused_attention_success();
                            if masked {
                                record_fused_attention_masked_success();
                            }
                            return Ok(Some(out.transpose(1, 2)?));
                        }
                        Ok(Err(_)) | Err(_) => {
                            fallback_reason = AttentionFallbackReason::FlashRuntimeError;
                        }
                    }
                }
            }
            CudaFlashAttentionDecision::Skip(reason) => {
                fallback_reason = reason;
            }
        }
    }

    if q.device().is_metal() {
        match should_try_metal_sdpa(q, k, v, mask)? {
            MetalSdpaDecision::Try => {
                let q_seq = q.dim(2)?;
                let use_f16_cast =
                    mask.is_none() && should_use_metal_sdpa_f16_cast(q.dtype(), q_seq);
                let sdpa_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if use_f16_cast {
                        run_metal_sdpa_with_f16_inputs(q, k, v, mask, causal, scale)
                    } else {
                        candle_nn::ops::sdpa(q, k, v, mask, causal, scale, 1.0)
                    }
                }));
                match sdpa_result {
                    Ok(Ok(out)) => {
                        record_fused_attention_success();
                        if masked {
                            record_fused_attention_masked_success();
                        }
                        return Ok(Some(out));
                    }
                    Ok(Err(_)) | Err(_) => {
                        fallback_reason = AttentionFallbackReason::MetalSdpaRuntimeError;
                    }
                }
            }
            MetalSdpaDecision::Skip(reason) => {
                fallback_reason = reason;
            }
        }
    }

    record_fused_attention_fallback(fallback_reason);
    if masked {
        record_fused_attention_masked_fallback();
    }
    Ok(None)
}

/// Conservative preflight for Metal SDPA.
///
/// This gate mirrors Candle's shape support checks, while keeping F32 full-SDPA
/// behind a guarded cast route to avoid known threadgroup-memory failures.
enum MetalSdpaDecision {
    Try,
    Skip(AttentionFallbackReason),
}

fn should_try_metal_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<MetalSdpaDecision> {
    if let Some(mask) = mask {
        return should_try_metal_sdpa_masked(q, k, v, mask);
    }

    let q_heads = q.dim(1)?;
    let kv_heads = k.dim(1)?;
    let v_heads = v.dim(1)?;
    let q_seq = q.dim(2)?;
    let k_seq = k.dim(2)?;
    let q_head = q.dim(3)?;
    let k_head = k.dim(3)?;

    if q.dtype() != k.dtype() || k.dtype() != v.dtype() {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::UnsupportedBackend,
        ));
    }
    let dtype_supported = matches!(q.dtype(), DType::F16 | DType::BF16 | DType::F32);
    if !dtype_supported {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::UnsupportedBackend,
        ));
    }

    if !metal_sdpa_shape_supported(q_heads, kv_heads, v_heads, q_seq, k_seq, q_head, k_head) {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::UnsupportedBackend,
        ));
    }

    // F32 + full-SDPA prefill has triggered oversized threadgroup plans on some
    // Apple GPUs. We only enable this shape when the guarded F16-cast route is on.
    if q_seq > 8 && q.dtype() == DType::F32 && !metal_sdpa_f32_prefill_cast_enabled() {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::UnsupportedBackend,
        ));
    }

    Ok(MetalSdpaDecision::Try)
}

fn should_try_metal_sdpa_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
) -> Result<MetalSdpaDecision> {
    if !metal_sdpa_masked_enabled() {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::MetalSdpaMaskPolicyDisabled,
        ));
    }

    let q_batch = q.dim(0)?;
    let q_heads = q.dim(1)?;
    let kv_heads = k.dim(1)?;
    let v_heads = v.dim(1)?;
    let q_seq = q.dim(2)?;
    let k_seq = k.dim(2)?;
    let q_head = q.dim(3)?;
    let k_head = k.dim(3)?;

    if q.dtype() != k.dtype() || k.dtype() != v.dtype() {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::MetalSdpaMaskDTypeUnsupported,
        ));
    }
    let dtype_supported = matches!(q.dtype(), DType::F16 | DType::BF16);
    if !dtype_supported {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::MetalSdpaMaskDTypeUnsupported,
        ));
    }
    if !metal_sdpa_shape_supported(q_heads, kv_heads, v_heads, q_seq, k_seq, q_head, k_head) {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::MetalSdpaMaskShapeUnsupported,
        ));
    }
    if !metal_sdpa_mask_shape_supported(mask, q_batch, q_heads, q_seq, k_seq)? {
        return Ok(MetalSdpaDecision::Skip(
            AttentionFallbackReason::MetalSdpaMaskShapeUnsupported,
        ));
    }

    Ok(MetalSdpaDecision::Try)
}

fn metal_sdpa_shape_supported(
    q_heads: usize,
    kv_heads: usize,
    v_heads: usize,
    q_seq: usize,
    k_seq: usize,
    q_head: usize,
    k_head: usize,
) -> bool {
    if kv_heads == 0 {
        return false;
    }
    if q_head != k_head {
        return false;
    }
    if v_heads != kv_heads {
        return false;
    }
    if q_heads % kv_heads != 0 {
        return false;
    }
    if q_seq > k_seq {
        return false;
    }

    let supported_head_dim = matches!(q_head, 32 | 64 | 72 | 80 | 96 | 128 | 256);
    if !supported_head_dim {
        return false;
    }

    true
}

fn metal_sdpa_mask_shape_supported(
    mask: &Tensor,
    q_batch: usize,
    q_heads: usize,
    q_seq: usize,
    k_seq: usize,
) -> Result<bool> {
    if mask.rank() != 4 {
        return Ok(false);
    }
    let (mask_batch, mask_heads, mask_q, mask_k) = mask.dims4()?;
    if mask_q != q_seq || mask_k != k_seq {
        return Ok(false);
    }
    if !(mask_batch == 1 || mask_batch == q_batch) {
        return Ok(false);
    }
    if !(mask_heads == 1 || mask_heads == q_heads) {
        return Ok(false);
    }
    Ok(true)
}

fn run_metal_sdpa_with_f16_inputs(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: f32,
) -> candle_core::Result<Tensor> {
    let q_f16 = q.to_dtype(DType::F16)?;
    let k_f16 = k.to_dtype(DType::F16)?;
    let v_f16 = v.to_dtype(DType::F16)?;
    let out = candle_nn::ops::sdpa(&q_f16, &k_f16, &v_f16, mask, causal, scale, 1.0)?;
    out.to_dtype(q.dtype())
}

fn should_use_metal_sdpa_f16_cast(dtype: DType, q_seq: usize) -> bool {
    q_seq > 8 && dtype == DType::F32 && metal_sdpa_f32_prefill_cast_enabled()
}

fn metal_sdpa_f32_prefill_cast_enabled() -> bool {
    env_bool("IZWI_METAL_SDPA_F32_PREFILL_F16", true)
}

fn metal_sdpa_masked_enabled() -> bool {
    env_bool("IZWI_METAL_SDPA_MASKED", false)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

#[inline]
fn dtype_supported_for_flash(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

#[cfg(test)]
mod tests {
    use super::{
        metal_sdpa_mask_shape_supported, metal_sdpa_shape_supported,
        should_try_cuda_flash_attention, should_use_metal_sdpa_f16_cast,
        CudaFlashAttentionDecision,
    };
    use crate::models::shared::telemetry::AttentionFallbackReason;
    use candle_core::DType;
    use candle_core::{Device, Tensor};

    #[test]
    fn metal_sdpa_shape_gate_accepts_supported_shapes() {
        assert!(metal_sdpa_shape_supported(32, 8, 8, 1, 128, 128, 128));
        assert!(metal_sdpa_shape_supported(32, 8, 8, 23, 23, 128, 128));
    }

    #[test]
    fn metal_sdpa_shape_gate_rejects_invalid_shapes() {
        assert!(!metal_sdpa_shape_supported(32, 8, 8, 24, 23, 128, 128));
        assert!(!metal_sdpa_shape_supported(30, 8, 8, 8, 8, 128, 128));
        assert!(!metal_sdpa_shape_supported(32, 8, 7, 8, 8, 128, 128));
        assert!(!metal_sdpa_shape_supported(32, 8, 8, 8, 8, 120, 120));
    }

    #[test]
    fn metal_sdpa_f16_cast_policy_only_applies_to_f32_prefill() {
        let _guard = crate::env_test_lock().lock().expect("env lock");

        std::env::remove_var("IZWI_METAL_SDPA_F32_PREFILL_F16");
        assert!(should_use_metal_sdpa_f16_cast(DType::F32, 23));
        assert!(!should_use_metal_sdpa_f16_cast(DType::F16, 23));
        assert!(!should_use_metal_sdpa_f16_cast(DType::F32, 8));

        std::env::set_var("IZWI_METAL_SDPA_F32_PREFILL_F16", "0");
        assert!(!should_use_metal_sdpa_f16_cast(DType::F32, 23));

        std::env::remove_var("IZWI_METAL_SDPA_F32_PREFILL_F16");
    }

    #[test]
    fn cuda_flash_attention_policy_has_explicit_fallback_reasons() {
        assert_eq!(
            should_try_cuda_flash_attention(false, true, false, DType::F16, DType::F16, DType::F16),
            CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashNotRequested)
        );
        assert_eq!(
            should_try_cuda_flash_attention(true, false, false, DType::F16, DType::F16, DType::F16),
            CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashNotCompiled)
        );
        assert_eq!(
            should_try_cuda_flash_attention(true, true, true, DType::F16, DType::F16, DType::F16),
            CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashMaskUnsupported)
        );
        assert_eq!(
            should_try_cuda_flash_attention(true, true, false, DType::F32, DType::F32, DType::F32),
            CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashDTypeUnsupported)
        );
        assert_eq!(
            should_try_cuda_flash_attention(true, true, false, DType::F16, DType::BF16, DType::F16),
            CudaFlashAttentionDecision::Skip(AttentionFallbackReason::FlashDTypeMismatch)
        );
        assert_eq!(
            should_try_cuda_flash_attention(true, true, false, DType::F16, DType::F16, DType::F16),
            CudaFlashAttentionDecision::Try
        );
    }

    #[test]
    fn metal_sdpa_mask_shape_gate_accepts_supported_broadcast_masks() {
        let device = Device::Cpu;
        let mask = Tensor::zeros((1, 1, 32, 32), DType::F32, &device).expect("mask");
        assert!(metal_sdpa_mask_shape_supported(&mask, 1, 8, 32, 32).expect("shape check"));

        let per_head = Tensor::zeros((1, 8, 32, 64), DType::F32, &device).expect("mask");
        assert!(metal_sdpa_mask_shape_supported(&per_head, 1, 8, 32, 64).expect("shape check"));
    }

    #[test]
    fn metal_sdpa_mask_shape_gate_rejects_invalid_masks() {
        let device = Device::Cpu;
        let bad_rank = Tensor::zeros((32, 32), DType::F32, &device).expect("mask");
        assert!(!metal_sdpa_mask_shape_supported(&bad_rank, 1, 8, 32, 32).expect("shape check"));

        let bad_dims = Tensor::zeros((2, 8, 32, 32), DType::F32, &device).expect("mask");
        assert!(!metal_sdpa_mask_shape_supported(&bad_dims, 1, 8, 32, 32).expect("shape check"));
    }
}
