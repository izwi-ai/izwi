//! Qwen3.8-only optimization evidence.
//!
//! These process-lifetime counters intentionally describe execution-path
//! selection, not timings. CUDA runtime correctness and performance still
//! require the separately captured, SHA-bound device evidence bundle.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(crate) struct Qwen38OptimizationTelemetrySnapshot {
    pub cuda_projection_calls_total: u64,
    pub cuda_q8_projection_calls_total: u64,
    pub cuda_dense_projection_calls_total: u64,
    pub cuda_attention_dtype_casts_total: u64,
    pub cuda_bf16_kv_provider_selected_total: u64,
    pub cuda_f16_kv_fallback_selected_total: u64,
    pub cuda_state_initial_allocations_total: u64,
    pub cuda_head_expansion_materializations_total: u64,
    pub cuda_silu_mul_attempts_total: u64,
    pub cuda_silu_mul_success_total: u64,
    pub cuda_silu_mul_fallback_total: u64,
    pub cuda_gated_rms_norm_attempts_total: u64,
    pub cuda_gated_rms_norm_success_total: u64,
    pub cuda_gated_rms_norm_fallback_total: u64,
    pub cuda_l2_norm_attempts_total: u64,
    pub cuda_l2_norm_success_total: u64,
    pub cuda_l2_norm_fallback_total: u64,
    pub cuda_deltanet_decode_attempts_total: u64,
    pub cuda_deltanet_decode_success_total: u64,
    pub cuda_deltanet_decode_fallback_total: u64,
    pub cuda_deltanet_specialized_decode_attempts_total: u64,
    pub cuda_deltanet_specialized_decode_success_total: u64,
    pub cuda_deltanet_specialized_decode_fallback_total: u64,
    pub cuda_deltanet_prefill_attempts_total: u64,
    pub cuda_deltanet_prefill_success_total: u64,
    pub cuda_deltanet_prefill_fallback_total: u64,
    pub cuda_causal_conv_prefill_attempts_total: u64,
    pub cuda_causal_conv_prefill_success_total: u64,
    pub cuda_causal_conv_prefill_fallback_total: u64,
    pub cuda_causal_conv_decode_attempts_total: u64,
    pub cuda_causal_conv_decode_success_total: u64,
    pub cuda_causal_conv_decode_fallback_total: u64,
    pub cuda_rope_kernel_success_total: u64,
    pub cuda_rope_manual_fallback_total: u64,
    pub sampling_device_argmax_total: u64,
    pub sampling_bounded_cuda_attempts_total: u64,
    pub sampling_bounded_cuda_success_total: u64,
    pub sampling_bounded_cuda_fallback_to_host_total: u64,
    pub sampling_host_total: u64,
    pub mtp_enabled_loads_total: u64,
    pub mtp_disabled_loads_total: u64,
    pub mtp_rounds_total: u64,
    pub mtp_draft_tokens_total: u64,
    pub mtp_accepted_draft_tokens_total: u64,
    pub mtp_rejected_rounds_total: u64,
    pub mtp_bonus_tokens_total: u64,
    pub mtp_target_verified_tokens_total: u64,
    pub mtp_target_replay_tokens_total: u64,
}

macro_rules! counters {
    ($($name:ident),+ $(,)?) => {
        $(static $name: AtomicU64 = AtomicU64::new(0);)+
    };
}

counters!(
    CUDA_PROJECTION_CALLS,
    CUDA_Q8_PROJECTION_CALLS,
    CUDA_DENSE_PROJECTION_CALLS,
    CUDA_ATTENTION_DTYPE_CASTS,
    CUDA_BF16_KV_PROVIDER_SELECTED,
    CUDA_F16_KV_FALLBACK_SELECTED,
    CUDA_STATE_INITIAL_ALLOCATIONS,
    CUDA_HEAD_EXPANSION_MATERIALIZATIONS,
    CUDA_SILU_MUL_ATTEMPTS,
    CUDA_SILU_MUL_SUCCESS,
    CUDA_SILU_MUL_FALLBACK,
    CUDA_GATED_RMS_NORM_ATTEMPTS,
    CUDA_GATED_RMS_NORM_SUCCESS,
    CUDA_GATED_RMS_NORM_FALLBACK,
    CUDA_L2_NORM_ATTEMPTS,
    CUDA_L2_NORM_SUCCESS,
    CUDA_L2_NORM_FALLBACK,
    CUDA_DELTANET_DECODE_ATTEMPTS,
    CUDA_DELTANET_DECODE_SUCCESS,
    CUDA_DELTANET_DECODE_FALLBACK,
    CUDA_DELTANET_SPECIALIZED_DECODE_ATTEMPTS,
    CUDA_DELTANET_SPECIALIZED_DECODE_SUCCESS,
    CUDA_DELTANET_SPECIALIZED_DECODE_FALLBACK,
    CUDA_DELTANET_PREFILL_ATTEMPTS,
    CUDA_DELTANET_PREFILL_SUCCESS,
    CUDA_DELTANET_PREFILL_FALLBACK,
    CUDA_CAUSAL_CONV_PREFILL_ATTEMPTS,
    CUDA_CAUSAL_CONV_PREFILL_SUCCESS,
    CUDA_CAUSAL_CONV_PREFILL_FALLBACK,
    CUDA_CAUSAL_CONV_DECODE_ATTEMPTS,
    CUDA_CAUSAL_CONV_DECODE_SUCCESS,
    CUDA_CAUSAL_CONV_DECODE_FALLBACK,
    CUDA_ROPE_KERNEL_SUCCESS,
    CUDA_ROPE_MANUAL_FALLBACK,
    SAMPLING_DEVICE_ARGMAX,
    SAMPLING_BOUNDED_CUDA_ATTEMPTS,
    SAMPLING_BOUNDED_CUDA_SUCCESS,
    SAMPLING_BOUNDED_CUDA_FALLBACK_TO_HOST,
    SAMPLING_HOST,
    MTP_ENABLED_LOADS,
    MTP_DISABLED_LOADS,
    MTP_ROUNDS,
    MTP_DRAFT_TOKENS,
    MTP_ACCEPTED_DRAFT_TOKENS,
    MTP_REJECTED_ROUNDS,
    MTP_BONUS_TOKENS,
    MTP_TARGET_VERIFIED_TOKENS,
    MTP_TARGET_REPLAY_TOKENS,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaProjectionPath {
    Q8,
    Dense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaKernelPath {
    SiluMul,
    GatedRmsNorm,
    L2Norm,
    DeltaNetDecode,
    DeltaNetSpecializedDecode,
    DeltaNetPrefill,
    CausalConvPrefill,
    CausalConvDecode,
}

pub(crate) fn record_cuda_projection(path: CudaProjectionPath) {
    CUDA_PROJECTION_CALLS.fetch_add(1, Ordering::Relaxed);
    match path {
        CudaProjectionPath::Q8 => CUDA_Q8_PROJECTION_CALLS.fetch_add(1, Ordering::Relaxed),
        CudaProjectionPath::Dense => CUDA_DENSE_PROJECTION_CALLS.fetch_add(1, Ordering::Relaxed),
    };
}

pub(crate) fn record_cuda_attention_dtype_casts(count: usize) {
    CUDA_ATTENTION_DTYPE_CASTS.fetch_add(count as u64, Ordering::Relaxed);
}

pub(crate) fn record_cuda_kv_provider(bf16_selected: bool) {
    if bf16_selected {
        CUDA_BF16_KV_PROVIDER_SELECTED.fetch_add(1, Ordering::Relaxed);
    } else {
        CUDA_F16_KV_FALLBACK_SELECTED.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_cuda_state_initial_allocation() {
    CUDA_STATE_INITIAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_cuda_head_expansion_materialization() {
    CUDA_HEAD_EXPANSION_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_cuda_kernel(path: CudaKernelPath, selected: bool) {
    let (attempts, success, fallback) = match path {
        CudaKernelPath::SiluMul => (
            &CUDA_SILU_MUL_ATTEMPTS,
            &CUDA_SILU_MUL_SUCCESS,
            &CUDA_SILU_MUL_FALLBACK,
        ),
        CudaKernelPath::GatedRmsNorm => (
            &CUDA_GATED_RMS_NORM_ATTEMPTS,
            &CUDA_GATED_RMS_NORM_SUCCESS,
            &CUDA_GATED_RMS_NORM_FALLBACK,
        ),
        CudaKernelPath::L2Norm => (
            &CUDA_L2_NORM_ATTEMPTS,
            &CUDA_L2_NORM_SUCCESS,
            &CUDA_L2_NORM_FALLBACK,
        ),
        CudaKernelPath::DeltaNetDecode => (
            &CUDA_DELTANET_DECODE_ATTEMPTS,
            &CUDA_DELTANET_DECODE_SUCCESS,
            &CUDA_DELTANET_DECODE_FALLBACK,
        ),
        CudaKernelPath::DeltaNetSpecializedDecode => (
            &CUDA_DELTANET_SPECIALIZED_DECODE_ATTEMPTS,
            &CUDA_DELTANET_SPECIALIZED_DECODE_SUCCESS,
            &CUDA_DELTANET_SPECIALIZED_DECODE_FALLBACK,
        ),
        CudaKernelPath::DeltaNetPrefill => (
            &CUDA_DELTANET_PREFILL_ATTEMPTS,
            &CUDA_DELTANET_PREFILL_SUCCESS,
            &CUDA_DELTANET_PREFILL_FALLBACK,
        ),
        CudaKernelPath::CausalConvPrefill => (
            &CUDA_CAUSAL_CONV_PREFILL_ATTEMPTS,
            &CUDA_CAUSAL_CONV_PREFILL_SUCCESS,
            &CUDA_CAUSAL_CONV_PREFILL_FALLBACK,
        ),
        CudaKernelPath::CausalConvDecode => (
            &CUDA_CAUSAL_CONV_DECODE_ATTEMPTS,
            &CUDA_CAUSAL_CONV_DECODE_SUCCESS,
            &CUDA_CAUSAL_CONV_DECODE_FALLBACK,
        ),
    };
    attempts.fetch_add(1, Ordering::Relaxed);
    if selected {
        success.fetch_add(1, Ordering::Relaxed);
    } else {
        fallback.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_cuda_rope(selected: bool) {
    if selected {
        CUDA_ROPE_KERNEL_SUCCESS.fetch_add(1, Ordering::Relaxed);
    } else {
        CUDA_ROPE_MANUAL_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_sampling_device_argmax() {
    SAMPLING_DEVICE_ARGMAX.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sampling_bounded_cuda(selected: bool) {
    SAMPLING_BOUNDED_CUDA_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    if selected {
        SAMPLING_BOUNDED_CUDA_SUCCESS.fetch_add(1, Ordering::Relaxed);
    } else {
        SAMPLING_BOUNDED_CUDA_FALLBACK_TO_HOST.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_sampling_host() {
    SAMPLING_HOST.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_mtp_policy(enabled: bool) {
    if enabled {
        MTP_ENABLED_LOADS.fetch_add(1, Ordering::Relaxed);
    } else {
        MTP_DISABLED_LOADS.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_mtp_round(
    drafted: usize,
    accepted: usize,
    emitted_bonus: bool,
    target_verified: usize,
    target_replayed: usize,
) {
    MTP_ROUNDS.fetch_add(1, Ordering::Relaxed);
    MTP_DRAFT_TOKENS.fetch_add(drafted as u64, Ordering::Relaxed);
    MTP_ACCEPTED_DRAFT_TOKENS.fetch_add(accepted as u64, Ordering::Relaxed);
    if accepted < drafted {
        MTP_REJECTED_ROUNDS.fetch_add(1, Ordering::Relaxed);
    }
    if emitted_bonus {
        MTP_BONUS_TOKENS.fetch_add(1, Ordering::Relaxed);
    }
    MTP_TARGET_VERIFIED_TOKENS.fetch_add(target_verified as u64, Ordering::Relaxed);
    MTP_TARGET_REPLAY_TOKENS.fetch_add(target_replayed as u64, Ordering::Relaxed);
}

pub(crate) fn snapshot() -> Qwen38OptimizationTelemetrySnapshot {
    macro_rules! load {
        ($name:ident) => {
            $name.load(Ordering::Relaxed)
        };
    }
    Qwen38OptimizationTelemetrySnapshot {
        cuda_projection_calls_total: load!(CUDA_PROJECTION_CALLS),
        cuda_q8_projection_calls_total: load!(CUDA_Q8_PROJECTION_CALLS),
        cuda_dense_projection_calls_total: load!(CUDA_DENSE_PROJECTION_CALLS),
        cuda_attention_dtype_casts_total: load!(CUDA_ATTENTION_DTYPE_CASTS),
        cuda_bf16_kv_provider_selected_total: load!(CUDA_BF16_KV_PROVIDER_SELECTED),
        cuda_f16_kv_fallback_selected_total: load!(CUDA_F16_KV_FALLBACK_SELECTED),
        cuda_state_initial_allocations_total: load!(CUDA_STATE_INITIAL_ALLOCATIONS),
        cuda_head_expansion_materializations_total: load!(CUDA_HEAD_EXPANSION_MATERIALIZATIONS),
        cuda_silu_mul_attempts_total: load!(CUDA_SILU_MUL_ATTEMPTS),
        cuda_silu_mul_success_total: load!(CUDA_SILU_MUL_SUCCESS),
        cuda_silu_mul_fallback_total: load!(CUDA_SILU_MUL_FALLBACK),
        cuda_gated_rms_norm_attempts_total: load!(CUDA_GATED_RMS_NORM_ATTEMPTS),
        cuda_gated_rms_norm_success_total: load!(CUDA_GATED_RMS_NORM_SUCCESS),
        cuda_gated_rms_norm_fallback_total: load!(CUDA_GATED_RMS_NORM_FALLBACK),
        cuda_l2_norm_attempts_total: load!(CUDA_L2_NORM_ATTEMPTS),
        cuda_l2_norm_success_total: load!(CUDA_L2_NORM_SUCCESS),
        cuda_l2_norm_fallback_total: load!(CUDA_L2_NORM_FALLBACK),
        cuda_deltanet_decode_attempts_total: load!(CUDA_DELTANET_DECODE_ATTEMPTS),
        cuda_deltanet_decode_success_total: load!(CUDA_DELTANET_DECODE_SUCCESS),
        cuda_deltanet_decode_fallback_total: load!(CUDA_DELTANET_DECODE_FALLBACK),
        cuda_deltanet_specialized_decode_attempts_total: load!(
            CUDA_DELTANET_SPECIALIZED_DECODE_ATTEMPTS
        ),
        cuda_deltanet_specialized_decode_success_total: load!(
            CUDA_DELTANET_SPECIALIZED_DECODE_SUCCESS
        ),
        cuda_deltanet_specialized_decode_fallback_total: load!(
            CUDA_DELTANET_SPECIALIZED_DECODE_FALLBACK
        ),
        cuda_deltanet_prefill_attempts_total: load!(CUDA_DELTANET_PREFILL_ATTEMPTS),
        cuda_deltanet_prefill_success_total: load!(CUDA_DELTANET_PREFILL_SUCCESS),
        cuda_deltanet_prefill_fallback_total: load!(CUDA_DELTANET_PREFILL_FALLBACK),
        cuda_causal_conv_prefill_attempts_total: load!(CUDA_CAUSAL_CONV_PREFILL_ATTEMPTS),
        cuda_causal_conv_prefill_success_total: load!(CUDA_CAUSAL_CONV_PREFILL_SUCCESS),
        cuda_causal_conv_prefill_fallback_total: load!(CUDA_CAUSAL_CONV_PREFILL_FALLBACK),
        cuda_causal_conv_decode_attempts_total: load!(CUDA_CAUSAL_CONV_DECODE_ATTEMPTS),
        cuda_causal_conv_decode_success_total: load!(CUDA_CAUSAL_CONV_DECODE_SUCCESS),
        cuda_causal_conv_decode_fallback_total: load!(CUDA_CAUSAL_CONV_DECODE_FALLBACK),
        cuda_rope_kernel_success_total: load!(CUDA_ROPE_KERNEL_SUCCESS),
        cuda_rope_manual_fallback_total: load!(CUDA_ROPE_MANUAL_FALLBACK),
        sampling_device_argmax_total: load!(SAMPLING_DEVICE_ARGMAX),
        sampling_bounded_cuda_attempts_total: load!(SAMPLING_BOUNDED_CUDA_ATTEMPTS),
        sampling_bounded_cuda_success_total: load!(SAMPLING_BOUNDED_CUDA_SUCCESS),
        sampling_bounded_cuda_fallback_to_host_total: load!(SAMPLING_BOUNDED_CUDA_FALLBACK_TO_HOST),
        sampling_host_total: load!(SAMPLING_HOST),
        mtp_enabled_loads_total: load!(MTP_ENABLED_LOADS),
        mtp_disabled_loads_total: load!(MTP_DISABLED_LOADS),
        mtp_rounds_total: load!(MTP_ROUNDS),
        mtp_draft_tokens_total: load!(MTP_DRAFT_TOKENS),
        mtp_accepted_draft_tokens_total: load!(MTP_ACCEPTED_DRAFT_TOKENS),
        mtp_rejected_rounds_total: load!(MTP_REJECTED_ROUNDS),
        mtp_bonus_tokens_total: load!(MTP_BONUS_TOKENS),
        mtp_target_verified_tokens_total: load!(MTP_TARGET_VERIFIED_TOKENS),
        mtp_target_replay_tokens_total: load!(MTP_TARGET_REPLAY_TOKENS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_serializes_stable_evidence_names() {
        let value = serde_json::to_value(Qwen38OptimizationTelemetrySnapshot::default()).unwrap();
        assert_eq!(value["cuda_projection_calls_total"], 0);
        assert_eq!(value["cuda_bf16_kv_provider_selected_total"], 0);
        assert_eq!(value["cuda_f16_kv_fallback_selected_total"], 0);
        assert_eq!(value["cuda_deltanet_decode_fallback_total"], 0);
        assert_eq!(value["cuda_deltanet_specialized_decode_fallback_total"], 0);
        assert_eq!(value["sampling_bounded_cuda_fallback_to_host_total"], 0);
        assert_eq!(value["mtp_rounds_total"], 0);
        assert_eq!(value["mtp_target_replay_tokens_total"], 0);
    }

    #[test]
    fn projection_and_kernel_outcomes_remain_reconcilable() {
        let before = snapshot();
        record_cuda_projection(CudaProjectionPath::Q8);
        record_cuda_projection(CudaProjectionPath::Dense);
        record_cuda_kernel(CudaKernelPath::DeltaNetDecode, true);
        record_cuda_kernel(CudaKernelPath::DeltaNetDecode, false);
        let after = snapshot();
        assert_eq!(
            after.cuda_projection_calls_total - before.cuda_projection_calls_total,
            2
        );
        assert_eq!(
            after.cuda_q8_projection_calls_total - before.cuda_q8_projection_calls_total,
            1
        );
        assert_eq!(
            after.cuda_dense_projection_calls_total - before.cuda_dense_projection_calls_total,
            1
        );
        assert_eq!(
            after.cuda_deltanet_decode_attempts_total - before.cuda_deltanet_decode_attempts_total,
            2
        );
        assert_eq!(
            after.cuda_deltanet_decode_success_total - before.cuda_deltanet_decode_success_total,
            1
        );
        assert_eq!(
            after.cuda_deltanet_decode_fallback_total - before.cuda_deltanet_decode_fallback_total,
            1
        );
    }

    #[test]
    fn mtp_acceptance_and_replay_outcomes_remain_reconcilable() {
        let before = snapshot();
        record_mtp_policy(true);
        record_mtp_policy(false);
        record_mtp_round(3, 3, true, 4, 0);
        record_mtp_round(3, 1, false, 4, 2);
        let after = snapshot();
        assert_eq!(
            after.mtp_enabled_loads_total - before.mtp_enabled_loads_total,
            1
        );
        assert_eq!(
            after.mtp_disabled_loads_total - before.mtp_disabled_loads_total,
            1
        );
        assert_eq!(after.mtp_rounds_total - before.mtp_rounds_total, 2);
        assert_eq!(
            after.mtp_draft_tokens_total - before.mtp_draft_tokens_total,
            6
        );
        assert_eq!(
            after.mtp_accepted_draft_tokens_total - before.mtp_accepted_draft_tokens_total,
            4
        );
        assert_eq!(
            after.mtp_rejected_rounds_total - before.mtp_rejected_rounds_total,
            1
        );
        assert_eq!(
            after.mtp_bonus_tokens_total - before.mtp_bonus_tokens_total,
            1
        );
        assert_eq!(
            after.mtp_target_verified_tokens_total - before.mtp_target_verified_tokens_total,
            8
        );
        assert_eq!(
            after.mtp_target_replay_tokens_total - before.mtp_target_replay_tokens_total,
            2
        );
    }

    #[test]
    fn specialized_deltanet_decode_outcomes_remain_reconcilable() {
        let before = snapshot();
        record_cuda_kernel(CudaKernelPath::DeltaNetSpecializedDecode, true);
        record_cuda_kernel(CudaKernelPath::DeltaNetSpecializedDecode, false);
        let after = snapshot();
        assert_eq!(
            after.cuda_deltanet_specialized_decode_attempts_total
                - before.cuda_deltanet_specialized_decode_attempts_total,
            2
        );
        assert_eq!(
            after.cuda_deltanet_specialized_decode_success_total
                - before.cuda_deltanet_specialized_decode_success_total,
            1
        );
        assert_eq!(
            after.cuda_deltanet_specialized_decode_fallback_total
                - before.cuda_deltanet_specialized_decode_fallback_total,
            1
        );
    }

    #[test]
    fn kv_provider_outcomes_remain_reconcilable() {
        let before = snapshot();
        record_cuda_kv_provider(true);
        record_cuda_kv_provider(false);
        let after = snapshot();
        assert_eq!(
            after.cuda_bf16_kv_provider_selected_total
                - before.cuda_bf16_kv_provider_selected_total,
            1
        );
        assert_eq!(
            after.cuda_f16_kv_fallback_selected_total - before.cuda_f16_kv_fallback_selected_total,
            1
        );
    }
}
