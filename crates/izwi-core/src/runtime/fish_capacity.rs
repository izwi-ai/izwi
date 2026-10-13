//! Immutable Fish load-time limits. Live resource admission remains authoritative.
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedFishServingCapacity {
    pub ar_rows: usize,
    pub prefill_tokens: u64,
    pub prefill_workspace_per_token: u64,
    pub codec_rows: usize,
    pub active_rows: usize,
    pub staged_rows: usize,
    pub ar_workspace_per_row: u64,
    pub codec_workspace_per_row: u64,
    pub planning_headroom_bytes: Option<u64>,
    pub native_batching: bool,
}

impl ResolvedFishServingCapacity {
    pub(crate) fn scalar() -> Result<Self> {
        Ok(Self {
            ar_rows: 1,
            prefill_tokens: 1,
            prefill_workspace_per_token: 512 * 1024 * 1024,
            codec_rows: 1,
            active_rows: 1,
            staged_rows: 1,
            ar_workspace_per_row: 512 * 1024 * 1024,
            codec_workspace_per_row:
                crate::models::architectures::fish_s2::codec::streaming_decode_workspace_bytes(
                    crate::models::architectures::fish_s2::FISH_S2_AUDIO_CHUNK_FRAMES,
                )?,
            planning_headroom_bytes: None,
            native_batching: false,
        })
    }

    pub(crate) fn resolve(
        model: &FishS2Config,
        config: &EngineConfig,
        headroom: Option<u64>,
        native_batching: bool,
    ) -> Result<Self> {
        let mut profile = Self::scalar()?;
        profile.planning_headroom_bytes = headroom;
        let checked_product = |values: &[usize]| -> Result<u64> {
            values.iter().try_fold(1u64, |value, next| {
                value.checked_mul(*next as u64).ok_or_else(|| {
                    Error::ModelLoadError("Fish serving capacity geometry overflow".into())
                })
            })
        };
        let effective_context = config
            .max_sequence_length
            .explicit_tokens()
            .unwrap_or(model.max_seq_len)
            .min(model.max_seq_len);
        let slow = &model.text_config;
        let fast = &model.audio_decoder_config;
        // F32 is a conservative bound for F32/F16/BF16 KV. Price the requested
        // context (native when automatic); do not reduce requested output/context to advertise more rows.
        let slow_head = slow
            .head_dim
            .unwrap_or(slow.hidden_size / slow.num_attention_heads.max(1));
        let fast_head = fast
            .head_dim
            .unwrap_or(fast.hidden_size / fast.num_attention_heads.max(1));
        // Price packed dense buffers, the logits row and the paged-prefill
        // attention working set. The slow stack prefills through the paged
        // kernels (`paged_prefill` on every backend), which keep online-softmax
        // state per query and head, never a query-by-context score matrix, so
        // a token costs its query, output and partial vectors whatever the
        // context (FSH-6: pricing full-context scores per token resolved the
        // prefill chunk to 29 tokens). Bound aggregate prefill tokens, not just
        // rows.
        let hidden_buffers = slow
            .hidden_size
            .checked_mul(16)
            .and_then(|bytes| {
                bytes.checked_add(
                    slow.intermediate_size
                        .unwrap_or(slow.hidden_size.saturating_mul(4))
                        .checked_mul(6)?,
                )
            })
            .ok_or_else(|| Error::ModelLoadError("Fish prefill geometry overflow".into()))?;
        let attention = checked_product(&[4, slow.num_attention_heads, slow_head + 2, 4])?;
        let dense = checked_product(&[4, hidden_buffers])?;
        profile.prefill_workspace_per_token = attention
            .checked_add(dense)
            .and_then(|bytes| bytes.checked_add((slow.vocab_size as u64).checked_mul(8)?))
            .ok_or_else(|| Error::ModelLoadError("Fish prefill workspace overflow".into()))?
            .max(1);
        profile.ar_workspace_per_row = profile
            .ar_workspace_per_row
            .max(profile.prefill_workspace_per_token);
        // Metal splits long contexts across partitions only for tiny query
        // batches (at most 64 query-head workgroups); their partials are a
        // fixed charge per prefill pass.
        let split_partials =
            checked_product(&[64, effective_context.div_ceil(512), slow_head + 2, 4])?;
        let chunk_budget = profile.ar_workspace_per_row.saturating_sub(split_partials);
        profile.prefill_tokens = (chunk_budget / profile.prefill_workspace_per_token)
            .clamp(1, 512)
            .min(effective_context as u64);
        // Scalar tensor execution can still interleave multiple independently
        // retained requests. Unknown memory leaves final fitting to the existing
        // state authority, without promoting native tensor width.
        if headroom.is_none() {
            profile.active_rows = config.max_retained_sequences.max(1);
            profile.staged_rows = config
                .max_staged_transactions
                .max(1)
                .min(profile.active_rows);
            return Ok(profile);
        }
        // Native codec packing retains one additional history copy beyond the
        // ordinary request-owned rollback pair.
        profile.codec_workspace_per_row = profile
            .codec_workspace_per_row
            .checked_add(
                crate::models::architectures::fish_s2::dac::FishS2DacConfig::current()
                    .streaming_history_bound_bytes()?,
            )
            .ok_or_else(|| Error::ModelLoadError("Fish codec packing workspace overflow".into()))?;
        let slow_bytes = checked_product(&[
            2,
            4,
            slow.num_hidden_layers,
            slow.num_key_value_heads,
            slow_head,
            effective_context,
        ])?;
        let fast_bytes = checked_product(&[
            2,
            4,
            fast.num_hidden_layers,
            fast.num_key_value_heads,
            fast_head,
            model.num_codebooks,
        ])?;
        // Both public capabilities have independent invocation banks. Price
        // prefill+decode scratch and fast state for each, plus codec candidates.
        let row_bytes = slow_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(fast_bytes.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(profile.ar_workspace_per_row.checked_mul(4)?))
            .and_then(|bytes| bytes.checked_add(profile.codec_workspace_per_row.checked_mul(4)?))
            .ok_or_else(|| Error::ModelLoadError("Fish serving row budget overflow".into()))?;
        let fixed_preparation =
            crate::models::architectures::fish_s2::codec::maximum_preparation_workspace_bytes()?
                .checked_mul(2)
                .ok_or_else(|| Error::ModelLoadError("Fish preparation budget overflow".into()))?;
        let available = headroom.unwrap_or(0);
        // Keep 20% untouched for allocator/transient uncertainty. The resource
        // authority still performs exact retained/invocation allocation and admission.
        let fitting = usize::try_from(
            available
                .saturating_sub(available / 5)
                .saturating_sub(fixed_preparation)
                / row_bytes.max(1),
        )
        .unwrap_or(usize::MAX)
        .max(1);
        let ceiling = |value: usize| {
            if value == 0 {
                fitting
            } else {
                value.min(fitting)
            }
        };
        profile.active_rows = ceiling(config.max_retained_sequences);
        profile.staged_rows = ceiling(config.max_staged_transactions).min(profile.active_rows);
        if native_batching {
            profile.ar_rows = ceiling(config.max_scheduler_batch_size)
                .min(profile.staged_rows)
                .min(config.max_batch_size.fixed_rows().unwrap_or(fitting));
        }
        profile.codec_rows = profile.ar_rows;
        profile.native_batching = profile.ar_rows > 1;
        Ok(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::fish_s2::FishS2TtsModel;

    #[test]
    fn unknown_or_unqualified_capacity_stays_scalar() {
        let model = FishS2TtsModel::for_test();
        let config = EngineConfig::default();
        for (memory, enabled) in [(None, true), (Some(u64::MAX / 2), false)] {
            let profile =
                ResolvedFishServingCapacity::resolve(model.config(), &config, memory, enabled)
                    .unwrap();
            assert_eq!(profile.ar_rows, 1);
            assert!(!profile.native_batching);
        }
    }

    #[test]
    fn explicit_context_prices_the_same_limit_as_loaded_state() {
        let model = FishS2TtsModel::for_test();
        let mut config = EngineConfig::default();
        let native =
            ResolvedFishServingCapacity::resolve(model.config(), &config, Some(256 << 30), true)
                .unwrap();
        config.max_sequence_length = crate::config::ContextLengthPreference::explicit(32).unwrap();
        let shorter =
            ResolvedFishServingCapacity::resolve(model.config(), &config, Some(256 << 30), true)
                .unwrap();
        // Paged prefill works per query, so a token costs the same at any
        // context; the shorter context still frees KV for more rows.
        assert_eq!(
            shorter.prefill_workspace_per_token,
            native.prefill_workspace_per_token
        );
        assert!(shorter.ar_rows >= native.ar_rows);
        assert!(shorter.prefill_tokens <= 32);
    }

    #[test]
    fn scalar_execution_preserves_multiple_retained_requests() {
        let model = FishS2TtsModel::for_test();
        let config = EngineConfig {
            max_retained_sequences: 12,
            max_staged_transactions: 4,
            ..Default::default()
        };
        for memory in [None, Some(256 << 30)] {
            let profile =
                ResolvedFishServingCapacity::resolve(model.config(), &config, memory, false)
                    .unwrap();
            assert_eq!(profile.ar_rows, 1);
            assert!(profile.active_rows > 1);
            assert!(profile.active_rows <= 12);
            assert!(profile.staged_rows <= 4);
        }
    }

    #[test]
    fn prefill_quantum_is_bounded_by_geometry_even_in_scalar_mode() {
        let model = FishS2TtsModel::for_test();
        let capacity = ResolvedFishServingCapacity::resolve(
            model.config(),
            &EngineConfig::default(),
            None,
            false,
        )
        .unwrap();
        assert!(capacity.prefill_tokens > 0);
        assert!(capacity.prefill_tokens <= 512);
        assert!(
            capacity.prefill_tokens * capacity.prefill_workspace_per_token
                <= capacity.ar_workspace_per_row
        );
        assert!(capacity.prefill_tokens < model.config().max_seq_len as u64);
    }

    /// FSH-6: pricing a full-context score matrix per prompt token resolved
    /// the slow-stack prefill chunk to 29 tokens, so a prompt took about 25
    /// full passes. Priced as the paged kernels run, a typical prompt
    /// prefills in a pass or two, and the chunk still fits the row's
    /// workspace.
    #[test]
    fn prefill_chunk_follows_the_paged_kernel_working_set() {
        let model = FishS2TtsModel::for_test();
        let capacity = ResolvedFishServingCapacity::resolve(
            model.config(),
            &EngineConfig::default(),
            Some(64 << 30),
            false,
        )
        .unwrap();
        assert!(
            capacity.prefill_tokens >= 256,
            "{}",
            capacity.prefill_tokens
        );
        assert!(
            capacity.prefill_tokens * capacity.prefill_workspace_per_token
                <= capacity.ar_workspace_per_row
        );
    }

    #[test]
    fn capacity_scales_with_memory_and_preserves_operator_ceilings() {
        let model = FishS2TtsModel::for_test();
        let mut config = EngineConfig {
            max_scheduler_batch_size: 0,
            max_retained_sequences: 0,
            max_staged_transactions: 0,
            ..Default::default()
        };
        let small =
            ResolvedFishServingCapacity::resolve(model.config(), &config, Some(8 << 30), true)
                .unwrap();
        let large =
            ResolvedFishServingCapacity::resolve(model.config(), &config, Some(256 << 30), true)
                .unwrap();
        assert!(large.ar_rows > small.ar_rows);
        assert!(large.ar_rows > 5);
        config.max_batch_size = crate::config::BatchSizePreference::fixed(3).unwrap();
        let capped =
            ResolvedFishServingCapacity::resolve(model.config(), &config, Some(256 << 30), true)
                .unwrap();
        assert_eq!(capped.ar_rows, 3);
        config.max_staged_transactions = 2;
        assert_eq!(
            ResolvedFishServingCapacity::resolve(model.config(), &config, Some(256 << 30), true)
                .unwrap()
                .ar_rows,
            2
        );
    }
}
