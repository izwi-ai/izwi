//! Qwen3.5/3.6-MoE MTP (multi-token prediction) draft head.
//!
//! The published qwen3_5_moe checkpoints carry ONE recurrent draft layer that
//! shares the target's token embeddings and LM head. A draft round seeds the
//! layer with a pair — the candidate token's embedding plus the predecessor
//! position's post-`output_norm` trunk hidden — and its `mtp.norm` output is
//! projected through the raw LM head directly (the trunk's `output_norm`
//! never applies to draft logits).
//!
//! The head is deliberately built through [`Qwen35WeightSource`] with the
//! `mtpblk.{layer}` logical prefix so the trunk's own attention/MLP/norm
//! loaders construct it exactly like a trunk layer; the native source
//! translates those names to the checkpoint's `mtp.layers.0.*` layout.

use candle_core::{Device, Module, Tensor, D};

use crate::backends::kv::KvWriteCompletionCollector;
use crate::error::{Error, Result};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::architectures::qwen35::chat::Qwen35TextConfig;
use crate::models::architectures::qwen35::text::{
    Qwen35FullAttention, Qwen35Mlp, Qwen35Projection, Qwen35RmsNorm, Qwen35TextModel,
    Qwen35WeightSource,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use std::sync::Arc;

/// The MTP layer occupies one virtual layer id past the last trunk layer —
/// the id the MTP KV domain binds in the state contract.
pub(crate) fn mtp_model_layer(cfg: &Qwen35TextConfig) -> u32 {
    u32::try_from(cfg.block_count).unwrap_or(u32::MAX)
}

/// Per-request latency controller. It compares elapsed time per committed
/// token, including draft, verification and prefix commit. Exploration is
/// bounded to four arms (scalar and depths 1..3), one probe every eight rounds.
/// Scheduler-limited tails do not train the controller.
#[derive(Clone, Debug)]
pub(crate) struct AdaptiveMtp {
    pub(crate) enabled: bool,
    speculation_disabled: bool,
    pub(crate) fixed_depth: usize,
    pub(crate) selected: usize,
    pub(crate) samples: [u32; 4],
    pub(crate) cost_per_token: [f64; 4],
    pub(crate) rounds: u64,
    pub(crate) probe: usize,
}

impl AdaptiveMtp {
    pub(crate) fn new(enabled: bool, starting_depth: usize) -> Self {
        let depth = starting_depth.clamp(1, 3);
        Self {
            enabled,
            speculation_disabled: false,
            fixed_depth: depth,
            selected: depth,
            samples: [0; 4],
            cost_per_token: [0.0; 4],
            rounds: 0,
            probe: 0,
        }
    }

    /// A numerical draft failure disables speculation for the entire request,
    /// including fixed-depth mode and any delayed timing observations.
    pub(crate) fn disable_after_nonfinite_draft(&mut self) {
        self.speculation_disabled = true;
    }

    pub(crate) fn speculation_disabled(&self) -> bool {
        self.speculation_disabled
    }

    /// Roll back timing policy without forgetting a numerical failure observed
    /// either before the checkpoint or in the cancelled quantum.
    pub(crate) fn restore_from_checkpoint(&mut self, checkpoint: Self) {
        let speculation_disabled = self.speculation_disabled || checkpoint.speculation_disabled;
        *self = checkpoint;
        self.speculation_disabled = speculation_disabled;
    }

    pub(crate) fn can_train(&self, budget: usize) -> bool {
        !self.speculation_disabled() && self.enabled && budget >= 4
    }

    pub(crate) fn depth(&self, budget: usize) -> usize {
        if self.speculation_disabled {
            return 0;
        }
        let ceiling = budget.saturating_sub(1).min(3);
        if !self.enabled {
            return self.fixed_depth.min(ceiling);
        }
        let arm = if self.rounds > 0 && self.rounds.is_multiple_of(8) {
            self.probe
        } else {
            self.selected
        };
        arm.min(ceiling)
    }

    pub(crate) fn observe(
        &mut self,
        depth: usize,
        committed: usize,
        elapsed: std::time::Duration,
        budget: usize,
    ) {
        if self.speculation_disabled
            || !self.enabled
            || depth > 3
            || committed == 0
            || budget < 4
            || elapsed.is_zero()
        {
            return;
        }
        let cost = elapsed.as_secs_f64() / committed as f64;
        self.cost_per_token[depth] = if self.samples[depth] == 0 {
            cost
        } else {
            self.cost_per_token[depth] * 0.75 + cost * 0.25
        };
        self.samples[depth] = self.samples[depth].saturating_add(1);
        if self.rounds > 0 && self.rounds.is_multiple_of(8) {
            self.probe = (self.probe + 1) % 4;
        }
        self.rounds = self.rounds.saturating_add(1);
        // Require a 5% advantage before switching to reduce timer noise churn.
        for candidate in 0..4 {
            if self.samples[candidate] > 0
                && self.cost_per_token[candidate] < self.cost_per_token[self.selected] * 0.95
            {
                self.selected = candidate;
            }
        }
    }
}

pub(crate) struct Qwen35MtpHead {
    hidden_size: usize,
    draft_depth: usize,
    pre_fc_norm_embedding: Qwen35RmsNorm,
    pre_fc_norm_hidden: Qwen35RmsNorm,
    fc: Qwen35Projection,
    input_layernorm: Qwen35RmsNorm,
    attention: Qwen35FullAttention,
    post_attention_norm: Qwen35RmsNorm,
    mlp: Qwen35Mlp,
    norm: Qwen35RmsNorm,
}

impl Qwen35MtpHead {
    pub(crate) fn load_via(
        source: &dyn Qwen35WeightSource,
        cfg: &Qwen35TextConfig,
        device: &Device,
        draft_depth: usize,
    ) -> Result<Self> {
        let hidden = cfg.embedding_length;
        let eps = cfg.attention_layer_norm_rms_epsilon;
        Ok(Self {
            hidden_size: hidden,
            draft_depth: draft_depth.clamp(1, 3),
            pre_fc_norm_embedding: source.rms_norm(
                "mtpblk.0.mtp_pre_fc_norm_embedding.weight",
                eps,
                device,
            )?,
            pre_fc_norm_hidden: source.rms_norm(
                "mtpblk.0.mtp_pre_fc_norm_hidden.weight",
                eps,
                device,
            )?,
            fc: source.projection("mtpblk.0.mtp_fc.weight", device)?,
            input_layernorm: source.rms_norm("mtpblk.0.attn_norm.weight", eps, device)?,
            attention: Qwen35FullAttention::load_via(source, device, "mtpblk.0", cfg)?,
            post_attention_norm: source.rms_norm(
                "mtpblk.0.post_attention_norm.weight",
                eps,
                device,
            )?,
            mlp: Qwen35Mlp::load_via(source, device, "mtpblk.0")?,
            norm: source.rms_norm("mtpblk.0.mtp_norm.weight", eps, device)?,
        })
    }

    /// Configured greedy draft depth (tokens per speculative round).
    pub(crate) fn draft_depth(&self) -> usize {
        self.draft_depth
    }

    /// Execute one (candidate embedding, predecessor hidden) pair at
    /// `position_id`, committing its MTP KV row. The output is post-`mtp.norm`
    /// and must be projected through the target's raw LM head.
    pub(crate) fn forward_step(
        &self,
        token_embedding: &Tensor,
        predecessor_hidden: &Tensor,
        position_id: [usize; 3],
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        if token_embedding.dims() != [1, 1, self.hidden_size]
            || predecessor_hidden.dims() != [1, 1, self.hidden_size]
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 MTP pair expects [1, 1, {}] inputs, got {:?} and {:?}",
                self.hidden_size,
                token_embedding.dims(),
                predecessor_hidden.dims()
            )));
        }
        let embedding = self.pre_fc_norm_embedding.forward(token_embedding)?;
        let predecessor = self.pre_fc_norm_hidden.forward(predecessor_hidden)?;
        let fused = Tensor::cat(&[&embedding, &predecessor], D::Minus1)?;
        let hidden = self.fc.forward(&fused)?;

        let mut prepared = cache.prepare_append(cache.context_len(), 1)?;
        let result = (|| -> Result<Tensor> {
            let residual = hidden.clone();
            let normalized = self.input_layernorm.forward(&hidden)?;
            let attended =
                self.attention
                    .forward_physical(&normalized, &[position_id], cache, &mut prepared, 0)?;
            let hidden = (&residual + &attended)?;
            let residual = hidden.clone();
            let normalized = self.post_attention_norm.forward(&hidden)?;
            let mlp = self.mlp.forward(&normalized)?;
            let hidden = (&residual + &mlp)?;
            self.norm
                .forward(&hidden)
                .map_err(crate::error::Error::from)
        })();
        match result {
            Ok(hidden) => {
                cache.commit_prepared(prepared)?;
                Ok(hidden)
            }
            Err(error) => match cache.abort_prepared(prepared) {
                Ok(()) => Err(error),
                Err(abort) => Err(Error::InferenceError(format!(
                    "Qwen3.5 MTP forward failed: {error}; provisional cache abort also failed: {abort}"
                ))),
            },
        }
    }

    /// Advance one MTP pair for each independently retained decode row while
    /// sharing the projection, attention, and MLP tensor dimensions. The rows'
    /// MTP caches must share one arena — the shared slot lowering and the
    /// common write-completion fence are what make the step one batch op.
    /// Returns one post-`mtp.norm` hidden per row.
    pub(crate) fn forward_steps_batch(
        &self,
        token_embeddings: &Tensor,
        predecessor_hidden: &Tensor,
        position_ids: &[[usize; 3]],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let (batch_size, token_count, hidden) = token_embeddings.dims3().map_err(|_| {
            Error::InvalidInput(
                "Qwen3.5 MTP batch embeddings must have shape [batch,1,hidden]".into(),
            )
        })?;
        if batch_size == 0
            || token_count != 1
            || hidden != self.hidden_size
            || predecessor_hidden.dims3()? != (batch_size, 1, hidden)
            || position_ids.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 MTP decode batch rows do not match".into(),
            ));
        }
        let start_positions = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        let first = &*caches[0];
        let slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(start_positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let lowered = first.arena().lower_slots(&slots)?;
        if lowered.arena_id() != first.arena().id() || lowered.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.5 MTP batch produced an incompatible slot map".into(),
            ));
        }
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(start_positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;

        let execution = (|| -> Result<Tensor> {
            let embedding = self.pre_fc_norm_embedding.forward(token_embeddings)?;
            let predecessor = self.pre_fc_norm_hidden.forward(predecessor_hidden)?;
            let fused = Tensor::cat(&[&embedding, &predecessor], D::Minus1)?;
            let hidden_states = self.fc.forward(&fused)?;
            let residual = hidden_states.clone();
            let normalized = self.input_layernorm.forward(&hidden_states)?;
            let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            let attended = self.attention.forward_physical_decode_batch(
                &normalized,
                position_ids,
                &cache_refs,
                lowered.as_ref(),
                &metadata,
                &mut completions,
                0,
            )?;
            let hidden_states = (&residual + &attended)?;
            let residual = hidden_states.clone();
            let normalized = self.post_attention_norm.forward(&hidden_states)?;
            let mlp = self.mlp.forward(&normalized)?;
            let hidden_states = (&residual + &mlp)?;
            self.norm.forward(&hidden_states).map_err(Error::from)
        })();
        let hidden_states = match execution {
            Ok(hidden) => hidden,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                        "Qwen3.5 MTP batch failed: {error}; write-fence drain also failed: {drain}"
                    ))),
                }
            }
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(start_positions[row], 1, completion.clone())?;
        }
        Ok(hidden_states)
    }

    /// Greedy recurrent draft: project the current head output through the
    /// target's raw LM head, take the argmax token, and feed
    /// `(embedding(token), head output)` back through the layer at the next
    /// continuation position. The first token never writes the MTP cache —
    /// it is selected from the seed alone — so `depth` tokens cost
    /// `depth - 1` pair forwards.
    /// Greedy recurrent draft: project the current head output through the
    /// target's raw LM head, take the argmax token, and feed
    /// `(embedding(token), head output)` back through the layer at the next
    /// continuation position. The first token never writes the MTP cache —
    /// it is selected from the seed alone — so `depth` tokens cost
    /// `depth - 1` pair forwards.
    pub(crate) fn draft_greedy(
        &self,
        text: &Qwen35TextModel,
        seed_hidden: &Tensor,
        depth: usize,
        continuation_positions: &[[usize; 3]],
        vocab_size: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Vec<u32>> {
        self.draft_recurrently(
            text,
            seed_hidden,
            depth,
            continuation_positions,
            cache,
            |logits| greedy_argmax(logits, vocab_size),
        )
    }

    /// Recurrent draft with a caller-owned selection policy: `select`
    /// receives the head output already projected through the target's raw
    /// LM head and returns the token to draft. Stochastic policies sample
    /// through the shared lossless proposal sampler; greedy policies take
    /// the clamped argmax. The first token never writes the MTP cache.
    pub(crate) fn draft_recurrently<S>(
        &self,
        text: &Qwen35TextModel,
        seed_hidden: &Tensor,
        depth: usize,
        continuation_positions: &[[usize; 3]],
        cache: &mut PhysicalPagedKvCache,
        mut select: S,
    ) -> Result<Vec<u32>>
    where
        S: FnMut(&Tensor) -> Result<u32>,
    {
        if depth == 0 {
            return Ok(Vec::new());
        }
        if continuation_positions.len() != depth - 1 {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 MTP depth {depth} requires {} continuation positions, got {}",
                depth - 1,
                continuation_positions.len()
            )));
        }
        let mut current = seed_hidden.clone();
        let mut token_ids = Vec::with_capacity(depth);
        for position_id in continuation_positions
            .iter()
            .map(|position| Some(*position))
            .chain(std::iter::once(None))
        {
            let logits = text.project_with_shared_lm_head(&current)?;
            let token = select(&logits)?;
            token_ids.push(token);
            match position_id {
                Some(position_id) => {
                    let embedding = text.embed_token_ids(&[token])?;
                    current = self.forward_step(
                        &embedding,
                        &current,
                        position_id,
                        cache,
                    )?;
                }
                None => break,
            }
        }
        Ok(token_ids)
    }
}

/// Argmax over `[1, 1, vocab]` (or `[vocab]`) draft logits, clamped to the
/// tokenizer vocabulary, or `None` when any clamped value is non-finite (the
/// caller decides between a hard error and a fallback). First-index-wins on
/// ties, mirroring the target's greedy sampler.
pub(crate) fn draft_argmax(logits: &Tensor, vocab_size: usize) -> Result<Option<u32>> {
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 MTP draft received vocab_size=0".to_string(),
        ));
    }
    let flat = logits.flatten_all()?;
    let cols = flat.dim(0)?;
    let clamped = if vocab_size < cols {
        flat.narrow(0, 0, vocab_size)?
    } else {
        flat
    };
    let values = clamped.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Ok(None);
        }
        if value > best_value {
            best = index;
            best_value = value;
        }
    }
    Ok(Some(best as u32))
}

/// Argmax over `[1, 1, vocab]` (or `[vocab]`) draft logits, clamped to the
/// tokenizer vocabulary. Greedy draft selection mirrors the target's greedy
/// sampler: first-index-wins on ties, non-finite logits are a hard error.
pub(crate) fn greedy_argmax(logits: &Tensor, vocab_size: usize) -> Result<u32> {
    draft_argmax(logits, vocab_size)?.ok_or_else(|| {
        Error::InferenceError("Qwen3.5 MTP draft produced non-finite logits".into())
    })
}

#[cfg(test)]
mod adaptive_tests {
    use super::AdaptiveMtp;
    use std::time::Duration;

    #[test]
    fn starts_shallow_explores_bounded_depths_and_selects_elapsed_cost() {
        let mut policy = AdaptiveMtp::new(true, 1);
        assert_eq!(policy.depth(4), 1);
        let mut seen = [false; 4];
        for _ in 0..160 {
            let depth = policy.depth(4);
            seen[depth] = true;
            let committed = depth + 1;
            // Depth two is fastest despite depth three accepting more tokens.
            let cost = [20, 15, 8, 12][depth];
            policy.observe(
                depth,
                committed,
                Duration::from_millis(cost * committed as u64),
                4,
            );
        }
        assert_eq!(seen, [true; 4]);
        assert_eq!(policy.selected, 2);
        assert_eq!(policy.depth(1), 0);
        assert!(policy.depth(2) <= 1);
    }

    #[test]
    fn poor_speculation_selects_scalar_and_opt_out_is_fixed() {
        let mut policy = AdaptiveMtp::new(true, 1);
        for _ in 0..80 {
            let depth = policy.depth(4);
            policy.observe(
                depth,
                1,
                Duration::from_millis(if depth == 0 { 5 } else { 30 }),
                4,
            );
        }
        assert_eq!(policy.selected, 0);
        let mut fixed = AdaptiveMtp::new(false, 3);
        fixed.observe(0, 1, Duration::from_nanos(1), 4);
        assert_eq!(fixed.depth(4), 3);
        assert_eq!(fixed.depth(2), 1);
    }

    #[test]
    fn cancellation_clone_and_scheduler_limited_tails_do_not_change_policy() {
        let base = AdaptiveMtp::new(true, 1);
        let mut cancelled = base.clone();
        cancelled.observe(1, 2, Duration::from_millis(1), 4);
        assert_eq!(base.rounds, 0);
        let mut limited = base.clone();
        limited.observe(0, 1, Duration::from_millis(1), 1);
        assert_eq!(limited.rounds, 0);
    }

    #[test]
    fn numerical_disable_blocks_fixed_depth_probes_and_delayed_observations() {
        for adaptive in [false, true] {
            let mut policy = AdaptiveMtp::new(adaptive, 3);
            assert!(!policy.speculation_disabled());
            assert_eq!(policy.depth(4), 3);
            for _ in 0..8 {
                policy.observe(3, 4, Duration::from_millis(4), 4);
            }
            let before = policy.clone();
            policy.disable_after_nonfinite_draft();
            policy.disable_after_nonfinite_draft();
            assert!(policy.speculation_disabled());
            // Events queued before the failure must not train or re-enable
            // the controller, even across multiple exploration intervals.
            for _ in 0..32 {
                for depth in 0..=3 {
                    policy.observe(depth, depth + 1, Duration::from_nanos(1), 4);
                }
            }
            for budget in [0, 1, 2, 4, usize::MAX] {
                assert_eq!(policy.depth(budget), 0);
                assert!(!policy.can_train(budget));
            }
            assert_eq!(policy.samples, before.samples);
            assert_eq!(policy.cost_per_token, before.cost_per_token);
            assert_eq!(policy.rounds, before.rounds);
            assert_eq!(policy.probe, before.probe);
            assert_eq!(policy.selected, before.selected);
        }
    }

    #[test]
    fn checkpoint_restore_keeps_either_numerical_latch_and_restores_timing_policy() {
        for current_disabled in [false, true] {
            for checkpoint_disabled in [false, true] {
                let mut checkpoint = AdaptiveMtp::new(true, 1);
                checkpoint.observe(1, 2, Duration::from_millis(2), 4);
                if checkpoint_disabled {
                    checkpoint.disable_after_nonfinite_draft();
                }
                let mut current = AdaptiveMtp::new(false, 3);
                if current_disabled {
                    current.disable_after_nonfinite_draft();
                }
                current.restore_from_checkpoint(checkpoint.clone());
                let disabled = current_disabled || checkpoint_disabled;
                assert_eq!(current.speculation_disabled(), disabled);
                assert_eq!(current.enabled, checkpoint.enabled);
                assert_eq!(current.fixed_depth, checkpoint.fixed_depth);
                assert_eq!(current.selected, checkpoint.selected);
                assert_eq!(current.samples, checkpoint.samples);
                assert_eq!(current.cost_per_token, checkpoint.cost_per_token);
                assert_eq!(current.rounds, checkpoint.rounds);
                assert_eq!(current.probe, checkpoint.probe);
                if disabled {
                    current.observe(3, 4, Duration::from_nanos(1), 4);
                    assert_eq!(current.rounds, checkpoint.rounds);
                    assert_eq!(current.depth(4), 0);
                    assert!(!current.can_train(4));
                }
            }
        }
    }
}
