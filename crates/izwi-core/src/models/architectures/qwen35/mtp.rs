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
    pub(crate) fn draft_greedy(
        &self,
        text: &Qwen35TextModel,
        seed_hidden: &Tensor,
        depth: usize,
        continuation_positions: &[[usize; 3]],
        vocab_size: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Vec<u32>> {
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
            let token = greedy_argmax(&logits, vocab_size)?;
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
/// tokenizer vocabulary. Greedy draft selection mirrors the target's greedy
/// sampler: first-index-wins on ties, non-finite logits are a hard error.
pub(crate) fn greedy_argmax(logits: &Tensor, vocab_size: usize) -> Result<u32> {
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
            return Err(Error::InferenceError(
                "Qwen3.5 MTP draft produced non-finite logits".into(),
            ));
        }
        if value > best_value {
            best = index;
            best_value = value;
        }
    }
    Ok(best as u32)
}
