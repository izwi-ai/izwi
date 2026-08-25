//! Code Predictor for multi-codebook RVQ token generation.
//!
//! The code predictor generates the residual codebook tokens after the talker
//! has produced the first (semantic) codebook. It uses a smaller transformer
//! for efficient multi-token prediction.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};
use crate::error::{Error, Result};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::architectures::qwen3::tts::config::CodePredictorConfig;
use crate::models::architectures::qwen3::tts::rope::{
    build_rope_inv_freq, build_rope_window_full, qwen_rotate_half,
};
pub use crate::models::shared::attention::physical::PhysicalPagedKvCache as CodePredictorPhysicalCache;
use crate::models::shared::attention::physical::PreparedPhysicalPagedStep;
use crate::models::shared::weights::mlx;

/// The predictor starts each semantic frame from talker hidden state followed
/// by the selected semantic embedding.
pub const CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS: usize = 2;

/// Exact physical context occupied by one predictor frame.
pub const fn code_predictor_physical_context_tokens(acoustic_groups: usize) -> usize {
    CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS.saturating_add(acoustic_groups.saturating_sub(1))
}

/// Code Predictor model
pub struct CodePredictor {
    codec_embeddings: Vec<Embedding>,
    small_to_mtp_projection: Option<Linear>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_heads: Vec<Linear>,
    device: Device,
    cfg: CodePredictorConfig,
    num_code_groups: usize,
}

impl CodePredictor {
    /// Load the code predictor from VarBuilder
    pub fn load(cfg: CodePredictorConfig, vb: VarBuilder, num_code_groups: usize) -> Result<Self> {
        // Use text_hidden_size for codec embeddings if specified, otherwise hidden_size
        let codec_embed_dim = cfg.text_hidden_size.unwrap_or(cfg.hidden_size);

        // Load codec embeddings (one per codebook, but weights only have 15)
        // The model has embeddings 0-14 (15 total), not 16
        let num_codec_embeddings = num_code_groups.min(15);
        let mut codec_embeddings = Vec::with_capacity(num_codec_embeddings);
        for idx in 0..num_codec_embeddings {
            let embed = mlx::load_embedding(
                cfg.vocab_size,
                codec_embed_dim,
                vb.pp(format!("model.codec_embedding.{idx}")),
            )?;
            codec_embeddings.push(embed);
        }

        let small_to_mtp_projection = if codec_embed_dim != cfg.hidden_size {
            Some(mlx::load_linear(
                codec_embed_dim,
                cfg.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // Load output heads (one per code group, but weights only have 15)
        let num_lm_heads = num_code_groups.min(15);
        let mut lm_heads = Vec::with_capacity(num_lm_heads);
        for idx in 0..num_lm_heads {
            let head = mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp(format!("lm_head.{idx}")),
            )?;
            lm_heads.push(head);
        }

        Ok(Self {
            codec_embeddings,
            small_to_mtp_projection,
            layers,
            norm,
            lm_heads,
            device: vb.device().clone(),
            cfg,
            num_code_groups,
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get number of code groups
    pub fn num_code_groups(&self) -> usize {
        self.num_code_groups
    }

    /// Number of acoustic code groups predicted after the semantic codebook.
    pub fn num_acoustic_groups(&self) -> usize {
        self.codec_embeddings.len()
    }

    /// Exact physical KV capacity required by one predictor invocation.
    ///
    /// The two-token prefill produces the first acoustic code. Each remaining
    /// acoustic group appends one dependent token, so a standard 15-group
    /// predictor ends at cursor 16.
    pub fn physical_context_tokens_per_frame(&self) -> usize {
        code_predictor_physical_context_tokens(self.lm_heads.len())
    }

    /// Validate the fresh invocation workspace required for one semantic frame.
    pub fn validate_physical_workspace(&self, cache: &CodePredictorPhysicalCache) -> Result<()> {
        if self.lm_heads.len() != self.codec_embeddings.len() {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical predictor has {} heads for {} acoustic embeddings",
                self.lm_heads.len(),
                self.codec_embeddings.len()
            )));
        }
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor requires a fresh cursor-0 workspace, got {}",
                cache.context_len()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
        )?;
        let required_tokens = self.physical_context_tokens_per_frame();
        if cache.capacity_tokens() < required_tokens {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor workspace holds {} tokens, requires {required_tokens}",
                cache.capacity_tokens()
            )));
        }
        Ok(())
    }

    /// Forward a predictor input against scheduler-owned invocation pages.
    ///
    /// The supplied start position must be the workspace's exact cursor.
    pub fn forward_physical(
        &self,
        first_codebook: &Tensor,
        start_pos: usize,
        cache: &mut CodePredictorPhysicalCache,
    ) -> Result<Vec<Tensor>> {
        let mut x = self.codec_embeddings[0].forward(first_codebook)?;
        if let Some(proj) = &self.small_to_mtp_projection {
            x = proj.forward(&x)?;
        }

        let (x, prepared) = self.forward_physical_hidden_uncommitted(&x, start_pos, cache)?;
        let mut outputs = Vec::with_capacity(self.num_code_groups);
        for head in &self.lm_heads {
            outputs.push(head.forward(&x)?);
        }
        cache.commit_prepared(prepared)?;
        Ok(outputs)
    }

    /// Generate one frame's acoustic groups using a fresh physical workspace.
    ///
    /// The workspace is invocation-local: callers must provide cursor 0 for
    /// every semantic frame and discard it on error. Successful generation
    /// advances the cursor from 0 to [`Self::physical_context_tokens_per_frame`].
    pub fn generate_acoustic_codes_physical(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        cache: &mut CodePredictorPhysicalCache,
    ) -> Result<Vec<u32>> {
        self.validate_physical_workspace(cache)?;
        let initial_cursor = cache.context_len();
        let checkpoint = cache.logical_checkpoint();
        let result =
            self.generate_acoustic_codes_physical_inner(talker_hidden, semantic_embed, cache);
        if result.is_err() && cache.context_len() != initial_cursor {
            cache.restore_logical_checkpoint(checkpoint)?;
        }
        result
    }

    fn generate_acoustic_codes_physical_inner(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        cache: &mut CodePredictorPhysicalCache,
    ) -> Result<Vec<u32>> {
        let required_tokens = self.physical_context_tokens_per_frame();

        let (talker_batch, talker_tokens, talker_dim) = talker_hidden.dims3()?;
        let (semantic_batch, semantic_tokens, semantic_dim) = semantic_embed.dims3()?;
        if talker_batch != 1
            || talker_tokens != 1
            || semantic_batch != 1
            || semantic_tokens != 1
            || talker_dim != semantic_dim
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor expects matching [1,1,hidden] inputs, got {:?} and {:?}",
                talker_hidden.dims(),
                semantic_embed.dims()
            )));
        }

        let input = Tensor::cat(&[talker_hidden, semantic_embed], 1)?;
        let mut hidden = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(&input)?
        } else {
            input
        };
        let prefill_tokens = hidden.dim(1)?;
        if prefill_tokens != CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical predictor formed {prefill_tokens} prefill tokens, expected {CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS}"
            )));
        }
        let (next_hidden, prefill) = self.forward_physical_hidden_uncommitted(&hidden, 0, cache)?;
        hidden = next_hidden;

        let last_hidden = hidden.i((.., prefill_tokens - 1..prefill_tokens, ..))?;
        let num_acoustic = self.lm_heads.len();
        if num_acoustic == 0 {
            cache.commit_prepared(prefill)?;
            return Ok(Vec::new());
        }

        let first_logits = self.lm_heads[0].forward(&last_hidden)?;
        let mut prev_code = argmax_token(&first_logits.i((0, 0))?)?;
        cache.commit_prepared(prefill)?;

        let mut all_codes = Vec::with_capacity(num_acoustic);
        all_codes.push(prev_code);
        for group_idx in 1..num_acoustic {
            let mut step_hidden = self
                .codec_embedding_row(group_idx - 1, prev_code)?
                .unsqueeze(0)?;
            if let Some(proj) = &self.small_to_mtp_projection {
                step_hidden = proj.forward(&step_hidden)?;
            }

            let step_start = cache.context_len();
            let (next_hidden, step_prepared) =
                self.forward_physical_hidden_uncommitted(&step_hidden, step_start, cache)?;
            step_hidden = next_hidden;
            let logits = self.lm_heads[group_idx].forward(&step_hidden)?;
            prev_code = argmax_token(&logits.i((0, 0))?)?;
            cache.commit_prepared(step_prepared)?;
            all_codes.push(prev_code);
        }

        if cache.context_len() != required_tokens {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical predictor ended at cursor {}, expected {required_tokens}",
                cache.context_len()
            )));
        }
        Ok(all_codes)
    }

    /// Generate acoustic codebooks for compatible semantic-frame rows.
    ///
    /// The predictor is an independently schedulable stage: every row begins at
    /// invocation cursor zero, executes the same two-token prefill, then advances
    /// through identical codebook positions. Rows with different cursor/shape or
    /// codebook geometry are rejected and must be split by the scheduler. One
    /// row deliberately falls back to the scalar implementation.
    pub fn generate_acoustic_codes_physical_batch(
        &self,
        talker_hidden: &Tensor,
        semantic_embeds: &Tensor,
        caches: &mut [&mut CodePredictorPhysicalCache],
    ) -> Result<Vec<Vec<u32>>> {
        let (batch_size, talker_tokens, talker_dim) = talker_hidden.dims3()?;
        let (semantic_batch, semantic_tokens, semantic_dim) = semantic_embeds.dims3()?;
        if batch_size == 0
            || talker_tokens != 1
            || semantic_batch != batch_size
            || semantic_tokens != 1
            || talker_dim != semantic_dim
            || talker_dim != self.cfg.hidden_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS predictor batch expects matching [batch,1,{}] inputs and caches, got {:?}, {:?}, and {} caches",
                self.cfg.hidden_size,
                talker_hidden.dims(),
                semantic_embeds.dims(),
                caches.len()
            )));
        }
        if batch_size == 1 {
            return self
                .generate_acoustic_codes_physical(talker_hidden, semantic_embeds, caches[0])
                .map(|codes| vec![codes]);
        }
        for cache in caches.iter() {
            self.validate_physical_workspace(cache)?;
        }
        let first = &*caches[0];
        if caches
            .iter()
            .any(|cache| !Arc::ptr_eq(cache.arena(), first.arena()))
        {
            return Err(Error::InvalidInput(
                "Qwen3-TTS predictor batch rows must share one invocation arena".into(),
            ));
        }
        let initial_checkpoints = caches
            .iter()
            .map(|cache| cache.logical_checkpoint())
            .collect::<Vec<_>>();
        let initial_cursors = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        let result = (|| {
            let input = Tensor::cat(&[talker_hidden, semantic_embeds], 1)?;
            let hidden = if let Some(proj) = &self.small_to_mtp_projection {
                proj.forward(&input)?
            } else {
                input
            };
            if hidden.dim(1)? != CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS {
                return Err(Error::InferenceError(
                    "Qwen3-TTS predictor batch formed an invalid prefill width".into(),
                ));
            }
            let hidden = self.forward_physical_hidden_batch_committed(&hidden, 0, caches)?;
            let last_hidden = hidden.i((.., CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS - 1, ..))?;
            let num_acoustic = self.lm_heads.len();
            if num_acoustic == 0 {
                return Ok(vec![Vec::new(); batch_size]);
            }

            let first_logits = self.lm_heads[0].forward(&last_hidden)?;
            let mut previous_codes = argmax_tokens(&first_logits, batch_size)?;
            let mut all_codes = previous_codes
                .iter()
                .map(|code| vec![*code])
                .collect::<Vec<_>>();
            for group_idx in 1..num_acoustic {
                if caches
                    .iter()
                    .any(|cache| cache.context_len() != group_idx + 1)
                {
                    return Err(Error::InvalidInput(format!(
                        "Qwen3-TTS predictor batch rows diverged before codebook {group_idx}"
                    )));
                }
                let ids = Tensor::from_vec(previous_codes.clone(), (batch_size, 1), &self.device)?;
                let mut step_hidden = self.codec_embeddings[group_idx - 1].forward(&ids)?;
                if let Some(proj) = &self.small_to_mtp_projection {
                    step_hidden = proj.forward(&step_hidden)?;
                }
                let step_hidden = self.forward_physical_hidden_batch_committed(
                    &step_hidden,
                    group_idx + 1,
                    caches,
                )?;
                let logits = self.lm_heads[group_idx].forward(&step_hidden)?;
                previous_codes = argmax_tokens(&logits, batch_size)?;
                for (row, code) in previous_codes.iter().enumerate() {
                    all_codes[row].push(*code);
                }
            }
            let required_tokens = self.physical_context_tokens_per_frame();
            if caches
                .iter()
                .any(|cache| cache.context_len() != required_tokens)
            {
                return Err(Error::InferenceError(format!(
                    "Qwen3-TTS predictor batch did not finish at cursor {required_tokens}"
                )));
            }
            Ok(all_codes)
        })();
        if let Err(error) = result {
            let mut rollback_error = None;
            for (row, cache) in caches.iter_mut().enumerate() {
                if cache.context_len() != initial_cursors[row] {
                    if let Err(rollback) =
                        cache.restore_logical_checkpoint(initial_checkpoints[row].clone())
                    {
                        rollback_error.get_or_insert(rollback);
                    }
                }
            }
            return if let Some(rollback) = rollback_error {
                Err(Error::InferenceError(format!(
                    "Qwen3-TTS predictor batch failed: {error}; rollback also failed: {rollback}"
                )))
            } else {
                Err(error)
            };
        }
        result
    }

    fn forward_physical_hidden_uncommitted(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &mut CodePredictorPhysicalCache,
    ) -> Result<(Tensor, PreparedPhysicalPagedStep)> {
        let (batch_size, sequence_len, hidden_size) = x.dims3()?;
        if batch_size != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor expects [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                x.dims()
            )));
        }
        if start_pos != cache.context_len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor starts at {start_pos}, expected invocation cursor {}",
                cache.context_len()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
        )?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let execution = (|| {
            let mut hidden = x.clone();
            for (idx, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_physical(&hidden, start_pos, cache, &mut prepared, idx)?;
            }
            self.norm.forward(&hidden).map_err(Error::from)
        })();
        match execution {
            Ok(hidden) => Ok((hidden, prepared)),
            Err(error) => match cache.abort_prepared(prepared) {
                Ok(()) => Err(error),
                Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3-TTS predictor position failed: {error}; write-fence drain also failed: {drain}"
                ))),
            },
        }
    }

    /// Execute one exact shared predictor position and commit all participating
    /// invocation rows atomically. `sequence_len` may be two for predictor
    /// prefill or one for a dependent codebook position.
    fn forward_physical_hidden_batch_committed(
        &self,
        input: &Tensor,
        start_pos: usize,
        caches: &mut [&mut CodePredictorPhysicalCache],
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, hidden_size) = input.dims3()?;
        if batch_size < 2
            || !(sequence_len == 1 || sequence_len == CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS)
            || hidden_size != self.cfg.hidden_size
            || caches.len() != batch_size
            || caches.iter().any(|cache| cache.context_len() != start_pos)
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS predictor batch position {start_pos} has incompatible shape {:?} or row cursors",
                input.dims()
            )));
        }
        for cache in caches.iter() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim(),
            )?;
            cache.slots_for_append(start_pos, sequence_len)?;
        }
        let first = &*caches[0];
        if caches
            .iter()
            .any(|cache| !Arc::ptr_eq(cache.arena(), first.arena()))
        {
            return Err(Error::InvalidInput(
                "Qwen3-TTS predictor batch rows must share one arena".into(),
            ));
        }
        let combined_slots = caches
            .iter()
            .map(|cache| cache.slots_for_append(start_pos, sequence_len))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let lowered = first.arena().lower_slots(&combined_slots)?;
        let decode = if sequence_len == 1 {
            Some(KvDecodeBatchMetadata {
                sequences: caches
                    .iter()
                    .map(|cache| cache.sequence_table(start_pos + 1))
                    .collect::<Result<Vec<_>>>()?,
            })
        } else {
            None
        };
        let prefill = if sequence_len > 1 {
            caches
                .iter()
                .enumerate()
                .map(|(row, cache)| {
                    let table = cache.sequence_table(start_pos + sequence_len)?;
                    Ok(PagedKvPrefillRow {
                        blocks: table.blocks,
                        first_page_offset: table.first_page_offset,
                        query_start: u32::try_from(row * sequence_len).map_err(|_| {
                            Error::InvalidInput(
                                "Qwen3-TTS predictor batch query offset exceeds u32".into(),
                            )
                        })?,
                        query_len: u32::try_from(sequence_len).map_err(|_| {
                            Error::InvalidInput(
                                "Qwen3-TTS predictor batch query width exceeds u32".into(),
                            )
                        })?,
                        context_len: table.context_len,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let checkpoints = caches
            .iter()
            .map(|cache| cache.logical_checkpoint())
            .collect::<Vec<_>>();
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;
        let execution = (|| -> Result<Tensor> {
            let mut hidden = input.clone();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let cache_refs = caches
                    .iter()
                    .map(|cache| &**cache)
                    .collect::<Vec<&CodePredictorPhysicalCache>>();
                hidden = layer.forward_physical_batch(
                    &hidden,
                    start_pos,
                    &cache_refs,
                    lowered.as_ref(),
                    decode.as_ref(),
                    &prefill,
                    &mut completions,
                    layer_idx,
                )?;
            }
            self.norm.forward(&hidden).map_err(Error::from)
        })();
        let hidden = match execution {
            Ok(hidden) => hidden,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                        "Qwen3-TTS predictor batch position failed: {error}; write-fence drain also failed: {drain}"
                    ))),
                };
            }
        };
        let completion = completions.seal()?;
        for (committed, row) in (0..batch_size).enumerate() {
            let row_completion = completion
                .project_slot_range(row * sequence_len, sequence_len)
                .map(Arc::new);
            if let Err(error) = row_completion.and_then(|completion| {
                caches[row].commit_shared_completion(start_pos, sequence_len, completion)
            }) {
                let mut rollback_error = None;
                for rollback_row in 0..committed {
                    if let Err(rollback) = caches[rollback_row]
                        .restore_logical_checkpoint(checkpoints[rollback_row].clone())
                    {
                        rollback_error.get_or_insert(rollback);
                    }
                }
                return if let Some(rollback) = rollback_error {
                    Err(Error::InferenceError(format!(
                        "Qwen3-TTS predictor batch commit failed: {error}; rollback also failed: {rollback}"
                    )))
                } else {
                    Err(error)
                };
            }
        }
        Ok(hidden)
    }

    /// Sum acoustic embeddings for the 15 generated acoustic codes.
    /// Returned tensor shape is [1, 1, codec_embed_dim].
    pub fn get_acoustic_embeddings_sum(&self, acoustic_codes: &[u32]) -> Result<Tensor> {
        if acoustic_codes.len() != self.codec_embeddings.len() {
            return Err(Error::InvalidInput(format!(
                "Expected {} acoustic codes, got {}",
                self.codec_embeddings.len(),
                acoustic_codes.len()
            )));
        }

        let mut sum = self
            .codec_embedding_row(0, acoustic_codes[0])?
            .unsqueeze(0)?;

        for (group_idx, code) in acoustic_codes.iter().enumerate().skip(1) {
            let embed = self.codec_embedding_row(group_idx, *code)?.unsqueeze(0)?;
            sum = sum.broadcast_add(&embed)?;
        }

        Ok(sum)
    }

    fn codec_embedding_row(&self, group_idx: usize, code: u32) -> Result<Tensor> {
        if self.device.is_cuda() {
            self.codec_embeddings[group_idx]
                .embeddings()
                .i(code as usize)?
                .unsqueeze(0)
                .map_err(Error::from)
        } else {
            let code_tensor = Tensor::from_vec(vec![code], (1,), &self.device)?;
            self.codec_embeddings[group_idx]
                .forward(&code_tensor)
                .map_err(Error::from)
        }
    }
}

/// Transformer layer for code predictor
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Attention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward_physical(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &CodePredictorPhysicalCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_physical(&normed, start_pos, cache, prepared, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_physical_batch(
        &self,
        x: &Tensor,
        start_pos: usize,
        caches: &[&CodePredictorPhysicalCache],
        slots: &dyn KvSlotMap,
        decode: Option<&KvDecodeBatchMetadata>,
        prefill: &[PagedKvPrefillRow],
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_physical_batch(
            &normed,
            start_pos,
            caches,
            slots,
            decode,
            prefill,
            completions,
            layer_idx,
        )?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Multi-head attention for code predictor
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_inv_freq: Vec<f32>,
}

impl Attention {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();

        let q_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = mlx::load_linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_inv_freq: build_rope_inv_freq(head_dim, cfg.rope_theta),
        })
    }

    fn apply_qk_norm(
        &self,
        x: Tensor,
        heads: usize,
        seq_len: usize,
        norm: &RmsNorm,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let reshaped = x.reshape((bsz * seq_len * heads, self.head_dim))?;
        let normed = norm.forward(&reshaped)?;
        normed
            .reshape((bsz, seq_len, heads, self.head_dim))
            .map_err(Error::from)
    }

    fn apply_rope(&self, x: Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let half_dim = self.head_dim / 2;

        let (cos, sin) = build_rope_window_full(
            seq_len,
            start_pos,
            &self.rope_inv_freq,
            x.device(),
            x.dtype(),
        )?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let rotated = qwen_rotate_half(&x, half_dim)?;

        let out = x.broadcast_mul(&cos)?;
        out.broadcast_add(&rotated.broadcast_mul(&sin)?)
            .map_err(Error::from)
    }

    /// Direct grouped-query attention over scheduler-owned predictor pages.
    fn forward_physical(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &CodePredictorPhysicalCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        if bsz != 1 || seq_len == 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor attention expects [1,sequence,hidden], got {:?}",
                x.dims()
            )));
        }

        let mut q =
            self.q_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let mut k =
            self.k_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;

        q = self.apply_qk_norm(q, self.num_heads, seq_len, &self.q_norm)?;
        k = self.apply_qk_norm(k, self.num_kv_heads, seq_len, &self.k_norm)?;
        q = self.apply_rope(q, start_pos)?;
        k = self.apply_rope(k, start_pos)?;

        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .contiguous()?;
        let k = k
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let v = v
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let out = cache.write_and_attend(
            layer_idx,
            prepared,
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }

    /// Execute one shared predictor position. All rows have the same sequence
    /// width and RoPE position; the initial two-token call uses paged prefill,
    /// while dependent codebook calls use paged decode.
    #[allow(clippy::too_many_arguments)]
    fn forward_physical_batch(
        &self,
        x: &Tensor,
        start_pos: usize,
        caches: &[&CodePredictorPhysicalCache],
        slots: &dyn KvSlotMap,
        decode: Option<&KvDecodeBatchMetadata>,
        prefill: &[PagedKvPrefillRow],
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, _) = x.dims3()?;
        if batch_size < 2
            || !(sequence_len == 1 || sequence_len == CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS)
            || caches.len() != batch_size
            || decode.is_some() != (sequence_len == 1)
            || prefill.is_empty() == (sequence_len > 1)
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor batch received incompatible position shape {:?}",
                x.dims()
            )));
        }
        let first = caches[0];
        if caches.iter().any(|cache| {
            cache.context_len() != start_pos
                || !Arc::ptr_eq(cache.arena(), first.arena())
                || cache.layer_binding(layer_idx).ok() != first.layer_binding(layer_idx).ok()
        }) {
            return Err(Error::InvalidInput(
                "Qwen3-TTS physical predictor batch rows must share a position, arena, and layer binding"
                    .into(),
            ));
        }

        let mut q = self.q_proj.forward(x)?.reshape((
            batch_size,
            sequence_len,
            self.num_heads,
            self.head_dim,
        ))?;
        let mut k = self.k_proj.forward(x)?.reshape((
            batch_size,
            sequence_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = self.v_proj.forward(x)?.reshape((
            batch_size,
            sequence_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        q = self.apply_qk_norm(q, self.num_heads, sequence_len, &self.q_norm)?;
        k = self.apply_qk_norm(k, self.num_kv_heads, sequence_len, &self.k_norm)?;
        q = self.apply_rope(q, start_pos)?;
        k = self.apply_rope(k, start_pos)?;
        let total_tokens = batch_size.checked_mul(sequence_len).ok_or_else(|| {
            Error::InvalidInput("Qwen3-TTS predictor batch token count overflow".into())
        })?;
        let queries = q
            .reshape((total_tokens, self.num_heads, self.head_dim))?
            .contiguous()?;
        let keys = k
            .reshape((total_tokens, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let values = v
            .reshape((total_tokens, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        if slots.arena_id() != first.arena().id() || slots.len() != total_tokens {
            return Err(Error::InvalidInput(
                "Qwen3-TTS predictor batch received an incompatible slot map".into(),
            ));
        }
        let binding = first.layer_binding(layer_idx)?;
        let completion = first.arena().write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let (out, completion) = submit_ordered_after_write(completion, || {
            if let Some(decode) = decode {
                first.arena().paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries: &queries,
                        batch: decode,
                        softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                        softcap: None,
                    },
                )
            } else {
                first.arena().paged_prefill(
                    binding,
                    PagedKvPrefillArgs {
                        queries: &queries,
                        rows: prefill,
                        softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                        softcap: None,
                        window_tokens: None,
                    },
                )
            }
        })?;
        completions.collect(completion)?;
        self.o_proj
            .forward(&out.reshape((batch_size, sequence_len, self.num_heads * self.head_dim))?)
            .map_err(Error::from)
    }
}

/// SwiGLU MLP
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            mlx::load_linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

fn argmax_token(logits: &Tensor) -> Result<u32> {
    if !logits.device().is_cuda() {
        return argmax_token_reference(logits);
    }

    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn argmax_token_reference(logits: &Tensor) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS predictor logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS predictor logits rank: {rank}"
            )))
        }
    };
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in values.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    Ok(max_idx as u32)
}

fn argmax_tokens(logits: &Tensor, expected_batch: usize) -> Result<Vec<u32>> {
    let logits = match logits.rank() {
        3 if logits.dim(1)? == 1 => logits.i((.., 0, ..))?,
        2 => logits.clone(),
        _ => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS predictor batch logits shape: {:?}",
                logits.dims()
            )))
        }
    };
    if logits.dim(0)? != expected_batch {
        return Err(Error::InferenceError(format!(
            "Qwen3-TTS predictor batch produced {} rows, expected {expected_batch}",
            logits.dim(0)?
        )));
    }
    if logits.device().is_cuda() {
        return logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()
            .map_err(Error::from);
    }
    (0..expected_batch)
        .map(|row| argmax_token_reference(&logits.i(row)?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};

    fn test_linear(output: usize, input: usize, offset: usize, device: &Device) -> Linear {
        let values = (0..output * input)
            .map(|index| {
                let value = (index.saturating_mul(7).saturating_add(offset)) % 29;
                (value as f32 - 14.0) / 32.0
            })
            .collect::<Vec<_>>();
        Linear::new(
            Tensor::from_vec(values, (output, input), device).unwrap(),
            None,
        )
    }

    fn tiny_predictor(device: &Device) -> CodePredictor {
        let cfg = CodePredictorConfig {
            model_type: "test-predictor".into(),
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 32,
            vocab_size: 8,
            num_code_groups: 3,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            hidden_act: "silu".into(),
            use_cache: true,
            layer_types: Vec::new(),
            text_hidden_size: None,
        };
        let attention = Attention {
            q_proj: test_linear(4, 4, 1, device),
            k_proj: test_linear(2, 4, 2, device),
            v_proj: test_linear(2, 4, 3, device),
            o_proj: test_linear(4, 4, 4, device),
            q_norm: RmsNorm::new(Tensor::ones(2, DType::F32, device).unwrap(), 1e-5),
            k_norm: RmsNorm::new(Tensor::ones(2, DType::F32, device).unwrap(), 1e-5),
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_inv_freq: build_rope_inv_freq(2, cfg.rope_theta),
        };
        let layer = Layer {
            input_layernorm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            self_attn: attention,
            post_attention_layernorm: RmsNorm::new(
                Tensor::ones(4, DType::F32, device).unwrap(),
                1e-5,
            ),
            mlp: Mlp {
                gate_proj: test_linear(8, 4, 5, device),
                up_proj: test_linear(8, 4, 6, device),
                down_proj: test_linear(4, 8, 7, device),
            },
        };
        let codec_embeddings = (0..3)
            .map(|group| {
                let values = (0..32)
                    .map(|index| {
                        let value = (index * 5 + group * 3) % 19;
                        (value as f32 - 9.0) / 16.0
                    })
                    .collect::<Vec<_>>();
                Embedding::new(Tensor::from_vec(values, (8, 4), device).unwrap(), 4)
            })
            .collect();
        CodePredictor {
            codec_embeddings,
            small_to_mtp_projection: None,
            layers: vec![layer],
            norm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            lm_heads: (0..3)
                .map(|head| test_linear(8, 4, 8 + head, device))
                .collect(),
            device: device.clone(),
            cfg,
            num_code_groups: 3,
        }
    }

    fn test_arena(instance: u64) -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = CpuKvArena::new(KvArenaConfig {
            id: KvArenaId {
                model_instance: ModelInstanceId::new(instance),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            page_tokens: 2,
            capacity_pages: 24,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })
        .unwrap();
        (Arc::new(arena), vec![binding])
    }

    fn test_cache(
        arena: Arc<dyn KvArena>,
        bindings: &[KvLayerBinding],
        first_page: u32,
        context_len: usize,
    ) -> CodePredictorPhysicalCache {
        let blocks = (first_page..first_page + 6)
            .map(|index| CacheBlockRef {
                arena: arena.id(),
                group: arena.config().group,
                index,
                slot_generation: 1,
            })
            .collect();
        CodePredictorPhysicalCache::new(arena, bindings.to_vec(), blocks, context_len).unwrap()
    }

    fn frame_input(seed: usize, device: &Device) -> Tensor {
        let values = (0..4)
            .map(|index| {
                let value = (index * 11 + seed) % 23;
                (value as f32 - 11.0) / 16.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, 1, 4), device).unwrap()
    }

    #[test]
    fn non_cuda_predictor_argmax_uses_reference_ordering() {
        let logits = Tensor::new(vec![0.0f32, 4.0, 4.0, 3.0], &Device::Cpu).unwrap();

        assert_eq!(argmax_token(&logits).unwrap(), 1);
    }

    #[test]
    fn predictor_argmax_reference_accepts_single_row_logits() {
        let logits = Tensor::new(&[[0.0f32, 1.0, 7.0, 3.0]], &Device::Cpu).unwrap();

        assert_eq!(argmax_token(&logits).unwrap(), 2);
    }

    #[test]
    fn physical_predictor_cursor_matches_prefill_and_dependent_groups() {
        assert_eq!(code_predictor_physical_context_tokens(0), 2);
        assert_eq!(code_predictor_physical_context_tokens(1), 2);
        assert_eq!(code_predictor_physical_context_tokens(15), 16);
    }

    #[test]
    fn predictor_batch_matches_scalar_codebooks_and_isolates_receipts() {
        let device = Device::Cpu;
        let model = tiny_predictor(&device);
        let (scalar_arena, scalar_bindings) = test_arena(711);
        let (batch_arena, batch_bindings) = test_arena(712);
        let mut scalar_a = test_cache(scalar_arena.clone(), &scalar_bindings, 0, 0);
        let mut scalar_b = test_cache(scalar_arena, &scalar_bindings, 6, 0);
        let mut batch_a = test_cache(batch_arena.clone(), &batch_bindings, 0, 0);
        let mut batch_b = test_cache(batch_arena, &batch_bindings, 6, 0);
        let talker_a = frame_input(3, &device);
        let talker_b = frame_input(5, &device);
        let semantic_a = frame_input(7, &device);
        let semantic_b = frame_input(9, &device);
        let scalar_codes_a = model
            .generate_acoustic_codes_physical(&talker_a, &semantic_a, &mut scalar_a)
            .unwrap();
        let scalar_codes_b = model
            .generate_acoustic_codes_physical(&talker_b, &semantic_b, &mut scalar_b)
            .unwrap();
        let talker = Tensor::cat(&[&talker_a, &talker_b], 0).unwrap();
        let semantic = Tensor::cat(&[&semantic_a, &semantic_b], 0).unwrap();
        let mut caches = [&mut batch_a, &mut batch_b];
        let batch_codes = model
            .generate_acoustic_codes_physical_batch(&talker, &semantic, &mut caches)
            .unwrap();

        assert_eq!(batch_codes, vec![scalar_codes_a, scalar_codes_b]);
        assert_eq!((batch_a.context_len(), batch_b.context_len()), (4, 4));
        let completions_a = batch_a.take_completed_writes();
        let completions_b = batch_b.take_completed_writes();
        assert_eq!(completions_a.len(), 3);
        assert_eq!(completions_b.len(), 3);
        assert!(completions_a
            .iter()
            .zip(&completions_b)
            .all(|(left, right)| !Arc::ptr_eq(left, right)
                && left
                    .slots()
                    .iter()
                    .all(|slot| !right.slots().contains(slot))));
    }

    #[test]
    fn predictor_batch_rejects_position_mismatch_without_advancing_any_row() {
        let device = Device::Cpu;
        let model = tiny_predictor(&device);
        let (arena, bindings) = test_arena(713);
        let mut fresh = test_cache(arena.clone(), &bindings, 0, 0);
        let mut incompatible = test_cache(arena, &bindings, 6, 1);
        let talker =
            Tensor::cat(&[&frame_input(11, &device), &frame_input(13, &device)], 0).unwrap();
        let semantic =
            Tensor::cat(&[&frame_input(15, &device), &frame_input(17, &device)], 0).unwrap();
        let mut caches = [&mut fresh, &mut incompatible];

        assert!(model
            .generate_acoustic_codes_physical_batch(&talker, &semantic, &mut caches)
            .is_err());
        assert_eq!((fresh.context_len(), incompatible.context_len()), (0, 1));
        assert!(fresh.take_completed_writes().is_empty());
        assert!(incompatible.take_completed_writes().is_empty());
    }

    #[test]
    fn one_row_predictor_batch_uses_scalar_fallback_exactly() {
        let device = Device::Cpu;
        let model = tiny_predictor(&device);
        let (scalar_arena, scalar_bindings) = test_arena(714);
        let (batch_arena, batch_bindings) = test_arena(715);
        let mut scalar = test_cache(scalar_arena, &scalar_bindings, 0, 0);
        let mut fallback = test_cache(batch_arena, &batch_bindings, 0, 0);
        let talker = frame_input(19, &device);
        let semantic = frame_input(21, &device);
        let expected = model
            .generate_acoustic_codes_physical(&talker, &semantic, &mut scalar)
            .unwrap();
        let mut caches = [&mut fallback];
        let actual = model
            .generate_acoustic_codes_physical_batch(&talker, &semantic, &mut caches)
            .unwrap();

        assert_eq!(actual, vec![expected]);
        assert_eq!(fallback.context_len(), scalar.context_len());
    }
}
