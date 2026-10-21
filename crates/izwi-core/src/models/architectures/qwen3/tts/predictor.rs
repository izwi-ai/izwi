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
use crate::kernels::f16_weight_linear::{projection_var_builder, ProjectionLinear};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::architectures::qwen3::tts::config::CodePredictorConfig;
use crate::models::architectures::qwen3::tts::rope::{
    build_rope_inv_freq, build_rope_window_full, qwen_rotate_half,
};
pub use crate::models::shared::attention::physical::PhysicalPagedKvCache as CodePredictorPhysicalCache;
use crate::models::shared::attention::physical::PreparedPhysicalPagedStep;
use crate::models::shared::sampling::{
    bounded_device_sampling_candidates, device_candidates_cover_top_p, resident_top_k_sample,
    sample_device_candidates,
};
use crate::models::shared::weights::mlx;

/// The predictor starts each semantic frame from talker hidden state followed
/// by the selected semantic embedding.
pub const CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS: usize = 2;

/// Exact physical context occupied by one predictor frame.
pub const fn code_predictor_physical_context_tokens(acoustic_groups: usize) -> usize {
    CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS.saturating_add(acoustic_groups.saturating_sub(1))
}

/// Sampling for the residual codebooks: the reference's `subtalker_*`
/// generation settings (`generation_config.json`: do_sample true,
/// temperature 0.9, top_k 50, top_p 1.0), passed to
/// `code_predictor.generate`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubtalkerSampling {
    pub do_sample: bool,
    pub temperature: f32,
    /// 0 disables top-k.
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SubtalkerSampling {
    fn default() -> Self {
        Self {
            do_sample: true,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
        }
    }
}

impl SubtalkerSampling {
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            ..Self::default()
        }
    }

    pub fn is_greedy(&self) -> bool {
        !self.do_sample || self.temperature <= 1e-5
    }
}

/// How one row picks each residual code: argmax, or a draw from the
/// sub-talker distribution using the row's own uniform source.
pub enum CodeSelection<'a> {
    Greedy,
    Sample {
        params: SubtalkerSampling,
        draw: &'a mut dyn FnMut() -> f32,
    },
}

impl CodeSelection<'_> {
    fn is_greedy(&self) -> bool {
        match self {
            Self::Greedy => true,
            Self::Sample { params, .. } => params.is_greedy(),
        }
    }

    /// Pick one code from a single row of logits as a `[1]` U32 tensor on
    /// the logits' device. Greedy codes and top-k draws with top-p 1 (the
    /// sub-talker default) never leave the device; other settings sample on
    /// the host and upload the code.
    fn select(&mut self, logits: &Tensor) -> Result<Tensor> {
        match self {
            Self::Sample { params, draw } if !params.is_greedy() => {
                let row = logits_row(logits)?;
                if params.top_p >= 1.0 {
                    if let Some(code) = resident_top_k_sample(
                        &row,
                        row.dim(0)?,
                        params.top_k,
                        params.temperature,
                        &mut **draw,
                    )? {
                        return Ok(code);
                    }
                }
                let code = sample_code(&row, params, &mut **draw)?;
                Tensor::from_vec(vec![code], 1, row.device()).map_err(Error::from)
            }
            _ => argmax_device(&logits_row(logits)?),
        }
    }
}

/// Code Predictor model
pub struct CodePredictor {
    codec_embeddings: Vec<CodecEmbedding>,
    small_to_mtp_projection: Option<Linear>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_heads: Vec<ProjectionLinear>,
    device: Device,
    cfg: CodePredictorConfig,
    num_code_groups: usize,
}

/// One codebook's embedding table: dense, or (weight-only F16) stored in F16
/// on the device with gathered rows widened to F32, exactly the F32 table's
/// values for a BF16 checkpoint.
#[derive(Debug, Clone)]
enum CodecEmbedding {
    Dense(Embedding),
    F16(Tensor),
}

impl CodecEmbedding {
    /// `embedding` was read on the host when `f16` is set.
    fn new(embedding: Embedding, f16: Option<&Device>) -> Result<Self> {
        let Some(device) = f16 else {
            return Ok(Self::Dense(embedding));
        };
        Ok(Self::F16(
            embedding
                .embeddings()
                .to_dtype(DType::F16)?
                .to_device(device)?,
        ))
    }

    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(embedding) => embedding.forward(ids),
            Self::F16(table) => {
                let mut dims = ids.dims().to_vec();
                dims.push(table.dim(1)?);
                table
                    .index_select(&ids.flatten_all()?, 0)?
                    .to_dtype(DType::F32)?
                    .reshape(dims)
            }
        }
    }

    /// Row `index` as F32, `[hidden]`.
    fn row(&self, index: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(embedding) => embedding.embeddings().i(index),
            Self::F16(table) => table.i(index)?.to_dtype(DType::F32),
        }
    }
}

impl CodePredictor {
    /// Load the code predictor. With `f16`, its attention and MLP
    /// projections, its logits heads and its codec embedding tables are
    /// stored in F16 on that device (activations and norms stay F32).
    pub fn load(
        cfg: CodePredictorConfig,
        vb: VarBuilder,
        num_code_groups: usize,
        f16: Option<&Device>,
    ) -> Result<Self> {
        let codec_embed_dim = cfg.input_hidden_size();

        // Load codec embeddings (one per codebook, but weights only have 15)
        // The model has embeddings 0-14 (15 total), not 16
        let num_codec_embeddings = num_code_groups.min(15);
        let mut codec_embeddings = Vec::with_capacity(num_codec_embeddings);
        for idx in 0..num_codec_embeddings {
            let embed = mlx::load_embedding(
                cfg.vocab_size,
                codec_embed_dim,
                projection_var_builder(&vb, f16).pp(format!("model.codec_embedding.{idx}")),
            )?;
            codec_embeddings.push(CodecEmbedding::new(embed, f16)?);
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
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")), f16)?;
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
                projection_var_builder(&vb, f16).pp(format!("lm_head.{idx}")),
            )?;
            lm_heads.push(ProjectionLinear::new(head, f16)?);
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

    /// Whether every layer's attention and MLP projections are stored as
    /// weight-only F16.
    pub fn projections_f16(&self) -> bool {
        !self.layers.is_empty()
            && self.layers.iter().all(|layer| {
                [
                    &layer.self_attn.q_proj,
                    &layer.self_attn.k_proj,
                    &layer.self_attn.v_proj,
                    &layer.self_attn.o_proj,
                    &layer.mlp.gate_proj,
                    &layer.mlp.up_proj,
                    &layer.mlp.down_proj,
                ]
                .iter()
                .all(|projection| projection.as_f16().is_some())
            })
    }

    pub fn config(&self) -> &CodePredictorConfig {
        &self.cfg
    }

    /// Checks the first layer's weight-only F16 MLP gate projection against
    /// its original F32 weight re-read from `vb` (the predictor's builder).
    /// Returns the relative L2; a dense predictor checks nothing.
    pub(crate) fn check_f16_conversion(&self, vb: &VarBuilder) -> Result<f64> {
        let Some(gate) = self.layers.first().and_then(|l| l.mlp.gate_proj.as_f16()) else {
            return Ok(0.0);
        };
        let host = projection_var_builder(vb, Some(&Device::Cpu));
        let original = mlx::load_linear_no_bias(
            self.cfg.hidden_size,
            self.cfg.intermediate_size,
            host.pp("model.layers.0.mlp.gate_proj"),
        )?;
        Ok(gate.check_against(original.weight(), None, 8)?)
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
        selection: &mut CodeSelection<'_>,
    ) -> Result<Vec<u32>> {
        self.validate_physical_workspace(cache)?;
        let initial_cursor = cache.context_len();
        let checkpoint = cache.logical_checkpoint();
        let result = self.generate_acoustic_codes_physical_inner(
            talker_hidden,
            semantic_embed,
            cache,
            selection,
        );
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
        selection: &mut CodeSelection<'_>,
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
        let mut prev_code = selection.select(&first_logits.i((0, 0))?)?;
        cache.commit_prepared(prefill)?;

        // Codes stay on the device and feed the next embedding directly; the
        // frame's codes are read back once at the end.
        let mut all_codes = Vec::with_capacity(num_acoustic);
        all_codes.push(prev_code.clone());
        for group_idx in 1..num_acoustic {
            let mut step_hidden = self.codec_embeddings[group_idx - 1]
                .forward(&prev_code)?
                .unsqueeze(0)?;
            if let Some(proj) = &self.small_to_mtp_projection {
                step_hidden = proj.forward(&step_hidden)?;
            }

            let step_start = cache.context_len();
            let (next_hidden, step_prepared) =
                self.forward_physical_hidden_uncommitted(&step_hidden, step_start, cache)?;
            step_hidden = next_hidden;
            let logits = self.lm_heads[group_idx].forward(&step_hidden)?;
            prev_code = selection.select(&logits.i((0, 0))?)?;
            cache.commit_prepared(step_prepared)?;
            all_codes.push(prev_code.clone());
        }

        if cache.context_len() != required_tokens {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical predictor ended at cursor {}, expected {required_tokens}",
                cache.context_len()
            )));
        }
        super::record_qwen3_tts_host_readback();
        Tensor::cat(&all_codes, 0)?
            .to_vec1::<u32>()
            .map_err(Error::from)
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
        selections: &mut [CodeSelection<'_>],
    ) -> Result<Vec<Vec<u32>>> {
        // Inputs arrive at the talker's width; `small_to_mtp_projection`
        // narrows them to the predictor's `hidden_size` below.
        let input_hidden_size = self.cfg.input_hidden_size();
        let (batch_size, talker_tokens, talker_dim) = talker_hidden.dims3()?;
        let (semantic_batch, semantic_tokens, semantic_dim) = semantic_embeds.dims3()?;
        if batch_size == 0
            || talker_tokens != 1
            || semantic_batch != batch_size
            || semantic_tokens != 1
            || talker_dim != semantic_dim
            || talker_dim != input_hidden_size
            || caches.len() != batch_size
            || selections.len() != batch_size
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS predictor batch expects matching [batch,1,{input_hidden_size}] inputs and caches, got {:?}, {:?}, and {} caches",
                talker_hidden.dims(),
                semantic_embeds.dims(),
                caches.len()
            )));
        }
        if batch_size == 1 {
            return self
                .generate_acoustic_codes_physical(
                    talker_hidden,
                    semantic_embeds,
                    caches[0],
                    &mut selections[0],
                )
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
            // Selecting the last token leaves a gap between batch rows. BLAS
            // providers require compact rows at the vocabulary projection.
            let last_hidden = hidden
                .i((.., CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS - 1, ..))?
                .contiguous()?;
            let num_acoustic = self.lm_heads.len();
            if num_acoustic == 0 {
                return Ok(vec![Vec::new(); batch_size]);
            }

            let first_logits = self.lm_heads[0].forward(&last_hidden)?;
            let mut previous_codes = select_codes(&first_logits, batch_size, selections)?;
            let mut steps = Vec::with_capacity(num_acoustic);
            steps.push(previous_codes.clone());
            for group_idx in 1..num_acoustic {
                if caches
                    .iter()
                    .any(|cache| cache.context_len() != group_idx + 1)
                {
                    return Err(Error::InvalidInput(format!(
                        "Qwen3-TTS predictor batch rows diverged before codebook {group_idx}"
                    )));
                }
                let ids = previous_codes.reshape((batch_size, 1))?;
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
                previous_codes = select_codes(&logits, batch_size, selections)?;
                steps.push(previous_codes.clone());
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
            super::record_qwen3_tts_host_readback();
            Tensor::stack(&steps, 1)?
                .to_vec2::<u32>()
                .map_err(Error::from)
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
                .row(code as usize)?
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
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder, f16: Option<&Device>) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Attention::load(cfg, vb.pp("self_attn"), f16)?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"), f16)?;

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
    q_proj: ProjectionLinear,
    k_proj: ProjectionLinear,
    v_proj: ProjectionLinear,
    o_proj: ProjectionLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_inv_freq: Vec<f32>,
}

impl Attention {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder, f16: Option<&Device>) -> Result<Self> {
        let pvb = projection_var_builder(&vb, f16);
        let head_dim = cfg.head_dim();

        let q_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                pvb.pp("q_proj"),
            )?,
            f16,
        )?;
        let k_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                pvb.pp("k_proj"),
            )?,
            f16,
        )?;
        let v_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                pvb.pp("v_proj"),
            )?,
            f16,
        )?;
        let o_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                pvb.pp("o_proj"),
            )?,
            f16,
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
        let compute_dtype = x.dtype();
        let state_dtype = cache.arena().config().dtype;
        let q = q.to_dtype(state_dtype)?;
        let k = k.to_dtype(state_dtype)?;
        let v = v.to_dtype(state_dtype)?;
        let out = cache.write_and_attend(
            layer_idx,
            prepared,
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let out =
            out.to_dtype(compute_dtype)?
                .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
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
        let compute_dtype = x.dtype();
        let state_dtype = first.arena().config().dtype;
        let queries = queries.to_dtype(state_dtype)?;
        let keys = keys.to_dtype(state_dtype)?;
        let values = values.to_dtype(state_dtype)?;
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
            .forward(&out.to_dtype(compute_dtype)?.reshape((
                batch_size,
                sequence_len,
                self.num_heads * self.head_dim,
            ))?)
            .map_err(Error::from)
    }
}

/// SwiGLU MLP
struct Mlp {
    gate_proj: ProjectionLinear,
    up_proj: ProjectionLinear,
    down_proj: ProjectionLinear,
}

impl Mlp {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder, f16: Option<&Device>) -> Result<Self> {
        let pvb = projection_var_builder(&vb, f16);
        let gate_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, pvb.pp("gate_proj"))?,
            f16,
        )?;
        let up_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, pvb.pp("up_proj"))?,
            f16,
        )?;
        let down_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(cfg.intermediate_size, cfg.hidden_size, pvb.pp("down_proj"))?,
            f16,
        )?;

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

/// One code per row as a `[batch]` U32 device tensor: the batched argmax
/// when every row is greedy, otherwise each row through its own selection.
fn select_codes(
    logits: &Tensor,
    expected_batch: usize,
    selections: &mut [CodeSelection<'_>],
) -> Result<Tensor> {
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
    if logits.dim(0)? != expected_batch || selections.len() != expected_batch {
        return Err(Error::InferenceError(format!(
            "Qwen3-TTS predictor batch produced {} rows for {} selections, expected {expected_batch}",
            logits.dim(0)?,
            selections.len()
        )));
    }
    if selections.iter().all(CodeSelection::is_greedy) {
        return argmax_device(&logits);
    }
    let codes = selections
        .iter_mut()
        .enumerate()
        .map(|(row, selection)| selection.select(&logits.i(row)?))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&codes, 0).map_err(Error::from)
}

/// Argmax over the last axis as U32 on the logits' device: `[1]` for one row,
/// `[rows]` for a matrix. Ties take the first index, as `torch.argmax` does.
fn argmax_device(logits: &Tensor) -> Result<Tensor> {
    let codes = logits.argmax(D::Minus1)?.to_dtype(DType::U32)?;
    if codes.rank() == 0 {
        codes.reshape(1).map_err(Error::from)
    } else {
        Ok(codes)
    }
}

fn logits_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        1 => Ok(logits.clone()),
        2 if logits.dim(0)? == 1 => logits.i(0).map_err(Error::from),
        _ => Err(Error::InferenceError(format!(
            "Unexpected Qwen3-TTS predictor logits shape: {:?}",
            logits.dims()
        ))),
    }
}

/// Draw one code from a logits row with the sub-talker settings. On CUDA and
/// Metal the top-k candidates are selected on device and only k values are
/// read back; otherwise the row is read back and sampled on the host.
fn sample_code(
    logits: &Tensor,
    params: &SubtalkerSampling,
    draw: &mut dyn FnMut() -> f32,
) -> Result<u32> {
    let row = match logits.rank() {
        1 => logits.clone(),
        2 if logits.dim(0)? == 1 => logits.i(0)?,
        _ => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS predictor logits shape: {:?}",
                logits.dims()
            )))
        }
    };
    let vocab = row.dim(0)?;
    if params.top_k > 0 {
        if let Some(candidates) = bounded_device_sampling_candidates(
            &row,
            vocab,
            params.top_k,
            params.temperature,
            &[],
            1.0,
            0.0,
            None,
        )? {
            super::record_qwen3_tts_host_readback();
            if device_candidates_cover_top_p(&candidates, params.top_p) {
                if let Some(code) = sample_device_candidates(&candidates, params.top_p, draw()) {
                    return Ok(code);
                }
            }
        }
    }
    super::record_qwen3_tts_host_readback();
    let values = row.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    sample_code_from_values(&values, params, draw())
        .ok_or_else(|| Error::InferenceError("Qwen3-TTS predictor logits are not finite".into()))
}

/// Host sampling in Hugging Face's warper order: temperature, top-k, then
/// top-p over the renormalized distribution, then one inverse-CDF draw
/// (`draw` in [0, 1)). Ties keep the lower index first.
fn sample_code_from_values(values: &[f32], params: &SubtalkerSampling, draw: f32) -> Option<u32> {
    let temperature = params.temperature.max(1e-5);
    let mut candidates: Vec<(u32, f32)> = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| (index as u32, value / temperature))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if params.top_k > 0 {
        candidates.truncate(params.top_k);
    }
    let max = candidates[0].1;
    let mut probabilities: Vec<f32> = candidates
        .iter()
        .map(|(_, value)| (value - max).exp())
        .collect();
    let total: f32 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= total;
    }
    if params.top_p < 1.0 {
        let cutoff = params.top_p.max(1e-6);
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for probability in &probabilities {
            cumulative += probability;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        probabilities.truncate(keep.max(1));
        let kept: f32 = probabilities.iter().sum();
        for probability in &mut probabilities {
            *probability /= kept;
        }
    }
    let mut remaining = draw.clamp(0.0, 1.0 - f32::EPSILON);
    for (index, probability) in probabilities.iter().enumerate() {
        if remaining < *probability {
            return Some(candidates[index].0);
        }
        remaining -= probability;
    }
    Some(candidates[probabilities.len() - 1].0)
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
        tiny_predictor_with_input_width(4, device)
    }

    /// A wider `input_width` than the hidden size of 4 adds
    /// `small_to_mtp_projection`, as the 1.7B checkpoints do.
    fn tiny_predictor_with_input_width(input_width: usize, device: &Device) -> CodePredictor {
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
            text_hidden_size: (input_width != 4).then_some(input_width),
        };
        let attention = Attention {
            q_proj: test_linear(4, 4, 1, device).into(),
            k_proj: test_linear(2, 4, 2, device).into(),
            v_proj: test_linear(2, 4, 3, device).into(),
            o_proj: test_linear(4, 4, 4, device).into(),
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
                gate_proj: test_linear(8, 4, 5, device).into(),
                up_proj: test_linear(8, 4, 6, device).into(),
                down_proj: test_linear(4, 8, 7, device).into(),
            },
        };
        let codec_embeddings = (0..3)
            .map(|group| {
                let values = (0..8 * input_width)
                    .map(|index| {
                        let value = (index * 5 + group * 3) % 19;
                        (value as f32 - 9.0) / 16.0
                    })
                    .collect::<Vec<_>>();
                CodecEmbedding::Dense(Embedding::new(
                    Tensor::from_vec(values, (8, input_width), device).unwrap(),
                    input_width,
                ))
            })
            .collect();
        CodePredictor {
            codec_embeddings,
            small_to_mtp_projection: (input_width != 4)
                .then(|| test_linear(4, input_width, 13, device)),
            layers: vec![layer],
            norm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            lm_heads: (0..3)
                .map(|head| test_linear(8, 4, 8 + head, device).into())
                .collect(),
            device: device.clone(),
            cfg,
            num_code_groups: 3,
        }
    }

    fn test_arena(instance: u64) -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        test_arena_with_dtype(instance, DType::F32)
    }

    fn test_arena_with_dtype(
        instance: u64,
        dtype: DType,
    ) -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
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
            dtype,
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
        frame_input_with_width(seed, 4, device)
    }

    fn frame_input_with_width(seed: usize, width: usize, device: &Device) -> Tensor {
        let values = (0..width)
            .map(|index| {
                let value = (index * 11 + seed) % 23;
                (value as f32 - 11.0) / 16.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, 1, width), device).unwrap()
    }

    #[test]
    fn non_cuda_predictor_argmax_uses_reference_ordering() {
        let logits = Tensor::new(vec![0.0f32, 4.0, 4.0, 3.0], &Device::Cpu).unwrap();

        assert_eq!(
            argmax_device(&logits).unwrap().to_vec1::<u32>().unwrap(),
            [1]
        );
    }

    #[test]
    fn predictor_argmax_reference_accepts_single_row_logits() {
        let logits = Tensor::new(&[[0.0f32, 1.0, 7.0, 3.0]], &Device::Cpu).unwrap();

        assert_eq!(
            argmax_device(&logits_row(&logits).unwrap())
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            [2]
        );
    }

    #[test]
    fn physical_predictor_cursor_matches_prefill_and_dependent_groups() {
        assert_eq!(code_predictor_physical_context_tokens(0), 2);
        assert_eq!(code_predictor_physical_context_tokens(1), 2);
        assert_eq!(code_predictor_physical_context_tokens(15), 16);
    }

    #[test]
    fn f32_predictor_compute_runs_against_f16_physical_kv() {
        let device = Device::Cpu;
        let model = tiny_predictor(&device);
        let (arena, bindings) = test_arena_with_dtype(710, DType::F16);
        let mut cache = test_cache(arena, &bindings, 0, 0);

        let codes = model
            .generate_acoustic_codes_physical(
                &frame_input(3, &device),
                &frame_input(7, &device),
                &mut cache,
                &mut CodeSelection::Greedy,
            )
            .unwrap();

        assert_eq!(codes.len(), 3);
        assert_eq!(cache.context_len(), 4);
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
            .generate_acoustic_codes_physical(
                &talker_a,
                &semantic_a,
                &mut scalar_a,
                &mut CodeSelection::Greedy,
            )
            .unwrap();
        let scalar_codes_b = model
            .generate_acoustic_codes_physical(
                &talker_b,
                &semantic_b,
                &mut scalar_b,
                &mut CodeSelection::Greedy,
            )
            .unwrap();
        let talker = Tensor::cat(&[&talker_a, &talker_b], 0).unwrap();
        let semantic = Tensor::cat(&[&semantic_a, &semantic_b], 0).unwrap();
        let mut caches = [&mut batch_a, &mut batch_b];
        let batch_codes = model
            .generate_acoustic_codes_physical_batch(
                &talker,
                &semantic,
                &mut caches,
                &mut [CodeSelection::Greedy, CodeSelection::Greedy],
            )
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

    /// The 1.7B checkpoints feed the predictor at the talker's width (2048)
    /// and project to its hidden size (1024). A batch takes inputs at the
    /// talker's width, as the scalar path does, and matches its codes.
    #[test]
    fn projected_predictor_batch_takes_talker_width_inputs() {
        let device = Device::Cpu;
        let model = tiny_predictor_with_input_width(6, &device);
        let (scalar_arena, scalar_bindings) = test_arena(716);
        let (batch_arena, batch_bindings) = test_arena(717);
        let mut scalar_a = test_cache(scalar_arena.clone(), &scalar_bindings, 0, 0);
        let mut scalar_b = test_cache(scalar_arena, &scalar_bindings, 6, 0);
        let mut batch_a = test_cache(batch_arena.clone(), &batch_bindings, 0, 0);
        let mut batch_b = test_cache(batch_arena, &batch_bindings, 6, 0);
        let talker_a = frame_input_with_width(3, 6, &device);
        let talker_b = frame_input_with_width(5, 6, &device);
        let semantic_a = frame_input_with_width(7, 6, &device);
        let semantic_b = frame_input_with_width(9, 6, &device);
        let scalar_codes_a = model
            .generate_acoustic_codes_physical(
                &talker_a,
                &semantic_a,
                &mut scalar_a,
                &mut CodeSelection::Greedy,
            )
            .unwrap();
        let scalar_codes_b = model
            .generate_acoustic_codes_physical(
                &talker_b,
                &semantic_b,
                &mut scalar_b,
                &mut CodeSelection::Greedy,
            )
            .unwrap();
        let batch_codes = model
            .generate_acoustic_codes_physical_batch(
                &Tensor::cat(&[&talker_a, &talker_b], 0).unwrap(),
                &Tensor::cat(&[&semantic_a, &semantic_b], 0).unwrap(),
                &mut [&mut batch_a, &mut batch_b],
                &mut [CodeSelection::Greedy, CodeSelection::Greedy],
            )
            .unwrap();

        assert_eq!(batch_codes, vec![scalar_codes_a, scalar_codes_b]);
        assert_eq!((batch_a.context_len(), batch_b.context_len()), (4, 4));
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
            .generate_acoustic_codes_physical_batch(
                &talker,
                &semantic,
                &mut caches,
                &mut [CodeSelection::Greedy, CodeSelection::Greedy],
            )
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
            .generate_acoustic_codes_physical(
                &talker,
                &semantic,
                &mut scalar,
                &mut CodeSelection::Greedy,
            )
            .unwrap();
        let mut caches = [&mut fallback];
        let actual = model
            .generate_acoustic_codes_physical_batch(
                &talker,
                &semantic,
                &mut caches,
                &mut [CodeSelection::Greedy],
            )
            .unwrap();

        assert_eq!(actual, vec![expected]);
        assert_eq!(fallback.context_len(), scalar.context_len());
    }

    /// QTTS-4: one predictor frame reads its codes back once (it read one
    /// code back per codebook), greedy or sampled with the sub-talker
    /// defaults, and a two-row batch reads back once for both rows.
    #[test]
    fn predictor_frame_reads_its_codes_back_once() {
        let device = Device::Cpu;
        let model = tiny_predictor(&device);
        let talker = frame_input(31, &device);
        let semantic = frame_input(33, &device);
        let readbacks = || super::super::thread_qwen3_tts_host_readbacks();
        let mut draw = {
            let mut state = 0x2545_F491u64;
            move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32
            }
        };
        for (instance, sampled) in [(730u64, false), (731, true)] {
            let (arena, bindings) = test_arena(instance);
            let mut cache = test_cache(arena, &bindings, 0, 0);
            let mut selection = if sampled {
                CodeSelection::Sample {
                    params: SubtalkerSampling::default(),
                    draw: &mut draw,
                }
            } else {
                CodeSelection::Greedy
            };
            let before = readbacks();
            let codes = model
                .generate_acoustic_codes_physical(&talker, &semantic, &mut cache, &mut selection)
                .unwrap();
            assert_eq!(codes.len(), 3);
            assert_eq!(readbacks() - before, 1, "sampled: {sampled}");
        }
        let (arena, bindings) = test_arena(732);
        let mut row0 = test_cache(arena.clone(), &bindings, 0, 0);
        let mut row1 = test_cache(arena, &bindings, 6, 0);
        let before = readbacks();
        model
            .generate_acoustic_codes_physical_batch(
                &Tensor::cat(&[&talker, &talker], 0).unwrap(),
                &Tensor::cat(&[&semantic, &semantic], 0).unwrap(),
                &mut [&mut row0, &mut row1],
                &mut [CodeSelection::Greedy, CodeSelection::Greedy],
            )
            .unwrap();
        assert_eq!(readbacks() - before, 1);
    }

    /// QTTS-6: host sampling follows the warper order (temperature, top-k,
    /// top-p) and draws by inverse CDF.
    #[test]
    fn subtalker_host_sampling_honours_top_k_top_p_and_temperature() {
        let values = [2.0f32, 1.0, 0.0, -1.0, f32::NAN];
        let top2 = SubtalkerSampling {
            do_sample: true,
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        // p(0) = e^2 / (e^2 + e^1) = 0.7311 within the top two.
        assert_eq!(sample_code_from_values(&values, &top2, 0.0), Some(0));
        assert_eq!(sample_code_from_values(&values, &top2, 0.730), Some(0));
        assert_eq!(sample_code_from_values(&values, &top2, 0.732), Some(1));
        assert_eq!(sample_code_from_values(&values, &top2, 0.999_99), Some(1));
        // Top-k = 0 keeps every finite logit: the last bucket is index 3.
        let all = SubtalkerSampling { top_k: 0, ..top2 };
        assert_eq!(sample_code_from_values(&values, &all, 0.999_99), Some(3));
        // Top-p 0.5 keeps only the head (0.731 >= 0.5).
        let nucleus = SubtalkerSampling { top_p: 0.5, ..all };
        assert_eq!(
            sample_code_from_values(&values, &nucleus, 0.999_99),
            Some(0)
        );
        // Temperature 0.5 sharpens: p(0) = e^4 / (e^4 + e^2) = 0.8808.
        let sharp = SubtalkerSampling {
            temperature: 0.5,
            ..top2
        };
        assert_eq!(sample_code_from_values(&values, &sharp, 0.880), Some(0));
        assert_eq!(sample_code_from_values(&values, &sharp, 0.882), Some(1));
        assert_eq!(sample_code_from_values(&[f32::NAN], &top2, 0.5), None);
    }

    /// QTTS-6: the predictor samples every residual codebook through the
    /// row's selection: top-1 sampling equals greedy, a high draw under
    /// top-2 picks the runner-up, and a batch row keeps its own selection.
    #[test]
    fn predictor_samples_residual_codebooks_per_row() {
        let device = Device::Cpu;
        let model = tiny_predictor(&device);
        let talker = frame_input(23, &device);
        let semantic = frame_input(25, &device);
        let run = |instance: u64, selection: &mut CodeSelection<'_>| {
            let (arena, bindings) = test_arena(instance);
            let mut cache = test_cache(arena, &bindings, 0, 0);
            model
                .generate_acoustic_codes_physical(&talker, &semantic, &mut cache, selection)
                .unwrap()
        };
        let greedy = run(720, &mut CodeSelection::Greedy);
        let mut high = || 0.999_99f32;
        let top1 = run(
            721,
            &mut CodeSelection::Sample {
                params: SubtalkerSampling {
                    top_k: 1,
                    ..SubtalkerSampling::default()
                },
                draw: &mut high,
            },
        );
        assert_eq!(top1, greedy);
        let top2_params = SubtalkerSampling {
            top_k: 2,
            ..SubtalkerSampling::default()
        };
        // Gumbel-max over the top two: a near-zero uniform for the leader and a
        // near-one uniform for the runner-up makes the runner-up win.
        let mut flip = false;
        let mut runner_up = move || {
            flip = !flip;
            if flip {
                1e-4f32
            } else {
                0.999_9
            }
        };
        let sampled = run(
            722,
            &mut CodeSelection::Sample {
                params: top2_params,
                draw: &mut runner_up,
            },
        );
        assert_ne!(sampled[0], greedy[0], "the draw must pick the runner-up");

        // Batch: row 0 greedy, row 1 sampled, each equal to its scalar run.
        let (arena, bindings) = test_arena(723);
        let mut row0 = test_cache(arena.clone(), &bindings, 0, 0);
        let mut row1 = test_cache(arena, &bindings, 6, 0);
        let batch_talker = Tensor::cat(&[&talker, &talker], 0).unwrap();
        let batch_semantic = Tensor::cat(&[&semantic, &semantic], 0).unwrap();
        let mut flip = false;
        let mut runner_up = move || {
            flip = !flip;
            if flip {
                1e-4f32
            } else {
                0.999_9
            }
        };
        let codes = model
            .generate_acoustic_codes_physical_batch(
                &batch_talker,
                &batch_semantic,
                &mut [&mut row0, &mut row1],
                &mut [
                    CodeSelection::Greedy,
                    CodeSelection::Sample {
                        params: top2_params,
                        draw: &mut runner_up,
                    },
                ],
            )
            .unwrap();
        assert_eq!(codes, vec![greedy, sampled]);
    }
}
