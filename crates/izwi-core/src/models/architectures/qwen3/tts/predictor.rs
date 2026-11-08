//! Code Predictor for multi-codebook RVQ token generation.
//!
//! The code predictor generates the residual codebook tokens after the talker
//! has produced the first (semantic) codebook. It uses a smaller transformer
//! for efficient multi-token prediction.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};
use crate::kernels::f16_weight_linear::{projection_var_builder, ProjectionLinear};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::architectures::qwen3::tts::config::CodePredictorConfig;
use crate::models::architectures::qwen3::tts::decode_rows::{self, DECODE_ROWS};
use crate::models::architectures::qwen3::tts::rope::{
    build_rope_inv_freq, build_rope_window_full, qwen_rotate_half,
};
pub use crate::models::shared::attention::physical::PhysicalPagedKvCache as CodePredictorPhysicalCache;
use crate::models::shared::attention::physical::PreparedPhysicalPagedStep;
use crate::models::shared::sampling::{
    bounded_device_sampling_candidates, device_candidates_cover_top_p, gumbel_noise,
    resident_top_k_rows_with_noise, resident_top_k_sample, resident_top_k_sample_rows,
    sample_device_candidates, DEVICE_TOP_K_LIMIT,
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

/// Kill switch for the code predictor's CUDA graphs (a boolean; on when
/// unset). Read once: load pricing and the predictor read it.
pub(crate) const QWEN3_TTS_PREDICTOR_GRAPHS_ENV: &str = "IZWI_QWEN3_TTS_PREDICTOR_GRAPHS";

fn predictor_graphs_requested() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(QWEN3_TTS_PREDICTOR_GRAPHS_ENV, true));
    ON.get()
}

/// The rows every graphed frame runs. Spare rows are zero rows, so every
/// kernel of a frame runs one shape however many requests share the step
/// (a row's codes never depend on the rows beside it), and one graph per
/// sampling setting serves every width. A CUDA frame the graphs do not
/// take runs the same rows eagerly.
pub(crate) const PREDICTOR_GRAPH_ROWS: usize = DECODE_ROWS;

const MIB: u64 = 1024 * 1024;

/// What the predictor graphs may hold: one graph per input dtype and
/// sampling setting (greedy, and each sub-talker top-k and temperature in
/// use). A 32-row frame's working set is a few MiB (15 codebooks of
/// `[32, 2048]` logits and their sorts, 17 positions of dense KV); the
/// retained buffers are its inputs (two `[32, 1, 2048]` hiddens and the
/// noise) and its codes.
const PREDICTOR_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 4,
    retained_bytes: 16 * MIB,
    pool_bytes: 112 * MIB,
};

/// Device bytes a load on `backend` prices for the predictor graphs, which
/// the load's self-check allocates: zero off CUDA or with the graphs
/// switched off.
pub(crate) fn qwen3_tts_predictor_graph_device_bytes(backend: BackendKind) -> u64 {
    if backend == BackendKind::Cuda && predictor_graphs_requested() {
        PREDICTOR_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// How every row of a graphed frame picks its codes, as the eager frame
/// picks them.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FrameSelection {
    /// Every row greedy: one argmax per codebook.
    Greedy,
    /// Every row draws with the same sub-talker top-k and temperature at
    /// top-p 1 ([`resident_top_k_sample_rows`]), its Gumbel noise drawn
    /// before the frame.
    TopK { top_k: usize, temperature: f32 },
}

impl FrameSelection {
    /// The selection every row shares, or `None` when rows pick apart (or
    /// draw on the host).
    fn of(selections: &[CodeSelection<'_>]) -> Option<Self> {
        if selections.is_empty() {
            return None;
        }
        if selections.iter().all(CodeSelection::is_greedy) {
            return Some(Self::Greedy);
        }
        let resident = |selection: &CodeSelection<'_>| match selection {
            CodeSelection::Sample { params, .. }
                if !params.is_greedy()
                    && params.top_p >= 1.0
                    && (1..=DEVICE_TOP_K_LIMIT).contains(&params.top_k) =>
            {
                Some((params.top_k, params.temperature.to_bits()))
            }
            _ => None,
        };
        let first = resident(&selections[0])?;
        selections
            .iter()
            .all(|selection| resident(selection) == Some(first))
            .then(|| Self::TopK {
                top_k: first.0,
                temperature: f32::from_bits(first.1),
            })
    }
}

/// A frame graph's key: its selection and its inputs' dtype.
type PredictorGraphKey = (FrameSelection, DType);

/// The code predictor's frame graphs ([`CodePredictor::resolve_graphs`]),
/// or why there are none, and how many frames ran through them.
struct PredictorGraphs {
    graphs: std::result::Result<ShapeGraphs<PredictorGraphKey>, String>,
    graphed_frames: AtomicU64,
    /// Frames the graphs could not take: rows that pick apart, or more rows
    /// than [`PREDICTOR_GRAPH_ROWS`].
    eager_frames: AtomicU64,
    /// The heads' column ids `0..vocab` (U32), every frame's last input:
    /// the top-k reads them from the graph's own copy, never from a cache
    /// that could free them under a captured graph.
    column_ids: std::sync::OnceLock<Tensor>,
}

impl PredictorGraphs {
    fn off(reason: &str) -> Self {
        Self {
            graphs: Err(reason.into()),
            graphed_frames: AtomicU64::new(0),
            eager_frames: AtomicU64::new(0),
            column_ids: std::sync::OnceLock::new(),
        }
    }

    /// `0..vocab` as U32 on `device`, uploaded once.
    fn column_ids(&self, vocab: usize, device: &Device) -> Result<Tensor> {
        if let Some(ids) = self.column_ids.get() {
            return Ok(ids.clone());
        }
        let ids = Tensor::arange(0u32, vocab as u32, device)?;
        Ok(self.column_ids.get_or_init(|| ids).clone())
    }
}

/// Every sampling row's Gumbel noise for all `groups` codebooks of a frame,
/// `[groups, PREDICTOR_GRAPH_ROWS, keep]`: drawn from each row's own source
/// codebook after codebook, `keep` draws each, as the eager frame draws it
/// ([`resident_top_k_sample`]); spare rows get zeros.
fn frame_noise(
    selections: &mut [CodeSelection<'_>],
    groups: usize,
    keep: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut noise = vec![0f32; groups * PREDICTOR_GRAPH_ROWS * keep];
    for (row, selection) in selections.iter_mut().enumerate() {
        let CodeSelection::Sample { draw, .. } = selection else {
            return Err(Error::InferenceError(
                "a sampled Qwen3-TTS predictor frame has a greedy row".into(),
            ));
        };
        for group in 0..groups {
            let start = (group * PREDICTOR_GRAPH_ROWS + row) * keep;
            noise[start..start + keep].copy_from_slice(&gumbel_noise(keep, &mut **draw));
        }
    }
    Tensor::from_vec(noise, (groups, PREDICTOR_GRAPH_ROWS, keep), device).map_err(Error::from)
}

/// One codebook's codes of a dense frame (`logits` `[rows, vocab]`) as a
/// `[rows]` U32 tensor, picked as the eager frame picks them, with the
/// codebook's pre-drawn noise and the heads' column ids.
fn dense_select(
    logits: &Tensor,
    selection: FrameSelection,
    noise: &Tensor,
    column_ids: &Tensor,
    group: usize,
) -> Result<Tensor> {
    match selection {
        FrameSelection::Greedy => argmax_device(logits),
        FrameSelection::TopK { top_k, temperature } => resident_top_k_rows_with_noise(
            logits,
            top_k,
            temperature,
            &noise.i(group)?,
            column_ids,
        )?
        .ok_or_else(|| {
            Error::InferenceError("Qwen3-TTS predictor top-k frame drew nothing".into())
        }),
    }
}

/// One frame's acoustic codes for a batch of rows: on the device, one row
/// of codes per codebook (`[groups, rows]` U32), and read back per row.
pub struct AcousticCodeRows {
    pub by_group: Tensor,
    pub rows: Vec<Vec<u32>>,
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
    graphs: PredictorGraphs,
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
            graphs: PredictorGraphs::off("not resolved"),
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
        if decode_rows::row_invariant(&self.device) {
            // CUDA: a lone row takes the batch frame, padded as every frame.
            return self
                .generate_acoustic_code_rows_physical_batch(
                    talker_hidden,
                    semantic_embed,
                    &mut [cache],
                    std::slice::from_mut(selection),
                )?
                .rows
                .into_iter()
                .next()
                .ok_or_else(|| {
                    Error::InferenceError("Qwen3-TTS predictor frame returned no row".into())
                });
        }
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
    /// codebook geometry are rejected and must be split by the scheduler. Off
    /// CUDA one row falls back to the scalar implementation; on CUDA every
    /// frame runs the dense frame at 32 rows, graphed or eagerly
    /// ([`Self::eager_dense_frame`]), and touches no invocation page.
    pub fn generate_acoustic_codes_physical_batch(
        &self,
        talker_hidden: &Tensor,
        semantic_embeds: &Tensor,
        caches: &mut [&mut CodePredictorPhysicalCache],
        selections: &mut [CodeSelection<'_>],
    ) -> Result<Vec<Vec<u32>>> {
        self.generate_acoustic_code_rows_physical_batch(
            talker_hidden,
            semantic_embeds,
            caches,
            selections,
        )
        .map(|codes| codes.rows)
    }

    /// [`Self::generate_acoustic_codes_physical_batch`] that also keeps the
    /// frame's codes on the device, `[groups, rows]`, for
    /// [`Self::acoustic_embeddings_sum_rows`].
    pub fn generate_acoustic_code_rows_physical_batch(
        &self,
        talker_hidden: &Tensor,
        semantic_embeds: &Tensor,
        caches: &mut [&mut CodePredictorPhysicalCache],
        selections: &mut [CodeSelection<'_>],
    ) -> Result<AcousticCodeRows> {
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
        let row_invariant = decode_rows::row_invariant(&self.device);
        if batch_size == 1 && !row_invariant {
            let codes = self.generate_acoustic_codes_physical(
                talker_hidden,
                semantic_embeds,
                caches[0],
                &mut selections[0],
            )?;
            let by_group = Tensor::from_vec(codes.clone(), (codes.len(), 1), &self.device)?;
            return Ok(AcousticCodeRows {
                by_group,
                rows: vec![codes],
            });
        }
        for cache in caches.iter() {
            self.validate_physical_workspace(cache)?;
        }
        if let Some(codes) = self.graphed_frame(talker_hidden, semantic_embeds, selections)? {
            return Ok(codes);
        }
        if row_invariant && batch_size <= DECODE_ROWS {
            return self.eager_dense_frame(talker_hidden, semantic_embeds, selections);
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
                return Ok(AcousticCodeRows {
                    by_group: Tensor::zeros((0, batch_size), DType::U32, &self.device)?,
                    rows: vec![Vec::new(); batch_size],
                });
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
            let by_group = Tensor::stack(&steps, 0)?;
            let codes = by_group.to_vec2::<u32>()?;
            let rows = (0..batch_size)
                .map(|row| codes.iter().map(|group| group[row]).collect())
                .collect();
            Ok(AcousticCodeRows { by_group, rows })
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

    /// Turns the frame graphs on (CUDA only, unless
    /// [`QWEN3_TTS_PREDICTOR_GRAPHS_ENV`] is off) once a self-check passes:
    /// for greedy frames and for the sub-talker default draw, two frames of
    /// xorshift inputs at `input_dtype` (the talker's hidden dtype, which
    /// every frame's inputs have), each captured or replayed and compared
    /// with the eager dense frame bit for bit ([`ShapeGraphs::run`]). A
    /// mismatch or a CUDA error leaves every frame on the eager paged path,
    /// with the reason in the diagnostics.
    pub(crate) fn resolve_graphs(&mut self, input_dtype: DType) {
        let resolved = if predictor_graphs_requested() {
            ShapeGraphs::resolve(
                "Qwen3-TTS code predictor",
                &self.device,
                PREDICTOR_GRAPH_LIMITS,
            )
        } else {
            Err(format!("{QWEN3_TTS_PREDICTOR_GRAPHS_ENV} is off"))
        };
        let resolved =
            resolved.and_then(
                |graphs| match self.self_check_graphs(&graphs, input_dtype) {
                    Ok(()) if graphs.is_on() => Ok(graphs),
                    Ok(()) => Err(format!(
                        "self-check failed: {}",
                        graphs.diagnostics()["off"]
                    )),
                    Err(error) => Err(format!("self-check failed: {error}")),
                },
            );
        match &resolved {
            Ok(_) => tracing::info!(
                rows = PREDICTOR_GRAPH_ROWS,
                "Qwen3-TTS code predictor frames run as CUDA graphs"
            ),
            Err(reason) if self.device.is_cuda() => tracing::warn!(
                %reason,
                "Qwen3-TTS code predictor frames run eagerly"
            ),
            Err(_) => {}
        }
        self.graphs.graphs = resolved;
    }

    /// Two frames per default selection through `graphs` (see
    /// [`Self::resolve_graphs`]): the first captures the graph and checks a
    /// replay against the eager frame, the second checks a replay over new
    /// inputs, so a value baked into the graph shows.
    fn self_check_graphs(
        &self,
        graphs: &ShapeGraphs<PredictorGraphKey>,
        input_dtype: DType,
    ) -> Result<()> {
        let width = self.cfg.input_hidden_size();
        let groups = self.lm_heads.len();
        let default = SubtalkerSampling::default();
        let keep = default.top_k.min(self.cfg.vocab_size);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let selections = [
            FrameSelection::Greedy,
            FrameSelection::TopK {
                top_k: default.top_k,
                temperature: default.temperature,
            },
        ];
        for selection in selections {
            for _ in 0..2 {
                let mut hidden = || -> Result<Tensor> {
                    let values = (0..PREDICTOR_GRAPH_ROWS * width)
                        .map(|_| uniform() * 2.0 - 1.0)
                        .collect::<Vec<_>>();
                    Tensor::from_vec(values, (PREDICTOR_GRAPH_ROWS, 1, width), &self.device)?
                        .to_dtype(input_dtype)
                        .map_err(Error::from)
                };
                let talker_hidden = hidden()?;
                let semantic = hidden()?;
                let noise = match selection {
                    FrameSelection::Greedy => Tensor::zeros(1, DType::F32, &self.device)?,
                    FrameSelection::TopK { .. } => Tensor::from_vec(
                        gumbel_noise(groups * PREDICTOR_GRAPH_ROWS * keep, &mut uniform),
                        (groups, PREDICTOR_GRAPH_ROWS, keep),
                        &self.device,
                    )?,
                };
                let column_ids = self.graphs.column_ids(self.cfg.vocab_size, &self.device)?;
                self.run_frame_graph(
                    graphs,
                    selection,
                    &[talker_hidden, semantic, noise, column_ids],
                )?;
            }
        }
        Ok(())
    }

    /// The frame graphs' state and counters, for the model diagnostics.
    pub fn graphs_diagnostics(&self) -> serde_json::Value {
        let mut diagnostics = match &self.graphs.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        };
        diagnostics["rows"] = PREDICTOR_GRAPH_ROWS.into();
        diagnostics["graphed_frames"] = self.graphs.graphed_frames.load(Relaxed).into();
        diagnostics["eager_frames"] = self.graphs.eager_frames.load(Relaxed).into();
        diagnostics
    }

    /// One frame through the graphs, when they are on and every row shares
    /// a selection: rows padded to [`PREDICTOR_GRAPH_ROWS`] with zero rows,
    /// every sampling row's noise for all codebooks drawn up front in the
    /// eager frame's order, the codes read back once. The invocation caches
    /// are not touched (a frame's KV is dense, inside the graph). `None`
    /// runs the eager frame.
    fn graphed_frame(
        &self,
        talker_hidden: &Tensor,
        semantic_embeds: &Tensor,
        selections: &mut [CodeSelection<'_>],
    ) -> Result<Option<AcousticCodeRows>> {
        let Ok(graphs) = &self.graphs.graphs else {
            return Ok(None);
        };
        if !graphs.is_on() {
            return Ok(None);
        }
        let rows = talker_hidden.dim(0)?;
        let selection = FrameSelection::of(selections).filter(|_| {
            rows == selections.len()
                && rows <= PREDICTOR_GRAPH_ROWS
                && !self.lm_heads.is_empty()
                && semantic_embeds.dims() == talker_hidden.dims()
        });
        let Some(selection) = selection else {
            self.graphs.eager_frames.fetch_add(1, Relaxed);
            return Ok(None);
        };
        let groups = self.lm_heads.len();
        let spare = PREDICTOR_GRAPH_ROWS - rows;
        let noise = match selection {
            FrameSelection::Greedy => Tensor::zeros(1, DType::F32, &self.device)?,
            FrameSelection::TopK { top_k, .. } => frame_noise(
                selections,
                groups,
                top_k.min(self.cfg.vocab_size),
                &self.device,
            )?,
        };
        let inputs = [
            talker_hidden.pad_with_zeros(0, 0, spare)?,
            semantic_embeds.pad_with_zeros(0, 0, spare)?,
            noise,
            self.graphs.column_ids(self.cfg.vocab_size, &self.device)?,
        ];
        let codes = self.run_frame_graph(graphs, selection, &inputs)?;
        let by_group = codes.narrow(1, 0, rows)?.contiguous()?;
        super::record_qwen3_tts_host_readback();
        let host = by_group.to_vec2::<u32>()?;
        self.graphs.graphed_frames.fetch_add(1, Relaxed);
        Ok(Some(AcousticCodeRows {
            rows: (0..rows)
                .map(|row| host.iter().map(|group| group[row]).collect())
                .collect(),
            by_group,
        }))
    }

    /// [`Self::dense_frame_codes`] over `inputs` (talker hidden, semantic
    /// embeddings, noise, column ids) through the graph of its key.
    fn run_frame_graph(
        &self,
        graphs: &ShapeGraphs<PredictorGraphKey>,
        selection: FrameSelection,
        inputs: &[Tensor],
    ) -> Result<Tensor> {
        let bytes = |tensor: &Tensor| (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64;
        // A graph keeps a stable copy of each input, and its codes.
        let codes =
            (self.lm_heads.len() * PREDICTOR_GRAPH_ROWS * DType::U32.size_in_bytes()) as u64;
        let retained = inputs.iter().map(bytes).sum::<u64>() + codes;
        let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            let [talker_hidden, semantic, noise, column_ids] = inputs else {
                return Err(Error::InferenceError(format!(
                    "Qwen3-TTS predictor graph received {} inputs, expected 4",
                    inputs.len()
                )));
            };
            Ok(vec![self.dense_frame_codes(
                talker_hidden,
                semantic,
                selection,
                noise,
                column_ids,
            )?])
        };
        let key = (selection, inputs[0].dtype());
        graphs
            .run(key, retained, inputs, &body)?
            .pop()
            .ok_or_else(|| {
                Error::InferenceError("Qwen3-TTS predictor graph returned nothing".into())
            })
    }

    /// One frame's codes, `[groups, rows]` U32, from `[rows, 1, input]`
    /// talker hidden states and semantic embeddings, entirely on the device:
    /// the eager batch frame with each row's KV in a dense per-row cache
    /// instead of paged invocation pages ([`Attention::forward_dense`]) and
    /// each codebook's noise read from `noise` (`[groups, rows, keep]`; any
    /// tensor when greedy) and the heads' column ids from `column_ids`
    /// (`0..vocab`). A pure function of its inputs and the weights, so a
    /// CUDA graph can capture it.
    fn dense_frame_codes(
        &self,
        talker_hidden: &Tensor,
        semantic: &Tensor,
        selection: FrameSelection,
        noise: &Tensor,
        column_ids: &Tensor,
    ) -> Result<Tensor> {
        self.dense_frame(talker_hidden, semantic, &mut |logits, group| {
            dense_select(logits, selection, noise, column_ids, group)
        })
    }

    /// A frame that the graphs do not take, on CUDA: the graphs' dense frame
    /// run eagerly at [`DECODE_ROWS`] rows ([`Self::padded_dense_frame`]),
    /// each row picking its codes through its own selection, as the eager
    /// paged frame picks them. A row's logits are those of a graphed frame,
    /// at any width, so its codes do not depend on the rows beside it or on
    /// whether the graphs took the frame.
    fn eager_dense_frame(
        &self,
        talker_hidden: &Tensor,
        semantic_embeds: &Tensor,
        selections: &mut [CodeSelection<'_>],
    ) -> Result<AcousticCodeRows> {
        let rows = talker_hidden.dim(0)?;
        let by_group =
            self.padded_dense_frame(talker_hidden, semantic_embeds, &mut |logits, _| {
                select_codes(logits, rows, selections)
            })?;
        super::record_qwen3_tts_host_readback();
        let host = by_group.to_vec2::<u32>()?;
        Ok(AcousticCodeRows {
            rows: (0..rows)
                .map(|row| host.iter().map(|group| group[row]).collect())
                .collect(),
            by_group,
        })
    }

    /// [`Self::dense_frame`] over `[rows, 1, input]` inputs padded with zero
    /// rows to [`DECODE_ROWS`], so every kernel runs one shape at any width;
    /// `select` sees the real rows' `[rows, vocab]` logits and returns their
    /// `[rows]` codes (the spare rows take code 0). Returns the real rows'
    /// codes, `[groups, rows]`.
    fn padded_dense_frame(
        &self,
        talker_hidden: &Tensor,
        semantic_embeds: &Tensor,
        select: &mut dyn FnMut(&Tensor, usize) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let rows = talker_hidden.dim(0)?;
        if rows == 0 || rows > DECODE_ROWS || semantic_embeds.dims() != talker_hidden.dims() {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS padded predictor frame takes 1 to {DECODE_ROWS} matching rows, got {:?} and {:?}",
                talker_hidden.dims(),
                semantic_embeds.dims()
            )));
        }
        let spare = DECODE_ROWS - rows;
        let codes = self.dense_frame(
            &talker_hidden.pad_with_zeros(0, 0, spare)?,
            &semantic_embeds.pad_with_zeros(0, 0, spare)?,
            &mut |logits, group| {
                select(&logits.narrow(0, 0, rows)?, group)?
                    .pad_with_zeros(0, 0, spare)
                    .map_err(Error::from)
            },
        )?;
        codes.narrow(1, 0, rows)?.contiguous().map_err(Error::from)
    }

    /// The dense frame's body: [`Self::dense_frame_codes`] with each
    /// codebook's `[rows]` U32 codes picked by `select` from its
    /// `[rows, vocab]` logits and the codebook's index.
    fn dense_frame(
        &self,
        talker_hidden: &Tensor,
        semantic: &Tensor,
        select: &mut dyn FnMut(&Tensor, usize) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let rows = talker_hidden.dim(0)?;
        let input = Tensor::cat(&[talker_hidden, semantic], 1)?;
        let hidden = match &self.small_to_mtp_projection {
            Some(proj) => proj.forward(&input)?,
            None => input,
        };
        let mut kv = (0..self.layers.len()).map(|_| None).collect::<Vec<_>>();
        let hidden = self.dense_forward(&hidden, 0, &mut kv)?;
        let last = hidden
            .i((.., CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS - 1, ..))?
            .contiguous()?;
        let Some(first_head) = self.lm_heads.first() else {
            return Tensor::zeros((0, rows), DType::U32, &self.device).map_err(Error::from);
        };
        let mut previous = select(&first_head.forward(&last)?, 0)?;
        let mut codes = vec![previous.clone()];
        for group_idx in 1..self.lm_heads.len() {
            let ids = previous.reshape((rows, 1))?;
            let mut step = self.codec_embeddings[group_idx - 1].forward(&ids)?;
            if let Some(proj) = &self.small_to_mtp_projection {
                step = proj.forward(&step)?;
            }
            let step = self.dense_forward(&step, group_idx + 1, &mut kv)?;
            let logits = self.lm_heads[group_idx].forward(&step)?;
            previous = select(&logits.i((.., 0, ..))?, group_idx)?;
            codes.push(previous.clone());
        }
        Tensor::stack(&codes, 0).map_err(Error::from)
    }

    /// The layers and final norm over `x` (`[rows, seq, hidden]`) at
    /// positions `start_pos..`, each layer attending over its dense cache in
    /// `kv` (grown here).
    fn dense_forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        kv: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let mut hidden = x.clone();
        for (layer, cache) in self.layers.iter().zip(kv.iter_mut()) {
            hidden = layer.forward_dense(&hidden, start_pos, cache)?;
        }
        self.norm.forward(&hidden).map_err(Error::from)
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

    /// [`Self::get_acoustic_embeddings_sum`] of every row at once, from
    /// `codes_by_group` (`[groups, rows]` U32 on the device): one gather per
    /// codebook for all rows, added in the per-row order, so each row of the
    /// `[rows, 1, codec_embed_dim]` result is bit for bit its per-row sum. A
    /// step runs 15 gathers and 14 adds instead of that many per row.
    pub fn acoustic_embeddings_sum_rows(&self, codes_by_group: &Tensor) -> Result<Tensor> {
        let (groups, _rows) = codes_by_group.dims2()?;
        if groups == 0 || groups != self.codec_embeddings.len() {
            return Err(Error::InvalidInput(format!(
                "Expected {} acoustic code groups, got {groups}",
                self.codec_embeddings.len()
            )));
        }
        let codes = codes_by_group.contiguous()?;
        let mut sum = self.codec_embeddings[0].forward(&codes.i(0)?)?;
        for (group_idx, embedding) in self.codec_embeddings.iter().enumerate().skip(1) {
            sum = sum.broadcast_add(&embedding.forward(&codes.i(group_idx)?)?)?;
        }
        sum.unsqueeze(1).map_err(Error::from)
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

    fn forward_dense(
        &self,
        x: &Tensor,
        start_pos: usize,
        kv: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_dense(&normed, start_pos, kv)?;
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
    rope: PredictorRope,
}

/// Positions whose RoPE tables [`PredictorRope`] keeps: a frame runs
/// positions 0 to 16 (the two-token prefill, then one per codebook).
const ROPE_CACHED_POSITIONS: usize = 32;

/// The predictor's RoPE tables. Every frame runs the same few positions, so
/// the cos/sin rows of the first [`ROPE_CACHED_POSITIONS`] positions are
/// built once per dtype and device by [`build_rope_window_full`] (host
/// angles, device cos/sin) and each call takes a view of its rows: each
/// position's row depends on that position alone, so the values are the
/// ones a per-call build gives, without its upload and six kernels for q
/// and for k in every layer at every codebook.
struct PredictorRope {
    inv_freq: Vec<f32>,
    tables: std::sync::Mutex<Vec<RopeTable>>,
}

struct RopeTable {
    dtype: DType,
    device: candle_core::DeviceLocation,
    cos: Tensor,
    sin: Tensor,
}

impl PredictorRope {
    fn new(inv_freq: Vec<f32>) -> Self {
        Self {
            inv_freq,
            tables: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// cos/sin `[seq_len, head_dim]` for positions `start_pos..`.
    fn window(
        &self,
        seq_len: usize,
        start_pos: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        if start_pos + seq_len > ROPE_CACHED_POSITIONS {
            return build_rope_window_full(seq_len, start_pos, &self.inv_freq, device, dtype);
        }
        let mut tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        let location = device.location();
        let index = match tables
            .iter()
            .position(|table| table.dtype == dtype && table.device == location)
        {
            Some(index) => index,
            None => {
                let (cos, sin) = build_rope_window_full(
                    ROPE_CACHED_POSITIONS,
                    0,
                    &self.inv_freq,
                    device,
                    dtype,
                )?;
                tables.push(RopeTable {
                    dtype,
                    device: location,
                    cos,
                    sin,
                });
                tables.len() - 1
            }
        };
        let table = &tables[index];
        Ok((
            table.cos.narrow(0, start_pos, seq_len)?,
            table.sin.narrow(0, start_pos, seq_len)?,
        ))
    }
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
            rope: PredictorRope::new(build_rope_inv_freq(head_dim, cfg.rope_theta)),
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

        let (cos, sin) = self
            .rope
            .window(seq_len, start_pos, x.device(), x.dtype())?;

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

    /// [`Self::forward_physical_batch`] over a dense per-row cache instead
    /// of paged KV: `x` is `[rows, seq, hidden]` at positions
    /// `start_pos..`, and `kv` holds the rows' earlier keys and values (F32
    /// `[rows, kv_heads, positions, head_dim]`) and gains this call's. Query
    /// head `h` reads key head `h / group`, as the paged kernels and the
    /// reference's `repeat_kv` do; scores, the causal mask of a multi-token
    /// call and the softmax run in F32, as the paged kernels run them.
    fn forward_dense(
        &self,
        x: &Tensor,
        start_pos: usize,
        kv: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (rows, seq, _) = x.dims3()?;
        let (heads, kv_heads, head_dim) = (self.num_heads, self.num_kv_heads, self.head_dim);
        if kv_heads == 0 || heads % kv_heads != 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS predictor has {heads} query heads for {kv_heads} key heads"
            )));
        }
        let q = self
            .q_proj
            .forward(x)?
            .reshape((rows, seq, heads, head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((rows, seq, kv_heads, head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((rows, seq, kv_heads, head_dim))?;
        let q = self.apply_qk_norm(q, heads, seq, &self.q_norm)?;
        let k = self.apply_qk_norm(k, kv_heads, seq, &self.k_norm)?;
        let q = self.apply_rope(q, start_pos)?;
        let k = self.apply_rope(k, start_pos)?;
        let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let (keys, values) = match kv.take() {
            Some((keys, values)) => (
                Tensor::cat(&[&keys, &k], 2)?,
                Tensor::cat(&[&values, &v], 2)?,
            ),
            None => (k, v),
        };
        let context = keys.dim(2)?;
        if context != start_pos + seq {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS dense predictor cache holds {context} positions at {start_pos}+{seq}"
            )));
        }
        // [rows, kv_heads, group * seq, head_dim]: query head h = kv * group
        // + g, row g * seq + s of its key head's block.
        let group = heads / kv_heads;
        let queries = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((rows, kv_heads, group * seq, head_dim))?
            .to_dtype(DType::F32)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut scores = (queries.matmul(&keys.t()?)? * scale)?;
        if seq > 1 {
            let mask = (0..group * seq)
                .flat_map(|row| {
                    let position = start_pos + row % seq;
                    (0..context).map(move |key| {
                        if key > position {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect::<Vec<_>>();
            let mask = Tensor::from_vec(mask, (group * seq, context), x.device())?;
            scores = scores.broadcast_add(&mask)?;
        }
        let out = ops::softmax_last_dim(&scores)?
            .matmul(&values)?
            .to_dtype(x.dtype())?
            .reshape((rows, heads, seq, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((rows, seq, heads * head_dim))?;
        *kv = Some((keys, values));
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
    if let Some(codes) = select_resident_top_k_rows(&logits, selections)? {
        return Ok(codes);
    }
    let codes = selections
        .iter_mut()
        .enumerate()
        .map(|(row, selection)| selection.select(&logits.i(row)?))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&codes, 0).map_err(Error::from)
}

/// Every row's code in one device pass when all rows sample the sub-talker
/// way (top-k with top-p 1) with the same `top_k` and temperature, which
/// every request with default sub-talker settings does: each row's code is
/// the one [`CodeSelection::select`] picks for it alone, but the 15
/// codebooks of a frame no longer run a sampling chain and a noise upload per
/// row. `None` sends the rows through [`CodeSelection::select`] one by one.
fn select_resident_top_k_rows(
    logits: &Tensor,
    selections: &mut [CodeSelection<'_>],
) -> Result<Option<Tensor>> {
    let shared = |selection: &CodeSelection<'_>| match selection {
        CodeSelection::Sample { params, .. } if !params.is_greedy() && params.top_p >= 1.0 => {
            Some((params.top_k, params.temperature.to_bits()))
        }
        _ => None,
    };
    let Some(first) = selections.first().and_then(shared) else {
        return Ok(None);
    };
    if selections.len() < 2 || selections.iter().any(|selection| shared(selection) != Some(first)) {
        return Ok(None);
    }
    let mut draws = selections
        .iter_mut()
        .map(|selection| match selection {
            CodeSelection::Sample { draw, .. } => &mut **draw,
            CodeSelection::Greedy => unreachable!("every row samples"),
        })
        .collect::<Vec<&mut (dyn FnMut() -> f32 + '_)>>();
    resident_top_k_sample_rows(logits, first.0, f32::from_bits(first.1), &mut draws)
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
            rope: PredictorRope::new(build_rope_inv_freq(2, cfg.rope_theta)),
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
            graphs: PredictorGraphs::off("test predictor"),
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
            .generate_acoustic_code_rows_physical_batch(
                &talker,
                &semantic,
                &mut caches,
                &mut [CodeSelection::Greedy, CodeSelection::Greedy],
            )
            .unwrap();

        assert_eq!(batch_codes.rows, vec![scalar_codes_a, scalar_codes_b]);
        // The device codes are the same codes, one row per codebook.
        assert_eq!(
            batch_codes.by_group.t().unwrap().to_vec2::<u32>().unwrap(),
            batch_codes.rows
        );
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

    /// Unstructured weights: xorshift values in [-0.5, 0.5).
    fn xorshift_values(state: &mut u64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn random_tensor(state: &mut u64, shape: &[usize], device: &Device) -> Tensor {
        let len = shape.iter().product();
        Tensor::from_vec(xorshift_values(state, len), shape, device).unwrap()
    }

    /// A predictor of xorshift weights with grouped query heads (4 query
    /// heads on 2 key heads of 4 dims), two layers, 4 codebooks of 16 codes
    /// and an `input_width` input (projected to the hidden 8 when wider).
    fn random_predictor(input_width: usize, device: &Device) -> CodePredictor {
        random_predictor_sized(input_width, (8, 4, 2, 4, 16, 4, 16), device)
    }

    /// [`random_predictor`] of other sizes: `(hidden, query heads, key
    /// heads, head dim, codes, codebooks, MLP width)`.
    fn random_predictor_sized(
        input_width: usize,
        (hidden, heads, kv_heads, head_dim, vocab, groups, mlp): (
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        ),
        device: &Device,
    ) -> CodePredictor {
        let mut state = 0x2545_F491_4F6C_DD1Du64 ^ input_width as u64;
        let mut linear = |output: usize, input: usize| -> ProjectionLinear {
            Linear::new(random_tensor(&mut state, &[output, input], device), None).into()
        };
        let mut layers = Vec::new();
        for _ in 0..2 {
            layers.push(Layer {
                input_layernorm: RmsNorm::new(
                    Tensor::ones(hidden, DType::F32, device).unwrap(),
                    1e-5,
                ),
                self_attn: Attention {
                    q_proj: linear(heads * head_dim, hidden),
                    k_proj: linear(kv_heads * head_dim, hidden),
                    v_proj: linear(kv_heads * head_dim, hidden),
                    o_proj: linear(hidden, heads * head_dim),
                    q_norm: RmsNorm::new(Tensor::ones(head_dim, DType::F32, device).unwrap(), 1e-5),
                    k_norm: RmsNorm::new(Tensor::ones(head_dim, DType::F32, device).unwrap(), 1e-5),
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim,
                    rope: PredictorRope::new(build_rope_inv_freq(head_dim, 10_000.0)),
                },
                post_attention_layernorm: RmsNorm::new(
                    Tensor::ones(hidden, DType::F32, device).unwrap(),
                    1e-5,
                ),
                mlp: Mlp {
                    gate_proj: linear(mlp, hidden),
                    up_proj: linear(mlp, hidden),
                    down_proj: linear(hidden, mlp),
                },
            });
        }
        let lm_heads = (0..groups).map(|_| linear(vocab, hidden)).collect();
        let small_to_mtp_projection = (input_width != hidden).then(|| {
            Linear::new(
                random_tensor(&mut state, &[hidden, input_width], device),
                None,
            )
        });
        let codec_embeddings = (0..groups)
            .map(|_| {
                CodecEmbedding::Dense(Embedding::new(
                    random_tensor(&mut state, &[vocab, input_width], device),
                    input_width,
                ))
            })
            .collect();
        let mut model = tiny_predictor_with_input_width(input_width, device);
        model.cfg.hidden_size = hidden;
        model.cfg.num_attention_heads = heads;
        model.cfg.num_key_value_heads = kv_heads;
        model.cfg.head_dim = head_dim;
        model.cfg.num_hidden_layers = 2;
        model.cfg.vocab_size = vocab;
        model.cfg.num_code_groups = groups;
        model.codec_embeddings = codec_embeddings;
        model.small_to_mtp_projection = small_to_mtp_projection;
        model.layers = layers;
        model.norm = RmsNorm::new(Tensor::ones(hidden, DType::F32, device).unwrap(), 1e-5);
        model.lm_heads = lm_heads;
        model.num_code_groups = groups;
        model
    }

    /// A CPU arena for [`random_predictor`]: two layers of 2 key heads of 4
    /// dims, pages of 2 tokens.
    fn random_arena(instance: u64) -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        random_arena_sized(instance, 4, 64)
    }

    /// [`random_arena`] with key heads of `head_dim` dims and `pages` pages.
    fn random_arena_sized(
        instance: u64,
        head_dim: usize,
        pages: usize,
    ) -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        let bindings = (0..2)
            .map(|layer| KvLayerBinding {
                model_layer: layer,
                physical_layer: layer,
            })
            .collect::<Vec<_>>();
        let arena = CpuKvArena::new(KvArenaConfig {
            id: KvArenaId {
                model_instance: ModelInstanceId::new(instance),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            page_tokens: 2,
            capacity_pages: pages as u32,
            growth: None,
            dtype: DType::F32,
            layers: bindings
                .iter()
                .map(|&binding| KvLayerConfig {
                    binding,
                    num_kv_heads: 2,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                })
                .collect(),
        })
        .unwrap();
        (Arc::new(arena), bindings)
    }

    /// A predictor wide enough that the CPU's matmul rounds a row
    /// differently at different row counts (hidden 64, MLP 96, 4 query
    /// heads on 2 key heads of 16 dims, 4 codebooks of 48 codes), and its
    /// arena (`rows` invocation workspaces).
    fn wide_predictor(
        rows: usize,
        instance: u64,
        device: &Device,
    ) -> (CodePredictor, Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        let model = random_predictor_sized(64, (64, 4, 2, 16, 48, 4, 96), device);
        let (arena, bindings) = random_arena_sized(instance, 16, rows * 8);
        (model, arena, bindings)
    }

    /// D3: on CUDA's path (forced here) a frame the graphs do not take runs
    /// the graphs' dense frame eagerly at 32 rows: a request's logits at
    /// every codebook are bit for bit the same alone and beside 1 to 31
    /// other requests, wherever it sits, and a lone request takes the same
    /// frame through the single-row entry.
    #[test]
    fn padded_eager_frame_logits_are_the_same_alone_and_in_any_batch() {
        let _force = decode_rows::ForceRowInvariant::new(true);
        let device = Device::Cpu;
        let (model, _, _) = wide_predictor(1, 0, &device);
        let requests = 32;
        let inputs = (0..requests)
            .map(|request| {
                let mut state = 0xA11CE_u64 + request as u64;
                (
                    random_tensor(&mut state, &[1, 1, 64], &device),
                    random_tensor(&mut state, &[1, 1, 64], &device),
                )
            })
            .collect::<Vec<_>>();
        // Each request's logits bits per codebook, in `order`'s frame.
        let frame = |order: &[usize]| {
            let talker =
                Tensor::cat(&order.iter().map(|&r| &inputs[r].0).collect::<Vec<_>>(), 0).unwrap();
            let semantic =
                Tensor::cat(&order.iter().map(|&r| &inputs[r].1).collect::<Vec<_>>(), 0).unwrap();
            let mut seen = vec![Vec::new(); order.len()];
            model
                .padded_dense_frame(&talker, &semantic, &mut |logits, _| {
                    for (row, values) in logits.to_vec2::<f32>().unwrap().into_iter().enumerate() {
                        seen[row].push(values.into_iter().map(f32::to_bits).collect::<Vec<_>>());
                    }
                    argmax_device(logits)
                })
                .unwrap();
            let mut by_request = vec![Vec::new(); requests];
            for (slot, &request) in order.iter().enumerate() {
                by_request[request] = std::mem::take(&mut seen[slot]);
            }
            by_request
        };
        let alone = (0..requests)
            .map(|request| frame(&[request]).swap_remove(request))
            .collect::<Vec<_>>();
        for order in [
            vec![3, 0],
            vec![7, 1, 30, 12, 5],
            (0..requests).rev().collect::<Vec<_>>(),
        ] {
            let batched = frame(&order);
            for &request in &order {
                assert!(
                    batched[request] == alone[request],
                    "request {request} differs in a frame of {}",
                    order.len()
                );
            }
        }
    }

    /// D3: on CUDA's path (forced here) a request's codes, and where its
    /// draw source ends, are the same alone and beside requests that pick
    /// their codes otherwise (greedy, the sub-talker draw, another top-k
    /// and temperature, a top-p draw on the host), through the eager frame
    /// the graphs leave such mixed frames to.
    #[test]
    fn padded_eager_frame_codes_are_the_same_beside_other_selections() {
        let _force = decode_rows::ForceRowInvariant::new(true);
        let device = Device::Cpu;
        let settings = [
            None,
            Some(SubtalkerSampling::default()),
            Some(SubtalkerSampling {
                do_sample: true,
                temperature: 0.7,
                top_k: 7,
                top_p: 1.0,
            }),
            Some(SubtalkerSampling {
                do_sample: true,
                temperature: 1.1,
                top_k: 20,
                top_p: 0.8,
            }),
        ];
        let requests = 8;
        let inputs = (0..requests)
            .map(|request| {
                let mut state = 0xF00D_u64 + request as u64;
                (
                    random_tensor(&mut state, &[1, 1, 64], &device),
                    random_tensor(&mut state, &[1, 1, 64], &device),
                )
            })
            .collect::<Vec<_>>();
        let run = |order: &[usize], instance: u64| {
            let (model, arena, bindings) = wide_predictor(order.len(), instance, &device);
            let mut caches = (0..order.len())
                .map(|slot| test_cache(arena.clone(), &bindings, slot as u32 * 8, 0))
                .collect::<Vec<_>>();
            let mut rngs = order
                .iter()
                .map(|&request| super::super::SimpleRng::seeded(request as u64 + 5))
                .collect::<Vec<_>>();
            let mut draws = rngs
                .iter_mut()
                .map(|rng| move || rng.next_f32())
                .collect::<Vec<_>>();
            let mut selections = order
                .iter()
                .zip(draws.iter_mut())
                .map(
                    |(&request, draw)| match settings[request % settings.len()] {
                        Some(params) => CodeSelection::Sample { params, draw },
                        None => CodeSelection::Greedy,
                    },
                )
                .collect::<Vec<_>>();
            let talker =
                Tensor::cat(&order.iter().map(|&r| &inputs[r].0).collect::<Vec<_>>(), 0).unwrap();
            let semantic =
                Tensor::cat(&order.iter().map(|&r| &inputs[r].1).collect::<Vec<_>>(), 0).unwrap();
            let mut refs = caches.iter_mut().collect::<Vec<_>>();
            let codes = if order.len() == 1 {
                vec![model
                    .generate_acoustic_codes_physical(
                        &talker,
                        &semantic,
                        refs[0],
                        &mut selections[0],
                    )
                    .unwrap()]
            } else {
                model
                    .generate_acoustic_code_rows_physical_batch(
                        &talker,
                        &semantic,
                        &mut refs,
                        &mut selections,
                    )
                    .unwrap()
                    .rows
            };
            drop(selections);
            drop(draws);
            let mut by_request = vec![(Vec::new(), 0u64); requests];
            for ((&request, codes), rng) in order.iter().zip(codes).zip(&rngs) {
                by_request[request] = (codes, rng.state);
            }
            by_request
        };
        let alone = (0..requests)
            .map(|request| run(&[request], 60 + request as u64).swap_remove(request))
            .collect::<Vec<_>>();
        for (index, order) in [vec![1, 0, 3], (0..requests).rev().collect::<Vec<_>>()]
            .into_iter()
            .enumerate()
        {
            let batched = run(&order, 70 + index as u64);
            for &request in &order {
                assert_eq!(batched[request], alone[request], "request {request}");
            }
        }
    }

    /// The dense layers (the graphs' body) give the paged layers' hidden
    /// states, every position of the two-token prefill (so the causal mask)
    /// and the next codebook positions, within F32 rounding.
    #[test]
    fn dense_layers_match_the_paged_layers() {
        let device = Device::Cpu;
        let model = random_predictor(8, &device);
        let (arena, bindings) = random_arena(741);
        let mut cache = test_cache(arena, &bindings, 0, 0);
        let mut state = 0x51ED_2701u64;
        let mut kv = vec![None, None];
        for (start_pos, seq) in [(0, 2), (2, 1), (3, 1)] {
            let x = random_tensor(&mut state, &[1, seq, 8], &device);
            let (paged, prepared) = model
                .forward_physical_hidden_uncommitted(&x, start_pos, &mut cache)
                .unwrap();
            cache.commit_prepared(prepared).unwrap();
            let dense = model.dense_forward(&x, start_pos, &mut kv).unwrap();
            let paged = paged.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let dense = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (index, (paged, dense)) in paged.iter().zip(&dense).enumerate() {
                let error = (paged - dense).abs();
                assert!(
                    error <= 1e-5 * paged.abs().max(1.0),
                    "position {start_pos}+{seq} element {index}: paged {paged} dense {dense}"
                );
            }
        }
    }

    /// The dense frame picks the eager paged frame's codes, greedy and with
    /// a sub-talker draw whose noise is drawn before the frame, with spare
    /// zero rows up to the graph rows, and leaves every row's draw source
    /// where the eager frame leaves it: three rows, xorshift weights and
    /// inputs, with and without the input projection.
    #[test]
    fn dense_frame_picks_the_eager_frames_codes() {
        let device = Device::Cpu;
        let rows = 3;
        for input_width in [8, 12] {
            let model = random_predictor(input_width, &device);
            let mut state = 0x9E37_79B9u64 ^ input_width as u64;
            let talker = random_tensor(&mut state, &[rows, 1, input_width], &device);
            let semantic = random_tensor(&mut state, &[rows, 1, input_width], &device);
            let pad = |tensor: &Tensor| {
                tensor
                    .pad_with_zeros(0, 0, PREDICTOR_GRAPH_ROWS - rows)
                    .unwrap()
            };
            let params = SubtalkerSampling {
                do_sample: true,
                temperature: 0.9,
                top_k: 5,
                top_p: 1.0,
            };
            for sampled in [false, true] {
                let (arena, bindings) = random_arena(742 + input_width as u64);
                let mut caches = (0..rows)
                    .map(|row| test_cache(arena.clone(), &bindings, row as u32 * 8, 0))
                    .collect::<Vec<_>>();
                let mut eager_draws = (0..rows)
                    .map(|row| super::super::SimpleRng::seeded(row as u64 + 17))
                    .collect::<Vec<_>>();
                let eager = {
                    let mut draws = eager_draws
                        .iter_mut()
                        .map(|rng| move || rng.next_f32())
                        .collect::<Vec<_>>();
                    let mut selections = draws
                        .iter_mut()
                        .map(|draw| {
                            if sampled {
                                CodeSelection::Sample { params, draw }
                            } else {
                                CodeSelection::Greedy
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut refs = caches.iter_mut().collect::<Vec<_>>();
                    model
                        .generate_acoustic_codes_physical_batch(
                            &talker,
                            &semantic,
                            &mut refs,
                            &mut selections,
                        )
                        .unwrap()
                };
                let mut dense_draws = (0..rows)
                    .map(|row| super::super::SimpleRng::seeded(row as u64 + 17))
                    .collect::<Vec<_>>();
                let (selection, noise) = {
                    let mut draws = dense_draws
                        .iter_mut()
                        .map(|rng| move || rng.next_f32())
                        .collect::<Vec<_>>();
                    let mut selections = draws
                        .iter_mut()
                        .map(|draw| {
                            if sampled {
                                CodeSelection::Sample { params, draw }
                            } else {
                                CodeSelection::Greedy
                            }
                        })
                        .collect::<Vec<_>>();
                    let selection = FrameSelection::of(&selections).unwrap();
                    let noise = if sampled {
                        frame_noise(&mut selections, 4, 5, &device).unwrap()
                    } else {
                        Tensor::zeros(1, DType::F32, &device).unwrap()
                    };
                    (selection, noise)
                };
                let ids = Tensor::arange(0u32, 16, &device).unwrap();
                let codes = model
                    .dense_frame_codes(&pad(&talker), &pad(&semantic), selection, &noise, &ids)
                    .unwrap();
                let dense = codes
                    .narrow(1, 0, rows)
                    .unwrap()
                    .t()
                    .unwrap()
                    .to_vec2::<u32>()
                    .unwrap();
                assert_eq!(dense, eager, "width {input_width}, sampled {sampled}");
                for (eager, dense) in eager_draws.iter().zip(&dense_draws) {
                    assert_eq!(eager.state, dense.state, "width {input_width}");
                }
                if !sampled {
                    // Unpadded, the dense frame picks the same codes.
                    let unpadded = model
                        .dense_frame_codes(&talker, &semantic, selection, &noise, &ids)
                        .unwrap()
                        .t()
                        .unwrap()
                        .to_vec2::<u32>()
                        .unwrap();
                    assert_eq!(unpadded, eager, "width {input_width}");
                }
            }
        }
    }

    /// A graphed frame needs every row to pick its codes the same resident
    /// way: all greedy, or one sub-talker top-k and temperature at top-p 1.
    #[test]
    fn graphed_frames_need_one_resident_selection() {
        let sample = |top_k: usize, temperature: f32, top_p: f32| SubtalkerSampling {
            do_sample: true,
            temperature,
            top_k,
            top_p,
        };
        let of = |params: &[Option<SubtalkerSampling>]| {
            let mut draws = params.iter().map(|_| || 0.5f32).collect::<Vec<_>>();
            let selections = params
                .iter()
                .zip(draws.iter_mut())
                .map(|(params, draw)| match params {
                    Some(params) => CodeSelection::Sample {
                        params: *params,
                        draw,
                    },
                    None => CodeSelection::Greedy,
                })
                .collect::<Vec<_>>();
            FrameSelection::of(&selections)
        };
        assert_eq!(of(&[]), None);
        assert_eq!(of(&[None, None]), Some(FrameSelection::Greedy));
        assert_eq!(
            of(&[None, Some(SubtalkerSampling::greedy())]),
            Some(FrameSelection::Greedy)
        );
        assert_eq!(
            of(&[Some(sample(50, 0.9, 1.0))]),
            Some(FrameSelection::TopK {
                top_k: 50,
                temperature: 0.9
            })
        );
        for apart in [
            vec![None, Some(sample(50, 0.9, 1.0))],
            vec![Some(sample(50, 0.9, 1.0)), Some(sample(40, 0.9, 1.0))],
            vec![Some(sample(50, 0.9, 1.0)), Some(sample(50, 0.8, 1.0))],
            vec![Some(sample(50, 0.9, 0.95))],
            vec![Some(sample(0, 0.9, 1.0))],
            vec![Some(sample(DEVICE_TOP_K_LIMIT + 1, 0.9, 1.0))],
        ] {
            assert_eq!(of(&apart), None, "{apart:?}");
        }
    }

    /// Off CUDA the graphs never resolve and price nothing; the kill switch
    /// turns their pricing off on CUDA and names itself.
    #[test]
    fn predictor_graphs_run_on_cuda_only_and_obey_the_kill_switch() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os(QWEN3_TTS_PREDICTOR_GRAPHS_ENV);
        let device = Device::Cpu;
        std::env::remove_var(QWEN3_TTS_PREDICTOR_GRAPHS_ENV);
        assert_eq!(qwen3_tts_predictor_graph_device_bytes(BackendKind::Cpu), 0);
        assert_eq!(
            qwen3_tts_predictor_graph_device_bytes(BackendKind::Metal),
            0
        );
        assert_eq!(
            qwen3_tts_predictor_graph_device_bytes(BackendKind::Cuda),
            128 * MIB
        );
        let mut model = tiny_predictor(&device);
        model.resolve_graphs(DType::F32);
        assert_eq!(
            model.graphs_diagnostics()["off"],
            "shape graphs run on CUDA only"
        );
        std::env::set_var(QWEN3_TTS_PREDICTOR_GRAPHS_ENV, "0");
        assert_eq!(qwen3_tts_predictor_graph_device_bytes(BackendKind::Cuda), 0);
        model.resolve_graphs(DType::F32);
        let off = model.graphs_diagnostics()["off"]
            .as_str()
            .unwrap()
            .to_owned();
        match previous {
            Some(value) => std::env::set_var(QWEN3_TTS_PREDICTOR_GRAPHS_ENV, value),
            None => std::env::remove_var(QWEN3_TTS_PREDICTOR_GRAPHS_ENV),
        }
        assert!(off.contains(QWEN3_TTS_PREDICTOR_GRAPHS_ENV), "{off}");
    }

    /// The cached RoPE rows are the per-call tables bit for bit, for every
    /// window a frame runs and beyond the cache, in each compute dtype.
    #[test]
    fn cached_rope_windows_equal_per_call_tables() {
        let device = Device::Cpu;
        let inv_freq = build_rope_inv_freq(128, 1_000_000.0);
        let rope = PredictorRope::new(inv_freq.clone());
        let bits = |tensor: &Tensor| {
            tensor
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            for (seq_len, start_pos) in [(2, 0), (1, 2), (1, 9), (1, 16), (3, 29), (2, 31)] {
                let (cos, sin) = rope.window(seq_len, start_pos, &device, dtype).unwrap();
                let (expected_cos, expected_sin) =
                    build_rope_window_full(seq_len, start_pos, &inv_freq, &device, dtype).unwrap();
                let window = format!("{dtype:?} {start_pos}+{seq_len}");
                assert_eq!(cos.dtype(), dtype);
                assert_eq!(cos.dims(), expected_cos.dims());
                assert_eq!(bits(&cos), bits(&expected_cos), "{window}");
                assert_eq!(bits(&sin), bits(&expected_sin), "{window}");
            }
        }
        assert_eq!(rope.tables.lock().unwrap().len(), 3);
    }

    /// A step's acoustic embedding sums gathered for all rows at once equal
    /// each row's own sum bit for bit (the same codebook order), over 15
    /// xorshift tables of mixed magnitudes, dense F32 and widened F16, so a
    /// different order would round differently.
    #[test]
    fn batched_acoustic_embedding_sums_equal_each_rows_own() {
        let device = Device::Cpu;
        let mut state = 0x51ED_2701_9E37_79B9u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (groups, vocab, width, rows) = (15, 8, 4, 5);
        for f16 in [false, true] {
            let mut model = tiny_predictor(&device);
            model.codec_embeddings = (0..groups)
                .map(|_| {
                    let values = (0..vocab * width)
                        .map(|_| {
                            let unit = (next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                            unit * 2f32.powi((next() % 12) as i32 - 6)
                        })
                        .collect::<Vec<_>>();
                    let table = Tensor::from_vec(values, (vocab, width), &device).unwrap();
                    CodecEmbedding::new(Embedding::new(table, width), f16.then_some(&device))
                        .unwrap()
                })
                .collect();
            let codes = (0..rows)
                .map(|_| {
                    (0..groups)
                        .map(|_| (next() % vocab as u64) as u32)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let by_group = (0..groups)
                .flat_map(|group| codes.iter().map(move |row| row[group]))
                .collect::<Vec<_>>();
            let by_group = Tensor::from_vec(by_group, (groups, rows), &device).unwrap();
            let batched = model.acoustic_embeddings_sum_rows(&by_group).unwrap();
            assert_eq!(batched.dims(), &[rows, 1, width]);
            for (row, row_codes) in codes.iter().enumerate() {
                let bits = |tensor: Tensor| {
                    tensor
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(batched.i(row).unwrap()),
                    bits(model.get_acoustic_embeddings_sum(row_codes).unwrap()),
                    "row {row}, f16 tables {f16}"
                );
            }
        }
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
