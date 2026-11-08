//! Fast codebook decoder for Fish S2 DualAR generation.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

pub use super::sampling::FishS2Sampler;
use super::sampling::{
    resident_rows_codes, sample_resident, sample_resident_rows, FishS2FastPolicy,
    FishS2ResidentDraws, FishS2ResidentRowDraws, FishS2SamplingDistribution,
};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};
use crate::models::architectures::fish_s2::config::FishS2Config;
use crate::models::architectures::fish_s2::rotary::FishS2RotaryCache;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::sampling::gumbel_noise;
use crate::models::shared::telemetry;

/// Kill switch for the fast AR's CUDA frame graphs (a boolean; on when
/// unset). Load pricing and the decoder read it.
pub(crate) const FISH_S2_FAST_GRAPHS_ENV: &str = "IZWI_FISH_S2_FAST_GRAPHS";

fn fast_graphs_requested() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(FISH_S2_FAST_GRAPHS_ENV, true));
    ON.get()
}

/// The rows every graphed frame runs. Spare rows are zero rows, so every
/// kernel of a frame runs one shape however many requests share the step (a
/// row's codes never depend on the rows beside it), and one graph per
/// sampling policy serves every width.
pub(crate) const FAST_GRAPH_ROWS: usize = 32;

const MIB: u64 = 1024 * 1024;

/// What the fast-AR graphs may hold: one graph per input dtype and sampling
/// policy. A 32-row frame's working set is a few MiB (nine depths of
/// `[32, 4096]` F32 logits and their chunked sorts, `[32, 9728]` MLP
/// activations, ten positions of dense KV per layer); the retained buffers
/// are its inputs (`[32, 1, 2560]` hiddens, the noise, the top-p matrix,
/// the column ids) and its codes.
const FAST_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 4,
    retained_bytes: 16 * MIB,
    pool_bytes: 112 * MIB,
};

/// Device bytes a load on `backend` prices for the fast-AR graphs, which the
/// load's self-check allocates: zero off CUDA or with the graphs switched
/// off.
pub(crate) fn fish_s2_fast_graph_device_bytes(backend: BackendKind) -> u64 {
    if backend == BackendKind::Cuda && fast_graphs_requested() {
        FAST_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// A frame graph's key: its rows' sampling policy and its inputs' dtype.
type FastGraphKey = (FishS2FastPolicy, DType);

/// The fast AR's frame graphs ([`FishS2FastDecoder::resolve_graphs`]), or
/// why there are none, and how many frames ran through them.
struct FastGraphs {
    graphs: std::result::Result<ShapeGraphs<FastGraphKey>, String>,
    graphed_frames: AtomicU64,
    /// Frames the graphs could not take: a row whose policy only the host
    /// serves.
    eager_frames: AtomicU64,
    /// The codebook's column ids `0..codebook_size` (U32), read by the top-k
    /// from the graph's own copy, never from a cache that could free them
    /// under a captured graph.
    column_ids: std::sync::OnceLock<Tensor>,
}

impl FastGraphs {
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

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2FastConfig {
    pub input_hidden_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishS2GeneratedFrame {
    pub semantic_token_id: u32,
    pub codebooks: Vec<u32>,
}

pub struct FishS2FastDecoder {
    cfg: FishS2FastConfig,
    project_in: FishS2FastProjectIn,
    embeddings: Embedding,
    layers: Vec<FishS2FastLayer>,
    norm: RmsNorm,
    output: Linear,
    graphs: FastGraphs,
}

enum FishS2FastProjectIn {
    Identity,
    Linear(Linear),
}

struct FishS2FastLayer {
    input_layernorm: RmsNorm,
    self_attn: FishS2FastAttention,
    post_attention_layernorm: RmsNorm,
    mlp: FishS2FastMlp,
}

struct FishS2FastAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: FishS2RotaryCache,
}

struct FishS2FastMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FishS2FastConfig {
    pub fn from_config(config: &FishS2Config) -> Result<Self> {
        let audio = &config.audio_decoder_config;
        let text = &config.text_config;
        let head_dim = audio
            .head_dim
            .unwrap_or_else(|| audio.hidden_size / audio.num_attention_heads);
        let intermediate_size = audio
            .intermediate_size
            .unwrap_or_else(|| audio.hidden_size * 3);
        Ok(Self {
            input_hidden_size: text.hidden_size,
            hidden_size: audio.hidden_size,
            intermediate_size,
            num_hidden_layers: audio.num_hidden_layers,
            num_attention_heads: audio.num_attention_heads,
            num_key_value_heads: audio.num_key_value_heads,
            head_dim,
            codebook_size: config.codebook_size,
            num_codebooks: config.num_codebooks,
            semantic_start_token_id: config.semantic_start_token_id,
            semantic_end_token_id: config.semantic_end_token_id,
            rope_theta: audio.rope_theta.unwrap_or(1_000_000.0),
            rms_norm_eps: audio.rms_norm_eps.unwrap_or(1e-6),
        })
    }

    fn q_size(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_size(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

impl FishS2FastDecoder {
    pub fn load(cfg: FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
        let rotary = FishS2RotaryCache::new(
            cfg.num_codebooks,
            cfg.head_dim,
            cfg.rope_theta,
            DType::BF16,
            vb.device(),
        )?;
        let project_in = if cfg.input_hidden_size == cfg.hidden_size {
            FishS2FastProjectIn::Identity
        } else {
            let prefix = if vb.contains_tensor("fast_project_in.weight") {
                "fast_project_in"
            } else if vb.contains_tensor("project_in.weight") {
                "project_in"
            } else {
                "fast_project_in"
            };
            FishS2FastProjectIn::Linear(candle_nn::linear(
                cfg.input_hidden_size,
                cfg.hidden_size,
                vb.pp(prefix),
            )?)
        };
        let embeddings =
            candle_nn::embedding(cfg.codebook_size, cfg.hidden_size, vb.pp("fast_embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(FishS2FastLayer::load(
                &cfg,
                &rotary,
                vb.pp(format!("fast_layers.{idx}")),
            )?);
        }
        let norm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["fast_norm", "norm"],
        )?;
        let output =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.codebook_size, vb.pp("fast_output"))?;
        Ok(Self {
            cfg,
            project_in,
            embeddings,
            layers,
            norm,
            output,
            graphs: FastGraphs::off("not resolved"),
        })
    }

    pub fn config(&self) -> &FishS2FastConfig {
        &self.cfg
    }

    /// Persistent RoPE table bytes, shared by every fast layer.
    pub fn rotary_cache_bytes(&self) -> u64 {
        self.layers
            .first()
            .map(|layer| layer.self_attn.rotary.storage_bytes())
            .unwrap_or(0)
    }

    pub fn project_slow_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.project_in {
            FishS2FastProjectIn::Identity => Ok(hidden.clone()),
            FishS2FastProjectIn::Linear(linear) => linear.forward(hidden).map_err(Error::from),
        }
    }

    pub fn codebook_embedding(&self, code: u32) -> Result<Tensor> {
        if code as usize >= self.cfg.codebook_size {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast code {code} exceeds codebook size {}",
                self.cfg.codebook_size
            )));
        }
        let ids = Tensor::from_vec(vec![code], (1, 1), self.embeddings.embeddings().device())?;
        self.embeddings.forward(&ids).map_err(Error::from)
    }

    pub fn forward_step(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        self.forward_step_inner(x, input_pos, cache, true)
    }

    fn forward_step_inner(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        project_logits: bool,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, hidden_size) = x.dims3()?;
        if batch_size != 1 || sequence_len != 1 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast physical paging expects [1,1,{}], got {:?}",
                self.cfg.hidden_size,
                x.dims()
            )));
        }
        if input_pos >= self.cfg.num_codebooks || input_pos >= cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast codebook position {input_pos} exceeds model/cache capacity {}/{}",
                self.cfg.num_codebooks,
                cache.capacity_tokens()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        )?;
        let mut prepared = cache.prepare_append(input_pos, sequence_len)?;
        let mut hidden = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, input_pos, cache, &mut prepared, idx)?;
        }
        let logits = if project_logits {
            self.output.forward(&self.norm.forward(&hidden)?)?
        } else {
            hidden
        };
        cache.commit_prepared(prepared)?;
        Ok(logits)
    }

    fn forward_step_batch(
        &self,
        x: &Tensor,
        input_pos: usize,
        caches: &mut [&mut PhysicalPagedKvCache],
        project_logits: bool,
    ) -> Result<Tensor> {
        let (rows, tokens, hidden) = x.dims3()?;
        if rows == 0
            || rows != caches.len()
            || tokens != 1
            || hidden != self.cfg.hidden_size
            || input_pos >= self.cfg.num_codebooks
        {
            return Err(Error::InvalidInput(
                "Fish Fast AR batch shape or clock is invalid".into(),
            ));
        }
        for cache in caches.iter() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim,
            )?;
            if cache.context_len() != input_pos {
                return Err(Error::InvalidInput(
                    "Fish Fast AR batch clocks differ".into(),
                ));
            }
        }
        let mut batch = super::batch::FishPhysicalBatch::new(
            caches,
            &vec![1; rows],
            self.cfg.num_hidden_layers,
        )?;
        let execution = (|| {
            let mut hidden = x.clone();
            for (index, layer) in self.layers.iter().enumerate() {
                let norm = layer.input_layernorm.forward(&hidden)?;
                let attn = layer
                    .self_attn
                    .forward_batch(&norm, input_pos, caches[0], &mut batch, index)?;
                hidden = hidden.broadcast_add(&attn)?;
                hidden = hidden.broadcast_add(
                    &layer
                        .mlp
                        .forward(&layer.post_attention_layernorm.forward(&hidden)?)?,
                )?;
            }
            if project_logits {
                self.output
                    .forward(&self.norm.forward(&hidden)?)
                    .map_err(Error::from)
            } else {
                Ok(hidden)
            }
        })();
        batch.finish(caches, execution)
    }

    /// Depth steps remain sequential; every dense projection and attention step
    /// within a depth operates on the complete batch of independent requests.
    pub(crate) fn generate_frames_batch(
        &self,
        semantic_tokens: &[u32],
        slow_hidden: &Tensor,
        samplers: &mut [&mut FishS2Sampler],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Vec<FishS2GeneratedFrame>> {
        let resident = resident_fast_sampling(slow_hidden.device());
        self.generate_frames_batch_with(semantic_tokens, slow_hidden, samplers, caches, resident)
    }

    /// [`Self::generate_frames_batch`] with the fast-code sampling chosen
    /// explicitly: `resident` keeps codes on the device and reads the frame
    /// back once (falling back to the host when a sampler's policy needs it).
    fn generate_frames_batch_with(
        &self,
        semantic_tokens: &[u32],
        slow_hidden: &Tensor,
        samplers: &mut [&mut FishS2Sampler],
        caches: &mut [&mut PhysicalPagedKvCache],
        resident: bool,
    ) -> Result<Vec<FishS2GeneratedFrame>> {
        let rows = semantic_tokens.len();
        if rows == 0
            || rows != caches.len()
            || rows != samplers.len()
            || slow_hidden.dims() != [rows, 1, self.cfg.input_hidden_size]
        {
            return Err(Error::InvalidInput(
                "Fish Fast AR frame batch rows do not match".into(),
            ));
        }
        let semantic_codes = semantic_tokens
            .iter()
            .map(|&token| semantic_code_from_token_id_from_fast_config(&self.cfg, token))
            .collect::<Result<Vec<_>>>()?;
        for cache in caches.iter_mut() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim,
            )?;
            if cache.capacity_tokens() < self.cfg.num_codebooks {
                return Err(Error::InvalidInput(
                    "Fish Fast AR batch cache has insufficient capacity".into(),
                ));
            }
        }
        if resident {
            let run = |key: FastGraphKey, inputs: &[Tensor]| self.run_frame_graph(key, inputs);
            if let Some(frames) = self.graphed_frames(
                semantic_tokens,
                &semantic_codes,
                slow_hidden,
                samplers,
                &run,
            )? {
                for cache in caches.iter_mut() {
                    cache.reset_invocation()?;
                }
                return Ok(frames);
            }
        }
        for cache in caches.iter_mut() {
            cache.reset_invocation()?;
        }
        let hidden = self.project_slow_hidden(slow_hidden)?;
        self.forward_step_batch(&hidden, 0, caches, false)?;
        let mut frames = semantic_tokens
            .iter()
            .zip(&semantic_codes)
            .map(|(&semantic_token_id, &code)| FishS2GeneratedFrame {
                semantic_token_id,
                codebooks: vec![code],
            })
            .collect::<Vec<_>>();
        let ids = Tensor::from_slice(&semantic_codes, (rows, 1), slow_hidden.device())?;
        let mut current = self.embeddings.forward(&ids)?;
        let row_draws = self.resident_row_draws(samplers, slow_hidden.device(), resident)?;
        let draws = match row_draws {
            Some(_) => None,
            None => self.resident_draws(samplers, slow_hidden.device(), resident)?,
        };
        let mut resident = ResidentCodes::default();
        for depth in 1..self.cfg.num_codebooks {
            let logits = self.forward_step_batch(&current, depth, caches, true)?;
            let next_ids = if let Some(row_draws) = &row_draws {
                // Every row's code at this depth from one pass.
                resident.sample_rows(&logits.i((.., 0))?, row_draws, depth - 1)?
            } else if let Some(draws) = &draws {
                let mut codes = Vec::with_capacity(rows);
                for (row, (sampler, draws)) in samplers.iter().zip(draws).enumerate() {
                    codes.push(resident.sample(&logits.i((row, 0))?, sampler, draws, depth - 1)?);
                }
                Tensor::cat(&codes, 0)?
            } else {
                let mut codes = Vec::with_capacity(rows);
                for (sampler, frame) in samplers.iter_mut().zip(&mut frames) {
                    let row = codes.len();
                    let code = sample_logits(&logits.i((row, 0))?, sampler)?;
                    frame.codebooks.push(code);
                    codes.push(code);
                }
                Tensor::from_slice(&codes, rows, slow_hidden.device())?
            };
            if depth + 1 < self.cfg.num_codebooks {
                current = self.embeddings.forward(&next_ids.reshape((rows, 1))?)?;
            }
        }
        if row_draws.is_some() || draws.is_some() {
            // Codes were drawn depth by depth, rows within a depth.
            let codes = resident.read_back()?;
            for (index, code) in codes.into_iter().enumerate() {
                frames[index % rows].codebooks.push(code);
            }
        }
        Ok(frames)
    }

    /// The load-time precision probe for one frame from `slow_hidden`
    /// (`[1, 1, input_hidden]`) whose semantic code is `semantic_code`: every
    /// depth's logits, stacked, choosing each next code greedily on the
    /// device. Uses `cache` as a fresh invocation.
    pub(crate) fn numeric_probe(
        &self,
        slow_hidden: &Tensor,
        semantic_code: u32,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        cache.reset_invocation()?;
        let hidden = self.project_slow_hidden(slow_hidden)?;
        self.forward_step_inner(&hidden, 0, cache, false)?;
        let mut current = self.codebook_embedding(semantic_code)?;
        let mut logits = Vec::with_capacity(self.cfg.num_codebooks);
        for depth in 1..self.cfg.num_codebooks {
            let step = self.forward_step(&current, depth, cache)?;
            let code = step.i((0, 0))?.argmax_keepdim(0)?.reshape((1, 1))?;
            current = self.embeddings.forward(&code)?;
            logits.push(step);
        }
        Tensor::cat(&logits, 1).map_err(Error::from)
    }

    /// Per-row resident draws for one frame, or `None` for the host path
    /// (CPU, or a policy the device sampler does not serve).
    fn resident_draws(
        &self,
        samplers: &mut [&mut FishS2Sampler],
        device: &Device,
        resident: bool,
    ) -> Result<Option<Vec<FishS2ResidentDraws>>> {
        if !resident {
            return Ok(None);
        }
        let tokens = self.cfg.num_codebooks.saturating_sub(1);
        let mut draws = Vec::with_capacity(samplers.len());
        for sampler in samplers.iter_mut() {
            match sampler.resident_draws(tokens, device)? {
                Some(row) => draws.push(row),
                None => return Ok(None),
            }
        }
        Ok(Some(draws))
    }

    /// One frame's draws for every row at once, or `None` for the per-row
    /// draws (host path, differing policies, or
    /// `IZWI_FISH_S2_BATCHED_SAMPLING=false`).
    fn resident_row_draws(
        &self,
        samplers: &mut [&mut FishS2Sampler],
        device: &Device,
        resident: bool,
    ) -> Result<Option<FishS2ResidentRowDraws>> {
        if !resident || !super::fish_s2_batched_sampling_enabled() {
            return Ok(None);
        }
        FishS2ResidentRowDraws::new(samplers, self.cfg.num_codebooks.saturating_sub(1), device)
    }

    pub fn generate_frame(
        &self,
        semantic_token_id: u32,
        slow_hidden: &Tensor,
        sampler: &mut FishS2Sampler,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<FishS2GeneratedFrame> {
        let resident = resident_fast_sampling(slow_hidden.device());
        self.generate_frame_with(semantic_token_id, slow_hidden, sampler, cache, resident)
    }

    fn generate_frame_with(
        &self,
        semantic_token_id: u32,
        slow_hidden: &Tensor,
        sampler: &mut FishS2Sampler,
        cache: &mut PhysicalPagedKvCache,
        resident: bool,
    ) -> Result<FishS2GeneratedFrame> {
        let semantic_code =
            semantic_code_from_token_id_from_fast_config(&self.cfg, semantic_token_id)?;
        if resident {
            // A row alone takes the same 32-row graph as beside others.
            let run = |key: FastGraphKey, inputs: &[Tensor]| self.run_frame_graph(key, inputs);
            if let Some(mut frames) = self.graphed_frames(
                &[semantic_token_id],
                &[semantic_code],
                slow_hidden,
                &mut [&mut *sampler],
                &run,
            )? {
                cache.reset_invocation()?;
                return frames.pop().ok_or_else(|| {
                    Error::InferenceError("Fish S2 fast graph returned no frame".into())
                });
            }
        }
        cache.reset_invocation()?;
        let hidden = self.project_slow_hidden(slow_hidden)?;
        let _ = self.forward_step_inner(&hidden, 0, cache, false)?;

        let mut codebooks = vec![semantic_code];
        let mut current = self.codebook_embedding(semantic_code)?;
        let tokens = self.cfg.num_codebooks.saturating_sub(1);
        // A row alone draws with the batched sampler at width one, so it
        // samples as it does beside others.
        let row_draws =
            self.resident_row_draws(&mut [&mut *sampler], slow_hidden.device(), resident)?;
        let draws = if resident && row_draws.is_none() {
            sampler.resident_draws(tokens, slow_hidden.device())?
        } else {
            None
        };
        let mut resident = ResidentCodes::default();
        for codebook_idx in 1..self.cfg.num_codebooks {
            let logits = self.forward_step(&current, codebook_idx, cache)?;
            let row = logits.i((0, 0))?;
            let next = match (&row_draws, &draws) {
                (Some(row_draws), _) => {
                    resident.sample_rows(&row.unsqueeze(0)?, row_draws, codebook_idx - 1)?
                }
                (None, Some(draws)) => resident.sample(&row, sampler, draws, codebook_idx - 1)?,
                (None, None) => {
                    let code = sample_logits(&row, sampler)?;
                    codebooks.push(code);
                    Tensor::from_vec(vec![code], 1, row.device())?
                }
            };
            if codebook_idx + 1 < self.cfg.num_codebooks {
                current = self.embeddings.forward(&next.reshape((1, 1))?)?;
            }
        }
        if row_draws.is_some() || draws.is_some() {
            codebooks.extend(resident.read_back()?);
        }
        Ok(FishS2GeneratedFrame {
            semantic_token_id,
            codebooks,
        })
    }
}

/// The graph inputs of one chunk of rows that share a policy, and the rows'
/// places in the frame.
struct GraphChunk {
    rows: Vec<usize>,
    policy: FishS2FastPolicy,
}

impl FishS2FastDecoder {
    /// Turns the frame graphs on (CUDA only, unless
    /// [`FISH_S2_FAST_GRAPHS_ENV`] is off) once a self-check passes: for
    /// greedy frames and for the default draw (temperature 0.7, top-p 0.7,
    /// top-k 30), two frames of xorshift inputs at `input_dtype` (the slow
    /// transformer's hidden dtype, which every frame's input has), each
    /// captured or replayed and compared with the eager dense frame bit for
    /// bit ([`ShapeGraphs::run`]). A mismatch or a CUDA error leaves every
    /// frame on the eager paged path, with the reason in the diagnostics.
    pub(crate) fn resolve_graphs(&mut self, input_dtype: DType) {
        let device = self.embeddings.embeddings().device().clone();
        let resolved = if fast_graphs_requested() {
            ShapeGraphs::resolve("Fish S2 fast AR", &device, FAST_GRAPH_LIMITS)
        } else {
            Err(format!("{FISH_S2_FAST_GRAPHS_ENV} is off"))
        };
        let resolved = resolved.and_then(|graphs| {
            match self.self_check_graphs(&graphs, input_dtype, &device) {
                Ok(()) if graphs.is_on() => Ok(graphs),
                Ok(()) => Err(format!(
                    "self-check failed: {}",
                    graphs.diagnostics()["off"]
                )),
                Err(error) => Err(format!("self-check failed: {error}")),
            }
        });
        match &resolved {
            Ok(_) => tracing::info!(
                rows = FAST_GRAPH_ROWS,
                "Fish S2 fast AR frames run as CUDA graphs"
            ),
            Err(reason) if device.is_cuda() => {
                tracing::warn!(%reason, "Fish S2 fast AR frames run eagerly")
            }
            Err(_) => {}
        }
        self.graphs.graphs = resolved;
    }

    /// Two frames per policy through `graphs` (see [`Self::resolve_graphs`]):
    /// the first captures the graph and checks a replay against the eager
    /// frame, the second checks a replay over new inputs, so a value baked
    /// into the graph shows.
    fn self_check_graphs(
        &self,
        graphs: &ShapeGraphs<FastGraphKey>,
        input_dtype: DType,
        device: &Device,
    ) -> Result<()> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let tokens = self.cfg.num_codebooks.saturating_sub(1);
        let policies = [0.0, 0.7].map(|temperature| FishS2FastPolicy {
            temperature,
            top_p: 0.7,
            top_k: 30,
        });
        for policy in policies {
            for _ in 0..2 {
                let width = self.cfg.input_hidden_size;
                let hidden = (0..FAST_GRAPH_ROWS * width)
                    .map(|_| uniform() * 2.0 - 1.0)
                    .collect::<Vec<_>>();
                let hidden = Tensor::from_vec(hidden, (FAST_GRAPH_ROWS, 1, width), device)?
                    .to_dtype(input_dtype)?;
                let codes = (0..FAST_GRAPH_ROWS)
                    .map(|_| (uniform() * self.cfg.codebook_size as f32) as u32)
                    .map(|code| code.min(self.cfg.codebook_size as u32 - 1))
                    .collect::<Vec<_>>();
                let codes = Tensor::from_vec(codes, FAST_GRAPH_ROWS, device)?;
                let noise = if policy.samples() {
                    let count = FAST_GRAPH_ROWS * tokens * policy.top_k;
                    Tensor::from_vec(
                        gumbel_noise(count, &mut uniform),
                        (FAST_GRAPH_ROWS, tokens, policy.top_k),
                        device,
                    )?
                } else {
                    Tensor::zeros(1, DType::F32, device)?
                };
                let inputs = [
                    hidden,
                    codes,
                    noise,
                    Tensor::triu2(policy.top_k, DType::F32, device)?,
                    self.graphs.column_ids(self.cfg.codebook_size, device)?,
                ];
                self.graph_run(graphs, (policy, input_dtype), &inputs)?;
            }
        }
        Ok(())
    }

    /// The frame graphs' state and counters, for the model diagnostics.
    pub(crate) fn graphs_diagnostics(&self) -> serde_json::Value {
        let mut diagnostics = match &self.graphs.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        };
        diagnostics["rows"] = FAST_GRAPH_ROWS.into();
        diagnostics["graphed_frames"] = self.graphs.graphed_frames.load(Ordering::Relaxed).into();
        diagnostics["eager_frames"] = self.graphs.eager_frames.load(Ordering::Relaxed).into();
        diagnostics
    }

    /// [`Self::dense_frame`] over `inputs` through the graph of `key`.
    fn run_frame_graph(&self, key: FastGraphKey, inputs: &[Tensor]) -> Result<Tensor> {
        let Ok(graphs) = &self.graphs.graphs else {
            return Err(Error::InferenceError("Fish S2 fast graphs are off".into()));
        };
        self.graph_run(graphs, key, inputs)
    }

    /// [`Self::dense_frame`] over `inputs` through `graphs` under `key`,
    /// pricing what the graph keeps: its stable copy of each input and its
    /// packed codes.
    fn graph_run(
        &self,
        graphs: &ShapeGraphs<FastGraphKey>,
        key: FastGraphKey,
        inputs: &[Tensor],
    ) -> Result<Tensor> {
        let bytes = |tensor: &Tensor| (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64;
        let depths = self.cfg.num_codebooks.saturating_sub(1);
        let codes = (2 * depths * FAST_GRAPH_ROWS * DType::U32.size_in_bytes()) as u64;
        let retained = inputs.iter().map(bytes).sum::<u64>() + codes;
        let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            Ok(vec![self.dense_frame(key.0, inputs)?])
        };
        graphs
            .run(key, retained, inputs, &body)?
            .pop()
            .ok_or_else(|| Error::InferenceError("Fish S2 fast graph returned nothing".into()))
    }

    /// One frame's fast codes for `semantic_codes` rows through the graphs,
    /// when they are on and every row's policy is one the device serves:
    /// rows grouped by policy in frame order, each group in chunks of
    /// [`FAST_GRAPH_ROWS`] padded with zero rows, every sampling row's noise
    /// for the frame drawn from its own sampler up front exactly as the
    /// eager frame draws it, and each chunk's codes read back once. `run`
    /// replays a chunk (the graph cache in production). `None` runs the
    /// eager paged frame.
    fn graphed_frames(
        &self,
        semantic_tokens: &[u32],
        semantic_codes: &[u32],
        slow_hidden: &Tensor,
        samplers: &mut [&mut FishS2Sampler],
        run: &dyn Fn(FastGraphKey, &[Tensor]) -> Result<Tensor>,
    ) -> Result<Option<Vec<FishS2GeneratedFrame>>> {
        match &self.graphs.graphs {
            Ok(graphs) if graphs.is_on() => {}
            _ => return Ok(None),
        }
        self.graph_frames_with(semantic_tokens, semantic_codes, slow_hidden, samplers, run)
    }

    /// [`Self::graphed_frames`] without the graphs' state check: what a
    /// replay computes, with `run` standing in for it.
    fn graph_frames_with(
        &self,
        semantic_tokens: &[u32],
        semantic_codes: &[u32],
        slow_hidden: &Tensor,
        samplers: &mut [&mut FishS2Sampler],
        run: &dyn Fn(FastGraphKey, &[Tensor]) -> Result<Tensor>,
    ) -> Result<Option<Vec<FishS2GeneratedFrame>>> {
        let rows = semantic_codes.len();
        let width = self.cfg.input_hidden_size;
        if rows == 0
            || rows != samplers.len()
            || rows != semantic_tokens.len()
            || slow_hidden.dims() != [rows, 1, width]
            || self.cfg.num_codebooks < 2
        {
            self.graphs.eager_frames.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let mut chunks: Vec<GraphChunk> = Vec::new();
        for (row, sampler) in samplers.iter().enumerate() {
            let Some(policy) = sampler.fast_policy()? else {
                self.graphs.eager_frames.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            };
            match chunks
                .iter_mut()
                .find(|chunk| chunk.policy == policy && chunk.rows.len() < FAST_GRAPH_ROWS)
            {
                Some(chunk) => chunk.rows.push(row),
                None => chunks.push(GraphChunk {
                    rows: vec![row],
                    policy,
                }),
            }
        }
        let device = slow_hidden.device();
        let tokens = self.cfg.num_codebooks - 1;
        let mut codebooks = vec![Vec::new(); rows];
        for chunk in &chunks {
            let count = chunk.rows.len();
            let spare = FAST_GRAPH_ROWS - count;
            let contiguous = chunk.rows.windows(2).all(|pair| pair[1] == pair[0] + 1);
            let hidden = if contiguous {
                slow_hidden.narrow(0, chunk.rows[0], count)?
            } else {
                let index = chunk.rows.iter().map(|&row| row as u32).collect::<Vec<_>>();
                slow_hidden.index_select(&Tensor::from_vec(index, count, device)?, 0)?
            };
            let mut codes = vec![0u32; FAST_GRAPH_ROWS];
            for (slot, &row) in chunk.rows.iter().enumerate() {
                codes[slot] = semantic_codes[row];
            }
            let noise = if chunk.policy.samples() {
                let per_row = tokens * chunk.policy.top_k;
                let mut noise = vec![0f32; FAST_GRAPH_ROWS * per_row];
                for (slot, &row) in chunk.rows.iter().enumerate() {
                    samplers[row].draw_fast_noise(&mut noise[slot * per_row..][..per_row]);
                }
                Tensor::from_vec(noise, (FAST_GRAPH_ROWS, tokens, chunk.policy.top_k), device)?
            } else {
                Tensor::zeros(1, DType::F32, device)?
            };
            let inputs = [
                hidden.pad_with_zeros(0, 0, spare)?,
                Tensor::from_vec(codes, FAST_GRAPH_ROWS, device)?,
                noise,
                Tensor::triu2(chunk.policy.top_k, DType::F32, device)?,
                self.graphs.column_ids(self.cfg.codebook_size, device)?,
            ];
            let packed = run((chunk.policy, slow_hidden.dtype()), &inputs)?;
            // [2, tokens, rows]: the codes, then each depth's non-finite
            // logit counts, read back once.
            let packed = packed.narrow(2, 0, count)?.contiguous()?;
            telemetry::record_host_read(DType::U32, packed.elem_count());
            let host = packed.to_vec3::<u32>()?;
            let non_finite = host[1].iter().flatten().map(|&v| u64::from(v)).sum::<u64>();
            if non_finite > 0 {
                return Err(Error::InferenceError(format!(
                    "Fish S2 fast logits contain {non_finite} non-finite values"
                )));
            }
            for (slot, &row) in chunk.rows.iter().enumerate() {
                codebooks[row] = std::iter::once(semantic_codes[row])
                    .chain(host[0].iter().map(|depth| depth[slot]))
                    .collect();
            }
        }
        self.graphs.graphed_frames.fetch_add(1, Ordering::Relaxed);
        FISH_S2_RESIDENT_FAST_FRAMES.fetch_add(1, Ordering::Relaxed);
        Ok(Some(
            semantic_tokens
                .iter()
                .zip(codebooks)
                .map(|(&semantic_token_id, codebooks)| FishS2GeneratedFrame {
                    semantic_token_id,
                    codebooks,
                })
                .collect(),
        ))
    }

    /// One frame's fast codes for every row, entirely on the device, from
    /// `inputs`: slow hiddens `[rows, 1, input_hidden]`, semantic codes
    /// `[rows]` (U32), the frame's Gumbel noise `[rows, depths, top_k]` (any
    /// tensor when greedy), the top-p matrix and the column ids. The eager
    /// batch frame with each row's KV in a dense per-row cache instead of
    /// paged invocation pages ([`FishS2FastAttention::forward_dense`]), and
    /// each depth drawn as [`sample_resident_rows`] draws it. Returns
    /// `[2, depths, rows]` U32: the codes, then each depth's non-finite logit
    /// counts. A pure function of its inputs and the weights, so a CUDA graph
    /// can capture it.
    fn dense_frame(&self, policy: FishS2FastPolicy, inputs: &[Tensor]) -> Result<Tensor> {
        let [slow_hidden, semantic, noise, upper, column_ids] = inputs else {
            return Err(Error::InferenceError(format!(
                "Fish S2 fast frame received {} inputs, expected 5",
                inputs.len()
            )));
        };
        let rows = slow_hidden.dim(0)?;
        let mut kv = (0..self.layers.len()).map(|_| None).collect::<Vec<_>>();
        let hidden = self.project_slow_hidden(slow_hidden)?;
        self.dense_step(&hidden, 0, &mut kv, false)?;
        let mut current = self.embeddings.forward(&semantic.reshape((rows, 1))?)?;
        let mut codes = Vec::with_capacity(self.cfg.num_codebooks);
        let mut non_finite = Vec::with_capacity(self.cfg.num_codebooks);
        for depth in 1..self.cfg.num_codebooks {
            let logits = self.dense_step(&current, depth, &mut kv, true)?;
            let depth_noise = if policy.samples() {
                Some(noise.i((.., depth - 1))?)
            } else {
                None
            };
            let (next, counts) = resident_rows_codes(
                &logits.i((.., 0))?,
                policy,
                upper,
                depth_noise.as_ref(),
                Some(column_ids),
            )?;
            if depth + 1 < self.cfg.num_codebooks {
                current = self.embeddings.forward(&next.reshape((rows, 1))?)?;
            }
            codes.push(next);
            non_finite.push(counts);
        }
        Ok(Tensor::stack(
            &[Tensor::stack(&codes, 0)?, Tensor::stack(&non_finite, 0)?],
            0,
        )?)
    }

    /// One position of every row through the layers (as
    /// `forward_step_batch`), each layer attending over its dense cache in
    /// `kv` (grown here); the logits when `project_logits`, else the hidden
    /// state.
    fn dense_step(
        &self,
        x: &Tensor,
        position: usize,
        kv: &mut [Option<(Tensor, Tensor)>],
        project_logits: bool,
    ) -> Result<Tensor> {
        let mut hidden = x.clone();
        for (layer, cache) in self.layers.iter().zip(kv.iter_mut()) {
            let norm = layer.input_layernorm.forward(&hidden)?;
            let attn = layer.self_attn.forward_dense(&norm, position, cache)?;
            hidden = hidden.broadcast_add(&attn)?;
            hidden = hidden.broadcast_add(
                &layer
                    .mlp
                    .forward(&layer.post_attention_layernorm.forward(&hidden)?)?,
            )?;
        }
        if project_logits {
            self.output
                .forward(&self.norm.forward(&hidden)?)
                .map_err(Error::from)
        } else {
            Ok(hidden)
        }
    }
}

impl FishS2FastLayer {
    fn load(cfg: &FishS2FastConfig, rotary: &FishS2RotaryCache, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["input_layernorm", "attention_norm"],
        )?;
        let self_attn = FishS2FastAttention::load(cfg, rotary, vb.pp("self_attn"))?;
        let post_attention_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["post_attention_layernorm", "ffn_norm"],
        )?;
        let mlp = FishS2FastMlp::load(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self
            .self_attn
            .forward(&normed, input_pos, cache, prepared, layer_idx)?;
        let x = x.broadcast_add(&attn)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp).map_err(Error::from)
    }
}

impl FishS2FastAttention {
    fn load(cfg: &FishS2FastConfig, rotary: &FishS2RotaryCache, vb: VarBuilder) -> Result<Self> {
        let total = cfg.q_size() + 2 * cfg.kv_size();
        if vb.contains_tensor("q_norm.weight") || vb.contains_tensor("k_norm.weight") {
            return Err(Error::ModelLoadError(
                "Fish S2 fast attention does not use Q/K normalization".into(),
            ));
        }
        Ok(Self {
            qkv_proj: candle_nn::linear_no_bias(cfg.hidden_size, total, vb.pp("qkv_proj"))?,
            o_proj: candle_nn::linear_no_bias(cfg.q_size(), cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: if vb.contains_tensor("q_norm.weight") {
                Some(candle_nn::rms_norm(
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("q_norm"),
                )?)
            } else {
                None
            },
            k_norm: if vb.contains_tensor("k_norm.weight") {
                Some(candle_nn::rms_norm(
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("k_norm"),
                )?)
            } else {
                None
            },
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary: rotary.clone(),
        })
    }

    fn forward_batch(
        &self,
        x: &Tensor,
        position: usize,
        cache: &PhysicalPagedKvCache,
        batch: &mut super::batch::FishPhysicalBatch,
        layer: usize,
    ) -> Result<Tensor> {
        let rows = x.dim(0)?;
        let qsize = self.num_heads * self.head_dim;
        let kvsize = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = self
            .rotary
            .apply(
                &qkv.narrow(2, 0, qsize)?
                    .reshape((rows, 1, self.num_heads, self.head_dim))?,
                position,
            )?
            .reshape((rows, self.num_heads, self.head_dim))?
            .contiguous()?;
        let k = self
            .rotary
            .apply(
                &qkv.narrow(2, qsize, kvsize)?.reshape((
                    rows,
                    1,
                    self.num_kv_heads,
                    self.head_dim,
                ))?,
                position,
            )?
            .reshape((rows, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let v = qkv
            .narrow(2, qsize + kvsize, kvsize)?
            .reshape((rows, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let output = batch.attend(
            cache,
            layer,
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        self.o_proj
            .forward(&output.reshape((rows, 1, qsize))?)
            .map_err(Error::from)
    }

    /// [`Self::forward_batch`] over a dense per-row cache instead of paged
    /// KV: `x` is `[rows, 1, hidden]` at `position`, and `kv` holds the rows'
    /// earlier keys and values (F32 `[rows, kv_heads, positions, head_dim]`)
    /// and gains this call's. Query head `h` reads key head `h / group`, as
    /// the paged kernels do; scores and the softmax run in F32, as the paged
    /// kernels run them.
    fn forward_dense(
        &self,
        x: &Tensor,
        position: usize,
        kv: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let rows = x.dim(0)?;
        let (heads, kv_heads, head_dim) = (self.num_heads, self.num_kv_heads, self.head_dim);
        if kv_heads == 0 || heads % kv_heads != 0 {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast attention has {heads} query heads for {kv_heads} key heads"
            )));
        }
        let qsize = heads * head_dim;
        let kvsize = kv_heads * head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = self.rotary.apply(
            &qkv.narrow(2, 0, qsize)?
                .reshape((rows, 1, heads, head_dim))?,
            position,
        )?;
        let k = self.rotary.apply(
            &qkv.narrow(2, qsize, kvsize)?
                .reshape((rows, 1, kv_heads, head_dim))?,
            position,
        )?;
        let v = qkv
            .narrow(2, qsize + kvsize, kvsize)?
            .reshape((rows, 1, kv_heads, head_dim))?;
        let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let (keys, values) = match kv.take() {
            Some((keys, values)) => (
                Tensor::cat(&[&keys, &k], 2)?,
                Tensor::cat(&[&values, &v], 2)?,
            ),
            None => (k, v),
        };
        if keys.dim(2)? != position + 1 {
            return Err(Error::InvalidInput(format!(
                "Fish S2 dense fast cache holds {} positions at {position}",
                keys.dim(2)?
            )));
        }
        // [rows, kv_heads, group, head_dim]: query head h = kv * group + g.
        let queries = q
            .reshape((rows, kv_heads, heads / kv_heads, head_dim))?
            .to_dtype(DType::F32)?;
        let scale = f64::from(1.0f32 / (head_dim as f32).sqrt());
        let scores = (queries.matmul(&keys.t()?)? * scale)?;
        let out = ops::softmax_last_dim(&scores)?
            .matmul(&values)?
            .to_dtype(x.dtype())?
            .reshape((rows, 1, qsize))?;
        *kv = Some((keys, values));
        self.o_proj.forward(&out).map_err(Error::from)
    }

    fn forward(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        if bsz != 1 {
            return Err(Error::InvalidInput(
                "Fish S2 fast physical paged attention expects one sequence".into(),
            ));
        }
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = qkv
            .narrow(2, 0, q_size)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = qkv.narrow(2, q_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let q = self.rotary.apply(&q, input_pos)?.squeeze(0)?;
        let k = self.rotary.apply(&k, input_pos)?.squeeze(0)?;
        let v = v.squeeze(0)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = cache.write_and_attend(layer_idx, prepared, &q, &k, &v, scale)?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

impl FishS2FastMlp {
    fn load(cfg: &FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = ops::silu(&gate)?.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

static FISH_S2_RESIDENT_FAST_FRAMES: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of Fish fast-codebook frames sampled on the device
/// with a single code readback (FSH-4).
pub fn fish_s2_resident_fast_frames() -> u64 {
    FISH_S2_RESIDENT_FAST_FRAMES.load(Ordering::Relaxed)
}

/// Fast codes are sampled on the device where readbacks are synchronizing
/// device round trips (CUDA, Metal). On the CPU the host sampler reads
/// memory that is already local.
fn resident_fast_sampling(device: &Device) -> bool {
    device.is_cuda() || device.is_metal()
}

/// One frame's device-resident fast codes and the non-finite logit counts
/// of the rows they were drawn from, read back together once.
#[derive(Default)]
struct ResidentCodes {
    codes: Vec<Tensor>,
    non_finite: Vec<Tensor>,
}

impl ResidentCodes {
    /// Draw one code (left on the device as a `[1]` U32 tensor).
    fn sample(
        &mut self,
        row: &Tensor,
        sampler: &FishS2Sampler,
        draws: &FishS2ResidentDraws,
        token: usize,
    ) -> Result<Tensor> {
        let (code, non_finite) =
            sample_resident(row, sampler.temperature, sampler.top_p, draws, token)?.ok_or_else(
                || Error::InferenceError("Fish S2 resident fast sampler lost its policy".into()),
            )?;
        self.codes.push(code.clone());
        self.non_finite.push(non_finite);
        Ok(code)
    }

    /// Draw every row's code at one depth (left on the device as a `[rows]`
    /// U32 tensor).
    fn sample_rows(
        &mut self,
        logits: &Tensor,
        draws: &FishS2ResidentRowDraws,
        token: usize,
    ) -> Result<Tensor> {
        let (codes, non_finite) = sample_resident_rows(logits, draws, token)?;
        self.codes.push(codes.clone());
        self.non_finite.push(non_finite);
        Ok(codes)
    }

    /// The codes in draw order, from one readback; an error if any sampled
    /// row held a non-finite logit.
    fn read_back(self) -> Result<Vec<u32>> {
        let count = self.codes.iter().map(Tensor::elem_count).sum::<usize>();
        if count == 0 {
            return Ok(Vec::new());
        }
        let packed = Tensor::cat(&[self.codes, self.non_finite].concat(), 0)?;
        telemetry::record_host_read(DType::U32, packed.elem_count());
        let values = packed.to_vec1::<u32>()?;
        FISH_S2_RESIDENT_FAST_FRAMES.fetch_add(1, Ordering::Relaxed);
        let non_finite = values[count..].iter().map(|&v| u64::from(v)).sum::<u64>();
        if non_finite > 0 {
            return Err(Error::InferenceError(format!(
                "Fish S2 fast logits contain {non_finite} non-finite values"
            )));
        }
        Ok(values[..count].to_vec())
    }
}

pub(crate) fn sample_logits(row: &Tensor, sampler: &mut FishS2Sampler) -> Result<u32> {
    super::sampling::validate_policy(sampler.temperature, sampler.top_p)?;
    let distribution =
        FishS2SamplingDistribution::from_logits(row, None, sampler.top_k, sampler.top_p, "fast")?;
    sampler.sample(&distribution)
}

fn semantic_code_from_token_id_from_fast_config(
    cfg: &FishS2FastConfig,
    token_id: u32,
) -> Result<u32> {
    if !(cfg.semantic_start_token_id..=cfg.semantic_end_token_id).contains(&token_id) {
        return Err(Error::InvalidInput(format!(
            "Fish S2 token {token_id} is outside the semantic range"
        )));
    }
    Ok(token_id - cfg.semantic_start_token_id)
}

fn load_rms_norm_alias(dim: usize, eps: f64, vb: &VarBuilder, aliases: &[&str]) -> Result<RmsNorm> {
    for alias in aliases {
        if vb.contains_tensor(&format!("{alias}.weight")) {
            return candle_nn::rms_norm(dim, eps, vb.pp(*alias)).map_err(Error::from);
        }
    }
    candle_nn::rms_norm(dim, eps, vb.pp(aliases[0])).map_err(Error::from)
}

#[cfg(test)]
pub(super) fn tiny_for_batch_tests() -> FishS2FastDecoder {
    let mut cfg = tests::tiny_cfg();
    cfg.input_hidden_size = 4;
    cfg.num_codebooks = 2;
    tests::tiny_decoder_config(&candle_core::Device::Cpu, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape};
    use std::collections::HashMap;

    pub(super) fn tiny_cfg() -> FishS2FastConfig {
        FishS2FastConfig {
            input_hidden_size: 3,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            codebook_size: 8,
            num_codebooks: 3,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    fn tensor(device: &Device, shape: impl Into<Shape>, value: f32) -> Tensor {
        Tensor::full(value, shape, device).unwrap()
    }

    fn tiny_decoder(device: &Device) -> FishS2FastDecoder {
        tiny_decoder_config(device, tiny_cfg())
    }

    pub(super) fn tiny_decoder_config(device: &Device, cfg: FishS2FastConfig) -> FishS2FastDecoder {
        let mut tensors = HashMap::new();
        tensors.insert(
            "fast_project_in.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.input_hidden_size), 0.01),
        );
        tensors.insert(
            "fast_project_in.bias".to_string(),
            tensor(device, (cfg.hidden_size,), 0.0),
        );
        tensors.insert(
            "fast_embeddings.weight".to_string(),
            tensor(device, (cfg.codebook_size, cfg.hidden_size), 0.02),
        );
        tensors.insert(
            "fast_norm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_output.weight".to_string(),
            tensor(device, (cfg.codebook_size, cfg.hidden_size), 0.03),
        );
        tensors.insert(
            "fast_layers.0.input_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_layers.0.post_attention_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_layers.0.self_attn.qkv_proj.weight".to_string(),
            tensor(
                device,
                (cfg.q_size() + 2 * cfg.kv_size(), cfg.hidden_size),
                0.01,
            ),
        );
        tensors.insert(
            "fast_layers.0.self_attn.o_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.q_size()), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.gate_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.up_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.down_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.intermediate_size), 0.01),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        FishS2FastDecoder::load(cfg, vb).unwrap()
    }

    /// FSH-4: a device-resident frame reads its fast codes back once (the
    /// host sampler reads every code: num_codebooks - 1 per frame) and, when
    /// greedy, picks exactly the host's codes. Batched resident rows match
    /// scalar resident rows, and a non-finite logit fails the frame.
    #[test]
    fn resident_fast_frame_reads_its_codes_back_once() {
        let device = Device::Cpu;
        let decoder = tiny_decoder(&device);
        let cfg = decoder.config().clone();
        let cache = || {
            super::super::physical::test_physical_cache(
                303,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.num_codebooks,
            )
        };
        let slow_hidden = Tensor::from_vec(vec![0.4f32, -0.2, 0.9], (1, 1, 3), &device).unwrap();
        let mut host_cache = cache();
        let mut resident_cache = cache();
        let mut host_sampler = FishS2Sampler::new(0.0, 0.9, 5);
        let mut resident_sampler = FishS2Sampler::new(0.0, 0.9, 5);
        for semantic in [22u32, 25, 27] {
            let before = telemetry::thread_host_reads();
            let host = decoder
                .generate_frame_with(
                    semantic,
                    &slow_hidden,
                    &mut host_sampler,
                    &mut host_cache,
                    false,
                )
                .unwrap();
            let host_reads = telemetry::thread_host_reads() - before;
            let before = telemetry::thread_host_reads();
            let resident = decoder
                .generate_frame_with(
                    semantic,
                    &slow_hidden,
                    &mut resident_sampler,
                    &mut resident_cache,
                    true,
                )
                .unwrap();
            assert_eq!(telemetry::thread_host_reads() - before, 1);
            assert_eq!(host_reads, (cfg.num_codebooks - 1) as u64);
            assert_eq!(resident, host);
        }

        // Sampled rows: the batch draws each row from its own sampler.
        let caches = || {
            super::super::physical::test_physical_caches(
                304,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.num_codebooks,
                3,
            )
        };
        let mut batch_caches = caches();
        let mut scalar_caches = caches();
        let mut samplers = (0..3)
            .map(|row| FishS2Sampler::new(0.8, 0.8, row + 40))
            .collect::<Vec<_>>();
        let mut scalar_samplers = samplers.clone();
        let hidden = Tensor::from_vec(
            (0..9).map(|i| (i as f32 * 0.7).cos()).collect::<Vec<_>>(),
            (3, 1, 3),
            &device,
        )
        .unwrap();
        let ids = [21u32, 24, 26];
        let expected = (0..3)
            .map(|row| {
                decoder
                    .generate_frame_with(
                        ids[row],
                        &hidden.narrow(0, row, 1).unwrap(),
                        &mut scalar_samplers[row],
                        &mut scalar_caches[row],
                        true,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let before = telemetry::thread_host_reads();
        let actual = decoder
            .generate_frames_batch_with(
                &ids,
                &hidden,
                &mut samplers.iter_mut().collect::<Vec<_>>(),
                &mut batch_caches.iter_mut().collect::<Vec<_>>(),
                true,
            )
            .unwrap();
        assert_eq!(telemetry::thread_host_reads() - before, 1);
        assert_eq!(actual, expected);

        let mut codes = ResidentCodes::default();
        let nan = Tensor::from_vec(vec![0.0f32, f32::NAN, 1.0], 3, &device).unwrap();
        let mut sampler = FishS2Sampler::new(0.7, 0.9, 3);
        let draws = sampler.resident_draws(1, &device).unwrap().unwrap();
        codes.sample(&nan, &sampler, &draws, 0).unwrap();
        assert!(codes
            .read_back()
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
    }

    #[test]
    fn sampler_argmax_is_deterministic_at_zero_temperature() {
        let device = Device::Cpu;
        let row = Tensor::from_vec(vec![0.1f32, 2.0, 1.0], (3,), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 7);
        assert_eq!(sample_logits(&row, &mut sampler).unwrap(), 1);
    }

    #[test]
    fn sampler_rejects_empty_greedy_logits_before_readback() {
        let device = Device::Cpu;
        let row = Tensor::from_vec(Vec::<f32>::new(), (0,), &device).unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 7);
        let err = sample_logits(&row, &mut sampler).expect_err("empty logits should fail");

        assert!(format!("{err}").contains("Fish S2 fast sampler received empty logits"));
    }

    #[test]
    fn native_fast_batch_matches_independent_sampling_and_reordered_rows() {
        let decoder = tiny_decoder(&Device::Cpu);
        let cfg = decoder.config();
        let cache = || {
            super::super::physical::test_physical_caches(
                901,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.num_codebooks,
                7,
            )
        };
        let mut batch_caches = cache();
        let mut scalar_caches = cache();
        let mut samplers = (0..7)
            .map(|row| FishS2Sampler::new(0.7, 0.9, row + 17))
            .collect::<Vec<_>>();
        let mut scalar_samplers = samplers.clone();
        for iteration in 0..3 {
            let ids = (0..7)
                .map(|row| 20 + (row + iteration) % 8)
                .collect::<Vec<_>>();
            let hidden = Tensor::from_vec(
                (0..21).map(|i| (i as f32 * 0.31).sin()).collect::<Vec<_>>(),
                (7, 1, 3),
                &Device::Cpu,
            )
            .unwrap();
            let expected = (0..7)
                .map(|row| {
                    decoder
                        .generate_frame(
                            ids[row],
                            &hidden.narrow(0, row, 1).unwrap(),
                            &mut scalar_samplers[row],
                            &mut scalar_caches[row],
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let actual = decoder
                .generate_frames_batch(
                    &ids,
                    &hidden,
                    &mut samplers.iter_mut().collect::<Vec<_>>(),
                    &mut batch_caches.iter_mut().collect::<Vec<_>>(),
                )
                .unwrap();
            assert_eq!(actual, expected);
            for cache in &mut batch_caches {
                assert_eq!(cache.context_len(), cfg.num_codebooks);
                assert_eq!(cache.take_completed_writes().len(), cfg.num_codebooks);
            }
            samplers.reverse();
            scalar_samplers.reverse();
            batch_caches.reverse();
            scalar_caches.reverse();
        }
    }

    /// The batched resident sampler: a batch of frames equals each row's
    /// frame drawn alone with the resident sampler (which takes the same
    /// sampler at width one), rows reordered between frames.
    #[test]
    fn resident_fast_batch_matches_rows_drawn_alone() {
        let decoder = tiny_decoder(&Device::Cpu);
        let cfg = decoder.config();
        let cache = || {
            super::super::physical::test_physical_caches(
                904,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.num_codebooks,
                5,
            )
        };
        let mut batch_caches = cache();
        let mut scalar_caches = cache();
        let mut samplers = (0..5)
            .map(|row| FishS2Sampler::new(0.8, 0.85, row + 3))
            .collect::<Vec<_>>();
        let mut scalar_samplers = samplers.clone();
        for iteration in 0..3 {
            let ids = (0..5)
                .map(|row| 20 + (row * 3 + iteration) % 8)
                .collect::<Vec<_>>();
            let hidden = Tensor::from_vec(
                (0..15)
                    .map(|i| ((i + iteration) as f32 * 0.47).cos())
                    .collect::<Vec<_>>(),
                (5, 1, 3),
                &Device::Cpu,
            )
            .unwrap();
            let expected = (0..5)
                .map(|row| {
                    decoder
                        .generate_frame_with(
                            ids[row],
                            &hidden.narrow(0, row, 1).unwrap(),
                            &mut scalar_samplers[row],
                            &mut scalar_caches[row],
                            true,
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let actual = decoder
                .generate_frames_batch_with(
                    &ids,
                    &hidden,
                    &mut samplers.iter_mut().collect::<Vec<_>>(),
                    &mut batch_caches.iter_mut().collect::<Vec<_>>(),
                    true,
                )
                .unwrap();
            assert_eq!(actual, expected);
            samplers.rotate_left(2);
            scalar_samplers.rotate_left(2);
            batch_caches.rotate_left(2);
            scalar_caches.rotate_left(2);
        }
    }

    /// Unstructured values in `[-scale, scale)`.
    fn xorshift(seed: u64, count: usize, scale: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Two layers, grouped query heads, an input projection and a codebook
    /// of `codebook_size` codes.
    fn xorshift_cfg(codebook_size: usize) -> FishS2FastConfig {
        FishS2FastConfig {
            input_hidden_size: 6,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            codebook_size,
            num_codebooks: 5,
            semantic_start_token_id: 100,
            semantic_end_token_id: 100 + codebook_size as u32 - 1,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    /// Every weight of `cfg` from xorshift, norms near one.
    fn xorshift_decoder(cfg: FishS2FastConfig, seed: u64) -> FishS2FastDecoder {
        let device = Device::Cpu;
        let mut salt = seed;
        let mut tensor = |shape: &[usize], offset: f32| {
            salt += 1;
            let count = shape.iter().product();
            let values = xorshift(salt * 7919, count, 0.6)
                .into_iter()
                .map(|value| value + offset)
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, &device).unwrap()
        };
        let (hidden, input) = (cfg.hidden_size, cfg.input_hidden_size);
        let mut tensors = HashMap::new();
        tensors.insert(
            "fast_project_in.weight".into(),
            tensor(&[hidden, input], 0.0),
        );
        tensors.insert("fast_project_in.bias".into(), tensor(&[hidden], 0.0));
        tensors.insert(
            "fast_embeddings.weight".into(),
            tensor(&[cfg.codebook_size, hidden], 0.0),
        );
        tensors.insert("fast_norm.weight".into(), tensor(&[hidden], 1.0));
        tensors.insert(
            "fast_output.weight".into(),
            tensor(&[cfg.codebook_size, hidden], 0.0),
        );
        for layer in 0..cfg.num_hidden_layers {
            let name = |suffix: &str| format!("fast_layers.{layer}.{suffix}");
            tensors.insert(name("input_layernorm.weight"), tensor(&[hidden], 1.0));
            tensors.insert(
                name("post_attention_layernorm.weight"),
                tensor(&[hidden], 1.0),
            );
            tensors.insert(
                name("self_attn.qkv_proj.weight"),
                tensor(&[cfg.q_size() + 2 * cfg.kv_size(), hidden], 0.0),
            );
            tensors.insert(
                name("self_attn.o_proj.weight"),
                tensor(&[hidden, cfg.q_size()], 0.0),
            );
            for projection in ["gate_proj", "up_proj"] {
                tensors.insert(
                    name(&format!("mlp.{projection}.weight")),
                    tensor(&[cfg.intermediate_size, hidden], 0.0),
                );
            }
            tensors.insert(
                name("mlp.down_proj.weight"),
                tensor(&[hidden, cfg.intermediate_size], 0.0),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        FishS2FastDecoder::load(cfg, vb).unwrap()
    }

    /// What a graph replay computes: the dense frame itself.
    fn dense_run(
        decoder: &FishS2FastDecoder,
    ) -> impl Fn(FastGraphKey, &[Tensor]) -> Result<Tensor> + '_ {
        move |key, inputs| decoder.dense_frame(key.0, inputs)
    }

    /// Each sampler's next draws, to show where its stream stands.
    fn next_draws(samplers: &mut [FishS2Sampler]) -> Vec<Vec<u32>> {
        samplers
            .iter_mut()
            .map(|sampler| {
                let mut values = [0f32; 6];
                let mut stochastic = sampler.clone();
                stochastic.temperature = 1.0;
                stochastic.draw_fast_noise(&mut values);
                values.iter().map(|value| value.to_bits()).collect()
            })
            .collect()
    }

    /// The dense layers give the paged layers' logits at every depth within
    /// F32 rounding, rows batched.
    #[test]
    fn dense_steps_match_the_paged_steps() {
        let cfg = xorshift_cfg(40);
        let decoder = xorshift_decoder(cfg.clone(), 3);
        let rows = 3;
        let hidden = Tensor::from_vec(
            xorshift(41, rows * cfg.input_hidden_size, 1.0),
            (rows, 1, cfg.input_hidden_size),
            &Device::Cpu,
        )
        .unwrap();
        let mut caches = super::super::physical::test_physical_caches(
            905,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.num_codebooks,
            rows,
        );
        let mut caches = caches.iter_mut().collect::<Vec<_>>();
        let projected = decoder.project_slow_hidden(&hidden).unwrap();
        let mut kv = vec![None; cfg.num_hidden_layers];
        let paged = decoder
            .forward_step_batch(&projected, 0, &mut caches, false)
            .unwrap();
        let dense = decoder.dense_step(&projected, 0, &mut kv, false).unwrap();
        let mut pairs = vec![(paged, dense)];
        for depth in 1..cfg.num_codebooks {
            let codes = (0..rows)
                .map(|row| ((row * 7 + depth * 3) % cfg.codebook_size) as u32)
                .collect::<Vec<_>>();
            let ids = Tensor::from_vec(codes, (rows, 1), &Device::Cpu).unwrap();
            let x = decoder.embeddings.forward(&ids).unwrap();
            let paged = decoder
                .forward_step_batch(&x, depth, &mut caches, true)
                .unwrap();
            let dense = decoder.dense_step(&x, depth, &mut kv, true).unwrap();
            pairs.push((paged, dense));
        }
        for (depth, (paged, dense)) in pairs.iter().enumerate() {
            assert_eq!(paged.dims(), dense.dims());
            let diff = (paged - dense)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-5, "depth {depth} differs by {diff}");
        }
    }

    /// The frame batches the graph tests run: (codebook size, rows, the
    /// rows' policies in turn) for greedy, sampled, three mixed policies,
    /// 34 rows in two chunks, and a codebook below top-k.
    #[allow(clippy::type_complexity)]
    const GRAPH_CASES: [(usize, usize, &[(f32, f32, usize)]); 6] = [
        (1500, 1, &[(0.0, 0.7, 30)]),
        (1500, 1, &[(0.7, 0.7, 30)]),
        (1500, 3, &[(0.7, 0.7, 30)]),
        (1500, 5, &[(0.7, 0.7, 30), (1.1, 0.95, 12), (0.0, 0.7, 30)]),
        (60, 34, &[(0.7, 0.7, 30)]),
        (20, 3, &[(1.1, 0.95, 12), (0.7, 0.7, 30)]),
    ];

    /// One graph case's decoder, slow hiddens, semantic tokens and codes,
    /// and samplers.
    #[allow(clippy::type_complexity)]
    fn graph_case(
        case: usize,
    ) -> (
        FishS2FastDecoder,
        Tensor,
        Vec<u32>,
        Vec<u32>,
        Vec<FishS2Sampler>,
    ) {
        let (codebook_size, rows, policies) = GRAPH_CASES[case];
        let cfg = xorshift_cfg(codebook_size);
        let decoder = xorshift_decoder(cfg.clone(), case as u64 + 5);
        let hidden = Tensor::from_vec(
            xorshift(case as u64 + 77, rows * cfg.input_hidden_size, 1.5),
            (rows, 1, cfg.input_hidden_size),
            &Device::Cpu,
        )
        .unwrap();
        let tokens = (0..rows)
            .map(|row| cfg.semantic_start_token_id + ((row * 13 + case) % codebook_size) as u32)
            .collect::<Vec<_>>();
        let codes = tokens
            .iter()
            .map(|&token| token - cfg.semantic_start_token_id)
            .collect::<Vec<_>>();
        let samplers = (0..rows)
            .map(|row| {
                let (temperature, top_p, top_k) = policies[row % policies.len()];
                FishS2Sampler::with_top_k(temperature, top_p, top_k, row as u64 + 31)
            })
            .collect::<Vec<_>>();
        (decoder, hidden, tokens, codes, samplers)
    }

    /// A row's frame alone, as a lone request runs it: the dense frame in
    /// slot 0 of 32 rows, the rest zero, its noise drawn from `sampler`.
    fn dense_alone(
        decoder: &FishS2FastDecoder,
        hidden: &Tensor,
        code: u32,
        sampler: &mut FishS2Sampler,
    ) -> Vec<u32> {
        let device = Device::Cpu;
        let rows = FAST_GRAPH_ROWS;
        let policy = sampler.fast_policy().unwrap().unwrap();
        let depths = decoder.cfg.num_codebooks - 1;
        let noise = if policy.samples() {
            let mut noise = vec![0f32; rows * depths * policy.top_k];
            sampler.draw_fast_noise(&mut noise[..depths * policy.top_k]);
            Tensor::from_vec(noise, (rows, depths, policy.top_k), &device).unwrap()
        } else {
            Tensor::zeros(1, DType::F32, &device).unwrap()
        };
        let mut codes = vec![0u32; rows];
        codes[0] = code;
        let inputs = [
            hidden.pad_with_zeros(0, 0, rows - 1).unwrap(),
            Tensor::from_vec(codes, rows, &device).unwrap(),
            noise,
            Tensor::triu2(policy.top_k, DType::F32, &device).unwrap(),
            Tensor::arange(0u32, decoder.cfg.codebook_size as u32, &device).unwrap(),
        ];
        let packed = decoder
            .dense_frame(policy, &inputs)
            .unwrap()
            .to_vec3::<u32>()
            .unwrap();
        assert!(packed[1].iter().flatten().all(|&count| count == 0));
        std::iter::once(code)
            .chain(packed[0].iter().map(|depth| depth[0]))
            .collect()
    }

    /// A graphed frame (rows grouped by policy, chunks of 32 padded with
    /// zero rows, the noise drawn up front) gives every row the bits it gets
    /// alone (in slot 0 of 32 rows, as a lone request runs), one readback
    /// per chunk, two frames running: a row's codes do not depend on the
    /// rows beside it, its slot or its chunk.
    #[test]
    fn graphed_frames_give_each_row_its_frame_alone() {
        for case in 0..GRAPH_CASES.len() {
            let (decoder, hidden, tokens, codes, mut samplers) = graph_case(case);
            let mut alone_samplers = samplers.clone();
            let policies = GRAPH_CASES[case].2.len().min(tokens.len());
            for frame in 0..2 {
                let before = telemetry::thread_host_reads();
                let graphed = decoder
                    .graph_frames_with(
                        &tokens,
                        &codes,
                        &hidden,
                        &mut samplers.iter_mut().collect::<Vec<_>>(),
                        &dense_run(&decoder),
                    )
                    .unwrap()
                    .expect("every policy is graphed");
                let chunks = tokens.len().div_ceil(FAST_GRAPH_ROWS) + policies - 1;
                assert_eq!(
                    telemetry::thread_host_reads() - before,
                    chunks as u64,
                    "case {case}: one readback per chunk"
                );
                for (row, sampler) in alone_samplers.iter_mut().enumerate() {
                    let alone = dense_alone(
                        &decoder,
                        &hidden.narrow(0, row, 1).unwrap(),
                        codes[row],
                        sampler,
                    );
                    assert_eq!(graphed[row].semantic_token_id, tokens[row]);
                    assert_eq!(
                        graphed[row].codebooks, alone,
                        "case {case} frame {frame} row {row}"
                    );
                }
            }
        }
    }

    /// Graphed frames pick the eager paged frame's codes (the dense and
    /// paged attention agree to F32 rounding, so a near tie may rarely
    /// flip one code: at most 1% of them) and leave every sampler's stream
    /// where the eager frame does.
    #[test]
    fn graphed_frames_pick_the_eager_frames_codes() {
        let (mut compared, mut differing) = (0usize, 0usize);
        for case in 0..GRAPH_CASES.len() {
            let (decoder, hidden, tokens, codes, mut samplers) = graph_case(case);
            let cfg = decoder.config().clone();
            let mut eager_samplers = samplers.clone();
            let mut caches = super::super::physical::test_physical_caches(
                906 + case as u64,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.num_codebooks,
                tokens.len(),
            );
            for frame in 0..2 {
                let eager = decoder
                    .generate_frames_batch_with(
                        &tokens,
                        &hidden,
                        &mut eager_samplers.iter_mut().collect::<Vec<_>>(),
                        &mut caches.iter_mut().collect::<Vec<_>>(),
                        true,
                    )
                    .unwrap();
                let graphed = decoder
                    .graph_frames_with(
                        &tokens,
                        &codes,
                        &hidden,
                        &mut samplers.iter_mut().collect::<Vec<_>>(),
                        &dense_run(&decoder),
                    )
                    .unwrap()
                    .expect("every policy is graphed");
                for (graphed, eager) in graphed.iter().zip(&eager) {
                    assert_eq!(graphed.codebooks.len(), eager.codebooks.len());
                    compared += eager.codebooks.len() - 1;
                    differing += graphed
                        .codebooks
                        .iter()
                        .zip(&eager.codebooks)
                        .filter(|(a, b)| a != b)
                        .count();
                }
                assert_eq!(
                    next_draws(&mut samplers),
                    next_draws(&mut eager_samplers),
                    "case {case} frame {frame}: streams"
                );
            }
        }
        assert!(
            differing * 100 <= compared,
            "{differing} of {compared} codes differ from the eager frame's"
        );
    }

    /// A row whose policy only the host serves keeps the frame eager.
    #[test]
    fn a_host_only_policy_keeps_the_frame_eager() {
        let cfg = xorshift_cfg(40);
        let decoder = xorshift_decoder(cfg.clone(), 9);
        let hidden =
            Tensor::zeros((2, 1, cfg.input_hidden_size), DType::F32, &Device::Cpu).unwrap();
        let mut samplers = [
            FishS2Sampler::with_top_k(0.7, 0.7, 30, 1),
            FishS2Sampler::with_top_k(0.7, 0.7, 0, 2),
        ];
        let frames = decoder
            .graph_frames_with(
                &[100, 101],
                &[0, 1],
                &hidden,
                &mut samplers.iter_mut().collect::<Vec<_>>(),
                &dense_run(&decoder),
            )
            .unwrap();
        assert!(frames.is_none());
        assert_eq!(decoder.graphs_diagnostics()["eager_frames"], 1);
    }

    /// Off CUDA the graphs never resolve and price nothing; the kill switch
    /// turns their pricing off on CUDA and names itself.
    #[test]
    fn fast_graphs_run_on_cuda_only_and_obey_the_kill_switch() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os(FISH_S2_FAST_GRAPHS_ENV);
        std::env::remove_var(FISH_S2_FAST_GRAPHS_ENV);
        assert_eq!(fish_s2_fast_graph_device_bytes(BackendKind::Cpu), 0);
        assert_eq!(fish_s2_fast_graph_device_bytes(BackendKind::Metal), 0);
        assert_eq!(
            fish_s2_fast_graph_device_bytes(BackendKind::Cuda),
            128 * MIB
        );
        let mut decoder = xorshift_decoder(xorshift_cfg(40), 1);
        decoder.resolve_graphs(DType::F32);
        assert_eq!(
            decoder.graphs_diagnostics()["off"],
            "shape graphs run on CUDA only"
        );
        std::env::set_var(FISH_S2_FAST_GRAPHS_ENV, "0");
        assert_eq!(fish_s2_fast_graph_device_bytes(BackendKind::Cuda), 0);
        decoder.resolve_graphs(DType::F32);
        let off = decoder.graphs_diagnostics()["off"]
            .as_str()
            .unwrap()
            .to_owned();
        match previous {
            Some(value) => std::env::set_var(FISH_S2_FAST_GRAPHS_ENV, value),
            None => std::env::remove_var(FISH_S2_FAST_GRAPHS_ENV),
        }
        assert!(off.contains(FISH_S2_FAST_GRAPHS_ENV), "{off}");
    }

    #[test]
    fn fast_decoder_generates_full_codebook_frame() {
        let device = Device::Cpu;
        let decoder = tiny_decoder(&device);
        let slow_hidden = Tensor::full(0.5f32, (1, 1, 3), &device).unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 11);
        let cfg = decoder.config();
        let mut cache = super::super::physical::test_physical_cache(
            202,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.num_codebooks,
        );
        let frame = decoder
            .generate_frame(22, &slow_hidden, &mut sampler, &mut cache)
            .expect("frame");
        assert_eq!(frame.semantic_token_id, 22);
        assert_eq!(frame.codebooks.len(), 3);
        assert_eq!(frame.codebooks[0], 2);
        assert!(frame.codebooks[1] < 8);
        assert!(frame.codebooks[2] < 8);
        assert_eq!(cache.context_len(), cfg.num_codebooks);

        let second = decoder
            .generate_frame(23, &slow_hidden, &mut sampler, &mut cache)
            .expect("second frame");
        assert_eq!(second.codebooks.len(), 3);
        assert_eq!(cache.context_len(), cfg.num_codebooks);
    }
}
