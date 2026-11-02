//! The codebook-MTP tower (§3.5), the CPU reference (`fish_omni/tower.py`).
//!
//! 13 dense layers at the trunk's width: `shared_layers` (4) shared ones,
//! then one per residual codebook (9), each with a head after it. They run
//! on every row of the sequence, text included, on top of the trunk's
//! final-norm hidden, at the trunk's positions. Per row `t`, with input
//! frame `F_t` and label (prefill) or predicted (decode) frame `F_{t+1}`:
//!
//! ```text
//! h     = the trunk's final-norm hidden at t
//! lab   = is_semantic(F_{t+1}.main)
//! extra = is_semantic(F_t.main) ? Σ_{k=2..9} CB[k·V + F_t.c_k] : 0      # unscaled, the INPUT frame
//! for l in 0..4:  h = Shared_l((lab ? semEmb_l[c0'] : 0) + (l == 0 ? extra : 0), h, t)
//! for j in 0..9:  h = Residual_j(lab ? cbEmb_j[c'_j] : 0, h, t);  c'_{j+1} = sample(head_j(h))
//!
//! Layer(e, h, t) = final_ln(r + SwiGLU(ffn_ln(r)))
//!   r = m + wo(Attn(attn_ln(m), t) * sigmoid(gate))
//!   m = eh_proj([enorm(e) ; hnorm(h)])
//! ```
//!
//! * `wqkv` rows are contiguous `[Q | Gate | K | V]` (`tower.py:147, 196`),
//!   not the trunk's per-head `[q_h | gate_h]`.
//! * q/k norms are always applied, whatever `attention_qk_norm` says
//!   (`tower.py:164-169`), with every other norm `(1 + w)`, eps 1e-6.
//! * RoPE rotates interleaved pairs `(2i, 2i + 1)` of the first
//!   `rotary_dims` (`tower.py:180-190`), from the family's F64 tables.
//! * Attention is causal over every position `≤ t`, GQA, scale
//!   `1 / sqrt(head_dim)`, through the tower's own paged domain.
//! * Layer `k` of the residual stack injects code `k - 1` of `F_{t+1}` (the
//!   first injects `c0`, as every shared layer does) and its head predicts
//!   code `k`. Codes index the injection tables clamped to `[0, V - 1]`
//!   (`tower.py:312`); a head may emit a code the codec's residual
//!   codebooks do not hold, and it goes back in as is.
//!
//! **Prefill** ([`FishDrama3Tower::forward_span`]) is teacher-forced: prompt
//! row `p` is labelled with row `p + 1`, so the tower runs on the trunk's
//! rows except the last prompt row, which the first decode step completes
//! (`generate.py:245-252`). No head runs.
//!
//! **Decode** ([`FishDrama3Tower::forward_step`]) runs one row per request
//! at its position: the caller samples the main token from the compact head
//! (eos stops before the tower), the tower runs its 13 layers at `t`,
//! sampling each head's code with the row's sampler before the next layer
//! injects it, and returns `F_{t+1}`. On the CPU it is a loop over the rows.
//!
//! The 13 layers share the tower's paged domain ("tower", §7.2, keys stored
//! after RoPE): one prepared append covers them, committed when the forward
//! succeeds and aborted when it fails. A step's rows commit together.
//! Across a request the domain trails the trunk's by one row once the
//! prompt is in ([`check_lengths`]).
//!
//! On the CPU activations and the residual are F32. Projections and heads
//! follow the CPU policy (Q8_0 after the self-check, or F32), the code
//! embeddings stay BF16 on the host, and `extra` reads the trunk
//! embedding's codebook table ([`FishDrama3Embedding::add_codebook_row`]).
//! On Metal and CUDA the layers run in the device column's forms
//! (`device::tower`), and a decode step samples its codes on the device
//! (`device::sampling`).

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Tensor};
use half::bf16;
use serde_json::json;

use super::checkpoint::{
    tower_head, tower_residual_layer, tower_shared_layer, FishDrama3Checkpoint,
};
use super::config::{FishDrama3Config, FishDrama3TokenIds, FishDrama3TowerConfig};
use super::device::sampling::{count_fallback, count_frames, device_cap, read_back};
use super::device::tower::{DeviceCodes, DeviceStepCodes, DeviceTower};
use super::device::DeviceRow;
use super::embed::{bf16_table, FishDrama3Embedding, TRUNK_CODEBOOK};
use super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::precision::{decide_cpu_forms, FishDrama3PrecisionReport, Q8_0_TOLERANCE};
use super::residency::{
    FishDrama3ClassForms, FishDrama3CpuPrecision, FishDrama3ProjectionClass, FishDrama3Residency,
};
use super::sampling::{FishDrama3DrawPolicy, FishDrama3RowSampler};
use super::trunk::attention::FishDrama3Rope;
use super::trunk::{load_linear, norm_gain, rms_norm_rows, sigmoid, silu, NormGain};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{
    PhysicalPagedKvCache, PhysicalPagedKvCheckpoint, PreparedPhysicalPagedStep,
};

/// How a tower load stores its weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3TowerOptions {
    pub precision: FishDrama3CpuPrecision,
    /// The Q8_0 self-check's tolerance ([`Q8_0_TOLERANCE`]).
    pub q8_0_tolerance: f64,
}

impl Default for FishDrama3TowerOptions {
    fn default() -> Self {
        Self {
            precision: FishDrama3CpuPrecision::Q8_0,
            q8_0_tolerance: Q8_0_TOLERANCE,
        }
    }
}

/// One tower layer: a code embedding merged into the hidden state, one
/// pre-norm block, then the final norm (`TowerLayer`, `tower.py:280-318`).
struct TowerLayer {
    /// `semantic_embedding` (shared) or `codebook_embedding` (residual),
    /// `[V, dim]`.
    embedding: Vec<bf16>,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    /// `[dim, 2 dim]`: input columns `0..dim` take the normed embedding,
    /// the rest the normed hidden.
    eh_proj: FishDrama3Linear,
    attention_norm: Vec<f32>,
    /// `[2 q_width + 2 kv_width, dim]`, contiguous `[Q | Gate | K | V]`.
    wqkv: FishDrama3Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: FishDrama3Linear,
    ffn_norm: Vec<f32>,
    w1: FishDrama3Linear,
    w3: FishDrama3Linear,
    w2: FishDrama3Linear,
    final_norm: Vec<f32>,
}

impl TowerLayer {
    fn load(
        checkpoint: &FishDrama3Checkpoint,
        prefix: &str,
        embedding: &str,
        tower: &FishDrama3TowerConfig,
        forms: &FishDrama3ClassForms,
        device: &Device,
    ) -> Result<Self> {
        let d = tower.dim;
        let a = &tower.attention;
        let gain = |name: &str, width: usize| {
            norm_gain(
                checkpoint,
                &format!("{prefix}.{name}.weight"),
                width,
                NormGain::ZeroCentred,
            )
        };
        let linear = |name: &str, rows: usize, cols: usize, class| {
            load_linear(
                checkpoint,
                &format!("{prefix}.{name}.weight"),
                rows,
                cols,
                forms.form(class),
                device,
            )
        };
        use FishDrama3ProjectionClass::{TowerAttention, TowerFeedForward, TowerMerge};
        let v = tower.codebook_vocab;
        Ok(Self {
            embedding: bf16_table(
                checkpoint,
                &format!("{prefix}.{embedding}.weight"),
                [v, d],
                0..v,
            )?,
            enorm: gain("enorm", d)?,
            hnorm: gain("hnorm", d)?,
            eh_proj: linear("eh_proj", d, 2 * d, TowerMerge)?,
            attention_norm: gain("transformer_layer.attention_norm", d)?,
            wqkv: linear(
                "transformer_layer.attention.wqkv",
                tower.wqkv_rows(),
                d,
                TowerAttention,
            )?,
            q_norm: gain("transformer_layer.attention.q_norm", a.head_dim)?,
            k_norm: gain("transformer_layer.attention.k_norm", a.head_dim)?,
            wo: linear(
                "transformer_layer.attention.wo",
                d,
                a.q_width(),
                TowerAttention,
            )?,
            ffn_norm: gain("transformer_layer.ffn_norm", d)?,
            w1: linear(
                "transformer_layer.feed_forward.w1",
                tower.intermediate,
                d,
                TowerFeedForward,
            )?,
            w3: linear(
                "transformer_layer.feed_forward.w3",
                tower.intermediate,
                d,
                TowerFeedForward,
            )?,
            w2: linear(
                "transformer_layer.feed_forward.w2",
                d,
                tower.intermediate,
                TowerFeedForward,
            )?,
            final_norm: gain("final_layernorm", d)?,
        })
    }

    fn projections(&self) -> [&FishDrama3Linear; 6] {
        [
            &self.eh_proj,
            &self.wqkv,
            &self.wo,
            &self.w1,
            &self.w3,
            &self.w2,
        ]
    }

    fn resident_bytes(&self) -> usize {
        let norms = self.enorm.len()
            + self.hnorm.len()
            + self.attention_norm.len()
            + self.q_norm.len()
            + self.k_norm.len()
            + self.ffn_norm.len()
            + self.final_norm.len();
        self.projections()
            .iter()
            .map(|p| p.resident_bytes())
            .sum::<usize>()
            + norms * 4
            + self.embedding.len() * 2
    }

    /// Adds `embedding[clamp(code)]` to `out` (`[dim]`).
    fn add_code(&self, code: u32, out: &mut [f32]) {
        let d = out.len();
        let rows = self.embedding.len() / d;
        let row = (code as usize).min(rows - 1);
        for (o, v) in out.iter_mut().zip(&self.embedding[row * d..(row + 1) * d]) {
            *o += v.to_f32();
        }
    }
}

/// Which of a row's frame codes a layer injects: code 0 for every shared
/// layer, code `j` for residual layer `j`.
fn injected_code(layer: usize, shared_layers: usize) -> usize {
    layer.saturating_sub(shared_layers)
}

/// How a step chooses the codes it injects after `c0`.
enum Codes<'a> {
    /// Each head's code from a working copy of the row's sampler, installed
    /// when the step commits.
    Sample(FishDrama3RowSampler),
    /// The codes of a given frame; each head's logits and each layer's
    /// output are kept.
    Forced {
        label: &'a [u32],
        logits: Vec<f32>,
        outputs: Vec<f32>,
    },
}

/// One request's row in a decode step.
pub struct FishDrama3TowerStepRow<'a> {
    pub cache: &'a mut PhysicalPagedKvCache,
    /// `F_t`, the row the trunk ran at `t`, `[main, c0, …]`.
    pub input: &'a [u32],
    /// The main token sampled for `F_{t+1}`. It must be semantic: on eos
    /// the request stops before the tower.
    pub main: u32,
    pub sampler: &'a mut FishDrama3RowSampler,
}

/// One request's row in a resident decode step
/// ([`FishDrama3Tower::begin_resident_step`]).
pub(crate) struct FishDrama3ResidentTowerRow<'a> {
    pub(crate) cache: &'a mut PhysicalPagedKvCache,
    /// `F_t`, `[main, c0, …]`.
    pub(crate) input: &'a [u32],
    pub(crate) sampler: &'a FishDrama3RowSampler,
}

/// A resident decode step in flight: its appends prepared, its codes on the
/// device, its rows' samplers past their heads' exponentials.
pub(crate) struct FishDrama3ResidentTowerStep {
    prepared: Vec<PreparedPhysicalPagedStep>,
    samplers: Vec<FishDrama3RowSampler>,
    pub(crate) codes: DeviceCodes,
}

/// Each row's heads policy and its heads' exponentials `[heads, rows, V]`
/// F32 (head-major, so each head's rows are one contiguous block; each
/// row's drawn from its sampler in head order), uploaded once.
fn head_noise<'s>(
    samplers: impl Iterator<Item = &'s mut FishDrama3RowSampler>,
    count: usize,
    heads: usize,
    vocab: usize,
    device: &Device,
) -> Result<(Vec<FishDrama3DrawPolicy>, Tensor)> {
    let mut noise = vec![0f32; count * heads * vocab];
    let mut policies = Vec::with_capacity(count);
    for (row, sampler) in samplers.enumerate() {
        policies.push(sampler.params().heads);
        for head in 0..heads {
            let at = (head * count + row) * vocab;
            noise[at..at + vocab].copy_from_slice(&sampler.exponentials(vocab)?);
        }
    }
    if policies.len() != count {
        return Err(Error::InferenceError(format!(
            "{} samplers for {count} tower rows",
            policies.len()
        )));
    }
    super::device::count_uploads(1);
    Ok((
        policies,
        Tensor::from_vec(noise, (heads, count, vocab), device)?,
    ))
}

/// One request's row in a step under given codes (tests and parity runs).
pub struct FishDrama3TowerForcedRow<'a> {
    pub cache: &'a mut PhysicalPagedKvCache,
    /// `F_t`, as in [`FishDrama3TowerStepRow`].
    pub input: &'a [u32],
    /// `F_{t+1}`: its main sets the injection mask and its codes are
    /// injected, exactly as a prefill label's.
    pub label: &'a [u32],
}

/// A rollback point of one request's tower paged view.
pub struct FishDrama3TowerCheckpoint {
    cache: PhysicalPagedKvCheckpoint,
    len: usize,
}

#[derive(Debug, Default)]
struct Counters {
    spans: AtomicU64,
    span_rows: AtomicU64,
    steps: AtomicU64,
    /// Rows a step ran, one tower pass each.
    step_rows: AtomicU64,
}

/// A forward's per-layer outputs, for parity tests.
#[cfg(test)]
pub(crate) type TowerTrace = Vec<Vec<f32>>;

/// The tower's weights in its backend's form.
enum Weights {
    /// The CPU reference.
    Host {
        /// The shared layers, then the residual ones.
        layers: Vec<TowerLayer>,
        /// `[V, dim]` each; head `j` follows residual layer `j`.
        heads: Vec<FishDrama3Linear>,
    },
    /// The device column's forms on Metal or CUDA (`device::tower`).
    Device(DeviceTower),
}

pub struct FishDrama3Tower {
    config: FishDrama3TowerConfig,
    weights: Weights,
    rope: FishDrama3Rope,
    /// Model layers of the paged domain, in slot order.
    attention_layers: Vec<u32>,
    ids: FishDrama3TokenIds,
    max_positions: usize,
    device: Device,
    eps: f32,
    precision: FishDrama3PrecisionReport,
    counters: Counters,
}

impl std::fmt::Debug for FishDrama3Tower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3Tower")
            .field("layers", &self.attention_layers.len())
            .field("heads", &self.config.residual_layers())
            .field("attention_layers", &self.attention_layers)
            .finish()
    }
}

fn clock_error(message: impl Into<String>) -> Error {
    Error::InferenceError(format!("Fish Audio Drama 3 tower: {}", message.into()))
}

/// Checks one request's two paged lengths against §7.2: the tower is level
/// with the trunk while the prompt is still being prefilled, and one row
/// behind once the whole prompt is in (a decode step writes the tower at
/// `t` before the trunk at `t + 1`).
pub fn check_lengths(trunk_len: usize, tower_len: usize, prompt_rows: usize) -> Result<()> {
    if prompt_rows == 0 {
        return Err(clock_error("a request has no prompt rows"));
    }
    let expected = if trunk_len < prompt_rows {
        trunk_len
    } else {
        trunk_len - 1
    };
    if tower_len != expected {
        return Err(clock_error(format!(
            "the tower holds {tower_len} rows beside the trunk's {trunk_len} of a \
             {prompt_rows}-row prompt; expected {expected}"
        )));
    }
    Ok(())
}

impl FishDrama3Tower {
    /// Loads the tower on `device`: on the CPU after the Q8_0 self-check
    /// decides each tower class's form; on Metal or CUDA in the device
    /// forms, which the model's load-time checks may then move per layer.
    pub fn load(
        checkpoint: &FishDrama3Checkpoint,
        config: &FishDrama3Config,
        device: &Device,
        options: FishDrama3TowerOptions,
    ) -> Result<Self> {
        let residency = FishDrama3Residency::of(device);
        residency.require_supported(device)?;
        if residency != FishDrama3Residency::Cpu {
            let forms = FishDrama3ClassForms::of(residency);
            let precision = FishDrama3PrecisionReport {
                forms: FishDrama3ProjectionClass::TOWER
                    .iter()
                    .map(|&class| (class, forms.form(class)))
                    .collect(),
                ..Default::default()
            };
            let weights = DeviceTower::load(checkpoint, config, device)?;
            return Ok(Self::assemble(
                config,
                Weights::Device(weights),
                device,
                precision,
            ));
        }
        let (forms, precision) = decide_cpu_forms(
            checkpoint,
            config,
            options.precision,
            options.q8_0_tolerance,
            &FishDrama3ProjectionClass::TOWER,
        )?;
        let t = config.tower;
        let shared = (0..t.shared_layers).map(|index| {
            TowerLayer::load(
                checkpoint,
                &tower_shared_layer(index),
                "semantic_embedding",
                &t,
                &forms,
                device,
            )
        });
        let residual = (0..t.residual_layers()).map(|index| {
            TowerLayer::load(
                checkpoint,
                &tower_residual_layer(index),
                "codebook_embedding",
                &t,
                &forms,
                device,
            )
        });
        let layers = shared.chain(residual).collect::<Result<Vec<_>>>()?;
        let heads = (0..t.residual_layers())
            .map(|index| {
                load_linear(
                    checkpoint,
                    &tower_head(index),
                    t.codebook_vocab,
                    t.dim,
                    forms.form(FishDrama3ProjectionClass::TowerHead),
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::assemble(
            config,
            Weights::Host { layers, heads },
            device,
            precision,
        ))
    }

    fn assemble(
        config: &FishDrama3Config,
        weights: Weights,
        device: &Device,
        precision: FishDrama3PrecisionReport,
    ) -> Self {
        let t = config.tower;
        let layers = (t.shared_layers + t.residual_layers()) as u32;
        // After the trunk's layers, as sglang numbers them (40..52), so the
        // two paged domains of one group never share a model layer.
        let first = config.trunk.layers.len() as u32;
        let tower = Self {
            config: t,
            weights,
            attention_layers: (first..first + layers).collect(),
            rope: FishDrama3Rope::new(&t.attention),
            ids: config.ids,
            max_positions: config.trunk.max_positions,
            device: device.clone(),
            eps: t.norm_eps as f32,
            precision,
            counters: Counters::default(),
        };
        tracing::info!(
            "Loaded the Fish Audio Drama 3 tower on {:?}: {:.2} GiB, {} Q8_0 classes kept F32",
            device.location(),
            tower.resident_bytes() as f64 / (1u64 << 30) as f64,
            tower.precision.q8_fallback_classes
        );
        tower
    }

    /// The device weights, for the load-time checks.
    pub(crate) fn device_tower(&self) -> Option<&DeviceTower> {
        match &self.weights {
            Weights::Device(weights) => Some(weights),
            Weights::Host { .. } => None,
        }
    }

    pub(crate) fn device_tower_mut(&mut self) -> Option<&mut DeviceTower> {
        match &mut self.weights {
            Weights::Device(weights) => Some(weights),
            Weights::Host { .. } => None,
        }
    }

    pub(crate) fn precision_mut(&mut self) -> &mut FishDrama3PrecisionReport {
        &mut self.precision
    }

    /// The CPU layers and heads (the CPU path and tests).
    fn host(&self) -> Result<(&[TowerLayer], &[FishDrama3Linear])> {
        match &self.weights {
            Weights::Host { layers, heads } => Ok((layers, heads)),
            Weights::Device(_) => Err(Error::InferenceError(
                "the Fish Audio Drama 3 host tower runs the CPU weights".into(),
            )),
        }
    }

    pub fn config(&self) -> &FishDrama3TowerConfig {
        &self.config
    }

    /// The model layers the tower's paged domain binds, in slot order.
    pub fn attention_layers(&self) -> &[u32] {
        &self.attention_layers
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn precision_report(&self) -> &FishDrama3PrecisionReport {
        &self.precision
    }

    /// Resident bytes of the weights (the code embeddings on the host on
    /// the CPU, on the device on Metal and CUDA).
    pub fn resident_bytes(&self) -> usize {
        match &self.weights {
            Weights::Host { layers, heads } => {
                layers.iter().map(TowerLayer::resident_bytes).sum::<usize>()
                    + heads.iter().map(|h| h.resident_bytes()).sum::<usize>()
            }
            Weights::Device(weights) => weights.resident_bytes(),
        }
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        let c = &self.counters;
        json!({
            "resident_bytes": self.resident_bytes(),
            "precision": self.precision.diagnostics(),
            "spans": c.spans.load(Ordering::Relaxed),
            "span_rows": c.span_rows.load(Ordering::Relaxed),
            "steps": c.steps.load(Ordering::Relaxed),
            "step_rows": c.step_rows.load(Ordering::Relaxed),
            "decode_graphs": self
                .device_tower()
                .map_or(serde_json::Value::Null, DeviceTower::graph_diagnostics),
        })
    }

    /// Rows a step has run so far, one tower pass each.
    pub fn step_rows(&self) -> u64 {
        self.counters.step_rows.load(Ordering::Relaxed)
    }

    /// The prompt rows a prefill span `span` of the trunk teacher-forces
    /// through the tower: the span's rows, short of the last prompt row,
    /// which has no label until the first decode step (`generate.py:245`).
    /// It may be empty.
    pub fn teacher_forced_rows(span: Range<usize>, prompt_rows: usize) -> Range<usize> {
        span.start..span.end.min(prompt_rows.saturating_sub(1)).max(span.start)
    }

    /// The cache must bind exactly the tower's layers.
    pub fn validate_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        let a = &self.config.attention;
        cache.validate_sparse_model(&self.attention_layers, a.kv_heads, a.head_dim, a.head_dim)
    }

    /// Checks a forward of `n` rows at `start` against `cache`.
    fn check_clocks(&self, n: usize, start: usize, cache: &PhysicalPagedKvCache) -> Result<()> {
        self.validate_cache(cache)?;
        if cache.context_len() != start {
            return Err(clock_error(format!(
                "rows start at {start}; the KV holds {}",
                cache.context_len()
            )));
        }
        let end = start + n;
        if n == 0 || end > self.max_positions || end > cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 tower rows {start}..{end} exceed the {} positions or the KV's {}",
                self.max_positions,
                cache.capacity_tokens()
            )));
        }
        Ok(())
    }

    fn host_rows(&self, hidden: &Tensor, rows: Option<usize>) -> Result<(Vec<f32>, usize)> {
        let (n, width) = hidden.dims2()?;
        if width != self.config.dim || rows.is_some_and(|rows| rows != n) {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 tower hidden rows are {:?}; expected [{}, {}]",
                hidden.dims(),
                rows.map_or("n".to_string(), |rows| rows.to_string()),
                self.config.dim
            )));
        }
        Ok((
            hidden
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?,
            n,
        ))
    }

    /// Checks one frame's width.
    fn frame<'a>(&self, values: &'a [u32], what: &str) -> Result<&'a [u32]> {
        if values.len() != self.config.frame_width() {
            return Err(Error::InvalidInput(format!(
                "a Fish Audio Drama 3 {what} frame holds {} values, not {}",
                values.len(),
                self.config.frame_width()
            )));
        }
        Ok(values)
    }

    /// `extra` of input frame `input` into `out` (`[dim]`, zeroed): the
    /// unscaled sum of codebooks 2.. when its main is semantic, else zero.
    fn add_extra(
        &self,
        embedding: &FishDrama3Embedding,
        input: &[u32],
        out: &mut [f32],
    ) -> Result<()> {
        if self.ids.is_semantic(input[0]) {
            for codebook in TRUNK_CODEBOOK + 1..self.config.codebooks {
                embedding.add_codebook_row(codebook, input[1 + codebook], out)?;
            }
        }
        Ok(())
    }

    fn check_embedding(&self, embedding: &FishDrama3Embedding) -> Result<()> {
        if embedding.hidden() != self.config.dim
            || embedding.frame_width() != self.config.frame_width()
        {
            return Err(Error::InvalidInput(format!(
                "the embedding is {} wide over frames of {}; the tower is {} over {}",
                embedding.hidden(),
                embedding.frame_width(),
                self.config.dim,
                self.config.frame_width()
            )));
        }
        Ok(())
    }

    /// The mixer output `[n, dim]` of the normed rows `x` at positions
    /// `start..start + n`; their K/V go into cache layer `slot` through
    /// `prepared`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &TowerLayer,
        x: &[f32],
        n: usize,
        start: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        slot: usize,
        phase: FishDrama3KernelPhase,
    ) -> Result<Vec<f32>> {
        let a = &self.config.attention;
        let (heads, kv_heads, d) = (a.heads, a.kv_heads, a.head_dim);
        let (q_width, kv_width) = (a.q_width(), a.kv_width());
        let packed = layer.wqkv.forward_host(x, n, phase)?;
        // Contiguous blocks per row: Q, Gate, K, V.
        let mut queries = Vec::with_capacity(n * q_width);
        let mut gate = Vec::with_capacity(n * q_width);
        let mut keys = Vec::with_capacity(n * kv_width);
        let mut values = Vec::with_capacity(n * kv_width);
        for row in packed.chunks_exact(2 * q_width + 2 * kv_width) {
            let (q, rest) = row.split_at(q_width);
            let (g, rest) = rest.split_at(q_width);
            let (k, v) = rest.split_at(kv_width);
            queries.extend_from_slice(q);
            gate.extend_from_slice(g);
            keys.extend_from_slice(k);
            values.extend_from_slice(v);
        }
        let mut queries = rms_norm_rows(&queries, d, &layer.q_norm, self.eps);
        let mut keys = rms_norm_rows(&keys, d, &layer.k_norm, self.eps);
        for t in 0..n {
            let (cos, sin) = self.rope.cos_sin(start + t);
            self.rope
                .apply(&mut queries[t * q_width..(t + 1) * q_width], &cos, &sin);
            self.rope
                .apply(&mut keys[t * kv_width..(t + 1) * kv_width], &cos, &sin);
        }
        let dtype = cache.arena().config().dtype;
        let tensor = |values: Vec<f32>, heads: usize| -> Result<Tensor> {
            Ok(Tensor::from_vec(values, (n, heads, d), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&self.device)?)
        };
        let scale = (1.0 / (d as f64).sqrt()) as f32;
        let attended = cache.write_and_attend(
            slot,
            prepared,
            &tensor(queries, heads)?,
            &tensor(keys, kv_heads)?,
            &tensor(values, kv_heads)?,
            scale,
        )?;
        let mut attended = attended
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        if attended.len() != gate.len() {
            return Err(Error::InferenceError(format!(
                "tower attention returned {} values for {} gates",
                attended.len(),
                gate.len()
            )));
        }
        for (o, g) in attended.iter_mut().zip(&gate) {
            *o *= sigmoid(*g);
        }
        layer.wo.forward_host(&attended, n, phase)
    }

    /// `layer` over `n` rows at `start`, its K/V in cache layer `slot`:
    /// injections `e` and hidden `h` (`[n, dim]` each) to the layer's
    /// final-norm output `[n, dim]`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        layer: &TowerLayer,
        slot: usize,
        e: &[f32],
        h: &[f32],
        n: usize,
        start: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        phase: FishDrama3KernelPhase,
    ) -> Result<Vec<f32>> {
        let d = self.config.dim;
        let e = rms_norm_rows(e, d, &layer.enorm, self.eps);
        let h = rms_norm_rows(h, d, &layer.hnorm, self.eps);
        let mut merged = Vec::with_capacity(2 * n * d);
        for (e, h) in e.chunks_exact(d).zip(h.chunks_exact(d)) {
            merged.extend_from_slice(e);
            merged.extend_from_slice(h);
        }
        let mut x = layer.eh_proj.forward_host(&merged, n, phase)?;
        let normed = rms_norm_rows(&x, d, &layer.attention_norm, self.eps);
        let mixed = self.attention(layer, &normed, n, start, cache, prepared, slot, phase)?;
        x.iter_mut().zip(&mixed).for_each(|(x, m)| *x += m);
        let normed = rms_norm_rows(&x, d, &layer.ffn_norm, self.eps);
        let gate = layer.w1.forward_host(&normed, n, phase)?;
        let mut up = layer.w3.forward_host(&normed, n, phase)?;
        up.iter_mut().zip(&gate).for_each(|(u, g)| *u *= silu(*g));
        let down = layer.w2.forward_host(&up, n, phase)?;
        x.iter_mut().zip(&down).for_each(|(x, f)| *x += f);
        Ok(rms_norm_rows(&x, d, &layer.final_norm, self.eps))
    }

    /// Codebook head `index`'s F32 logits `[n, V]` of layer outputs `h`.
    pub(crate) fn head_logits(
        &self,
        index: usize,
        h: &[f32],
        n: usize,
        phase: FishDrama3KernelPhase,
    ) -> Result<Vec<f32>> {
        if let Weights::Device(weights) = &self.weights {
            let h = Tensor::from_slice(h, (n, self.config.dim), &self.device)?;
            return Ok(weights
                .head_logits(index, &h, phase, None)?
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?);
        }
        self.host()?.1[index].forward_host(h, n, phase)
    }

    /// The layers over `n` teacher-forced rows: `inputs` and `labels` hold
    /// `n` frames each.
    #[allow(clippy::too_many_arguments)]
    fn run_span(
        &self,
        embedding: &FishDrama3Embedding,
        mut x: Vec<f32>,
        inputs: &[u32],
        labels: &[u32],
        n: usize,
        start: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<()> {
        let (layers, _) = self.host()?;
        let d = self.config.dim;
        let width = self.config.frame_width();
        let mut extra = vec![0f32; n * d];
        for (input, out) in inputs.chunks_exact(width).zip(extra.chunks_exact_mut(d)) {
            self.add_extra(embedding, input, out)?;
        }
        for slot in 0..layers.len() {
            let column = 1 + injected_code(slot, self.config.shared_layers);
            let mut e = if slot == 0 {
                extra.clone()
            } else {
                vec![0f32; n * d]
            };
            for (label, out) in labels.chunks_exact(width).zip(e.chunks_exact_mut(d)) {
                if self.ids.is_semantic(label[0]) {
                    layers[slot].add_code(label[column], out);
                }
            }
            x = self.layer(
                &layers[slot],
                slot,
                &e,
                &x,
                n,
                start,
                cache,
                prepared,
                FishDrama3KernelPhase::Span,
            )?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(x.clone());
            }
        }
        Ok(())
    }

    fn span(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        frames: &[u32],
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<()> {
        self.check_embedding(embedding)?;
        if let Weights::Device(weights) = &self.weights {
            return self.span_device(weights, embedding, hidden, frames, start_pos, cache, trace);
        }
        let (x, n) = self.host_rows(hidden, None)?;
        let width = self.config.frame_width();
        if frames.len() != (n + 1) * width {
            return Err(Error::InvalidInput(format!(
                "a Fish Audio Drama 3 tower span of {n} rows takes {} frames, its rows and the \
                 next one; got {} values",
                n + 1,
                frames.len()
            )));
        }
        self.check_clocks(n, start_pos, cache)?;
        let mut prepared = cache.prepare_append(start_pos, n)?;
        match self.run_span(
            embedding,
            x,
            &frames[..n * width],
            &frames[width..],
            n,
            start_pos,
            cache,
            &mut prepared,
            trace,
        ) {
            Ok(()) => {
                cache.commit_prepared(prepared)?;
                self.counters.spans.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .span_rows
                    .fetch_add(n as u64, Ordering::Relaxed);
                Ok(())
            }
            Err(error) => {
                let _ = cache.abort_prepared(prepared);
                Err(error)
            }
        }
    }

    /// Prefill, teacher-forced: the trunk's final-norm `hidden` `[n, dim]`
    /// of one request's rows `start_pos..start_pos + n`, and `frames`, the
    /// `n + 1` rows `start_pos..start_pos + n + 1` (`[main, c0, …]` each):
    /// row `i` is row `i`'s input frame and row `i + 1` its label. The
    /// cache must stand at `start_pos`; it gains the `n` rows. A request's
    /// spans cover [`Self::teacher_forced_rows`] of the trunk's. On an error
    /// nothing moves.
    pub fn forward_span(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        frames: &[u32],
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<()> {
        self.span(embedding, hidden, frames, start_pos, cache, None)
    }

    /// [`Self::forward_span`] recording every layer's output.
    #[cfg(test)]
    pub(crate) fn forward_span_traced(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        frames: &[u32],
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<TowerTrace> {
        let mut trace = Vec::new();
        self.span(
            embedding,
            hidden,
            frames,
            start_pos,
            cache,
            Some(&mut trace),
        )?;
        Ok(trace)
    }

    /// One row through every layer at `position`: the injection mask and
    /// `c0` from the label or the sampled main, the later codes from
    /// `codes`. Returns the frame `[main, c0, …]`.
    #[allow(clippy::too_many_arguments)]
    fn run_row(
        &self,
        embedding: &FishDrama3Embedding,
        mut x: Vec<f32>,
        input: &[u32],
        main: u32,
        codes: &mut Codes<'_>,
        position: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
    ) -> Result<Vec<u32>> {
        let d = self.config.dim;
        let shared = self.config.shared_layers;
        let (labelled, c0) = match codes {
            Codes::Sample(_) => (true, main - self.ids.semantic_start),
            Codes::Forced { label, .. } => (self.ids.is_semantic(main), label[1]),
        };
        let (layers, _) = self.host()?;
        let mut frame = Vec::with_capacity(self.config.frame_width());
        frame.extend([main, c0]);
        let mut extra = vec![0f32; d];
        self.add_extra(embedding, input, &mut extra)?;
        for slot in 0..layers.len() {
            let mut e = if slot == 0 {
                std::mem::take(&mut extra)
            } else {
                vec![0f32; d]
            };
            if labelled {
                layers[slot].add_code(frame[1 + injected_code(slot, shared)], &mut e);
            }
            x = self.layer(
                &layers[slot],
                slot,
                &e,
                &x,
                1,
                position,
                cache,
                prepared,
                FishDrama3KernelPhase::Step,
            )?;
            if let Codes::Forced { outputs, .. } = codes {
                outputs.extend_from_slice(&x);
            }
            if slot >= shared {
                let head = slot - shared;
                let logits = self.head_logits(head, &x, 1, FishDrama3KernelPhase::Step)?;
                frame.push(match codes {
                    Codes::Sample(sampler) => sampler.sample_code(&logits)?,
                    Codes::Forced {
                        label,
                        logits: kept,
                        ..
                    } => {
                        kept.extend(logits);
                        label[2 + head]
                    }
                });
            }
        }
        Ok(frame)
    }

    /// Decode over `rows` (cache, input frame, main, codes), row `r` on
    /// hidden row `r`: each row runs alone at its KV length; they commit
    /// together, and on an error none moves. Returns each row's frame and
    /// its codes source.
    fn steps<'a>(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        rows: Vec<(&mut PhysicalPagedKvCache, &[u32], u32, Codes<'a>)>,
    ) -> Result<Vec<(Vec<u32>, Codes<'a>)>> {
        self.check_embedding(embedding)?;
        if let Weights::Device(weights) = &self.weights {
            return self.steps_device(weights, embedding, hidden, rows);
        }
        let (x, count) = self.host_rows(hidden, Some(rows.len()))?;
        if count == 0 {
            return Err(Error::InvalidInput("a decode step needs a row".into()));
        }
        let d = self.config.dim;
        let mut done = Vec::with_capacity(count);
        let mut failure = None;
        for (r, (cache, input, main, mut codes)) in rows.into_iter().enumerate() {
            let position = cache.context_len();
            let attempt = (|| {
                self.frame(input, "input")?;
                if let Codes::Forced { label, .. } = &codes {
                    self.frame(label, "label")?;
                } else if !self.ids.is_semantic(main) {
                    return Err(Error::InvalidInput(format!(
                        "a Fish Audio Drama 3 frame's main token {main} is not semantic; \
                         eos stops before the tower"
                    )));
                }
                self.check_clocks(1, position, cache)?;
                cache.prepare_append(position, 1)
            })();
            let mut prepared = match attempt {
                Ok(prepared) => prepared,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            };
            match self.run_row(
                embedding,
                x[r * d..(r + 1) * d].to_vec(),
                input,
                main,
                &mut codes,
                position,
                cache,
                &mut prepared,
            ) {
                Ok(frame) => {
                    self.counters.step_rows.fetch_add(1, Ordering::Relaxed);
                    done.push((cache, prepared, frame, codes));
                }
                Err(error) => {
                    let _ = cache.abort_prepared(prepared);
                    failure = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = failure {
            for (cache, prepared, _, _) in done {
                let _ = cache.abort_prepared(prepared);
            }
            return Err(error);
        }
        let mut out = Vec::with_capacity(count);
        for (cache, prepared, frame, codes) in done {
            cache.commit_prepared(prepared)?;
            out.push((frame, codes));
        }
        self.counters.steps.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }

    /// Decode: row `r` of `hidden` `[rows.len(), dim]` is the trunk's
    /// final-norm hidden of request `rows[r]` at its position `t` (its tower
    /// KV length). Runs the 13 layers at `t`, sampling codes 1.. with the
    /// row's sampler after each head, and returns the predicted frames
    /// `F_{t+1}` row-major (`[main, c0, …]` each, the embedding's input
    /// layout). Each row runs alone; the rows commit together (KV and
    /// samplers), and on an error none moves.
    pub fn forward_step(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        rows: &mut [FishDrama3TowerStepRow<'_>],
    ) -> Result<Vec<u32>> {
        let jobs = rows
            .iter_mut()
            .map(|row| {
                (
                    &mut *row.cache,
                    row.input,
                    row.main,
                    Codes::Sample(row.sampler.clone()),
                )
            })
            .collect();
        let out = self.steps(embedding, hidden, jobs)?;
        let mut frames = Vec::with_capacity(out.len() * self.config.frame_width());
        for (row, (frame, codes)) in rows.iter_mut().zip(out) {
            if let Codes::Sample(sampler) = codes {
                *row.sampler = sampler;
            }
            frames.extend(frame);
        }
        Ok(frames)
    }

    /// [`Self::forward_step`] injecting each row's `label` codes instead of
    /// sampling (a teacher-forced decode step: it writes and commits the
    /// KV the same way). Returns every head's logits per row, row-major
    /// `[rows, heads, V]` F32.
    pub fn forward_step_forced(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        rows: &mut [FishDrama3TowerForcedRow<'_>],
    ) -> Result<Vec<f32>> {
        Ok(self.steps_forced(embedding, hidden, rows)?.0)
    }

    /// [`Self::forward_step_forced`] also returning every layer's output per
    /// row, row-major `[rows, layers, dim]`.
    #[cfg(test)]
    pub(crate) fn forward_step_forced_traced(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        rows: &mut [FishDrama3TowerForcedRow<'_>],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.steps_forced(embedding, hidden, rows)
    }

    /// The head logits and the layer outputs of a forced step.
    fn steps_forced(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        rows: &mut [FishDrama3TowerForcedRow<'_>],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let jobs = rows
            .iter_mut()
            .map(|row| {
                (
                    &mut *row.cache,
                    row.input,
                    row.label.first().copied().unwrap_or_default(),
                    Codes::Forced {
                        label: row.label,
                        logits: Vec::new(),
                        outputs: Vec::new(),
                    },
                )
            })
            .collect();
        let (mut logits, mut outputs) = (Vec::new(), Vec::new());
        for (_, codes) in self.steps(embedding, hidden, jobs)? {
            if let Codes::Forced {
                logits: row,
                outputs: layers,
                ..
            } = codes
            {
                logits.extend(row);
                outputs.extend(layers);
            }
        }
        Ok((logits, outputs))
    }

    #[allow(clippy::too_many_arguments)]
    fn span_device(
        &self,
        weights: &DeviceTower,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        frames: &[u32],
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<()> {
        let (n, width) = hidden.dims2()?;
        let fw = self.config.frame_width();
        if width != self.config.dim || frames.len() != (n + 1) * fw {
            return Err(Error::InvalidInput(format!(
                "a Fish Audio Drama 3 tower span of {:?} hidden rows takes {} frames; got {} \
                 values",
                hidden.dims(),
                n + 1,
                frames.len()
            )));
        }
        self.check_clocks(n, start_pos, cache)?;
        let hidden = hidden.to_device(&self.device)?.to_dtype(DType::F32)?;
        let mut prepared = cache.prepare_append(start_pos, n)?;
        let result = {
            let mut rows = [DeviceRow {
                start: start_pos,
                rows: n,
                cache: &*cache,
                prepared: &mut prepared,
            }];
            weights.span(
                embedding,
                &hidden,
                &frames[..n * fw],
                &frames[fw..],
                fw,
                |main| self.ids.is_semantic(main),
                &mut rows,
            )
        };
        match result {
            Ok(outputs) => {
                cache.commit_prepared(prepared)?;
                self.counters.spans.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .span_rows
                    .fetch_add(n as u64, Ordering::Relaxed);
                if let Some(trace) = trace {
                    for output in outputs {
                        trace.push(
                            output
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_device(&Device::Cpu)?
                                .to_vec1::<f32>()?,
                        );
                    }
                }
                Ok(())
            }
            Err(error) => {
                let _ = cache.abort_prepared(prepared);
                Err(error)
            }
        }
    }

    /// A decode step on a device. Sampled rows draw their codes on the device
    /// (each row's exponentials taken from its sampler in head order, as
    /// the host draws them) unless a row's head cap exceeds the device's
    /// ranks; forced rows inject their label's codes.
    fn steps_device<'a>(
        &self,
        weights: &DeviceTower,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        rows: Vec<(&mut PhysicalPagedKvCache, &[u32], u32, Codes<'a>)>,
    ) -> Result<Vec<(Vec<u32>, Codes<'a>)>> {
        let count = rows.len();
        let (n, width) = hidden.dims2()?;
        if count == 0 || n != count || width != self.config.dim {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 tower hidden rows are {:?} for {count} requests",
                hidden.dims()
            )));
        }
        let fw = self.config.frame_width();
        let mut caches = Vec::with_capacity(count);
        let mut prepared = Vec::with_capacity(count);
        let mut sources = Vec::with_capacity(count);
        let mut inputs = Vec::with_capacity(count * fw);
        let mut mains = Vec::with_capacity(count);
        let mut failure = None;
        for (cache, input, main, codes) in rows {
            let position = cache.context_len();
            let attempt = (|| {
                self.frame(input, "input")?;
                if let Codes::Forced { label, .. } = &codes {
                    self.frame(label, "label")?;
                } else if !self.ids.is_semantic(main) {
                    return Err(Error::InvalidInput(format!(
                        "a Fish Audio Drama 3 frame's main token {main} is not semantic; \
                         eos stops before the tower"
                    )));
                }
                self.check_clocks(1, position, cache)?;
                cache.prepare_append(position, 1)
            })();
            match attempt {
                Ok(step) => {
                    prepared.push(step);
                    caches.push(cache);
                    inputs.extend_from_slice(input);
                    mains.push(main);
                    sources.push(codes);
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = failure {
            for (cache, step) in caches.into_iter().zip(prepared) {
                let _ = cache.abort_prepared(step);
            }
            return Err(error);
        }
        let mut c0 = Vec::with_capacity(count);
        let mut labelled = Vec::with_capacity(count);
        for (main, source) in mains.iter().zip(&sources) {
            match source {
                Codes::Sample(_) => {
                    c0.push(main - self.ids.semantic_start);
                    labelled.push(true);
                }
                Codes::Forced { label, .. } => {
                    c0.push(label[1]);
                    labelled.push(self.ids.is_semantic(*main));
                }
            }
        }
        let resident = sources.iter().all(|source| match source {
            Codes::Sample(sampler) => device_cap(&sampler.params().heads).is_some(),
            Codes::Forced { .. } => false,
        });
        let vocab = self.config.codebook_vocab;
        let heads = weights.heads();
        let hidden = hidden.to_device(&self.device)?.to_dtype(DType::F32)?;
        let result = (|| {
            let mut device_rows = caches
                .iter()
                .zip(prepared.iter_mut())
                .map(|(cache, step)| DeviceRow {
                    start: cache.context_len(),
                    rows: 1,
                    cache,
                    prepared: step,
                })
                .collect::<Vec<_>>();
            let first = weights.codes(c0.iter().copied(), &self.device)?;
            let labelled = DeviceTower::labelled(&labelled, &self.device)?;
            if resident {
                let samplers = sources.iter_mut().filter_map(|source| match source {
                    Codes::Sample(sampler) => Some(sampler),
                    Codes::Forced { .. } => None,
                });
                let (policies, noise) = head_noise(samplers, count, heads, vocab, &self.device)?;
                let step = weights.step(
                    embedding,
                    &hidden,
                    &inputs,
                    &first,
                    &labelled,
                    &mut device_rows,
                    DeviceStepCodes::Device { policies, noise },
                )?;
                let codes = step.device.as_ref().ok_or_else(|| {
                    Error::InferenceError("a device tower step drew no codes".into())
                })?;
                // Readback 2: every head's pick, then every draw's non-finite
                // count, in one transfer.
                let values = read_back(&codes.packed()?)?;
                return Ok((DeviceCodes::rows(&values, count)?, step.outputs));
            }
            let mut pick = |head: usize, logits: &Tensor| -> Result<Vec<u32>> {
                let host = logits.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
                sources
                    .iter_mut()
                    .zip(host)
                    .map(|(source, row)| match source {
                        Codes::Sample(sampler) => {
                            count_fallback();
                            sampler.sample_code(&row)
                        }
                        Codes::Forced {
                            label,
                            logits: kept,
                            ..
                        } => {
                            kept.extend(row);
                            Ok(label[2 + head])
                        }
                    })
                    .collect()
            };
            let step = weights.step(
                embedding,
                &hidden,
                &inputs,
                &first,
                &labelled,
                &mut device_rows,
                DeviceStepCodes::Host(&mut pick),
            )?;
            Ok((step.host_codes, step.outputs))
        })();
        let (codes, outputs) = match result {
            Ok(done) => done,
            Err(error) => {
                for (cache, step) in caches.into_iter().zip(prepared) {
                    let _ = cache.abort_prepared(step);
                }
                return Err(error);
            }
        };
        if sources
            .iter()
            .any(|source| matches!(source, Codes::Forced { .. }))
        {
            let layers = outputs
                .iter()
                .map(|output| -> Result<Vec<Vec<f32>>> {
                    Ok(output
                        .to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .to_vec2::<f32>()?)
                })
                .collect::<Result<Vec<_>>>()?;
            for (r, source) in sources.iter_mut().enumerate() {
                if let Codes::Forced { outputs, .. } = source {
                    for layer in &layers {
                        outputs.extend_from_slice(&layer[r]);
                    }
                }
            }
        }
        if resident {
            count_frames(count);
        }
        let mut out = Vec::with_capacity(count);
        for (((cache, step), source), (main, (c0, codes))) in caches
            .into_iter()
            .zip(prepared)
            .zip(sources)
            .zip(mains.iter().zip(c0.iter().zip(codes)))
        {
            cache.commit_prepared(step)?;
            let mut frame = Vec::with_capacity(fw);
            frame.extend([*main, *c0]);
            frame.extend(codes);
            out.push((frame, source));
        }
        self.counters
            .step_rows
            .fetch_add(count as u64, Ordering::Relaxed);
        self.counters.steps.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }

    /// A decode step whose codes stay on the device (the model's
    /// one-readback frame): row `r` on hidden row `r`, `c0` (`[m]` U32,
    /// clamped) each row's code 0. Each row draws its heads from a working
    /// copy of its sampler; the appends stay prepared until
    /// [`Self::finish_resident_step`] or [`Self::abort_resident_step`]. On an
    /// error nothing moves.
    pub(crate) fn begin_resident_step(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        c0: &Tensor,
        rows: &mut [FishDrama3ResidentTowerRow<'_>],
    ) -> Result<FishDrama3ResidentTowerStep> {
        self.check_embedding(embedding)?;
        let Weights::Device(weights) = &self.weights else {
            return Err(Error::InferenceError(
                "a resident tower step needs device weights".into(),
            ));
        };
        let count = rows.len();
        if count == 0 || hidden.dims() != [count, self.config.dim] || c0.dims() != [count] {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 resident tower rows are {:?} and {:?} for {count} requests",
                hidden.dims(),
                c0.dims()
            )));
        }
        let mut prepared = Vec::with_capacity(count);
        let mut inputs = Vec::with_capacity(count * self.config.frame_width());
        for row in rows.iter() {
            let position = row.cache.context_len();
            let attempt = (|| {
                self.frame(row.input, "input")?;
                if device_cap(&row.sampler.params().heads).is_none() {
                    return Err(Error::InvalidInput(
                        "a resident tower step needs device-capped heads".into(),
                    ));
                }
                self.check_clocks(1, position, row.cache)?;
                row.cache.prepare_append(position, 1)
            })();
            match attempt {
                Ok(step) => {
                    prepared.push(step);
                    inputs.extend_from_slice(row.input);
                }
                Err(error) => {
                    for (row, step) in rows.iter_mut().zip(prepared) {
                        let _ = row.cache.abort_prepared(step);
                    }
                    return Err(error);
                }
            }
        }
        let mut samplers = rows
            .iter()
            .map(|row| row.sampler.clone())
            .collect::<Vec<_>>();
        let hidden = hidden.to_dtype(DType::F32)?;
        let result = (|| {
            let (policies, noise) = head_noise(
                samplers.iter_mut(),
                count,
                weights.heads(),
                self.config.codebook_vocab,
                &self.device,
            )?;
            let labelled = DeviceTower::labelled(&vec![true; count], &self.device)?;
            let mut device_rows = rows
                .iter()
                .zip(prepared.iter_mut())
                .map(|(row, step)| DeviceRow {
                    start: row.cache.context_len(),
                    rows: 1,
                    cache: &*row.cache,
                    prepared: step,
                })
                .collect::<Vec<_>>();
            weights.step(
                embedding,
                &hidden,
                &inputs,
                c0,
                &labelled,
                &mut device_rows,
                DeviceStepCodes::Device { policies, noise },
            )
        })();
        let codes = match result.and_then(|step| {
            step.device
                .ok_or_else(|| Error::InferenceError("a resident tower step drew no codes".into()))
        }) {
            Ok(codes) => codes,
            Err(error) => {
                for (row, step) in rows.iter_mut().zip(prepared) {
                    let _ = row.cache.abort_prepared(step);
                }
                return Err(error);
            }
        };
        Ok(FishDrama3ResidentTowerStep {
            prepared,
            samplers,
            codes,
        })
    }

    /// Commits the rows of `step` where `keep` (installing their samplers)
    /// and aborts the others; `caches` are its rows' in order.
    pub(crate) fn finish_resident_step(
        &self,
        step: FishDrama3ResidentTowerStep,
        caches: &mut [&mut PhysicalPagedKvCache],
        samplers: &mut [&mut FishDrama3RowSampler],
        keep: &[bool],
    ) -> Result<()> {
        let count = step.prepared.len();
        if caches.len() != count || samplers.len() != count || keep.len() != count {
            return Err(Error::InferenceError(
                "a resident tower step finishes with other rows than it began".into(),
            ));
        }
        let mut failure = None;
        let mut kept = 0;
        for ((((cache, prepared), sampler), working), &keep) in caches
            .iter_mut()
            .zip(step.prepared)
            .zip(samplers.iter_mut())
            .zip(step.samplers)
            .zip(keep)
        {
            let outcome = if keep && failure.is_none() {
                kept += 1;
                **sampler = working;
                cache.commit_prepared(prepared)
            } else {
                cache.abort_prepared(prepared)
            };
            if let Err(error) = outcome {
                failure.get_or_insert(error);
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        count_frames(kept);
        self.counters
            .step_rows
            .fetch_add(kept as u64, Ordering::Relaxed);
        self.counters.steps.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Aborts every row of `step`.
    pub(crate) fn abort_resident_step(
        &self,
        step: FishDrama3ResidentTowerStep,
        caches: &mut [&mut PhysicalPagedKvCache],
    ) {
        for (cache, prepared) in caches.iter_mut().zip(step.prepared) {
            let _ = cache.abort_prepared(prepared);
        }
    }

    /// Begins a quantum: a rollback point of `cache`, the request's tower
    /// paged view.
    pub fn begin_quantum(&self, cache: &PhysicalPagedKvCache) -> Result<FishDrama3TowerCheckpoint> {
        self.validate_cache(cache)?;
        Ok(FishDrama3TowerCheckpoint {
            cache: cache.logical_checkpoint(),
            len: cache.context_len(),
        })
    }

    /// Accepts a quantum: the view never moved backwards.
    pub fn commit_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &FishDrama3TowerCheckpoint,
    ) -> Result<()> {
        self.validate_cache(cache)?;
        if cache.context_len() < checkpoint.len {
            return Err(clock_error("a quantum moved the KV backwards"));
        }
        Ok(())
    }

    /// Rewinds `cache` to `checkpoint`.
    pub fn abort_quantum(
        &self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: FishDrama3TowerCheckpoint,
    ) -> Result<()> {
        cache.restore_logical_checkpoint(checkpoint.cache)
    }
}

#[cfg(test)]
mod tests {
    //! Tower parity on the synthetic checkpoint (4 + 9 layers at width 128)
    //! against an independent F64 reference written from upstream's
    //! `TowerLayer` and `FishOmniAudioDecoder.prefill` (`tower.py:256-418`),
    //! reading the weights straight from the checkpoint.
    //!
    //! The reference takes its conventions as switches. With upstream's it
    //! must match the tower within the tolerances below; with any one switch
    //! flipped (a RED variant) it must miss by far more. Like the trunk's, it
    //! can hold the Q8_0 weights a CPU load computes with.
    //!
    //! Tolerances, relative L2 over every output value (measured on aarch64
    //! in brackets):
    //! * against the weights the load holds, F32 or Q8_0, on spans and split
    //!   spans (and on F32 steps): 1e-5 per layer [≤ 4.6e-7], 2e-5 for every
    //!   layer and head of the tower [≤ 3.3e-6]. Only F32 rounding differs:
    //!   summation order, RoPE's cos and sin rounded to F32.
    //! * Q8_0 steps, whose row-wise kernel quantizes the activations to Q8_0
    //!   as well: 3e-2 per layer [≤ 8.5e-3], 0.2 for the tower's heads
    //!   [0.080];
    //! * Q8_0 spans against the BF16 weights (the cost of Q8_0 weights): 3e-2
    //!   per layer [≤ 6.1e-3], 0.15 for the tower [≤ 0.073], on an untrained
    //!   13-layer stack where a 1% perturbation grows layer by layer.
    //! * RED variants against the F32 reference: at least 0.03 per layer
    //!   [≥ 0.053] and 0.05 for the tower [≥ 0.10], 2,500 times the F32
    //!   tolerances. The weakest are the q/k norms on one layer (the
    //!   synthetic gains sit near 1) and `extra` scaled, which only moves
    //!   rows whose input and label are both semantic.

    use candle_core::quantized::{GgmlDType, QTensor};

    use super::super::checkpoint::CODEBOOK_EMBEDDINGS;
    use super::super::config::FishDrama3RopeLayout;
    use super::super::head::FishDrama3CompactHead;
    use super::super::residency::FishDrama3WeightForm;
    use super::super::sampling::{race, FishDrama3SamplingParams};
    use super::super::synthetic::SyntheticCheckpoint;
    use super::super::trunk::{FishDrama3StepRow, FishDrama3Trunk, FishDrama3TrunkOptions};
    use super::*;
    use crate::kernels::numeric_check::{relative_l2, xorshift_values};
    use crate::models::shared::attention::physical::test_physical_caches_for_model_layers_on;

    const F32_LAYER_TOLERANCE: f64 = 1e-5;
    const F32_TOWER_TOLERANCE: f64 = 2e-5;
    const Q8_0_LAYER_TOLERANCE: f64 = 3e-2;
    const Q8_0_STEP_TOWER_TOLERANCE: f64 = 0.2;
    const Q8_0_TOWER_TOLERANCE: f64 = 0.15;
    const RED_LAYER_MARGIN: f64 = 0.03;
    const RED_TOWER_MARGIN: f64 = 0.05;

    /// The conventions the reference follows.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Conventions {
        /// `wqkv` rows `[Q | Gate | K | V]`; `false` reads the query and
        /// gate per head, `[q_h | gate_h]`, as the trunk's `q_proj`.
        contiguous_qkv: bool,
        rope: FishDrama3RopeLayout,
        qk_norm: bool,
        /// `extra` unscaled; `false` scales it by `1 / sqrt(2)`.
        extra_unscaled: bool,
        /// Injections (and their mask) from the label frame; `false` from
        /// the input frame.
        label_injections: bool,
        /// Residual layer `j` injects code `j`; `false` code `j + 1`.
        previous_code: bool,
    }

    const UPSTREAM: Conventions = Conventions {
        contiguous_qkv: true,
        rope: FishDrama3RopeLayout::InterleavedPairs,
        qk_norm: true,
        extra_unscaled: true,
        label_injections: true,
        previous_code: true,
    };

    fn sigmoid64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn f32s(x: &[f64]) -> Vec<f32> {
        x.iter().map(|v| *v as f32).collect()
    }

    fn f64s(x: &[f32]) -> Vec<f64> {
        x.iter().map(|v| f64::from(*v)).collect()
    }

    /// `x @ w.T`, `w` `[rows, cols]`.
    fn linear64(x: &[f64], w: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        x.chunks_exact(cols)
            .flat_map(|x| {
                w.chunks_exact(cols)
                    .take(rows)
                    .map(|w| x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn rms64(x: &[f64], gain: &[f64], eps: f64) -> Vec<f64> {
        let width = gain.len();
        x.chunks_exact(width)
            .flat_map(|row| {
                let r = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + eps).sqrt();
                row.iter()
                    .zip(gain)
                    .map(move |(v, g)| v * r * g)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    struct Reference<'a> {
        ck: &'a FishDrama3Checkpoint,
        config: &'a FishDrama3Config,
        conv: Conventions,
        /// The projections a Q8_0 load quantizes, round-tripped through
        /// Candle's Q8_0 quantizer.
        q8: bool,
    }

    impl<'a> Reference<'a> {
        fn new(ck: &'a FishDrama3Checkpoint, config: &'a FishDrama3Config) -> Self {
            Self {
                ck,
                config,
                conv: UPSTREAM,
                q8: false,
            }
        }

        fn with(self, conv: Conventions) -> Self {
            Self { conv, ..self }
        }

        fn holding(self, precision: FishDrama3CpuPrecision) -> Self {
            Self {
                q8: precision == FishDrama3CpuPrecision::Q8_0,
                ..self
            }
        }

        fn tensor(&self, name: &str, shape: &[usize]) -> Vec<f64> {
            f64s(&self.ck.bf16_to_f32(name, shape).unwrap())
        }

        fn projection(&self, name: &str, rows: usize, cols: usize) -> Vec<f64> {
            let values = self.ck.bf16_to_f32(name, &[rows, cols]).unwrap();
            if !self.q8 {
                return f64s(&values);
            }
            let tensor = Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap();
            let q = QTensor::quantize(&tensor, GgmlDType::Q8_0).unwrap();
            f64s(&host(&q.dequantize(&Device::Cpu).unwrap()))
        }

        /// `1 + w` of the norm `name` (without `.weight`).
        fn gain(&self, name: &str, width: usize) -> Vec<f64> {
            self.tensor(&format!("{name}.weight"), &[width])
                .into_iter()
                .map(|w| 1.0 + w)
                .collect()
        }

        /// The F64 rotation of every head of `x` (`[heads * head_dim]`) at
        /// `position`.
        fn rotate(&self, x: &mut [f64], position: usize) {
            let a = &self.config.tower.attention;
            let half = a.rotary_dims / 2;
            for head in x.chunks_exact_mut(a.head_dim) {
                let original = head.to_vec();
                for i in 0..half {
                    let angle =
                        position as f64 / a.rope_theta.powf((2 * i) as f64 / a.rotary_dims as f64);
                    let (p, q) = match self.conv.rope {
                        FishDrama3RopeLayout::HalfSplit => (i, i + half),
                        FishDrama3RopeLayout::InterleavedPairs => (2 * i, 2 * i + 1),
                    };
                    head[p] = original[p] * angle.cos() - original[q] * angle.sin();
                    head[q] = original[q] * angle.cos() + original[p] * angle.sin();
                }
            }
        }

        /// The tower layer `prefix` over rows `0..n` (positions `0..n`):
        /// injections `e`, hidden `h`.
        fn layer(&self, prefix: &str, e: &[f64], h: &[f64], n: usize) -> Vec<f64> {
            let t = &self.config.tower;
            let a = &t.attention;
            let (d, eps) = (t.dim, t.norm_eps);
            let e = rms64(e, &self.gain(&format!("{prefix}.enorm"), d), eps);
            let h = rms64(h, &self.gain(&format!("{prefix}.hnorm"), d), eps);
            let merged = e
                .chunks_exact(d)
                .zip(h.chunks_exact(d))
                .flat_map(|(e, h)| e.iter().chain(h).copied().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let w = self.projection(&format!("{prefix}.eh_proj.weight"), d, 2 * d);
            let m = linear64(&merged, &w, d, 2 * d);
            let b = format!("{prefix}.transformer_layer");
            let x = rms64(&m, &self.gain(&format!("{b}.attention_norm"), d), eps);
            let rows = t.wqkv_rows();
            let w = self.projection(&format!("{b}.attention.wqkv.weight"), rows, d);
            let packed = linear64(&x, &w, rows, d);
            let (heads, kv_heads, hd) = (a.heads, a.kv_heads, a.head_dim);
            let (qw, kvw) = (a.q_width(), a.kv_width());
            let (mut q, mut gate, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for row in packed.chunks_exact(rows) {
                if self.conv.contiguous_qkv {
                    q.extend_from_slice(&row[..qw]);
                    gate.extend_from_slice(&row[qw..2 * qw]);
                } else {
                    for head in row[..2 * qw].chunks_exact(2 * hd) {
                        q.extend_from_slice(&head[..hd]);
                        gate.extend_from_slice(&head[hd..]);
                    }
                }
                k.extend_from_slice(&row[2 * qw..2 * qw + kvw]);
                v.extend_from_slice(&row[2 * qw + kvw..]);
            }
            if self.conv.qk_norm {
                q = rms64(&q, &self.gain(&format!("{b}.attention.q_norm"), hd), eps);
                k = rms64(&k, &self.gain(&format!("{b}.attention.k_norm"), hd), eps);
            }
            for row in 0..n {
                self.rotate(&mut q[row * qw..(row + 1) * qw], row);
                self.rotate(&mut k[row * kvw..(row + 1) * kvw], row);
            }
            let group = heads / kv_heads;
            let mut o = vec![0f64; n * qw];
            for i in 0..n {
                for head in 0..heads {
                    let g = head / group;
                    let query = &q[i * qw + head * hd..i * qw + (head + 1) * hd];
                    let scores = (0..=i)
                        .map(|j| {
                            let key = &k[j * kvw + g * hd..j * kvw + (g + 1) * hd];
                            query.iter().zip(key).map(|(a, b)| a * b).sum::<f64>()
                                / (hd as f64).sqrt()
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                    let total = weights.iter().sum::<f64>();
                    for (j, weight) in weights.iter().enumerate() {
                        let value = &v[j * kvw + g * hd..j * kvw + (g + 1) * hd];
                        for (out, value) in o[i * qw + head * hd..i * qw + (head + 1) * hd]
                            .iter_mut()
                            .zip(value)
                        {
                            *out += weight / total * value;
                        }
                    }
                }
            }
            for (o, g) in o.iter_mut().zip(&gate) {
                *o *= sigmoid64(*g);
            }
            let w = self.projection(&format!("{b}.attention.wo.weight"), d, qw);
            let mut r = m
                .iter()
                .zip(linear64(&o, &w, d, qw))
                .map(|(m, a)| m + a)
                .collect::<Vec<_>>();
            let f = rms64(&r, &self.gain(&format!("{b}.ffn_norm"), d), eps);
            let i = t.intermediate;
            let x = format!("{b}.feed_forward");
            let gate = linear64(&f, &self.projection(&format!("{x}.w1.weight"), i, d), i, d);
            let up = linear64(&f, &self.projection(&format!("{x}.w3.weight"), i, d), i, d);
            let act = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| g * sigmoid64(*g) * u)
                .collect::<Vec<_>>();
            let down = linear64(
                &act,
                &self.projection(&format!("{x}.w2.weight"), d, i),
                d,
                i,
            );
            r.iter_mut().zip(&down).for_each(|(r, f)| *r += f);
            rms64(&r, &self.gain(&format!("{prefix}.final_layernorm"), d), eps)
        }

        /// The tower teacher-forced over rows `0..n` of `frames` (`n + 1`
        /// frames): every layer's output, in order.
        fn tower(&self, hidden: &[f64], frames: &[u32], n: usize) -> Vec<Vec<f64>> {
            let t = &self.config.tower;
            let ids = self.config.ids;
            let (d, width, v) = (t.dim, t.frame_width(), t.codebook_vocab);
            let cb = self.tensor(CODEBOOK_EMBEDDINGS, &[t.codebooks * v, d]);
            let inputs = frames[..n * width].chunks_exact(width).collect::<Vec<_>>();
            let labels = frames[width..(n + 1) * width]
                .chunks_exact(width)
                .collect::<Vec<_>>();
            let source = if self.conv.label_injections {
                &labels
            } else {
                &inputs
            };
            let scale = if self.conv.extra_unscaled {
                1.0
            } else {
                1.0 / 2f64.sqrt()
            };
            let mut extra = vec![0f64; n * d];
            for (input, out) in inputs.iter().zip(extra.chunks_exact_mut(d)) {
                if ids.is_semantic(input[0]) {
                    for k in 2..t.codebooks {
                        let row = k * v + (input[1 + k] as usize).min(v - 1);
                        for (o, c) in out.iter_mut().zip(&cb[row * d..(row + 1) * d]) {
                            *o += c * scale;
                        }
                    }
                }
            }
            let inject = |table: &[f64], column: usize| {
                source
                    .iter()
                    .flat_map(|frame| {
                        if ids.is_semantic(frame[0]) {
                            let row = (frame[column] as usize).min(v - 1);
                            table[row * d..(row + 1) * d].to_vec()
                        } else {
                            vec![0f64; d]
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let mut x = hidden.to_vec();
            let mut out = Vec::new();
            for l in 0..t.shared_layers {
                let prefix = tower_shared_layer(l);
                let table = self.tensor(&format!("{prefix}.semantic_embedding.weight"), &[v, d]);
                let mut e = inject(&table, 1);
                if l == 0 {
                    e.iter_mut().zip(&extra).for_each(|(e, x)| *e += x);
                }
                x = self.layer(&prefix, &e, &x, n);
                out.push(x.clone());
            }
            for j in 0..t.residual_layers() {
                let prefix = tower_residual_layer(j);
                let table = self.tensor(&format!("{prefix}.codebook_embedding.weight"), &[v, d]);
                let column = if self.conv.previous_code {
                    1 + j
                } else {
                    2 + j
                };
                x = self.layer(&prefix, &inject(&table, column), &x, n);
                out.push(x.clone());
            }
            out
        }

        /// Every head's logits over the residual layers' outputs of
        /// [`Self::tower`], `[n, heads, V]`.
        fn heads(&self, layers: &[Vec<f64>], n: usize) -> Vec<f64> {
            let t = &self.config.tower;
            let (d, v, r) = (t.dim, t.codebook_vocab, t.residual_layers());
            let per_head = (0..r)
                .map(|j| {
                    let w = self.projection(&tower_head(j), v, d);
                    linear64(&layers[t.shared_layers + j], &w, v, d)
                })
                .collect::<Vec<_>>();
            interleave_heads(&per_head, n, v)
        }
    }

    /// `[heads][n, V]` to `[n, heads, V]`.
    fn interleave_heads<T: Copy>(per_head: &[Vec<T>], n: usize, v: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(per_head.len() * n * v);
        for row in 0..n {
            for head in per_head {
                out.extend_from_slice(&head[row * v..(row + 1) * v]);
            }
        }
        out
    }

    fn host(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn load(synthetic: &SyntheticCheckpoint, precision: FishDrama3CpuPrecision) -> FishDrama3Tower {
        FishDrama3Tower::load(
            &synthetic.open(),
            &synthetic.config,
            &Device::Cpu,
            FishDrama3TowerOptions {
                precision,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn caches_for(
        layers: &[u32],
        config: &FishDrama3Config,
        rows: usize,
    ) -> Vec<PhysicalPagedKvCache> {
        let a = &config.tower.attention;
        test_physical_caches_for_model_layers_on(
            &Device::Cpu,
            DType::F32,
            layers,
            a.kv_heads,
            a.head_dim,
            64,
            rows,
        )
    }

    fn caches(
        tower: &FishDrama3Tower,
        config: &FishDrama3Config,
        rows: usize,
    ) -> Vec<PhysicalPagedKvCache> {
        caches_for(tower.attention_layers(), config, rows)
    }

    /// 21 frames: text, a run of semantic frames (codes over the whole
    /// codebook, one past it), then text. Row 5 has a semantic label and a
    /// text input, rows 6..14 both semantic, row 14 a semantic input and a
    /// text label.
    fn frames(config: &FishDrama3Config) -> Vec<u32> {
        let ids = config.ids;
        let v = config.tower.codebook_vocab as u32;
        let draws = xorshift_values(21 * 11, 78, 1.0);
        let pick = |i: usize, limit: u32| ((draws[i].abs() * limit as f32) as u32).min(limit - 1);
        let mut out = Vec::new();
        for t in 0..21 {
            let mut row = vec![0u32; 11];
            if (6..15).contains(&t) {
                let c0 = pick(t * 11, v);
                row[0] = ids.semantic_start + c0;
                row[1] = c0;
                for (k, code) in row.iter_mut().enumerate().skip(2) {
                    *code = pick(t * 11 + k, v);
                }
                if t == 9 {
                    row[4] = 900;
                }
            } else {
                row[0] = pick(t * 11, ids.eos);
            }
            out.extend(row);
        }
        out
    }

    /// Measured tolerances against `reference`: `(name, got, tolerance)`.
    fn check(results: &[(&str, &[f32], f64)], reference: &[f32], context: &str) {
        for (name, got, tolerance) in results {
            let error = relative_l2(got, reference);
            assert!(
                error < *tolerance,
                "{context} {name}: {error:.3e} >= {tolerance:.1e}"
            );
        }
    }

    fn step_tolerance(
        precision: FishDrama3CpuPrecision,
        f32_tolerance: f64,
        q8_tolerance: f64,
    ) -> f64 {
        match precision {
            FishDrama3CpuPrecision::F32 => f32_tolerance,
            FishDrama3CpuPrecision::Q8_0 => q8_tolerance,
        }
    }

    /// The first shared layer and the last residual layer alone, on spans
    /// and steps, against F64 on the weights the load holds; per-layer RED
    /// variants (the `wqkv` packing, the RoPE pairs, the q/k norms) far off.
    #[test]
    fn a_tower_layer_matches_f64() {
        let synthetic = SyntheticCheckpoint::write(21);
        let config = &synthetic.config;
        let ck = synthetic.open();
        let (n, d) = (20, config.tower.dim);
        let mut e = xorshift_values(n * d, 7, 1.0);
        // The first rows inject nothing, as text labels do (enorm(0) = 0).
        e[..3 * d].fill(0.0);
        let h = xorshift_values(n * d, 8, 1.0);
        let towers = [FishDrama3CpuPrecision::F32, FishDrama3CpuPrecision::Q8_0]
            .map(|precision| (precision, load(&synthetic, precision)));
        let last = config.tower.shared_layers + config.tower.residual_layers() - 1;
        for (index, prefix) in [
            (0, tower_shared_layer(0)),
            (
                last,
                tower_residual_layer(config.tower.residual_layers() - 1),
            ),
        ] {
            let upstream =
                f32s(&Reference::new(&ck, config).layer(&prefix, &f64s(&e), &f64s(&h), n));
            for (precision, tower) in &towers {
                let held = f32s(&Reference::new(&ck, config).holding(*precision).layer(
                    &prefix,
                    &f64s(&e),
                    &f64s(&h),
                    n,
                ));
                let one = || caches_for(&[tower.attention_layers()[index]], config, 1).remove(0);
                let layer = &tower.host().unwrap().0[index];
                let mut cache = one();
                let mut prepared = cache.prepare_append(0, n).unwrap();
                let span = tower
                    .layer(
                        layer,
                        0,
                        &e,
                        &h,
                        n,
                        0,
                        &cache,
                        &mut prepared,
                        FishDrama3KernelPhase::Span,
                    )
                    .unwrap();
                cache.commit_prepared(prepared).unwrap();
                let mut cache = one();
                let mut steps = Vec::new();
                for t in 0..n {
                    let mut prepared = cache.prepare_append(t, 1).unwrap();
                    steps.extend(
                        tower
                            .layer(
                                layer,
                                0,
                                &e[t * d..(t + 1) * d],
                                &h[t * d..(t + 1) * d],
                                1,
                                t,
                                &cache,
                                &mut prepared,
                                FishDrama3KernelPhase::Step,
                            )
                            .unwrap(),
                    );
                    cache.commit_prepared(prepared).unwrap();
                }
                let context = format!("layer {index} {precision:?}");
                check(
                    &[
                        ("span", &span, F32_LAYER_TOLERANCE),
                        (
                            "step",
                            &steps,
                            step_tolerance(*precision, F32_LAYER_TOLERANCE, Q8_0_LAYER_TOLERANCE),
                        ),
                    ],
                    &held,
                    &context,
                );
                check(
                    &[("span vs BF16 weights", &span, Q8_0_LAYER_TOLERANCE)],
                    &upstream,
                    &context,
                );
            }
            for (name, conv) in [
                (
                    "wqkv read per head",
                    Conventions {
                        contiguous_qkv: false,
                        ..UPSTREAM
                    },
                ),
                (
                    "NeoX RoPE",
                    Conventions {
                        rope: FishDrama3RopeLayout::HalfSplit,
                        ..UPSTREAM
                    },
                ),
                (
                    "no q/k norm",
                    Conventions {
                        qk_norm: false,
                        ..UPSTREAM
                    },
                ),
            ] {
                let red = f32s(&Reference::new(&ck, config).with(conv).layer(
                    &prefix,
                    &f64s(&e),
                    &f64s(&h),
                    n,
                ));
                let error = relative_l2(&red, &upstream);
                assert!(error > RED_LAYER_MARGIN, "layer {index} {name}: {error}");
            }
        }
    }

    /// The span's outputs per layer, concatenated layer by layer.
    fn concat(layers: &[Vec<f32>]) -> Vec<f32> {
        layers.concat()
    }

    /// Head logits `[n, heads, V]` over a span trace's residual outputs.
    fn trace_heads(tower: &FishDrama3Tower, trace: &[Vec<f32>], n: usize) -> Vec<f32> {
        let t = tower.config();
        let per_head = (0..t.residual_layers())
            .map(|j| {
                tower
                    .head_logits(
                        j,
                        &trace[t.shared_layers + j],
                        n,
                        FishDrama3KernelPhase::Span,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        interleave_heads(&per_head, n, t.codebook_vocab)
    }

    /// The whole tower over 20 teacher-forced rows on every path (one span,
    /// split spans, row-by-row forced steps), every layer and every head
    /// against F64 on F32 and Q8_0 weights; every RED variant far off.
    #[test]
    fn the_tower_matches_f64_on_every_path() {
        let synthetic = SyntheticCheckpoint::write(22);
        let config = &synthetic.config;
        let ck = synthetic.open();
        let embedding = FishDrama3Embedding::load(&ck, config).unwrap();
        let frames = frames(config);
        let width = config.tower.frame_width();
        let n = frames.len() / width - 1;
        let d = config.tower.dim;
        let hidden = xorshift_values(n * d, 31, 1.0);
        let hidden_tensor = Tensor::from_vec(hidden.clone(), (n, d), &Device::Cpu).unwrap();
        let upstream_ref = Reference::new(&ck, config);
        let upstream = upstream_ref.tower(&f64s(&hidden), &frames, n);
        let upstream_layers = f32s(&upstream.concat());
        let upstream_heads = f32s(&upstream_ref.heads(&upstream, n));
        let mut f32_layers = Vec::new();
        for precision in [FishDrama3CpuPrecision::F32, FishDrama3CpuPrecision::Q8_0] {
            let held_ref = Reference::new(&ck, config).holding(precision);
            let held = held_ref.tower(&f64s(&hidden), &frames, n);
            let held_layers = f32s(&held.concat());
            let held_heads = f32s(&held_ref.heads(&held, n));
            let tower = load(&synthetic, precision);
            assert_eq!(tower.attention_layers(), (8..21).collect::<Vec<u32>>());
            // One span.
            let mut cache = caches(&tower, config, 1).remove(0);
            let trace = tower
                .forward_span_traced(&embedding, &hidden_tensor, &frames, 0, &mut cache)
                .unwrap();
            assert_eq!(cache.context_len(), n);
            assert_eq!(trace.len(), 13);
            let span = concat(&trace);
            let span_heads = trace_heads(&tower, &trace, n);
            // Split spans: each takes its rows and the next one.
            let mut cache = caches(&tower, config, 1).remove(0);
            let mut split = vec![Vec::new(); 13];
            for (a, b) in [(0, 7), (7, 8), (8, n)] {
                let trace = tower
                    .forward_span_traced(
                        &embedding,
                        &hidden_tensor.narrow(0, a, b - a).unwrap(),
                        &frames[a * width..(b + 1) * width],
                        a,
                        &mut cache,
                    )
                    .unwrap();
                for (all, part) in split.iter_mut().zip(trace) {
                    all.extend(part);
                }
            }
            assert_eq!(cache.context_len(), n);
            let split = concat(&split);
            // Row by row, forced to the labels.
            let mut cache = caches(&tower, config, 1).remove(0);
            let mut steps = Vec::new();
            for t in 0..n {
                let mut rows = [FishDrama3TowerForcedRow {
                    cache: &mut cache,
                    input: &frames[t * width..(t + 1) * width],
                    label: &frames[(t + 1) * width..(t + 2) * width],
                }];
                steps.extend(
                    tower
                        .forward_step_forced(
                            &embedding,
                            &hidden_tensor.narrow(0, t, 1).unwrap(),
                            &mut rows,
                        )
                        .unwrap(),
                );
            }
            assert_eq!(cache.context_len(), n);
            let context = format!("tower {precision:?}");
            let step = step_tolerance(precision, F32_TOWER_TOLERANCE, Q8_0_STEP_TOWER_TOLERANCE);
            check(
                &[
                    ("span", &span, F32_TOWER_TOLERANCE),
                    ("split", &split, F32_TOWER_TOLERANCE),
                ],
                &held_layers,
                &context,
            );
            check(
                &[
                    ("span heads", &span_heads, F32_TOWER_TOLERANCE),
                    ("step heads", &steps, step),
                ],
                &held_heads,
                &context,
            );
            check(
                &[("split vs span", &split, F32_TOWER_TOLERANCE)],
                &span,
                &context,
            );
            check(
                &[("step vs span heads", &steps, step)],
                &span_heads,
                &context,
            );
            check(
                &[("span vs BF16 weights", &span, Q8_0_TOWER_TOLERANCE)],
                &upstream_layers,
                &context,
            );
            check(
                &[(
                    "span heads vs BF16 weights",
                    &span_heads,
                    Q8_0_TOWER_TOLERANCE,
                )],
                &upstream_heads,
                &context,
            );
            if precision == FishDrama3CpuPrecision::F32 {
                f32_layers = span;
            }
        }

        // RED: each convention flipped misses the F32 tower by far more
        // than its tolerance.
        for (name, conv) in [
            (
                "wqkv read per head",
                Conventions {
                    contiguous_qkv: false,
                    ..UPSTREAM
                },
            ),
            (
                "NeoX RoPE",
                Conventions {
                    rope: FishDrama3RopeLayout::HalfSplit,
                    ..UPSTREAM
                },
            ),
            (
                "no q/k norm",
                Conventions {
                    qk_norm: false,
                    ..UPSTREAM
                },
            ),
            (
                "extra scaled by 1/sqrt(2)",
                Conventions {
                    extra_unscaled: false,
                    ..UPSTREAM
                },
            ),
            (
                "injections from the input frame",
                Conventions {
                    label_injections: false,
                    ..UPSTREAM
                },
            ),
            (
                "residual layer k injects code k",
                Conventions {
                    previous_code: false,
                    ..UPSTREAM
                },
            ),
        ] {
            let red = Reference::new(&ck, config).with(conv);
            let red = f32s(&red.tower(&f64s(&hidden), &frames, n).concat());
            let error = relative_l2(&f32_layers, &red);
            assert!(error > RED_TOWER_MARGIN, "{name}: {error}");
        }
    }

    /// Prefills the first `rows` + 1 frames' rows teacher-forced: `rows`
    /// tower rows.
    fn prefill(
        tower: &FishDrama3Tower,
        embedding: &FishDrama3Embedding,
        config: &FishDrama3Config,
        frames: &[u32],
        hidden: &Tensor,
        rows: usize,
        cache: &mut PhysicalPagedKvCache,
    ) {
        let width = config.tower.frame_width();
        tower
            .forward_span(
                embedding,
                &hidden.narrow(0, 0, rows).unwrap(),
                &frames[..(rows + 1) * width],
                0,
                cache,
            )
            .unwrap();
    }

    /// A sampled step draws each head's code with the row's sampler after
    /// that head and injects it into the next layer: under recorded noise
    /// the codes are exactly what the race picks over the logits a forced
    /// step gets with those codes. A head code is fed back as is up to the
    /// table's last row (past it, clamped), and a step refuses a main that
    /// is not semantic without moving anything.
    #[test]
    fn a_sampled_step_samples_each_head_and_feeds_its_code_back() {
        let synthetic = SyntheticCheckpoint::write(23);
        let config = &synthetic.config;
        let ck = synthetic.open();
        let embedding = FishDrama3Embedding::load(&ck, config).unwrap();
        let ids = config.ids;
        let frames = frames(config);
        let width = config.tower.frame_width();
        let (d, v, r) = (
            config.tower.dim,
            config.tower.codebook_vocab,
            config.tower.residual_layers(),
        );
        let hidden =
            Tensor::from_vec(xorshift_values(21 * d, 41, 1.0), (21, d), &Device::Cpu).unwrap();
        let prompt_rows = 9;
        let input = &frames[8 * width..9 * width];
        let h_t = hidden.narrow(0, 8, 1).unwrap();
        let params = FishDrama3SamplingParams::from_request(None, None, None, 0).unwrap();
        let noise = xorshift_values(r * v, 5, 1.0)
            .into_iter()
            .map(|u| -((u.abs() * 0.999 + 0.0005).ln()))
            .collect::<Vec<_>>();
        let main = ids.semantic(5).unwrap();
        for precision in [FishDrama3CpuPrecision::F32, FishDrama3CpuPrecision::Q8_0] {
            let tower = load(&synthetic, precision);
            let mut cache = caches(&tower, config, 1).remove(0);
            prefill(
                &tower,
                &embedding,
                config,
                &frames,
                &hidden,
                prompt_rows - 1,
                &mut cache,
            );
            let mut sampler = FishDrama3RowSampler::with_recorded_noise(
                params,
                ids,
                &frames
                    .chunks_exact(width)
                    .take(prompt_rows)
                    .map(|f| f[0])
                    .collect::<Vec<_>>(),
                noise.clone(),
            )
            .unwrap();
            let window = sampler.window();
            // Refused before anything moves: eos, a text token, a short input.
            for (bad_main, bad_input) in [(ids.eos, input), (ids.voice, input), (main, &input[..3])]
            {
                let mut rows = [FishDrama3TowerStepRow {
                    cache: &mut cache,
                    input: bad_input,
                    main: bad_main,
                    sampler: &mut sampler,
                }];
                assert!(tower.forward_step(&embedding, &h_t, &mut rows).is_err());
            }
            assert_eq!(cache.context_len(), prompt_rows - 1);
            let mut rows = [FishDrama3TowerStepRow {
                cache: &mut cache,
                input,
                main,
                sampler: &mut sampler,
            }];
            let frame = tower.forward_step(&embedding, &h_t, &mut rows).unwrap();
            assert_eq!(cache.context_len(), prompt_rows);
            assert_eq!(&frame[..2], &[main, 5]);
            assert_eq!(frame.len(), width);
            assert!(frame[2..].iter().all(|c| (*c as usize) < v));
            // The heads consumed exactly their noise, and the main token's
            // window is the sampler's business, not the tower's.
            assert!(sampler.exponentials(1).is_err());
            assert_eq!(sampler.window(), window);

            // The same step forced to the sampled frame: the race over its
            // logits under the same noise picks every code again.
            let mut cache = caches(&tower, config, 1).remove(0);
            prefill(
                &tower,
                &embedding,
                config,
                &frames,
                &hidden,
                prompt_rows - 1,
                &mut cache,
            );
            let mut rows = [FishDrama3TowerForcedRow {
                cache: &mut cache,
                input,
                label: &frame,
            }];
            let logits = tower
                .forward_step_forced(&embedding, &h_t, &mut rows)
                .unwrap();
            assert_eq!(logits.len(), r * v);
            for j in 0..r {
                let code = race(
                    &logits[j * v..(j + 1) * v],
                    &params.heads,
                    0,
                    &noise[j * v..(j + 1) * v],
                )
                .unwrap();
                assert_eq!(code as u32, frame[2 + j], "{precision:?} head {j}");
            }

            // Codes index the injection tables clamped at their last row.
            let forced = |label: &[u32]| {
                let mut cache = caches(&tower, config, 1).remove(0);
                prefill(
                    &tower,
                    &embedding,
                    config,
                    &frames,
                    &hidden,
                    prompt_rows - 1,
                    &mut cache,
                );
                let mut rows = [FishDrama3TowerForcedRow {
                    cache: &mut cache,
                    input,
                    label,
                }];
                tower
                    .forward_step_forced(&embedding, &h_t, &mut rows)
                    .unwrap()
            };
            let mut top = frame.clone();
            top[2..].fill(v as u32 - 1);
            let mut past = frame.clone();
            past[2..].fill(10_000);
            let mut low = frame.clone();
            low[2..].fill(v as u32 / 4 - 1);
            assert_eq!(forced(&top), forced(&past));
            assert_ne!(forced(&top), forced(&low));
        }
    }

    /// A step over several requests is a loop over them: each row's frame
    /// and sampler are what it gets alone, whatever the order; a failing row
    /// leaves every row where it was.
    #[test]
    fn a_step_batch_is_a_per_row_loop() {
        let synthetic = SyntheticCheckpoint::write(24);
        let config = &synthetic.config;
        let ck = synthetic.open();
        let embedding = FishDrama3Embedding::load(&ck, config).unwrap();
        let tower = load(&synthetic, FishDrama3CpuPrecision::Q8_0);
        let ids = config.ids;
        let frames = frames(config);
        let width = config.tower.frame_width();
        let d = config.tower.dim;
        let hidden =
            Tensor::from_vec(xorshift_values(21 * d, 42, 1.0), (21, d), &Device::Cpu).unwrap();
        let lengths = [3usize, 9, 6];
        let mains = [
            ids.semantic(1).unwrap(),
            ids.semantic(60).unwrap(),
            ids.semantic(33).unwrap(),
        ];
        let prefilled = || {
            let mut caches = caches(&tower, config, 3);
            for (cache, length) in caches.iter_mut().zip(lengths) {
                prefill(&tower, &embedding, config, &frames, &hidden, length, cache);
            }
            let samplers = (0..3)
                .map(|r| {
                    FishDrama3RowSampler::new(
                        FishDrama3SamplingParams::from_request(None, None, None, 100 + r).unwrap(),
                        ids,
                        &[ids.voice],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            (caches, samplers)
        };
        let input = |r: usize| &frames[lengths[r] * width..(lengths[r] + 1) * width];
        let h = |r: usize| hidden.narrow(0, lengths[r], 1).unwrap();
        // Alone.
        let mut alone = Vec::new();
        for r in 0..3 {
            let (mut caches, mut samplers) = prefilled();
            let mut rows = [FishDrama3TowerStepRow {
                cache: &mut caches[r],
                input: input(r),
                main: mains[r],
                sampler: &mut samplers[r],
            }];
            let frame = tower.forward_step(&embedding, &h(r), &mut rows).unwrap();
            assert_eq!(caches[r].context_len(), lengths[r] + 1);
            alone.push((frame, samplers[r].checkpoint()));
        }
        // Together, in two orders.
        for order in [[0usize, 1, 2], [2, 0, 1]] {
            let (mut caches, mut samplers) = prefilled();
            let before = tower.step_rows();
            let batch = Tensor::cat(&order.iter().map(|r| h(*r)).collect::<Vec<_>>(), 0).unwrap();
            let mut refs = caches
                .iter_mut()
                .zip(samplers.iter_mut())
                .map(Some)
                .collect::<Vec<_>>();
            let mut rows = order
                .iter()
                .map(|r| {
                    let (cache, sampler) = refs[*r].take().unwrap();
                    FishDrama3TowerStepRow {
                        cache,
                        input: input(*r),
                        main: mains[*r],
                        sampler,
                    }
                })
                .collect::<Vec<_>>();
            let out = tower.forward_step(&embedding, &batch, &mut rows).unwrap();
            assert_eq!(tower.step_rows() - before, 3);
            drop(rows);
            for (slot, r) in order.iter().enumerate() {
                assert_eq!(
                    out[slot * width..(slot + 1) * width],
                    alone[*r].0[..],
                    "request {r} at {slot}"
                );
                assert_eq!(samplers[*r].checkpoint(), alone[*r].1);
                assert_eq!(caches[*r].context_len(), lengths[*r] + 1);
            }
        }
        // A failing second row (eos as its main) leaves both rows as they were.
        let (mut caches, mut samplers) = prefilled();
        let before = samplers.iter().map(|s| s.checkpoint()).collect::<Vec<_>>();
        let (first_cache, rest_caches) = caches.split_at_mut(1);
        let (first_sampler, rest_samplers) = samplers.split_at_mut(1);
        let mut rows = [
            FishDrama3TowerStepRow {
                cache: &mut first_cache[0],
                input: input(0),
                main: mains[0],
                sampler: &mut first_sampler[0],
            },
            FishDrama3TowerStepRow {
                cache: &mut rest_caches[0],
                input: input(1),
                main: ids.eos,
                sampler: &mut rest_samplers[0],
            },
        ];
        let batch = Tensor::cat(&[h(0), h(1)], 0).unwrap();
        assert!(tower.forward_step(&embedding, &batch, &mut rows).is_err());
        assert_eq!((caches[0].context_len(), caches[1].context_len()), (3, 9));
        for (sampler, before) in samplers.iter().zip(before) {
            assert_eq!(sampler.checkpoint(), before);
        }
    }

    /// Prefill spans and decode steps through the trunk and the tower keep
    /// §7.2's invariant: level while the prompt is prefilled, the tower one
    /// row behind once it is in.
    #[test]
    fn the_kv_lengths_follow_the_trunk_through_prefill_and_decode() {
        let synthetic = SyntheticCheckpoint::write(25);
        let config = &synthetic.config;
        let ck = synthetic.open();
        let embedding = FishDrama3Embedding::load(&ck, config).unwrap();
        let trunk =
            FishDrama3Trunk::load(&ck, config, &Device::Cpu, FishDrama3TrunkOptions::default())
                .unwrap();
        let tower = load(&synthetic, FishDrama3CpuPrecision::Q8_0);
        let head = FishDrama3CompactHead::load(&ck, config, &Device::Cpu).unwrap();
        let width = config.tower.frame_width();
        let prompt_rows = 12;
        let prompt = &frames(config)[..prompt_rows * width];
        let a = &config.trunk.attention;
        let mut trunk_cache = test_physical_caches_for_model_layers_on(
            &Device::Cpu,
            DType::F32,
            trunk.attention_layers(),
            a.kv_heads,
            a.head_dim,
            64,
            1,
        )
        .remove(0);
        let mut tower_cache = caches(&tower, config, 1).remove(0);
        let mut state = trunk.new_state().unwrap();
        let mains = prompt
            .chunks_exact(width)
            .map(|row| row[0])
            .collect::<Vec<_>>();
        let mut sampler = FishDrama3RowSampler::new(
            FishDrama3SamplingParams::from_request(None, None, None, 9).unwrap(),
            config.ids,
            &mains,
        )
        .unwrap();
        assert_eq!(
            FishDrama3Tower::teacher_forced_rows(11..12, prompt_rows),
            11..11
        );
        let mut last_hidden = None;
        for span in [0..5, 5..11, 11..12] {
            let embeds = embedding
                .embed_tensor(&prompt[span.start * width..span.end * width], &Device::Cpu)
                .unwrap();
            let hidden = trunk
                .forward_span(&embeds, span.start, &mut state, &mut trunk_cache)
                .unwrap();
            let rows = FishDrama3Tower::teacher_forced_rows(span.clone(), prompt_rows);
            if !rows.is_empty() {
                tower
                    .forward_span(
                        &embedding,
                        &hidden.narrow(0, 0, rows.len()).unwrap(),
                        &prompt[rows.start * width..(rows.end + 1) * width],
                        rows.start,
                        &mut tower_cache,
                    )
                    .unwrap();
            }
            check_lengths(
                trunk_cache.context_len(),
                tower_cache.context_len(),
                prompt_rows,
            )
            .unwrap();
            last_hidden = Some(hidden.narrow(0, span.len() - 1, 1).unwrap());
        }
        assert_eq!(
            (trunk_cache.context_len(), tower_cache.context_len()),
            (12, 11)
        );
        let mut h_t = last_hidden.unwrap();
        let mut input = prompt[11 * width..].to_vec();
        for _ in 0..3 {
            let logits = host(&head.logits(&h_t).unwrap());
            // Eos stays masked so the rollout keeps going.
            let draw = sampler.sample_main(&logits, false).unwrap();
            let mut rows = [FishDrama3TowerStepRow {
                cache: &mut tower_cache,
                input: &input,
                main: draw.token,
                sampler: &mut sampler,
            }];
            let frame = tower.forward_step(&embedding, &h_t, &mut rows).unwrap();
            let embeds = embedding.embed_tensor(&frame, &Device::Cpu).unwrap();
            let mut rows = [FishDrama3StepRow {
                state: &mut state,
                cache: &mut trunk_cache,
            }];
            h_t = trunk.forward_step(&embeds, &mut rows).unwrap();
            check_lengths(
                trunk_cache.context_len(),
                tower_cache.context_len(),
                prompt_rows,
            )
            .unwrap();
            input = frame;
        }
        assert_eq!(
            (trunk_cache.context_len(), tower_cache.context_len()),
            (15, 14)
        );
        // The invariant refuses the other combinations.
        for (trunk_len, tower_len) in [(15, 15), (15, 13), (5, 4), (12, 12)] {
            assert!(check_lengths(trunk_len, tower_len, prompt_rows).is_err());
        }
        assert!(check_lengths(0, 0, 0).is_err());
        assert!(check_lengths(5, 5, prompt_rows).is_ok());
    }

    /// An aborted quantum rewinds the tower's paged view, a replay gives the
    /// same bits, and refused forwards move nothing.
    #[test]
    fn an_aborted_quantum_rewinds_the_tower_kv() {
        let synthetic = SyntheticCheckpoint::write(26);
        let config = &synthetic.config;
        let ck = synthetic.open();
        let embedding = FishDrama3Embedding::load(&ck, config).unwrap();
        let tower = load(&synthetic, FishDrama3CpuPrecision::Q8_0);
        let ids = config.ids;
        let frames = frames(config);
        let width = config.tower.frame_width();
        let d = config.tower.dim;
        let hidden =
            Tensor::from_vec(xorshift_values(21 * d, 43, 1.0), (21, d), &Device::Cpu).unwrap();
        let mut cache = caches(&tower, config, 1).remove(0);
        prefill(&tower, &embedding, config, &frames, &hidden, 6, &mut cache);
        assert_eq!(cache.context_len(), 6);

        // Refused before anything moves: a wrong start, the wrong width, a
        // frame short, a cache of other layers.
        let span = |start: usize, rows: usize| {
            (
                hidden.narrow(0, start, rows).unwrap(),
                &frames[start * width..(start + rows + 1) * width],
            )
        };
        let (h, f) = span(6, 2);
        assert!(tower
            .forward_span(&embedding, &h, f, 5, &mut cache)
            .is_err());
        assert!(tower
            .forward_span(&embedding, &h.narrow(1, 0, 64).unwrap(), f, 6, &mut cache)
            .is_err());
        assert!(tower
            .forward_span(&embedding, &h, &f[..2 * width], 6, &mut cache)
            .is_err());
        let mut other = caches_for(&tower.attention_layers()[1..], config, 1).remove(0);
        assert!(tower
            .forward_span(&embedding, &h, f, 0, &mut other)
            .is_err());
        assert!(tower.begin_quantum(&other).is_err());
        assert_eq!(cache.context_len(), 6);

        let mut sampler = FishDrama3RowSampler::new(
            FishDrama3SamplingParams::from_request(None, None, None, 3).unwrap(),
            ids,
            &[ids.voice],
        )
        .unwrap();
        let quantum = |cache: &mut PhysicalPagedKvCache, sampler: &mut FishDrama3RowSampler| {
            let (h, f) = span(6, 4);
            tower.forward_span(&embedding, &h, f, 6, cache).unwrap();
            let mut rows = [FishDrama3TowerStepRow {
                cache,
                input: &frames[10 * width..11 * width],
                main: ids.semantic(17).unwrap(),
                sampler,
            }];
            tower
                .forward_step(&embedding, &hidden.narrow(0, 10, 1).unwrap(), &mut rows)
                .unwrap()
        };
        let checkpoint = tower.begin_quantum(&cache).unwrap();
        let sampled = sampler.checkpoint();
        let first = quantum(&mut cache, &mut sampler);
        assert_eq!(cache.context_len(), 11);
        tower.abort_quantum(&mut cache, checkpoint).unwrap();
        sampler.restore(sampled);
        assert_eq!(cache.context_len(), 6);
        let checkpoint = tower.begin_quantum(&cache).unwrap();
        let again = quantum(&mut cache, &mut sampler);
        assert_eq!(again, first);
        tower.commit_quantum(&cache, &checkpoint).unwrap();
        // A view behind its checkpoint does not commit.
        let ahead = tower.begin_quantum(&cache).unwrap();
        let mut rewound = caches(&tower, config, 1).remove(0);
        prefill(
            &tower,
            &embedding,
            config,
            &frames,
            &hidden,
            6,
            &mut rewound,
        );
        assert!(tower.commit_quantum(&rewound, &ahead).is_err());
        let summary = tower.diagnostics_summary();
        assert_eq!(summary["step_rows"], 2);
        assert_eq!(summary["span_rows"], 6 + 4 + 4 + 6);
    }

    /// The self-check covers the tower's classes: one failing its Q8_0
    /// check (the heads' block scales underflow F16) stays F32 and is
    /// counted; F32 loads check nothing.
    #[test]
    fn a_tower_class_failing_its_q8_0_check_stays_f32() {
        let healthy = SyntheticCheckpoint::write(27);
        let tower = load(&healthy, FishDrama3CpuPrecision::Q8_0);
        let q8_0_bytes = tower.resident_bytes();
        let report = tower.precision_report();
        assert_eq!(report.q8_fallback_classes, 0);
        assert_eq!(report.checks.len(), 1 + 2 + 3 + 1);
        assert_eq!(report.forms.len(), 4);
        assert!(report
            .forms
            .iter()
            .all(|(_, form)| *form == FishDrama3WeightForm::Q8_0));
        assert!(report
            .checks
            .iter()
            .all(|c| c.relative_l2 < Q8_0_TOLERANCE / 2.0));

        let tiny = SyntheticCheckpoint::write_with(27, |tensors| {
            for (name, _, _, bytes) in tensors.iter_mut() {
                if name.starts_with("audio_decoder_mtp.heads.") {
                    for pair in bytes.as_chunks_mut::<2>().0 {
                        let v = bf16::from_le_bytes(*pair).to_f32() * 1e-5;
                        pair.copy_from_slice(&bf16::from_f32(v).to_le_bytes());
                    }
                }
            }
        });
        let tower = load(&tiny, FishDrama3CpuPrecision::Q8_0);
        let report = tower.precision_report();
        assert_eq!(report.q8_fallback_classes, 1);
        for (class, form) in &report.forms {
            let expected = if *class == FishDrama3ProjectionClass::TowerHead {
                FishDrama3WeightForm::F32
            } else {
                FishDrama3WeightForm::Q8_0
            };
            assert_eq!(*form, expected, "{class:?}");
        }
        let (layers, heads) = tower.host().unwrap();
        assert!(heads.iter().all(|h| h.form() == FishDrama3WeightForm::F32));
        assert_eq!(layers[0].wqkv.form(), FishDrama3WeightForm::Q8_0);
        assert_eq!(
            tower.diagnostics_summary()["precision"]["q8_0_fallback_classes"],
            1
        );
        let exact = load(&healthy, FishDrama3CpuPrecision::F32);
        assert!(exact.precision_report().checks.is_empty());
        assert!(exact.resident_bytes() > 2 * q8_0_bytes);
    }

    #[test]
    fn teacher_forced_rows_stop_short_of_the_last_prompt_row() {
        let rows = FishDrama3Tower::teacher_forced_rows;
        assert_eq!(rows(0..5, 12), 0..5);
        assert_eq!(rows(5..12, 12), 5..11);
        assert_eq!(rows(0..12, 12), 0..11);
        assert_eq!(rows(11..12, 12), 11..11);
        assert_eq!(rows(0..1, 1), 0..0);
    }
}
