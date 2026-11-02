//! The hybrid trunk: Qwen3.5-35B-A3B's language model, the CPU reference
//! (§3.4).
//!
//! 40 pre-norm layers, `layer_types` 3 Gated DeltaNet : 1 gated full
//! attention, each followed by the sparse MoE block:
//!
//! ```text
//! x += mixer((1 + w)-RMSNorm(x));  x += moe((1 + w)-RMSNorm(x))
//! hidden = (1 + w)-RMSNorm(x)                                   # model.norm
//! ```
//!
//! On the CPU the residual stream and every activation are F32; on Metal
//! and CUDA the layers run in the device column's forms (`device::trunk`:
//! an F32 residual and F16 layers on Metal, BF16 throughout on CUDA). The
//! caller picks the [`FishDrama3KernelPhase`]:
//! * [`FishDrama3Trunk::forward_span`] runs one request's rows from its
//!   position (prefill) and returns the final-norm hidden of every row: the
//!   teacher-forced tower reads them all;
//! * [`FishDrama3Trunk::forward_step`] runs one row for each of several
//!   requests, each with its own paged KV and state (decode). On the CPU it
//!   is a loop over the rows; on a device the rows run together layer by
//!   layer through row-invariant kernels only. Either way a row's bits never
//!   depend on its batch.
//!
//! Full-attention layers share one "trunk" paged domain holding only them
//! (model layers from `layer_types`); one prepared append covers all of
//! them, committed when the forward succeeds and aborted when it fails, so
//! the KV length, the state and the position always move together.
//! Logits come from the compact head (`head.rs`) on selected hidden rows.

pub mod attention;
pub mod deltanet;
pub mod moe;
pub mod state;

#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Tensor};
use serde_json::json;

use super::checkpoint::{trunk_layer, FishDrama3Checkpoint, TRUNK_PREFIX};
use super::config::{FishDrama3Config, FishDrama3LayerKind, FishDrama3TrunkConfig};
use super::device::trunk::{DeviceTrunk, FishDrama3MoeBackend};
use super::device::DeviceRow;
pub use super::linear::FishDrama3KernelPhase;
use super::linear::FishDrama3Linear;
use super::precision::{decide_cpu_forms, FishDrama3PrecisionReport, Q8_0_TOLERANCE};
use super::residency::{
    FishDrama3ClassForms, FishDrama3CpuPrecision, FishDrama3ProjectionClass, FishDrama3Residency,
    FishDrama3WeightForm,
};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

use attention::FishDrama3Attention;
use deltanet::FishDrama3DeltaNet;
use moe::FishDrama3Moe;
pub use moe::{route, FishDrama3Route};
pub use state::{FishDrama3TrunkCheckpoint, FishDrama3TrunkState};

/// `x / (1 + exp(-x))`.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `log(1 + exp(x))` without overflow: `max(x, 0) + log1p(exp(-|x|))`.
pub(crate) fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `x * rsqrt(mean(x²) + eps) * gain` per row of `gain.len()`.
pub(crate) fn rms_norm_rows(x: &[f32], width: usize, gain: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(gain.len(), width);
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inverse = 1.0 / (mean + eps).sqrt();
        out.extend(row.iter().zip(gain).map(|(v, g)| v * inverse * g));
    }
    out
}

/// How a stored norm weight becomes its gain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormGain {
    /// `1 + w`: every trunk norm but the DeltaNet output norm.
    ZeroCentred,
    /// `w`: the DeltaNet output norm (`GatedRMSNorm`).
    Plain,
}

/// The F32 gain of the BF16 norm weight `name` (`[width]`).
pub(crate) fn norm_gain(
    checkpoint: &FishDrama3Checkpoint,
    name: &str,
    width: usize,
    kind: NormGain,
) -> Result<Vec<f32>> {
    let w = checkpoint.bf16_to_f32(name, &[width])?;
    Ok(match kind {
        NormGain::ZeroCentred => w.into_iter().map(|w| 1.0 + w).collect(),
        NormGain::Plain => w,
    })
}

/// The BF16 projection `name` (`[rows, cols]`) in `form`.
pub(crate) fn load_linear(
    checkpoint: &FishDrama3Checkpoint,
    name: &str,
    rows: usize,
    cols: usize,
    form: FishDrama3WeightForm,
    device: &Device,
) -> Result<FishDrama3Linear> {
    FishDrama3Linear::from_values(
        checkpoint.bf16_to_f32(name, &[rows, cols])?,
        rows,
        cols,
        form,
        device,
    )
}

/// How a trunk load stores its weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3TrunkOptions {
    pub precision: FishDrama3CpuPrecision,
    /// The Q8_0 self-check's tolerance ([`Q8_0_TOLERANCE`]).
    pub q8_0_tolerance: f64,
    /// How a Metal or CUDA load runs its routed experts
    /// (`IZWI_FISH_DRAMA3_MOE_BACKEND`); the CPU always loops.
    pub moe_backend: FishDrama3MoeBackend,
}

impl Default for FishDrama3TrunkOptions {
    fn default() -> Self {
        Self {
            precision: FishDrama3CpuPrecision::Q8_0,
            q8_0_tolerance: Q8_0_TOLERANCE,
            moe_backend: FishDrama3MoeBackend::Fused,
        }
    }
}

enum Mixer {
    /// `slot` is the layer's place in the trunk's paged domain.
    Attention {
        slot: usize,
        attention: FishDrama3Attention,
    },
    /// `slot` is the layer's place among the linear layers' states.
    DeltaNet {
        slot: usize,
        net: FishDrama3DeltaNet,
    },
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    moe: FishDrama3Moe,
}

/// One request's row in a decode step.
pub struct FishDrama3StepRow<'a> {
    pub state: &'a mut FishDrama3TrunkState,
    pub cache: &'a mut PhysicalPagedKvCache,
}

#[derive(Debug, Default)]
struct Counters {
    spans: AtomicU64,
    span_rows: AtomicU64,
    steps: AtomicU64,
    /// Rows a step ran, one trunk pass each.
    step_rows: AtomicU64,
}

/// The trunk's weights in its backend's form.
enum Weights {
    /// The CPU reference: host norms, Q8_0 or F32 projections.
    Host {
        layers: Vec<Layer>,
        final_norm: Vec<f32>,
    },
    /// The device column's forms on Metal or CUDA (`device::trunk`).
    Device(DeviceTrunk),
}

pub struct FishDrama3Trunk {
    config: FishDrama3TrunkConfig,
    weights: Weights,
    /// Model layers of the paged domain, in slot order.
    attention_layers: Vec<u32>,
    device: Device,
    eps: f32,
    precision: FishDrama3PrecisionReport,
    counters: Counters,
}

impl std::fmt::Debug for FishDrama3Trunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3Trunk")
            .field("layers", &self.config.layers.len())
            .field("attention_layers", &self.attention_layers)
            .finish()
    }
}

/// What a traced forward saw at each layer, for parity tests. Every entry is
/// row-major over the forward's rows; a traced step of several rows holds
/// each row's layers in turn.
#[derive(Debug, Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct LayerTrace {
    /// The residual stream after the layer, `[n, hidden]`.
    pub(crate) outputs: Vec<Vec<f32>>,
    /// The mixer's output (DeltaNet or attention) before its residual add.
    pub(crate) mixers: Vec<Vec<f32>>,
    /// The MoE block's output (routed plus gated shared expert) before its
    /// residual add.
    pub(crate) moes: Vec<Vec<f32>>,
    /// The router's F32 logits, `[n, experts]`.
    pub(crate) router_logits: Vec<Vec<f32>>,
}

impl FishDrama3Trunk {
    /// Loads the trunk on `device`: on the CPU after the Q8_0 self-check
    /// decides each projection class's form; on Metal or CUDA in the device
    /// forms, which the model's load-time checks (`metal::self_check`,
    /// `cuda::self_check`) may then move per layer.
    pub fn load(
        checkpoint: &FishDrama3Checkpoint,
        config: &FishDrama3Config,
        device: &Device,
        options: FishDrama3TrunkOptions,
    ) -> Result<Self> {
        let residency = FishDrama3Residency::of(device);
        residency.require_supported(device)?;
        if residency != FishDrama3Residency::Cpu {
            return Self::load_device(checkpoint, config, device, options);
        }
        let (forms, precision) = decide_cpu_forms(
            checkpoint,
            config,
            options.precision,
            options.q8_0_tolerance,
            &FishDrama3ProjectionClass::TRUNK,
        )?;
        let t = &config.trunk;
        let h = t.hidden;
        let mut attention_slots = 0;
        let mut linear_slots = 0;
        let layers = t
            .layers
            .iter()
            .enumerate()
            .map(|(layer, kind)| {
                let l = trunk_layer(layer);
                let mixer = match kind {
                    FishDrama3LayerKind::FullAttention => {
                        attention_slots += 1;
                        Mixer::Attention {
                            slot: attention_slots - 1,
                            attention: FishDrama3Attention::load(
                                checkpoint, layer, t, &forms, device,
                            )?,
                        }
                    }
                    FishDrama3LayerKind::LinearAttention => {
                        linear_slots += 1;
                        Mixer::DeltaNet {
                            slot: linear_slots - 1,
                            net: FishDrama3DeltaNet::load(checkpoint, layer, t, &forms, device)?,
                        }
                    }
                };
                Ok(Layer {
                    input_norm: norm_gain(
                        checkpoint,
                        &format!("{l}.input_layernorm.weight"),
                        h,
                        NormGain::ZeroCentred,
                    )?,
                    post_norm: norm_gain(
                        checkpoint,
                        &format!("{l}.post_attention_layernorm.weight"),
                        h,
                        NormGain::ZeroCentred,
                    )?,
                    mixer,
                    moe: FishDrama3Moe::load(
                        checkpoint,
                        layer,
                        t,
                        &forms,
                        config.fp8_block,
                        device,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let trunk = Self {
            config: t.clone(),
            weights: Weights::Host {
                layers,
                final_norm: norm_gain(
                    checkpoint,
                    &format!("{TRUNK_PREFIX}.norm.weight"),
                    h,
                    NormGain::ZeroCentred,
                )?,
            },
            attention_layers: t
                .full_attention_layers()
                .into_iter()
                .map(|layer| layer as u32)
                .collect(),
            device: device.clone(),
            eps: t.rms_norm_eps as f32,
            precision,
            counters: Counters::default(),
        };
        tracing::info!(
            "Loaded the Fish Audio Drama 3 trunk on {:?}: {:.2} GiB, {} Q8_0 classes kept F32",
            device.location(),
            trunk.resident_bytes() as f64 / (1u64 << 30) as f64,
            trunk.precision.q8_fallback_classes
        );
        Ok(trunk)
    }

    fn load_device(
        checkpoint: &FishDrama3Checkpoint,
        config: &FishDrama3Config,
        device: &Device,
        options: FishDrama3TrunkOptions,
    ) -> Result<Self> {
        let t = &config.trunk;
        let weights = DeviceTrunk::load(checkpoint, config, options.moe_backend, device)?;
        let forms = FishDrama3ClassForms::of(FishDrama3Residency::of(device));
        let precision = FishDrama3PrecisionReport {
            forms: FishDrama3ProjectionClass::TRUNK
                .iter()
                .map(|&class| (class, forms.form(class)))
                .collect(),
            ..Default::default()
        };
        let trunk = Self {
            config: t.clone(),
            weights: Weights::Device(weights),
            attention_layers: t
                .full_attention_layers()
                .into_iter()
                .map(|layer| layer as u32)
                .collect(),
            device: device.clone(),
            eps: t.rms_norm_eps as f32,
            precision,
            counters: Counters::default(),
        };
        tracing::info!(
            "Loaded the Fish Audio Drama 3 trunk on {:?}: {:.2} GiB",
            device.location(),
            trunk.resident_bytes() as f64 / (1u64 << 30) as f64
        );
        Ok(trunk)
    }

    /// The device weights, for the load-time checks.
    pub(crate) fn device_trunk(&self) -> Option<&DeviceTrunk> {
        match &self.weights {
            Weights::Device(weights) => Some(weights),
            Weights::Host { .. } => None,
        }
    }

    pub(crate) fn device_trunk_mut(&mut self) -> Option<&mut DeviceTrunk> {
        match &mut self.weights {
            Weights::Device(weights) => Some(weights),
            Weights::Host { .. } => None,
        }
    }

    pub(crate) fn precision_mut(&mut self) -> &mut FishDrama3PrecisionReport {
        &mut self.precision
    }

    /// The CPU layers (tests reach into them).
    #[cfg(test)]
    fn host_layers(&self) -> &[Layer] {
        match &self.weights {
            Weights::Host { layers, .. } => layers,
            Weights::Device(_) => &[],
        }
    }

    pub fn config(&self) -> &FishDrama3TrunkConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The model layers the trunk's paged domain binds, in slot order.
    pub fn attention_layers(&self) -> &[u32] {
        &self.attention_layers
    }

    /// The zero state of a new request.
    pub fn new_state(&self) -> Result<FishDrama3TrunkState> {
        FishDrama3TrunkState::new(&self.config, &self.device)
    }

    pub fn precision_report(&self) -> &FishDrama3PrecisionReport {
        &self.precision
    }

    /// Resident bytes of the weights.
    pub fn resident_bytes(&self) -> usize {
        let (layers, final_norm) = match &self.weights {
            Weights::Host { layers, final_norm } => (layers, final_norm),
            Weights::Device(weights) => return weights.resident_bytes(),
        };
        let layers = layers
            .iter()
            .map(|layer| {
                let mixer = match &layer.mixer {
                    Mixer::Attention { attention, .. } => attention.resident_bytes(),
                    Mixer::DeltaNet { net, .. } => net.resident_bytes(),
                };
                mixer
                    + layer.moe.resident_bytes()
                    + (layer.input_norm.len() + layer.post_norm.len()) * 4
            })
            .sum::<usize>();
        layers + final_norm.len() * 4
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
        })
    }

    /// Rows a step has run so far, one trunk pass each.
    pub fn step_rows(&self) -> u64 {
        self.counters.step_rows.load(Ordering::Relaxed)
    }

    /// The cache must bind exactly the full-attention layers.
    pub fn validate_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        let a = &self.config.attention;
        cache.validate_sparse_model(&self.attention_layers, a.kv_heads, a.head_dim, a.head_dim)
    }

    /// Checks a forward of `n` rows at `start` against `state` and `cache`.
    fn check_clocks(
        &self,
        n: usize,
        start: usize,
        state: &FishDrama3TrunkState,
        cache: &PhysicalPagedKvCache,
    ) -> Result<()> {
        self.validate_cache(cache)?;
        if state.position() != start || cache.context_len() != start {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 trunk rows start at {start}; the state is at {} and the KV at {}",
                state.position(),
                cache.context_len()
            )));
        }
        let end = start + n;
        if n == 0 || end > self.config.max_positions || end > cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 trunk rows {start}..{end} exceed the {} positions or the KV's {}",
                self.config.max_positions,
                cache.capacity_tokens()
            )));
        }
        Ok(())
    }

    fn host_rows(&self, embeds: &Tensor, rows: Option<usize>) -> Result<(Vec<f32>, usize)> {
        let (n, width) = embeds.dims2()?;
        if width != self.config.hidden || rows.is_some_and(|rows| rows != n) {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 trunk embeddings are {:?}; expected [{}, {}]",
                embeds.dims(),
                rows.map_or("n".to_string(), |rows| rows.to_string()),
                self.config.hidden
            )));
        }
        Ok((
            embeds
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?,
            n,
        ))
    }

    /// The layers over `n` rows at `start`, updating `next` (the linear
    /// layers' states); returns the final-norm hidden `[n, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        mut x: Vec<f32>,
        n: usize,
        start: usize,
        next: &mut [state::FishDrama3LinearState],
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        phase: FishDrama3KernelPhase,
        mut trace: Option<&mut LayerTrace>,
    ) -> Result<Vec<f32>> {
        let Weights::Host { layers, final_norm } = &self.weights else {
            return Err(Error::InferenceError(
                "the Fish Audio Drama 3 host forward runs the CPU weights".into(),
            ));
        };
        let h = self.config.hidden;
        for layer in layers {
            let normed = rms_norm_rows(&x, h, &layer.input_norm, self.eps);
            let mixed = match &layer.mixer {
                Mixer::Attention { slot, attention } => {
                    attention.forward(&normed, n, start, cache, prepared, *slot, phase)?
                }
                Mixer::DeltaNet { slot, net } => {
                    net.forward(&normed, n, &mut next[*slot], phase)?
                }
            };
            x.iter_mut().zip(&mixed).for_each(|(x, m)| *x += m);
            let normed = rms_norm_rows(&x, h, &layer.post_norm, self.eps);
            let moe = layer.moe.forward(&normed, n, phase)?;
            x.iter_mut().zip(&moe).for_each(|(x, m)| *x += m);
            if let Some(trace) = trace.as_deref_mut() {
                trace
                    .router_logits
                    .push(layer.moe.router_logits_rows(&normed, n));
                trace.mixers.push(mixed);
                trace.moes.push(moe);
                trace.outputs.push(x.clone());
            }
        }
        Ok(rms_norm_rows(&x, h, final_norm, self.eps))
    }

    fn span(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        state: &mut FishDrama3TrunkState,
        cache: &mut PhysicalPagedKvCache,
        trace: Option<&mut LayerTrace>,
    ) -> Result<Tensor> {
        if let Weights::Device(weights) = &self.weights {
            return self.span_device(weights, embeds, start_pos, state, cache, trace);
        }
        let (x, n) = self.host_rows(embeds, None)?;
        self.check_clocks(n, start_pos, state, cache)?;
        let mut next = state.layers()?.to_vec();
        let mut prepared = cache.prepare_append(start_pos, n)?;
        match self.run(
            x,
            n,
            start_pos,
            &mut next,
            cache,
            &mut prepared,
            FishDrama3KernelPhase::Span,
            trace,
        ) {
            Ok(hidden) => {
                cache.commit_prepared(prepared)?;
                state.install(next, start_pos + n);
                self.counters.spans.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .span_rows
                    .fetch_add(n as u64, Ordering::Relaxed);
                Ok(Tensor::from_vec(
                    hidden,
                    (n, self.config.hidden),
                    &self.device,
                )?)
            }
            Err(error) => {
                let _ = cache.abort_prepared(prepared);
                Err(error)
            }
        }
    }

    /// Prefill: the rows `embeds` `[n, hidden]` of one request at positions
    /// `start_pos..start_pos + n`, which must be where `state` and `cache`
    /// stand. Returns the final-norm hidden of every row, `[n, hidden]` F32.
    /// On an error nothing moves.
    pub fn forward_span(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        state: &mut FishDrama3TrunkState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        self.span(embeds, start_pos, state, cache, None)
    }

    /// [`Self::forward_span`] recording every layer ([`LayerTrace`]).
    #[cfg(test)]
    pub(crate) fn forward_span_traced(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        state: &mut FishDrama3TrunkState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<(Tensor, LayerTrace)> {
        let mut trace = LayerTrace::default();
        let hidden = self.span(embeds, start_pos, state, cache, Some(&mut trace))?;
        Ok((hidden, trace))
    }

    /// Decode: row `r` of `embeds` `[rows.len(), hidden]` is the next row of
    /// request `rows[r]`, at its own position. Returns `[rows.len(), hidden]`
    /// final-norm hidden. Each row runs the whole trunk on its own (row-wise
    /// kernels); the rows commit together, and on an error none moves.
    pub fn forward_step(
        &self,
        embeds: &Tensor,
        rows: &mut [FishDrama3StepRow<'_>],
    ) -> Result<Tensor> {
        self.steps(embeds, rows, None)
    }

    /// [`Self::forward_step`] recording every layer of every row, the rows
    /// in turn.
    #[cfg(test)]
    pub(crate) fn forward_step_traced(
        &self,
        embeds: &Tensor,
        rows: &mut [FishDrama3StepRow<'_>],
    ) -> Result<(Tensor, LayerTrace)> {
        let mut trace = LayerTrace::default();
        let hidden = self.steps(embeds, rows, Some(&mut trace))?;
        Ok((hidden, trace))
    }

    /// The device rows of `embeds` `[n, hidden]`, checked.
    fn device_rows(&self, embeds: &Tensor, rows: Option<usize>) -> Result<(Tensor, usize)> {
        let (n, width) = embeds.dims2()?;
        if width != self.config.hidden || rows.is_some_and(|rows| rows != n) {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 trunk embeddings are {:?}; expected [{}, {}]",
                embeds.dims(),
                rows.map_or("n".to_string(), |rows| rows.to_string()),
                self.config.hidden
            )));
        }
        Ok((embeds.to_device(&self.device)?.to_dtype(DType::F32)?, n))
    }

    fn span_device(
        &self,
        weights: &DeviceTrunk,
        embeds: &Tensor,
        start_pos: usize,
        state: &mut FishDrama3TrunkState,
        cache: &mut PhysicalPagedKvCache,
        trace: Option<&mut LayerTrace>,
    ) -> Result<Tensor> {
        let (x, n) = self.device_rows(embeds, None)?;
        self.check_clocks(n, start_pos, state, cache)?;
        let mut next = state.layers()?.to_vec();
        let mut prepared = cache.prepare_append(start_pos, n)?;
        let result = {
            let mut rows = [DeviceRow {
                start: start_pos,
                rows: n,
                cache: &*cache,
                prepared: &mut prepared,
            }];
            let mut states = [next.as_mut_slice()];
            weights.run(
                x,
                &mut rows,
                &mut states,
                FishDrama3KernelPhase::Span,
                trace,
            )
        };
        match result {
            Ok(hidden) => {
                cache.commit_prepared(prepared)?;
                state.install(next, start_pos + n);
                self.counters.spans.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .span_rows
                    .fetch_add(n as u64, Ordering::Relaxed);
                Ok(hidden)
            }
            Err(error) => {
                let _ = cache.abort_prepared(prepared);
                Err(error)
            }
        }
    }

    /// A decode step on a device: every row's append is prepared, the rows
    /// run together layer by layer, and they commit together.
    fn steps_device(
        &self,
        weights: &DeviceTrunk,
        embeds: &Tensor,
        rows: &mut [FishDrama3StepRow<'_>],
        trace: Option<&mut LayerTrace>,
    ) -> Result<Tensor> {
        let (x, count) = self.device_rows(embeds, Some(rows.len()))?;
        if count == 0 {
            return Err(Error::InvalidInput("a decode step needs a row".into()));
        }
        let mut prepared = Vec::with_capacity(count);
        let mut next = Vec::with_capacity(count);
        for row in rows.iter_mut() {
            let start = row.state.position();
            let attempt = (|| {
                self.check_clocks(1, start, row.state, row.cache)?;
                let layers = row.state.layers()?.to_vec();
                Ok::<_, Error>((layers, row.cache.prepare_append(start, 1)?))
            })();
            match attempt {
                Ok((layers, step)) => {
                    next.push(layers);
                    prepared.push(step);
                }
                Err(error) => {
                    for (row, step) in rows.iter_mut().zip(prepared) {
                        let _ = row.cache.abort_prepared(step);
                    }
                    return Err(error);
                }
            }
        }
        let result = {
            let mut device_rows = rows
                .iter()
                .zip(prepared.iter_mut())
                .map(|(row, step)| DeviceRow {
                    start: row.state.position(),
                    rows: 1,
                    cache: &*row.cache,
                    prepared: step,
                })
                .collect::<Vec<_>>();
            let mut states = next.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
            weights.run(
                x,
                &mut device_rows,
                &mut states,
                FishDrama3KernelPhase::Step,
                trace,
            )
        };
        let hidden = match result {
            Ok(hidden) => hidden,
            Err(error) => {
                for (row, step) in rows.iter_mut().zip(prepared) {
                    let _ = row.cache.abort_prepared(step);
                }
                return Err(error);
            }
        };
        for ((row, step), layers) in rows.iter_mut().zip(prepared).zip(next) {
            let start = row.state.position();
            row.cache.commit_prepared(step)?;
            row.state.install(layers, start + 1);
        }
        self.counters
            .step_rows
            .fetch_add(count as u64, Ordering::Relaxed);
        self.counters.steps.fetch_add(1, Ordering::Relaxed);
        Ok(hidden)
    }

    fn steps(
        &self,
        embeds: &Tensor,
        rows: &mut [FishDrama3StepRow<'_>],
        mut trace: Option<&mut LayerTrace>,
    ) -> Result<Tensor> {
        if let Weights::Device(weights) = &self.weights {
            // A device runs a step's rows together layer by layer; a trace
            // keeps the CPU's row-major layout for one row only.
            if trace.is_some() && rows.len() != 1 {
                return Err(Error::InferenceError(
                    "a traced device step takes one row".into(),
                ));
            }
            return self.steps_device(weights, embeds, rows, trace);
        }
        let (x, count) = self.host_rows(embeds, Some(rows.len()))?;
        if count == 0 {
            return Err(Error::InvalidInput("a decode step needs a row".into()));
        }
        let h = self.config.hidden;
        let mut done = Vec::with_capacity(count);
        let mut failure = None;
        for (r, row) in rows.iter_mut().enumerate() {
            let start = row.state.position();
            let attempt = (|| {
                self.check_clocks(1, start, row.state, row.cache)?;
                let next = row.state.layers()?.to_vec();
                let prepared = row.cache.prepare_append(start, 1)?;
                Ok::<_, Error>((next, prepared))
            })();
            let (mut next, mut prepared) = match attempt {
                Ok(ready) => ready,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            };
            match self.run(
                x[r * h..(r + 1) * h].to_vec(),
                1,
                start,
                &mut next,
                row.cache,
                &mut prepared,
                FishDrama3KernelPhase::Step,
                trace.as_deref_mut(),
            ) {
                Ok(hidden) => {
                    self.counters.step_rows.fetch_add(1, Ordering::Relaxed);
                    done.push((next, prepared, hidden));
                }
                Err(error) => {
                    let _ = row.cache.abort_prepared(prepared);
                    failure = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = failure {
            for (row, (_, prepared, _)) in rows.iter_mut().zip(done) {
                let _ = row.cache.abort_prepared(prepared);
            }
            return Err(error);
        }
        let mut out = Vec::with_capacity(count * h);
        for (row, (next, prepared, hidden)) in rows.iter_mut().zip(done) {
            let start = row.state.position();
            row.cache.commit_prepared(prepared)?;
            row.state.install(next, start + 1);
            out.extend(hidden);
        }
        self.counters.steps.fetch_add(1, Ordering::Relaxed);
        Ok(Tensor::from_vec(out, (count, h), &self.device)?)
    }
}
