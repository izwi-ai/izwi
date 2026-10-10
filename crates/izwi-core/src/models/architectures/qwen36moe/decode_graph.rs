//! Piecewise CUDA graph decode for one row (Phase 3).
//!
//! The decode step splits at the attention layers. Everything else runs as
//! captured graph segments:
//! - segment 0: the DeltaNet layers before the first attention layer, then
//!   that layer's norm and q/k/v projection;
//! - each later segment: the previous attention layer's gate, `o_proj`, norm
//!   and MoE, the DeltaNet layers that follow, then the next attention
//!   layer's norm and q/k/v projection;
//! - the last segment: the final attention layer's tail.
//!
//! Attention itself stays eager between segments: its q/k RoPE takes the
//! step's positions as kernel scalars, and the paged KV write and attention
//! depend on per-step slots and context length. The LM head and token
//! selection also stay eager.
//!
//! DeltaNet state moves through a device address table (`gdn::StateTable`).
//! Each step allocates one slab of fresh state outputs and writes every
//! layer's six addresses into the table with one upload. The captured graphs
//! read the engine's current state tensors and write the slab, so there are
//! no state copies, and committed state is never mutated in place.
//!
//! Phases, in lockstep across the segments:
//! 1. **warm**: run eagerly under the htod-cache guard.
//! 2. **capture**: capture each segment, then verify it.
//! 3. **verify**: verify again on fresh inputs, which catches values a
//!    capture baked in.
//! 4. **replay**: replay only.
//!
//! A verification runs the segment eagerly into the published state buffers,
//! replays the graph into scratch buffers, and compares outputs and state.
//! The step always continues on the eager results, so a mismatch or a
//! capture failure disables graph decode without affecting the step's output.
//!
//! Off CUDA there is no capture: every step runs the same segments eagerly,
//! with explicit state tensors. Tests use that to check the segmentation
//! against the standard decode path.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Tensor};

use super::{
    ConvRingState, Qwen36FeedForward, Qwen36FullAttention, Qwen36Hidden, Qwen36Layer,
    Qwen36LayerRuntimeState, Qwen36LinearAttention, Qwen36Mixer, Qwen36TextModel,
    Qwen36TextRuntimeState,
};
use crate::backends::kv::{KvSlotMap, KvWriteCompletionCollector};
use crate::error::{Error, Result};
use crate::kernels::cuda::gdn;
use crate::kernels::cuda::segment_graph::{DeviceTable, SegmentGraph, SegmentInput};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::architectures::qwen36moe::fast_path::{
    compare_values, legacy_requested, Qwen36CudaSwitches, Qwen36FusedPath,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// Environment switch for graph decode (`legacy`, `off`, `0`, `false`).
pub(super) const GRAPHS_ENV: &str = "IZWI_QWEN36_CUDA_GRAPHS";

/// Where one DeltaNet layer reads its state and writes the next one.
pub(super) enum LinearStateIo {
    /// CUDA: addresses in the step's device table.
    #[cfg(feature = "cuda")]
    Table(gdn::StateTable),
    /// Other devices: explicit tensors; the outputs are written in place.
    Direct {
        history: [Tensor; 3],
        state: Tensor,
        history_out: Tensor,
        state_out: Tensor,
    },
}

/// One segment: everything between two attention cores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Segment {
    /// Full-attention layer whose tail (gate, `o_proj`, norm, MoE) opens it.
    opens_after: Option<usize>,
    /// DeltaNet layers run whole, in order.
    linear: Vec<usize>,
    /// Full-attention layer whose norm and q/k/v projection close it.
    closes_before: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Phase {
    Warm,
    Capture,
    Verify,
    Replay,
    Disabled(String),
}

impl Phase {
    fn name(&self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Capture => "capture",
            Self::Verify => "verify",
            Self::Replay => "replay",
            Self::Disabled(_) => "disabled",
        }
    }
}

struct Runtime {
    phase: Phase,
    graphs: Vec<SegmentGraph>,
    table: Option<DeviceTable>,
}

/// Graph decode for one model (see the module docs).
pub(super) struct Qwen36DecodeGraphs {
    segments: Vec<Segment>,
    /// Per layer: the DeltaNet ordinal (table slot `ordinal * TABLE_ENTRIES`).
    linear_ordinal: Vec<Option<usize>>,
    linear_count: usize,
    /// Capture graphs (CUDA). Otherwise every step runs the segments eagerly.
    capture: bool,
    runtime: Mutex<Runtime>,
    warmups: AtomicU64,
    captures: AtomicU64,
    replays: AtomicU64,
    verified_segments: AtomicU64,
    eager_steps: AtomicU64,
}

/// Plan the segments from the layer types.
pub(super) fn plan_segments(layers: &[Qwen36Layer]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current = Segment {
        opens_after: None,
        linear: Vec::new(),
        closes_before: None,
    };
    for (index, layer) in layers.iter().enumerate() {
        match layer.mixer {
            Qwen36Mixer::Linear(_) => current.linear.push(index),
            Qwen36Mixer::Full(_) => {
                current.closes_before = Some(index);
                segments.push(std::mem::replace(
                    &mut current,
                    Segment {
                        opens_after: Some(index),
                        linear: Vec::new(),
                        closes_before: None,
                    },
                ));
            }
        }
    }
    segments.push(current);
    segments
}

impl Qwen36DecodeGraphs {
    /// Whether graph decode applies to `model`, and why not. `allow_eager`
    /// admits non-CUDA devices with eager segments (tests).
    pub(super) fn resolve(
        model: &Qwen36TextModel,
        switches: &Qwen36CudaSwitches,
        allow_eager: bool,
    ) -> (Option<Self>, Qwen36FusedPath) {
        let device = &model.device;
        let refuse = |reason: String| (None, Qwen36FusedPath::legacy(reason));
        if legacy_requested(GRAPHS_ENV) {
            return refuse(format!("{GRAPHS_ENV}=legacy"));
        }
        if let Some(reason) = switches.graphs_off {
            return refuse(reason.to_string());
        }
        if !device.is_cuda() && !allow_eager {
            return refuse("CUDA graph decode runs on CUDA only".into());
        }
        if model.finite_diagnostics_enabled {
            return refuse("CUDA graph decode is off while finite diagnostics read values".into());
        }
        let mut linear_ordinal = Vec::with_capacity(model.layers.len());
        let mut linear_count = 0usize;
        for layer in &model.layers {
            match &layer.mixer {
                Qwen36Mixer::Linear(mixer) => {
                    if mixer.fused_decode.is_none() {
                        return refuse(
                            "CUDA graph decode needs the fused DeltaNet decode on every layer"
                                .into(),
                        );
                    }
                    linear_ordinal.push(Some(linear_count));
                    linear_count += 1;
                }
                Qwen36Mixer::Full(_) => linear_ordinal.push(None),
            }
            if let Qwen36FeedForward::Sparse(moe) = &layer.ffn {
                if !moe.backend().is_fused() {
                    return refuse("CUDA graph decode needs the fused MoE on every layer".into());
                }
            }
        }
        let segments = plan_segments(&model.layers);
        let graphs = segments.iter().map(|_| SegmentGraph::default()).collect();
        (
            Some(Self {
                segments,
                linear_ordinal,
                linear_count,
                capture: device.is_cuda(),
                runtime: Mutex::new(Runtime {
                    phase: Phase::Warm,
                    graphs,
                    table: None,
                }),
                warmups: AtomicU64::new(0),
                captures: AtomicU64::new(0),
                replays: AtomicU64::new(0),
                verified_segments: AtomicU64::new(0),
                eager_steps: AtomicU64::new(0),
            }),
            Qwen36FusedPath::Fused,
        )
    }

    /// Diagnostics: phase, counters and the reason graph decode was turned
    /// off at runtime, if it was.
    pub(super) fn summary(&self) -> serde_json::Value {
        let phase = self
            .runtime
            .lock()
            .map(|runtime| runtime.phase.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().phase.clone());
        let disabled = match &phase {
            Phase::Disabled(reason) => Some(reason.clone()),
            _ => None,
        };
        serde_json::json!({
            "backend": if disabled.is_some() { "legacy" } else { "fused" },
            "legacy_reasons": disabled.iter().collect::<Vec<_>>(),
            "phase": phase.name(),
            "capture": self.capture,
            "segments": self.segments.len(),
            "warmups": self.warmups.load(Relaxed),
            "captures": self.captures.load(Relaxed),
            "replays": self.replays.load(Relaxed),
            "verified_segments": self.verified_segments.load(Relaxed),
            "eager_steps": self.eager_steps.load(Relaxed),
        })
    }

    /// Run one decode row (`embedded` `[1, 1, hidden]`) through the segments.
    /// Returns the pre-norm hidden, or `None` when graph decode does not take
    /// the step (disabled, busy, or a state it cannot address); the caller
    /// then runs the eager path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        model: &Qwen36TextModel,
        embedded: &Tensor,
        position: [usize; 3],
        state: &mut Qwen36TextRuntimeState,
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<Option<Tensor>> {
        let Ok(mut runtime) = self.runtime.try_lock() else {
            self.eager_steps.fetch_add(1, Relaxed);
            return Ok(None);
        };
        if matches!(runtime.phase, Phase::Disabled(_)) {
            self.eager_steps.fetch_add(1, Relaxed);
            return Ok(None);
        }
        let device = model.device.clone();
        for (index, layer) in model.layers.iter().enumerate() {
            layer.ensure_state_initialized(&mut state.layers[index], &device)?;
        }
        let Some(inputs) = StateInputs::gather(model, state)? else {
            self.eager_steps.fetch_add(1, Relaxed);
            return Ok(None);
        };
        let step = self.run_step(
            &mut runtime,
            model,
            &device,
            &inputs,
            embedded,
            position,
            cache,
            slots,
            metadata,
            completions,
        );
        let (hidden, phase, published) = match step {
            Ok(done) => done,
            Err(error) => {
                // A failure while graphs are active may be the graphs'
                // fault: decode eagerly from now on.
                if self.capture && runtime.phase != Phase::Warm {
                    runtime.phase = Phase::Disabled(format!("graph decode step failed: {error}"));
                    for graph in runtime.graphs.iter_mut() {
                        graph.reset();
                    }
                }
                return Err(error);
            }
        };
        runtime.phase = match phase {
            Phase::Warm if self.capture => Phase::Capture,
            Phase::Warm => Phase::Warm,
            Phase::Capture => Phase::Verify,
            Phase::Verify => Phase::Replay,
            other => other,
        };
        if let Phase::Disabled(reason) = runtime.phase.clone() {
            for graph in runtime.graphs.iter_mut() {
                graph.reset();
            }
            tracing::warn!(%reason, "Qwen3.6 CUDA graph decode disabled; decoding eagerly");
        }
        drop(runtime);
        published.publish(model, state)?;
        Ok(Some(hidden))
    }

    /// The segment loop for one step. Returns the pre-norm hidden, the phase
    /// the step ended in, and the state outputs to publish.
    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &self,
        runtime: &mut Runtime,
        model: &Qwen36TextModel,
        device: &Device,
        inputs: &StateInputs,
        embedded: &Tensor,
        position: [usize; 3],
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<(Tensor, Phase, StateOutputs)> {
        let published = StateOutputs::allocate(model, device)?;
        let ios = self.state_io(runtime, model, inputs, &published, device)?;
        let run = |segment: &Segment, inputs: &[Tensor]| run_segment(model, segment, &ios, inputs);
        // The segment runner speaks Candle errors.
        let run_candle = |segment: &Segment, inputs: &[Tensor]| {
            run(segment, inputs).map_err(candle_core::Error::wrap)
        };
        let mut phase = runtime.phase.clone();
        let mut carried: Vec<Tensor> = vec![embedded.clone()];
        // The previous segment's retained graph outputs, adopted by the next
        // capture so replays chain without copies.
        let mut previous_graph: Option<Vec<Tensor>> = None;
        let mut physical_layer = 0usize;
        for (index, segment) in self.segments.iter().enumerate() {
            let step_inputs = carried.clone();
            let outputs = match &phase {
                Phase::Warm => {
                    self.warmups.fetch_add(1, Relaxed);
                    SegmentGraph::warm(device, &step_inputs, |inputs| run_candle(segment, inputs))?
                }
                Phase::Capture | Phase::Verify => {
                    let adopted = previous_graph.take();
                    if phase == Phase::Capture {
                        let capture_inputs = segment_inputs(&step_inputs, adopted.as_deref());
                        let captured = runtime.graphs[index]
                            .capture(&capture_inputs, |inputs| run_candle(segment, inputs));
                        if let Err(error) = captured {
                            phase = Phase::Disabled(format!(
                                "capture of segment {index} failed: {error}"
                            ));
                        } else {
                            self.captures.fetch_add(1, Relaxed);
                        }
                    }
                    let eager = run(segment, &step_inputs)?;
                    if !matches!(phase, Phase::Disabled(_)) {
                        match self.verify(
                            runtime,
                            index,
                            segment,
                            model,
                            inputs,
                            &published,
                            device,
                            &step_inputs,
                            &eager,
                        ) {
                            Ok(graph_outputs) => {
                                self.verified_segments.fetch_add(1, Relaxed);
                                previous_graph = Some(graph_outputs);
                            }
                            Err(reason) => phase = Phase::Disabled(reason),
                        }
                    }
                    eager
                }
                Phase::Replay => {
                    let replay_inputs = segment_inputs(&step_inputs, previous_graph.as_deref());
                    let outputs = runtime.graphs[index].replay(&replay_inputs)?;
                    self.replays.fetch_add(1, Relaxed);
                    previous_graph = Some(outputs.clone());
                    outputs
                }
                Phase::Disabled(_) => run(segment, &step_inputs)?,
            };
            carried = match segment.closes_before {
                Some(layer_index) => {
                    let [residual, q_proj, keys, values] = <[Tensor; 4]>::try_from(outputs)
                        .map_err(|_| segment_error("a closing segment returns 4 outputs"))?;
                    let attention = full_attention(&model.layers[layer_index])?;
                    let attended = attention.decode_attend(
                        &q_proj,
                        &keys,
                        &values,
                        &[position],
                        &[cache],
                        slots,
                        metadata,
                        completions,
                        physical_layer,
                    )?;
                    physical_layer += 1;
                    vec![residual, q_proj, attended]
                }
                None => outputs,
            };
        }
        let attention_layers = model
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, Qwen36Mixer::Full(_)))
            .count();
        if physical_layer != attention_layers {
            return Err(segment_error(
                "graph decode did not cover every attention layer",
            ));
        }
        let [hidden] = <[Tensor; 1]>::try_from(carried)
            .map_err(|_| segment_error("the last segment returns the hidden state"))?;
        // A graph's retained output is overwritten by its next replay; hand
        // the caller its own copy.
        let hidden = if phase == Phase::Replay {
            hidden.copy()?
        } else {
            hidden
        };
        Ok((hidden, phase, published))
    }

    /// The per-layer state IO for this step. On CUDA this refreshes the
    /// device table (one upload) and hands out table slots.
    fn state_io(
        &self,
        runtime: &mut Runtime,
        model: &Qwen36TextModel,
        inputs: &StateInputs,
        outputs: &StateOutputs,
        device: &Device,
    ) -> Result<Vec<Option<LinearStateIo>>> {
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            let entries = self.linear_count * gdn::TABLE_ENTRIES;
            if runtime.table.as_ref().map(DeviceTable::entries) != Some(entries) {
                runtime.table = Some(DeviceTable::new(device, entries)?);
            }
            let table = runtime.table.as_mut().expect("table allocated");
            let mut values = Vec::with_capacity(entries);
            for (layer, ordinal) in self.linear_ordinal.iter().enumerate() {
                if ordinal.is_some() {
                    values.extend(table_entries(inputs, outputs, layer)?);
                }
            }
            table.write(0, &values)?;
            let address = table.address();
            return Ok(self
                .linear_ordinal
                .iter()
                .map(|ordinal| {
                    ordinal.map(|ordinal| {
                        LinearStateIo::Table(gdn::StateTable {
                            address,
                            slot: ordinal * gdn::TABLE_ENTRIES,
                        })
                    })
                })
                .collect());
        }
        let _ = (runtime, device);
        (0..model.layers.len())
            .map(|layer| {
                Ok(
                    match (
                        inputs.layers[layer].as_ref(),
                        outputs.layers[layer].as_ref(),
                    ) {
                        (Some(input), Some(output)) => Some(LinearStateIo::Direct {
                            history: input.history.clone(),
                            state: input.state.clone(),
                            history_out: output.history.clone(),
                            state_out: output.state.clone(),
                        }),
                        _ => None,
                    },
                )
            })
            .collect()
    }

    /// Replay segment `index` into scratch state buffers and compare it with
    /// the eager run's outputs and published state. Returns the graph's
    /// retained outputs for the next capture to adopt.
    #[allow(clippy::too_many_arguments)]
    fn verify(
        &self,
        runtime: &mut Runtime,
        index: usize,
        segment: &Segment,
        model: &Qwen36TextModel,
        inputs: &StateInputs,
        published: &StateOutputs,
        device: &Device,
        step_inputs: &[Tensor],
        eager: &[Tensor],
    ) -> std::result::Result<Vec<Tensor>, String> {
        let mut check = || -> Result<Vec<Tensor>> {
            let scratch = StateOutputs::allocate(model, device)?;
            self.point_segment(runtime, segment, inputs, &scratch)?;
            let replay_inputs = segment_inputs(step_inputs, None);
            let graph = runtime.graphs[index].replay(&replay_inputs)?;
            self.point_segment(runtime, segment, inputs, published)?;
            if graph.len() != eager.len() {
                return Err(segment_error("graph and eager outputs differ in count"));
            }
            for (slot, (graph_out, eager_out)) in graph.iter().zip(eager).enumerate() {
                compare_values(
                    &format!("graph segment {index} output {slot}"),
                    &host(graph_out)?,
                    &host(eager_out)?,
                    0.01,
                    0.02,
                )?;
            }
            for &layer in &segment.linear {
                let (Some(graph_state), Some(eager_state)) = (
                    scratch.layers[layer].as_ref(),
                    published.layers[layer].as_ref(),
                ) else {
                    return Err(segment_error("verified layer has no state outputs"));
                };
                compare_values(
                    &format!("graph segment {index} layer {layer} conv history"),
                    &host(&graph_state.history)?,
                    &host(&eager_state.history)?,
                    1e-6,
                    1e-6,
                )?;
                compare_values(
                    &format!("graph segment {index} layer {layer} recurrent state"),
                    &host(&graph_state.state)?,
                    &host(&eager_state.state)?,
                    0.01,
                    0.02,
                )?;
            }
            Ok(graph)
        };
        check().map_err(|error| format!("verification failed: {error}"))
    }

    /// Point `segment`'s DeltaNet table entries at `outputs` (CUDA).
    fn point_segment(
        &self,
        runtime: &mut Runtime,
        segment: &Segment,
        inputs: &StateInputs,
        outputs: &StateOutputs,
    ) -> Result<()> {
        let Some(table) = runtime.table.as_mut() else {
            return Ok(());
        };
        for &layer in &segment.linear {
            let ordinal = self.linear_ordinal[layer]
                .ok_or_else(|| segment_error("segment layer is not a DeltaNet layer"))?;
            table.write(
                ordinal * gdn::TABLE_ENTRIES,
                &table_entries(inputs, outputs, layer)?,
            )?;
        }
        Ok(())
    }
}

/// The segment's inputs as runner inputs. With `adopted` (the previous
/// segment's retained graph outputs `[residual, q_proj, k, v]`), the carried
/// residual and `q_proj` are adopted in place.
fn segment_inputs<'a>(
    carried: &'a [Tensor],
    adopted: Option<&'a [Tensor]>,
) -> Vec<SegmentInput<'a>> {
    carried
        .iter()
        .enumerate()
        .map(|(slot, tensor)| match adopted {
            Some(previous) if carried.len() == 3 && slot < 2 => SegmentInput {
                tensor: &previous[slot],
                adopt: true,
            },
            _ => SegmentInput {
                tensor,
                adopt: false,
            },
        })
        .collect()
}

/// One segment's computation. `inputs`: `[embedded]` for the first segment,
/// `[residual, q_proj, attention output]` for the others. Returns
/// `[residual, q_proj, k, v]` when the segment closes before an attention
/// layer, `[hidden]` for the last one.
fn run_segment(
    model: &Qwen36TextModel,
    segment: &Segment,
    ios: &[Option<LinearStateIo>],
    inputs: &[Tensor],
) -> Result<Vec<Tensor>> {
    let layers = &model.layers;
    let mut hidden = match (segment.opens_after, inputs) {
        (None, [embedded]) => Qwen36Hidden::new(embedded.clone()),
        (Some(index), [residual, q_proj, attended]) => {
            let mixed = full_attention(&layers[index])?.decode_output(attended, q_proj)?;
            layers[index].post_mixer(residual, &mixed)?
        }
        _ => return Err(segment_error("segment inputs do not match its plan")),
    };
    for &index in &segment.linear {
        let layer = &layers[index];
        let (residual, normalized) = hidden.normalized(&layer.attn_norm)?;
        let Qwen36Mixer::Linear(mixer) = &layer.mixer else {
            return Err(segment_error("planned DeltaNet layer has another mixer"));
        };
        let io = ios[index]
            .as_ref()
            .ok_or_else(|| segment_error("DeltaNet layer has no state IO"))?;
        let mixed = mixer.forward_segment(&normalized, io)?;
        hidden = layer.post_mixer(&residual, &mixed)?;
    }
    match segment.closes_before {
        Some(index) => {
            let (residual, normalized) = hidden.normalized(&layers[index].attn_norm)?;
            let (q_proj, keys, values) =
                full_attention(&layers[index])?.decode_projections(&normalized)?;
            Ok(vec![residual, q_proj, keys, values])
        }
        None => Ok(vec![hidden.materialize()?]),
    }
}

impl Qwen36LinearAttention {
    /// One token through the fused DeltaNet kernels with explicit state IO:
    /// the graph-segment form of the fused decode.
    fn forward_segment(&self, hidden_states: &Tensor, io: &LinearStateIo) -> Result<Tensor> {
        let spec = self
            .fused_decode
            .as_ref()
            .ok_or_else(|| segment_error("graph decode needs the fused DeltaNet decode"))?;
        let (mixed_qkv, z, beta_raw, alpha) = self.decode_projections(hidden_states)?;
        let (x, z, beta_raw, alpha) = (mixed_qkv.i(0)?, z.i(0)?, beta_raw.i(0)?, alpha.i(0)?);
        let y = match io {
            #[cfg(feature = "cuda")]
            LinearStateIo::Table(table) => {
                let conv = gdn::conv_decode_table(&x, &self.conv_kernel, *table)?;
                gdn::recurrent_decode_table(
                    &conv,
                    &z,
                    &beta_raw,
                    &alpha,
                    &self.dt_bias,
                    &self.a,
                    &self.norm.weight,
                    *table,
                    spec,
                )?
            }
            LinearStateIo::Direct {
                history,
                state,
                history_out,
                state_out,
            } => {
                let (conv, next_history) = gdn::conv_decode(
                    &x,
                    &self.conv_kernel,
                    [&history[0], &history[1], &history[2]],
                )?;
                let (y, next) = gdn::recurrent_decode(
                    &conv,
                    &z,
                    &beta_raw,
                    &alpha,
                    &self.dt_bias,
                    &self.a,
                    &self.norm.weight,
                    state,
                    spec,
                )?;
                history_out.slice_set(&next_history.reshape(history_out.shape())?, 0, 0)?;
                state_out.slice_set(&next.reshape(state_out.shape())?, 0, 0)?;
                y
            }
        };
        self.out_proj
            .forward(&y.reshape((1, 1, self.num_v_heads * self.head_v_dim))?)
    }
}

/// One DeltaNet layer's current state, contiguous and F32.
struct LayerStateIn {
    history: [Tensor; 3],
    state: Tensor,
}

struct StateInputs {
    layers: Vec<Option<LayerStateIn>>,
}

impl StateInputs {
    /// The row's current DeltaNet state, or `None` when some layer's state is
    /// not in the fused-decode form (3 F32 history slots, F32 state).
    fn gather(model: &Qwen36TextModel, state: &Qwen36TextRuntimeState) -> Result<Option<Self>> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for (layer, layer_state) in model.layers.iter().zip(&state.layers) {
            let (
                Qwen36Mixer::Linear(_),
                Qwen36LayerRuntimeState::Linear {
                    conv_state: Some(ring),
                    recurrent_state: Some(recurrent),
                },
            ) = (&layer.mixer, layer_state)
            else {
                if matches!(layer.mixer, Qwen36Mixer::Linear(_)) {
                    return Ok(None);
                }
                layers.push(None);
                continue;
            };
            if ring.slots.len() != gdn::CONV_TAPS - 1
                || ring.next_idx >= ring.slots.len()
                || ring.slots.iter().any(|slot| slot.dtype() != DType::F32)
                || recurrent.dtype() != DType::F32
            {
                return Ok(None);
            }
            let ordered = ring
                .ordered_slots()
                .map(|slot| slot.contiguous().map_err(Error::from))
                .collect::<Result<Vec<_>>>()?;
            let history = <[Tensor; 3]>::try_from(ordered)
                .map_err(|_| segment_error("conv history has three slots"))?;
            layers.push(Some(LayerStateIn {
                history,
                state: recurrent.contiguous()?,
            }));
        }
        Ok(Some(Self { layers }))
    }
}

/// One DeltaNet layer's fresh state outputs: views into the step's slab.
struct LayerStateOut {
    history: Tensor,
    state: Tensor,
}

struct StateOutputs {
    layers: Vec<Option<LayerStateOut>>,
}

impl StateOutputs {
    /// One slab for every DeltaNet layer's next history `[3, conv_dim]` and
    /// state `[1, Hv, Dk, Dv]` (F32). The kernels write every element.
    fn allocate(model: &Qwen36TextModel, device: &Device) -> Result<Self> {
        let sizes = model
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                Qwen36Mixer::Linear(mixer) => Some((
                    mixer.conv_dim,
                    (mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim),
                )),
                Qwen36Mixer::Full(_) => None,
            })
            .collect::<Vec<_>>();
        let total = sizes
            .iter()
            .flatten()
            .map(|(conv_dim, (hv, dk, dv))| (gdn::CONV_TAPS - 1) * conv_dim + hv * dk * dv)
            .sum::<usize>();
        let slab = uninit_f32(total, device)?;
        let mut offset = 0usize;
        let mut layers = Vec::with_capacity(sizes.len());
        for size in sizes {
            layers.push(match size {
                Some((conv_dim, (hv, dk, dv))) => {
                    let history_len = (gdn::CONV_TAPS - 1) * conv_dim;
                    let history = slab
                        .narrow(0, offset, history_len)?
                        .reshape((gdn::CONV_TAPS - 1, conv_dim))?;
                    offset += history_len;
                    let state = slab
                        .narrow(0, offset, hv * dk * dv)?
                        .reshape((1, hv, dk, dv))?;
                    offset += hv * dk * dv;
                    Some(LayerStateOut { history, state })
                }
                None => None,
            });
        }
        Ok(Self { layers })
    }

    /// Hand the written state to the row: each ring becomes the packed history.
    fn publish(self, model: &Qwen36TextModel, state: &mut Qwen36TextRuntimeState) -> Result<()> {
        for ((layer, out), layer_state) in model
            .layers
            .iter()
            .zip(self.layers)
            .zip(state.layers.iter_mut())
        {
            let (Qwen36Mixer::Linear(mixer), Some(out)) = (&layer.mixer, out) else {
                continue;
            };
            let Qwen36LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer_state
            else {
                return Err(segment_error("DeltaNet layer lost its runtime state"));
            };
            *conv_state = Some(ConvRingState::from_history(out.history.reshape((
                gdn::CONV_TAPS - 1,
                mixer.conv_dim,
                1,
            ))?)?);
            *recurrent_state = Some(out.state);
        }
        Ok(())
    }
}

/// Table entries for `layer`: h0, h1, h2, next history, state in, state out.
#[cfg(feature = "cuda")]
fn table_entries(
    inputs: &StateInputs,
    outputs: &StateOutputs,
    layer: usize,
) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    let (Some(input), Some(output)) = (
        inputs.layers[layer].as_ref(),
        outputs.layers[layer].as_ref(),
    ) else {
        return Err(segment_error("DeltaNet layer has no state to address"));
    };
    Ok([
        gdn::f32_address(&input.history[0])?,
        gdn::f32_address(&input.history[1])?,
        gdn::f32_address(&input.history[2])?,
        gdn::f32_address(&output.history)?,
        gdn::f32_address(&input.state)?,
        gdn::f32_address(&output.state)?,
    ])
}

#[cfg(not(feature = "cuda"))]
fn table_entries(
    _inputs: &StateInputs,
    _outputs: &StateOutputs,
    _layer: usize,
) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    Err(segment_error("device tables need CUDA"))
}

/// An F32 buffer whose every element the step's kernels write.
fn uninit_f32(len: usize, device: &Device) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = device {
        use candle_core::backend::BackendDevice;
        // SAFETY: the DeltaNet kernels (or the verification) write every
        // element before the buffer is read or published.
        let storage = unsafe { cuda.alloc_uninit(&candle_core::Shape::from(len), DType::F32)? };
        return Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            len,
            candle_core::op::BackpropOp::none(),
            false,
        ));
    }
    Tensor::zeros(len, DType::F32, device).map_err(Error::from)
}

fn full_attention(layer: &Qwen36Layer) -> Result<&Qwen36FullAttention> {
    match &layer.mixer {
        Qwen36Mixer::Full(attention) => Ok(attention),
        Qwen36Mixer::Linear(_) => Err(segment_error("planned attention layer is DeltaNet")),
    }
}

fn host(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?)
}

fn segment_error(message: &str) -> Error {
    Error::InferenceError(format!("Qwen3.6 graph decode: {message}"))
}
