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
use crate::kv::{KvDecodeBatchMetadata, KvSlotRef};
use crate::models::architectures::qwen36moe::fast_path::{
    compare_values, legacy_requested, Qwen36CudaSwitches, Qwen36FusedPath,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

#[path = "step_graph.rs"]
mod step_graph;

/// Environment switch for graph decode (`legacy`, `off`, `0`, `false`).
pub(super) const GRAPHS_ENV: &str = "IZWI_QWEN36_CUDA_GRAPHS";

/// What a graph-decoded step returns: logits when the LM head ran inside the
/// whole-step graph, otherwise the pre-norm hidden for the caller's head.
pub(super) enum GraphDecodeOutput {
    Hidden(Tensor),
    Logits(Tensor),
}

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
    /// The whole-step graph, tried before the segments when planned.
    whole: step_graph::WholeStepRuntime,
    /// Recorded after each step's queued work. The next step's stream waits
    /// on it: consecutive steps may run on different threads, whose
    /// per-thread streams are not ordered, while they share the table and
    /// the graphs' buffers.
    #[cfg(feature = "cuda")]
    fence: Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
}

impl Runtime {
    /// Order this step's work after the previous step's (a GPU-side wait).
    fn order_after_previous(&self, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let (Some(fence), Device::Cuda(cuda)) = (&self.fence, device) {
            return cuda
                .cuda_stream()
                .wait(fence)
                .map_err(|error| segment_error(&format!("step fence wait: {error}")));
        }
        let _ = device;
        Ok(())
    }

    /// Mark the end of this step's queued work for the next step.
    fn mark_step_end(&mut self, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            let stream = cuda.cuda_stream();
            if self.fence.is_none() {
                self.fence = Some(
                    stream
                        .context()
                        .new_event(None)
                        .map_err(|error| segment_error(&format!("step fence: {error}")))?,
                );
            }
            if let Some(fence) = &self.fence {
                fence
                    .record(&stream)
                    .map_err(|error| segment_error(&format!("step fence record: {error}")))?;
            }
        }
        let _ = device;
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // Wait for the last step, which may have run on another thread,
        // before the graphs and the table it used are destroyed.
        #[cfg(feature = "cuda")]
        if let Some(fence) = &self.fence {
            let _ = fence.synchronize();
        }
    }
}

/// A failed step: `Graph` errors come from the graph machinery (table,
/// replay) and disable graph decode; `Other` errors are the step's own.
enum StepError {
    Graph(Error),
    Other(Error),
}

impl From<Error> for StepError {
    fn from(error: Error) -> Self {
        Self::Other(error)
    }
}

impl From<candle_core::Error> for StepError {
    fn from(error: candle_core::Error) -> Self {
        Self::Other(error.into())
    }
}

/// Graph decode for one model (see the module docs).
pub(super) struct Qwen36DecodeGraphs {
    segments: Vec<Segment>,
    /// Per layer: the DeltaNet ordinal (table slot `ordinal * TABLE_ENTRIES`).
    linear_ordinal: Vec<Option<usize>>,
    linear_count: usize,
    /// Capture graphs (CUDA). Otherwise every step runs the segments eagerly.
    capture: bool,
    /// The whole-step graph, or why it does not apply.
    whole: std::result::Result<step_graph::WholeStepPlan, String>,
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
        whole_step: bool,
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
        let whole = if whole_step {
            step_graph::WholeStepPlan::resolve(model, device.is_cuda())
        } else {
            Err("not requested".into())
        };
        (
            Some(Self {
                segments,
                linear_ordinal,
                linear_count,
                capture: device.is_cuda(),
                whole,
                runtime: Mutex::new(Runtime {
                    phase: Phase::Warm,
                    graphs,
                    table: None,
                    whole: step_graph::WholeStepRuntime::new(),
                    #[cfg(feature = "cuda")]
                    fence: None,
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
        let (phase, whole_phase) = {
            let runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (runtime.phase.clone(), runtime.whole.phase.clone())
        };
        let whole_step = match &self.whole {
            Ok(plan) => plan.summary(&whole_phase),
            Err(reason) => serde_json::json!({ "phase": "off", "disabled_reason": reason }),
        };
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
            "whole_step": whole_step,
        })
    }

    /// Run one decode row through the whole-step graph when it applies, else
    /// through the segments (`embedded` `[1, 1, hidden]` is `token`'s
    /// embedding). Returns `None` when graph decode does not take the step
    /// (disabled, busy, or a state it cannot address); the caller then runs
    /// the eager path. `reference` computes the standard decode's logits for
    /// the whole-step graph's warm check.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        model: &Qwen36TextModel,
        token: u32,
        embedded: &Tensor,
        position: [usize; 3],
        slot: KvSlotRef,
        state: &mut Qwen36TextRuntimeState,
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<Option<GraphDecodeOutput>> {
        let Ok(mut runtime) = self.runtime.try_lock() else {
            self.eager_steps.fetch_add(1, Relaxed);
            return Ok(None);
        };
        let whole = match &self.whole {
            Ok(plan)
                if !matches!(runtime.whole.phase, Phase::Disabled(_))
                    && position[0] == position[1]
                    && position[0] == position[2] =>
            {
                Some(plan)
            }
            _ => None,
        };
        if whole.is_none() && matches!(runtime.phase, Phase::Disabled(_)) {
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
        runtime.order_after_previous(&device)?;
        if let Some(plan) = whole {
            let request = step_graph::StepRequest {
                token: Some(token),
                position: position[0],
                slot,
                cache,
                slots,
                metadata,
            };
            match step_graph::run(
                plan,
                &mut runtime.whole,
                self,
                model,
                &request,
                &inputs,
                completions,
                reference,
            ) {
                Ok(Some(done)) => {
                    runtime.mark_step_end(&device)?;
                    drop(runtime);
                    done.published.publish(model, state)?;
                    return Ok(Some(if plan.head_in_graph() {
                        GraphDecodeOutput::Logits(done.output)
                    } else {
                        GraphDecodeOutput::Hidden(done.output)
                    }));
                }
                Ok(None) => {}
                Err(StepError::Graph(error)) => runtime.whole.disable(error.to_string()),
                Err(StepError::Other(error)) => {
                    let _ = runtime.mark_step_end(&device);
                    return Err(error);
                }
            }
            if matches!(runtime.phase, Phase::Disabled(_)) {
                self.eager_steps.fetch_add(1, Relaxed);
                return Ok(None);
            }
        }
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
        let marked = runtime.mark_step_end(&device);
        let (hidden, phase, published) = match step {
            Ok(done) => done,
            Err(StepError::Graph(error)) => {
                // The graph machinery failed: decode eagerly from now on.
                runtime.phase = Phase::Disabled(format!("graph decode step failed: {error}"));
                for graph in runtime.graphs.iter_mut() {
                    graph.reset();
                }
                tracing::warn!(%error, "Qwen3.6 CUDA graph decode disabled; decoding eagerly");
                return Err(error);
            }
            Err(StepError::Other(error)) => return Err(error),
        };
        marked?;
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
        Ok(Some(GraphDecodeOutput::Hidden(hidden)))
    }

    /// Whether a step can go through the whole-step graph with a
    /// device-resident input token: the graph is replaying (or, off CUDA in
    /// tests, the eager body runs every step).
    pub(super) fn whole_step_replaying(&self) -> bool {
        let Ok(plan) = &self.whole else {
            return false;
        };
        self.runtime
            .lock()
            .map(|runtime| plan.takes_device_tokens(&runtime.whole.phase))
            .unwrap_or(false)
    }

    /// The whole-step graph's token slot (see `step_graph::token_slot`).
    pub(super) fn whole_step_token_slot(&self) -> Option<Result<Tensor>> {
        let runtime = self.runtime.lock().ok()?;
        step_graph::token_slot(&runtime.whole)
    }

    /// Queue one step through the replaying whole-step graph, its input token
    /// already in the step buffer, without falling back: `None` means the
    /// whole-step graph did not take the step (busy, not replaying, or it
    /// failed and turned itself off), and nothing was published.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_whole(
        &self,
        model: &Qwen36TextModel,
        token: Option<u32>,
        position: usize,
        slot: KvSlotRef,
        state: &mut Qwen36TextRuntimeState,
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<Option<GraphDecodeOutput>> {
        let Ok(plan) = &self.whole else {
            return Ok(None);
        };
        let Ok(mut runtime) = self.runtime.try_lock() else {
            return Ok(None);
        };
        if !plan.takes_device_tokens(&runtime.whole.phase) {
            return Ok(None);
        }
        let device = model.device.clone();
        let Some(inputs) = StateInputs::gather(model, state)? else {
            return Ok(None);
        };
        runtime.order_after_previous(&device)?;
        let request = step_graph::StepRequest {
            token,
            position,
            slot,
            cache,
            slots,
            metadata,
        };
        let mut no_reference =
            || -> Result<Tensor> { Err(segment_error("a replaying step has no warm check")) };
        let step = step_graph::run(
            plan,
            &mut runtime.whole,
            self,
            model,
            &request,
            &inputs,
            completions,
            &mut no_reference,
        );
        let marked = runtime.mark_step_end(&device);
        let done = match step {
            Ok(Some(done)) => done,
            Ok(None) => return Ok(None),
            Err(StepError::Graph(error)) => {
                runtime.whole.disable(error.to_string());
                return Ok(None);
            }
            Err(StepError::Other(error)) => return Err(error),
        };
        marked?;
        drop(runtime);
        done.published.publish(model, state)?;
        if token.is_none() {
            plan.pipelined
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(Some(if plan.head_in_graph() {
            GraphDecodeOutput::Logits(done.output)
        } else {
            GraphDecodeOutput::Hidden(done.output)
        }))
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
    ) -> std::result::Result<(Tensor, Phase, StateOutputs), StepError> {
        let published = StateOutputs::allocate(model, device, None)?;
        let ios = self
            .state_io(runtime, model, inputs, &published, device)
            .map_err(StepError::Graph)?;
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
                    let outputs = runtime.graphs[index]
                        .replay(&replay_inputs)
                        .map_err(|error| StepError::Graph(error.into()))?;
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
            return Err(segment_error("graph decode did not cover every attention layer").into());
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
        if device.is_cuda() {
            let values = self.table_values(inputs, outputs)?;
            if runtime.table.as_ref().map(DeviceTable::entries) != Some(values.len()) {
                runtime.table = Some(DeviceTable::new(device, values.len())?);
            }
            let table = runtime.table.as_mut().expect("table allocated");
            table.write(0, &values)?;
            return self.table_io(table.address());
        }
        self.direct_io(model, inputs, outputs)
    }

    /// Every DeltaNet layer's six table entries, `inputs` to `outputs`, in
    /// table order (CUDA).
    fn table_values(&self, inputs: &StateInputs, outputs: &StateOutputs) -> Result<Vec<u64>> {
        let mut values = Vec::with_capacity(self.linear_count * gdn::TABLE_ENTRIES);
        for (layer, ordinal) in self.linear_ordinal.iter().enumerate() {
            if ordinal.is_some() {
                values.extend(table_entries(inputs, outputs, layer)?);
            }
        }
        Ok(values)
    }

    /// Table-addressed IO for every DeltaNet layer, the table at `address`.
    fn table_io(&self, address: u64) -> Result<Vec<Option<LinearStateIo>>> {
        #[cfg(feature = "cuda")]
        {
            Ok(self
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
                .collect())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = address;
            Err(segment_error("device tables need CUDA"))
        }
    }

    /// Explicit-tensor IO for every DeltaNet layer (devices without tables).
    fn direct_io(
        &self,
        model: &Qwen36TextModel,
        inputs: &StateInputs,
        outputs: &StateOutputs,
    ) -> Result<Vec<Option<LinearStateIo>>> {
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
            let scratch = StateOutputs::allocate(model, device, Some(&segment.linear))?;
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
    /// One slab for the next history `[3, conv_dim]` and state
    /// `[1, Hv, Dk, Dv]` (F32) of every DeltaNet layer, or of `only` those
    /// layers. The kernels write every element.
    fn allocate(model: &Qwen36TextModel, device: &Device, only: Option<&[usize]>) -> Result<Self> {
        let sizes = model
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| match &layer.mixer {
                Qwen36Mixer::Linear(mixer) if only.is_none_or(|only| only.contains(&index)) => {
                    Some((
                        mixer.conv_dim,
                        (mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim),
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let total = sizes
            .iter()
            .flatten()
            .map(|(conv_dim, (hv, dk, dv))| (gdn::CONV_TAPS - 1) * conv_dim + hv * dk * dv)
            .sum::<usize>();
        if total == 0 {
            return Ok(Self {
                layers: sizes.iter().map(|_| None).collect(),
            });
        }
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
