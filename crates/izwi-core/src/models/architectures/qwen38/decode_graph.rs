//! Piecewise CUDA graph decode for one Qwen3.8 row.
//!
//! The decode step splits at the 16 full-attention layers. Everything else
//! runs as captured graph segments (17 for the 64-layer trunk):
//! - segment 0: DeltaNet layers 0-2, then layer 3's input norm and q/k/v
//!   projection;
//! - each later segment: the previous attention layer's gate, `o_proj`,
//!   post-attention norm and MLP, the three DeltaNet layers that follow, then
//!   the next attention layer's norm and q/k/v projection;
//! - the last segment: the final attention layer's tail.
//!
//! Attention itself stays eager between segments: its q/k RoPE takes the
//! step's position as a kernel scalar, and the paged KV write and attention
//! read per-step slots and context lengths. The final norm, the LM head and
//! token selection also stay eager.
//!
//! DeltaNet state moves through a device address table (the `gdn::StateTable`
//! layout). Each step allocates one slab of fresh state outputs (the next
//! `[C, 3]` conv history and the next recurrent state of every DeltaNet
//! layer) and writes every layer's addresses into the table with one upload.
//! The captured graphs read the row's current state and write the slab, so
//! there are no state copies, and committed state is never mutated in place.
//! Qwen3.8 keeps its conv history channel-major (`[C, 3]`, oldest first), so
//! the conv runs through q38's own table-addressed kernel and the recurrence
//! through the generic `gdn::recurrent_decode_table`.
//!
//! Phases, in lockstep across the segments:
//! 1. **warm**: run eagerly under the htod-cache guard. On CUDA the warm step
//!    also runs the standard decode on a copy of the row state and compares
//!    the logits, which checks the segmentation and the table-addressed
//!    kernels against the eager path on the live arena.
//! 2. **capture**: capture each segment, then verify it.
//! 3. **verify**: verify again on fresh inputs, which catches values a
//!    capture baked in.
//! 4. **replay**: replay only.
//!
//! A verification runs the segment eagerly into the published state buffers,
//! replays the graph into scratch buffers, and compares outputs and state.
//! The step continues on the eager results. Any failure (a capture, a
//! mismatch, a replay or table error) turns graph decode off for the rest of
//! the process with the reason in the diagnostics, and hands the step to the
//! standard decode: nothing from a failed step is published, and the standard
//! decode rewrites the step's KV slot.
//!
//! Off CUDA there is no capture: every step runs the same segments eagerly
//! with explicit state tensors. Tests use that to check the segmentation
//! against the standard decode path bit for bit.
//!
//! When it applies, the whole-step graph (`step_graph.rs`) is tried first: the
//! entire step, attention and the LM head included, as one captured graph.
//! The segments take any step it does not.
//!
//! Qwen3.8 owns this module. Qwen3.6 keeps its own decode graphs; the two
//! families share only the generic pieces under `kernels/`.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::{DType, Device, Module, Tensor};

use super::{
    qwen38_conv_decode, ConvHistoryStorage, ConvRingState, Qwen38FullAttention, Qwen38Hidden,
    Qwen38LayerRuntimeState, Qwen38LinearAttention, Qwen38Mixer, Qwen38Mlp, Qwen38Projection,
    Qwen38ProjectionGroup, Qwen38TextModel, Qwen38TextRuntimeState,
};
use crate::backends::kv::{KvSlotMap, KvWriteCompletionCollector};
use crate::error::{Error, Result};
use crate::kernels::cuda::gdn;
use crate::kernels::cuda::segment_graph::{DeviceTable, SegmentGraph, SegmentInput};
use crate::kv::{KvDecodeBatchMetadata, KvSlotRef};
use crate::models::architectures::qwen38::fast_path::{
    compare_values, legacy_requested, Qwen38CudaSwitches, Qwen38FastPath, DECODE_GRAPHS_ENV,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

#[path = "step_graph.rs"]
mod step_graph;

/// Logit agreement the warm step demands of the segmented decode against the
/// standard decode. Both run the same kernels except the table-addressed conv
/// (the same arithmetic), so this catches addressing and composition errors,
/// not rounding.
const WARM_CHECK_REL_L2: f64 = 0.02;
const WARM_CHECK_MAX_FRACTION: f32 = 0.05;

/// What a graph-decoded step returns: logits `[1, 1, vocab]` when the LM
/// head ran inside the whole-step graph, otherwise the pre-norm hidden
/// `[1, 1, hidden]` for the caller's final norm and LM head.
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
        /// `[C, 3]` F32, oldest first.
        history: Tensor,
        /// `[1, Hv, Dk, Dv]` F32.
        state: Tensor,
        history_out: Tensor,
        state_out: Tensor,
    },
}

/// One segment: everything between two attention cores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Segment {
    /// Full-attention layer whose tail (gate, `o_proj`, norm, MLP) opens it.
    pub(super) opens_after: Option<usize>,
    /// DeltaNet layers run whole, in order.
    pub(super) linear: Vec<usize>,
    /// Full-attention layer whose norm and q/k/v projection close it.
    pub(super) closes_before: Option<usize>,
}

/// Plan the segments from the layer kinds (`true`: full attention).
pub(super) fn plan_segments(attention: &[bool]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current = Segment {
        opens_after: None,
        linear: Vec::new(),
        closes_before: None,
    };
    for (index, &full) in attention.iter().enumerate() {
        if full {
            current.closes_before = Some(index);
            segments.push(std::mem::replace(
                &mut current,
                Segment {
                    opens_after: Some(index),
                    linear: Vec::new(),
                    closes_before: None,
                },
            ));
        } else {
            current.linear.push(index);
        }
    }
    segments.push(current);
    segments
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Phase {
    Warm,
    Capture,
    Verify,
    Replay,
    Disabled(String),
}

impl Phase {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Capture => "capture",
            Self::Verify => "verify",
            Self::Replay => "replay",
            Self::Disabled(_) => "disabled",
        }
    }

    /// The phase after a step that ended in `self`.
    fn advance(self, capture: bool) -> Self {
        match self {
            Self::Warm if capture => Self::Capture,
            Self::Warm => Self::Warm,
            Self::Capture => Self::Verify,
            Self::Verify => Self::Replay,
            other => other,
        }
    }
}

/// The step fence: recorded after each step's queued work, waited on by the
/// next step's stream (a GPU-side wait). Consecutive steps may run on
/// different threads, whose per-thread default streams are not ordered, while
/// they share the table and the graphs' buffers. Raw driver calls only: the
/// wrapper's `bind_to_thread` first returns (and clears) any error another
/// component latched on the context, which would turn graph decode off for an
/// error that is not its own, and could skip the teardown wait.
#[derive(Default)]
pub(super) struct StepFence {
    #[cfg(feature = "cuda")]
    event: Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
}

impl StepFence {
    /// Order this step's work after the previous step's.
    pub(super) fn order_after_previous(&self, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let (Some(event), Device::Cuda(cuda)) = (&self.event, device) {
            use candle_core::cuda_backend::cudarc::driver::{result, sys};
            let stream = cuda.cuda_stream();
            // SAFETY: the event and the stream belong to this device's
            // context, which outlives both handles.
            return unsafe {
                result::ctx::set_current(stream.context().cu_ctx()).and_then(|()| {
                    result::stream::wait_event(
                        stream.cu_stream(),
                        event.cu_event(),
                        sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                    )
                })
            }
            .map_err(|error| segment_error(&format!("step fence wait: {error}")));
        }
        let _ = device;
        Ok(())
    }

    /// Mark the end of this step's queued work for the next step.
    pub(super) fn mark_step_end(&mut self, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            use candle_core::cuda_backend::cudarc::driver::result;
            let stream = cuda.cuda_stream();
            if self.event.is_none() {
                self.event = Some(
                    stream
                        .context()
                        .new_event(None)
                        .map_err(|error| segment_error(&format!("step fence: {error}")))?,
                );
            }
            if let Some(event) = &self.event {
                // SAFETY: as in `order_after_previous`.
                unsafe {
                    result::ctx::set_current(stream.context().cu_ctx())
                        .and_then(|()| result::event::record(event.cu_event(), stream.cu_stream()))
                }
                .map_err(|error| segment_error(&format!("step fence record: {error}")))?;
            }
        }
        let _ = device;
        Ok(())
    }

    /// Block until the last marked step finished. `false` when that cannot be
    /// proven (the caller then leaks what the step may still touch).
    #[cfg(feature = "cuda")]
    fn wait_for_last_step(&self) -> bool {
        use candle_core::cuda_backend::cudarc::driver::result;
        let Some(event) = &self.event else {
            return true;
        };
        // SAFETY: the event is live and its context outlives it.
        unsafe {
            result::ctx::set_current(event.context().cu_ctx())
                .and_then(|()| result::event::synchronize(event.cu_event()))
                .is_ok()
        }
    }
}

struct Runtime {
    phase: Phase,
    graphs: Vec<SegmentGraph>,
    table: Option<DeviceTable>,
    /// The whole-step graph, tried before the segments when planned.
    whole: step_graph::WholeStepRuntime,
    fence: StepFence,
}

impl Runtime {
    /// Turn graph decode off: the segments, and the whole-step graph too.
    fn disable(&mut self, reason: String) {
        tracing::warn!(%reason, "Qwen3.8 CUDA graph decode disabled; decoding eagerly");
        for graph in self.graphs.iter_mut() {
            graph.reset();
        }
        if !matches!(self.whole.phase, Phase::Disabled(_)) {
            self.whole.disable(reason.clone());
        }
        self.phase = Phase::Disabled(reason);
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // Wait for the last step, which may have run on another thread,
        // before the graphs and the table it used are destroyed. A failed
        // wait cannot prove it finished: leak everything it may touch.
        // (Off CUDA nothing is in flight and the wait always succeeds.)
        #[cfg(feature = "cuda")]
        if !self.fence.wait_for_last_step() {
            tracing::warn!(
                "Qwen3.8 graph decode teardown could not wait for the last step; leaking its graphs"
            );
            for graph in std::mem::take(&mut self.graphs) {
                std::mem::forget(graph);
            }
            std::mem::forget(self.table.take());
            self.whole.leak();
        }
    }
}

/// Graph decode for one model (see the module docs).
pub(super) struct Qwen38DecodeGraphs {
    segments: Vec<Segment>,
    /// Per layer: the DeltaNet ordinal (table slot `ordinal * TABLE_ENTRIES`).
    linear_ordinal: Vec<Option<usize>>,
    linear_count: usize,
    attention_count: usize,
    /// Capture graphs (CUDA). Otherwise every step runs the segments eagerly.
    capture: bool,
    /// The whole-step graph, or why it does not apply.
    whole: std::result::Result<step_graph::WholeStepPlan, String>,
    runtime: Mutex<Runtime>,
    warmups: AtomicU64,
    warm_checks: AtomicU64,
    captures: AtomicU64,
    replays: AtomicU64,
    verified_segments: AtomicU64,
    eager_steps: AtomicU64,
}

impl Qwen38DecodeGraphs {
    /// Whether graph decode applies to `model`, and why not. `allow_eager`
    /// admits non-CUDA devices with eager segments (tests); in production the
    /// residency policy must also have planned (and admission reserved) graph
    /// decode.
    pub(super) fn resolve(
        model: &Qwen38TextModel,
        switches: &Qwen38CudaSwitches,
        allow_eager: bool,
        whole_step: bool,
    ) -> (Option<Self>, Qwen38FastPath) {
        let device = &model.device;
        let refuse = |reason: String| (None, Qwen38FastPath::legacy(reason));
        if legacy_requested(DECODE_GRAPHS_ENV) {
            return refuse(format!("{DECODE_GRAPHS_ENV}=legacy"));
        }
        if let Some(reason) = switches.decode_graphs_off {
            return refuse(reason.to_string());
        }
        if !device.is_cuda() && !allow_eager {
            return refuse("CUDA graph decode runs on CUDA only".into());
        }
        if !allow_eager && !model.residency.graph_decode_planned() {
            return refuse(model.residency.graph_decode_refusal());
        }
        if model.finite_diagnostics_enabled {
            return refuse("CUDA graph decode is off while finite diagnostics read values".into());
        }
        let mut linear_ordinal = Vec::with_capacity(model.layers.len());
        let mut attention = Vec::with_capacity(model.layers.len());
        let mut linear_count = 0usize;
        for layer in &model.layers {
            if !layer.mlp.capturable() {
                return refuse(CAPTURE_REFUSAL.into());
            }
            match &layer.mixer {
                Qwen38Mixer::Linear(mixer) => {
                    if mixer.fused_gdn.is_none() {
                        return refuse(
                            "CUDA graph decode needs the fused DeltaNet decode on every layer"
                                .into(),
                        );
                    }
                    if !mixer.capturable() {
                        return refuse(CAPTURE_REFUSAL.into());
                    }
                    linear_ordinal.push(Some(linear_count));
                    linear_count += 1;
                    attention.push(false);
                }
                Qwen38Mixer::Full(full) => {
                    if !full.capturable() {
                        return refuse(CAPTURE_REFUSAL.into());
                    }
                    linear_ordinal.push(None);
                    attention.push(true);
                }
            }
        }
        let attention_count = attention.iter().filter(|full| **full).count();
        if attention_count == 0 {
            return refuse("the model has no full-attention layer".into());
        }
        let segments = plan_segments(&attention);
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
                attention_count,
                capture: device.is_cuda(),
                whole,
                runtime: Mutex::new(Runtime {
                    phase: Phase::Warm,
                    graphs,
                    table: None,
                    whole: step_graph::WholeStepRuntime::new(),
                    fence: StepFence::default(),
                }),
                warmups: AtomicU64::new(0),
                warm_checks: AtomicU64::new(0),
                captures: AtomicU64::new(0),
                replays: AtomicU64::new(0),
                verified_segments: AtomicU64::new(0),
                eager_steps: AtomicU64::new(0),
            }),
            Qwen38FastPath::Fused,
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
            "warm_checks": self.warm_checks.load(Relaxed),
            "captures": self.captures.load(Relaxed),
            "replays": self.replays.load(Relaxed),
            "verified_segments": self.verified_segments.load(Relaxed),
            "eager_steps": self.eager_steps.load(Relaxed),
            "whole_step": whole_step,
        })
    }

    /// Run one decode row through the whole-step graph when it applies, else
    /// through the segments (`embedded` `[1, 1, hidden]` is `token`'s
    /// embedding). `None` hands the step to the standard decode: graph decode
    /// is off or busy, the row's state is not in the fused form, or the step
    /// failed (which turns graph decode off). Nothing is published then.
    /// `reference` computes the standard decode's logits on a copy of the row
    /// state for the warm checks.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        model: &Qwen38TextModel,
        token: u32,
        embedded: &Tensor,
        position: [usize; 3],
        slot: KvSlotRef,
        state: &mut Qwen38TextRuntimeState,
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Option<GraphDecodeOutput> {
        let Ok(mut runtime) = self.runtime.try_lock() else {
            self.eager_steps.fetch_add(1, Relaxed);
            return None;
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
            return None;
        }
        let step = self.step(
            &mut runtime,
            whole,
            model,
            token,
            embedded,
            position,
            slot,
            state,
            cache,
            slots,
            metadata,
            completions,
            reference,
        );
        match step {
            Ok(Some(output)) => Some(output),
            Ok(None) => {
                self.eager_steps.fetch_add(1, Relaxed);
                None
            }
            Err(error) => {
                runtime.disable(format!("graph decode step failed: {error}"));
                self.eager_steps.fetch_add(1, Relaxed);
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        runtime: &mut Runtime,
        whole: Option<&step_graph::WholeStepPlan>,
        model: &Qwen38TextModel,
        token: u32,
        embedded: &Tensor,
        position: [usize; 3],
        slot: KvSlotRef,
        state: &mut Qwen38TextRuntimeState,
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<Option<GraphDecodeOutput>> {
        let device = model.device.clone();
        for (layer, layer_state) in model.layers.iter().zip(state.layers.iter_mut()) {
            layer.ensure_state_initialized(layer_state, &device)?;
        }
        let Some(inputs) = StateInputs::gather(model, state)? else {
            return Ok(None);
        };
        runtime.fence.order_after_previous(&device)?;
        if let Some(plan) = whole {
            let request = step_graph::StepRequest {
                token: Some(token),
                position: position[0],
                slot,
                cache,
                slots,
                metadata,
            };
            let done = step_graph::run(
                plan,
                &mut runtime.whole,
                self,
                model,
                &request,
                &inputs,
                completions,
                reference,
            );
            // Whatever the outcome, the step may have queued work.
            let marked = runtime.fence.mark_step_end(&device);
            if let Some(done) = done? {
                marked?;
                done.published.publish(model, state)?;
                return Ok(Some(if plan.head_in_graph() {
                    GraphDecodeOutput::Logits(done.output)
                } else {
                    GraphDecodeOutput::Hidden(done.output)
                }));
            }
            // The whole-step graph turned itself off: the segments take the
            // step.
            marked?;
            if matches!(runtime.phase, Phase::Disabled(_)) {
                return Ok(None);
            }
        }
        // The warm check's standard decode runs first: it writes the step's
        // KV slot, which the segments then rewrite with the same values.
        let expected = if self.capture && runtime.phase == Phase::Warm {
            Some(reference()?)
        } else {
            None
        };
        let step = self.run_step(
            runtime,
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
        let marked = runtime.fence.mark_step_end(&device);
        let (hidden, phase, published) = step?;
        marked?;
        if let Some(expected) = expected {
            let logits = model.project_target_hidden_span(&hidden)?;
            compare_values(
                "segmented decode against the standard decode",
                &host(&logits)?,
                &host(&expected)?,
                WARM_CHECK_REL_L2,
                WARM_CHECK_MAX_FRACTION,
            )?;
            self.warm_checks.fetch_add(1, Relaxed);
        }
        match phase.advance(self.capture) {
            Phase::Disabled(reason) => runtime.disable(reason),
            next => runtime.phase = next,
        }
        published.publish(model, state)?;
        Ok(Some(GraphDecodeOutput::Hidden(hidden)))
    }

    /// The segment loop for one step. Returns the pre-norm hidden, the phase
    /// the step ended in, and the state outputs to publish.
    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &self,
        runtime: &mut Runtime,
        model: &Qwen38TextModel,
        device: &Device,
        inputs: &StateInputs,
        embedded: &Tensor,
        position: [usize; 3],
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<(Tensor, Phase, StateOutputs)> {
        let published = StateOutputs::allocate(model, device, None)?;
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
                    let attended = full_attention(model, layer_index)?.decode_attend(
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
        if physical_layer != self.attention_count {
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
        model: &Qwen38TextModel,
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

    /// Every DeltaNet layer's table entries, `inputs` to `outputs`, in table
    /// order (CUDA).
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
        model: &Qwen38TextModel,
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
        model: &Qwen38TextModel,
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

/// The reason graph decode refuses a projection form it cannot capture.
const CAPTURE_REFUSAL: &str =
    "CUDA graph decode needs capturable projections (Candle Q8_0 QMatMul keeps a hidden device scratch a capture cannot retain)";

impl Qwen38Projection {
    /// Whether a captured segment may run this projection: everything but
    /// Candle's QMatMul.
    fn capturable(&self) -> bool {
        !matches!(self, Self::Quantized(_))
    }
}

impl Qwen38ProjectionGroup {
    fn capturable(&self) -> bool {
        match self {
            Self::Separate(parts) => parts.iter().all(Qwen38Projection::capturable),
            Self::Packed { projection, .. } => projection.capturable(),
        }
    }
}

impl Qwen38Mlp {
    fn capturable(&self) -> bool {
        self.gate_up.capturable() && self.down.capturable()
    }
}

impl Qwen38FullAttention {
    fn capturable(&self) -> bool {
        self.qkv_proj.capturable() && self.o_proj.capturable()
    }
}

impl Qwen38LinearAttention {
    fn capturable(&self) -> bool {
        self.qkv_z_proj.capturable()
            && self.out_proj.capturable()
            && (self.alpha_beta_rows.is_some() || self.alpha_beta_proj.capturable())
    }

    /// One token through the fused DeltaNet kernels with explicit state IO:
    /// the graph-segment form of the fused decode (`fused_decode_row`), with
    /// the same projections, kernels and casts.
    fn forward_segment(&self, hidden_states: &Tensor, io: &LinearStateIo) -> Result<Tensor> {
        let spec = self
            .fused_gdn
            .as_ref()
            .ok_or_else(|| segment_error("graph decode needs the fused DeltaNet decode"))?;
        let (mixed_qkv, z, alpha, beta) = self.decode_projections(hidden_states)?;
        let value_width = self.num_v_heads * self.head_v_dim;
        let y = match io {
            #[cfg(feature = "cuda")]
            LinearStateIo::Table(table) => {
                let conv = crate::kernels::cuda::qwen38_causal_conv_decode_table(
                    &mixed_qkv,
                    &self.conv_kernel,
                    *table,
                )?;
                gdn::recurrent_decode_table(
                    &conv,
                    &z,
                    &beta,
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
                let (conv, next_history) = qwen38_conv_decode(&mixed_qkv, &self.conv_kernel, history)?;
                let (y, next) = gdn::recurrent_decode(
                    &conv,
                    &z,
                    &beta,
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
            .forward(
                &y.reshape((1, 1, value_width))?
                    .to_dtype(hidden_states.dtype())?,
            )
            .map_err(Error::from)
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
/// layer, `[hidden]` for the last one. The op sequence is the standard
/// decode's (`forward_physical_decode_batch` for one row).
fn run_segment(
    model: &Qwen38TextModel,
    segment: &Segment,
    ios: &[Option<LinearStateIo>],
    inputs: &[Tensor],
) -> Result<Vec<Tensor>> {
    let layers = &model.layers;
    let mut hidden = match (segment.opens_after, inputs) {
        (None, [embedded]) => Qwen38Hidden::new(embedded.clone()),
        (Some(index), [residual, q_proj, attended]) => {
            let mixed = full_attention(model, index)?.decode_output(attended, q_proj)?;
            layers[index].post_mixer(residual, &mixed)?
        }
        _ => return Err(segment_error("segment inputs do not match its plan")),
    };
    for &index in &segment.linear {
        let layer = &layers[index];
        let (residual, normalized) = hidden.normalized(&layer.attn_norm)?;
        let Qwen38Mixer::Linear(mixer) = &layer.mixer else {
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
            let (q_proj, keys, values) = full_attention(model, index)?.decode_projections(&normalized)?;
            Ok(vec![residual, q_proj, keys, values])
        }
        None => Ok(vec![hidden.materialize()?]),
    }
}

/// One DeltaNet layer's current state, contiguous and F32.
struct LayerStateIn {
    /// `[C, 3]`, oldest first.
    history: Tensor,
    state: Tensor,
}

pub(super) struct StateInputs {
    layers: Vec<Option<LayerStateIn>>,
}

impl StateInputs {
    /// The row's current DeltaNet state, or `None` when some layer's state is
    /// not in the fused-decode form (a 3-slot F32 history, an F32 state).
    pub(super) fn gather(
        model: &Qwen38TextModel,
        state: &Qwen38TextRuntimeState,
    ) -> Result<Option<Self>> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for (layer, layer_state) in model.layers.iter().zip(&state.layers) {
            let mixer = match &layer.mixer {
                Qwen38Mixer::Linear(mixer) => mixer,
                Qwen38Mixer::Full(_) => {
                    layers.push(None);
                    continue;
                }
            };
            let Qwen38LayerRuntimeState::Linear {
                conv_state: Some(ring),
                recurrent_state: Some(recurrent),
            } = layer_state
            else {
                return Ok(None);
            };
            if ring.history_len().ok() != Some(gdn::CONV_TAPS - 1)
                || !ring.is_f32()
                || recurrent.dtype() != DType::F32
                || recurrent.elem_count()
                    != mixer.num_v_heads * mixer.head_k_dim * mixer.head_v_dim
            {
                return Ok(None);
            }
            let history = ring.contiguous_history()?.contiguous()?;
            if history.dims() != [mixer.conv_dim, gdn::CONV_TAPS - 1] {
                return Ok(None);
            }
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
    /// `[C, 3]`.
    history: Tensor,
    /// `[1, Hv, Dk, Dv]`.
    state: Tensor,
}

pub(super) struct StateOutputs {
    layers: Vec<Option<LayerStateOut>>,
}

impl StateOutputs {
    /// One slab for the next history `[C, 3]` and state `[1, Hv, Dk, Dv]`
    /// (F32) of every DeltaNet layer, or of `only` those layers. The kernels
    /// write every element.
    pub(super) fn allocate(
        model: &Qwen38TextModel,
        device: &Device,
        only: Option<&[usize]>,
    ) -> Result<Self> {
        let sizes = model
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| match &layer.mixer {
                Qwen38Mixer::Linear(mixer) if only.is_none_or(|only| only.contains(&index)) => {
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
                        .reshape((conv_dim, gdn::CONV_TAPS - 1))?;
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

    /// Hand the written state to the row: each history becomes the
    /// channel-major contiguous history the fused decode also produces.
    pub(super) fn publish(
        self,
        model: &Qwen38TextModel,
        state: &mut Qwen38TextRuntimeState,
    ) -> Result<()> {
        for ((layer, out), layer_state) in model
            .layers
            .iter()
            .zip(self.layers)
            .zip(state.layers.iter_mut())
        {
            let (Qwen38Mixer::Linear(_), Some(out)) = (&layer.mixer, out) else {
                continue;
            };
            let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer_state
            else {
                return Err(segment_error("DeltaNet layer lost its runtime state"));
            };
            *conv_state = Some(ConvRingState {
                storage: ConvHistoryStorage::Contiguous(out.history),
            });
            *recurrent_state = Some(out.state);
        }
        Ok(())
    }
}

impl StateOutputs {
    /// Compare this (graph-written) state with `eager`'s, layer by layer:
    /// the conv history exactly (to 1e-6), the recurrent state to rounding.
    /// Layers either side lacks are skipped.
    fn compare_with(&self, eager: &StateOutputs, label: &str) -> Result<()> {
        for (layer, (graph_state, eager_state)) in self.layers.iter().zip(&eager.layers).enumerate()
        {
            let (Some(graph_state), Some(eager_state)) = (graph_state, eager_state) else {
                continue;
            };
            compare_values(
                &format!("{label} layer {layer} conv history"),
                &host(&graph_state.history)?,
                &host(&eager_state.history)?,
                1e-6,
                1e-6,
            )?;
            compare_values(
                &format!("{label} layer {layer} recurrent state"),
                &host(&graph_state.state)?,
                &host(&eager_state.state)?,
                0.01,
                0.02,
            )?;
        }
        Ok(())
    }
}

/// Table entries for `layer` in the `gdn::StateTable` layout: the `[C, 3]`
/// history (entry 0; entries 1 and 2 repeat it, unused by q38's conv), the
/// next history, the recurrent state in, the recurrent state out.
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
    let history = gdn::f32_address(&input.history)?;
    Ok([
        history,
        history,
        history,
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

fn full_attention(model: &Qwen38TextModel, layer: usize) -> Result<&Qwen38FullAttention> {
    match &model.layers[layer].mixer {
        Qwen38Mixer::Full(attention) => Ok(attention),
        Qwen38Mixer::Linear(_) => Err(segment_error("planned attention layer is DeltaNet")),
    }
}

pub(super) fn host(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?)
}

pub(super) fn segment_error(message: &str) -> Error {
    Error::InferenceError(format!("Qwen3.8 graph decode: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 27B trunk: 48 DeltaNet layers and a full-attention layer every
    /// fourth, so 17 segments with three DeltaNet layers each but the last.
    #[test]
    fn segments_split_the_27b_trunk_at_its_attention_layers() {
        let attention = (0..64).map(|layer| layer % 4 == 3).collect::<Vec<_>>();
        let segments = plan_segments(&attention);
        assert_eq!(segments.len(), 17);
        assert_eq!(
            segments[0],
            Segment {
                opens_after: None,
                linear: vec![0, 1, 2],
                closes_before: Some(3),
            }
        );
        for (index, segment) in segments.iter().enumerate().skip(1).take(15) {
            let opens = 4 * index - 1;
            assert_eq!(
                segment,
                &Segment {
                    opens_after: Some(opens),
                    linear: vec![opens + 1, opens + 2, opens + 3],
                    closes_before: Some(opens + 4),
                },
                "segment {index}"
            );
        }
        assert_eq!(
            segments[16],
            Segment {
                opens_after: Some(63),
                linear: Vec::new(),
                closes_before: None,
            }
        );
        // Every layer is covered once: DeltaNet layers inside a segment,
        // attention layers as one segment's close and the next one's open.
        let mut linear = segments
            .iter()
            .flat_map(|segment| segment.linear.iter().copied())
            .collect::<Vec<_>>();
        linear.sort_unstable();
        assert_eq!(
            linear,
            (0..64).filter(|layer| layer % 4 != 3).collect::<Vec<_>>()
        );
        for pair in segments.windows(2) {
            assert_eq!(pair[0].closes_before, pair[1].opens_after);
        }
        // A trunk that starts or ends with attention gets empty runs.
        let edges = plan_segments(&[true, false, true]);
        assert_eq!(edges.len(), 3);
        assert_eq!(edges[0].linear, Vec::<usize>::new());
        assert_eq!(edges[2].closes_before, None);
    }

    #[test]
    fn phases_advance_to_replay_only_with_capture() {
        assert_eq!(Phase::Warm.advance(false), Phase::Warm);
        assert_eq!(Phase::Warm.advance(true), Phase::Capture);
        assert_eq!(Phase::Capture.advance(true), Phase::Verify);
        assert_eq!(Phase::Verify.advance(true), Phase::Replay);
        assert_eq!(Phase::Replay.advance(true), Phase::Replay);
        let off = Phase::Disabled("x".into());
        assert_eq!(off.clone().advance(true), off);
    }
}
