//! Whole-step CUDA graph decode for one Qwen3.5 row (opt-in).
//!
//! A decode step on CUDA is ~600-900 small launches; eager Candle pays host
//! time for each, so the step is bound by the host, not the GPU. Here the
//! whole step is captured once and replayed: the token embedding, every
//! layer, the final norm and the LM head. Every per-step value lives in a
//! device **step buffer** (U32): the token id, the position, the flat KV
//! slot and the decode metadata `[context, first-page offset, block table]`.
//! - Projections use the izwi-owned K-quant MMVQ (`kernels/cuda/kquant.rs`);
//!   Candle's `QMatMul` keeps a global scratch buffer and cannot be captured.
//! - DeltaNet layers run the fused decode kernels through a device address
//!   table (`gdn::StateTable`). Each step allocates one slab of fresh state
//!   outputs; committed state is never mutated in place.
//! - Attention runs `rope::qk_norm_rope_kv` (q/k norm, RoPE and the K/V slot
//!   write) and `split_decode::paged_decode_adaptive`, which read the step
//!   buffer. Both take 16-bit activations, so q/k/v are cast to the KV
//!   storage dtype (F16 on CUDA) inside the graph.
//!
//! Phases: warm (eager, checked against the standard decode on a copy of the
//! row state), capture + verify, verify again on fresh inputs (catches
//! values a capture baked in), then replay. The step always continues on the
//! eager result until replay. Any failure turns the graph off for the
//! process and the step runs the standard decode.
//!
//! Off CUDA there is no capture: with `allow_eager` (tests) every step runs
//! the same body eagerly with the arena's own attention and explicit state
//! tensors, which checks the composition against the standard decode.
//!
//! Default off: `IZWI_QWEN35_CUDA_GRAPH=1` enables it until it has been
//! validated on the GPU, because a capture fault can leave a sticky CUDA
//! error that only a restart clears.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Module, Tensor};

use super::{
    ConvRingState, Qwen35FullAttention, Qwen35LayerRuntimeState, Qwen35LinearAttention,
    Qwen35Mixer, Qwen35TextModel, Qwen35TextRuntimeState,
};
use crate::backends::kv::{KvSlotMap, KvWriteCompletion, KvWriteCompletionCollector};
use crate::error::{Error, Result};
use crate::kernels::cuda::gdn;
use crate::kernels::cuda::rope::{self, QkNormRopeSpec};
use crate::kernels::cuda::segment_graph::{DeviceTable, SegmentGraph, SegmentInput};
use crate::kernels::cuda::split_decode;
use crate::kv::{KvArenaId, KvDecodeBatchMetadata, KvSlotRef};
use crate::models::architectures::qwen35::fast_path::{compare_values, Qwen35FastPath};
use crate::models::architectures::qwen35::linear::forward_shared;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// Opt-in switch (a boolean; off when unset).
pub(super) const GRAPH_ENV: &str = "IZWI_QWEN35_CUDA_GRAPH";

/// Step buffer layout (U32).
const TOKEN: usize = 0;
const POSITION: usize = 1;
const SLOT: usize = 2;
const METADATA: usize = 4;

/// Logit agreement the warm step demands of the graph body against the
/// standard decode. They differ in attention precision (16-bit q/k norm
/// inputs) and summation order, so this catches composition bugs, not
/// rounding.
const WARM_CHECK_REL_L2: f64 = 0.05;
const WARM_CHECK_MAX_FRACTION: f32 = 0.1;

fn graph_requested() -> bool {
    crate::env::flag_or(GRAPH_ENV, false)
}

/// The q/k norm + RoPE + KV write inputs one attention layer needs in the
/// graph: the kernel spec, F32 inverse frequencies and F32 head-norm gains.
pub(super) struct GraphQk {
    spec: QkNormRopeSpec,
    inv_freq: Tensor,
    q_gain: Tensor,
    k_gain: Tensor,
}

impl GraphQk {
    pub(super) fn new(attention: &Qwen35FullAttention, device: &Device) -> Result<Self> {
        Ok(Self {
            spec: QkNormRopeSpec {
                num_heads: attention.num_heads,
                num_kv_heads: attention.num_kv_heads,
                head_dim: attention.head_dim,
                rope_dim: attention.rope_dim,
                eps: attention.qk_norm_eps as f32,
            },
            inv_freq: Tensor::from_vec(
                attention.rope_inv_freqs.clone(),
                attention.rope_inv_freqs.len(),
                device,
            )?,
            q_gain: attention.q_norm_gain.clone(),
            k_gain: attention.k_norm_gain.clone(),
        })
    }
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
    graph: SegmentGraph,
    table: Option<DeviceTable>,
    step: Option<StepBuffer>,
    /// The arena and backing generation the graph was captured against.
    captured_for: Option<(KvArenaId, u64)>,
    /// Recorded after each step's queued work. The next step's stream waits
    /// on it: consecutive steps may run on different threads, whose
    /// per-thread streams are not ordered, while they share the table, the
    /// step buffer and the graph's buffers.
    #[cfg(feature = "cuda")]
    fence: Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
}

impl Runtime {
    fn disable(&mut self, reason: String) {
        self.graph.reset();
        tracing::warn!(%reason, "Qwen3.5 CUDA graph decode disabled; using the standard decode");
        self.phase = Phase::Disabled(reason);
    }

    /// Order this step's work after the previous step's (a GPU-side wait).
    fn order_after_previous(&self, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let (Some(fence), Device::Cuda(cuda)) = (&self.fence, device) {
            return cuda
                .cuda_stream()
                .wait(fence)
                .map_err(|error| graph_error(&format!("step fence wait: {error}")));
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
                        .map_err(|error| graph_error(&format!("step fence: {error}")))?,
                );
            }
            if let Some(fence) = &self.fence {
                fence
                    .record(&stream)
                    .map_err(|error| graph_error(&format!("step fence record: {error}")))?;
            }
        }
        let _ = device;
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // Wait for the last step, which may have run on another thread,
        // before the graph and the buffers it used are destroyed.
        #[cfg(feature = "cuda")]
        if let Some(fence) = &self.fence {
            let _ = fence.synchronize();
        }
    }
}

/// Graph decode for one model (see the module docs).
pub(super) struct Qwen35DecodeGraph {
    /// Per layer: the DeltaNet ordinal (table slot `ordinal * TABLE_ENTRIES`).
    linear_ordinal: Vec<Option<usize>>,
    linear_count: usize,
    /// Full-attention layers (arena bindings `0..attention`).
    attention: usize,
    /// Longest context the step buffer's block table must cover.
    max_context: usize,
    /// Capture graphs (CUDA); otherwise the body runs eagerly every step.
    capture: bool,
    runtime: Mutex<Runtime>,
    warmups: AtomicU64,
    captures: AtomicU64,
    replays: AtomicU64,
    verified: AtomicU64,
    standard_steps: AtomicU64,
}

impl Qwen35DecodeGraph {
    /// Whether graph decode applies to `model`, and why not. `allow_eager`
    /// admits non-CUDA devices with an eager body (tests).
    pub(super) fn resolve(
        model: &Qwen35TextModel,
        graphs_off: Option<&'static str>,
        max_context: usize,
        allow_eager: bool,
    ) -> (Option<Self>, Qwen35FastPath) {
        let device = &model.device;
        let refuse = |reason: String| (None, Qwen35FastPath::legacy(reason));
        if !graph_requested() && !allow_eager {
            return refuse(format!(
                "off until validated on the GPU; set {GRAPH_ENV}=1 to enable"
            ));
        }
        if let Some(reason) = graphs_off {
            return refuse(reason.to_string());
        }
        if !device.is_cuda() && !allow_eager {
            return refuse("CUDA graph decode runs on CUDA only".into());
        }
        if model.finite_diagnostics_enabled {
            return refuse("CUDA graph decode is off while finite diagnostics read values".into());
        }
        if device.is_cuda() && !model.projections_all_kquant() {
            return refuse(
                "CUDA graph decode needs the K-quant MMVQ on every projection (QMatMul cannot be captured)"
                    .into(),
            );
        }
        let mut linear_ordinal = Vec::with_capacity(model.layers.len());
        let mut linear_count = 0usize;
        let mut attention = 0usize;
        for layer in &model.layers {
            match &layer.mixer {
                Qwen35Mixer::Linear(mixer) => {
                    if mixer.fused_decode.is_none() {
                        return refuse(
                            "CUDA graph decode needs the fused DeltaNet decode on every layer"
                                .into(),
                        );
                    }
                    linear_ordinal.push(Some(linear_count));
                    linear_count += 1;
                }
                Qwen35Mixer::Full(full) => {
                    if device.is_cuda() {
                        if full.graph_qk.is_none() {
                            return refuse(
                                "CUDA graph decode needs the fused q/k norm + RoPE + KV write"
                                    .into(),
                            );
                        }
                        if !split_decode::supported(
                            DType::F16,
                            full.num_heads,
                            full.num_kv_heads,
                            full.head_dim,
                            full.head_dim,
                        ) {
                            return refuse(
                                "split decode does not cover this attention geometry".into(),
                            );
                        }
                    }
                    linear_ordinal.push(None);
                    attention += 1;
                }
            }
        }
        if attention == 0 {
            return refuse("the model has no full-attention layer".into());
        }
        (
            Some(Self {
                linear_ordinal,
                linear_count,
                attention,
                max_context,
                capture: device.is_cuda(),
                runtime: Mutex::new(Runtime {
                    phase: Phase::Warm,
                    graph: SegmentGraph::default(),
                    table: None,
                    step: None,
                    captured_for: None,
                    #[cfg(feature = "cuda")]
                    fence: None,
                }),
                warmups: AtomicU64::new(0),
                captures: AtomicU64::new(0),
                replays: AtomicU64::new(0),
                verified: AtomicU64::new(0),
                standard_steps: AtomicU64::new(0),
            }),
            Qwen35FastPath::Fused,
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
            "warmups": self.warmups.load(Relaxed),
            "captures": self.captures.load(Relaxed),
            "replays": self.replays.load(Relaxed),
            "verified": self.verified.load(Relaxed),
            "standard_steps": self.standard_steps.load(Relaxed),
        })
    }

    /// Run one decode row through the graph. `Ok(None)` hands the step to
    /// the standard decode: graph decode is off or busy, the position is
    /// multimodal, or the state is not in fused-decode form. `reference`
    /// computes the standard decode's logits for the warm check.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        model: &Qwen35TextModel,
        token: u32,
        position: [usize; 3],
        slot: KvSlotRef,
        state: &mut Qwen35TextRuntimeState,
        cache: &PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<Option<Tensor>> {
        let Ok(mut runtime) = self.runtime.try_lock() else {
            self.standard_steps.fetch_add(1, Relaxed);
            return Ok(None);
        };
        if matches!(runtime.phase, Phase::Disabled(_))
            || position[0] != position[1]
            || position[0] != position[2]
        {
            self.standard_steps.fetch_add(1, Relaxed);
            return Ok(None);
        }
        let device = model.device.clone();
        for (layer, layer_state) in model.layers.iter().zip(state.layers.iter_mut()) {
            layer.ensure_state_initialized(layer_state, &device)?;
        }
        let Some(inputs) = StateInputs::gather(model, state)? else {
            self.standard_steps.fetch_add(1, Relaxed);
            return Ok(None);
        };
        runtime.order_after_previous(&device)?;
        let request = StepRequest {
            token,
            position: position[0],
            slot,
            cache,
            slots,
            metadata,
        };
        let step = if self.capture {
            match self.captured(&mut runtime, model, &request, &inputs, reference) {
                Ok((done, graph_completions)) => {
                    for completion in graph_completions {
                        completions.collect(completion)?;
                    }
                    Ok(Some(done))
                }
                Err(error) => {
                    runtime.disable(format!("graph decode step failed: {error}"));
                    Ok(None)
                }
            }
        } else {
            self.eager(&mut runtime, model, &request, &inputs, completions)
                .map(Some)
        };
        let marked = runtime.mark_step_end(&device);
        let done = match step {
            Ok(Some(done)) => done,
            Ok(None) => {
                self.standard_steps.fetch_add(1, Relaxed);
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        marked?;
        drop(runtime);
        done.published.publish(model, state)?;
        Ok(Some(done.logits))
    }

    /// Eager devices: the body with the arena's attention, every step.
    fn eager(
        &self,
        runtime: &mut Runtime,
        model: &Qwen35TextModel,
        request: &StepRequest<'_>,
        inputs: &StateInputs,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<StepDone> {
        let device = model.device.clone();
        let max_blocks = self.max_blocks(request.cache)?;
        let step = self.step_buffer(runtime, &device, max_blocks, request)?;
        let published = StateOutputs::allocate(model, &device)?;
        let ios = direct_io(model, inputs, &published);
        let rope = model.rope_table(
            model.first_attention()?,
            &[[request.position; 3]],
            (1, 1),
            DType::F32,
        )?;
        let mut attention = Attention::Arena {
            rope: rope.as_ref(),
            cache: request.cache,
            slots: request.slots,
            metadata: request.metadata,
            completions,
        };
        self.warmups.fetch_add(1, Relaxed);
        let logits = body(model, &step, &ios, &mut attention)?;
        Ok(StepDone { logits, published })
    }

    /// CUDA: warm, capture and verify, or replay, inside the arena's
    /// graph-write bookkeeping.
    fn captured(
        &self,
        runtime: &mut Runtime,
        model: &Qwen35TextModel,
        request: &StepRequest<'_>,
        inputs: &StateInputs,
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<(StepDone, Vec<KvWriteCompletion>)> {
        let device = model.device.clone();
        let arena = request.cache.arena().clone();
        let config = arena.config();
        if config.dtype != DType::F16 && config.dtype != DType::BF16 {
            return Err(graph_error("graph attention needs a 16-bit KV cache"));
        }
        let kv_dtype = config.dtype;
        let page_tokens = config.page_tokens as usize;
        let max_blocks = self.max_blocks(request.cache)?;
        // The warm step checks the body against the standard decode first:
        // the standard path calls the arena, which `with_graph_writes`
        // serializes.
        let expected = if runtime.phase == Phase::Warm {
            Some(reference()?)
        } else {
            None
        };
        let step = self.step_buffer(runtime, &device, max_blocks, request)?;
        let published = StateOutputs::allocate(model, &device)?;
        let entries = self.table_values(inputs, &published)?;
        if runtime.table.as_ref().map(DeviceTable::entries) != Some(entries.len()) {
            runtime.table = Some(DeviceTable::new(&device, entries.len())?);
            runtime.graph.reset();
            runtime.phase = Phase::Warm;
        }
        runtime
            .table
            .as_mut()
            .expect("table allocated")
            .write(0, &entries)?;
        let ios = self.table_io(runtime.table.as_ref().expect("table").address())?;
        let bindings = (0..self.attention)
            .map(|layer| request.cache.layer_binding(layer))
            .collect::<Result<Vec<_>>>()?;
        let metadata = runtime
            .step
            .as_ref()
            .expect("step buffer allocated")
            .metadata()?;
        let max_partitions =
            split_decode::partitions(1, model.first_attention()?.num_kv_heads, self.max_context);
        let mut output = None;
        let completions = arena.with_graph_writes(
            &bindings,
            request.slots,
            self.attention as u64,
            &mut |caches| {
                let key = (arena.id(), caches.generation);
                if runtime.captured_for != Some(key) {
                    runtime.graph.reset();
                    if runtime.phase != Phase::Warm {
                        runtime.phase = Phase::Capture;
                    }
                    runtime.captured_for = Some(key);
                }
                let mut attention = Attention::Graph {
                    caches: &caches.layers,
                    step: &step,
                    metadata: metadata.clone(),
                    kv_dtype,
                    page_tokens,
                    max_blocks,
                    max_partitions,
                };
                let run_body = |attention: &mut Attention<'_>, step: &Tensor| {
                    body(model, step, &ios, attention)
                        .map(|out| vec![out])
                        .map_err(candle_core::Error::wrap)
                };
                output = match runtime.phase.clone() {
                    Phase::Warm => {
                        self.warmups.fetch_add(1, Relaxed);
                        SegmentGraph::warm(&device, std::slice::from_ref(&step), |inputs| {
                            run_body(&mut attention, &inputs[0])
                        })?
                        .into_iter()
                        .next()
                    }
                    Phase::Capture | Phase::Verify => {
                        if runtime.phase == Phase::Capture {
                            runtime.graph.capture(
                                &[SegmentInput {
                                    tensor: &step,
                                    adopt: true,
                                }],
                                |inputs| run_body(&mut attention, &inputs[0]),
                            )?;
                            self.captures.fetch_add(1, Relaxed);
                        }
                        let eager = body(model, &step, &ios, &mut attention)?;
                        self.verify(runtime, model, inputs, &published, &device, &step, &eager)?;
                        self.verified.fetch_add(1, Relaxed);
                        Some(eager)
                    }
                    Phase::Replay => {
                        let out = runtime.graph.replay(&[SegmentInput {
                            tensor: &step,
                            adopt: true,
                        }])?;
                        self.replays.fetch_add(1, Relaxed);
                        // The retained output is overwritten by the next replay.
                        out.into_iter().next().map(|out| out.copy()).transpose()?
                    }
                    Phase::Disabled(_) => None,
                };
                Ok(())
            },
        )?;
        let logits = output.ok_or_else(|| graph_error("the step produced no output"))?;
        if let Some(expected) = expected {
            compare_values(
                "graph decode against the standard decode",
                &host(&logits)?,
                &host(&expected)?,
                WARM_CHECK_REL_L2,
                WARM_CHECK_MAX_FRACTION,
            )?;
        }
        runtime.phase = match runtime.phase.clone() {
            Phase::Warm => Phase::Capture,
            Phase::Capture => Phase::Verify,
            Phase::Verify => Phase::Replay,
            other => other,
        };
        Ok((StepDone { logits, published }, completions))
    }

    /// Replay into scratch DeltaNet outputs and compare with the eager run.
    #[allow(clippy::too_many_arguments)]
    fn verify(
        &self,
        runtime: &mut Runtime,
        model: &Qwen35TextModel,
        inputs: &StateInputs,
        published: &StateOutputs,
        device: &Device,
        step: &Tensor,
        eager: &Tensor,
    ) -> Result<()> {
        let scratch = StateOutputs::allocate(model, device)?;
        let scratch_entries = self.table_values(inputs, &scratch)?;
        let published_entries = self.table_values(inputs, published)?;
        let table = runtime
            .table
            .as_mut()
            .ok_or_else(|| graph_error("verification without a state table"))?;
        table.write(0, &scratch_entries)?;
        let replayed = runtime.graph.replay(&[SegmentInput {
            tensor: step,
            adopt: true,
        }]);
        runtime
            .table
            .as_mut()
            .expect("table")
            .write(0, &published_entries)?;
        let replayed = replayed?;
        let replayed = replayed
            .first()
            .ok_or_else(|| graph_error("the graph returned nothing"))?;
        compare_values(
            "graph decode output",
            &host(replayed)?,
            &host(eager)?,
            0.01,
            0.02,
        )?;
        for (layer, (graph_state, eager_state)) in
            scratch.layers.iter().zip(&published.layers).enumerate()
        {
            let (Some(graph_state), Some(eager_state)) = (graph_state, eager_state) else {
                continue;
            };
            compare_values(
                &format!("graph decode layer {layer} conv history"),
                &host(&graph_state.history)?,
                &host(&eager_state.history)?,
                1e-6,
                1e-6,
            )?;
            compare_values(
                &format!("graph decode layer {layer} recurrent state"),
                &host(&graph_state.state)?,
                &host(&eager_state.state)?,
                0.01,
                0.02,
            )?;
        }
        Ok(())
    }

    /// Block-table capacity of the step buffer: the longest context in pages,
    /// plus one for a first-page offset.
    fn max_blocks(&self, cache: &PhysicalPagedKvCache) -> Result<usize> {
        let page_tokens = cache.arena().config().page_tokens as usize;
        if page_tokens == 0 {
            return Err(graph_error("the KV arena has zero-token pages"));
        }
        Ok(self.max_context.div_ceil(page_tokens) + 1)
    }

    /// The step buffer for `request`, allocated (or reallocated for a new
    /// block-table capacity, which forces a recapture) and brought up to
    /// date.
    fn step_buffer(
        &self,
        runtime: &mut Runtime,
        device: &Device,
        max_blocks: usize,
        request: &StepRequest<'_>,
    ) -> Result<Tensor> {
        let [sequence] = request.metadata.sequences.as_slice() else {
            return Err(graph_error("graph decode serves one row"));
        };
        if sequence.blocks.len() > max_blocks {
            return Err(graph_error("the row's block table exceeds the step buffer"));
        }
        if runtime.step.as_ref().map(|step| step.max_blocks) != Some(max_blocks) {
            runtime.step = Some(StepBuffer::new(device, max_blocks)?);
            runtime.graph.reset();
            if runtime.phase != Phase::Warm {
                runtime.phase = Phase::Capture;
            }
        }
        let page_tokens = request.cache.arena().config().page_tokens;
        let slot = request
            .slot
            .block
            .index
            .checked_mul(page_tokens)
            .and_then(|base| base.checked_add(request.slot.offset))
            .ok_or_else(|| graph_error("KV slot index overflow"))?;
        let position = u32::try_from(request.position)
            .map_err(|_| graph_error("decode position exceeds u32"))?;
        let step = runtime.step.as_mut().expect("step buffer allocated");
        let mut values = vec![0u32; METADATA + 2 + sequence.blocks.len()];
        values[TOKEN] = request.token;
        values[POSITION] = position;
        values[SLOT] = slot;
        values[METADATA] = sequence.context_len;
        values[METADATA + 1] = sequence.first_page_offset;
        for (entry, block) in values[METADATA + 2..].iter_mut().zip(&sequence.blocks) {
            *entry = block.index;
        }
        step.update(&values)?;
        Ok(step.tensor.clone())
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
            Err(graph_error("device tables need CUDA"))
        }
    }
}

/// One decode row's inputs for the step.
struct StepRequest<'a> {
    token: u32,
    position: usize,
    slot: KvSlotRef,
    cache: &'a PhysicalPagedKvCache,
    slots: &'a dyn KvSlotMap,
    metadata: &'a KvDecodeBatchMetadata,
}

/// A finished step: logits and the DeltaNet state to publish.
struct StepDone {
    logits: Tensor,
    published: StateOutputs,
}

/// The step buffer: per-step values the graph reads on the device.
struct StepBuffer {
    tensor: Tensor,
    max_blocks: usize,
    /// What the device holds, to upload only the changed range.
    mirror: Vec<u32>,
}

impl StepBuffer {
    fn new(device: &Device, max_blocks: usize) -> Result<Self> {
        let len = METADATA + 2 + max_blocks;
        Ok(Self {
            tensor: Tensor::zeros(len, DType::U32, device)?,
            max_blocks,
            mirror: vec![0; len],
        })
    }

    /// Bring the device copy to `values` (a prefix of the buffer), uploading
    /// only the range that changed, stream-ordered after queued work.
    fn update(&mut self, values: &[u32]) -> Result<()> {
        let differs = |index: &usize| values[*index] != self.mirror[*index];
        let (Some(first), Some(last)) = (
            (0..values.len()).find(differs),
            (0..values.len()).rev().find(differs),
        ) else {
            return Ok(());
        };
        upload(&self.tensor, first, &values[first..=last])?;
        self.mirror[first..=last].copy_from_slice(&values[first..=last]);
        Ok(())
    }

    fn metadata(&self) -> Result<Tensor> {
        Ok(self.tensor.narrow(0, METADATA, 2 + self.max_blocks)?)
    }
}

/// Write `values` into the U32 `tensor` at `offset`, in place.
fn upload(tensor: &Tensor, offset: usize, values: &[u32]) -> Result<()> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = tensor.device() {
        use candle_core::cuda_backend::cudarc::driver::{result, DevicePtr};
        let (storage, layout) = tensor.storage_and_layout();
        let candle_core::Storage::Cuda(storage) = &*storage else {
            return Err(graph_error("the step buffer is not a CUDA tensor"));
        };
        let stream = cuda.cuda_stream();
        let (base, _guard) = storage.as_cuda_slice::<u32>()?.device_ptr(&stream);
        let destination = base + ((layout.start_offset() + offset) * 4) as u64;
        stream
            .context()
            .bind_to_thread()
            .map_err(|error| graph_error(&format!("step buffer bind: {error}")))?;
        // SAFETY: the range lies inside the buffer (callers write within its
        // length), and a pageable source is staged before the call returns.
        unsafe { result::memcpy_htod_async(destination, values, stream.cu_stream()) }
            .map_err(|error| graph_error(&format!("step buffer upload: {error}")))?;
        return Ok(());
    }
    let update = Tensor::from_slice(values, values.len(), tensor.device())?;
    tensor.slice_set(&update, 0, offset)?;
    Ok(())
}

/// Where one DeltaNet layer reads its state and writes the next one.
enum LinearStateIo {
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

/// Attention for the step body.
enum Attention<'a> {
    /// The fused q/k RoPE + KV write and the adaptive split decode, both
    /// driven by the step buffer: capturable.
    Graph {
        caches: &'a [(Tensor, Tensor)],
        step: &'a Tensor,
        metadata: Tensor,
        kv_dtype: DType,
        page_tokens: usize,
        max_blocks: usize,
        max_partitions: usize,
    },
    /// The arena's own write and decode (eager devices, tests).
    Arena {
        rope: Option<&'a super::Qwen35RopeTable>,
        cache: &'a PhysicalPagedKvCache,
        slots: &'a dyn KvSlotMap,
        metadata: &'a KvDecodeBatchMetadata,
        completions: &'a mut KvWriteCompletionCollector,
    },
}

/// The step body: embedding of `step[TOKEN]`, every layer, the final norm
/// and the LM head. Returns logits `[1, 1, vocab]`, as the standard decode
/// does.
fn body(
    model: &Qwen35TextModel,
    step: &Tensor,
    ios: &[Option<LinearStateIo>],
    attention: &mut Attention<'_>,
) -> Result<Tensor> {
    let ids = step.narrow(0, TOKEN, 1)?.reshape((1, 1))?;
    let mut hidden = model.token_embeddings.forward(&ids)?;
    let mut physical = 0usize;
    for (index, layer) in model.layers.iter().enumerate() {
        let residual = hidden.clone();
        let normalized = layer.attn_norm.forward(&hidden)?;
        let mixed = match &layer.mixer {
            Qwen35Mixer::Linear(mixer) => {
                let io = ios[index]
                    .as_ref()
                    .ok_or_else(|| graph_error("DeltaNet layer has no state IO"))?;
                mixer.forward_graph(&normalized, io)?
            }
            Qwen35Mixer::Full(full) => {
                let mixed = attend(full, &normalized, attention, physical)?;
                physical += 1;
                mixed
            }
        };
        let hidden_states = (&residual + &mixed)?;
        let residual = hidden_states.clone();
        let hidden_states = layer.post_attention_norm.forward(&hidden_states)?;
        let hidden_states = layer.ffn.forward(&hidden_states)?;
        hidden = (&residual + &hidden_states)?;
    }
    model.project_hidden_span(&hidden)
}

fn attend(
    full: &Qwen35FullAttention,
    normalized: &Tensor,
    attention: &mut Attention<'_>,
    physical: usize,
) -> Result<Tensor> {
    let [q_out, k_out, v_out] =
        forward_shared(normalized, [&full.q_proj, &full.k_proj, &full.v_proj])?;
    let attended = match attention {
        Attention::Graph {
            caches,
            step,
            metadata,
            kv_dtype,
            page_tokens,
            max_blocks,
            max_partitions,
        } => {
            let qk = full
                .graph_qk
                .as_ref()
                .ok_or_else(|| graph_error("graph attention needs the fused q/k RoPE inputs"))?;
            let (cache_keys, cache_values) = caches
                .get(physical)
                .ok_or_else(|| graph_error("no KV cache for an attention layer"))?;
            // The fused q/k norm + RoPE + write and the split decode take
            // 16-bit activations; the trunk is F32.
            let q_proj =
                q_out
                    .to_dtype(*kv_dtype)?
                    .reshape((1, 1, full.num_heads, full.head_dim * 2))?;
            let keys =
                k_out
                    .to_dtype(*kv_dtype)?
                    .reshape((1, 1, full.num_kv_heads, full.head_dim))?;
            let values =
                v_out
                    .to_dtype(*kv_dtype)?
                    .reshape((1, 1, full.num_kv_heads, full.head_dim))?;
            let queries = rope::qk_norm_rope_kv(
                &q_proj,
                &keys,
                &values,
                &qk.q_gain,
                &qk.k_gain,
                &qk.inv_freq,
                &qk.spec,
                step,
                cache_keys,
                cache_values,
            )?;
            split_decode::paged_decode_adaptive(
                &queries.reshape((1, full.num_heads, full.head_dim))?,
                cache_keys,
                cache_values,
                metadata,
                1,
                full.num_heads,
                full.num_kv_heads,
                *page_tokens,
                *max_blocks,
                1.0 / (full.head_dim as f32).sqrt(),
                None,
                *max_partitions,
            )?
            .reshape((1, 1, full.num_heads * full.head_dim))?
            .to_dtype(q_out.dtype())?
        }
        Attention::Arena {
            rope,
            cache,
            slots,
            metadata,
            completions,
        } => full.decode_attend(
            &q_out,
            &k_out,
            &v_out,
            *rope,
            &[*cache],
            *slots,
            metadata,
            completions,
            physical,
        )?,
    };
    full.decode_output(&attended, &q_out)
}

impl Qwen35LinearAttention {
    /// One token through the fused DeltaNet kernels with explicit state IO:
    /// the graph form of the fused decode.
    fn forward_graph(&self, hidden_states: &Tensor, io: &LinearStateIo) -> Result<Tensor> {
        let spec = self
            .fused_decode
            .as_ref()
            .ok_or_else(|| graph_error("graph decode needs the fused DeltaNet decode"))?;
        let [mixed_qkv, z, beta_raw, alpha] = forward_shared(
            hidden_states,
            [
                &self.qkv_proj,
                &self.gate_proj,
                &self.beta_proj,
                &self.alpha_proj,
            ],
        )?;
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
    fn gather(model: &Qwen35TextModel, state: &Qwen35TextRuntimeState) -> Result<Option<Self>> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for (layer, layer_state) in model.layers.iter().zip(&state.layers) {
            let Qwen35Mixer::Linear(mixer) = &layer.mixer else {
                layers.push(None);
                continue;
            };
            let Qwen35LayerRuntimeState::Linear {
                conv_state: Some(ring),
                recurrent_state,
            } = layer_state
            else {
                return Ok(None);
            };
            if ring.slots.len() != gdn::CONV_TAPS - 1
                || ring.next_idx >= ring.slots.len()
                || ring.slots.iter().any(|slot| slot.dtype() != DType::F32)
                || recurrent_state
                    .as_ref()
                    .is_some_and(|state| state.dtype() != DType::F32)
            {
                return Ok(None);
            }
            let ordered = ring
                .ordered_slots()
                .into_iter()
                .map(|slot| slot.contiguous().map_err(Error::from))
                .collect::<Result<Vec<_>>>()?;
            let history = <[Tensor; 3]>::try_from(ordered)
                .map_err(|_| graph_error("conv history has three slots"))?;
            let state = match recurrent_state {
                Some(state) => state.contiguous()?,
                None => Tensor::zeros(
                    (1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim),
                    DType::F32,
                    &model.device,
                )?,
            };
            layers.push(Some(LayerStateIn { history, state }));
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
    /// `[1, Hv, Dk, Dv]` (F32) of every DeltaNet layer. The kernels write
    /// every element.
    fn allocate(model: &Qwen35TextModel, device: &Device) -> Result<Self> {
        let sizes = model
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                Qwen35Mixer::Linear(mixer) => Some((
                    mixer.conv_dim,
                    (mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim),
                )),
                Qwen35Mixer::Full(_) => None,
            })
            .collect::<Vec<_>>();
        let total = sizes
            .iter()
            .flatten()
            .map(|(conv_dim, (hv, dk, dv))| (gdn::CONV_TAPS - 1) * conv_dim + hv * dk * dv)
            .sum::<usize>();
        let slab = uninit_f32(total.max(1), device)?;
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

    /// Hand the written state to the row: each ring becomes the packed
    /// history and each recurrent state the fresh output.
    fn publish(self, model: &Qwen35TextModel, state: &mut Qwen35TextRuntimeState) -> Result<()> {
        for ((layer, out), layer_state) in model
            .layers
            .iter()
            .zip(self.layers)
            .zip(state.layers.iter_mut())
        {
            let (Qwen35Mixer::Linear(mixer), Some(out)) = (&layer.mixer, out) else {
                continue;
            };
            let Qwen35LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer_state
            else {
                return Err(graph_error("DeltaNet layer lost its runtime state"));
            };
            *conv_state = Some(ConvRingState::from_history(&out.history, mixer.conv_dim)?);
            *recurrent_state = Some(out.state);
        }
        Ok(())
    }
}

/// Explicit-tensor IO for every DeltaNet layer (devices without tables).
fn direct_io(
    model: &Qwen35TextModel,
    inputs: &StateInputs,
    outputs: &StateOutputs,
) -> Vec<Option<LinearStateIo>> {
    (0..model.layers.len())
        .map(|layer| {
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
            }
        })
        .collect()
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
        return Err(graph_error("DeltaNet layer has no state to address"));
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
    Err(graph_error("device tables need CUDA"))
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

fn host(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?)
}

fn graph_error(message: &str) -> Error {
    Error::InferenceError(format!("Qwen3.5 graph decode: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_decode_is_opt_in() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var(GRAPH_ENV);
        assert!(!graph_requested());
        for (value, expected) in [
            ("1", true),
            (" ON ", true),
            ("true", true),
            (" yes ", true),
            ("0", false),
            ("no", false),
        ] {
            std::env::set_var(GRAPH_ENV, value);
            assert_eq!(graph_requested(), expected, "{value}");
        }
        std::env::remove_var(GRAPH_ENV);
    }
}
