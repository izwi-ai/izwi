//! Whole-step CUDA graphs for Qwen3.6 decode batches of 2 to 32 rows.
//!
//! An eager batched decode step is thousands of small launches (40 layers of
//! projections, norms, DeltaNet or attention, router, experts), so Candle's
//! host time per launch bounds the step, not the GPU. As the one-row
//! whole-step graph (`step_graph.rs`) does for a single row, the whole step is
//! captured once per batch width and replayed: the token embeddings, every
//! layer, the final norm and (when it is block FP8) the LM head. Every
//! per-step value lives in a device **step buffer** (U32): each row's token
//! id, position and flat KV slot, then the decode metadata `[context[rows] |
//! first-page offset[rows] | block table[rows x max_blocks]]`.
//! - DeltaNet layers run the multi-row decode kernels (`gdn::decode_rows_in`)
//!   through one device address table per width, layer-major. Each step
//!   allocates one slab of fresh state outputs per row; committed state is
//!   never mutated in place.
//! - Attention runs `rope::qk_norm_rope_kv_rows` (q/k norm, RoPE and each
//!   row's K/V slot write) and `split_decode::paged_decode_adaptive`, which
//!   read the step buffer.
//! - The sparse MoE runs its per-pair kernels on the device routing. Widths
//!   from a layer's grouped-kernel threshold up are not graphed (that path is
//!   not proven capturable), and neither is anything a capture would bake a
//!   host value into.
//!
//! **Widths.** A step of `n` rows replays the graph of the smallest width in
//! [`WIDTHS`] that holds it. The spare rows decode token 0 at position 0 into
//! a KV slot past every cache (the write skips it), attend one token of page
//! 0, and read and write a scratch DeltaNet state. Every op of the step is
//! row-independent (per-row norms, projections, recurrences, attention and
//! routed experts), so a spare row changes no real row's values; its outputs
//! are dropped.
//!
//! Phases, per width: warm (eager, checked against the standard decode on a
//! copy of the rows' state), capture + verify, verify again on fresh inputs
//! (catches values a capture baked in), then replay; a verified width
//! captures again in one step when its buffers move. The step always
//! continues on the eager result until replay. Any failure turns every width
//! off for the process and the step runs the standard decode.
//!
//! Off CUDA there is no capture: with `allow_eager` (tests) every step runs
//! the same body eagerly over its own rows with the arena's attention and
//! explicit state tensors, which checks the composition against the standard
//! decode.
//!
//! On by default on CUDA: `IZWI_QWEN36_BATCH_GRAPHS=legacy` turns it off, and
//! so do `IZWI_QWEN36_CUDA_GRAPHS=legacy` and `cuda.decode_graphs=off`.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Module, Tensor};

use super::super::{
    Qwen36FeedForward, Qwen36FullAttention, Qwen36Hidden, Qwen36LinearAttention, Qwen36Mixer,
    Qwen36Projection, Qwen36TextModel, Qwen36TextRuntimeState,
};
use super::{segment_error, step_graph::upload, GraphDecodeOutput, StateInputs, StateOutputs};
use crate::backends::kv::{KvSlotMap, KvWriteCompletion, KvWriteCompletionCollector};
use crate::error::Result;
use crate::kernels::cuda::gdn;
use crate::kernels::cuda::rope;
use crate::kernels::cuda::segment_graph::{DeviceTable, SegmentGraph, SegmentInput};
use crate::kernels::cuda::split_decode;
use crate::kv::{KvArenaId, KvDecodeBatchMetadata, KvSlotRef};
use crate::models::architectures::qwen36moe::fast_path::{
    legacy_requested, Qwen36CudaSwitches, Qwen36FusedPath,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// Kill switch for the multi-row graphs (`legacy`, `off`, `0`, `false`).
pub(in crate::models::architectures::qwen36moe) const BATCH_GRAPHS_ENV: &str =
    "IZWI_QWEN36_BATCH_GRAPHS";

/// The graphed batch widths (one row has its own whole-step graph). Each
/// keeps its own graph, step buffer and table, so a step pads up to the next
/// width rather than every width being captured.
const WIDTHS: [usize; 9] = [2, 3, 4, 6, 8, 12, 16, 24, 32];

/// The width whose graph serves a step of `rows` rows (at least two).
fn graph_width(rows: usize) -> Option<usize> {
    (rows >= 2)
        .then(|| WIDTHS.iter().copied().find(|width| *width >= rows))
        .flatten()
}

/// A spare row's flat KV slot: past every cache, so the write skips it.
const SPARE_SLOT: u32 = u32::MAX;

/// Partitions a row's attention may take before its partitions grow past
/// `split_decode::ROW_PARTITION_TOKENS` (as Qwen3.5's graphs cap them).
const ROW_INVARIANT_PARTITIONS: usize = 64;

/// Logit agreement the warm step demands of the graph body against the
/// standard decode. They differ in attention summation order, so this catches
/// composition bugs, not rounding.
const WARM_CHECK_REL_L2: f64 = 0.05;
const WARM_CHECK_MAX_FRACTION: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Warm,
    Capture,
    Verify,
    /// A width that has passed verification captures again against moved
    /// buffers (a grown KV arena, a new step buffer): the same body, so one
    /// verified capture step suffices.
    Recapture,
    Replay,
}

impl Phase {
    fn name(&self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Capture => "capture",
            Self::Verify => "verify",
            Self::Recapture => "recapture",
            Self::Replay => "replay",
        }
    }
}

/// The step buffer's layout (U32) for `rows` rows of `max_blocks` block-table
/// entries: tokens, positions, flat KV slots, then the decode metadata
/// `[context[rows] | first-page offset[rows] | block table[rows x max_blocks]]`
/// that the split decode reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StepLayout {
    rows: usize,
    max_blocks: usize,
}

impl StepLayout {
    fn positions(&self) -> usize {
        self.rows
    }

    fn slots(&self) -> usize {
        2 * self.rows
    }

    fn metadata(&self) -> usize {
        3 * self.rows
    }

    fn metadata_len(&self) -> usize {
        self.rows * (2 + self.max_blocks)
    }

    fn len(&self) -> usize {
        self.metadata() + self.metadata_len()
    }

    /// The per-row values every step rewrites: tokens through the
    /// first-page offsets.
    fn head_len(&self) -> usize {
        self.metadata() + 2 * self.rows
    }

    fn blocks(&self, row: usize) -> usize {
        self.head_len() + row * self.max_blocks
    }
}

/// One width's graph and the device buffers it reads.
struct Runtime {
    phase: Phase,
    graph: SegmentGraph,
    /// Every DeltaNet layer's rows: `[layer ordinal][row][TABLE_ENTRIES]`.
    table: Option<DeviceTable>,
    step: Option<StepBuffer>,
    /// The arena and backing generation the graph was captured against.
    captured_for: Option<(KvArenaId, u64)>,
    /// The width has passed the full verification once.
    verified: bool,
}

impl Runtime {
    fn new() -> Self {
        Self {
            phase: Phase::Warm,
            graph: SegmentGraph::default(),
            table: None,
            step: None,
            captured_for: None,
            verified: false,
        }
    }

    /// Drop the captured graph: the next step at this width captures again,
    /// lightly once the width has been verified.
    fn recapture(&mut self) {
        self.graph.reset();
        self.phase = match self.phase {
            Phase::Warm => Phase::Warm,
            _ if self.verified => Phase::Recapture,
            _ => Phase::Capture,
        };
    }
}

/// Every width's runtime, and why the graphs were turned off, if they were.
struct Graphs {
    widths: BTreeMap<usize, Runtime>,
    /// The scratch DeltaNet state every width's spare rows read and write:
    /// graph steps never overlap (the lock and the fence order them).
    spare: Option<StateOutputs>,
    disabled: Option<String>,
    /// Recorded after each graph step's queued work. The next step's stream
    /// waits on it: consecutive steps may run on different threads, whose
    /// per-thread streams are not ordered, while they share the tables, the
    /// step buffers and the graphs' buffers.
    #[cfg(feature = "cuda")]
    fence: Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
}

impl Graphs {
    fn disable(&mut self, reason: String) {
        for runtime in self.widths.values_mut() {
            runtime.graph.reset();
        }
        tracing::warn!(%reason, "Qwen3.6 multi-row CUDA graphs disabled; using the standard decode");
        self.disabled = Some(reason);
    }

    /// Order this step's work after the previous step's (a GPU-side wait).
    fn order_after_previous(&self, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let (Some(fence), Device::Cuda(cuda)) = (&self.fence, device) {
            return cuda
                .cuda_stream()
                .wait(fence)
                .map_err(|error| segment_error(&format!("batch step fence wait: {error}")));
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
                        .map_err(|error| segment_error(&format!("batch step fence: {error}")))?,
                );
            }
            if let Some(fence) = &self.fence {
                fence
                    .record(&stream)
                    .map_err(|error| segment_error(&format!("batch step fence record: {error}")))?;
            }
        }
        let _ = device;
        Ok(())
    }
}

impl Drop for Graphs {
    fn drop(&mut self) {
        // Wait for the last step, which may have run on another thread,
        // before the graphs and the buffers they used are destroyed.
        #[cfg(feature = "cuda")]
        if let Some(fence) = &self.fence {
            let _ = fence.synchronize();
        }
    }
}

/// Multi-row graph decode for one model (see the module docs).
pub(in crate::models::architectures::qwen36moe) struct Qwen36BatchGraphs {
    /// Per layer: the DeltaNet ordinal (the layer's block of every table).
    linear_ordinal: Vec<Option<usize>>,
    linear_count: usize,
    /// Full-attention layers (arena bindings `0..attention`).
    attention: usize,
    /// Longest context the step buffer's block tables must cover.
    max_context: usize,
    /// The LM head runs inside the graph (block FP8; a cuBLAS head stays out).
    head_in_graph: bool,
    /// Capture graphs (CUDA); otherwise the body runs eagerly every step.
    capture: bool,
    graphs: Mutex<Graphs>,
    warmups: AtomicU64,
    captures: AtomicU64,
    replays: AtomicU64,
    verified: AtomicU64,
    recaptures: AtomicU64,
    standard_steps: AtomicU64,
    graph_rows: AtomicU64,
    spare_rows: AtomicU64,
}

/// One step's rows, all on one KV arena.
pub(in crate::models::architectures::qwen36moe) struct BatchRows<'a> {
    pub(in crate::models::architectures::qwen36moe) tokens: &'a [u32],
    pub(in crate::models::architectures::qwen36moe) positions: &'a [[usize; 3]],
    pub(in crate::models::architectures::qwen36moe) slots: &'a [KvSlotRef],
    pub(in crate::models::architectures::qwen36moe) caches: &'a [&'a PhysicalPagedKvCache],
    pub(in crate::models::architectures::qwen36moe) slot_map: &'a dyn KvSlotMap,
    pub(in crate::models::architectures::qwen36moe) metadata: &'a KvDecodeBatchMetadata,
}

impl BatchRows<'_> {
    fn len(&self) -> usize {
        self.tokens.len()
    }
}

/// A finished step: logits (or the pre-norm hidden when the LM head stays out
/// of the graph) and each row's DeltaNet state to publish.
struct StepDone {
    output: Tensor,
    published: Vec<StateOutputs>,
}

impl Qwen36BatchGraphs {
    /// Whether multi-row graph decode applies to `model`, and why not.
    /// `allow_eager` admits non-CUDA devices with an eager body (tests).
    pub(in crate::models::architectures::qwen36moe) fn resolve(
        model: &Qwen36TextModel,
        switches: &Qwen36CudaSwitches,
        allow_eager: bool,
    ) -> (Option<Self>, Qwen36FusedPath) {
        let device = &model.device;
        let refuse = |reason: String| (None, Qwen36FusedPath::legacy(reason));
        if legacy_requested(BATCH_GRAPHS_ENV) {
            return refuse(format!("{BATCH_GRAPHS_ENV}=legacy"));
        }
        if legacy_requested(super::GRAPHS_ENV) {
            return refuse(format!("{}=legacy", super::GRAPHS_ENV));
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
        let mut attention = 0usize;
        for layer in &model.layers {
            match &layer.mixer {
                Qwen36Mixer::Linear(mixer) => {
                    if mixer.fused_decode.is_none() {
                        return refuse(
                            "multi-row graphs need the fused DeltaNet decode on every layer".into(),
                        );
                    }
                    if device.is_cuda() && !mixer.batched_decode {
                        return refuse(
                            "multi-row graphs need the batched DeltaNet decode on every layer"
                                .into(),
                        );
                    }
                    linear_ordinal.push(Some(linear_count));
                    linear_count += 1;
                }
                Qwen36Mixer::Full(full) => {
                    if full.fused_qk.is_none() {
                        return refuse("multi-row graphs need the fused q/k norm + RoPE".into());
                    }
                    if device.is_cuda() {
                        if !full.batched_qk {
                            return refuse(
                                "multi-row graphs need the batched q/k norm + RoPE".into(),
                            );
                        }
                        if !split_decode::supported(
                            model.dtype(),
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
            if let Qwen36FeedForward::Sparse(moe) = &layer.ffn {
                if !moe.backend().is_fused() {
                    return refuse("multi-row graphs need the fused MoE on every layer".into());
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
                max_context: model.max_context_tokens,
                head_in_graph: matches!(model.output, Qwen36Projection::CompactFp8 { .. }),
                capture: device.is_cuda(),
                graphs: Mutex::new(Graphs {
                    widths: BTreeMap::new(),
                    spare: None,
                    disabled: None,
                    #[cfg(feature = "cuda")]
                    fence: None,
                }),
                warmups: AtomicU64::new(0),
                captures: AtomicU64::new(0),
                replays: AtomicU64::new(0),
                verified: AtomicU64::new(0),
                recaptures: AtomicU64::new(0),
                standard_steps: AtomicU64::new(0),
                graph_rows: AtomicU64::new(0),
                spare_rows: AtomicU64::new(0),
            }),
            Qwen36FusedPath::Fused,
        )
    }

    /// Diagnostics: each width's phase, the counters and the reason the
    /// graphs were turned off at runtime, if they were.
    pub(in crate::models::architectures::qwen36moe) fn summary(&self) -> serde_json::Value {
        let graphs = self
            .graphs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let widths = graphs
            .widths
            .iter()
            .map(|(width, runtime)| (width.to_string(), runtime.phase.name().into()))
            .collect::<serde_json::Map<String, serde_json::Value>>();
        serde_json::json!({
            "backend": if graphs.disabled.is_some() { "legacy" } else { "fused" },
            "legacy_reasons": graphs.disabled.iter().collect::<Vec<_>>(),
            "capture": self.capture,
            "head_in_graph": self.head_in_graph,
            "widths": widths,
            "warmups": self.warmups.load(Relaxed),
            "captures": self.captures.load(Relaxed),
            "replays": self.replays.load(Relaxed),
            "verified": self.verified.load(Relaxed),
            "recaptures": self.recaptures.load(Relaxed),
            "standard_steps": self.standard_steps.load(Relaxed),
            "graph_rows": self.graph_rows.load(Relaxed),
            "spare_rows": self.spare_rows.load(Relaxed),
        })
    }

    /// Run one decode step of `rows` (two or more) through the graph of its
    /// width. `Ok(None)` hands the step to the standard decode: the graphs
    /// are off or busy, the batch is wider than every graph or reaches a MoE
    /// layer's grouped-kernel threshold, a position is multimodal, the rows
    /// span arenas, or a state is not in fused-decode form. `reference`
    /// computes the standard decode's logits for the warm check.
    pub(in crate::models::architectures::qwen36moe) fn forward(
        &self,
        model: &Qwen36TextModel,
        rows: &BatchRows<'_>,
        states: &mut [&mut Qwen36TextRuntimeState],
        completions: &mut KvWriteCompletionCollector,
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<Option<GraphDecodeOutput>> {
        let n = rows.len();
        if n == 0
            || states.len() != n
            || rows.positions.len() != n
            || rows.slots.len() != n
            || rows.caches.len() != n
            || rows.metadata.sequences.len() != n
        {
            return Err(segment_error("the batch step's rows do not match"));
        }
        let standard = || -> Result<Option<GraphDecodeOutput>> {
            self.standard_steps.fetch_add(1, Relaxed);
            Ok(None)
        };
        let Some(width) = graph_width(n).filter(|width| *width < moe_grouped_rows(model)) else {
            return standard();
        };
        let Ok(mut graphs) = self.graphs.try_lock() else {
            return standard();
        };
        let arena = rows.caches[0].arena().id();
        if graphs.disabled.is_some()
            || rows
                .positions
                .iter()
                .any(|position| position[0] != position[1] || position[0] != position[2])
            || rows.caches.iter().any(|cache| cache.arena().id() != arena)
        {
            return standard();
        }
        let device = model.device.clone();
        let mut inputs = Vec::with_capacity(n);
        for state in states.iter_mut() {
            for (layer, layer_state) in model.layers.iter().zip(state.layers.iter_mut()) {
                layer.ensure_state_initialized(layer_state, &device)?;
            }
            let Some(input) = StateInputs::gather(model, state)? else {
                return standard();
            };
            inputs.push(input);
        }
        graphs.order_after_previous(&device)?;
        let step = if self.capture {
            match self.captured(&mut graphs, width, model, rows, &inputs, reference) {
                Ok((done, graph_completions)) => {
                    for completion in graph_completions {
                        completions.collect(completion)?;
                    }
                    Ok(Some(done))
                }
                Err(error) => {
                    graphs.disable(format!(
                        "graph decode step of {n} rows (width {width}) failed: {error}"
                    ));
                    Ok(None)
                }
            }
        } else {
            self.eager(model, rows, &inputs, completions).map(Some)
        };
        let marked = graphs.mark_step_end(&device);
        let done = match step {
            Ok(Some(done)) => done,
            Ok(None) => return standard(),
            Err(error) => return Err(error),
        };
        marked?;
        drop(graphs);
        for (published, state) in done.published.into_iter().zip(states.iter_mut()) {
            published.publish(model, state)?;
        }
        Ok(Some(if self.head_in_graph {
            GraphDecodeOutput::Logits(done.output)
        } else {
            GraphDecodeOutput::Hidden(done.output)
        }))
    }

    /// Eager devices: the body over the step's own rows with the arena's
    /// attention, every step.
    fn eager(
        &self,
        model: &Qwen36TextModel,
        rows: &BatchRows<'_>,
        inputs: &[StateInputs],
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<StepDone> {
        let device = model.device.clone();
        let n = rows.len();
        let layout = StepLayout {
            rows: n,
            max_blocks: self.max_blocks(rows.caches[0])?,
        };
        let mut step = StepBuffer::new(&device, layout)?;
        step.write(rows)?;
        let published = (0..n)
            .map(|_| StateOutputs::allocate(model, &device, None))
            .collect::<Result<Vec<_>>>()?;
        let ios = direct_io(model, inputs, &published);
        let mut attention = Attention::Arena {
            positions: rows.positions,
            caches: rows.caches,
            slots: rows.slot_map,
            metadata: rows.metadata,
            completions,
        };
        self.warmups.fetch_add(1, Relaxed);
        let output = body(
            model,
            &step.tensor,
            n,
            &ios,
            &mut attention,
            self.head_in_graph,
        )?;
        self.graph_rows.fetch_add(n as u64, Relaxed);
        Ok(StepDone { output, published })
    }

    /// CUDA: warm, capture and verify, or replay the graph of `width` rows,
    /// inside the arena's graph-write bookkeeping.
    fn captured(
        &self,
        graphs: &mut Graphs,
        width: usize,
        model: &Qwen36TextModel,
        rows: &BatchRows<'_>,
        inputs: &[StateInputs],
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<(StepDone, Vec<KvWriteCompletion>)> {
        let n = rows.len();
        let device = model.device.clone();
        let cache = rows.caches[0];
        let arena = cache.arena().clone();
        let config = arena.config();
        if config.dtype != DType::F16 && config.dtype != DType::BF16 {
            return Err(segment_error("graph attention needs a 16-bit KV cache"));
        }
        let page_tokens = config.page_tokens as usize;
        let layout = StepLayout {
            rows: width,
            max_blocks: self.max_blocks(cache)?,
        };
        let Graphs { widths, spare, .. } = graphs;
        if width > n && spare.is_none() {
            *spare = Some(StateOutputs::allocate(model, &device, None)?);
        }
        let spare = spare.as_ref();
        let runtime = widths.entry(width).or_insert_with(Runtime::new);
        // The warm step checks the body against the standard decode first:
        // the standard path calls the arena, which `with_graph_writes`
        // serializes.
        let expected = if runtime.phase == Phase::Warm {
            Some(reference()?)
        } else {
            None
        };
        if runtime.step.as_ref().map(|step| step.layout) != Some(layout) {
            runtime.step = Some(StepBuffer::new(&device, layout)?);
            runtime.recapture();
        }
        runtime
            .step
            .as_mut()
            .expect("step buffer allocated")
            .write(rows)?;
        let published = (0..n)
            .map(|_| StateOutputs::allocate(model, &device, None))
            .collect::<Result<Vec<_>>>()?;
        let entries = self.table_values(width, inputs, &published, spare)?;
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
        let ios = self.table_io(width, runtime.table.as_ref().expect("table"))?;
        let bindings = (0..self.attention)
            .map(|layer| cache.layer_binding(layer))
            .collect::<Result<Vec<_>>>()?;
        let step = runtime
            .step
            .as_ref()
            .expect("step buffer allocated")
            .tensor
            .clone();
        let positions = step.narrow(0, layout.positions(), width)?;
        let slots = step.narrow(0, layout.slots(), width)?;
        let metadata = step.narrow(0, layout.metadata(), layout.metadata_len())?;
        let kv_heads = super::full_attention(model.first_attention_layer()?)?.num_kv_heads;
        // Each row splits its own context into partitions, up to this cap:
        // the occupancy count, but at least enough for an 8k-token context.
        let max_partitions = split_decode::partitions(width, kv_heads, self.max_context).max(
            self.max_context
                .div_ceil(split_decode::ROW_PARTITION_TOKENS)
                .min(ROW_INVARIANT_PARTITIONS),
        );
        let head = self.head_in_graph;
        let mut output = None;
        let completions = arena.with_graph_writes(
            &bindings,
            rows.slot_map,
            self.attention as u64,
            &mut |caches| {
                let key = (arena.id(), caches.generation);
                if runtime.captured_for != Some(key) {
                    runtime.recapture();
                    runtime.captured_for = Some(key);
                }
                let mut attention = Attention::Graph {
                    caches: &caches.layers,
                    positions: &positions,
                    slots: &slots,
                    metadata: &metadata,
                    page_tokens,
                    max_blocks: layout.max_blocks,
                    max_partitions,
                };
                let run_body = |attention: &mut Attention<'_>, step: &Tensor| {
                    body(model, step, width, &ios, attention, head)
                        .map(|out| vec![out])
                        .map_err(candle_core::Error::wrap)
                };
                output = match runtime.phase {
                    Phase::Warm => {
                        self.warmups.fetch_add(1, Relaxed);
                        SegmentGraph::warm(&device, std::slice::from_ref(&step), |inputs| {
                            run_body(&mut attention, &inputs[0])
                        })?
                        .into_iter()
                        .next()
                    }
                    Phase::Capture | Phase::Verify | Phase::Recapture => {
                        if matches!(runtime.phase, Phase::Capture | Phase::Recapture) {
                            runtime.graph.capture(
                                &[SegmentInput {
                                    tensor: &step,
                                    adopt: true,
                                }],
                                |inputs| run_body(&mut attention, &inputs[0]),
                            )?;
                            self.captures.fetch_add(1, Relaxed);
                        }
                        let eager = body(model, &step, width, &ios, &mut attention, head)?;
                        self.verify(
                            runtime, spare, model, width, inputs, &published, &step, &eager,
                        )?;
                        self.verified.fetch_add(1, Relaxed);
                        if runtime.phase == Phase::Recapture {
                            self.recaptures.fetch_add(1, Relaxed);
                        }
                        Some(eager)
                    }
                    Phase::Replay => {
                        let out = runtime.graph.replay(&[SegmentInput {
                            tensor: &step,
                            adopt: true,
                        }])?;
                        self.replays.fetch_add(1, Relaxed);
                        // The retained output is overwritten by the next
                        // replay; only the step's own rows are kept.
                        out.into_iter()
                            .next()
                            .map(|out| out.narrow(0, 0, n)?.copy())
                            .transpose()?
                    }
                };
                Ok(())
            },
        )?;
        let output = output.ok_or_else(|| segment_error("the batch step produced no output"))?;
        let output = if output.dim(0)? == n {
            output
        } else {
            output.narrow(0, 0, n)?
        };
        if let Some(expected) = expected {
            let logits = if head {
                output.clone()
            } else {
                model.project_hidden_span(&output)?
            };
            device_compare(
                &[(
                    "multi-row graph decode against the standard decode",
                    logits,
                    expected,
                )],
                WARM_CHECK_REL_L2,
                WARM_CHECK_MAX_FRACTION,
            )?;
        }
        runtime.phase = match runtime.phase {
            Phase::Warm => Phase::Capture,
            Phase::Capture => Phase::Verify,
            Phase::Verify | Phase::Recapture | Phase::Replay => Phase::Replay,
        };
        if runtime.phase == Phase::Replay {
            runtime.verified = true;
        }
        self.graph_rows.fetch_add(n as u64, Relaxed);
        self.spare_rows.fetch_add((width - n) as u64, Relaxed);
        Ok((StepDone { output, published }, completions))
    }

    /// Replay into scratch DeltaNet outputs and compare with the eager run,
    /// over the step's own rows: the output and every row's state slab.
    #[allow(clippy::too_many_arguments)]
    fn verify(
        &self,
        runtime: &mut Runtime,
        spare: Option<&StateOutputs>,
        model: &Qwen36TextModel,
        width: usize,
        inputs: &[StateInputs],
        published: &[StateOutputs],
        step: &Tensor,
        eager: &Tensor,
    ) -> Result<()> {
        let n = inputs.len();
        let device = model.device.clone();
        let scratch = (0..n)
            .map(|_| StateOutputs::allocate(model, &device, None))
            .collect::<Result<Vec<_>>>()?;
        let scratch_entries = self.table_values(width, inputs, &scratch, spare)?;
        let published_entries = self.table_values(width, inputs, published, spare)?;
        let table = runtime
            .table
            .as_mut()
            .ok_or_else(|| segment_error("batch verification without a state table"))?;
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
            .ok_or_else(|| segment_error("the batch graph returned nothing"))?;
        let mut pairs = vec![(
            "multi-row graph output",
            replayed.narrow(0, 0, n)?,
            eager.narrow(0, 0, n)?,
        )];
        for (graph_row, eager_row) in scratch.iter().zip(published) {
            if let (Some(graph_slab), Some(eager_slab)) = (graph_row.slab(), eager_row.slab()) {
                pairs.push((
                    "multi-row graph DeltaNet state",
                    graph_slab.clone(),
                    eager_slab.clone(),
                ));
            }
        }
        device_compare(&pairs, 0.01, 0.02).map(drop)
    }

    /// Block-table capacity per row: the longest context in pages, plus one
    /// for a first-page offset.
    fn max_blocks(&self, cache: &PhysicalPagedKvCache) -> Result<usize> {
        let page_tokens = cache.arena().config().page_tokens as usize;
        if page_tokens == 0 {
            return Err(segment_error("the KV arena has zero-token pages"));
        }
        Ok(self.max_context.div_ceil(page_tokens) + 1)
    }

    /// Every DeltaNet layer's entries for `width` rows, layer-major: the
    /// step's rows `inputs` to `outputs`, then the spare rows on `spare`
    /// (CUDA).
    fn table_values(
        &self,
        width: usize,
        inputs: &[StateInputs],
        outputs: &[StateOutputs],
        spare: Option<&StateOutputs>,
    ) -> Result<Vec<u64>> {
        let mut values = Vec::with_capacity(self.linear_count * width * gdn::TABLE_ENTRIES);
        for (layer, ordinal) in self.linear_ordinal.iter().enumerate() {
            if ordinal.is_none() {
                continue;
            }
            for (input, output) in inputs.iter().zip(outputs) {
                values.extend(super::table_entries(input, output, layer)?);
            }
            if width > inputs.len() {
                let spare = spare.ok_or_else(|| segment_error("spare rows have no state"))?;
                let entries = spare_entries(spare, layer)?;
                for _ in inputs.len()..width {
                    values.extend(entries);
                }
            }
        }
        Ok(values)
    }

    /// Table-addressed IO for every DeltaNet layer of a `width`-row step.
    fn table_io(&self, width: usize, table: &DeviceTable) -> Result<Vec<Option<RowsIo>>> {
        #[cfg(feature = "cuda")]
        {
            Ok(self
                .linear_ordinal
                .iter()
                .map(|ordinal| {
                    ordinal.map(|ordinal| {
                        RowsIo::Table(gdn::RowsTable::of(
                            table,
                            ordinal * width * gdn::TABLE_ENTRIES,
                        ))
                    })
                })
                .collect())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (width, table);
            Err(segment_error("device tables need CUDA"))
        }
    }
}

/// The rows the sparse MoE layers serve through their per-pair kernels: below
/// every layer's grouped-kernel threshold (set by the load-time timings).
fn moe_grouped_rows(model: &Qwen36TextModel) -> usize {
    model
        .layers
        .iter()
        .filter_map(|layer| match &layer.ffn {
            Qwen36FeedForward::Sparse(moe) => moe
                .fused_experts()
                .filter(|experts| experts.grouped())
                .map(|experts| experts.grouped_min_tokens()),
            _ => None,
        })
        .min()
        .unwrap_or(usize::MAX)
}

/// The step buffer: per-step values the graph reads on the device.
struct StepBuffer {
    tensor: Tensor,
    layout: StepLayout,
    /// What the device holds, to upload only the changed ranges.
    mirror: Vec<u32>,
}

impl StepBuffer {
    fn new(device: &Device, layout: StepLayout) -> Result<Self> {
        let len = layout.len();
        let mut buffer = Self {
            tensor: Tensor::zeros(len, DType::U32, device)?,
            layout,
            mirror: vec![0; len],
        };
        buffer.update(0, &spare_head(layout))?;
        Ok(buffer)
    }

    /// Write `rows` into the buffer: each row's token, position, flat slot,
    /// context, first-page offset and block table; the rows past them stay
    /// spare.
    fn write(&mut self, rows: &BatchRows<'_>) -> Result<()> {
        let layout = self.layout;
        let n = rows.len();
        if n > layout.rows {
            return Err(segment_error("the step has more rows than its width"));
        }
        let page_tokens = rows.caches[0].arena().config().page_tokens;
        let mut head = spare_head(layout);
        for row in 0..n {
            let sequence = &rows.metadata.sequences[row];
            if sequence.blocks.len() > layout.max_blocks {
                return Err(segment_error("a row's block table exceeds the step buffer"));
            }
            let slot = rows.slots[row];
            head[row] = rows.tokens[row];
            head[layout.positions() + row] = u32::try_from(rows.positions[row][0])
                .map_err(|_| segment_error("decode position exceeds u32"))?;
            head[layout.slots() + row] = slot
                .block
                .index
                .checked_mul(page_tokens)
                .and_then(|base| base.checked_add(slot.offset))
                .filter(|flat| *flat != SPARE_SLOT)
                .ok_or_else(|| segment_error("KV slot index overflow"))?;
            head[layout.metadata() + row] = sequence.context_len;
            head[layout.metadata() + layout.rows + row] = sequence.first_page_offset;
        }
        self.update(0, &head)?;
        for row in 0..layout.rows {
            let blocks = match rows.metadata.sequences.get(row) {
                Some(sequence) => sequence.blocks.iter().map(|block| block.index).collect(),
                // A spare row reads page 0.
                None => vec![0u32],
            };
            self.update(layout.blocks(row), &blocks)?;
        }
        Ok(())
    }

    /// Bring the device range at `offset` to `values`, uploading only the
    /// span that changed, stream-ordered after queued work.
    fn update(&mut self, offset: usize, values: &[u32]) -> Result<()> {
        if offset + values.len() > self.mirror.len() {
            return Err(segment_error("a step buffer write past its end"));
        }
        let mirror = &self.mirror[offset..offset + values.len()];
        let differs = |index: &usize| values[*index] != mirror[*index];
        let (Some(first), Some(last)) = (
            (0..values.len()).find(differs),
            (0..values.len()).rev().find(differs),
        ) else {
            return Ok(());
        };
        upload(&self.tensor, offset + first, &values[first..=last])?;
        self.mirror[offset + first..=offset + last].copy_from_slice(&values[first..=last]);
        Ok(())
    }
}

/// The step buffer's head with every row spare: token 0 at position 0, the
/// spare slot, and a one-token context (a zero context would leave the
/// row's split-decode partitions empty).
fn spare_head(layout: StepLayout) -> Vec<u32> {
    let mut head = vec![0u32; layout.head_len()];
    for row in 0..layout.rows {
        head[layout.slots() + row] = SPARE_SLOT;
        head[layout.metadata() + row] = 1;
    }
    head
}

/// Where one DeltaNet layer reads its rows' state and writes the next one.
enum RowsIo {
    /// CUDA: the layer's rows in the step's device table.
    #[cfg(feature = "cuda")]
    Table(gdn::RowsTable),
    /// Other devices: explicit tensors per row; the outputs are written in
    /// place.
    Direct(Vec<DirectRow>),
}

struct DirectRow {
    history: [Tensor; 3],
    state: Tensor,
    history_out: Tensor,
    state_out: Tensor,
}

/// Explicit-tensor IO per row for every DeltaNet layer (devices without
/// tables).
fn direct_io(
    model: &Qwen36TextModel,
    inputs: &[StateInputs],
    outputs: &[StateOutputs],
) -> Vec<Option<RowsIo>> {
    (0..model.layers.len())
        .map(|layer| {
            let rows = inputs
                .iter()
                .zip(outputs)
                .map(|(input, output)| {
                    match (input.layers[layer].as_ref(), output.layers[layer].as_ref()) {
                        (Some(input), Some(output)) => Some(DirectRow {
                            history: input.history.clone(),
                            state: input.state.clone(),
                            history_out: output.history.clone(),
                            state_out: output.state.clone(),
                        }),
                        _ => None,
                    }
                })
                .collect::<Option<Vec<_>>>()?;
            Some(RowsIo::Direct(rows))
        })
        .collect()
}

/// A spare row's entries for `layer`: it reads and writes the spare slab in
/// place, three history slots from its history rows.
#[cfg(feature = "cuda")]
fn spare_entries(spare: &StateOutputs, layer: usize) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    let Some(spare) = spare.layers[layer].as_ref() else {
        return Err(segment_error("DeltaNet layer has no spare state"));
    };
    let history = |slot: usize| -> Result<u64> { Ok(gdn::f32_address(&spare.history.i(slot)?)?) };
    Ok([
        history(0)?,
        history(1)?,
        history(2)?,
        gdn::f32_address(&spare.history)?,
        gdn::f32_address(&spare.state)?,
        gdn::f32_address(&spare.state)?,
    ])
}

#[cfg(not(feature = "cuda"))]
fn spare_entries(_spare: &StateOutputs, _layer: usize) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    Err(segment_error("device tables need CUDA"))
}

/// Attention for the step body.
enum Attention<'a> {
    /// The fused q/k RoPE + KV write and the adaptive split decode, both
    /// driven by the step buffer: capturable.
    Graph {
        caches: &'a [(Tensor, Tensor)],
        positions: &'a Tensor,
        slots: &'a Tensor,
        metadata: &'a Tensor,
        page_tokens: usize,
        max_blocks: usize,
        max_partitions: usize,
    },
    /// The arena's own write and decode (eager devices, tests).
    Arena {
        positions: &'a [[usize; 3]],
        caches: &'a [&'a PhysicalPagedKvCache],
        slots: &'a dyn KvSlotMap,
        metadata: &'a KvDecodeBatchMetadata,
        completions: &'a mut KvWriteCompletionCollector,
    },
}

/// The step body over `rows` rows: the embeddings of the step buffer's
/// tokens, every layer, the final norm and, with `head`, the LM head.
/// Returns logits `[rows, 1, vocab]`, as the standard decode does, or the
/// pre-norm hidden `[rows, 1, hidden]`.
fn body(
    model: &Qwen36TextModel,
    step: &Tensor,
    rows: usize,
    ios: &[Option<RowsIo>],
    attention: &mut Attention<'_>,
    head: bool,
) -> Result<Tensor> {
    let ids = step.narrow(0, 0, rows)?.reshape((rows, 1))?;
    let mut hidden = Qwen36Hidden::new(model.token_embeddings.forward(&ids)?);
    let mut physical = 0usize;
    for (index, layer) in model.layers.iter().enumerate() {
        let (residual, normalized) = hidden.normalized(&layer.attn_norm)?;
        let mixed = match &layer.mixer {
            Qwen36Mixer::Linear(mixer) => {
                let io = ios[index]
                    .as_ref()
                    .ok_or_else(|| segment_error("DeltaNet layer has no state IO"))?;
                mixer.forward_graph_rows(&normalized, io, rows)?
            }
            Qwen36Mixer::Full(full) => {
                let mixed = attend(full, &normalized, attention, physical, rows)?;
                physical += 1;
                mixed
            }
        };
        hidden = layer.post_mixer(&residual, &mixed)?;
    }
    let hidden = hidden.materialize()?;
    if head {
        model.project_hidden_span(&hidden)
    } else {
        Ok(hidden)
    }
}

fn attend(
    full: &Qwen36FullAttention,
    normalized: &Tensor,
    attention: &mut Attention<'_>,
    physical: usize,
    rows: usize,
) -> Result<Tensor> {
    let (q_proj, keys, values) = full.decode_projections(normalized)?;
    let attended = match attention {
        Attention::Graph {
            caches,
            positions,
            slots,
            metadata,
            page_tokens,
            max_blocks,
            max_partitions,
        } => {
            let (spec, inv_freq) = full
                .fused_qk
                .as_ref()
                .ok_or_else(|| segment_error("graph attention needs the fused q/k RoPE"))?;
            let (cache_keys, cache_values) = caches
                .get(physical)
                .ok_or_else(|| segment_error("no KV cache for an attention layer"))?;
            let (heads, kv_heads, dim) = (full.num_heads, full.num_kv_heads, full.head_dim);
            let queries = rope::qk_norm_rope_kv_rows(
                &q_proj.reshape((rows, heads * 2 * dim))?,
                &keys.reshape((rows, kv_heads * dim))?,
                &values.reshape((rows, kv_heads * dim))?,
                &full.q_norm.weight,
                &full.k_norm.weight,
                inv_freq,
                spec,
                positions,
                slots,
                cache_keys,
                cache_values,
                rows,
            )?;
            split_decode::paged_decode_adaptive(
                &queries.reshape((rows, heads, dim))?,
                cache_keys,
                cache_values,
                metadata,
                rows,
                heads,
                kv_heads,
                *page_tokens,
                *max_blocks,
                1.0 / (dim as f32).sqrt(),
                None,
                *max_partitions,
            )?
            .reshape((rows, 1, heads * dim))?
            .to_dtype(q_proj.dtype())?
        }
        Attention::Arena {
            positions,
            caches,
            slots,
            metadata,
            completions,
        } => full.decode_attend(
            &q_proj,
            &keys,
            &values,
            positions,
            caches,
            *slots,
            metadata,
            completions,
            physical,
        )?,
    };
    full.decode_output(&attended, &q_proj)
}

impl Qwen36LinearAttention {
    /// `rows` tokens through the fused DeltaNet kernels with explicit state
    /// IO: the graph form of the batched fused decode.
    fn forward_graph_rows(
        &self,
        hidden_states: &Tensor,
        io: &RowsIo,
        rows: usize,
    ) -> Result<Tensor> {
        let spec = self
            .fused_decode
            .as_ref()
            .ok_or_else(|| segment_error("graph decode needs the fused DeltaNet decode"))?;
        let (mixed_qkv, z, beta_raw, alpha) = self.decode_projections(hidden_states)?;
        let value_width = self.num_v_heads * self.head_v_dim;
        let y = match io {
            #[cfg(feature = "cuda")]
            RowsIo::Table(table) => gdn::decode_rows_in(
                &mixed_qkv.reshape((rows, self.conv_dim))?,
                &self.conv_kernel,
                &z.reshape((rows, value_width))?,
                &beta_raw.reshape((rows, self.num_v_heads))?,
                &alpha.reshape((rows, self.num_v_heads))?,
                &self.dt_bias,
                &self.a,
                &self.norm.weight,
                *table,
                rows,
                spec,
            )?,
            RowsIo::Direct(direct) => {
                if direct.len() != rows {
                    return Err(segment_error("DeltaNet state IO does not cover every row"));
                }
                let mut outputs = Vec::with_capacity(rows);
                for (row, io) in direct.iter().enumerate() {
                    let (conv, next_history) = gdn::conv_decode(
                        &mixed_qkv.i(row)?,
                        &self.conv_kernel,
                        [&io.history[0], &io.history[1], &io.history[2]],
                    )?;
                    let (y, next) = gdn::recurrent_decode(
                        &conv,
                        &z.i(row)?,
                        &beta_raw.i(row)?,
                        &alpha.i(row)?,
                        &self.dt_bias,
                        &self.a,
                        &self.norm.weight,
                        &io.state,
                        spec,
                    )?;
                    io.history_out.slice_set(
                        &next_history.reshape(io.history_out.shape())?,
                        0,
                        0,
                    )?;
                    io.state_out
                        .slice_set(&next.reshape(io.state_out.shape())?, 0, 0)?;
                    outputs.push(y.reshape((1, value_width))?);
                }
                Tensor::cat(&outputs, 0)?
            }
        };
        self.out_proj.forward(&y.reshape((rows, 1, value_width))?)
    }
}

/// Compare `(label, actual, expected)` pairs with one host readback: the
/// relative L2 error and the largest error over the largest reference
/// magnitude, NaN-safe.
fn device_compare(pairs: &[(&str, Tensor, Tensor)], rel_l2: f64, max_frac: f32) -> Result<()> {
    let mut stats = Vec::with_capacity(pairs.len() * 4);
    for (label, actual, expected) in pairs {
        if actual.elem_count() != expected.elem_count() {
            return Err(segment_error(&format!(
                "{label}: {} values vs {} in the reference",
                actual.elem_count(),
                expected.elem_count()
            )));
        }
        let actual = actual.to_dtype(DType::F32)?.flatten_all()?;
        let expected = expected.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (&actual - &expected)?;
        stats.push(diff.sqr()?.sum_all()?);
        stats.push(expected.sqr()?.sum_all()?);
        stats.push(diff.abs()?.max(0)?);
        stats.push(expected.abs()?.max(0)?);
    }
    let values = Tensor::stack(&stats, 0)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    for ((label, _, _), stat) in pairs.iter().zip(values.chunks(4)) {
        let [err, norm, max_err, max_ref] = [stat[0], stat[1], stat[2], stat[3]];
        // A NaN or infinity anywhere makes the sums non-finite.
        if !(err.is_finite() && norm.is_finite() && max_err.is_finite() && max_ref.is_finite()) {
            return Err(segment_error(&format!(
                "{label}: non-finite values (error sum {err}, reference sum {norm})"
            )));
        }
        let observed = (f64::from(err) / f64::from(norm).max(f64::MIN_POSITIVE)).sqrt();
        if observed > rel_l2 || max_err > max_frac * max_ref.max(f32::MIN_POSITIVE) {
            return Err(segment_error(&format!(
                "{label}: the graph diverges from the eager step: relative L2 {observed:.3e}, max error {max_err} vs max |ref| {max_ref}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_pad_up_to_the_next_graphed_width() {
        assert_eq!(graph_width(1), None, "one row has its own graph");
        assert_eq!(graph_width(2), Some(2));
        assert_eq!(graph_width(5), Some(6));
        assert_eq!(graph_width(9), Some(12));
        assert_eq!(graph_width(17), Some(24));
        assert_eq!(graph_width(32), Some(32));
        assert_eq!(graph_width(33), None);
    }

    #[test]
    fn a_new_step_layout_marks_every_row_spare() {
        let layout = StepLayout {
            rows: 3,
            max_blocks: 2,
        };
        let head = spare_head(layout);
        assert_eq!(head.len(), layout.head_len());
        for row in 0..3 {
            assert_eq!(head[row], 0, "token");
            assert_eq!(head[layout.slots() + row], SPARE_SLOT);
            assert_eq!(head[layout.metadata() + row], 1, "one-token context");
            assert_eq!(head[layout.metadata() + 3 + row], 0, "first-page offset");
        }
        assert_eq!(layout.blocks(1), layout.head_len() + 2);
        assert_eq!(layout.len(), 3 * 3 + 3 * (2 + 2));
    }
}
