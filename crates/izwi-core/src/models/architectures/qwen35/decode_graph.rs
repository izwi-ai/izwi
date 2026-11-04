//! Whole-step CUDA graph decode for Qwen3.5 batches of 1 to 32 rows.
//!
//! A decode step on CUDA is ~600-900 small launches; eager Candle pays host
//! time for each, so the step is bound by the host, not the GPU. Here the
//! whole step is captured once per batch width and replayed: the token
//! embeddings, every layer, the final norm and the LM head. Every per-step
//! value lives in a device **step buffer** (U32): each row's token id,
//! position and flat KV slot, then the decode metadata `[context[rows] |
//! first-page offset[rows] | block table[rows x max_blocks]]`.
//! - Projections use the izwi-owned K-quant MMVQ (up to eight rows) and MMQ
//!   (`kernels/cuda/kquant.rs`); Candle's `QMatMul` keeps global scratch
//!   buffers and cannot be captured.
//! - DeltaNet layers run the multi-row decode kernels through one device
//!   address table per width (`gdn::RowsTable`, layer-major). Each step
//!   allocates one slab of fresh state outputs per row; committed state is
//!   never mutated in place.
//! - Attention runs `rope::qk_norm_rope_kv_rows` (q/k norm, RoPE and each
//!   row's K/V slot write) and `split_decode::paged_decode_adaptive`, which
//!   read the step buffer. Both take 16-bit activations, so q/k/v are cast to
//!   the KV storage dtype (F16 on CUDA) inside the graph.
//!
//! **Widths.** A step of `n` rows replays the graph of the smallest width in
//! [`WIDTHS`] that holds it. The spare rows decode token 0 at position 0 into
//! a KV slot past every cache (the write skips it), attend one token of page
//! 0, and read and write a scratch DeltaNet state of their width's. Every op
//! of the step is row-independent (per-row norms, projections, recurrences
//! and attention), so a spare row changes no real row's values; its outputs
//! are dropped.
//!
//! Phases, per width: warm (eager, checked against the standard decode on a
//! copy of the rows' state), capture + verify, verify again on fresh inputs
//! (catches values a capture baked in), then replay. The step always
//! continues on the eager result until replay. Any failure turns every width
//! off for the process and the step runs the standard decode.
//!
//! Off CUDA there is no capture: with `allow_eager` (tests) every step runs
//! the same body eagerly over its own rows with the arena's attention and
//! explicit state tensors, which checks the composition against the standard
//! decode.
//!
//! On by default on CUDA: `IZWI_QWEN35_CUDA_GRAPH=0` turns it off, and so
//! does `cuda.decode_graphs=off`.
use std::collections::BTreeMap;
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

/// Kill switch (a boolean; on when unset).
pub(super) const GRAPH_ENV: &str = "IZWI_QWEN35_CUDA_GRAPH";

/// The graphed batch widths. Each keeps its own graph, step buffer and table
/// (about a megabyte of graph memory per row for the retained logits), so a
/// step pads up to the next width rather than every width being captured.
const WIDTHS: [usize; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

/// The widest batch a graph serves: the CUDA chat width.
pub(super) const MAX_GRAPH_ROWS: usize = WIDTHS[WIDTHS.len() - 1];

/// The width whose graph serves a step of `rows` rows.
fn graph_width(rows: usize) -> Option<usize> {
    WIDTHS.iter().copied().find(|width| *width >= rows.max(1))
}

/// A spare row's flat KV slot: past every cache, so the write skips it.
const SPARE_SLOT: u32 = u32::MAX;

/// Logit agreement the warm step demands of the graph body against the
/// standard decode. They differ in attention precision (16-bit q/k norm
/// inputs) and summation order, so this catches composition bugs, not
/// rounding.
const WARM_CHECK_REL_L2: f64 = 0.05;
const WARM_CHECK_MAX_FRACTION: f32 = 0.1;

fn graph_requested() -> bool {
    crate::env::flag_or(GRAPH_ENV, true)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Warm,
    Capture,
    Verify,
    Replay,
}

impl Phase {
    fn name(&self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Capture => "capture",
            Self::Verify => "verify",
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
}

impl Runtime {
    fn new() -> Self {
        Self {
            phase: Phase::Warm,
            graph: SegmentGraph::default(),
            table: None,
            step: None,
            captured_for: None,
        }
    }

    /// Drop the captured graph: the next step at this width captures again.
    fn recapture(&mut self) {
        self.graph.reset();
        if self.phase != Phase::Warm {
            self.phase = Phase::Capture;
        }
    }
}

/// Every width's runtime, and why graph decode was turned off, if it was.
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
        tracing::warn!(%reason, "Qwen3.5 CUDA graph decode disabled; using the standard decode");
        self.disabled = Some(reason);
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

/// Graph decode for one model (see the module docs).
pub(super) struct Qwen35DecodeGraph {
    /// Per layer: the DeltaNet ordinal (the layer's block of every table).
    linear_ordinal: Vec<Option<usize>>,
    linear_count: usize,
    /// Full-attention layers (arena bindings `0..attention`).
    attention: usize,
    /// Longest context the step buffer's block tables must cover.
    max_context: usize,
    /// Capture graphs (CUDA); otherwise the body runs eagerly every step.
    capture: bool,
    graphs: Mutex<Graphs>,
    warmups: AtomicU64,
    captures: AtomicU64,
    replays: AtomicU64,
    verified: AtomicU64,
    standard_steps: AtomicU64,
    /// Rows replayed (or run) as padding past the step's own rows.
    spare_rows: AtomicU64,
    graph_rows: AtomicU64,
}

/// One step's rows, all on one KV arena.
pub(super) struct StepRows<'a> {
    pub(super) tokens: &'a [u32],
    pub(super) positions: &'a [[usize; 3]],
    pub(super) slots: &'a [KvSlotRef],
    pub(super) caches: &'a [&'a PhysicalPagedKvCache],
    pub(super) slot_map: &'a dyn KvSlotMap,
    pub(super) metadata: &'a KvDecodeBatchMetadata,
}

impl StepRows<'_> {
    fn len(&self) -> usize {
        self.tokens.len()
    }
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
            return refuse(format!("{GRAPH_ENV} is off"));
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
                "CUDA graph decode needs the K-quant MMVQ/MMQ on every projection (QMatMul cannot be captured)"
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
                standard_steps: AtomicU64::new(0),
                spare_rows: AtomicU64::new(0),
                graph_rows: AtomicU64::new(0),
            }),
            Qwen35FastPath::Fused,
        )
    }

    /// Diagnostics: each width's phase, the counters and the reason graph
    /// decode was turned off at runtime, if it was.
    pub(super) fn summary(&self) -> serde_json::Value {
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
            "widths": widths,
            "warmups": self.warmups.load(Relaxed),
            "captures": self.captures.load(Relaxed),
            "replays": self.replays.load(Relaxed),
            "verified": self.verified.load(Relaxed),
            "standard_steps": self.standard_steps.load(Relaxed),
            "graph_rows": self.graph_rows.load(Relaxed),
            "spare_rows": self.spare_rows.load(Relaxed),
        })
    }

    /// Run one decode step of `rows` through the graph of its width.
    /// `Ok(None)` hands the step to the standard decode: graph decode is off
    /// or busy, the batch is wider than every graph, a position is
    /// multimodal, the rows span arenas, or a state is not in fused-decode
    /// form. `reference` computes the standard decode's logits for the warm
    /// check. Returns logits `[rows, 1, vocab]`.
    pub(super) fn forward(
        &self,
        model: &Qwen35TextModel,
        rows: &StepRows<'_>,
        states: &mut [&mut Qwen35TextRuntimeState],
        completions: &mut KvWriteCompletionCollector,
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<Option<Tensor>> {
        let n = rows.len();
        if n == 0
            || states.len() != n
            || rows.positions.len() != n
            || rows.slots.len() != n
            || rows.caches.len() != n
            || rows.metadata.sequences.len() != n
        {
            return Err(graph_error("the step's rows do not match"));
        }
        let standard = || -> Result<Option<Tensor>> {
            self.standard_steps.fetch_add(1, Relaxed);
            Ok(None)
        };
        let Some(width) = graph_width(n) else {
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
        Ok(Some(done.logits))
    }

    /// Eager devices: the body over the step's own rows with the arena's
    /// attention, every step.
    fn eager(
        &self,
        model: &Qwen35TextModel,
        rows: &StepRows<'_>,
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
            .map(|_| StateOutputs::allocate(model, &device))
            .collect::<Result<Vec<_>>>()?;
        let ios = direct_io(model, inputs, &published);
        let rope =
            model.rope_table(model.first_attention()?, rows.positions, (n, 1), DType::F32)?;
        let mut attention = Attention::Arena {
            rope: rope.as_ref(),
            caches: rows.caches,
            slots: rows.slot_map,
            metadata: rows.metadata,
            completions,
        };
        self.warmups.fetch_add(1, Relaxed);
        let logits = body(model, &step.tensor, n, &ios, &mut attention)?;
        self.graph_rows.fetch_add(n as u64, Relaxed);
        Ok(StepDone { logits, published })
    }

    /// CUDA: warm, capture and verify, or replay the graph of `width` rows,
    /// inside the arena's graph-write bookkeeping.
    fn captured(
        &self,
        graphs: &mut Graphs,
        width: usize,
        model: &Qwen35TextModel,
        rows: &StepRows<'_>,
        inputs: &[StateInputs],
        reference: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<(StepDone, Vec<KvWriteCompletion>)> {
        let n = rows.len();
        let device = model.device.clone();
        let cache = rows.caches[0];
        let arena = cache.arena().clone();
        let config = arena.config();
        if config.dtype != DType::F16 && config.dtype != DType::BF16 {
            return Err(graph_error("graph attention needs a 16-bit KV cache"));
        }
        let kv_dtype = config.dtype;
        let page_tokens = config.page_tokens as usize;
        let layout = StepLayout {
            rows: width,
            max_blocks: self.max_blocks(cache)?,
        };
        let Graphs { widths, spare, .. } = graphs;
        if width > n && spare.is_none() {
            *spare = Some(StateOutputs::allocate(model, &device)?);
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
            .map(|_| StateOutputs::allocate(model, &device))
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
        let step_buffer = runtime.step.as_ref().expect("step buffer allocated");
        let step = step_buffer.tensor.clone();
        let positions = step.narrow(0, layout.positions(), width)?;
        let slots = step.narrow(0, layout.slots(), width)?;
        let metadata = step.narrow(0, layout.metadata(), layout.metadata_len())?;
        let max_partitions = split_decode::partitions(
            width,
            model.first_attention()?.num_kv_heads,
            self.max_context,
        );
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
                    kv_dtype,
                    page_tokens,
                    max_blocks: layout.max_blocks,
                    max_partitions,
                };
                let run_body = |attention: &mut Attention<'_>, step: &Tensor| {
                    body(model, step, width, &ios, attention)
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
                        let eager = body(model, &step, width, &ios, &mut attention)?;
                        self.verify(
                            runtime, spare, model, width, inputs, &published, &step, &eager,
                        )?;
                        self.verified.fetch_add(1, Relaxed);
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
        let logits = output.ok_or_else(|| graph_error("the step produced no output"))?;
        let logits = if logits.dim(0)? == n {
            logits
        } else {
            logits.narrow(0, 0, n)?
        };
        if let Some(expected) = expected {
            compare_values(
                "graph decode against the standard decode",
                &host(&logits)?,
                &host(&expected)?,
                WARM_CHECK_REL_L2,
                WARM_CHECK_MAX_FRACTION,
            )?;
        }
        runtime.phase = match runtime.phase {
            Phase::Warm => Phase::Capture,
            Phase::Capture => Phase::Verify,
            Phase::Verify | Phase::Replay => Phase::Replay,
        };
        self.graph_rows.fetch_add(n as u64, Relaxed);
        self.spare_rows.fetch_add((width - n) as u64, Relaxed);
        Ok((StepDone { logits, published }, completions))
    }

    /// Replay into scratch DeltaNet outputs and compare with the eager run,
    /// over the step's own rows.
    #[allow(clippy::too_many_arguments)]
    fn verify(
        &self,
        runtime: &mut Runtime,
        spare: Option<&StateOutputs>,
        model: &Qwen35TextModel,
        width: usize,
        inputs: &[StateInputs],
        published: &[StateOutputs],
        step: &Tensor,
        eager: &Tensor,
    ) -> Result<()> {
        let n = inputs.len();
        let device = model.device.clone();
        let scratch = (0..n)
            .map(|_| StateOutputs::allocate(model, &device))
            .collect::<Result<Vec<_>>>()?;
        let scratch_entries = self.table_values(width, inputs, &scratch, spare)?;
        let published_entries = self.table_values(width, inputs, published, spare)?;
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
            &host(&replayed.narrow(0, 0, n)?)?,
            &host(&eager.narrow(0, 0, n)?)?,
            0.01,
            0.02,
        )?;
        for (row, (graph_row, eager_row)) in scratch.iter().zip(published).enumerate() {
            for (layer, (graph_state, eager_state)) in
                graph_row.layers.iter().zip(&eager_row.layers).enumerate()
            {
                let (Some(graph_state), Some(eager_state)) = (graph_state, eager_state) else {
                    continue;
                };
                compare_values(
                    &format!("graph decode row {row} layer {layer} conv history"),
                    &host(&graph_state.history)?,
                    &host(&eager_state.history)?,
                    1e-6,
                    1e-6,
                )?;
                compare_values(
                    &format!("graph decode row {row} layer {layer} recurrent state"),
                    &host(&graph_state.state)?,
                    &host(&eager_state.state)?,
                    0.01,
                    0.02,
                )?;
            }
        }
        Ok(())
    }

    /// Block-table capacity per row: the longest context in pages, plus one
    /// for a first-page offset.
    fn max_blocks(&self, cache: &PhysicalPagedKvCache) -> Result<usize> {
        let page_tokens = cache.arena().config().page_tokens as usize;
        if page_tokens == 0 {
            return Err(graph_error("the KV arena has zero-token pages"));
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
                values.extend(table_entries(input, output, layer)?);
            }
            if width > inputs.len() {
                let spare = spare.ok_or_else(|| graph_error("spare rows have no state"))?;
                let entries = spare_entries(spare, layer)?;
                for _ in inputs.len()..width {
                    values.extend(entries);
                }
            }
        }
        Ok(values)
    }

    /// Table-addressed IO for every DeltaNet layer of a `width`-row step.
    fn table_io(&self, width: usize, table: &DeviceTable) -> Result<Vec<Option<LinearStateIo>>> {
        #[cfg(feature = "cuda")]
        {
            Ok(self
                .linear_ordinal
                .iter()
                .map(|ordinal| {
                    ordinal.map(|ordinal| {
                        LinearStateIo::Table(gdn::RowsTable::of(
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
            Err(graph_error("device tables need CUDA"))
        }
    }
}

/// A finished step: logits and each row's DeltaNet state to publish.
struct StepDone {
    logits: Tensor,
    published: Vec<StateOutputs>,
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
        // Spare rows attend one token: a zero context would leave their
        // split-decode partitions empty.
        let mut head = vec![0u32; layout.head_len()];
        for row in 0..layout.rows {
            head[layout.slots() + row] = SPARE_SLOT;
            head[layout.metadata() + row] = 1;
        }
        buffer.update(0, &head)?;
        Ok(buffer)
    }

    /// Write `rows` into the buffer: each row's token, position, flat slot,
    /// context, first-page offset and block table; the rows past them stay
    /// spare.
    fn write(&mut self, rows: &StepRows<'_>) -> Result<()> {
        let layout = self.layout;
        let n = rows.len();
        if n > layout.rows {
            return Err(graph_error("the step has more rows than its width"));
        }
        let page_tokens = rows.caches[0].arena().config().page_tokens;
        let mut head = vec![0u32; layout.head_len()];
        for row in 0..layout.rows {
            head[layout.slots() + row] = SPARE_SLOT;
            head[layout.metadata() + row] = 1;
        }
        for row in 0..n {
            let sequence = &rows.metadata.sequences[row];
            if sequence.blocks.len() > layout.max_blocks {
                return Err(graph_error("a row's block table exceeds the step buffer"));
            }
            let slot = rows.slots[row];
            head[row] = rows.tokens[row];
            head[layout.positions() + row] = u32::try_from(rows.positions[row][0])
                .map_err(|_| graph_error("decode position exceeds u32"))?;
            head[layout.slots() + row] = slot
                .block
                .index
                .checked_mul(page_tokens)
                .and_then(|base| base.checked_add(slot.offset))
                .filter(|flat| *flat != SPARE_SLOT)
                .ok_or_else(|| graph_error("KV slot index overflow"))?;
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

/// Write `values` into the U32 `tensor` at `offset`, in place.
fn upload(tensor: &Tensor, offset: usize, values: &[u32]) -> Result<()> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = tensor.device() {
        use candle_core::cuda_backend::cudarc::driver::{result, DevicePtr};
        let (storage, layout) = tensor.storage_and_layout();
        let candle_core::Storage::Cuda(storage) = &*storage else {
            return Err(graph_error("the step buffer is not a CUDA tensor"));
        };
        if offset + values.len() > tensor.elem_count() {
            return Err(graph_error("a step buffer write past its end"));
        }
        let stream = cuda.cuda_stream();
        let (base, _guard) = storage.as_cuda_slice::<u32>()?.device_ptr(&stream);
        let destination = base + ((layout.start_offset() + offset) * 4) as u64;
        stream
            .context()
            .bind_to_thread()
            .map_err(|error| graph_error(&format!("step buffer bind: {error}")))?;
        // SAFETY: the range lies inside the buffer (checked above), and a
        // pageable source is staged before the call returns.
        unsafe { result::memcpy_htod_async(destination, values, stream.cu_stream()) }
            .map_err(|error| graph_error(&format!("step buffer upload: {error}")))?;
        return Ok(());
    }
    let update = Tensor::from_slice(values, values.len(), tensor.device())?;
    tensor.slice_set(&update, 0, offset)?;
    Ok(())
}

/// Where one DeltaNet layer reads its rows' state and writes the next one.
enum LinearStateIo {
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

/// Attention for the step body.
enum Attention<'a> {
    /// The fused q/k RoPE + KV write and the adaptive split decode, both
    /// driven by the step buffer: capturable.
    Graph {
        caches: &'a [(Tensor, Tensor)],
        positions: &'a Tensor,
        slots: &'a Tensor,
        metadata: &'a Tensor,
        kv_dtype: DType,
        page_tokens: usize,
        max_blocks: usize,
        max_partitions: usize,
    },
    /// The arena's own write and decode (eager devices, tests).
    Arena {
        rope: Option<&'a super::Qwen35RopeTable>,
        caches: &'a [&'a PhysicalPagedKvCache],
        slots: &'a dyn KvSlotMap,
        metadata: &'a KvDecodeBatchMetadata,
        completions: &'a mut KvWriteCompletionCollector,
    },
}

/// The step body over `rows` rows: the embeddings of the step buffer's
/// tokens, every layer, the final norm and the LM head. Returns logits
/// `[rows, 1, vocab]`, as the standard decode does.
fn body(
    model: &Qwen35TextModel,
    step: &Tensor,
    rows: usize,
    ios: &[Option<LinearStateIo>],
    attention: &mut Attention<'_>,
) -> Result<Tensor> {
    let ids = step.narrow(0, 0, rows)?.reshape((rows, 1))?;
    let mut hidden = model.token_table.embed(&ids)?;
    let mut physical = 0usize;
    for (index, layer) in model.layers.iter().enumerate() {
        let residual = hidden.clone();
        let normalized = layer.attn_norm.forward(&hidden)?;
        let mixed = match &layer.mixer {
            Qwen35Mixer::Linear(mixer) => {
                let io = ios[index]
                    .as_ref()
                    .ok_or_else(|| graph_error("DeltaNet layer has no state IO"))?;
                mixer.forward_graph(&normalized, io, rows)?
            }
            Qwen35Mixer::Full(full) => {
                let mixed = attend(full, &normalized, attention, physical, rows)?;
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
    rows: usize,
) -> Result<Tensor> {
    let [q_out, k_out, v_out] =
        forward_shared(normalized, [&full.q_proj, &full.k_proj, &full.v_proj])?;
    let attended = match attention {
        Attention::Graph {
            caches,
            positions,
            slots,
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
            let q_proj = q_out
                .to_dtype(*kv_dtype)?
                .reshape((rows, full.num_heads * full.head_dim * 2))?;
            let keys = k_out
                .to_dtype(*kv_dtype)?
                .reshape((rows, full.num_kv_heads * full.head_dim))?;
            let values = v_out
                .to_dtype(*kv_dtype)?
                .reshape((rows, full.num_kv_heads * full.head_dim))?;
            let queries = rope::qk_norm_rope_kv_rows(
                &q_proj,
                &keys,
                &values,
                &qk.q_gain,
                &qk.k_gain,
                &qk.inv_freq,
                &qk.spec,
                positions,
                slots,
                cache_keys,
                cache_values,
                rows,
            )?;
            split_decode::paged_decode_adaptive(
                &queries.reshape((rows, full.num_heads, full.head_dim))?,
                cache_keys,
                cache_values,
                metadata,
                rows,
                full.num_heads,
                full.num_kv_heads,
                *page_tokens,
                *max_blocks,
                1.0 / (full.head_dim as f32).sqrt(),
                None,
                *max_partitions,
            )?
            .reshape((rows, 1, full.num_heads * full.head_dim))?
            .to_dtype(q_out.dtype())?
        }
        Attention::Arena {
            rope,
            caches,
            slots,
            metadata,
            completions,
        } => full.decode_attend(
            &q_out,
            &k_out,
            &v_out,
            *rope,
            caches,
            *slots,
            metadata,
            completions,
            physical,
        )?,
    };
    full.decode_output(&attended, &q_out)
}

impl Qwen35LinearAttention {
    /// `rows` tokens through the fused DeltaNet kernels with explicit state
    /// IO: the graph form of the batched fused decode.
    fn forward_graph(
        &self,
        hidden_states: &Tensor,
        io: &LinearStateIo,
        rows: usize,
    ) -> Result<Tensor> {
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
        let value_width = self.num_v_heads * self.head_v_dim;
        let y = match io {
            #[cfg(feature = "cuda")]
            LinearStateIo::Table(table) => gdn::decode_rows_in(
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
            LinearStateIo::Direct(direct) => {
                if direct.len() != rows {
                    return Err(graph_error("DeltaNet state IO does not cover every row"));
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

/// Explicit-tensor IO for every DeltaNet layer of every row (devices
/// without tables).
fn direct_io(
    model: &Qwen35TextModel,
    inputs: &[StateInputs],
    outputs: &[StateOutputs],
) -> Vec<Option<LinearStateIo>> {
    (0..model.layers.len())
        .map(|layer| {
            inputs
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
                .collect::<Option<Vec<_>>>()
                .map(LinearStateIo::Direct)
        })
        .collect()
}

/// Table entries for `layer` of one row: h0, h1, h2, next history, state in,
/// state out.
#[cfg(feature = "cuda")]
fn table_entries(
    input: &StateInputs,
    output: &StateOutputs,
    layer: usize,
) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    let (Some(input), Some(output)) = (input.layers[layer].as_ref(), output.layers[layer].as_ref())
    else {
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

/// A spare row's entries for `layer`: it reads and writes the spare slab in
/// place, three history slots from its history rows.
#[cfg(feature = "cuda")]
fn spare_entries(spare: &StateOutputs, layer: usize) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    let Some(spare) = spare.layers[layer].as_ref() else {
        return Err(graph_error("DeltaNet layer has no spare state"));
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
fn table_entries(
    _input: &StateInputs,
    _output: &StateOutputs,
    _layer: usize,
) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    Err(graph_error("device tables need CUDA"))
}

#[cfg(not(feature = "cuda"))]
fn spare_entries(_spare: &StateOutputs, _layer: usize) -> Result<[u64; gdn::TABLE_ENTRIES]> {
    Err(graph_error("device tables need CUDA"))
}

/// An F32 buffer whose every element the step's kernels write.
pub(super) fn uninit_f32(len: usize, device: &Device) -> Result<Tensor> {
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
    fn graph_decode_is_on_unless_switched_off() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var(GRAPH_ENV);
        assert!(graph_requested());
        for (value, expected) in [
            ("1", true),
            (" ON ", true),
            ("true", true),
            ("0", false),
            ("off", false),
            ("no", false),
        ] {
            std::env::set_var(GRAPH_ENV, value);
            assert_eq!(graph_requested(), expected, "{value}");
        }
        std::env::remove_var(GRAPH_ENV);
    }

    #[test]
    fn steps_pad_up_to_the_next_graphed_width() {
        let widths = (1..=33).map(graph_width).collect::<Vec<_>>();
        assert_eq!(widths[..8], [1, 2, 3, 4, 6, 6, 8, 8].map(Some));
        assert_eq!(widths[8], Some(12)); // 9 rows
        assert_eq!(widths[12], Some(16)); // 13 rows
        assert_eq!(widths[16], Some(24)); // 17 rows
        assert_eq!(widths[31], Some(32)); // 32 rows
        assert_eq!(widths[32], None); // 33 rows run the standard decode
        assert_eq!(MAX_GRAPH_ROWS, 32);
    }

    #[test]
    fn the_step_layout_packs_rows_then_metadata() {
        let layout = StepLayout {
            rows: 3,
            max_blocks: 5,
        };
        assert_eq!(layout.positions(), 3);
        assert_eq!(layout.slots(), 6);
        assert_eq!(layout.metadata(), 9);
        assert_eq!(layout.head_len(), 15);
        assert_eq!(layout.blocks(0), 15);
        assert_eq!(layout.blocks(2), 25);
        assert_eq!(layout.metadata_len(), 21);
        assert_eq!(layout.len(), 30);
    }

    #[test]
    fn a_new_step_buffer_holds_spare_rows() {
        let layout = StepLayout {
            rows: 2,
            max_blocks: 3,
        };
        let buffer = StepBuffer::new(&Device::Cpu, layout).unwrap();
        let values = buffer.tensor.to_vec1::<u32>().unwrap();
        assert_eq!(values, buffer.mirror);
        // Slots past every cache, one-token contexts on page 0.
        assert_eq!(values[layout.slots()..layout.metadata()], [SPARE_SLOT; 2]);
        assert_eq!(values[layout.metadata()..layout.metadata() + 2], [1, 1]);
        assert!(values[layout.head_len()..].iter().all(|v| *v == 0));
    }
}
