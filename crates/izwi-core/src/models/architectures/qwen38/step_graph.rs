//! Whole-step CUDA graph decode for one Qwen3.8 row.
//!
//! The piecewise segments leave attention eager between 17 captured graphs:
//! its q/k RoPE takes the position as a kernel scalar, and the paged KV write
//! and attention read host-built slot maps and metadata. Here every per-step
//! value lives in a device **step buffer** (U32): the token id, the position,
//! the flat KV slot and the decode metadata `[context, first-page offset,
//! block table]`. Attention reads it through `rope::qk_norm_rope_kv` (fused
//! q/k norm, partial RoPE and the K/V slot write) and `split_decode::
//! paged_decode_adaptive`, so the whole step is captured once: the token
//! embedding, all 64 layers, the final norm and (when it is block FP8) the
//! LM head. A step is then one host-to-device update of the step buffer and
//! the DeltaNet state table, plus one graph launch.
//!
//! The KV writes go through `KvArena::with_graph_writes`, which keeps the
//! arena's write bookkeeping (dirty pages, completions) and serializes the
//! enqueue against growth. The graph is recaptured when the arena's backing
//! generation changes. `with_graph_writes` fences once, after the enqueued
//! step, so sealing the step's writes waits for the whole step: the sync the
//! segmented path takes after the last attention layer's write (before the
//! LM head has run) is gone.
//!
//! Phases mirror the segments: warm (eager), capture + verify, verify,
//! replay. The warm step also runs the standard decode on a copy of the row
//! state and compares logits, which checks the new attention composition on
//! the live arena. Any failure turns the whole-step graph off and the step,
//! and every later one, runs through the segments.
//!
//! Off CUDA the body runs eagerly with the arena's own attention calls, so
//! tests can check it against the standard decode bit for bit.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use candle_core::{DType, Device, Module, Tensor};

use super::super::{
    Qwen38FullAttention, Qwen38Hidden, Qwen38Mixer, Qwen38Projection, Qwen38TextModel,
};
use super::{host, segment_error, LinearStateIo, Phase, StateInputs, StateOutputs};
use crate::backends::kv::{KvSlotMap, KvWriteCompletion, KvWriteCompletionCollector};
use crate::error::Result;
use crate::kernels::cuda::rope;
use crate::kernels::cuda::segment_graph::{DeviceTable, SegmentGraph, SegmentInput};
use crate::kernels::cuda::split_decode;
use crate::kv::{KvArenaId, KvDecodeBatchMetadata, KvSlotRef};
use crate::models::architectures::qwen38::fast_path::{
    compare_values, legacy_requested, STEP_GRAPH_ENV,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// Step buffer layout (U32). `rope::qk_norm_rope_kv` reads the position and
/// the flat slot at indices 1 and 2.
const TOKEN: usize = 0;
const POSITION: usize = 1;
const SLOT: usize = 2;
const METADATA: usize = 4;

/// Logit agreement the warm step demands of the whole-step body against the
/// standard decode. The two differ only in attention summation order (split
/// decode against the arena's provider) and the fused K/V write, so this
/// catches composition bugs, not rounding.
const WARM_CHECK_REL_L2: f64 = 0.05;
const WARM_CHECK_MAX_FRACTION: f32 = 0.1;

/// Load-time plan for the whole-step graph.
pub(super) struct WholeStepPlan {
    /// The LM head runs inside the graph (block FP8; a cuBLAS head stays out).
    head_in_graph: bool,
    /// Longest context the step buffer's block table must cover.
    max_context: usize,
    /// Number of full-attention layers (arena bindings `0..attention`).
    attention: usize,
    /// Capture graphs (CUDA); otherwise the body runs eagerly every step.
    capture: bool,
    warmups: AtomicU64,
    warm_checks: AtomicU64,
    captures: AtomicU64,
    replays: AtomicU64,
    verified: AtomicU64,
    /// KV backing-generation changes that forced a recapture.
    recaptures: AtomicU64,
    /// Steps queued with a device-selected input token (pipelined decode).
    pub(super) pipelined: AtomicU64,
}

/// Runtime state, held under the decode graphs' runtime lock.
pub(super) struct WholeStepRuntime {
    pub(super) phase: Phase,
    graph: SegmentGraph,
    table: Option<DeviceTable>,
    step: Option<StepBuffer>,
    /// The arena and backing generation the graph was captured against.
    captured_for: Option<(KvArenaId, u64)>,
}

impl WholeStepRuntime {
    pub(super) fn new() -> Self {
        Self {
            phase: Phase::Warm,
            graph: SegmentGraph::default(),
            table: None,
            step: None,
            captured_for: None,
        }
    }

    pub(super) fn disable(&mut self, reason: String) {
        self.graph.reset();
        tracing::warn!(%reason, "Qwen3.8 whole-step CUDA graph disabled; using the graph segments");
        self.phase = Phase::Disabled(reason);
    }

    /// Leak the graph and the buffers it may still read (teardown after a
    /// failed step-fence wait).
    #[cfg(feature = "cuda")]
    pub(super) fn leak(&mut self) {
        std::mem::forget(std::mem::take(&mut self.graph));
        std::mem::forget(self.table.take());
        std::mem::forget(self.step.take());
    }
}

impl WholeStepPlan {
    /// The whole-step plan for `model`, or why it does not apply. The decode
    /// graphs' own prerequisites (fused DeltaNet, capturable projections)
    /// are already met.
    pub(super) fn resolve(
        model: &Qwen38TextModel,
        capture: bool,
    ) -> std::result::Result<Self, String> {
        if legacy_requested(STEP_GRAPH_ENV) {
            return Err(format!("{STEP_GRAPH_ENV}=legacy"));
        }
        let dtype = model.activation_dtype();
        let mut attention = 0usize;
        for layer in &model.layers {
            if let Qwen38Mixer::Full(full) = &layer.mixer {
                if full.fused_qk.is_none() || !full.fused_kv_write {
                    return Err(
                        "the whole-step graph needs the fused q/k norm + RoPE with its K/V write"
                            .into(),
                    );
                }
                if capture
                    && !split_decode::supported(
                        dtype,
                        full.num_heads,
                        full.num_kv_heads,
                        full.head_dim,
                        full.head_dim,
                    )
                {
                    return Err("split decode does not cover this attention geometry".into());
                }
                attention += 1;
            }
        }
        if attention == 0 {
            return Err("the model has no full-attention layer".into());
        }
        Ok(Self {
            head_in_graph: matches!(model.output, Qwen38Projection::NativeFp8 { .. }),
            max_context: model.max_context_tokens,
            attention,
            capture,
            warmups: AtomicU64::new(0),
            warm_checks: AtomicU64::new(0),
            captures: AtomicU64::new(0),
            replays: AtomicU64::new(0),
            verified: AtomicU64::new(0),
            recaptures: AtomicU64::new(0),
            pipelined: AtomicU64::new(0),
        })
    }

    pub(super) fn head_in_graph(&self) -> bool {
        self.head_in_graph
    }

    /// Whether a step may take its input token from the step buffer: the
    /// graph replays (its warm check needs the token on the host), or the
    /// body runs eagerly every step.
    pub(super) fn takes_device_tokens(&self, phase: &Phase) -> bool {
        match phase {
            Phase::Disabled(_) => false,
            Phase::Replay => true,
            _ => !self.capture,
        }
    }

    pub(super) fn summary(&self, phase: &Phase) -> serde_json::Value {
        serde_json::json!({
            "phase": phase.name(),
            "disabled_reason": match phase {
                Phase::Disabled(reason) => Some(reason.clone()),
                _ => None,
            },
            "head_in_graph": self.head_in_graph,
            "warmups": self.warmups.load(Relaxed),
            "warm_checks": self.warm_checks.load(Relaxed),
            "captures": self.captures.load(Relaxed),
            "recaptures": self.recaptures.load(Relaxed),
            "replays": self.replays.load(Relaxed),
            "verified": self.verified.load(Relaxed),
            "pipelined": self.pipelined.load(Relaxed),
        })
    }
}

/// One decode row's inputs for the whole step.
pub(super) struct StepRequest<'a> {
    /// The input token, or `None` for the one the previous step's device-side
    /// token selection wrote into the step buffer (pipelined decode).
    pub(super) token: Option<u32>,
    pub(super) position: usize,
    pub(super) slot: KvSlotRef,
    pub(super) cache: &'a PhysicalPagedKvCache,
    pub(super) slots: &'a dyn KvSlotMap,
    pub(super) metadata: &'a KvDecodeBatchMetadata,
}

/// A finished whole step: logits (or the pre-norm hidden when the LM head
/// stays out of the graph) and the DeltaNet state to publish.
pub(super) struct StepDone {
    pub(super) output: Tensor,
    pub(super) published: StateOutputs,
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
            return Err(segment_error("the step buffer is not a CUDA tensor"));
        };
        let stream = cuda.cuda_stream();
        let (base, _guard) = storage.as_cuda_slice::<u32>()?.device_ptr(&stream);
        let destination = base + ((layout.start_offset() + offset) * 4) as u64;
        // SAFETY: the context is this device's and outlives the call.
        unsafe { result::ctx::set_current(stream.context().cu_ctx()) }
            .map_err(|error| segment_error(&format!("step buffer bind: {error}")))?;
        // SAFETY: the range lies inside the buffer (callers write within its
        // length), and a pageable source is staged before the call returns.
        unsafe { result::memcpy_htod_async(destination, values, stream.cu_stream()) }
            .map_err(|error| segment_error(&format!("step buffer upload: {error}")))?;
        return Ok(());
    }
    let update = Tensor::from_slice(values, values.len(), tensor.device())?;
    tensor.slice_set(&update, 0, offset)?;
    Ok(())
}

/// Attention for the step body.
enum Attention<'a> {
    /// The fused q/k RoPE + KV write and the adaptive split decode, both
    /// driven by the step buffer: capturable.
    Graph {
        caches: &'a [(Tensor, Tensor)],
        step: &'a Tensor,
        metadata: Tensor,
        page_tokens: usize,
        max_blocks: usize,
        max_partitions: usize,
    },
    /// The arena's own write and decode (eager devices, tests).
    Arena {
        position: [usize; 3],
        cache: &'a PhysicalPagedKvCache,
        slots: &'a dyn KvSlotMap,
        metadata: &'a KvDecodeBatchMetadata,
        completions: &'a mut KvWriteCompletionCollector,
    },
}

/// The step body: embedding of `step[TOKEN]`, every layer, and, with `head`,
/// the final norm and the LM head. Returns logits `[1, 1, vocab]` or the
/// pre-norm hidden `[1, 1, hidden]`. The op sequence is the standard
/// decode's, attention aside.
fn body(
    model: &Qwen38TextModel,
    step: &Tensor,
    ios: &[Option<LinearStateIo>],
    attention: &mut Attention<'_>,
    head: bool,
) -> Result<Tensor> {
    let ids = step.narrow(0, TOKEN, 1)?.reshape((1, 1))?;
    let mut hidden = Qwen38Hidden::new(model.token_embeddings.forward(&ids)?);
    let mut physical = 0usize;
    for (index, layer) in model.layers.iter().enumerate() {
        let (residual, normalized) = hidden.normalized(&layer.attn_norm)?;
        let mixed = match &layer.mixer {
            Qwen38Mixer::Linear(mixer) => {
                let io = ios[index]
                    .as_ref()
                    .ok_or_else(|| segment_error("DeltaNet layer has no state IO"))?;
                mixer.forward_segment(&normalized, io)?
            }
            Qwen38Mixer::Full(full) => {
                let mixed = attend(full, &normalized, attention, physical)?;
                physical += 1;
                mixed
            }
        };
        hidden = layer.post_mixer(&residual, &mixed)?;
    }
    let hidden = hidden.materialize()?;
    if head {
        model.project_target_hidden_span(&hidden)
    } else {
        Ok(hidden)
    }
}

fn attend(
    full: &Qwen38FullAttention,
    normalized: &Tensor,
    attention: &mut Attention<'_>,
    physical: usize,
) -> Result<Tensor> {
    let (q_proj, keys, values) = full.decode_projections(normalized)?;
    let attended = match attention {
        Attention::Graph {
            caches,
            step,
            metadata,
            page_tokens,
            max_blocks,
            max_partitions,
        } => {
            let (spec, inv_freq) = full
                .fused_qk
                .as_ref()
                .ok_or_else(|| segment_error("the whole-step graph needs the fused q/k RoPE"))?;
            let (cache_keys, cache_values) = caches
                .get(physical)
                .ok_or_else(|| segment_error("no KV cache for an attention layer"))?;
            let queries = rope::qk_norm_rope_kv(
                &q_proj,
                &keys,
                &values,
                &full.q_norm.weight,
                &full.k_norm.weight,
                inv_freq,
                spec,
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
        }
        Attention::Arena {
            position,
            cache,
            slots,
            metadata,
            completions,
        } => full.decode_attend(
            &q_proj,
            &keys,
            &values,
            &[*position],
            &[*cache],
            *slots,
            metadata,
            completions,
            physical,
        )?,
    };
    full.decode_output(&attended, &q_proj)
}

/// Run one step through the whole-step graph. `Ok(None)` hands the step to
/// the segments: the whole-step graph is off, or it failed here (which turns
/// it off; nothing it computed is used). An error is the eager body's own
/// (devices without capture).
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    plan: &WholeStepPlan,
    runtime: &mut WholeStepRuntime,
    graphs: &super::Qwen38DecodeGraphs,
    model: &Qwen38TextModel,
    request: &StepRequest<'_>,
    inputs: &StateInputs,
    state_completions: &mut KvWriteCompletionCollector,
    reference: &mut dyn FnMut() -> Result<Tensor>,
) -> Result<Option<StepDone>> {
    if matches!(runtime.phase, Phase::Disabled(_)) {
        return Ok(None);
    }
    if !plan.capture {
        return eager(
            plan,
            runtime,
            graphs,
            model,
            request,
            inputs,
            state_completions,
        )
        .map(Some);
    }
    match captured(plan, runtime, graphs, model, request, inputs, reference) {
        Ok((done, completions)) => {
            for completion in completions {
                state_completions.collect(completion)?;
            }
            Ok(Some(done))
        }
        Err(error) => {
            runtime.disable(format!("whole-step graph failed: {error}"));
            Ok(None)
        }
    }
}

/// Eager devices: the body with the arena's attention, every step.
fn eager(
    plan: &WholeStepPlan,
    runtime: &mut WholeStepRuntime,
    graphs: &super::Qwen38DecodeGraphs,
    model: &Qwen38TextModel,
    request: &StepRequest<'_>,
    inputs: &StateInputs,
    completions: &mut KvWriteCompletionCollector,
) -> Result<StepDone> {
    let device = model.device.clone();
    let max_blocks = max_blocks(plan, request.cache)?;
    let step = step_buffer(runtime, &device, max_blocks, request)?;
    let published = StateOutputs::allocate(model, &device, None)?;
    let ios = graphs.direct_io(model, inputs, &published)?;
    let mut attention = Attention::Arena {
        position: [request.position; 3],
        cache: request.cache,
        slots: request.slots,
        metadata: request.metadata,
        completions,
    };
    plan.warmups.fetch_add(1, Relaxed);
    let output = body(model, &step, &ios, &mut attention, plan.head_in_graph)?;
    Ok(StepDone { output, published })
}

/// CUDA: warm, capture and verify, or replay, inside the arena's graph-write
/// bookkeeping.
fn captured(
    plan: &WholeStepPlan,
    runtime: &mut WholeStepRuntime,
    graphs: &super::Qwen38DecodeGraphs,
    model: &Qwen38TextModel,
    request: &StepRequest<'_>,
    inputs: &StateInputs,
    reference: &mut dyn FnMut() -> Result<Tensor>,
) -> Result<(StepDone, Vec<KvWriteCompletion>)> {
    let device = model.device.clone();
    let arena = request.cache.arena().clone();
    let page_tokens = arena.config().page_tokens as usize;
    let max_blocks = max_blocks(plan, request.cache)?;
    // The warm step checks the body against the standard decode first: the
    // standard path calls the arena, which `with_graph_writes` serializes.
    let expected = if runtime.phase == Phase::Warm {
        Some(reference()?)
    } else {
        None
    };
    let step = step_buffer(runtime, &device, max_blocks, request)?;
    let published = StateOutputs::allocate(model, &device, None)?;
    let entries = graphs.table_values(inputs, &published)?;
    if runtime.table.as_ref().map(DeviceTable::entries) != Some(entries.len()) {
        runtime.table = Some(DeviceTable::new(&device, entries.len())?);
        runtime.graph.reset();
        runtime.phase = Phase::Warm;
    }
    let table = runtime.table.as_mut().expect("table allocated");
    table.write(0, &entries)?;
    let ios = graphs.table_io(table.address())?;
    let bindings = (0..plan.attention)
        .map(|layer| request.cache.layer_binding(layer))
        .collect::<Result<Vec<_>>>()?;
    let metadata = runtime
        .step
        .as_ref()
        .expect("step buffer allocated")
        .metadata()?;
    let kv_heads = model
        .full_attention_layers()
        .next()
        .ok_or_else(|| segment_error("the model has no full-attention layer"))?
        .num_kv_heads;
    let max_partitions = split_decode::partitions(1, kv_heads, plan.max_context);
    let mut output = None;
    let completions = arena.with_graph_writes(
        &bindings,
        request.slots,
        plan.attention as u64,
        &mut |caches| {
            let key = (arena.id(), caches.generation);
            if runtime.captured_for != Some(key) {
                runtime.graph.reset();
                if runtime.phase != Phase::Warm {
                    if runtime.captured_for.is_some() {
                        plan.recaptures.fetch_add(1, Relaxed);
                    }
                    runtime.phase = Phase::Capture;
                }
                runtime.captured_for = Some(key);
            }
            let mut attention = Attention::Graph {
                caches: &caches.layers,
                step: &step,
                metadata: metadata.clone(),
                page_tokens,
                max_blocks,
                max_partitions,
            };
            let run_body = |attention: &mut Attention<'_>, step: &Tensor| {
                body(model, step, &ios, attention, plan.head_in_graph)
                    .map(|out| vec![out])
                    .map_err(candle_core::Error::wrap)
            };
            let out = match runtime.phase.clone() {
                Phase::Warm => {
                    plan.warmups.fetch_add(1, Relaxed);
                    let out = SegmentGraph::warm(&device, std::slice::from_ref(&step), |inputs| {
                        run_body(&mut attention, &inputs[0])
                    })?;
                    out.into_iter().next()
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
                        plan.captures.fetch_add(1, Relaxed);
                    }
                    let eager = body(model, &step, &ios, &mut attention, plan.head_in_graph)?;
                    verify(
                        runtime, graphs, model, inputs, &published, &device, &step, &eager,
                    )?;
                    plan.verified.fetch_add(1, Relaxed);
                    Some(eager)
                }
                Phase::Replay => {
                    let out = runtime.graph.replay(&[SegmentInput {
                        tensor: &step,
                        adopt: true,
                    }])?;
                    plan.replays.fetch_add(1, Relaxed);
                    // The retained output is overwritten by the next replay.
                    out.into_iter().next().map(|out| out.copy()).transpose()?
                }
                Phase::Disabled(_) => None,
            };
            output = out;
            Ok(())
        },
    )?;
    let checked = (|| -> Result<Tensor> {
        let output = output.ok_or_else(|| segment_error("the whole step produced no output"))?;
        if let Some(expected) = expected {
            let logits = if plan.head_in_graph {
                output.clone()
            } else {
                model.project_target_hidden_span(&output)?
            };
            compare_values(
                "whole-step decode against the standard decode",
                &host(&logits)?,
                &host(&expected)?,
                WARM_CHECK_REL_L2,
                WARM_CHECK_MAX_FRACTION,
            )?;
            plan.warm_checks.fetch_add(1, Relaxed);
        }
        Ok(output)
    })();
    let output = match checked {
        Ok(output) => output,
        Err(error) => {
            // Nothing from this step is used, but its writes are in flight:
            // wait for them before the segments rewrite the slot.
            for completion in &completions {
                let _ = completion.wait();
            }
            return Err(error);
        }
    };
    runtime.phase = runtime.phase.clone().advance(true);
    Ok((StepDone { output, published }, completions))
}

/// Replay into scratch DeltaNet outputs and compare with the eager run.
#[allow(clippy::too_many_arguments)]
fn verify(
    runtime: &mut WholeStepRuntime,
    graphs: &super::Qwen38DecodeGraphs,
    model: &Qwen38TextModel,
    inputs: &StateInputs,
    published: &StateOutputs,
    device: &Device,
    step: &Tensor,
    eager: &Tensor,
) -> Result<()> {
    let scratch = StateOutputs::allocate(model, device, None)?;
    let table = runtime
        .table
        .as_mut()
        .ok_or_else(|| segment_error("whole-step verification without a state table"))?;
    table.write(0, &graphs.table_values(inputs, &scratch)?)?;
    let replayed = runtime.graph.replay(&[SegmentInput {
        tensor: step,
        adopt: true,
    }])?;
    table.write(0, &graphs.table_values(inputs, published)?)?;
    let replayed = replayed
        .first()
        .ok_or_else(|| segment_error("the whole-step graph returned nothing"))?;
    compare_values(
        "whole-step graph output",
        &host(replayed)?,
        &host(eager)?,
        0.01,
        0.02,
    )?;
    scratch.compare_with(published, "whole-step")
}

/// Block-table capacity of the step buffer: the longest context in pages,
/// plus one for a first-page offset.
fn max_blocks(plan: &WholeStepPlan, cache: &PhysicalPagedKvCache) -> Result<usize> {
    let page_tokens = cache.arena().config().page_tokens as usize;
    if page_tokens == 0 {
        return Err(segment_error("the KV arena has zero-token pages"));
    }
    Ok(plan.max_context.div_ceil(page_tokens) + 1)
}

/// The step buffer for `request`, allocated (or reallocated for a new
/// block-table capacity, which forces a recapture) and brought up to date.
fn step_buffer(
    runtime: &mut WholeStepRuntime,
    device: &Device,
    max_blocks: usize,
    request: &StepRequest<'_>,
) -> Result<Tensor> {
    let [sequence] = request.metadata.sequences.as_slice() else {
        return Err(segment_error("the whole-step graph decodes one row"));
    };
    if sequence.blocks.len() > max_blocks {
        return Err(segment_error(
            "the row's block table exceeds the step buffer",
        ));
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
        .ok_or_else(|| segment_error("KV slot index overflow"))?;
    let position = u32::try_from(request.position)
        .map_err(|_| segment_error("decode position exceeds u32"))?;
    let step = runtime.step.as_mut().expect("step buffer allocated");
    let mut values = vec![0u32; METADATA + 2 + sequence.blocks.len()];
    // A device-written token is left alone: matching the mirror keeps it out
    // of the upload.
    values[TOKEN] = request.token.unwrap_or(step.mirror[TOKEN]);
    values[POSITION] = position;
    values[SLOT] = slot;
    values[METADATA] = sequence.context_len;
    values[METADATA + 1] = sequence.first_page_offset;
    for (entry, block) in values[METADATA + 2..].iter_mut().zip(&sequence.blocks) {
        *entry = block.index;
    }
    step.update(&values)?;
    if request.token.is_none() {
        // The device holds a token the host never saw: the next host token
        // must upload whatever it is.
        step.mirror[TOKEN] = u32::MAX;
    }
    Ok(step.tensor.clone())
}

/// The step buffer's token slot (U32 `[1]`), where token selection writes the
/// next step's input, once the buffer exists.
pub(super) fn token_slot(runtime: &WholeStepRuntime) -> Option<Result<Tensor>> {
    runtime
        .step
        .as_ref()
        .map(|step| step.tensor.narrow(0, TOKEN, 1).map_err(Into::into))
}
