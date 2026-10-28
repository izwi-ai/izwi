use super::diagnostics;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use candle_transformers::models::with_tracing::QMatMul;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::backends::state::{
    InvocationRingDepthwiseConvTransaction, PhysicalStateSequenceId, PhysicalStateTransactionId,
    StateComponentValue, TensorStateArena,
};
use crate::engine::InvocationTensorLease;
use crate::error::{Error, Result};
use crate::kernels::{
    try_fused_qk_rms_norm, try_fused_rms_norm, try_fused_rope_pair_bshd,
    try_fused_silu_mul_with_status, try_lfm_shortconv_decode3, try_lfm_shortconv_ring_sequence,
    try_lfm_shortconv_sequence3,
};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention_with_options, CudaFlashAttentionOptions,
};
use crate::models::shared::attention::gqa::{
    banded_sdpa_bhsd, compact_gqa_sdpa_bhsd, AttentionBand, CompactGqaMask,
};
use crate::models::shared::attention::mixed::{MixedRowLayout, PreparedPhysicalMixedStep};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::telemetry::record_rope_kernel;
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::Lfm2BackboneConfig;
use super::physical::LFM2_SHORTCONV_STATE_DOMAIN;
use super::physical::Lfm2StateLayout;

#[derive(Clone)]
pub(crate) struct Lfm2ShortConvRuntimeState {
    cursor: u64,
    capacity: usize,
    hidden: usize,
    components: Vec<(crate::kv::v2::StateComponentId, Option<Tensor>)>,
}

#[derive(Debug)]
pub(crate) struct Lfm2RetainedEmbeddingBatchOutput {
    pub(crate) last_hidden: Vec<Tensor>,
    pub(crate) launch_widths: Vec<usize>,
}

fn retained_ragged_launch_rows(spans: &[usize]) -> Result<Vec<Vec<usize>>> {
    if spans.is_empty() || spans.contains(&0) {
        return Err(Error::InvalidInput(
            "LFM2 retained ragged spans must be non-empty".into(),
        ));
    }
    let max_span = spans.iter().copied().max().expect("non-empty spans");
    Ok((0..max_span)
        .map(|offset| {
            spans
                .iter()
                .enumerate()
                .filter_map(|(row, span)| (offset < *span).then_some(row))
                .collect::<Vec<_>>()
        })
        .collect())
}

impl Lfm2ShortConvRuntimeState {
    pub(crate) fn new(model: &QuantizedLfm2Backbone) -> Self {
        Self {
            cursor: 0,
            capacity: model.cfg.shortconv_l_cache,
            hidden: model.cfg.embedding_length,
            components: model
                .state_layout
                .shortconv_components
                .iter()
                .map(|(_, component)| (*component, None))
                .collect(),
        }
    }

    pub(crate) fn cursor(&self) -> u64 {
        self.cursor
    }

    pub(crate) fn restore(
        &mut self,
        arena: &TensorStateArena,
        sequence: PhysicalStateSequenceId,
    ) -> Result<()> {
        let snapshot = arena.read(sequence, LFM2_SHORTCONV_STATE_DOMAIN)?;
        let Some(snapshot) = snapshot else {
            if self.cursor == 0 {
                return Ok(());
            }
            return Err(Error::InferenceError(
                "LFM2 retained ShortConv state disappeared".into(),
            ));
        };
        if snapshot.components.len() != self.components.len() {
            return Err(Error::InferenceError(
                "LFM2 retained ShortConv component count changed".into(),
            ));
        }
        for ((expected, slot), value) in self.components.iter_mut().zip(snapshot.components.iter())
        {
            if *expected != value.component {
                return Err(Error::InferenceError(
                    "LFM2 retained ShortConv component identity changed".into(),
                ));
            }
            match value.tensor.as_ref() {
                Some(tensor)
                    if tensor.dims() == [self.capacity, 1, self.hidden]
                        && tensor.dtype() == DType::F32 =>
                {
                    *slot = Some(tensor.clone());
                }
                Some(_) => {
                    return Err(Error::InferenceError(
                        "LFM2 retained ShortConv tensor geometry changed".into(),
                    ))
                }
                None if snapshot.cursor == 0 => *slot = None,
                None => {
                    return Err(Error::InferenceError(
                        "LFM2 retained ShortConv tensor is absent after cursor zero".into(),
                    ))
                }
            }
        }
        self.cursor = snapshot.cursor;
        Ok(())
    }

    pub(crate) fn stage(
        &mut self,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        if self.cursor != target_cursor {
            return Err(Error::InferenceError(
                "LFM2 ShortConv cursor does not match the target KV cursor".into(),
            ));
        }
        let expected_cursor = arena
            .read_transaction_base(transaction, LFM2_SHORTCONV_STATE_DOMAIN)?
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        let values = self
            .components
            .iter()
            .map(|(component, tensor)| {
                let tensor = tensor.as_ref().ok_or_else(|| {
                    Error::InferenceError("LFM2 ShortConv state was not initialized".into())
                })?;
                Ok(StateComponentValue {
                    component: *component,
                    tensor: Some(tensor.clone()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        arena.stage_replace(
            transaction,
            LFM2_SHORTCONV_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            values,
        )?;
        for (_, tensor) in &mut self.components {
            *tensor = None;
        }
        Ok(())
    }

    fn apply(
        &mut self,
        component: crate::kv::v2::StateComponentId,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        let (_, hidden, steps) = input.dims3()?;
        if hidden != self.hidden || weight.dims2()? != (self.hidden, self.capacity) {
            return Err(Error::InvalidInput(
                "LFM2 retained ShortConv geometry mismatch".into(),
            ));
        }
        let (_, ring) = self
            .components
            .iter_mut()
            .find(|(candidate, _)| *candidate == component)
            .ok_or_else(|| Error::InferenceError("LFM2 ShortConv component is absent".into()))?;
        let current = match ring.as_ref() {
            Some(ring) => ring.clone(),
            None if self.cursor == 0 => {
                Tensor::zeros((self.capacity, 1, self.hidden), DType::F32, input.device())?
            }
            None => {
                return Err(Error::InferenceError(
                    "LFM2 ShortConv state is absent after cursor zero".into(),
                ))
            }
        };
        let valid = self.cursor.min(self.capacity as u64);
        if diagnostics::enabled() {
            diagnostics::check(
                input,
                &format!("ShortConv {component:?} gated input"),
                self.cursor as usize,
            )?;
            diagnostics::check(
                &current,
                &format!("ShortConv {component:?} ring"),
                self.cursor as usize,
            )?;
        }
        let output = try_lfm_shortconv_ring_sequence(
            &current,
            &input.contiguous()?,
            weight,
            self.cursor,
            valid,
        );
        // With fused kernels enabled, an accelerator kernel that declines is
        // a defect, not a reason to fall back silently to many small ops.
        // `IZWI_FUSED_KERNELS=0` asks for the device-agnostic op path.
        if (input.device().is_cuda() || input.device().is_metal())
            && output.is_none()
            && crate::kernels::use_fused_kernels_for_device(input.device())
        {
            return Err(Error::InferenceError(
                "LFM2 accelerator ShortConv ring kernel is unavailable".into(),
            ));
        }
        let output = match output {
            Some(output) => output,
            None => direct_shortconv_ring(&current, input, weight, self.cursor, valid)?,
        };
        if diagnostics::enabled() {
            diagnostics::check(
                &output,
                &format!("ShortConv {component:?} convolution output"),
                self.cursor as usize,
            )?;
        }
        let mut slots = (0..self.capacity)
            .map(|index| current.i(index))
            .collect::<candle_core::Result<Vec<_>>>()?;
        // Earlier tokens contribute to the convolution output, but only this
        // suffix survives in the ring after a full-span prefill.
        for step in steps.saturating_sub(self.capacity)..steps {
            let absolute = self.cursor.saturating_add(step as u64);
            let slot = (absolute % self.capacity as u64) as usize;
            slots[slot] = input.i((0, .., step))?.unsqueeze(0)?;
        }
        let refs = slots.iter().collect::<Vec<_>>();
        *ring = Some(Tensor::stack(&refs, 0)?.contiguous()?);
        Ok(output)
    }

    /// One decode token through the fused ShortConv step
    /// (`kernels::lfm_shortconv_step`): `projected` is the token's
    /// in-projection output `[1, 1, 3 * hidden]`; returns the gated
    /// convolution `[1, 1, hidden]` and installs the next ring. `None` when
    /// fused kernels are switched off, so the caller runs the op-by-op path.
    /// Like [`Self::apply`], the cursor advances only through
    /// [`Self::advance`].
    fn apply_fused_step(
        &mut self,
        component: crate::kv::v2::StateComponentId,
        projected: &Tensor,
        weight: &Tensor,
    ) -> Result<Option<Tensor>> {
        #[cfg(test)]
        if FORCE_UNFUSED_SHORTCONV.with(std::cell::Cell::get) {
            return Ok(None);
        }
        let device = projected.device();
        let enabled = if device.is_cpu() {
            crate::kernels::fused_kernels_enabled_by_env()
        } else {
            crate::kernels::use_fused_kernels_for_device(device)
        };
        if !enabled {
            return Ok(None);
        }
        if weight.dims2()? != (self.hidden, self.capacity) {
            return Err(Error::InvalidInput(
                "LFM2 retained ShortConv geometry mismatch".into(),
            ));
        }
        let cursor = self.cursor;
        let capacity = self.capacity;
        let hidden = self.hidden;
        let (_, ring) = self
            .components
            .iter_mut()
            .find(|(candidate, _)| *candidate == component)
            .ok_or_else(|| Error::InferenceError("LFM2 ShortConv component is absent".into()))?;
        let current = match ring.as_ref() {
            Some(ring) => ring.clone(),
            None if cursor == 0 => Tensor::zeros((capacity, 1, hidden), DType::F32, device)?,
            None => {
                return Err(Error::InferenceError(
                    "LFM2 ShortConv state is absent after cursor zero".into(),
                ))
            }
        };
        let valid = cursor.min(capacity as u64);
        let (y, next) = crate::kernels::lfm_shortconv_step::lfm_shortconv_gated_ring_step(
            projected, &current, weight, cursor, valid,
        )?;
        *ring = Some(next);
        Ok(Some(y))
    }

    fn advance(&mut self, steps: usize) -> Result<()> {
        self.cursor =
            self.cursor
                .checked_add(u64::try_from(steps).map_err(|_| {
                    Error::InvalidInput("LFM2 ShortConv step count exceeds u64".into())
                })?)
                .ok_or_else(|| Error::InvalidInput("LFM2 ShortConv cursor overflow".into()))?;
        Ok(())
    }
}

fn direct_shortconv_ring(
    ring: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    cursor: u64,
    valid: u64,
) -> Result<Tensor> {
    if ring.device().is_cpu() {
        return host_shortconv_ring(ring, input, weight, cursor, valid);
    }
    op_shortconv_ring(ring, input, weight, cursor, valid)
}

/// [`direct_shortconv_ring`] as Candle ops, one per tap and step.
fn op_shortconv_ring(
    ring: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    cursor: u64,
    valid: u64,
) -> Result<Tensor> {
    let (_, hidden, steps) = input.dims3()?;
    let capacity = ring.dim(0)?;
    let oldest = cursor.saturating_sub(valid);
    let mut outputs = Vec::with_capacity(steps);
    for step in 0..steps {
        let absolute = cursor + step as u64;
        let window_start = absolute as i128 + 1 - capacity as i128;
        let mut output = None::<Tensor>;
        for tap in 0..capacity {
            let source = window_start + tap as i128;
            if source < 0 {
                continue;
            }
            let source = source as u64;
            let value = if source < cursor {
                if source < oldest {
                    continue;
                }
                ring.i((source % capacity as u64) as usize)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?
            } else {
                let input_step = usize::try_from(source - cursor).map_err(|_| {
                    Error::InvalidInput("LFM2 ShortConv input cursor overflow".into())
                })?;
                if input_step > step {
                    continue;
                }
                input.narrow(2, input_step, 1)?
            };
            let contribution =
                value.broadcast_mul(&weight.narrow(1, tap, 1)?.reshape((1, hidden, 1))?)?;
            output = Some(match output {
                Some(current) => (&current + &contribution)?,
                None => contribution,
            });
        }
        outputs.push(output.ok_or_else(|| {
            Error::InferenceError("LFM2 ShortConv produced no current contribution".into())
        })?);
    }
    Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2).map_err(Error::from)
}

/// [`direct_shortconv_ring`] over host memory in one pass: each output sums
/// the same taps in the same order, so it is bit-identical, without one
/// Candle op per tap and step (LFM3's 7-tap convolution made a 1K-token CPU
/// prefill issue about 100K of them per layer stack).
fn host_shortconv_ring(
    ring: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    cursor: u64,
    valid: u64,
) -> Result<Tensor> {
    let (batch, hidden, steps) = input.dims3()?;
    let capacity = ring.dim(0)?;
    if batch != 1 || ring.elem_count() != capacity * hidden || weight.dims2()? != (hidden, capacity)
    {
        return Err(Error::InvalidInput(
            "LFM2 ShortConv ring geometry mismatch".into(),
        ));
    }
    let host = |tensor: &Tensor| -> Result<Vec<f32>> {
        Ok(tensor
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    let (ring_values, inputs, weights) = (host(ring)?, host(input)?, host(weight)?);
    let oldest = cursor.saturating_sub(valid);
    let mut output = vec![0f32; hidden * steps];
    for channel in 0..hidden {
        let taps = &weights[channel * capacity..][..capacity];
        let row = &inputs[channel * steps..][..steps];
        for step in 0..steps {
            let window_start = (cursor + step as u64) as i128 + 1 - capacity as i128;
            let mut sum = None::<f32>;
            for (tap, weight) in taps.iter().enumerate() {
                let source = window_start + tap as i128;
                if source < 0 {
                    continue;
                }
                let source = source as u64;
                let value = if source < cursor {
                    if source < oldest {
                        continue;
                    }
                    ring_values[(source % capacity as u64) as usize * hidden + channel]
                } else {
                    row[(source - cursor) as usize]
                };
                let contribution = value * weight;
                sum = Some(sum.map_or(contribution, |sum| sum + contribution));
            }
            output[channel * steps + step] = sum.ok_or_else(|| {
                Error::InferenceError("LFM2 ShortConv produced no current contribution".into())
            })?;
        }
    }
    Tensor::from_vec(output, (1, hidden, steps), input.device()).map_err(Error::from)
}

#[cfg(test)]
thread_local! {
    /// Test-only switch that sends this thread's ShortConv decode steps down
    /// the op-by-op path, for parity and A/B timing against the fused step.
    pub(crate) static FORCE_UNFUSED_SHORTCONV: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test-only switch that sends this thread's projections to Candle's
    /// `QMatMul`, for A/B timing against the small-batch mat-vec.
    pub(crate) static FORCE_CANDLE_MATVEC: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// One decode step's KV writes and next ShortConv states, not yet
/// committed. Dropping it without [`Self::commit`] or [`Self::abort_with`]
/// would orphan in-flight writes, so both consume it.
#[must_use = "a pending decode step must be committed or aborted"]
pub(crate) struct Lfm2PendingDecodeCommit {
    completions: KvWriteCompletionCollector,
    working: Vec<Lfm2ShortConvRuntimeState>,
    positions: Vec<usize>,
}

impl Lfm2PendingDecodeCommit {
    /// Wait for the step's KV writes (already done when the caller has read
    /// the logits back), then advance every row's cache and ShortConv state.
    pub(crate) fn commit(
        self,
        states: &mut [&mut Lfm2ShortConvRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<()> {
        if states.len() != self.working.len() || caches.len() != self.positions.len() {
            return Err(self.abort_with(Error::InvalidInput(
                "LFM2 pending decode commit rows do not match".into(),
            )));
        }
        let completion = Arc::new(self.completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(self.positions[row], 1, completion.clone())?;
        }
        for (state, working) in states.iter_mut().zip(self.working) {
            **state = working;
        }
        Ok(())
    }

    /// Drain the step's writes without committing them, returning `error`
    /// (annotated if the drain also fails).
    pub(crate) fn abort_with(self, error: Error) -> Error {
        match self.completions.drain() {
            Ok(()) => error,
            Err(drain) => Error::InferenceError(format!(
                "LFM2 decode step failed: {error}; fence drain failed: {drain}"
            )),
        }
    }
}

/// One mixed step's KV writes and next ShortConv states, not yet committed:
/// [`Self::commit`] or [`Self::abort_with`] consumes it.
#[must_use = "a pending mixed step must be committed or aborted"]
pub(crate) struct Lfm2PendingMixedCommit {
    step: PreparedPhysicalMixedStep,
    working: Vec<Lfm2ShortConvRuntimeState>,
}

impl Lfm2PendingMixedCommit {
    /// Seal the step's writes, then advance every row's cache by its span
    /// and install its ShortConv state.
    pub(crate) fn commit(
        self,
        states: &mut [&mut Lfm2ShortConvRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<()> {
        if states.len() != self.working.len() {
            return Err(self.abort_with(Error::InvalidInput(
                "LFM2 pending mixed commit rows do not match".into(),
            )));
        }
        self.step.finish(caches, Ok(()))?;
        for (state, working) in states.iter_mut().zip(self.working) {
            **state = working;
        }
        Ok(())
    }

    /// Drain the step's writes without committing them, returning `error`
    /// (annotated if the drain also fails).
    pub(crate) fn abort_with(self, error: Error) -> Error {
        match self.step.finish::<()>(&mut [], Err(error)) {
            Err(error) => error,
            Ok(()) => unreachable!("an aborted mixed step cannot finish"),
        }
    }
}

#[derive(Debug)]
struct Mlp {
    gate: Lfm2Linear,
    down: Lfm2Linear,
    up: Lfm2Linear,
}

#[derive(Debug)]
struct AttentionLayer {
    wq: Lfm2Linear,
    wk: Lfm2Linear,
    wv: Lfm2Linear,
    wo: Lfm2Linear,
    q_norm: LfmRmsNorm,
    k_norm: LfmRmsNorm,
    qk_norm_weight: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    cos: Tensor,
    sin: Tensor,
    cos_sin: Tensor,
    neg_inf: Tensor,
    physical_layer: usize,
    model_layer: usize,
    /// LFM3: `[hidden -> heads]`; each head's output is scaled by the sigmoid
    /// of its gate before the output projection.
    gate: Option<Lfm2Linear>,
}

#[derive(Debug)]
struct ShortConvLayer {
    in_proj: Lfm2Linear,
    out_proj: Lfm2Linear,
    conv: Tensor,
    l_cache: usize,
    component: crate::kv::v2::StateComponentId,
}

#[derive(Debug)]
enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Debug)]
struct LayerWeights {
    operator_norm: LfmRmsNorm,
    ffn_norm: LfmRmsNorm,
    mlp: Mlp,
    kind: LayerKind,
    /// LFM3 sandwich norms: applied to the operator and FFN outputs before
    /// their residual adds.
    post_operator_norm: Option<LfmRmsNorm>,
    post_ffn_norm: Option<LfmRmsNorm>,
}

impl LayerWeights {
    /// The operator output as the residual stream receives it.
    fn operator_output(&self, hidden: Tensor) -> Result<Tensor> {
        match &self.post_operator_norm {
            Some(norm) => norm.forward(&hidden.contiguous()?),
            None => Ok(hidden),
        }
    }

    /// The FFN of a normalized input, as the residual stream receives it.
    fn ffn_output(&self, normalized: &Tensor) -> Result<Tensor> {
        let output = self.mlp.forward(normalized)?;
        match &self.post_ffn_norm {
            Some(norm) => norm.forward(&output.contiguous()?),
            None => Ok(output),
        }
    }
}

#[derive(Debug)]
struct ProjectionHead {
    weight: Lfm2Linear,
    bias: Option<Tensor>,
}

pub struct QuantizedLfm2Backbone {
    cfg: Lfm2BackboneConfig,
    /// `None` for a backbone that only runs on supplied embeddings.
    token_embeddings: Option<Arc<candle_core::quantized::QTensor>>,
    /// `None` for a trunk that only produces hidden states (LFM3-Audio's
    /// encoder and codec decoder run their own heads).
    output_head: Option<ProjectionHead>,
    layers: Vec<LayerWeights>,
    norm: LfmRmsNorm,
    vocab_size: usize,
    state_layout: Lfm2StateLayout,
}

#[derive(Debug, Clone)]
struct LfmRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl LfmRmsNorm {
    fn from_qtensor(weight: candle_core::quantized::QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device()).map_err(Error::from)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.device().is_metal() {
            let input = if input.is_contiguous() {
                input.clone()
            } else {
                input.contiguous()?
            };
            if let Some(output) = try_fused_rms_norm(&input, &self.weight, self.eps) {
                return Ok(output);
            }
        }
        candle_nn::ops::rms_norm(input, &self.weight, self.eps as f32).map_err(Error::from)
    }

    fn eps(&self) -> f64 {
        self.eps
    }

    fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Mlp {
    fn load(loader: &GgufLoader, device: &Device, prefix: &str) -> Result<Self> {
        Self::load_with_prefixes(loader, device, &[prefix.to_string()])
    }

    fn load_with_prefixes(
        loader: &GgufLoader,
        device: &Device,
        prefixes: &[String],
    ) -> Result<Self> {
        let mut gate_names = Vec::new();
        let mut down_names = Vec::new();
        let mut up_names = Vec::new();
        for prefix in prefixes {
            gate_names.extend([
                format!("{prefix}.ffn_gate.weight"),
                format!("{prefix}.feed_forward.w1.weight"),
                format!("{prefix}.mlp.gate_proj.weight"),
            ]);
            down_names.extend([
                format!("{prefix}.ffn_down.weight"),
                format!("{prefix}.feed_forward.w2.weight"),
                format!("{prefix}.mlp.down_proj.weight"),
            ]);
            up_names.extend([
                format!("{prefix}.ffn_up.weight"),
                format!("{prefix}.feed_forward.w3.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
            ]);
        }

        Ok(Self {
            gate: load_linear_any(loader, device, &gate_names)?,
            down: load_linear_any(loader, device, &down_names)?,
            up: load_linear_any(loader, device, &up_names)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(hidden_states)?;
        let up = self.up.forward(hidden_states)?;
        let hidden = if let Some(fused) = try_fused_silu_mul_with_status(&gate, &up) {
            fused.tensor
        } else {
            let gate = candle_nn::ops::silu(&gate)?;
            gate.broadcast_mul(&up)?
        };
        self.down.forward(&hidden).map_err(Error::from)
    }
}

// Decode rows can have different absolute positions. Gather their rotary tables
// once and let Candle rotate all heads/rows in one launch per Q/K tensor.
fn rotary_decode_batch(
    query: &Tensor,
    key: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &[usize],
) -> Result<(Tensor, Tensor)> {
    RotaryDecodeRows::gather(cos, sin, positions, query.dim(0)?)?.rotate(query, key)
}

/// The rotary rows of one decode step. Attention layers share their tables
/// (LFM2 has one set, LFM3 one for global and one for sliding layers), so a
/// step gathers each table once (one position upload) instead of once per
/// layer.
struct RotaryDecodeRows {
    table: candle_core::TensorId,
    cos: Tensor,
    sin: Tensor,
}

impl RotaryDecodeRows {
    fn gather(cos: &Tensor, sin: &Tensor, positions: &[usize], batch: usize) -> Result<Self> {
        let table_len = cos.dim(0)?.min(sin.dim(0)?);
        if positions.len() != batch || positions.iter().any(|&position| position >= table_len) {
            return Err(Error::InvalidInput(
                "LFM2 rotary batch positions exceed the available tables".into(),
            ));
        }
        let positions = positions
            .iter()
            .map(|&position| u32::try_from(position))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::InvalidInput("LFM2 rotary position exceeds u32".into()))?;
        let indices = Tensor::from_vec(positions, batch, cos.device())?;
        Ok(Self {
            table: cos.id(),
            cos: cos.index_select(&indices, 0)?.unsqueeze(1)?.contiguous()?,
            sin: sin.index_select(&indices, 0)?.unsqueeze(1)?.contiguous()?,
        })
    }

    /// This step's rows of `cos`/`sin`, gathered on first use.
    fn for_step<'a>(
        rows: &'a mut Vec<Self>,
        cos: &Tensor,
        sin: &Tensor,
        positions: &[usize],
        batch: usize,
    ) -> Result<&'a Self> {
        let index = match rows.iter().position(|rows| rows.table == cos.id()) {
            Some(index) => index,
            None => {
                rows.push(Self::gather(cos, sin, positions, batch)?);
                rows.len() - 1
            }
        };
        Ok(&rows[index])
    }

    fn rotate(&self, query: &Tensor, key: &Tensor) -> Result<(Tensor, Tensor)> {
        if query.dim(0)? != self.cos.dim(0)? {
            return Err(Error::InvalidInput(
                "LFM2 rotary rows do not match the decode batch".into(),
            ));
        }
        Ok((
            candle_nn::rotary_emb::rope_thd(&query.contiguous()?, &self.cos, &self.sin)?,
            candle_nn::rotary_emb::rope_thd(&key.contiguous()?, &self.cos, &self.sin)?,
        ))
    }
}

impl AttentionLayer {
    /// The output projection of the attention heads `attn_output`
    /// (`[.., heads * head_dim]`) for the layer input `input` (`[.., hidden]`),
    /// after LFM3's per-head sigmoid gate when the layer has one.
    fn project_output(&self, input: &Tensor, attn_output: &Tensor) -> Result<Tensor> {
        let Some(gate) = &self.gate else {
            return self.wo.forward(attn_output).map_err(Error::from);
        };
        let dims = attn_output.dims();
        let lead = &dims[..dims.len() - 1];
        let gate = candle_nn::ops::sigmoid(&gate.forward(input)?)?;
        let heads = attn_output.reshape([lead, &[self.n_head, self.head_dim]].concat())?;
        let gate = gate.reshape([lead, &[self.n_head, 1]].concat())?;
        let gated = heads.broadcast_mul(&gate)?.reshape(dims)?;
        self.wo.forward(&gated).map_err(Error::from)
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        record_rope_kernel();
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Error::from)
    }

    fn try_apply_rotary_emb_pair_bshd(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let (_, seq_len, _, _) = q.dims4()?;
        if q.device().is_cuda() {
            let cos = self.cos.narrow(0, index_pos, seq_len)?.contiguous()?;
            let sin = self.sin.narrow(0, index_pos, seq_len)?.contiguous()?;
            record_rope_kernel();
            record_rope_kernel();
            return Ok(Some((
                candle_nn::rotary_emb::rope_thd(&q.contiguous()?, &cos, &sin)?,
                candle_nn::rotary_emb::rope_thd(&k.contiguous()?, &cos, &sin)?,
            )));
        }
        let packed = self.cos_sin.narrow(0, index_pos, seq_len)?.contiguous()?;
        if let Some((q, k)) = try_fused_rope_pair_bshd(&q.contiguous()?, &k.contiguous()?, &packed)
        {
            record_rope_kernel();
            record_rope_kernel();
            return Ok(Some((q, k)));
        }
        Ok(None)
    }

    fn project_qkv(
        &self,
        hidden_states: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let query_states = self.wq.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ))?;
        let key_states = self.wk.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.n_kv_head,
            self.head_dim,
        ))?;
        let value_states = self
            .wv
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query_states = query_states.contiguous()?;
        let key_states = key_states.contiguous()?;
        diagnostics::check_layer(&query_states, "raw Q", self.model_layer, index_pos)?;
        diagnostics::check_layer(&key_states, "raw K", self.model_layer, index_pos)?;
        diagnostics::check_layer(&value_states, "raw V", self.model_layer, index_pos)?;
        let (query_states, key_states) = if seq_len == 1
            && (query_states.device().is_metal() || query_states.device().is_cuda())
        {
            if let Some((query_states, key_states)) = try_fused_qk_rms_norm(
                &query_states,
                &key_states,
                &self.qk_norm_weight,
                self.q_norm.eps(),
            ) {
                (query_states, key_states)
            } else {
                (
                    self.q_norm.forward(&query_states)?,
                    self.k_norm.forward(&key_states)?,
                )
            }
        } else {
            (
                self.q_norm.forward(&query_states)?,
                self.k_norm.forward(&key_states)?,
            )
        };
        diagnostics::check_layer(&query_states, "normalized Q", self.model_layer, index_pos)?;
        diagnostics::check_layer(&key_states, "normalized K", self.model_layer, index_pos)?;
        let (query_states, key_states) = if let Some((query_states, key_states)) =
            self.try_apply_rotary_emb_pair_bshd(&query_states, &key_states, index_pos)?
        {
            (
                query_states.transpose(1, 2)?.contiguous()?,
                key_states.transpose(1, 2)?.contiguous()?,
            )
        } else {
            let query_states = query_states.transpose(1, 2)?.contiguous()?;
            let key_states = key_states.transpose(1, 2)?.contiguous()?;
            (
                self.apply_rotary_emb(&query_states, index_pos)?,
                self.apply_rotary_emb(&key_states, index_pos)?,
            )
        };
        diagnostics::check_layer(
            &query_states,
            "normalized/rotated Q",
            self.model_layer,
            index_pos,
        )?;
        diagnostics::check_layer(
            &key_states,
            "normalized/rotated K",
            self.model_layer,
            index_pos,
        )?;
        Ok((query_states, key_states, value_states))
    }

    fn forward_stateless(
        &self,
        hidden_states: &Tensor,
        mask: StatelessAttentionMask<'_>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let (query_states, key_states, value_states) =
            self.project_qkv(hidden_states, index_pos)?;

        if query_states.device().is_cuda() && flash_attention_requested() {
            let cuda_options = lfm25_cuda_flash_attention_options(
                !matches!(mask, StatelessAttentionMask::None),
                self.sliding_window,
            );
            if let Some(attn_output) = try_fused_self_attention_with_options(
                &query_states,
                &key_states,
                &value_states,
                None,
                self.head_dim,
                true,
                cuda_options,
            )? {
                let attn_output =
                    attn_output
                        .transpose(1, 2)?
                        .reshape((batch_size, seq_len, hidden_size))?;
                return self.project_output(hidden_states, &attn_output);
            }
        }

        if should_try_lfm25_metal_prefill_sdpa(
            &query_states,
            seq_len,
            index_pos,
            self.sliding_window,
        ) {
            if let Some(attn_output) = try_fused_self_attention_with_options(
                &query_states,
                &key_states,
                &value_states,
                None,
                self.head_dim,
                true,
                CudaFlashAttentionOptions::default(),
            )? {
                let attn_output =
                    attn_output
                        .transpose(1, 2)?
                        .reshape((batch_size, seq_len, hidden_size))?;
                return self.project_output(hidden_states, &attn_output);
            }
        }

        let softmax_scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = match mask {
            StatelessAttentionMask::None => compact_gqa_sdpa_bhsd(
                &query_states,
                &key_states,
                &value_states,
                None,
                softmax_scale,
            )?,
            StatelessAttentionMask::Dense(mask) => compact_gqa_sdpa_bhsd(
                &query_states,
                &key_states,
                &value_states,
                Some(CompactGqaMask::Boolean {
                    mask,
                    masked_value: &self.neg_inf,
                }),
                softmax_scale,
            )?,
            StatelessAttentionMask::Band(band) => {
                LFM2_BANDED_STATELESS_ATTENTION_CALLS.fetch_add(1, Ordering::Relaxed);
                banded_sdpa_bhsd(
                    &query_states,
                    &key_states,
                    &value_states,
                    band,
                    softmax_scale,
                )?
            }
        };
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((batch_size, seq_len, hidden_size))?;
        self.project_output(hidden_states, &attn_output)
    }

    fn forward_physical(
        &self,
        hidden_states: &Tensor,
        index_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        if batch_size != 1 {
            return Err(Error::InvalidInput(
                "LFM2 physical attention currently requires batch size one".into(),
            ));
        }
        let (queries, keys, values) = self.project_qkv(hidden_states, index_pos)?;
        let kv_dtype = cache.arena().config().dtype;
        let [queries, keys, values] = [queries, keys, values].map(|tensor| {
            tensor
                .squeeze(0)?
                .transpose(0, 1)?
                .contiguous()?
                .to_dtype(kv_dtype)
        });
        let (queries, keys, values) = (queries?, keys?, values?);
        let softmax_scale = (1.0f64 / (self.head_dim as f64).sqrt()) as f32;
        let output = match self.sliding_window {
            Some(window_tokens) => cache.write_and_attend_with_window(
                self.physical_layer,
                prepared,
                &queries,
                &keys,
                &values,
                softmax_scale,
                window_tokens,
            )?,
            None => cache.write_and_attend(
                self.physical_layer,
                prepared,
                &queries,
                &keys,
                &values,
                softmax_scale,
            )?,
        };
        let output = output.to_dtype(hidden_states.dtype())?;
        diagnostics::check_layer(
            &output,
            "paged attention output",
            self.model_layer,
            index_pos,
        )?;
        let output = output.reshape((batch_size, seq_len, hidden_size))?;
        self.project_output(hidden_states, &output)
    }

    fn forward_retained_decode_batch(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        completions: &mut KvWriteCompletionCollector,
        rotary_rows: &mut Vec<RotaryDecodeRows>,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden_states.dims3()?;
        if batch == 0 || seq_len != 1 || positions.len() != batch || caches.len() != batch {
            return Err(Error::InvalidInput(
                "LFM2 attention decode batch dimensions do not match".into(),
            ));
        }
        let first = caches[0];
        if caches
            .iter()
            .any(|cache| !Arc::ptr_eq(cache.arena(), first.arena()))
            || slots.arena_id() != first.arena().id()
            || slots.len() != batch
        {
            return Err(Error::InvalidInput(
                "LFM2 attention decode rows do not share one physical arena".into(),
            ));
        }
        let query =
            self.wq
                .forward(hidden_states)?
                .reshape((batch, 1, self.n_head, self.head_dim))?;
        let key =
            self.wk
                .forward(hidden_states)?
                .reshape((batch, 1, self.n_kv_head, self.head_dim))?;
        let values =
            self.wv
                .forward(hidden_states)?
                .reshape((batch, 1, self.n_kv_head, self.head_dim))?;
        diagnostics::check_batch_layer(&query, "raw Q", self.model_layer, positions)?;
        diagnostics::check_batch_layer(&key, "raw K", self.model_layer, positions)?;
        diagnostics::check_batch_layer(&values, "raw V", self.model_layer, positions)?;
        let query = self.q_norm.forward(&query.contiguous()?)?;
        let key = self.k_norm.forward(&key.contiguous()?)?;
        diagnostics::check_batch_layer(&query, "normalized Q", self.model_layer, positions)?;
        diagnostics::check_batch_layer(&key, "normalized K", self.model_layer, positions)?;
        let (queries, keys) = if query.device().is_cuda() {
            let (query, key) =
                RotaryDecodeRows::for_step(rotary_rows, &self.cos, &self.sin, positions, batch)?
                    .rotate(&query, &key)?;
            record_rope_kernel();
            record_rope_kernel();
            (
                query.reshape((batch, self.n_head, self.head_dim))?,
                key.reshape((batch, self.n_kv_head, self.head_dim))?,
            )
        } else {
            let mut queries = Vec::with_capacity(batch);
            let mut keys = Vec::with_capacity(batch);
            for row in 0..batch {
                let q = query.narrow(0, row, 1)?;
                let k = key.narrow(0, row, 1)?;
                let (q, k) = if let Some((q, k)) =
                    self.try_apply_rotary_emb_pair_bshd(&q, &k, positions[row])?
                {
                    (q.transpose(1, 2)?, k.transpose(1, 2)?)
                } else {
                    (
                        self.apply_rotary_emb(&q.transpose(1, 2)?.contiguous()?, positions[row])?,
                        self.apply_rotary_emb(&k.transpose(1, 2)?.contiguous()?, positions[row])?,
                    )
                };
                queries.push(q.reshape((self.n_head, self.head_dim))?);
                keys.push(k.reshape((self.n_kv_head, self.head_dim))?);
            }
            let queries = Tensor::stack(&queries.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
            let keys = Tensor::stack(&keys.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
            (queries, keys)
        };
        let values = values
            .reshape((batch, self.n_kv_head, self.head_dim))?
            .contiguous()?;
        diagnostics::check_batch_layer(
            &queries,
            "normalized/rotated Q",
            self.model_layer,
            positions,
        )?;
        diagnostics::check_batch_layer(&keys, "normalized/rotated K", self.model_layer, positions)?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| match self.sliding_window {
                    Some(window) => cache.sequence_table_with_window(positions[row] + 1, window),
                    None => cache.sequence_table(positions[row] + 1),
                })
                .collect::<Result<Vec<_>>>()?,
        };
        // The arena may store KV narrower than the F32 activations.
        let kv_dtype = first.arena().config().dtype;
        let queries = queries.to_dtype(kv_dtype)?;
        let keys = keys.to_dtype(kv_dtype)?;
        let values = values.to_dtype(kv_dtype)?;
        let binding = first.layer_binding(self.physical_layer)?;
        let completion = first.arena().write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let (output, completion) = submit_ordered_after_write(completion, || {
            first.arena().paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &queries,
                    batch: &metadata,
                    softmax_scale: (1.0 / (self.head_dim as f32).sqrt()),
                    softcap: None,
                },
            )
        })?;
        completions.collect(completion)?;
        let output = output.to_dtype(hidden_states.dtype())?;
        diagnostics::check_batch_layer(
            &output,
            "paged attention output",
            self.model_layer,
            positions,
        )?;
        self.project_output(hidden_states, &output.reshape((batch, 1, hidden_size))?)
    }
}

impl AttentionLayer {
    /// One mixed step: projections and norms on every token, RoPE per row
    /// from its own position (as each split call rotates), then one K/V
    /// write and paged attention over the step's rows.
    fn forward_mixed(
        &self,
        hidden_states: &Tensor,
        caches: &[&PhysicalPagedKvCache],
        step: &mut PreparedPhysicalMixedStep,
    ) -> Result<Tensor> {
        let (_, tokens, hidden_size) = hidden_states.dims3()?;
        let query =
            self.wq
                .forward(hidden_states)?
                .reshape((1, tokens, self.n_head, self.head_dim))?;
        let key =
            self.wk
                .forward(hidden_states)?
                .reshape((1, tokens, self.n_kv_head, self.head_dim))?;
        let values =
            self.wv
                .forward(hidden_states)?
                .reshape((tokens, self.n_kv_head, self.head_dim))?;
        let query = self.q_norm.forward(&query.contiguous()?)?;
        let key = self.k_norm.forward(&key.contiguous()?)?;
        let layout = step.layout();
        let mut queries = Vec::with_capacity(layout.rows().len());
        let mut keys = Vec::with_capacity(layout.rows().len());
        for (row, span) in layout.rows().iter().enumerate() {
            let offset = layout.token_offset(row);
            let q = query.narrow(1, offset, span.q_len)?;
            let k = key.narrow(1, offset, span.q_len)?;
            let (q, k) = match self.try_apply_rotary_emb_pair_bshd(&q, &k, span.start_pos)? {
                Some(rotated) => rotated,
                None => (
                    self.apply_rotary_emb(&q.transpose(1, 2)?.contiguous()?, span.start_pos)?
                        .transpose(1, 2)?,
                    self.apply_rotary_emb(&k.transpose(1, 2)?.contiguous()?, span.start_pos)?
                        .transpose(1, 2)?,
                ),
            };
            queries.push(q.reshape((span.q_len, self.n_head, self.head_dim))?);
            keys.push(k.reshape((span.q_len, self.n_kv_head, self.head_dim))?);
        }
        // The arena may store KV narrower than the F32 activations.
        let kv_dtype = caches[0].arena().config().dtype;
        let queries = Tensor::cat(&queries, 0)?.contiguous()?.to_dtype(kv_dtype)?;
        let keys = Tensor::cat(&keys, 0)?.contiguous()?.to_dtype(kv_dtype)?;
        let values = values.contiguous()?.to_dtype(kv_dtype)?;
        let output = step.write_and_attend(
            caches,
            self.physical_layer,
            &queries,
            &keys,
            &values,
            (1.0f64 / (self.head_dim as f64).sqrt()) as f32,
            None,
            self.sliding_window,
        )?;
        let output = output
            .to_dtype(hidden_states.dtype())?
            .reshape((1, tokens, hidden_size))?;
        self.project_output(hidden_states, &output)
    }
}

impl ShortConvLayer {
    fn forward_stateless(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let projected = self.in_proj.forward(hidden_states)?.transpose(1, 2)?;
        let b = projected.narrow(1, 0, hidden_size)?;
        let c = projected.narrow(1, hidden_size, hidden_size)?;
        let x = projected.narrow(1, hidden_size * 2, hidden_size)?;
        let bx = (&b * &x)?.contiguous()?;

        let conv_weight = &self.conv;

        let conv_out = if seq_len == 1 {
            let state = Tensor::zeros(
                (batch_size, hidden_size, self.l_cache),
                bx.dtype(),
                bx.device(),
            )?;
            let fused_conv_out = if self.l_cache == 3 {
                try_lfm_shortconv_decode3(&state, &bx, conv_weight)
            } else {
                None
            };

            if let Some(conv_out) = fused_conv_out {
                conv_out
            } else {
                let state = if self.l_cache > 1 {
                    let tail = state.narrow(2, 1, self.l_cache - 1)?;
                    Tensor::cat(&[&tail, &bx], 2)?
                } else {
                    bx.clone()
                };
                (&state * &conv_weight.unsqueeze(0)?)?
                    .sum_keepdim(2)?
                    .contiguous()?
            }
        } else {
            let bx = bx.contiguous()?;
            let out = if self.l_cache == 3 {
                try_lfm_shortconv_sequence3(&bx, conv_weight)
            } else {
                None
            };

            if let Some(out) = out {
                out
            } else if self.l_cache != 3 {
                // LFM3's width-7 filters: Candle runs a grouped `Conv1d` as
                // one convolution per channel, a dispatch each on a GPU.
                depthwise_causal_conv(&bx, conv_weight)?
            } else {
                let conv = Conv1d::new(
                    conv_weight
                        .reshape((hidden_size, 1, self.l_cache))?
                        .contiguous()?,
                    None,
                    Conv1dConfig {
                        padding: self.l_cache.saturating_sub(1),
                        groups: hidden_size,
                        ..Default::default()
                    },
                );
                conv.forward(&bx)?.narrow(2, 0, seq_len)?
            }
        };

        let conv_out = (&c * &conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out).map_err(Error::from)
    }

    fn forward_physical(
        &self,
        hidden_states: &Tensor,
        transaction: &mut InvocationRingDepthwiseConvTransaction<'_>,
    ) -> Result<Tensor> {
        let (_, _, hidden_size) = hidden_states.dims3()?;
        let projected = self.in_proj.forward(hidden_states)?.transpose(1, 2)?;
        let b = projected.narrow(1, 0, hidden_size)?;
        let c = projected.narrow(1, hidden_size, hidden_size)?;
        let x = projected.narrow(1, hidden_size * 2, hidden_size)?;
        let bx = (&b * &x)?.contiguous()?;
        let conv_out = transaction.apply(self.component, &bx, &self.conv)?;
        let conv_out = (&c * &conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out).map_err(Error::from)
    }

    fn forward_retained(
        &self,
        hidden_states: &Tensor,
        state: &mut Lfm2ShortConvRuntimeState,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden_states.dims3()?;
        if batch == 1 && seq_len == 1 {
            let projected = self.in_proj.forward(hidden_states)?;
            if let Some(y) = state.apply_fused_step(self.component, &projected, &self.conv)? {
                return self.out_proj.forward(&y).map_err(Error::from);
            }
            return self.forward_retained_unfused(projected, hidden_size, state);
        }
        let projected = self.in_proj.forward(hidden_states)?;
        self.forward_retained_unfused(projected, hidden_size, state)
    }

    fn forward_retained_unfused(
        &self,
        projected: Tensor,
        hidden_size: usize,
        state: &mut Lfm2ShortConvRuntimeState,
    ) -> Result<Tensor> {
        let conv_out = self.gated_conv_retained(projected, hidden_size, state)?;
        self.out_proj.forward(&conv_out).map_err(Error::from)
    }

    /// The gated convolution `[1, steps, hidden]` of one sequence's
    /// in-projection `[1, steps, 3 * hidden]`, continuing `state`.
    fn gated_conv_retained(
        &self,
        projected: Tensor,
        hidden_size: usize,
        state: &mut Lfm2ShortConvRuntimeState,
    ) -> Result<Tensor> {
        let projected = projected.transpose(1, 2)?;
        let b = projected.narrow(1, 0, hidden_size)?;
        let c = projected.narrow(1, hidden_size, hidden_size)?;
        let x = projected.narrow(1, hidden_size * 2, hidden_size)?;
        let bx = (&b * &x)?.contiguous()?;
        let conv_out = state.apply(self.component, &bx, &self.conv)?;
        (&c * &conv_out)?
            .transpose(1, 2)?
            .contiguous()
            .map_err(Error::from)
    }

    /// One mixed step: the in- and out-projections run on every token, and
    /// each row's segment convolves with its own state the way its split
    /// call does (a one-token row through the fused step, a span through
    /// the chunk path).
    fn forward_mixed(
        &self,
        hidden_states: &Tensor,
        layout: &MixedRowLayout,
        states: &mut [Lfm2ShortConvRuntimeState],
    ) -> Result<Tensor> {
        let (_, tokens, hidden_size) = hidden_states.dims3()?;
        if tokens != layout.total_tokens() || states.len() != layout.rows().len() {
            return Err(Error::InvalidInput(
                "LFM2 ShortConv mixed rows do not match the step".into(),
            ));
        }
        let projected = self.in_proj.forward(hidden_states)?;
        let mut rows = Vec::with_capacity(states.len());
        for (row, (span, state)) in layout.rows().iter().zip(states.iter_mut()).enumerate() {
            let projected = projected.narrow(1, layout.token_offset(row), span.q_len)?;
            let fused = if span.q_len == 1 {
                state.apply_fused_step(self.component, &projected, &self.conv)?
            } else {
                None
            };
            rows.push(match fused {
                Some(y) => y,
                None => self.gated_conv_retained(projected, hidden_size, state)?,
            });
        }
        let conv = Tensor::cat(&rows, 1)?;
        self.out_proj.forward(&conv).map_err(Error::from)
    }

    fn forward_retained_decode_batch(
        &self,
        hidden_states: &Tensor,
        states: &mut [&mut Lfm2ShortConvRuntimeState],
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden_states.dims3()?;
        if batch == 0 || seq_len != 1 || states.len() != batch {
            return Err(Error::InvalidInput(
                "LFM2 ShortConv decode batch dimensions do not match".into(),
            ));
        }
        let projected = self.in_proj.forward(hidden_states)?;
        let mut fused = Vec::with_capacity(batch);
        for row in 0..batch {
            match states[row].apply_fused_step(
                self.component,
                &projected.narrow(0, row, 1)?,
                &self.conv,
            )? {
                Some(y) => fused.push(y),
                None => break,
            }
        }
        if fused.len() == batch {
            let y = if batch == 1 {
                fused.pop().expect("one fused row")
            } else {
                Tensor::cat(&fused.iter().collect::<Vec<_>>(), 0)?
            };
            return self.out_proj.forward(&y).map_err(Error::from);
        }
        if !fused.is_empty() {
            return Err(Error::InferenceError(
                "LFM2 fused ShortConv switched off mid-batch".into(),
            ));
        }
        let projected = projected.transpose(1, 2)?;
        let b = projected.narrow(1, 0, hidden_size)?;
        let c = projected.narrow(1, hidden_size, hidden_size)?;
        let x = projected.narrow(1, hidden_size * 2, hidden_size)?;
        let bx = (&b * &x)?.contiguous()?;
        let mut rows = Vec::with_capacity(batch);
        for row in 0..batch {
            rows.push(states[row].apply(self.component, &bx.narrow(0, row, 1)?, &self.conv)?);
        }
        let conv = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?;
        let conv = (&c * &conv)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv).map_err(Error::from)
    }
}

impl ProjectionHead {
    /// The output projection, or for a checkpoint without one the tied token
    /// table: `token_embeddings` when the backbone already holds it, else a
    /// fresh read.
    fn load(
        loader: &GgufLoader,
        device: &Device,
        token_embeddings: Option<&Lfm2QuantizedWeight>,
    ) -> Result<Self> {
        let untied = [
            "output.weight",
            "lm_head.weight",
            "model.output.weight",
            "model.lm_head.weight",
            "lfm.output.weight",
            "lfm.lm_head.weight",
            "dense_2_out.weight",
            "dense_2.weight",
            "lin.weight",
        ]
        .map(String::from);
        let weight = match token_embeddings {
            Some(tied) if !untied.iter().any(|name| loader.has_tensor(name)) => {
                Lfm2Linear::new(tied.clone())?
            }
            _ => {
                let mut names = untied.to_vec();
                names.extend(["token_embd.weight", "tok_embeddings.weight"].map(String::from));
                load_linear_any(loader, device, &names)?
            }
        };
        let bias = load_optional_bias_any(
            loader,
            device,
            &[
                "output.bias".to_string(),
                "lm_head.bias".to_string(),
                "lfm.output.bias".to_string(),
                "lfm.lm_head.bias".to_string(),
                "dense_2_out.bias".to_string(),
                "dense_2.bias".to_string(),
                "lin.bias".to_string(),
            ],
        )?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let diagnose = diagnostics::enabled();
        if diagnose {
            diagnostics::validate_finite(hidden_states, "LFM2 output projection input")?;
        }
        let projected = self.weight.forward(hidden_states)?;
        if diagnose {
            diagnostics::validate_finite(&projected, "LFM2 output projection QMatMul")?;
        }
        if let Some(bias) = &self.bias {
            let output = projected.broadcast_add(bias)?;
            if diagnose {
                diagnostics::validate_finite(&output, "LFM2 output projection bias addition")?;
            }
            Ok(output)
        } else {
            Ok(projected)
        }
    }
}

impl QuantizedLfm2Backbone {
    pub fn load(loader: &GgufLoader, cfg: Lfm2BackboneConfig, device: &Device) -> Result<Self> {
        Self::load_inner(loader, cfg, device, BackboneParts::Full)
    }

    /// A backbone that only ever runs on supplied embeddings, such as the
    /// LFM2.5-Audio detokenizer: its token-embedding table is never read, so
    /// it is not loaded.
    pub fn load_for_embeds(
        loader: &GgufLoader,
        cfg: Lfm2BackboneConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::load_inner(loader, cfg, device, BackboneParts::ForEmbeds)
    }

    /// The layers and final norm alone, run on supplied embeddings and
    /// returning hidden states: neither a token table nor an output head
    /// (LFM3-Audio's encoder and LiquidCodec decoder).
    pub fn load_trunk(
        loader: &GgufLoader,
        cfg: Lfm2BackboneConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::load_inner(loader, cfg, device, BackboneParts::Trunk)
    }

    fn load_inner(
        loader: &GgufLoader,
        cfg: Lfm2BackboneConfig,
        device: &Device,
        parts: BackboneParts,
    ) -> Result<Self> {
        let state_layout = Lfm2StateLayout::from_config(&cfg)?;
        let (context_length, head_dim) = rope_table_geometry(&cfg)?;
        let lfm3 = cfg.lfm3.clone().unwrap_or_default();
        let (cos, sin) = precompute_freqs_scaled(
            head_dim,
            cfg.rope_freq_base,
            lfm3.rope_freq_factors.as_deref(),
            context_length,
            device,
        )?;
        let cos_sin = Tensor::cat(&[&cos, &sin], 1)?.contiguous()?;
        // LFM3's sliding-window layers rotate at their own base.
        let swa_tables = lfm3
            .rope_freq_base_swa
            .map(|base| -> Result<(Tensor, Tensor, Tensor)> {
                let (cos, sin) = precompute_freqs(head_dim, base, context_length, device)?;
                let cos_sin = Tensor::cat(&[&cos, &sin], 1)?.contiguous()?;
                Ok((cos, sin, cos_sin))
            })
            .transpose()?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let embedding_names = [
            "token_embd.weight".to_string(),
            "tok_embeddings.weight".to_string(),
            "model.embed_tokens.weight".to_string(),
            "lfm.embed_tokens.weight".to_string(),
            "emb.emb.weight".to_string(),
            "emb.weight".to_string(),
        ];
        // Token rows are gathered straight from the quantized table
        // (`QTensor::embedding`, all backends), which a tied LM head shares:
        // no dequantized copy (512 MiB F32 for LFM2.5's 65,536 x 2,048
        // vocabulary) and no second quantized read for the head.
        let (token_embeddings, vocab_size, hidden_size) = if parts == BackboneParts::Trunk {
            (None, 0, cfg.embedding_length)
        } else if parts == BackboneParts::Full {
            let token_embedding_q = load_weight_any(loader, device, &embedding_names)?;
            let (vocab_size, hidden_size) = token_embedding_q.tensor.shape().dims2()?;
            (Some(token_embedding_q), vocab_size, hidden_size)
        } else {
            let shape = embedding_names
                .iter()
                .find_map(|name| loader.tensor_shape(name))
                .ok_or_else(|| {
                    Error::ModelLoadError("LFM2 GGUF has no token-embedding tensor".into())
                })?;
            let [vocab_size, hidden_size] = shape[..] else {
                return Err(Error::ModelLoadError(format!(
                    "LFM2 token embedding has shape {shape:?}, expected two dimensions"
                )));
            };
            (None, vocab_size, hidden_size)
        };
        if hidden_size != cfg.embedding_length {
            return Err(Error::ModelLoadError(format!(
                "LFM2 embedding width mismatch: GGUF has {hidden_size}, metadata says {}",
                cfg.embedding_length
            )));
        }
        let norm = LfmRmsNorm::from_qtensor(
            load_qtensor_any(
                loader,
                device,
                &[
                    "output_norm.weight".to_string(),
                    "embedding_norm.weight".to_string(),
                    "model.embedding_norm.weight".to_string(),
                    "model.embedding_norm".to_string(),
                    "token_embd_norm.weight".to_string(),
                    "lfm.embedding_norm.weight".to_string(),
                ],
            )?,
            cfg.attention_layer_norm_rms_epsilon,
        )?;
        let output_head = (parts != BackboneParts::Trunk)
            .then(|| ProjectionHead::load(loader, device, token_embeddings.as_ref()))
            .transpose()?;

        let mut layers = Vec::with_capacity(cfg.block_count);
        for layer_idx in 0..cfg.block_count {
            let prefix = format!("blk.{layer_idx}");
            let legacy_prefix = format!("lfm.layers.{layer_idx}");
            let operator_norm = LfmRmsNorm::from_qtensor(
                load_qtensor_any(
                    loader,
                    device,
                    &[
                        format!("{prefix}.attn_norm.weight"),
                        format!("{prefix}.operator_norm.weight"),
                        format!("{prefix}.attention_norm.weight"),
                        format!("{legacy_prefix}.operator_norm.weight"),
                    ],
                )?,
                cfg.attention_layer_norm_rms_epsilon,
            )?;
            let ffn_norm = LfmRmsNorm::from_qtensor(
                load_qtensor_any(
                    loader,
                    device,
                    &[
                        format!("{prefix}.ffn_norm.weight"),
                        format!("{prefix}.ffn_norm"),
                        format!("{legacy_prefix}.ffn_norm.weight"),
                    ],
                )?,
                cfg.attention_layer_norm_rms_epsilon,
            )?;
            let mlp =
                Mlp::load_with_prefixes(loader, device, &[prefix.clone(), legacy_prefix.clone()])?;
            let (post_operator_norm, post_ffn_norm) = if lfm3.sandwich_norm {
                let norm = |names: [String; 3]| -> Result<LfmRmsNorm> {
                    LfmRmsNorm::from_qtensor(
                        load_qtensor_any(loader, device, &names)?,
                        cfg.attention_layer_norm_rms_epsilon,
                    )
                };
                (
                    Some(norm([
                        format!("{prefix}.post_attention_norm.weight"),
                        format!("{prefix}.post_operator_norm.weight"),
                        format!("{legacy_prefix}.post_operator_norm.weight"),
                    ])?),
                    Some(norm([
                        format!("{prefix}.post_ffw_norm.weight"),
                        format!("{prefix}.post_ffn_norm.weight"),
                        format!("{legacy_prefix}.post_ffn_norm.weight"),
                    ])?),
                )
            } else {
                (None, None)
            };

            let is_attention = cfg
                .attention_head_count_kv
                .get(layer_idx)
                .copied()
                .unwrap_or(cfg.attention_head_count)
                > 0;
            let kind = if is_attention {
                let n_kv_head = cfg.attention_head_count_kv[layer_idx];
                let physical_layer = state_layout
                    .attention_model_layers
                    .iter()
                    .position(|model_layer| *model_layer as usize == layer_idx)
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "LFM2 attention layer {layer_idx} is absent from physical layout"
                        ))
                    })?;
                let q_norm = LfmRmsNorm::from_qtensor(
                    load_qtensor_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_q_norm.weight"),
                            format!("{prefix}.self_attn.q_layernorm.weight"),
                            format!("{prefix}.attention.q_norm.weight"),
                            format!("{legacy_prefix}.self_attn.q_layernorm.weight"),
                        ],
                    )?,
                    cfg.attention_layer_norm_rms_epsilon,
                )?;
                let k_norm = LfmRmsNorm::from_qtensor(
                    load_qtensor_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_k_norm.weight"),
                            format!("{prefix}.self_attn.k_layernorm.weight"),
                            format!("{prefix}.attention.k_norm.weight"),
                            format!("{legacy_prefix}.self_attn.k_layernorm.weight"),
                        ],
                    )?,
                    cfg.attention_layer_norm_rms_epsilon,
                )?;
                let qk_norm_weight =
                    Tensor::cat(&[q_norm.weight(), k_norm.weight()], 0)?.contiguous()?;
                let (layer_cos, layer_sin, layer_cos_sin) = match &swa_tables {
                    Some((cos, sin, cos_sin)) if cfg.layer_uses_swa_rope(layer_idx) => {
                        (cos.clone(), sin.clone(), cos_sin.clone())
                    }
                    _ => (cos.clone(), sin.clone(), cos_sin.clone()),
                };
                let gate = lfm3
                    .attention_output_gate
                    .then(|| {
                        load_linear_any(
                            loader,
                            device,
                            &[
                                format!("{prefix}.attn_gate.weight"),
                                format!("{prefix}.self_attn.gate_proj.weight"),
                                format!("{legacy_prefix}.self_attn.gate_proj.weight"),
                            ],
                        )
                    })
                    .transpose()?;
                LayerKind::Attention(AttentionLayer {
                    wq: load_linear_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_q.weight"),
                            format!("{prefix}.self_attn.q_proj.weight"),
                            format!("{legacy_prefix}.self_attn.q_proj.weight"),
                        ],
                    )?,
                    wk: load_linear_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_k.weight"),
                            format!("{prefix}.self_attn.k_proj.weight"),
                            format!("{legacy_prefix}.self_attn.k_proj.weight"),
                        ],
                    )?,
                    wv: load_linear_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_v.weight"),
                            format!("{prefix}.self_attn.v_proj.weight"),
                            format!("{legacy_prefix}.self_attn.v_proj.weight"),
                        ],
                    )?,
                    wo: load_linear_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_output.weight"),
                            format!("{prefix}.self_attn.out_proj.weight"),
                            format!("{legacy_prefix}.self_attn.out_proj.weight"),
                        ],
                    )?,
                    q_norm,
                    k_norm,
                    qk_norm_weight,
                    n_head: cfg.attention_head_count,
                    n_kv_head,
                    head_dim: cfg.embedding_length / cfg.attention_head_count,
                    sliding_window: cfg.layer_sliding_window(layer_idx),
                    cos: layer_cos,
                    sin: layer_sin,
                    cos_sin: layer_cos_sin,
                    neg_inf: neg_inf.clone(),
                    physical_layer,
                    model_layer: layer_idx,
                    gate,
                })
            } else {
                LayerKind::ShortConv(ShortConvLayer {
                    in_proj: load_linear_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.shortconv.in_proj.weight"),
                            format!("{prefix}.conv.in_proj.weight"),
                            format!("{legacy_prefix}.conv.in_proj.weight"),
                        ],
                    )?,
                    out_proj: load_linear_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.shortconv.out_proj.weight"),
                            format!("{prefix}.conv.out_proj.weight"),
                            format!("{legacy_prefix}.conv.out_proj.weight"),
                        ],
                    )?,
                    conv: normalize_shortconv_weight(
                        load_dense_any(
                            loader,
                            device,
                            &[
                                format!("{prefix}.shortconv.conv.weight"),
                                format!("{prefix}.conv.conv.weight"),
                                format!("{prefix}.shortconv.conv"),
                                format!("{legacy_prefix}.conv.conv.weight"),
                            ],
                            Some(DType::F32),
                        )?,
                        cfg.shortconv_l_cache,
                        cfg.embedding_length,
                    )?,
                    l_cache: cfg.shortconv_l_cache,
                    component: state_layout.shortconv_component(layer_idx)?,
                })
            };

            layers.push(LayerWeights {
                operator_norm,
                ffn_norm,
                mlp,
                kind,
                post_operator_norm,
                post_ffn_norm,
            });
        }

        Ok(Self {
            cfg,
            token_embeddings: token_embeddings.map(|weight| weight.tensor),
            output_head,
            layers,
            norm,
            vocab_size,
            state_layout,
        })
    }

    pub fn config(&self) -> &Lfm2BackboneConfig {
        &self.cfg
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.embedding_length
    }

    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.token_embeddings
            .as_ref()
            .ok_or_else(|| {
                Error::InferenceError("LFM2 backbone was loaded without token embeddings".into())
            })?
            .embedding(token_ids)
            .map_err(Error::from)
    }

    fn head(&self) -> Result<&ProjectionHead> {
        self.output_head.as_ref().ok_or_else(|| {
            Error::InferenceError("LFM2 trunk was loaded without an output head".into())
        })
    }

    pub fn project_hidden(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.head()?.forward(hidden_states)
    }

    pub fn project_last_hidden(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        let last_hidden = hidden_states.i((.., seq_len - 1, ..))?;
        self.head()?.forward(&last_hidden)
    }

    pub(crate) fn forward_tokens_physical(
        &self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
    ) -> Result<Tensor> {
        let hidden_states = self.embed_tokens(token_ids)?;
        let hidden_states =
            self.forward_embeds_physical(&hidden_states, index_pos, cache, shortconv)?;
        self.project_last_hidden(&hidden_states)
    }

    pub(crate) fn new_shortconv_state(&self) -> Lfm2ShortConvRuntimeState {
        Lfm2ShortConvRuntimeState::new(self)
    }

    pub(crate) fn forward_tokens_retained(
        &self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut Lfm2ShortConvRuntimeState,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        let hidden_states = self.embed_tokens(token_ids)?;
        let hidden_states =
            self.forward_embeds_retained(&hidden_states, index_pos, cache, shortconv)?;
        if compute_logits {
            self.project_last_hidden(&hidden_states).map(Some)
        } else {
            Ok(None)
        }
    }

    pub(crate) fn forward_token_ids_retained_batch(
        &self,
        token_ids: &[u32],
        positions: &[usize],
        states: &mut [&mut Lfm2ShortConvRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let batch = token_ids.len();
        if batch == 0 {
            return Err(Error::InvalidInput(
                "LFM2 retained decode batch is empty".into(),
            ));
        }
        let input = Tensor::from_slice(token_ids, (batch, 1), self.norm.weight.device())?;
        let embeds = self.embed_tokens(&input)?;
        let hidden = self.forward_embeds_retained_step_batch(&embeds, positions, states, caches)?;
        self.project_last_hidden(&hidden)
    }

    /// [`Self::forward_token_ids_retained_batch`] without waiting for the
    /// step's KV writes: the logits are returned with the step's pending
    /// commit, and the rows' caches and ShortConv states stay unchanged until
    /// [`Lfm2PendingDecodeCommit::commit`]. A caller that reads the logits back
    /// first commits after the device has already drained, so the step pays
    /// one synchronization instead of a fence wait before the LM head plus a
    /// readback after it.
    pub(crate) fn forward_token_ids_retained_batch_deferred(
        &self,
        token_ids: &[u32],
        positions: &[usize],
        states: &[&Lfm2ShortConvRuntimeState],
        caches: &[&PhysicalPagedKvCache],
    ) -> Result<(Tensor, Lfm2PendingDecodeCommit)> {
        let batch = token_ids.len();
        if batch == 0 {
            return Err(Error::InvalidInput(
                "LFM2 retained decode batch is empty".into(),
            ));
        }
        let input = Tensor::from_slice(token_ids, (batch, 1), self.norm.weight.device())?;
        let embeds = self.embed_tokens(&input)?;
        let (hidden, pending) =
            self.forward_embeds_retained_step_batch_deferred(&embeds, positions, states, caches)?;
        match self.project_last_hidden(&hidden) {
            Ok(logits) => Ok((logits, pending)),
            Err(error) => Err(pending.abort_with(error)),
        }
    }

    fn forward_embeds_retained_step_batch(
        &self,
        input_embeds: &Tensor,
        positions: &[usize],
        states: &mut [&mut Lfm2ShortConvRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let (hidden, pending) = {
            let state_refs = states.iter().map(|state| &**state).collect::<Vec<_>>();
            let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            self.forward_embeds_retained_step_batch_deferred(
                input_embeds,
                positions,
                &state_refs,
                &cache_refs,
            )?
        };
        pending.commit(states, caches)?;
        Ok(hidden)
    }

    fn forward_embeds_retained_step_batch_deferred(
        &self,
        input_embeds: &Tensor,
        positions: &[usize],
        states: &[&Lfm2ShortConvRuntimeState],
        caches: &[&PhysicalPagedKvCache],
    ) -> Result<(Tensor, Lfm2PendingDecodeCommit)> {
        let (batch, seq_len, hidden_size) = input_embeds.dims3()?;
        if batch == 0
            || seq_len != 1
            || hidden_size != self.cfg.embedding_length
            || positions.len() != batch
            || states.len() != batch
            || caches.len() != batch
        {
            return Err(Error::InvalidInput(
                "LFM2 retained embedding batch rows do not match".into(),
            ));
        }
        for row in 0..batch {
            if caches[row].context_len() != positions[row]
                || states[row].cursor() != positions[row] as u64
            {
                return Err(Error::InvalidInput(
                    "LFM2 retained embedding row cursors do not match".into(),
                ));
            }
        }
        let first = caches[0];
        let slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let lowered = first.arena().lower_slots(&slots)?;
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;
        let mut working = states
            .iter()
            .map(|state| (**state).clone())
            .collect::<Vec<_>>();
        let execution = (|| -> Result<Tensor> {
            let mut hidden_states = input_embeds.clone();
            let mut rotary_rows = Vec::new();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                diagnostics::check_batch_layer(&hidden_states, "input", layer_idx, positions)?;
                let residual = hidden_states.clone();
                let hidden = layer.operator_norm.forward(&hidden_states)?;
                diagnostics::check_batch_layer(&hidden, "operator norm", layer_idx, positions)?;
                let hidden = match &layer.kind {
                    LayerKind::Attention(attention) => attention.forward_retained_decode_batch(
                        &hidden,
                        positions,
                        caches,
                        lowered.as_ref(),
                        &mut completions,
                        &mut rotary_rows,
                    )?,
                    LayerKind::ShortConv(shortconv) => {
                        let mut refs = working.iter_mut().collect::<Vec<_>>();
                        shortconv.forward_retained_decode_batch(&hidden, &mut refs)?
                    }
                };
                let hidden = layer.operator_output(hidden)?;
                diagnostics::check_batch_layer(&hidden, "operator output", layer_idx, positions)?;
                hidden_states = (&hidden + &residual)?;
                diagnostics::check_batch_layer(&hidden_states, "residual", layer_idx, positions)?;
                let residual = hidden_states.clone();
                let hidden = layer.ffn_norm.forward(&hidden_states)?;
                diagnostics::check_batch_layer(&hidden, "FFN norm", layer_idx, positions)?;
                let mlp = layer.ffn_output(&hidden)?;
                diagnostics::check_batch_layer(&mlp, "MLP output", layer_idx, positions)?;
                hidden_states = (&mlp + &residual)?;
                diagnostics::check_batch_layer(
                    &hidden_states,
                    "MLP residual",
                    layer_idx,
                    positions,
                )?;
            }
            let normalized = self.norm.forward(&hidden_states)?;
            if diagnostics::enabled() {
                for (row, &position) in positions.iter().enumerate() {
                    diagnostics::check(
                        &normalized.narrow(0, row, 1)?,
                        &format!("row {row} final norm"),
                        position,
                    )?;
                }
            }
            Ok(normalized)
        })();
        let hidden = match execution {
            Ok(hidden) => hidden,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                        "LFM2 decode batch failed: {error}; fence drain failed: {drain}"
                    ))),
                }
            }
        };
        for state in &mut working {
            if let Err(error) = state.advance(1) {
                return Err(Lfm2PendingDecodeCommit {
                    completions,
                    working: Vec::new(),
                    positions: Vec::new(),
                }
                .abort_with(error));
            }
        }
        Ok((
            hidden,
            Lfm2PendingDecodeCommit {
                completions,
                working,
                positions: positions.to_vec(),
            },
        ))
    }

    /// One forward over a mixed step: `token_ids` holds every row's new
    /// tokens in `layout` order (one-token decode rows, then prefill spans);
    /// `states[i]` and `caches[i]` are row `i`'s. See
    /// [`Self::forward_embeds_mixed_deferred`].
    pub(crate) fn forward_token_ids_mixed_deferred(
        &self,
        token_ids: &[u32],
        layout: MixedRowLayout,
        states: &[&Lfm2ShortConvRuntimeState],
        caches: &[&PhysicalPagedKvCache],
        sample_rows: &[usize],
    ) -> Result<(Option<Tensor>, Lfm2PendingMixedCommit)> {
        let input = Tensor::from_slice(token_ids, (1, token_ids.len()), self.norm.weight.device())?;
        let embeds = self.embed_tokens(&input)?;
        self.forward_embeds_mixed_deferred(&embeds, layout, states, caches, sample_rows)
    }

    /// Every token of a mixed step (`[1, tokens, hidden]` in `layout` order)
    /// through the projections, norms and MLP together; attention writes
    /// and attends once per layer over all rows, and each row's ShortConv
    /// segment continues its own state. Returns the `[sample_rows, vocab]`
    /// logits of the last token of each row in `sample_rows` (`None` when no
    /// row samples) with the step's pending commit: no cache or state
    /// changes until [`Lfm2PendingMixedCommit::commit`].
    pub(crate) fn forward_embeds_mixed_deferred(
        &self,
        input_embeds: &Tensor,
        layout: MixedRowLayout,
        states: &[&Lfm2ShortConvRuntimeState],
        caches: &[&PhysicalPagedKvCache],
        sample_rows: &[usize],
    ) -> Result<(Option<Tensor>, Lfm2PendingMixedCommit)> {
        let (batch, tokens, hidden_size) = input_embeds.dims3()?;
        if batch != 1
            || tokens != layout.total_tokens()
            || hidden_size != self.cfg.embedding_length
            || states.len() != layout.rows().len()
            || caches.len() != layout.rows().len()
            || sample_rows.iter().any(|row| *row >= caches.len())
        {
            return Err(Error::InvalidInput(
                "LFM2 mixed step inputs do not match its row layout".into(),
            ));
        }
        let head_dim = self.cfg.embedding_length / self.cfg.attention_head_count;
        let sparse_geometry = self
            .state_layout
            .attention_model_layers
            .iter()
            .map(|model_layer| {
                (
                    *model_layer,
                    self.cfg.attention_head_count_kv[*model_layer as usize],
                    head_dim,
                    head_dim,
                )
            })
            .collect::<Vec<_>>();
        for ((cache, state), span) in caches.iter().zip(states).zip(layout.rows()) {
            if cache.context_len() != span.start_pos || state.cursor() != span.start_pos as u64 {
                return Err(Error::InvalidInput(
                    "LFM2 mixed row cursors do not match their state".into(),
                ));
            }
            cache.validate_sparse_model_layers(&sparse_geometry)?;
        }
        let sample_tokens = sample_rows
            .iter()
            .map(|row| u32::try_from(layout.last_token(*row)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::InvalidInput("LFM2 mixed step exceeds u32 tokens".into()))?;
        let mut step = PreparedPhysicalMixedStep::new(caches, layout)?;
        let mut working = states
            .iter()
            .map(|state| (**state).clone())
            .collect::<Vec<_>>();
        let execution = (|| -> Result<Option<Tensor>> {
            let layout = step.layout().clone();
            let mut hidden_states = input_embeds.clone();
            for layer in &self.layers {
                let residual = hidden_states.clone();
                let hidden = layer.operator_norm.forward(&hidden_states)?;
                let hidden = match &layer.kind {
                    LayerKind::Attention(attention) => {
                        attention.forward_mixed(&hidden, caches, &mut step)?
                    }
                    LayerKind::ShortConv(shortconv) => {
                        shortconv.forward_mixed(&hidden, &layout, &mut working)?
                    }
                };
                let hidden = layer.operator_output(hidden)?;
                hidden_states = (&hidden + &residual)?;
                let residual = hidden_states.clone();
                let hidden = layer.ffn_norm.forward(&hidden_states)?;
                let mlp = layer.ffn_output(&hidden)?;
                hidden_states = (&mlp + &residual)?;
            }
            if sample_tokens.is_empty() {
                return Ok(None);
            }
            let normalized = self.norm.forward(&hidden_states)?.squeeze(0)?;
            let sample_tokens =
                Tensor::from_vec(sample_tokens, sample_rows.len(), normalized.device())?;
            self.head()?
                .forward(&normalized.index_select(&sample_tokens, 0)?)
                .map(Some)
        })();
        let advanced = execution.and_then(|logits| {
            for (state, span) in working.iter_mut().zip(step.layout().rows()) {
                state.advance(span.q_len)?;
            }
            Ok(logits)
        });
        let pending = Lfm2PendingMixedCommit { step, working };
        match advanced {
            Ok(logits) => Ok((logits, pending)),
            Err(error) => Err(pending.abort_with(error)),
        }
    }

    pub(crate) fn forward_embeds_retained_batch(
        &self,
        input_embeds: &[Tensor],
        positions: &[usize],
        states: &mut [&mut Lfm2ShortConvRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Lfm2RetainedEmbeddingBatchOutput> {
        let batch = input_embeds.len();
        if batch == 0 || positions.len() != batch || states.len() != batch || caches.len() != batch
        {
            return Err(Error::InvalidInput(
                "LFM2 retained ragged embedding rows do not match".into(),
            ));
        }
        let mut spans = Vec::with_capacity(batch);
        for row in 0..batch {
            let (embed_batch, span, hidden) = input_embeds[row].dims3()?;
            if embed_batch != 1
                || span == 0
                || hidden != self.cfg.embedding_length
                || caches[row].context_len() != positions[row]
                || states[row].cursor() != positions[row] as u64
            {
                return Err(Error::InvalidInput(
                    "LFM2 retained ragged embedding geometry is invalid".into(),
                ));
            }
            spans.push(span);
        }
        let waves = retained_ragged_launch_rows(&spans)?;
        let mut last_hidden = (0..batch).map(|_| None).collect::<Vec<_>>();
        let mut launch_widths = Vec::with_capacity(waves.len());
        for (offset, active) in waves.into_iter().enumerate() {
            let embeds = Tensor::cat(
                &active
                    .iter()
                    .map(|row| input_embeds[*row].narrow(1, offset, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?,
                0,
            )?;
            let wave_positions = active
                .iter()
                .map(|row| positions[*row] + offset)
                .collect::<Vec<_>>();
            let mut wave_states = states
                .iter_mut()
                .enumerate()
                .filter_map(|(row, state)| active.contains(&row).then_some(&mut **state))
                .collect::<Vec<_>>();
            let mut wave_caches = caches
                .iter_mut()
                .enumerate()
                .filter_map(|(row, cache)| active.contains(&row).then_some(&mut **cache))
                .collect::<Vec<_>>();
            let hidden = self.forward_embeds_retained_step_batch(
                &embeds,
                &wave_positions,
                &mut wave_states,
                &mut wave_caches,
            )?;
            for (wave_row, row) in active.iter().copied().enumerate() {
                if offset + 1 == spans[row] {
                    last_hidden[row] = Some(hidden.narrow(0, wave_row, 1)?);
                }
            }
            launch_widths.push(active.len());
        }
        Ok(Lfm2RetainedEmbeddingBatchOutput {
            last_hidden: last_hidden
                .into_iter()
                .map(|hidden| {
                    hidden.ok_or_else(|| {
                        Error::InferenceError(
                            "LFM2 retained ragged batch lost a terminal hidden row".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            launch_widths,
        })
    }

    /// Advance the retained B1 backbone from caller-supplied embeddings.
    ///
    /// This is the model-neutral seam used by LFM2.5 Audio when its prompt
    /// contains projected audio spans rather than token IDs. The retained KV
    /// and ShortConv clocks remain authoritative and are advanced atomically
    /// only after the complete backbone forward succeeds.
    pub(crate) fn forward_embeds_retained(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut Lfm2ShortConvRuntimeState,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = input_embeds.dims3()?;
        if batch != 1
            || hidden != self.cfg.embedding_length
            || index_pos != cache.context_len()
            || shortconv.cursor() != index_pos as u64
        {
            return Err(Error::InvalidInput(
                "LFM2 retained input does not match its physical state".into(),
            ));
        }
        let head_dim = self.cfg.embedding_length / self.cfg.attention_head_count;
        let sparse_geometry = self
            .state_layout
            .attention_model_layers
            .iter()
            .map(|model_layer| {
                (
                    *model_layer,
                    self.cfg.attention_head_count_kv[*model_layer as usize],
                    head_dim,
                    head_dim,
                )
            })
            .collect::<Vec<_>>();
        cache.validate_sparse_model_layers(&sparse_geometry)?;
        let mut prepared = match self.cfg.attention_sliding_window {
            Some(window_tokens) => {
                cache.prepare_append_with_window(index_pos, seq_len, window_tokens)?
            }
            None => cache.prepare_append(index_pos, seq_len)?,
        };
        let mut working = shortconv.clone();
        let mut hidden_states = input_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            diagnostics::check_layer(&hidden_states, "input", layer_idx, index_pos)?;
            let residual = hidden_states.clone();
            let hidden = layer.operator_norm.forward(&hidden_states)?;
            diagnostics::check_layer(&hidden, "operator norm", layer_idx, index_pos)?;
            let hidden = match &layer.kind {
                LayerKind::Attention(attention) => {
                    attention.forward_physical(&hidden, index_pos, cache, &mut prepared)?
                }
                LayerKind::ShortConv(layer) => layer.forward_retained(&hidden, &mut working)?,
            };
            let hidden = layer.operator_output(hidden)?;
            diagnostics::check_layer(&hidden, "operator output", layer_idx, index_pos)?;
            hidden_states = (&hidden + &residual)?;
            diagnostics::check_layer(&hidden_states, "residual", layer_idx, index_pos)?;
            let residual = hidden_states.clone();
            let hidden = layer.ffn_norm.forward(&hidden_states)?;
            diagnostics::check_layer(&hidden, "FFN norm", layer_idx, index_pos)?;
            let mlp = layer.ffn_output(&hidden)?;
            diagnostics::check_layer(&mlp, "MLP output", layer_idx, index_pos)?;
            hidden_states = (&mlp + &residual)?;
            diagnostics::check_layer(&hidden_states, "MLP residual", layer_idx, index_pos)?;
        }
        let hidden_states = self.norm.forward(&hidden_states)?;
        diagnostics::check(&hidden_states, "final norm", index_pos)?;
        working.advance(seq_len)?;
        cache.commit_prepared(prepared)?;
        *shortconv = working;
        Ok(hidden_states)
    }

    pub(crate) fn forward_embeds_physical(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
    ) -> Result<Tensor> {
        let (batch, _seq_len, hidden) = input_embeds.dims3()?;
        if batch != 1 || hidden != self.cfg.embedding_length || index_pos != cache.context_len() {
            return Err(Error::InvalidInput(
                "LFM2 physical input does not match its invocation state".into(),
            ));
        }
        let head_dim = self.cfg.embedding_length / self.cfg.attention_head_count;
        let sparse_geometry = self
            .state_layout
            .attention_model_layers
            .iter()
            .map(|model_layer| {
                (
                    *model_layer,
                    self.cfg.attention_head_count_kv[*model_layer as usize],
                    head_dim,
                    head_dim,
                )
            })
            .collect::<Vec<_>>();
        cache.validate_sparse_model_layers(&sparse_geometry)?;

        self.forward_embeds_physical_chunk(
            input_embeds,
            index_pos,
            cache,
            shortconv,
            self.cfg.attention_sliding_window,
        )
    }

    fn forward_embeds_physical_chunk(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = input_embeds.dims3()?;
        let mut prepared = match sliding_window {
            Some(window_tokens) => {
                cache.prepare_append_with_window(index_pos, seq_len, window_tokens)?
            }
            None => cache.prepare_append(index_pos, seq_len)?,
        };
        let intent = self.state_layout.ring_step_intent(
            shortconv.domain(),
            index_pos,
            batch,
            hidden,
            seq_len,
        )?;
        let output = shortconv.with_ring_depthwise_conv(&intent, |transaction| {
            let mut hidden_states = input_embeds.clone();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                diagnostics::check_layer(&hidden_states, "input", layer_idx, index_pos)?;
                let residual = hidden_states.clone();
                let hidden = layer.operator_norm.forward(&hidden_states)?;
                diagnostics::check_layer(&hidden, "operator norm", layer_idx, index_pos)?;
                let hidden = match &layer.kind {
                    LayerKind::Attention(attention) => {
                        attention.forward_physical(&hidden, index_pos, cache, &mut prepared)?
                    }
                    LayerKind::ShortConv(shortconv) => {
                        shortconv.forward_physical(&hidden, transaction)?
                    }
                };
                let hidden = layer.operator_output(hidden)?;
                diagnostics::check_layer(&hidden, "operator output", layer_idx, index_pos)?;
                hidden_states = (&hidden + &residual)?;
                diagnostics::check_layer(&hidden_states, "residual", layer_idx, index_pos)?;

                let residual = hidden_states.clone();
                let hidden = layer.ffn_norm.forward(&hidden_states)?;
                let hidden = layer.ffn_output(&hidden)?;
                diagnostics::check_layer(&hidden, "MLP output", layer_idx, index_pos)?;
                hidden_states = (&hidden + &residual)?;
                diagnostics::check_layer(&hidden_states, "residual", layer_idx, index_pos)?;
            }
            {
                let normalized = self.norm.forward(&hidden_states)?;
                diagnostics::check(&normalized, "final norm", index_pos)?;
                Ok(normalized)
            }
        })?;
        cache.commit_prepared(prepared)?;
        Ok(output)
    }

    /// Whole-sequence forward without KV state (the LFM2.5-Audio
    /// detokenizer). With a sliding window, attention runs banded in query
    /// blocks, so memory grows with the sequence, not its square: the
    /// detokenizer's 6 x frames positions would otherwise build `[16, T, T]`
    /// score planes (6.75 GiB at the default 1,024 frames).
    pub(crate) fn forward_embeds_stateless(&self, input_embeds: &Tensor) -> Result<Tensor> {
        self.forward_embeds_stateless_with(input_embeds, false)
    }

    fn forward_embeds_stateless_with(
        &self,
        input_embeds: &Tensor,
        force_dense_mask: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = input_embeds.dims3()?;
        if batch != 1 || hidden != self.cfg.embedding_length || seq_len == 0 {
            return Err(Error::InvalidInput(
                "LFM2 stateless input does not match the loaded backbone".into(),
            ));
        }
        let dense_mask;
        let mask = match self.cfg.attention_sliding_window {
            _ if seq_len <= 1 => StatelessAttentionMask::None,
            Some(window) if !force_dense_mask => {
                StatelessAttentionMask::Band(AttentionBand::causal_window(window))
            }
            _ => {
                dense_mask = self.mask(seq_len, input_embeds.device())?;
                StatelessAttentionMask::Dense(&dense_mask)
            }
        };

        let mut hidden_states = input_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = Self::stateless_layer(layer, layer_idx, &hidden_states, mask, 0)?;
        }
        {
            let normalized = self.norm.forward(&hidden_states)?;
            diagnostics::check(&normalized, "final norm", 0)?;
            Ok(normalized)
        }
    }

    /// One layer of the stateless forward over `hidden_states`, whose first
    /// row is position `index_pos`.
    fn stateless_layer(
        layer: &LayerWeights,
        layer_idx: usize,
        hidden_states: &Tensor,
        mask: StatelessAttentionMask<'_>,
        index_pos: usize,
    ) -> Result<Tensor> {
        diagnostics::check_layer(hidden_states, "input", layer_idx, index_pos)?;
        let residual = hidden_states.clone();
        let hidden = layer.operator_norm.forward(hidden_states)?;
        diagnostics::check_layer(&hidden, "operator norm", layer_idx, index_pos)?;
        let hidden = match &layer.kind {
            LayerKind::Attention(attention) => {
                attention.forward_stateless(&hidden, mask, index_pos)?
            }
            LayerKind::ShortConv(shortconv) => shortconv.forward_stateless(&hidden)?,
        };
        let hidden = layer.operator_output(hidden)?;
        diagnostics::check_layer(&hidden, "operator output", layer_idx, index_pos)?;
        let hidden_states = (&hidden + &residual)?;
        diagnostics::check_layer(&hidden_states, "residual", layer_idx, index_pos)?;

        let residual = hidden_states.clone();
        let hidden = layer.ffn_norm.forward(&hidden_states)?;
        let hidden = layer.ffn_output(&hidden)?;
        diagnostics::check_layer(&hidden, "MLP output", layer_idx, index_pos)?;
        let hidden_states = (&hidden + &residual)?;
        diagnostics::check_layer(&hidden_states, "residual", layer_idx, index_pos)?;
        Ok(hidden_states)
    }

    /// The stateless forward fed incrementally (LFT-5): `input_embeds` are
    /// the positions after those `stream` has consumed, and the result is
    /// exactly their rows of [`Self::forward_embeds_stateless`] over the
    /// whole sequence. The model is causal with bounded reach, so each layer
    /// keeps only the inputs its operator can still see: the last
    /// `sliding_window - 1` rows before an attention layer and the last
    /// `l_cache - 1` rows before a ShortConv layer, and it runs them again
    /// with the new rows at their absolute positions.
    pub(crate) fn forward_embeds_stream(
        &self,
        stream: &mut Lfm2StatelessStream,
        input_embeds: &Tensor,
    ) -> Result<Tensor> {
        let (batch, rows, hidden) = input_embeds.dims3()?;
        if batch != 1 || hidden != self.cfg.embedding_length || rows == 0 {
            return Err(Error::InvalidInput(
                "LFM2 stream input does not match the loaded backbone".into(),
            ));
        }
        let Some(window) = self.cfg.attention_sliding_window else {
            return Err(Error::InvalidInput(
                "LFM2 stateless streaming needs a sliding attention window".into(),
            ));
        };
        if stream.tails.len() != self.layers.len() {
            stream.tails = vec![None; self.layers.len()];
        }
        let start = stream.positions;
        let mut hidden_states = input_embeds.clone();
        for (layer_idx, (layer, tail)) in self.layers.iter().zip(&mut stream.tails).enumerate() {
            let keep = match &layer.kind {
                LayerKind::Attention(_) => window.saturating_sub(1),
                LayerKind::ShortConv(shortconv) => shortconv.l_cache.saturating_sub(1),
            };
            let (full, cached) = match tail.as_ref() {
                Some(tail) => (Tensor::cat(&[tail, &hidden_states], 1)?, tail.dim(1)?),
                None => (hidden_states.clone(), 0),
            };
            let full_len = full.dim(1)?;
            let kept = keep.min(full_len);
            *tail = (kept > 0)
                .then(|| full.narrow(1, full_len - kept, kept))
                .transpose()?;
            let mask = if full_len <= 1 {
                StatelessAttentionMask::None
            } else {
                StatelessAttentionMask::Band(AttentionBand::causal_window(window))
            };
            hidden_states = Self::stateless_layer(layer, layer_idx, &full, mask, start - cached)?
                .narrow(1, cached, rows)?;
        }
        stream.positions += rows;
        let normalized = self.norm.forward(&hidden_states)?;
        diagnostics::check(&normalized, "final norm", start)?;
        Ok(normalized)
    }

    fn mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let sliding_window = self.cfg.attention_sliding_window;
        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|query| {
                (0..seq_len).map(move |key| {
                    u8::from(lfm2_attention_position_masked(query, key, sliding_window))
                })
            })
            .collect();
        Tensor::from_slice(&mask, (seq_len, seq_len), device).map_err(Error::from)
    }
}

/// A causal depthwise convolution of `x` `[batch, channels, time]` with
/// `weight` `[channels, width]`: `out[t] = sum_k weight[k] * x[t + k - (width
/// - 1)]`, zero before the start, as one shifted multiply-add per tap.
fn depthwise_causal_conv(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (batch, channels, time) = x.dims3()?;
    let width = weight.dim(1)?;
    let padded = Tensor::cat(
        &[
            &Tensor::zeros((batch, channels, width - 1), x.dtype(), x.device())?,
            x,
        ],
        2,
    )?;
    let mut out = None::<Tensor>;
    for tap in 0..width {
        let term = padded
            .narrow(2, tap, time)?
            .broadcast_mul(&weight.narrow(1, tap, 1)?.unsqueeze(0)?)?;
        out = Some(match out {
            Some(sum) => (sum + term)?,
            None => term,
        });
    }
    Ok(out.expect("a filter has at least one tap").contiguous()?)
}

/// Per-layer input tails of [`QuantizedLfm2Backbone::forward_embeds_stream`]:
/// the positions consumed and, before each layer, the rows its operator can
/// still reach. Tensors are immutable, so a clone is an exact checkpoint.
#[derive(Debug, Clone, Default)]
pub(crate) struct Lfm2StatelessStream {
    positions: usize,
    tails: Vec<Option<Tensor>>,
}

impl Lfm2StatelessStream {
    pub(crate) fn positions(&self) -> usize {
        self.positions
    }
}

/// What [`QuantizedLfm2Backbone::load_inner`] reads besides the layers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BackboneParts {
    /// Token table, layers and output head.
    Full,
    /// Layers and output head; the token table is never read.
    ForEmbeds,
    /// Layers and final norm only.
    Trunk,
}

/// Attention masking for [`QuantizedLfm2Backbone::forward_embeds_stateless`].
#[derive(Clone, Copy)]
enum StatelessAttentionMask<'a> {
    /// A single position: nothing to mask.
    None,
    /// A host-built `[T, T]` boolean mask (no sliding window).
    Dense(&'a Tensor),
    /// A causal sliding window, computed banded in query blocks.
    Band(AttentionBand),
}

/// Stateless attention layers computed banded (diagnostics and tests).
static LFM2_BANDED_STATELESS_ATTENTION_CALLS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn lfm2_banded_stateless_attention_calls() -> u64 {
    LFM2_BANDED_STATELESS_ATTENTION_CALLS.load(Ordering::Relaxed)
}

fn lfm25_cuda_flash_attention_options(
    masked_prefill: bool,
    sliding_window: Option<usize>,
) -> CudaFlashAttentionOptions<'static> {
    CudaFlashAttentionOptions {
        window_size_left: if masked_prefill {
            sliding_window.map(|window| window.saturating_sub(1))
        } else {
            None
        },
        ..CudaFlashAttentionOptions::default()
    }
}

fn lfm2_attention_position_masked(query: usize, key: usize, sliding_window: Option<usize>) -> bool {
    key > query || sliding_window.is_some_and(|window| query.saturating_sub(key) >= window)
}

fn should_try_lfm25_metal_prefill_sdpa(
    query_states: &Tensor,
    seq_len: usize,
    index_pos: usize,
    sliding_window: Option<usize>,
) -> bool {
    seq_len > 1
        && index_pos == 0
        && sliding_window.is_none()
        && query_states.device().is_metal()
        && lfm25_metal_prefill_sdpa_enabled()
}

fn lfm25_metal_prefill_sdpa_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_LFM25_METAL_PREFILL_SDPA", true));
    ENABLED.get()
}

fn normalize_shortconv_weight(
    mut conv_weight: Tensor,
    l_cache: usize,
    hidden_size: usize,
) -> Result<Tensor> {
    match conv_weight.rank() {
        3 => conv_weight = conv_weight.squeeze(1)?,
        2 => {}
        rank => {
            return Err(Error::ModelLoadError(format!(
                "Unexpected LFM2 shortconv weight rank: {rank}"
            )));
        }
    }

    let (mut rows, mut cols) = conv_weight.dims2()?;
    if rows == l_cache && cols == hidden_size {
        conv_weight = conv_weight.transpose(0, 1)?;
        (rows, cols) = conv_weight.dims2()?;
    }
    if rows != hidden_size || cols != l_cache {
        return Err(Error::ModelLoadError(format!(
            "Unexpected LFM2 shortconv weight shape: expected [{hidden_size}, {l_cache}], found [{rows}, {cols}]"
        )));
    }
    conv_weight.contiguous().map_err(Error::from)
}

/// The (positions, head dim) of the RoPE cos/sin tables a backbone
/// precomputes at load: F32 `cos` and `sin` of `[positions, head_dim / 2]`
/// plus their `[positions, head_dim]` concatenation. Load admission prices
/// them (`runtime/lifecycle/asr_memory.rs`).
pub(crate) fn rope_table_geometry(cfg: &Lfm2BackboneConfig) -> Result<(usize, usize)> {
    let head_dim = cfg
        .embedding_length
        .checked_div(cfg.attention_head_count)
        .ok_or_else(|| Error::ModelLoadError("LFM2 attention head count is zero".into()))?;
    Ok((cfg.context_length, head_dim))
}

/// NEOX RoPE `cos`/`sin` tables, `[context_length, head_dim / 2]` F32.
///
/// Angles and their cosines are computed on the host in F64 and uploaded
/// once. On the device, an F32 `position × inv_freq` loses about 0.008 rad at
/// position 128K before `cos` runs, and Candle's Metal kernels evaluate
/// `cos`/`sin` with fast math, so device tables drift with position on every
/// backend.
fn precompute_freqs(
    head_dim: usize,
    freq_base: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = rope_tables_f64(head_dim, freq_base, context_length);
    let half = head_dim / 2;
    Ok((
        Tensor::from_vec(cos, (context_length, half), device)?,
        Tensor::from_vec(sin, (context_length, half), device)?,
    ))
}

/// [`precompute_freqs`] with each inverse frequency divided by its factor
/// (GGUF `rope_freqs.weight`, as llama.cpp applies it). LFM3's global layers
/// carry 1e30 on their low-frequency half, which leaves it unrotated.
fn precompute_freqs_scaled(
    head_dim: usize,
    freq_base: f64,
    factors: Option<&[f64]>,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if factors.is_some_and(|factors| factors.len() != head_dim / 2) {
        return Err(Error::ModelLoadError(format!(
            "LFM RoPE frequency factors must cover {} frequencies",
            head_dim / 2
        )));
    }
    let (cos, sin) = rope_tables_f64_scaled(head_dim, freq_base, factors, context_length);
    let half = head_dim / 2;
    Ok((
        Tensor::from_vec(cos, (context_length, half), device)?,
        Tensor::from_vec(sin, (context_length, half), device)?,
    ))
}

fn rope_tables_f64(head_dim: usize, freq_base: f64, context_length: usize) -> (Vec<f32>, Vec<f32>) {
    rope_tables_f64_scaled(head_dim, freq_base, None, context_length)
}

fn rope_tables_f64_scaled(
    head_dim: usize,
    freq_base: f64,
    factors: Option<&[f64]>,
    context_length: usize,
) -> (Vec<f32>, Vec<f32>) {
    let inv_freq: Vec<f64> = (0..head_dim)
        .step_by(2)
        .enumerate()
        .map(|(pair, index)| {
            let factor = factors.map_or(1.0, |factors| factors[pair]);
            1.0 / freq_base.powf(index as f64 / head_dim as f64) / factor
        })
        .collect();
    let mut cos = Vec::with_capacity(context_length * inv_freq.len());
    let mut sin = Vec::with_capacity(context_length * inv_freq.len());
    for position in 0..context_length {
        for frequency in &inv_freq {
            let (s, c) = (position as f64 * frequency).sin_cos();
            cos.push(c as f32);
            sin.push(s as f32);
        }
    }
    (cos, sin)
}

/// A quantized weight and, on Metal, the small-batch k-quant mat-vec over
/// the same buffer.
#[derive(Clone)]
struct Lfm2QuantizedWeight {
    tensor: Arc<candle_core::quantized::QTensor>,
    small_batch: Option<SmallBatchMatVec>,
}

/// A quantized projection. Candle runs a `[rows, 1, k]` input (a decode
/// step) as one mat-vec dispatch per row, each streaming the whole weight.
/// On Metal, 2..=8 such rows take izwi's k-quant mat-vec instead
/// (`kernels::metal_kquant_mv`): one dispatch that reads the weight once.
/// Every other input (one row, prefill spans, CPU, CUDA) takes Candle's
/// `QMatMul` unchanged.
#[derive(Debug, Clone)]
struct Lfm2Linear {
    matmul: QMatMul,
    small_batch: Option<SmallBatchMatVec>,
}

impl Lfm2Linear {
    fn new(weight: Lfm2QuantizedWeight) -> Result<Self> {
        Ok(Self {
            matmul: QMatMul::from_weights(weight.tensor)?,
            small_batch: weight.small_batch,
        })
    }

    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let per_row_mat_vec = input.rank() >= 2 && input.dim(D::Minus2)? == 1;
        #[cfg(test)]
        let per_row_mat_vec = per_row_mat_vec && !FORCE_CANDLE_MATVEC.with(std::cell::Cell::get);
        match &self.small_batch {
            Some(small_batch) if per_row_mat_vec && small_batch.accepts(input) => {
                small_batch.forward(input)
            }
            _ => self.matmul.forward(input),
        }
    }
}

#[cfg(feature = "metal")]
type SmallBatchMatVec = crate::kernels::metal_kquant_mv::MetalKQuantMatVec;

/// Without Metal there is no small-batch kernel.
#[cfg(not(feature = "metal"))]
#[derive(Debug, Clone)]
enum SmallBatchMatVec {}

#[cfg(not(feature = "metal"))]
impl SmallBatchMatVec {
    fn accepts(&self, _: &Tensor) -> bool {
        match *self {}
    }

    fn forward(&self, _: &Tensor) -> candle_core::Result<Tensor> {
        match *self {}
    }
}

fn load_linear_any(loader: &GgufLoader, device: &Device, names: &[String]) -> Result<Lfm2Linear> {
    Lfm2Linear::new(load_weight_any(loader, device, names)?)
}

/// On Metal with fused kernels on, the weight is read on the host and
/// uploaded here (as Candle's GGUF load does) so the small-batch mat-vec can
/// share its buffer; elsewhere it loads as usual.
fn load_weight_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<Lfm2QuantizedWeight> {
    #[cfg(feature = "metal")]
    if device.is_metal() && crate::kernels::use_fused_kernels_for_device(device) {
        use candle_core::quantized::{QStorage, QTensor};
        let host = load_qtensor_any(loader, &Device::Cpu, names)?;
        let dtype = host.dtype();
        let (rows, cols) = host.shape().dims2()?;
        let storage = QStorage::from_data(host.data()?, device, dtype)?;
        let small_batch = match &storage {
            QStorage::Metal(metal) => crate::kernels::metal_kquant_mv::MetalKQuantMatVec::new(
                metal.buffer().clone(),
                dtype,
                rows,
                cols,
            ),
            _ => None,
        };
        return Ok(Lfm2QuantizedWeight {
            tensor: Arc::new(QTensor::new(storage, (rows, cols))?),
            small_batch,
        });
    }
    Ok(Lfm2QuantizedWeight {
        tensor: Arc::new(load_qtensor_any(loader, device, names)?),
        small_batch: None,
    })
}

fn load_qtensor_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<candle_core::quantized::QTensor> {
    for name in names {
        if loader.has_tensor(name) {
            return loader.load_qtensor(name, device);
        }
    }
    Err(Error::ModelLoadError(format!(
        "Missing GGUF tensor; tried {}",
        names.join(" | ")
    )))
}

fn load_dense_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    dtype: Option<DType>,
) -> Result<Tensor> {
    for name in names {
        if loader.has_tensor(name) {
            let mut tensor = loader.load_qtensor(name, device)?.dequantize(device)?;
            if let Some(dtype) = dtype {
                if tensor.dtype() != dtype {
                    tensor = tensor.to_dtype(dtype)?;
                }
            }
            return Ok(tensor);
        }
    }
    Err(Error::ModelLoadError(format!(
        "Missing GGUF tensor; tried {}",
        names.join(" | ")
    )))
}

fn load_optional_bias_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<Option<Tensor>> {
    for name in names {
        if loader.has_tensor(name) {
            return load_dense_any(loader, device, std::slice::from_ref(name), Some(DType::F32))
                .map(Some);
        }
    }
    Ok(None)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{
        lfm25_cuda_flash_attention_options, lfm2_attention_position_masked, precompute_freqs,
        Lfm2ShortConvRuntimeState,
    };
    use crate::kv::v2::StateComponentId;
    use candle_core::{DType, Device, Tensor};

    /// The tables on `device` match F64 `cos`/`sin` at far positions.
    fn assert_rope_tables_exact(device: &Device) {
        const HEAD_DIM: usize = 64;
        const BASE: f64 = 1_000_000.0;
        const CONTEXT: usize = 128_000;
        let (cos, sin) = precompute_freqs(HEAD_DIM, BASE, CONTEXT, device).unwrap();
        for position in [4_095_usize, 65_535, 127_999] {
            let cos_row: Vec<f32> = cos.get(position).unwrap().to_vec1().unwrap();
            let sin_row: Vec<f32> = sin.get(position).unwrap().to_vec1().unwrap();
            for (index, (c, s)) in cos_row.iter().zip(&sin_row).enumerate() {
                let inv = 1.0 / BASE.powf((2 * index) as f64 / HEAD_DIM as f64);
                let angle = position as f64 * inv;
                let cos_err = (f64::from(*c) - angle.cos()).abs();
                let sin_err = (f64::from(*s) - angle.sin()).abs();
                // `<` is false for NaN, so a non-finite entry fails.
                assert!(
                    cos_err < 1e-6 && sin_err < 1e-6,
                    "position {position} frequency {index}: cos {cos_err} sin {sin_err}"
                );
            }
        }
    }

    #[test]
    fn rope_tables_match_f64_at_far_positions_on_cpu() {
        assert_rope_tables_exact(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn rope_tables_match_f64_at_far_positions_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "rope_tables_match_f64_at_far_positions_on_metal",
        ) else {
            return;
        };
        assert_rope_tables_exact(&device);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU; run explicitly with --ignored"]
    fn rope_tables_match_f64_at_far_positions_on_cuda() {
        assert_rope_tables_exact(&Device::new_cuda(0).expect("CUDA device required"));
    }

    #[test]
    fn embeds_only_backbone_skips_the_token_table_and_matches_the_full_load() {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
        use gguf_file::Value;

        let path = std::env::temp_dir().join(format!(
            "izwi-lfm2-embeds-only-{}.gguf",
            uuid::Uuid::new_v4()
        ));
        let metadata = [
            ("general.architecture", Value::String("lfm2".into())),
            ("lfm2.block_count", Value::U32(2)),
            ("lfm2.context_length", Value::U32(32)),
            ("lfm2.embedding_length", Value::U32(4)),
            ("lfm2.attention.head_count", Value::U32(1)),
            (
                "lfm2.attention.head_count_kv",
                Value::Array(vec![Value::U32(1), Value::U32(0)]),
            ),
            ("lfm2.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("lfm2.shortconv.l_cache", Value::U32(3)),
        ];
        let mut weights = vec![];
        let mut seed = 0u32;
        let mut add = |name: String, shape: &[usize]| {
            let elements = shape.iter().product::<usize>();
            let values = (0..elements)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
            weights.push((name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()));
        };
        add("token_embd.weight".into(), &[7, 4]);
        add("output_norm.weight".into(), &[4]);
        add("dense_2.weight".into(), &[5, 4]);
        for layer in 0..2 {
            for name in ["attn_norm", "ffn_norm"] {
                add(format!("blk.{layer}.{name}.weight"), &[4]);
            }
            for name in ["ffn_gate", "ffn_up", "ffn_down"] {
                add(format!("blk.{layer}.{name}.weight"), &[4, 4]);
            }
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            add(format!("blk.0.{name}.weight"), &[4]);
        }
        for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            add(format!("blk.0.{name}.weight"), &[4, 4]);
        }
        add("blk.1.shortconv.in_proj.weight".into(), &[12, 4]);
        add("blk.1.shortconv.out_proj.weight".into(), &[4, 4]);
        add("blk.1.shortconv.conv.weight".into(), &[4, 3]);
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(
            &mut file,
            &metadata
                .iter()
                .map(|(name, value)| (*name, value))
                .collect::<Vec<_>>(),
            &weights
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        drop(file);
        let loader = crate::models::shared::weights::gguf::GgufLoader::from_path(&path).unwrap();
        let config = super::super::config::parse_lfm2_backbone_config(&loader).unwrap();
        let full =
            super::QuantizedLfm2Backbone::load(&loader, config.clone(), &Device::Cpu).unwrap();
        let embeds_only =
            super::QuantizedLfm2Backbone::load_for_embeds(&loader, config, &Device::Cpu).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert!(full.token_embeddings.is_some());
        assert!(embeds_only.token_embeddings.is_none());
        assert_eq!(embeds_only.vocab_size(), full.vocab_size());
        let ids = Tensor::new(&[[1u32, 2]], &Device::Cpu).unwrap();
        assert!(embeds_only.embed_tokens(&ids).is_err());

        let embeds = Tensor::from_vec(
            (0..12).map(|v| (v as f32 * 0.37).sin()).collect::<Vec<_>>(),
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let project = |backbone: &super::QuantizedLfm2Backbone| {
            let hidden = backbone.forward_embeds_stateless(&embeds).unwrap();
            backbone
                .project_hidden(&hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(project(&embeds_only), project(&full));
    }

    /// A three-block LFM2 backbone (attention, ShortConv, attention) with
    /// 8-wide embeddings, GQA 2:1 and an optional sliding window, written to
    /// a temporary F32 GGUF and loaded for embeds.
    fn tiny_windowed_backbone(sliding_window: Option<u32>) -> super::QuantizedLfm2Backbone {
        tiny_windowed_backbone_with_head(sliding_window, 5)
    }

    /// The tiny three-layer backbone (attention, ShortConv, attention) with
    /// an output head of `head_outputs` rows.
    pub(crate) fn tiny_windowed_backbone_with_head(
        sliding_window: Option<u32>,
        head_outputs: usize,
    ) -> super::QuantizedLfm2Backbone {
        tiny_windowed_backbone_on(sliding_window, head_outputs, &Device::Cpu)
    }

    /// [`tiny_windowed_backbone_with_head`] loaded on `device`.
    pub(crate) fn tiny_windowed_backbone_on(
        sliding_window: Option<u32>,
        head_outputs: usize,
        device: &Device,
    ) -> super::QuantizedLfm2Backbone {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
        use gguf_file::Value;

        let path =
            std::env::temp_dir().join(format!("izwi-lfm2-windowed-{}.gguf", uuid::Uuid::new_v4()));
        let mut metadata = vec![
            ("general.architecture", Value::String("lfm2".into())),
            ("lfm2.block_count", Value::U32(3)),
            ("lfm2.context_length", Value::U32(1024)),
            ("lfm2.embedding_length", Value::U32(8)),
            ("lfm2.attention.head_count", Value::U32(2)),
            (
                "lfm2.attention.head_count_kv",
                Value::Array(vec![Value::U32(1), Value::U32(0), Value::U32(1)]),
            ),
            ("lfm2.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("lfm2.shortconv.l_cache", Value::U32(3)),
        ];
        if let Some(window) = sliding_window {
            metadata.push(("lfm2.attention.sliding_window", Value::U32(window)));
        }
        let mut weights = vec![];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut add = |name: String, shape: &[usize]| {
            let elements = shape.iter().product::<usize>();
            let values = (0..elements)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
            weights.push((name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()));
        };
        add("token_embd.weight".into(), &[7, 8]);
        add("output_norm.weight".into(), &[8]);
        add("dense_2.weight".into(), &[head_outputs, 8]);
        for layer in 0..3 {
            for name in ["attn_norm", "ffn_norm"] {
                add(format!("blk.{layer}.{name}.weight"), &[8]);
            }
            for name in ["ffn_gate", "ffn_up", "ffn_down"] {
                add(format!("blk.{layer}.{name}.weight"), &[8, 8]);
            }
        }
        for layer in [0, 2] {
            for name in ["attn_q_norm", "attn_k_norm"] {
                add(format!("blk.{layer}.{name}.weight"), &[4]);
            }
            add(format!("blk.{layer}.attn_q.weight"), &[8, 8]);
            add(format!("blk.{layer}.attn_k.weight"), &[4, 8]);
            add(format!("blk.{layer}.attn_v.weight"), &[4, 8]);
            add(format!("blk.{layer}.attn_output.weight"), &[8, 8]);
        }
        add("blk.1.shortconv.in_proj.weight".into(), &[24, 8]);
        add("blk.1.shortconv.out_proj.weight".into(), &[8, 8]);
        add("blk.1.shortconv.conv.weight".into(), &[8, 3]);
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(
            &mut file,
            &metadata
                .iter()
                .map(|(name, value)| (*name, value))
                .collect::<Vec<_>>(),
            &weights
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        drop(file);
        let loader = crate::models::shared::weights::gguf::GgufLoader::from_path(&path).unwrap();
        let config = super::super::config::parse_lfm2_backbone_config(&loader).unwrap();
        let backbone =
            super::QuantizedLfm2Backbone::load_for_embeds(&loader, config, device).unwrap();
        std::fs::remove_file(&path).unwrap();
        backbone
    }

    /// LFT-5: feeding the stateless forward in uneven steps (single rows,
    /// steps shorter and longer than the 30-row window) reproduces the
    /// whole-sequence forward row for row, keeping only `window - 1` rows
    /// before each attention layer and `l_cache - 1` before the ShortConv.
    #[test]
    fn streamed_stateless_forward_matches_the_whole_sequence() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};

        let backbone = tiny_windowed_backbone(Some(30));
        let embeds = xorshift_tensor(&[1, 300, 8], 23, 1.0, &Device::Cpu).unwrap();
        let whole = host_f32(&backbone.forward_embeds_stateless(&embeds).unwrap()).unwrap();
        let mut stream = super::Lfm2StatelessStream::default();
        let mut parts = Vec::new();
        let mut start = 0;
        for step in [6usize, 1, 12, 47, 6, 100, 128] {
            parts.push(
                backbone
                    .forward_embeds_stream(&mut stream, &embeds.narrow(1, start, step).unwrap())
                    .unwrap(),
            );
            start += step;
        }
        assert_eq!(stream.positions(), 300);
        let tails = stream
            .tails
            .iter()
            .map(|tail| tail.as_ref().unwrap().dim(1).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(tails, [29, 2, 29]);
        let streamed = host_f32(&Tensor::cat(&parts, 1).unwrap()).unwrap();
        let scale = whole.iter().fold(0f32, |m, v| m.max(v.abs()));
        let error = max_abs_diff(&streamed, &whole);
        assert!(
            error < 1e-5 * scale.max(1.0),
            "streamed vs whole max abs {error}"
        );
    }

    /// LFT-6: the detokenizer's stateless forward runs its sliding-window
    /// attention banded (no `[heads, T, T]` planes) and matches the dense
    /// masked forward across query-block boundaries.
    #[test]
    fn stateless_sliding_window_attention_is_banded_and_matches_dense_mask() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};

        let backbone = tiny_windowed_backbone(Some(30));
        let embeds = xorshift_tensor(&[1, 300, 8], 11, 1.0, &Device::Cpu).unwrap();
        let before = super::lfm2_banded_stateless_attention_calls();
        let banded = host_f32(&backbone.forward_embeds_stateless(&embeds).unwrap()).unwrap();
        // Two attention layers each took the banded path (other tests may
        // run concurrently, so only a lower bound is exact).
        assert!(super::lfm2_banded_stateless_attention_calls() - before >= 2);
        let dense = host_f32(
            &backbone
                .forward_embeds_stateless_with(&embeds, true)
                .unwrap(),
        )
        .unwrap();
        let scale = dense.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.1, "backbone output is degenerate");
        let error = max_abs_diff(&banded, &dense);
        assert!(
            error < 1e-5 * scale.max(1.0),
            "banded vs dense max abs {error}"
        );

        // Without a window the dense causal mask stays in use and agrees too.
        let causal = tiny_windowed_backbone(None);
        let short = embeds.narrow(1, 0, 40).unwrap();
        assert_eq!(
            host_f32(&causal.forward_embeds_stateless(&short).unwrap()).unwrap(),
            host_f32(&causal.forward_embeds_stateless_with(&short, true).unwrap()).unwrap()
        );
    }

    /// LFT-5 on the published detokenizer backbone (dequantized, as above):
    /// 600 positions streamed in codec-frame steps (6 positions per frame)
    /// match the whole-sequence forward.
    #[test]
    #[ignore = "needs the local LFM2.5-Audio tokenizer GGUF"]
    fn published_detokenizer_stream_matches_whole_sequence() {
        use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_tensor};

        let path = {
            let _env = crate::env_test_lock().lock().unwrap();
            std::env::var_os("IZWI_TEST_LFM25_DETOKENIZER_GGUF")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        std::path::PathBuf::from(home).join(
                            "Library/Application Support/izwi/models/LFM2.5-Audio-1.5B-GGUF/\
                             tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf",
                        )
                    })
                })
                .unwrap()
        };
        let loader = crate::models::shared::weights::gguf::GgufLoader::from_path(&path).unwrap();
        let config = super::super::config::parse_lfm2_backbone_config(&loader).unwrap();
        let width = config.embedding_length;
        let backbone = {
            let _env = crate::env_test_lock().lock().unwrap();
            let previous = std::env::var_os("CANDLE_DEQUANTIZE_ALL");
            std::env::set_var("CANDLE_DEQUANTIZE_ALL", "1");
            let backbone =
                super::QuantizedLfm2Backbone::load_for_embeds(&loader, config, &Device::Cpu);
            match previous {
                Some(value) => std::env::set_var("CANDLE_DEQUANTIZE_ALL", value),
                None => std::env::remove_var("CANDLE_DEQUANTIZE_ALL"),
            }
            backbone.unwrap()
        };
        let embeds = xorshift_tensor(&[1, 600, width], 17, 1.0, &Device::Cpu).unwrap();
        let whole = host_f32(&backbone.forward_embeds_stateless(&embeds).unwrap()).unwrap();
        let mut stream = super::Lfm2StatelessStream::default();
        let mut parts = Vec::new();
        let mut start = 0;
        for frames in [1usize, 4, 2, 8, 16, 32, 37] {
            let rows = frames * 6;
            parts.push(
                backbone
                    .forward_embeds_stream(&mut stream, &embeds.narrow(1, start, rows).unwrap())
                    .unwrap(),
            );
            start += rows;
        }
        assert_eq!(start, 600);
        let streamed = host_f32(&Tensor::cat(&parts, 1).unwrap()).unwrap();
        let rel = relative_l2(&streamed, &whole);
        eprintln!("published detokenizer stream vs whole: relative L2 {rel:e}");
        assert!(rel < 1e-5, "relative L2 {rel}");
    }

    /// The published LFM2.5-Audio detokenizer (window 30, 16 heads) gives the
    /// same banded and dense output on 600 positions (100 codec frames). The
    /// weights are dequantized at load: with Q4_0 matmuls the Q8 activation
    /// quantization turns last-ulp differences between the two op shapes
    /// into ~6e-3 relative noise, which would hide a real band error. Set
    /// IZWI_TEST_LFM25_DETOKENIZER_GGUF or keep the default izwi model path.
    #[test]
    #[ignore = "needs the local LFM2.5-Audio tokenizer GGUF"]
    fn published_detokenizer_banded_attention_matches_dense_mask() {
        use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_tensor};

        let path = {
            let _env = crate::env_test_lock().lock().unwrap();
            std::env::var_os("IZWI_TEST_LFM25_DETOKENIZER_GGUF")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        std::path::PathBuf::from(home).join(
                            "Library/Application Support/izwi/models/LFM2.5-Audio-1.5B-GGUF/\
                             tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf",
                        )
                    })
                })
                .unwrap()
        };
        let loader = crate::models::shared::weights::gguf::GgufLoader::from_path(&path).unwrap();
        let config = super::super::config::parse_lfm2_backbone_config(&loader).unwrap();
        assert_eq!(config.attention_sliding_window, Some(30));
        let width = config.embedding_length;
        let backbone = {
            let _env = crate::env_test_lock().lock().unwrap();
            let previous = std::env::var_os("CANDLE_DEQUANTIZE_ALL");
            std::env::set_var("CANDLE_DEQUANTIZE_ALL", "1");
            let backbone =
                super::QuantizedLfm2Backbone::load_for_embeds(&loader, config, &Device::Cpu);
            match previous {
                Some(value) => std::env::set_var("CANDLE_DEQUANTIZE_ALL", value),
                None => std::env::remove_var("CANDLE_DEQUANTIZE_ALL"),
            }
            backbone.unwrap()
        };
        let embeds = xorshift_tensor(&[1, 600, width], 13, 1.0, &Device::Cpu).unwrap();
        let banded = host_f32(&backbone.forward_embeds_stateless(&embeds).unwrap()).unwrap();
        let dense = host_f32(
            &backbone
                .forward_embeds_stateless_with(&embeds, true)
                .unwrap(),
        )
        .unwrap();
        let rel = relative_l2(&banded, &dense);
        eprintln!("banded vs dense relative L2 {rel:e}");
        assert!(rel < 1e-5, "relative L2 {rel}");
    }

    #[test]
    fn lfm2_batched_rotary_matches_scalar_at_ragged_positions() {
        assert_batched_rotary_matches_scalar(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA CUDA device"]
    fn cuda_lfm2_batched_rotary_matches_scalar_at_ragged_positions() {
        assert_batched_rotary_matches_scalar(&Device::new_cuda(0).unwrap());
    }

    fn assert_batched_rotary_matches_scalar(device: &Device) {
        let cos = Tensor::from_vec(
            (0..64).map(|v| (v as f32 * 0.13).cos()).collect(),
            (16, 4),
            device,
        )
        .unwrap();
        let sin = Tensor::from_vec(
            (0..64).map(|v| (v as f32 * 0.13).sin()).collect(),
            (16, 4),
            device,
        )
        .unwrap();
        let query = Tensor::from_vec(
            (0..72).map(|v| v as f32 * 0.1 - 2.).collect(),
            (3, 1, 3, 8),
            device,
        )
        .unwrap();
        let key = query.narrow(2, 0, 1).unwrap().contiguous().unwrap();
        let positions = [0, 9, 15];
        assert!(super::rotary_decode_batch(&query, &key, &cos, &sin, &[0, 9, 16]).is_err());
        assert!(super::rotary_decode_batch(&query, &key, &cos, &sin, &[0, 9]).is_err());
        let (q, k) = super::rotary_decode_batch(&query, &key, &cos, &sin, &positions).unwrap();
        for (input, actual) in [(&query, &q), (&key, &k)] {
            for (row, &position) in positions.iter().enumerate() {
                let scalar = candle_nn::rotary_emb::rope(
                    &input
                        .narrow(0, row, 1)
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .contiguous()
                        .unwrap(),
                    &cos.narrow(0, position, 1).unwrap(),
                    &sin.narrow(0, position, 1).unwrap(),
                )
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap();
                let expected = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let actual = actual
                    .narrow(0, row, 1)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() <= 1e-6);
                }
            }
        }

        // One gather serves every layer of a step; another table gathers once
        // more, and the first stays cached for the layers that alternate.
        let mut rows = Vec::new();
        let first = super::RotaryDecodeRows::for_step(&mut rows, &cos, &sin, &positions, 3)
            .unwrap()
            .cos
            .id();
        let reused = super::RotaryDecodeRows::for_step(&mut rows, &cos, &sin, &positions, 3)
            .unwrap()
            .cos
            .id();
        assert_eq!(first, reused);
        let (cos2, sin2) = (cos.copy().unwrap(), sin.copy().unwrap());
        let other =
            super::RotaryDecodeRows::for_step(&mut rows, &cos2, &sin2, &positions, 3).unwrap();
        assert_ne!(other.cos.id(), first);
        let other_id = other.cos.id();
        let again = super::RotaryDecodeRows::for_step(&mut rows, &cos, &sin, &positions, 3)
            .unwrap()
            .cos
            .id();
        assert_eq!(again, first);
        assert_eq!(rows.len(), 2);
        let other = super::RotaryDecodeRows::for_step(&mut rows, &cos2, &sin2, &positions, 3)
            .unwrap();
        assert_eq!(other.cos.id(), other_id);
        let (q2, k2) = other.rotate(&query, &key).unwrap();
        for (left, right) in [(&q, &q2), (&k, &k2)] {
            assert_eq!(
                left.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                right.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
        assert!(other.rotate(&query.narrow(0, 0, 2).unwrap(), &key).is_err());
    }

    #[test]
    fn lfm25_cuda_flash_options_use_window_only_for_masked_prefill() {
        let options = lfm25_cuda_flash_attention_options(true, Some(512));
        assert_eq!(options.window_size_left, Some(511));
        assert_eq!(options.window_size_right, None);
        assert!(options.alibi_slopes.is_none());
        assert!(options.softcap.is_none());

        let decode_options = lfm25_cuda_flash_attention_options(false, Some(512));
        assert_eq!(decode_options.window_size_left, None);

        let full_causal_options = lfm25_cuda_flash_attention_options(true, None);
        assert_eq!(full_causal_options.window_size_left, None);
    }

    #[test]
    fn lfm2_sliding_window_counts_the_current_token() {
        let window = Some(3);
        assert!(!lfm2_attention_position_masked(4, 4, window));
        assert!(!lfm2_attention_position_masked(4, 2, window));
        assert!(lfm2_attention_position_masked(4, 1, window));
        assert!(lfm2_attention_position_masked(4, 5, window));
        assert!(!lfm2_attention_position_masked(4, 0, None));
    }

    #[test]
    fn retained_shortconv_ring_is_chunk_partition_invariant_across_wrap() {
        let device = &Device::Cpu;
        let component = StateComponentId::new(1);
        let initial = || Lfm2ShortConvRuntimeState {
            cursor: 0,
            capacity: 3,
            hidden: 2,
            components: vec![(component, None)],
        };
        let input = Tensor::from_vec(
            vec![
                0.1f32, 0.2, 0.3, 0.4, 0.5, // hidden row 0
                -0.2, 0.1, 0.6, -0.1, 0.2, // hidden row 1
            ],
            (1, 2, 5),
            device,
        )
        .unwrap();
        let weight =
            Tensor::from_vec(vec![0.2f32, 0.3, 0.5, -0.1, 0.4, 0.7], (2, 3), device).unwrap();
        let mut full = initial();
        let full_output = full.apply(component, &input, &weight).unwrap();
        full.advance(5).unwrap();
        let mut chunked = initial();
        let first = chunked
            .apply(component, &input.narrow(2, 0, 2).unwrap(), &weight)
            .unwrap();
        chunked.advance(2).unwrap();
        let second = chunked
            .apply(component, &input.narrow(2, 2, 3).unwrap(), &weight)
            .unwrap();
        chunked.advance(3).unwrap();
        let chunked_output = Tensor::cat(&[&first, &second], 2).unwrap();
        let full_values = full_output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let chunked_values = chunked_output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (full, chunked) in full_values.iter().zip(&chunked_values) {
            assert!((full - chunked).abs() < 1e-6, "{full} != {chunked}");
        }
        assert_eq!(full.cursor, chunked.cursor);
        let full_ring = full.components[0].1.as_ref().unwrap();
        let chunked_ring = chunked.components[0].1.as_ref().unwrap();
        assert_eq!(
            full_ring.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            chunked_ring
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert_eq!(full_ring.dtype(), DType::F32);
    }

    /// `IZWI_FUSED_KERNELS=0` falls back to the op path on an accelerator
    /// and matches the CPU ring across a chunk boundary.
    fn assert_unfused_ring_matches_cpu(device: &Device) {
        let _env = crate::env_test_lock().lock().unwrap();
        let previous = std::env::var_os("IZWI_FUSED_KERNELS");
        std::env::set_var("IZWI_FUSED_KERNELS", "0");
        let run = |device: &Device| {
            let component = StateComponentId::new(1);
            let mut state = Lfm2ShortConvRuntimeState {
                cursor: 0,
                capacity: 3,
                hidden: 2,
                components: vec![(component, None)],
            };
            let input = Tensor::from_vec(
                vec![0.1f32, 0.2, 0.3, 0.4, 0.5, -0.2, 0.1, 0.6, -0.1, 0.2],
                (1, 2, 5),
                device,
            )
            .unwrap();
            let weight =
                Tensor::from_vec(vec![0.2f32, 0.3, 0.5, -0.1, 0.4, 0.7], (2, 3), device).unwrap();
            let first = state
                .apply(component, &input.narrow(2, 0, 2).unwrap(), &weight)
                .unwrap();
            state.advance(2).unwrap();
            let second = state
                .apply(component, &input.narrow(2, 2, 3).unwrap(), &weight)
                .unwrap();
            Tensor::cat(&[&first, &second], 2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let result = (run(device), run(&Device::Cpu));
        match previous {
            Some(value) => std::env::set_var("IZWI_FUSED_KERNELS", value),
            None => std::env::remove_var("IZWI_FUSED_KERNELS"),
        }
        let (accelerated, cpu) = result;
        for (lhs, rhs) in accelerated.iter().zip(&cpu) {
            assert!((lhs - rhs).abs() < 1e-6, "{lhs} != {rhs}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn unfused_shortconv_ring_falls_back_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "unfused_shortconv_ring_falls_back_on_metal",
        ) else {
            return;
        };
        assert_unfused_ring_matches_cpu(&device);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU; run explicitly with --ignored"]
    fn unfused_shortconv_ring_falls_back_on_cuda() {
        assert_unfused_ring_matches_cpu(&Device::new_cuda(0).expect("CUDA device required"));
    }

    #[test]
    fn depthwise_causal_conv_matches_a_grouped_conv1d() {
        let device = Device::Cpu;
        let (channels, width, time) = (5, 7, 11);
        let x = Tensor::randn(0f32, 1.0, (2, channels, time), &device).unwrap();
        let weight = Tensor::randn(0f32, 1.0, (channels, width), &device).unwrap();
        let conv = candle_nn::Conv1d::new(
            weight.reshape((channels, 1, width)).unwrap(),
            None,
            candle_nn::Conv1dConfig {
                padding: width - 1,
                groups: channels,
                ..Default::default()
            },
        );
        let expected = candle_nn::Module::forward(&conv, &x)
            .unwrap()
            .narrow(2, 0, time)
            .unwrap();
        let actual = super::depthwise_causal_conv(&x, &weight).unwrap();
        let worst = (expected - actual)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(worst < 1e-5, "worst difference {worst}");
    }

    /// The host ring loop sums the same taps in the same order as the op
    /// path: bit-identical, across a wrapped ring with a partial history.
    #[test]
    fn host_shortconv_ring_is_bit_identical_to_the_op_path() {
        use crate::kernels::numeric_check::xorshift_tensor;
        let device = &Device::Cpu;
        for (capacity, cursor, valid, steps) in [(7, 0, 0, 12), (7, 23, 7, 5), (7, 9, 4, 1), (3, 40, 3, 9)] {
            let hidden = 6;
            let ring = xorshift_tensor(&[capacity, 1, hidden], 11, 1.0, device).unwrap();
            let input = xorshift_tensor(&[1, hidden, steps], 12, 1.0, device).unwrap();
            let weight = xorshift_tensor(&[hidden, capacity], 13, 1.0, device).unwrap();
            let host = super::host_shortconv_ring(&ring, &input, &weight, cursor, valid).unwrap();
            let ops = super::op_shortconv_ring(&ring, &input, &weight, cursor, valid).unwrap();
            assert_eq!(
                host.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                ops.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "capacity {capacity}, cursor {cursor}, valid {valid}, steps {steps}"
            );
        }
    }

    #[test]
    fn retained_shortconv_decode_rows_match_scalar_oracles_and_rollback() {
        let device = &Device::Cpu;
        let component = StateComponentId::new(2);
        let initial = || Lfm2ShortConvRuntimeState {
            cursor: 0,
            capacity: 3,
            hidden: 2,
            components: vec![(component, None)],
        };
        let weight =
            Tensor::from_vec(vec![0.2f32, 0.3, 0.5, -0.1, 0.4, 0.7], (2, 3), device).unwrap();
        let history =
            Tensor::from_vec(vec![0.1f32, 0.2, 0.3, -0.2, 0.1, 0.6], (1, 2, 3), device).unwrap();
        let mut rows = [initial(), initial()];
        rows[0].apply(component, &history, &weight).unwrap();
        rows[0].advance(3).unwrap();
        rows[1]
            .apply(component, &history.narrow(2, 1, 2).unwrap(), &weight)
            .unwrap();
        rows[1].advance(2).unwrap();
        let checkpoints = rows.clone();
        let inputs = Tensor::from_vec(vec![0.9f32, -0.4, -0.7, 0.8], (2, 2, 1), device).unwrap();

        let batch_outputs = (0..2)
            .map(|row| {
                rows[row]
                    .apply(component, &inputs.narrow(0, row, 1).unwrap(), &weight)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for row in &mut rows {
            row.advance(1).unwrap();
        }

        for row in 0..2 {
            let mut oracle = checkpoints[row].clone();
            let expected = oracle
                .apply(component, &inputs.narrow(0, row, 1).unwrap(), &weight)
                .unwrap();
            oracle.advance(1).unwrap();
            assert_eq!(
                batch_outputs[row]
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
            assert_eq!(rows[row].cursor, oracle.cursor);
            assert_eq!(
                rows[row].components[0]
                    .1
                    .as_ref()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                oracle.components[0]
                    .1
                    .as_ref()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            );
        }

        rows = checkpoints.clone();
        assert_eq!(rows[0].cursor, 3);
        assert_eq!(rows[1].cursor, 2);
    }
}
#[cfg(test)]
#[path = "backbone_mixed_tests.rs"]
mod mixed_tests;

#[test]
fn retained_ragged_launches_drop_shorter_rows_without_padding() {
    assert_eq!(
        retained_ragged_launch_rows(&[3, 1, 2]).unwrap(),
        vec![vec![0, 1, 2], vec![0, 2], vec![0]]
    );
    assert!(retained_ragged_launch_rows(&[]).is_err());
    assert!(retained_ragged_launch_rows(&[1, 0]).is_err());
}
