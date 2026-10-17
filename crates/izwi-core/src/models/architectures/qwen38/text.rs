use crate::kernels::cuda::graphs::{IslandOutput, TensorIsland};
use crate::performance::CudaPerformanceConfig;
const GRAPH_CACHE_BYTES: usize = 8 * 1024 * 1024;
// Keep an admitted region working set reusable across complete 64-layer passes.
const GRAPH_LAYER_LIMIT: usize = 8;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{ops, rotary_emb, Embedding, Linear};
use std::sync::Arc;

use super::cache::{CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN};
use super::chat::Qwen38TextConfig;
use super::fast_path::{
    compare_values, decode_gate, summarize, Qwen38CudaSwitches, Qwen38FastPath, FUSED_AB_GEMV_ENV,
    FUSED_ATTN_GATE_ENV, FUSED_GDN_ENV, FUSED_KV_WRITE_ENV, FUSED_NORM_ENV, FUSED_QK_ROPE_ENV,
    GDN_PREFILL_ENV,
};
use super::native::{
    IndexedSafetensors, ProjectionMaterialization, Qwen38LayerType, Qwen38NativeConfig,
};
use super::residency::{fp8_row_dot, probe_activations, Qwen38ResidencyPolicy, Qwen38WeightForm};
use super::telemetry::{
    record_cpu_q8_projection, record_cuda_attention_dtype_casts, record_cuda_attn_gate_composition,
    record_cuda_fp8_gemv, record_cuda_fp8_packed_projection, record_cuda_fp8_prefill_gemm,
    record_cuda_head_expansion_materialization, record_cuda_kernel, record_cuda_lm_head,
    record_cuda_projection, record_cuda_qk_rope_chain, record_cuda_rms_norm_composition,
    record_cuda_rope, record_cuda_state_initial_allocation, record_fast_path_self_check_failure,
    record_fused_attn_gate, record_fused_qk_norm_rope, record_fused_rms_norm,
    record_metal_fp8_projection, CudaKernelPath, CudaProjectionPath,
};
use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::kernels::cuda::gate;
use crate::kernels::cuda::gdn::{self, GdnDecodeSpec};
use crate::kernels::cuda::norm;
use crate::kernels::cuda::rope::{self, MropePosition, QkNormRopeSpec};
use crate::kernels::cuda::{qwen38_deltanet_sequence, qwen38_deltanet_sequence_supported};
use crate::kernels::{
    try_fused_gated_delta_recurrent, try_fused_gated_rms_norm, try_fused_l2_norm,
    try_fused_silu_mul, try_qwen38_causal_conv_decode, try_qwen38_causal_conv_sequence,
    try_qwen38_deltanet_decode, try_qwen38_gated_rms_norm_decode, try_qwen38_l2_norm_decode,
    try_qwen38_silu_mul_decode, try_tiled_deltanet_recurrence,
};
use crate::kv::v2::{StateComponentId, StateDomainId};
use crate::kv::{KvDecodeBatchMetadata, KvSlotRef};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::memory::accounting::{
    deep_copy_tensor_storage, TensorStorageAccounting,
};
use crate::models::shared::telemetry::{
    record_prefill_sequence_span, record_rope_kernel, record_rope_manual,
};

#[path = "decode_graph.rs"]
mod decode_graph;
mod kernel_timings;

pub struct Qwen38TextModel {
    device: Device,
    projection_representation: Qwen38ProjectionRepresentation,
    /// The residency policy the trunk was loaded under.
    residency: Qwen38ResidencyPolicy,
    /// One entry per block-FP8 projection group: packed, or why not.
    fp8_packing: Vec<Qwen38FastPath>,
    /// Metal: one entry per checked FP8 projection kind (GEMV or rerouted).
    fp8_gemv_checks: Vec<Qwen38FastPath>,
    /// CUDA: one entry per checked FP8 projection kind (prefill GEMM or why
    /// the kind keeps the compact kernel).
    fp8_prefill_gemm_checks: Vec<Qwen38FastPath>,
    /// The LM head as block FP8 (or why not), with the host gate's
    /// relative L2 error against the BF16 head.
    fp8_lm_head: (Qwen38FastPath, Option<f64>),
    token_embeddings: Embedding,
    layers: Vec<Qwen38Layer>,
    output_norm: Qwen38RmsNorm,
    output: Qwen38Projection,
    finite_diagnostics_enabled: bool,
    /// Fused residual add + RMSNorm over every trunk norm.
    fused_norm_path: Qwen38FastPath,
    /// Piecewise CUDA graph decode of single rows (`decode_graph.rs`), when
    /// it resolved at load, and why not otherwise.
    decode_graphs: Option<decode_graph::Qwen38DecodeGraphs>,
    decode_graphs_path: Qwen38FastPath,
    /// The checkpoint's context length: the longest block table the
    /// whole-step graph's step buffer must hold.
    max_context_tokens: usize,
}

/// One target-model prefill pass retained for MTP prompt alignment.
///
/// `hidden_states` is the complete decoder residual before the target output
/// norm. `logits`, when requested, is sampled from the final row of that same
/// pass; no target computation is repeated.
pub(super) struct Qwen38TextPrefillOutput {
    pub hidden_states: Tensor,
    pub logits: Option<Tensor>,
}

/// One decode step queued through the replaying whole-step graph: its logits
/// are on the device and its KV write is not committed yet
/// ([`Qwen38TextModel::commit_enqueued_step`]).
pub(super) struct EnqueuedGraphStep {
    /// `[1, 1, vocab]` on the device (the step's own copy).
    pub(super) logits: Tensor,
    position: usize,
    /// Taken by the commit. A step dropped uncommitted (a discarded
    /// pipelined step, an error) waits for its KV writes, so their pages
    /// cannot be reused under them.
    completions: Option<KvWriteCompletionCollector>,
}

impl Drop for EnqueuedGraphStep {
    fn drop(&mut self) {
        if let Some(completions) = self.completions.take() {
            if let Err(error) = completions.drain() {
                tracing::warn!(%error, "draining an uncommitted Qwen3.8 graph step failed");
            }
        }
    }
}

pub(super) struct Qwen38TextDecodeBatchOutput {
    /// The pre-norm hidden rows, or `None` when the whole-step CUDA graph
    /// ran the LM head inside the graph (only single rows without an MTP
    /// head take that path).
    pub hidden_states: Option<Tensor>,
    pub logits: Tensor,
}

/// Bounded verification intermediates. Q/K/V and discretization rows scale
/// with at most four positions; the initial state is an immutable handle to
/// the transaction checkpoint, never one full recurrent matrix per position.
struct Qwen38LinearVerification {
    initial: Qwen38LayerRuntimeState,
    convolution_input: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
}

impl Qwen38LinearVerification {
    fn recover(&self, prefix: usize) -> Result<Qwen38LayerRuntimeState> {
        let length = self.value.dim(1)?;
        if prefix > length {
            return Err(Error::InvalidInput(
                "Qwen3.8 verification prefix exceeds span".into(),
            ));
        }
        let mut restored = self.initial.clone();
        if prefix == 0 {
            return Ok(restored);
        }
        let Qwen38LayerRuntimeState::Linear {
            conv_state,
            recurrent_state,
        } = &mut restored
        else {
            return Err(Error::InferenceError(
                "verification requires linear state".into(),
            ));
        };
        let mut query = self.query.narrow(1, 0, prefix)?;
        let mut key = self.key.narrow(1, 0, prefix)?;
        let value = self.value.narrow(1, 0, prefix)?;
        let repeats = value.dim(2)? / query.dim(2)?;
        if repeats > 1 {
            query = repeat_interleave_head_states_seq(&query, repeats)?;
            key = repeat_interleave_head_states_seq(&key, repeats)?;
        }
        let base = recurrent_state.as_ref().ok_or_else(|| {
            Error::InferenceError("verification recurrent checkpoint missing".into())
        })?;
        let (_, next) = recurrent_gated_delta_sequence(
            &query,
            &key,
            &value,
            &self.g.narrow(1, 0, prefix)?,
            &self.beta.narrow(1, 0, prefix)?,
            base.clone(),
        )?;
        *recurrent_state = Some(deep_copy_tensor_storage(&next)?);
        if let Some(conv) = conv_state.as_mut() {
            for position in 0..prefix {
                conv.push_decode(&self.convolution_input.i((0, position))?.unsqueeze(1)?)?;
            }
            conv.compact_owned()?;
        }
        Ok(restored)
    }
}

pub(super) struct Qwen38TextVerification {
    pub hidden_states: Tensor,
    initial_state: Qwen38TextRuntimeState,
    final_state: Qwen38TextRuntimeState,
    layers: Vec<Option<Qwen38LinearVerification>>,
    start_position: usize,
    view_id: u64,
    generation: u64,
    token_count: usize,
}

impl Qwen38TextVerification {
    /// Install only the verified prefix. Full attention keeps already-written
    /// KV rows and their completion fences; hybrid state is reconstructed from
    /// compact intermediates without running target weights again.
    pub(super) fn commit_prefix(
        self,
        prefix: usize,
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        if cache.view_id() != self.view_id
            || cache.logical_generation() != self.generation
            || prefix > self.token_count
            || cache.context_len() != self.start_position + self.token_count
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 verification commit is stale or out of bounds".into(),
            ));
        }
        if prefix != self.token_count {
            let mut recovered = self.initial_state;
            for (layer, retained) in recovered.layers.iter_mut().zip(self.layers.iter()) {
                if let Some(retained) = retained {
                    *layer = retained.recover(prefix)?;
                }
            }
            cache.truncate_verified_prefix(self.start_position + prefix)?;
            *state = recovered;
        } else {
            *state = self.final_state;
        }
        // Detach the escaping canonical hidden rows from rejected positions.
        deep_copy_tensor_storage(&self.hidden_states.narrow(1, 0, prefix)?).map_err(Error::from)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen38ProjectionRepresentation {
    ExpandedF32,
    ExpandedF16,
    ExpandedBf16,
    PackedQ8WithDenseF16,
    PackedQ8WithDenseBf16,
    NativeFp8WithQ8FallbackF16,
    NativeFp8WithQ8FallbackBf16,
    NativeFp8WithDenseF16,
    PackedQ8WithDenseF32,
}

impl Qwen38ProjectionRepresentation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeFp8WithQ8FallbackF16 => "native_block_fp8_with_q8_fallback_dense_f16",
            Self::NativeFp8WithQ8FallbackBf16 => "native_block_fp8_with_q8_fallback_dense_bf16",
            Self::NativeFp8WithDenseF16 => "native_block_fp8_with_dense_f16",
            Self::PackedQ8WithDenseF32 => "q8_0_projections_with_dense_f32",
            Self::ExpandedF32 => "expanded_f32",
            Self::ExpandedF16 => "expanded_f16",
            Self::ExpandedBf16 => "expanded_bf16",
            Self::PackedQ8WithDenseF16 => "q8_0_requantized_projections_with_dense_f16",
            Self::PackedQ8WithDenseBf16 => "q8_0_requantized_projections_with_dense_bf16",
        }
    }

    pub const fn compute_dtype(self) -> &'static str {
        match self {
            Self::ExpandedF32 | Self::PackedQ8WithDenseF32 => "f32",
            Self::ExpandedF16
            | Self::PackedQ8WithDenseF16
            | Self::NativeFp8WithQ8FallbackF16
            | Self::NativeFp8WithDenseF16 => "f16",
            Self::ExpandedBf16
            | Self::PackedQ8WithDenseBf16
            | Self::NativeFp8WithQ8FallbackBf16 => "bf16",
        }
    }
}

#[derive(Clone)]
pub struct Qwen38TextRuntimeState {
    layers: Vec<Qwen38LayerRuntimeState>,
}

impl Qwen38TextRuntimeState {
    /// Backing allocations retained by the per-request text runtime state.
    ///
    /// This intentionally excludes model-global caches (notably full-attention
    /// RoPE windows), so callers requiring a complete scheduler claim must keep
    /// Qwen3.8 fail-closed until those caches are independently bounded.
    pub fn allocated_session_bytes(&self) -> Option<u64> {
        let mut accounting = TensorStorageAccounting::default();
        self.account_storage(&mut accounting)?;
        Some(accounting.bytes())
    }

    pub(crate) fn account_storage(&self, accounting: &mut TensorStorageAccounting) -> Option<()> {
        for layer in &self.layers {
            match layer {
                Qwen38LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    if let Some(conv_state) = conv_state {
                        conv_state.account_storage(accounting)?;
                    }
                    if let Some(recurrent_state) = recurrent_state {
                        accounting.add_tensor(recurrent_state)?;
                    }
                }
                Qwen38LayerRuntimeState::Full => {}
            }
        }
        Some(())
    }

    /// Whether every DeltaNet layer holds its recurrent state. A fresh state
    /// holds none until its first prefill zero-initializes it; a span that
    /// resumes above token zero must find the state its earlier spans (or an
    /// attached snapshot) left.
    pub(crate) fn linear_state_initialized(&self) -> bool {
        self.layers.iter().all(|layer| match layer {
            Qwen38LayerRuntimeState::Linear {
                recurrent_state, ..
            } => recurrent_state.is_some(),
            Qwen38LayerRuntimeState::Full => true,
        })
    }

    /// Host copy of the hybrid state in the order the tensor domains commit
    /// it: every DeltaNet layer's recurrent state, then every layer's
    /// convolution history.
    #[cfg(test)]
    pub(crate) fn linear_state_values(&self) -> Result<Vec<f32>> {
        let mut recurrent = Vec::new();
        let mut convolution = Vec::new();
        for layer in &self.layers {
            if let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            {
                let state = recurrent_state.as_ref().ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 recurrent state was not initialized".into())
                })?;
                let ring = conv_state.as_ref().ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 convolution state was not initialized".into())
                })?;
                recurrent.extend(host_values(state)?);
                convolution.extend(host_values(&ring.staged_tensor()?)?);
            }
        }
        recurrent.extend(convolution);
        Ok(recurrent)
    }

    pub(crate) fn restore_tensor_domains(
        &mut self,
        arena: &TensorStateArena,
        sequence: PhysicalStateSequenceId,
    ) -> Result<()> {
        let recurrent = arena.read(sequence, recurrent_domain_v2())?;
        let convolution = arena.read(sequence, convolution_domain_v2())?;
        if recurrent.is_none() && convolution.is_none() {
            return Ok(());
        }
        let recurrent = recurrent.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 recurrent state is missing its convolution peer".into())
        })?;
        let convolution = convolution.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 convolution state is missing its recurrent peer".into())
        })?;
        let mut recurrent_components = recurrent.components.iter();
        let mut convolution_components = convolution.components.iter();
        for layer in &mut self.layers {
            let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent = recurrent_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 recurrent component coverage is incomplete".into())
            })?;
            let convolution = convolution_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 convolution component coverage is incomplete".into())
            })?;
            let recurrent_tensor = recurrent.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 recurrent component is absent".into())
            })?;
            let convolution_tensor = convolution.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 convolution component is absent".into())
            })?;
            *recurrent_state = Some(recurrent_tensor.clone());
            *conv_state = Some(ConvRingState::from_staged(convolution_tensor.clone())?);
        }
        if recurrent_components.next().is_some() || convolution_components.next().is_some() {
            return Err(Error::InferenceError(
                "Qwen3.8 tensor state has components for unknown layers".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn stage_tensor_domains(
        &mut self,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        let recurrent_cursor = arena
            .read_transaction_base(transaction, recurrent_domain_v2())?
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        let convolution_cursor = arena
            .read_transaction_base(transaction, convolution_domain_v2())?
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        let mut recurrent = Vec::new();
        let mut convolution = Vec::new();
        for layer in &self.layers {
            let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent_tensor = recurrent_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 recurrent state was not initialized".into())
            })?;
            let ring = conv_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 convolution state was not initialized".into())
            })?;
            let ring_tensor = ring.staged_tensor()?;
            let component = u32::try_from(recurrent.len() + 1)
                .map_err(|_| Error::InvalidInput("Qwen3.8 state component overflow".into()))?;
            recurrent.push(StateComponentValue {
                component: StateComponentId::new(component),
                tensor: Some(recurrent_tensor.clone()),
            });
            convolution.push(StateComponentValue {
                component: StateComponentId::new(component),
                tensor: Some(ring_tensor),
            });
        }
        arena.stage_replace(
            transaction,
            recurrent_domain_v2(),
            recurrent_cursor,
            target_cursor,
            recurrent,
        )?;
        arena.stage_replace(
            transaction,
            convolution_domain_v2(),
            convolution_cursor,
            target_cursor,
            convolution,
        )?;
        // The arena now owns the only retained handles. Keep the decode state
        // as control metadata between quanta so engine abort cannot expose a
        // partially drained model state.
        for layer in &mut self.layers {
            if let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            {
                *conv_state = None;
                *recurrent_state = None;
            }
        }
        Ok(())
    }
}

fn recurrent_domain_v2() -> StateDomainId {
    RECURRENT_STATE_DOMAIN
}

fn convolution_domain_v2() -> StateDomainId {
    CONVOLUTION_STATE_DOMAIN
}

#[derive(Clone)]
struct ConvRingState {
    storage: ConvHistoryStorage,
}

#[derive(Clone)]
enum ConvHistoryStorage {
    /// Portable circular storage used by the generic convolution fallback.
    Ring { slots: Vec<Tensor>, next_idx: usize },
    /// Oldest-to-newest `[channels, history]` storage returned by the dedicated
    /// Qwen3.8 CUDA decode convolution. Keeping this tensor intact avoids
    /// rebuilding the same history with `Tensor::cat` on every decode token.
    Contiguous(Tensor),
}

impl ConvRingState {
    fn from_slots(slots: Vec<Tensor>) -> Self {
        Self {
            storage: ConvHistoryStorage::Ring { slots, next_idx: 0 },
        }
    }

    fn from_staged(history: Tensor) -> Result<Self> {
        match history.rank() {
            // Canonical Qwen3.8 physical representation. The dedicated CUDA
            // decode kernel can consume this handle on the next quantum.
            2 => {
                let (channels, history_len) = history.dims2()?;
                if channels == 0 || history_len == 0 {
                    return Err(Error::InferenceError(
                        "Qwen3.8 staged convolution history is empty".into(),
                    ));
                }
                Ok(Self {
                    storage: ConvHistoryStorage::Contiguous(history),
                })
            }
            // Read the legacy `[history, channels, 1]` representation so
            // sessions staged before this optimization remain portable.
            3 => {
                let (history_len, channels, singleton) = history.dims3()?;
                if history_len == 0 || channels == 0 || singleton != 1 {
                    return Err(Error::InferenceError(format!(
                        "Invalid legacy Qwen3.8 convolution history shape {:?}",
                        history.dims()
                    )));
                }
                Ok(Self {
                    storage: ConvHistoryStorage::Contiguous(history.squeeze(2)?.transpose(0, 1)?),
                })
            }
            rank => Err(Error::InferenceError(format!(
                "Invalid Qwen3.8 convolution history rank {rank}"
            ))),
        }
    }

    fn account_storage(&self, accounting: &mut TensorStorageAccounting) -> Option<()> {
        match &self.storage {
            ConvHistoryStorage::Ring { slots, .. } => {
                for slot in slots {
                    accounting.add_tensor(slot)?;
                }
            }
            ConvHistoryStorage::Contiguous(history) => accounting.add_tensor(history)?,
        }
        Some(())
    }

    fn history_len(&self) -> Result<usize> {
        match &self.storage {
            ConvHistoryStorage::Ring { slots, next_idx } => {
                if slots.is_empty() || *next_idx >= slots.len() {
                    return Err(Error::InferenceError(format!(
                        "Invalid Qwen3.8 convolution ring: slots={}, next_idx={next_idx}",
                        slots.len()
                    )));
                }
                Ok(slots.len())
            }
            ConvHistoryStorage::Contiguous(history) => history.dim(1).map_err(Error::from),
        }
    }

    /// Return oldest-to-newest CUDA kernel history without copying when the
    /// previous specialized decode already produced that representation.
    fn contiguous_history(&self) -> Result<Tensor> {
        match &self.storage {
            ConvHistoryStorage::Contiguous(history) => Ok(history.clone()),
            ConvHistoryStorage::Ring { slots, next_idx } => {
                if slots.is_empty() || *next_idx >= slots.len() {
                    return Err(Error::InferenceError(format!(
                        "Invalid Qwen3.8 convolution ring: slots={}, next_idx={next_idx}",
                        slots.len()
                    )));
                }
                let logical = (0..slots.len())
                    .map(|offset| &slots[(*next_idx + offset) % slots.len()])
                    .collect::<Vec<_>>();
                Tensor::cat(&logical, 1).map_err(Error::from)
            }
        }
    }

    fn replace_contiguous(&mut self, history: Tensor, expected_len: usize) -> Result<()> {
        let (channels, history_len) = history.dims2()?;
        if channels == 0 || history_len != expected_len {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.8 contiguous convolution history: channels={channels}, history={history_len}, expected_history={expected_len}"
            )));
        }
        self.storage = ConvHistoryStorage::Contiguous(history);
        Ok(())
    }

    fn ensure_ring(&mut self) -> Result<(&mut Vec<Tensor>, &mut usize)> {
        if let ConvHistoryStorage::Contiguous(history) = &self.storage {
            let history_len = history.dim(1)?;
            let slots = (0..history_len)
                .map(|index| history.narrow(1, index, 1).map_err(Error::from))
                .collect::<Result<Vec<_>>>()?;
            self.storage = ConvHistoryStorage::Ring { slots, next_idx: 0 };
        }
        match &mut self.storage {
            ConvHistoryStorage::Ring { slots, next_idx } => Ok((slots, next_idx)),
            ConvHistoryStorage::Contiguous(_) => unreachable!("history was converted to a ring"),
        }
    }

    fn staged_tensor(&self) -> Result<Tensor> {
        self.contiguous_history()
    }

    fn is_f32(&self) -> bool {
        match &self.storage {
            ConvHistoryStorage::Ring { slots, .. } => {
                slots.iter().all(|slot| slot.dtype() == DType::F32)
            }
            ConvHistoryStorage::Contiguous(history) => history.dtype() == DType::F32,
        }
    }

    /// Move every logical history slot into independent fixed-history storage.
    ///
    /// Sequence-prefill slots are views into the entire projected token span.
    /// Keeping those views in runtime state would retain the full projection
    /// instead of the fixed `kernel_size - 1` history required by the conv.
    fn compact_owned(&mut self) -> Result<()> {
        match &mut self.storage {
            ConvHistoryStorage::Ring { slots, .. } => {
                if slots.is_empty() {
                    return Ok(());
                }
                *slots = slots
                    .iter()
                    .map(deep_copy_tensor_storage)
                    .collect::<candle_core::Result<Vec<_>>>()?;
            }
            ConvHistoryStorage::Contiguous(history) => {
                *history = deep_copy_tensor_storage(history)?;
            }
        }
        Ok(())
    }

    /// Retain the current one-token projection as the newest decode slot.
    ///
    /// Decode projects exactly one token, so this view cannot retain a
    /// sequence-sized backing tensor. Sequence prefill detaches its final fixed
    /// history once at the sequence boundary instead.
    fn push_decode(&mut self, current: &Tensor) -> Result<()> {
        let (slots, next_idx) = self.ensure_ring()?;
        if slots.is_empty() || *next_idx >= slots.len() {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.8 convolution ring: slots={}, next_idx={}",
                slots.len(),
                *next_idx
            )));
        }
        // The ring holds the conv state arena's F32 dtype for the whole
        // session; an incoming projection in another dtype would poison the
        // ring with mixed-dtype slots that later `cat`/multiply fail on.
        let slot_dtype = slots[*next_idx].dtype();
        if current.dtype() != slot_dtype {
            return Err(Error::InferenceError(format!(
                "Qwen3.8 convolution ring dtype drift: ring {:?}, incoming {:?}",
                slot_dtype,
                current.dtype()
            )));
        }
        slots[*next_idx] = current.clone();
        *next_idx = (*next_idx + 1) % slots.len();
        Ok(())
    }
}

#[derive(Clone)]
enum Qwen38LayerRuntimeState {
    Linear {
        conv_state: Option<ConvRingState>,
        recurrent_state: Option<Tensor>,
    },
    Full,
}

struct Qwen38Layer {
    graphs: Arc<TensorIsland>,
    graphs_enabled: bool,
    graph_generation: u64,
    attn_norm: Qwen38RmsNorm,
    mixer: Qwen38Mixer,
    post_attention_norm: Qwen38RmsNorm,
    mlp: Qwen38Mlp,
}

enum Qwen38Mixer {
    Linear(Qwen38LinearAttention),
    Full(Qwen38FullAttention),
}

pub(super) struct Qwen38Mlp {
    graphs: Arc<TensorIsland>,
    graphs_enabled: bool,
    graph_generation: u64,
    fused_decode: bool,
    gate_up: Qwen38ProjectionGroup,
    down: Qwen38Projection,
}

pub(super) enum Qwen38Projection {
    /// The checkpoint's block-FP8 bytes and F32 block scales, executed by the
    /// compact FP8 kernels with the decode kernel chosen at load. With
    /// `prefill_gemm`, spans of [`FP8_PREFILL_GEMM_MIN_ROWS`] rows or more
    /// expand bounded weight tiles to the activation dtype for a dense GEMM
    /// (`fp8::block_fp8_projection_dequant_gemm`) instead; it is set only
    /// after the projection kind passed its load-time check.
    NativeFp8 {
        raw: super::native::RawBlockFp8Projection,
        kernel: Qwen38Fp8Kernel,
        prefill_gemm: bool,
    },
    Quantized(QMatMul),
    Dense(Linear),
}

/// Decode kernel of a native block-FP8 projection, fixed at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen38Fp8Kernel {
    /// The reference GEMV (and the prefill GEMM above four rows).
    Reference,
    /// The vectorized GEMV on eligible decode calls (CUDA, verified at load).
    Vectorized,
    /// Metal: always expand to F16 and multiply (the GEMV failed its check
    /// for this projection kind, or `IZWI_QWEN38_FP8_GEMV=legacy`).
    DequantOnly,
}

/// Metal runs the FP8 GEMV up to this many activation rows; a longer span
/// expands the weights to a transient F16 matrix for one matrix multiply.
const METAL_FP8_GEMV_MAX_ROWS: usize = 8;
/// The CPU runs Candle's Q8_0 matmul up to this many activation rows; a
/// longer span expands weight row tiles to F32 for one F32 matmul per tile.
const CPU_Q8_MATMUL_MAX_ROWS: usize = 8;
/// Weight rows the CPU Q8_0 prefill expands at a time (42 MB at K = 5120).
const CPU_Q8_PREFILL_TILE_ROWS: usize = 2048;
/// Activation rows from which a CUDA block-FP8 projection with the prefill
/// GEMM resolved runs it instead of the compact `mm` kernel. `mm` computes
/// 16-row output tiles and decodes every weight byte in software once per
/// tile, so at 16 rows and up its cost grows with the span while the
/// dequantized path's weight traffic (1 byte read, 2 written and read again
/// per weight, all tiles) stays fixed and cuBLAS runs at tensor-core rate.
/// Below one `mm` tile, `mm` reads each weight byte once and keeps the
/// smaller traffic. The load-time timings report both paths at 16 rows so the
/// crossover can be checked on the device.
pub(super) const FP8_PREFILL_GEMM_MIN_ROWS: usize = 16;

enum Qwen38ProjectionGroup {
    Separate(Vec<Qwen38Projection>),
    Packed {
        projection: Qwen38Projection,
        widths: Vec<usize>,
    },
}

/// Zero-centered RMSNorm (HF `Qwen3_5RMSNorm`): the checkpoint stores `w` and
/// the model applies `x̂ · (1 + w)`. The loader keeps the shifted gain F32
/// under every plan, so a half-precision activation is normalized in F32 and
/// cast back, which is HF's `(norm(x.float()) * (1 + w.float())).type_as(x)`.
pub(super) struct Qwen38RmsNorm {
    weight: Tensor,
    eps: f64,
    /// One-launch kernel (`kernels/cuda/norm.rs`) for 16-bit activations with
    /// this F32 gain, enabled by [`Qwen38TextModel::resolve_fast_paths`].
    fused: bool,
}

impl Qwen38RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            eps,
            fused: false,
        }
    }

    pub(super) fn set_fused(&mut self, fused: bool) {
        self.fused = fused;
    }

    fn fused_applies(&self, input: &Tensor) -> bool {
        self.fused
            && self.weight.dtype() == DType::F32
            && matches!(input.dtype(), DType::F16 | DType::BF16)
    }

    /// Normalize `input`, also returning the F32 temporaries a half-precision
    /// input needs, which a captured graph region must retain.
    fn forward_retaining(&self, input: &Tensor) -> candle_core::Result<(Tensor, Vec<Tensor>)> {
        if input.dtype() == self.weight.dtype() {
            let output = candle_nn::ops::rms_norm(input, &self.weight, self.eps as f32)?;
            return Ok((output, Vec::new()));
        }
        let wide = input.to_dtype(self.weight.dtype())?;
        let normalized = candle_nn::ops::rms_norm(&wide, &self.weight, self.eps as f32)?;
        Ok((normalized.to_dtype(input.dtype())?, vec![wide, normalized]))
    }

    /// `(residual + delta, norm(residual + delta))`: one launch when fused,
    /// otherwise the add and the composition. The sum is rounded to the
    /// activation dtype before the norm either way.
    pub(super) fn add_forward(
        &self,
        residual: &Tensor,
        delta: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.fused_applies(residual)
            && delta.dtype() == residual.dtype()
            && delta.dims() == residual.dims()
        {
            record_fused_rms_norm(true);
            return Ok(norm::add_rms_norm(
                residual,
                delta,
                &self.weight,
                self.eps as f32,
            )?);
        }
        let sum = (residual + delta)?;
        let normalized = self.forward(&sum)?;
        Ok((sum, normalized))
    }
}

impl Module for Qwen38RmsNorm {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        if self.fused_applies(input) {
            record_fused_rms_norm(false);
            return norm::rms_norm(input, &self.weight, self.eps as f32);
        }
        if input.device().is_cuda() {
            record_cuda_rms_norm_composition();
        }
        self.forward_retaining(input).map(|(output, _)| output)
    }
}

/// The residual stream at a layer boundary, with the previous layer's MLP
/// output optionally not yet added: the next norm folds that add into its
/// launch ([`Qwen38RmsNorm::add_forward`]).
struct Qwen38Hidden {
    residual: Tensor,
    pending: Option<Tensor>,
}

impl Qwen38Hidden {
    fn new(residual: Tensor) -> Self {
        Self {
            residual,
            pending: None,
        }
    }

    /// The pre-norm hidden state with any pending add applied.
    fn materialize(self) -> Result<Tensor> {
        match self.pending {
            Some(delta) => (&self.residual + &delta).map_err(Error::from),
            None => Ok(self.residual),
        }
    }

    /// `(residual stream, norm(residual stream))` for the next layer.
    fn normalized(self, norm: &Qwen38RmsNorm) -> Result<(Tensor, Tensor)> {
        match self.pending {
            Some(delta) => norm.add_forward(&self.residual, &delta),
            None => {
                let normalized = norm.forward(&self.residual)?;
                Ok((self.residual, normalized))
            }
        }
    }
}

impl Module for Qwen38Projection {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        if input.device().is_cuda() {
            record_cuda_projection(match self {
                Self::NativeFp8 { .. } => CudaProjectionPath::NativeFp8,
                Self::Quantized(_) => CudaProjectionPath::Q8,
                Self::Dense(_) => CudaProjectionPath::Dense,
            });
        }
        match self {
            Self::NativeFp8 {
                raw,
                kernel,
                prefill_gemm,
            } => {
                let vectorized = *kernel == Qwen38Fp8Kernel::Vectorized;
                if *prefill_gemm
                    && input.elem_count() / raw.shape[1].max(1) >= FP8_PREFILL_GEMM_MIN_ROWS
                {
                    if input.device().is_cuda() {
                        record_cuda_fp8_prefill_gemm();
                    }
                    let output = crate::kernels::cuda::fp8::block_fp8_projection_dequant_gemm(
                        input,
                        &raw.weights,
                        &raw.scales,
                    )?;
                    #[cfg(test)]
                    let output = prefill_gemm_fault::apply(output)?;
                    return Ok(output);
                }
                if input.device().is_metal() {
                    let rows = input.elem_count() / raw.shape[1].max(1);
                    if *kernel == Qwen38Fp8Kernel::DequantOnly || rows > METAL_FP8_GEMV_MAX_ROWS {
                        record_metal_fp8_projection(false);
                        let dense =
                            crate::kernels::metal_fp8::dequantize_f16(&raw.weights, &raw.scales)?
                                .to_dtype(input.dtype())?;
                        return Linear::new(dense, None).forward(input);
                    }
                    record_metal_fp8_projection(true);
                }
                if input.device().is_cuda() {
                    let rows = input.elem_count() / raw.shape[1].max(1);
                    if rows <= 4 {
                        record_cuda_fp8_gemv(vectorized);
                    }
                }
                crate::kernels::cuda::fp8::block_fp8_projection_selecting(
                    input,
                    &raw.weights,
                    &raw.scales,
                    vectorized,
                )
            }
            Self::Quantized(projection) => {
                if let (true, QMatMul::QTensor(weights)) = (input.device().is_cpu(), projection) {
                    let k = weights.shape().dims2()?.1;
                    let rows = input.elem_count() / k.max(1);
                    if input.dtype() == DType::F32
                        && weights.dtype() == candle_core::quantized::GgmlDType::Q8_0
                    {
                        let tiled = rows > CPU_Q8_MATMUL_MAX_ROWS;
                        record_cpu_q8_projection(tiled);
                        if tiled {
                            return crate::kernels::cpu_q8::dequant_tile_matmul(
                                input,
                                weights,
                                CPU_Q8_PREFILL_TILE_ROWS,
                            );
                        }
                    }
                }
                projection.forward(input)
            }
            Self::Dense(projection) => projection.forward(input),
        }
    }
}

impl Qwen38Projection {
    /// A native block-FP8 projection with the prefill GEMM switched to
    /// `enabled` (same resident tensors); `None` for other forms.
    pub(super) fn with_prefill_gemm(&self, enabled: bool) -> Option<Self> {
        match self {
            Self::NativeFp8 { raw, kernel, .. } => Some(Self::NativeFp8 {
                raw: raw.clone(),
                kernel: *kernel,
                prefill_gemm: enabled,
            }),
            Self::Quantized(_) | Self::Dense(_) => None,
        }
    }
}

/// Tests: a fault injected into the dequantized prefill GEMM's output only
/// (x1.5), so the load-time check and its fallback can be RED-checked.
#[cfg(test)]
pub(super) mod prefill_gemm_fault {
    use std::cell::Cell;

    thread_local! {
        static FAULT: Cell<bool> = const { Cell::new(false) };
    }

    pub(in crate::models::architectures::qwen38) struct Injected;

    impl Drop for Injected {
        fn drop(&mut self) {
            FAULT.with(|fault| fault.set(false));
        }
    }

    /// Corrupt every prefill GEMM on this thread until the guard drops.
    pub(in crate::models::architectures::qwen38) fn inject() -> Injected {
        FAULT.with(|fault| fault.set(true));
        Injected
    }

    pub(super) fn apply(output: candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        if FAULT.with(Cell::get) {
            output * 1.5
        } else {
            Ok(output)
        }
    }
}

impl Qwen38ProjectionGroup {
    fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        match self {
            Self::Separate(projections) => projections
                .iter()
                .map(|projection| projection.forward(input).map_err(Error::from))
                .collect(),
            Self::Packed { projection, widths } => {
                if input.device().is_cuda()
                    && matches!(projection, Qwen38Projection::NativeFp8 { .. })
                {
                    record_cuda_fp8_packed_projection();
                }
                let output = projection.forward(input)?;
                split_projection_output(&output, widths)
            }
        }
    }
}

pub(super) struct Qwen38FullAttention {
    qkv_proj: Qwen38ProjectionGroup,
    o_proj: Qwen38Projection,
    q_norm: Qwen38RmsNorm,
    k_norm: Qwen38RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    mrope_sections: Vec<usize>,
    rope_kernel_enabled: bool,
    rope_inv_freqs: Vec<f32>,
    /// Fused q/k norm + partial M-RoPE for a decode token, with its F32
    /// inverse frequencies on the device. Resolved at load.
    fused_qk: Option<(QkNormRopeSpec, Tensor)>,
    fused_qk_path: Qwen38FastPath,
    /// The fused q/k launch also writes the rotated key and the value into
    /// the paged cache (`write_fused_and_attend`).
    fused_kv_write: bool,
    fused_kv_write_path: Qwen38FastPath,
    /// The gated attention output in one launch.
    fused_gate: bool,
    fused_gate_path: Qwen38FastPath,
}

/// Forward-scoped rotary tensors shared by every Qwen3.8 full-attention layer.
///
/// Qwen3.8 uses the same rotary configuration in each full-attention layer, so
/// rebuilding these position-dependent tensors in every layer only repeats
/// device work. The plan deliberately contains no kernel-policy decision: each
/// layer still selects the rotary kernel or the exact manual fallback itself.
pub(super) struct Qwen38MropePlan {
    cos: Option<Tensor>,
    sin: Option<Tensor>,
    sequence_len: usize,
    rope_dim: usize,
}

/// Forward-scoped rotary inputs shared by every full-attention layer: the
/// token positions; the cos/sin plan, built on first use by a layer that runs
/// the composition (the fused kernel takes the positions as scalars); and,
/// for a text decode token whose layers write K/V from the fused kernel, the
/// device step `[0, position, flat slot]` (U32) that kernel reads.
pub(super) struct Qwen38RopeInputs<'a> {
    positions: &'a [[usize; 3]],
    device: Device,
    dtype: DType,
    plan: std::cell::OnceCell<Qwen38MropePlan>,
    step: Option<Tensor>,
}

impl<'a> Qwen38RopeInputs<'a> {
    pub(super) fn new(positions: &'a [[usize; 3]], device: &Device, dtype: DType) -> Self {
        Self {
            positions,
            device: device.clone(),
            dtype,
            plan: std::cell::OnceCell::new(),
            step: None,
        }
    }

    /// Upload the step buffer when the forward is one text token at the
    /// cache's next slot and some layer writes K/V from the fused kernel: one
    /// 12-byte copy per forward, shared by every such layer. The flat slot is
    /// `page * page_tokens + offset`, the arena's own slot lowering.
    pub(super) fn with_decode_step<'b>(
        mut self,
        cache: &PhysicalPagedKvCache,
        mut layers: impl Iterator<Item = &'b Qwen38FullAttention>,
    ) -> Result<Self> {
        let [[temporal, height, width]] = self.positions else {
            return Ok(self);
        };
        if temporal != height || temporal != width || !layers.any(|layer| layer.fused_kv_write) {
            return Ok(self);
        }
        let slot = cache.slots_for_append(cache.context_len(), 1)?[0];
        let flat = slot
            .block
            .index
            .checked_mul(cache.arena().config().page_tokens)
            .and_then(|base| base.checked_add(slot.offset));
        if let (Some(flat), Ok(position)) = (flat, u32::try_from(*temporal)) {
            self.step = Some(Tensor::from_vec(
                vec![0u32, position, flat],
                3,
                &self.device,
            )?);
        }
        Ok(self)
    }

    /// The cos/sin plan, built by the first layer that needs it.
    fn plan(&self, attention: &Qwen38FullAttention) -> Result<&Qwen38MropePlan> {
        if let Some(plan) = self.plan.get() {
            return Ok(plan);
        }
        let plan = attention.prepare_mrope_plan(self.positions, &self.device, self.dtype)?;
        Ok(self.plan.get_or_init(|| plan))
    }
}

struct Qwen38LinearAttention {
    fused_decode: bool,
    qkv_z_proj: Qwen38ProjectionGroup,
    alpha_beta_proj: Qwen38ProjectionGroup,
    dt_bias: Tensor,
    a: Tensor,
    conv_kernel: Tensor,
    conv_kernel_slices: Vec<Tensor>,
    norm: Qwen38GatedRmsNorm,
    out_proj: Qwen38Projection,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_dim: usize,
    kernel_size: usize,
    tiled_recurrence_enabled: bool,
    tiled_recurrence_tile_size_override: Option<usize>,
    /// Fused single-token decode: q38's own conv kernel, then the generic
    /// `kernels/cuda/gdn.rs` recurrence. Resolved at load.
    fused_gdn: Option<GdnDecodeSpec>,
    fused_gdn_path: Qwen38FastPath,
    /// The dense `in_proj_a`/`in_proj_b` weights as F32 rows `[2 * value
    /// heads, hidden]` (alpha rows, then beta rows) for the router GEMV
    /// (`moe::router_logits`): one izwi launch per decode token instead of a
    /// tiny cuBLAS GEMM, which also keeps cuBLAS out of captured decode
    /// graphs. Resolved at load.
    alpha_beta_rows: Option<Tensor>,
    alpha_beta_gemv_path: Qwen38FastPath,
    /// Grouped prefill (`kernels::cuda::qwen38_deltanet_sequence`): one
    /// launch per chunk straight from the conv output, no expanded q/k.
    /// Resolved at load; verification spans keep today's path.
    gdn_prefill: bool,
    gdn_prefill_path: Qwen38FastPath,
}

/// One synthetic row of DeltaNet decode projections for the fused self-check.
struct GdnSelfCheckRow {
    mixed_qkv: Tensor,
    z: Tensor,
    beta: Tensor,
    alpha: Tensor,
}

struct Qwen38GatedRmsNorm {
    fused_decode: bool,
    weight: Tensor,
    eps: f64,
}

impl Qwen38TextModel {
    pub fn load_native(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        device: &Device,
        target: ProjectionMaterialization,
    ) -> Result<Self> {
        let performance = crate::performance::PerformanceConfig::default().resolve_env()?;
        Self::load_native_with_performance(tensors, native, device, target, &performance.cuda)
    }

    /// Load under the residency policy `performance` resolves for `device`
    /// (the CUDA performance section; the other sections at their defaults).
    pub fn load_native_with_performance(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        device: &Device,
        target: ProjectionMaterialization,
        performance: &CudaPerformanceConfig,
    ) -> Result<Self> {
        let mut config = crate::performance::PerformanceConfig::default();
        config.cuda = performance.clone();
        let policy = Qwen38ResidencyPolicy::resolve(device, target, &config, tensors, native);
        Self::load_native_with_policy(tensors, native, device, target, performance, &policy)
    }

    /// Load under an already resolved residency `policy`: the one load
    /// admission priced.
    pub(crate) fn load_native_with_policy(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        device: &Device,
        target: ProjectionMaterialization,
        performance: &CudaPerformanceConfig,
        policy: &Qwen38ResidencyPolicy,
    ) -> Result<Self> {
        let cfg = &native.text;
        let loader = Qwen38WeightLoader::new(
            tensors,
            policy,
            device,
            native.block_fp8.block_shape,
            target,
        );
        let projection_representation = policy.representation();
        let embedding_weights = tensors.materialize_dense_tensor(
            "model.language_model.embed_tokens.weight",
            &[native.vocab_size, cfg.embedding_length],
            target,
            device,
        )?;
        let token_embeddings = Embedding::new(embedding_weights, cfg.embedding_length);
        let output_norm = load_native_zero_centered_norm(
            tensors,
            "model.language_model.norm.weight",
            cfg.embedding_length,
            cfg.attention_layer_norm_rms_epsilon,
            device,
        )?;
        let (output, fp8_lm_head) =
            load_lm_head(&loader, [native.vocab_size, cfg.embedding_length])?;

        let graphs = Arc::new(TensorIsland::default());
        let mut layers = Vec::with_capacity(cfg.block_count);
        for (layer_idx, layer_type) in native.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            let attn_norm = load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.input_layernorm.weight"),
                cfg.embedding_length,
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?;
            let post_attention_norm = load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                cfg.embedding_length,
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?;
            let mut mlp = Qwen38Mlp::load_native(&loader, &prefix, cfg, performance)?;
            let mixer = match layer_type {
                Qwen38LayerType::FullAttention => {
                    Qwen38Mixer::Full(Qwen38FullAttention::load_native(&loader, &prefix, cfg)?)
                }
                Qwen38LayerType::LinearAttention => Qwen38Mixer::Linear(
                    Qwen38LinearAttention::load_native(&loader, &prefix, cfg, performance)?,
                ),
            };
            mlp.graphs = graphs.clone();
            mlp.graph_generation = layer_idx as u64;
            mlp.graphs_enabled &= layer_idx < GRAPH_LAYER_LIMIT;
            layers.push(Qwen38Layer {
                graphs: graphs.clone(),
                graphs_enabled: layer_idx < GRAPH_LAYER_LIMIT
                    && device.is_cuda()
                    && performance.enabled()
                    && performance.decode_graphs.enabled(),
                graph_generation: layer_idx as u64,
                attn_norm,
                mixer,
                post_attention_norm,
                mlp,
            });
        }
        let outcomes = loader.into_outcomes();
        let mut model = Self {
            device: device.clone(),
            projection_representation,
            residency: policy.clone(),
            fp8_packing: outcomes.packing,
            fp8_gemv_checks: outcomes.gemv,
            fp8_prefill_gemm_checks: outcomes.prefill_gemm,
            fp8_lm_head,
            token_embeddings,
            layers,
            output_norm,
            output,
            finite_diagnostics_enabled: qwen38_env_bool("IZWI_QWEN38_FINITE_DIAGNOSTICS", false),
            fused_norm_path: Qwen38FastPath::default(),
            decode_graphs: None,
            decode_graphs_path: Qwen38FastPath::default(),
            max_context_tokens: cfg.context_length,
        };
        let switches = Qwen38CudaSwitches::for_device(device, performance);
        model.resolve_fast_paths(&switches, false);
        model.resolve_decode_graphs(&switches, false, true);
        Ok(model)
    }

    /// The residency policy this trunk was loaded under.
    pub(crate) fn residency(&self) -> &Qwen38ResidencyPolicy {
        &self.residency
    }

    pub(super) fn graph_diagnostics(&self) -> serde_json::Value {
        self.layers
            .first()
            .map_or_else(|| serde_json::json!({}), |layer| layer.graphs.diagnostics())
    }

    pub fn new_state(&self) -> Qwen38TextRuntimeState {
        Qwen38TextRuntimeState {
            layers: self.layers.iter().map(Qwen38Layer::new_state).collect(),
        }
    }

    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    pub fn hidden_size(&self) -> usize {
        self.token_embeddings.hidden_size()
    }

    pub fn projection_representation(&self) -> Qwen38ProjectionRepresentation {
        self.projection_representation
    }

    pub(crate) fn forward_token_id_at_physical(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_token_id_hidden_at_physical(token_id, position_ids, state, cache)?;
        self.forward_hidden_to_logits(&hidden)
    }

    /// Execute one token for a ragged set of retained hybrid sessions.
    ///
    /// Model-wide projections and MLPs retain a real batch dimension. Full
    /// attention writes all rows through one paged decode dispatch per layer;
    /// DeltaNet keeps each row's convolution and recurrent state independent
    /// while sharing its input/output projections.
    pub(super) fn forward_token_ids_batch_at_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        states: &mut [&mut Qwen38TextRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Qwen38TextDecodeBatchOutput> {
        let batch_size = token_ids.len();
        if batch_size == 0
            || position_ids.len() != batch_size
            || states.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 hybrid decode batch rows do not match".into(),
            ));
        }
        for state in states.iter() {
            self.validate_runtime_state(state)?;
        }
        let sparse_layers = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                matches!(layer.mixer, Qwen38Mixer::Full(_)).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let first_full = self
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Qwen38Mixer::Full(attention) => Some(attention),
                Qwen38Mixer::Linear(_) => None,
            })
            .ok_or_else(|| {
                Error::InferenceError("Qwen3.8 model has no full-attention layer".into())
            })?;
        let start_positions = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        for cache in caches.iter() {
            cache.validate_sparse_model(
                &sparse_layers,
                first_full.num_kv_heads,
                first_full.head_dim,
                first_full.head_dim,
            )?;
        }

        let first = &*caches[0];
        let combined_slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(start_positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let lowered = first.arena().lower_slots(&combined_slots)?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(start_positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        if let (1, Some(graphs)) = (batch_size, &self.decode_graphs) {
            if let Some(output) = self.graph_decode_row(
                graphs,
                token_ids[0],
                position_ids[0],
                combined_slots[0],
                states[0],
                caches[0],
                lowered.as_ref(),
                &metadata,
                sparse_layers.len(),
            )? {
                return Ok(output);
            }
        }
        let first = &*caches[0];
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;
        let execution = (|| -> Result<(Tensor, Tensor)> {
            let embedded = self.embed_decode_token_ids(token_ids)?;
            let cache_refs = caches
                .iter()
                .map(|cache| &**cache)
                .collect::<Vec<&PhysicalPagedKvCache>>();
            let hidden = self.standard_decode_hidden(
                embedded,
                position_ids,
                states,
                &cache_refs,
                lowered.as_ref(),
                &metadata,
                &mut completions,
                sparse_layers.len(),
            )?;
            let logits = self.project_target_hidden_span(&hidden)?;
            Ok((hidden, logits))
        })();
        let (hidden, logits) = match execution {
            Ok(output) => output,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3.8 target batch failed: {error}; write-fence drain also failed: {drain}"
                ))),
                }
            }
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(start_positions[row], 1, completion.clone())?;
        }
        Ok(Qwen38TextDecodeBatchOutput {
            hidden_states: Some(hidden),
            logits,
        })
    }

    /// The standard (uncaptured) decode of a batch of rows: every layer in
    /// turn, attention through the arena. Returns the pre-norm hidden.
    #[allow(clippy::too_many_arguments)]
    fn standard_decode_hidden(
        &self,
        embedded: Tensor,
        position_ids: &[[usize; 3]],
        states: &mut [&mut Qwen38TextRuntimeState],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        attention_layers: usize,
    ) -> Result<Tensor> {
        let mut hidden = Qwen38Hidden::new(embedded);
        let mut physical_layer = 0usize;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_states = states
                .iter_mut()
                .map(|state| &mut state.layers[layer_index])
                .collect::<Vec<_>>();
            for layer_state in layer_states.iter_mut() {
                layer.ensure_state_initialized(layer_state, &self.device)?;
            }
            hidden = layer.forward_physical_decode_batch(
                hidden,
                &mut layer_states,
                position_ids,
                caches,
                slots,
                metadata,
                completions,
                &mut physical_layer,
            )?;
            hidden =
                self.validate_layer_output(hidden, layer_index, layer.decode_diagnostic_path())?;
        }
        if physical_layer != attention_layers {
            return Err(Error::InferenceError(
                "Qwen3.8 batched attention did not cover every sparse layer".into(),
            ));
        }
        hidden.materialize()
    }

    /// One row through graph decode, its KV writes collected apart and
    /// committed only when the graphs took the step. `Ok(None)`: the graphs
    /// did not take it (nothing was published, and any KV write they queued
    /// was waited for), so the caller runs the standard decode, which
    /// rewrites the slot.
    #[allow(clippy::too_many_arguments)]
    fn graph_decode_row(
        &self,
        graphs: &decode_graph::Qwen38DecodeGraphs,
        token: u32,
        position: [usize; 3],
        slot: KvSlotRef,
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        attention_layers: usize,
    ) -> Result<Option<Qwen38TextDecodeBatchOutput>> {
        let start = cache.context_len();
        let embedded = self.embed_decode_token_ids(&[token])?;
        let view: &PhysicalPagedKvCache = cache;
        let mut completions =
            KvWriteCompletionCollector::new(view.arena().config(), slots.logical_slots())?;
        // The warm check: the standard decode on a copy of the row state, its
        // writes collected and sealed apart.
        let reference_state = state.clone();
        let mut reference = || -> Result<Tensor> {
            let mut reference_state = reference_state.clone();
            let mut collector =
                KvWriteCompletionCollector::new(view.arena().config(), slots.logical_slots())?;
            let hidden = self.standard_decode_hidden(
                embedded.clone(),
                &[position],
                &mut [&mut reference_state],
                &[view],
                slots,
                metadata,
                &mut collector,
                attention_layers,
            );
            let sealed = collector.seal();
            let logits = self.project_target_hidden_span(&hidden?)?;
            sealed?;
            Ok(logits)
        };
        let output = graphs.forward(
            self,
            token,
            &embedded,
            position,
            slot,
            state,
            view,
            slots,
            metadata,
            &mut completions,
            &mut reference,
        );
        let (hidden_states, logits) = match output {
            Some(decode_graph::GraphDecodeOutput::Logits(logits)) => (None, logits),
            Some(decode_graph::GraphDecodeOutput::Hidden(hidden)) => {
                match self.project_target_hidden_span(&hidden) {
                    Ok(logits) => (Some(hidden), logits),
                    Err(error) => {
                        return match completions.drain() {
                            Ok(()) => Err(error),
                            Err(drain) => Err(Error::InferenceError(format!(
                                "Qwen3.8 graph decode head failed: {error}; write-fence drain also failed: {drain}"
                            ))),
                        }
                    }
                }
            }
            None => {
                completions.drain()?;
                return Ok(None);
            }
        };
        let completion = Arc::new(completions.seal()?);
        cache.commit_shared_completion(start, 1, completion)?;
        Ok(Some(Qwen38TextDecodeBatchOutput {
            hidden_states,
            logits,
        }))
    }

    pub(super) fn forward_token_id_hidden_at_physical(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let hidden = self.embed_token_ids(&[token_id])?;
        self.forward_hidden_physical(&hidden, &[position_ids], state, cache)
    }

    pub(crate) fn prefill_token_ids_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        Ok(self
            .prefill_token_ids_with_hidden_physical(
                token_ids,
                position_ids,
                state,
                cache,
                compute_logits,
            )?
            .and_then(|output| output.logits))
    }

    pub(super) fn prefill_token_ids_with_hidden_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Option<Qwen38TextPrefillOutput>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 physical prefill span mismatch: {} token ids for {} position ids",
                token_ids.len(),
                position_ids.len()
            )));
        }
        record_prefill_sequence_span(token_ids.len());
        let input = self.embed_token_ids(token_ids)?;
        let hidden = self.forward_hidden_physical(&input, position_ids, state, cache)?;
        let logits = if compute_logits {
            let last = hidden.narrow(1, token_ids.len() - 1, 1)?;
            Some(self.forward_hidden_to_logits(&last)?)
        } else {
            None
        };
        Ok(Some(Qwen38TextPrefillOutput {
            hidden_states: hidden,
            logits,
        }))
    }

    pub(super) fn embed_token_ids(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen3.8 shared embedding received no token ids".into(),
            ));
        }
        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        self.token_embeddings.forward(&input).map_err(Error::from)
    }

    pub(super) fn embed_decode_token_ids(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen3.8 batched decode embedding received no token ids".into(),
            ));
        }
        let input = Tensor::from_slice(token_ids, (token_ids.len(), 1), &self.device)?;
        self.token_embeddings.forward(&input).map_err(Error::from)
    }

    /// Project an already normalized MTP hidden span through the target LM
    /// head without applying the target model's output norm.
    pub(super) fn project_with_shared_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head_forward(hidden)
    }

    /// The LM head, shared by the target logits and every MTP draft.
    fn lm_head_forward(&self, hidden: &Tensor) -> Result<Tensor> {
        if hidden.device().is_cuda() {
            record_cuda_lm_head(matches!(self.output, Qwen38Projection::NativeFp8 { .. }));
        }
        self.output.forward(hidden).map_err(Error::from)
    }

    pub(crate) fn forward_input_embedding_at_physical(
        &self,
        input_embedding: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_hidden_physical(input_embedding, &[position_ids], state, cache)?;
        self.forward_hidden_to_logits(&hidden)
    }

    pub(super) fn forward_hidden_physical(
        &self,
        input: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        self.forward_hidden_physical_recording(input, position_ids, state, cache, None)
    }

    pub(super) fn verify_token_ids_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Qwen38TextVerification> {
        if !(2..=4).contains(&token_ids.len()) || token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(
                "Qwen3.8 verification requires two through four positions".into(),
            ));
        }
        // Initialize before retaining immutable checkpoint handles.
        for (layer, state) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.ensure_state_initialized(state, &self.device)?;
        }
        let initial_state = state.clone();
        let start_position = cache.context_len();
        let mut layers = (0..self.layers.len()).map(|_| None).collect::<Vec<_>>();
        let hidden_states = self.forward_hidden_physical_recording(
            &self.embed_token_ids(token_ids)?,
            position_ids,
            state,
            cache,
            Some(&mut layers),
        )?;
        Ok(Qwen38TextVerification {
            hidden_states,
            initial_state,
            final_state: state.clone(),
            layers,
            start_position,
            token_count: token_ids.len(),
            view_id: cache.view_id(),
            generation: cache.logical_generation(),
        })
    }

    fn forward_hidden_physical_recording(
        &self,
        input: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        mut verification: Option<&mut Vec<Option<Qwen38LinearVerification>>>,
    ) -> Result<Tensor> {
        self.validate_runtime_state(state)?;
        let (_, sequence_len, hidden_size) = input.dims3()?;
        if sequence_len == 0
            || sequence_len != position_ids.len()
            || hidden_size != self.hidden_size()
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 physical hidden span does not match its positions or model width".into(),
            ));
        }
        let sparse_layers = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                matches!(layer.mixer, Qwen38Mixer::Full(_)).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let first_full = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen38Mixer::Full(attention) => Some(attention),
            Qwen38Mixer::Linear(_) => None,
        });
        let first_full = first_full.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 model has no full-attention layer".into())
        })?;
        cache.validate_sparse_model(
            &sparse_layers,
            first_full.num_kv_heads,
            first_full.head_dim,
            first_full.head_dim,
        )?;
        let start_pos = cache.context_len();
        let rope = Qwen38RopeInputs::new(position_ids, input.device(), input.dtype())
            .with_decode_step(cache, self.full_attention_layers())?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.ensure_state_initialized(layer_state, &self.device)?;
        }
        let mut hidden = Qwen38Hidden::new(input.clone());
        let mut physical_layer = 0usize;
        for (layer_index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            hidden = layer.forward_physical(
                hidden,
                layer_state,
                &rope,
                cache,
                &mut prepared,
                &mut physical_layer,
                verification.as_mut().map(|layers| &mut layers[layer_index]),
            )?;
            hidden = self.validate_layer_output(
                hidden,
                layer_index,
                if sequence_len == 1 {
                    layer.decode_diagnostic_path()
                } else {
                    layer.prefill_diagnostic_path()
                },
            )?;
        }
        if physical_layer != sparse_layers.len() {
            return Err(Error::InferenceError(
                "Qwen3.8 physical attention did not cover every sparse layer".into(),
            ));
        }
        let hidden = hidden.materialize()?;
        cache.commit_prepared(prepared)?;
        Ok(hidden)
    }

    /// Finite-value diagnostics over a layer's output. Only when they are
    /// enabled does the pending residual add materialize here instead of in
    /// the next norm.
    fn validate_layer_output(
        &self,
        hidden: Qwen38Hidden,
        layer_index: usize,
        path: &'static str,
    ) -> Result<Qwen38Hidden> {
        if !self.finite_diagnostics_enabled {
            return Ok(hidden);
        }
        let hidden = hidden.materialize()?;
        validate_qwen38_finite_tensor(&hidden, layer_index, path, true)?;
        Ok(Qwen38Hidden::new(hidden))
    }

    pub fn forward_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.project_target_hidden_span(hidden)?
            .i((0, 0))
            .map_err(Error::from)
    }

    pub(super) fn project_target_hidden_span(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.output_norm.forward(hidden)?;
        validate_qwen38_finite_tensor(
            &hidden,
            self.layers.len(),
            "output.norm",
            self.finite_diagnostics_enabled,
        )?;
        let logits = self.lm_head_forward(&hidden)?;
        validate_qwen38_finite_tensor(
            &logits,
            self.layers.len(),
            "output.logits",
            self.finite_diagnostics_enabled,
        )?;
        Ok(logits)
    }

    /// Resolve the fused fast paths at load: CUDA in production, while
    /// `allow_cpu` resolves them against the portable CPU references in tests.
    /// Each path turns on only after its self-check on this device passes.
    pub(super) fn resolve_fast_paths(&mut self, switches: &Qwen38CudaSwitches, allow_cpu: bool) {
        self.resolve_fused_norms(switches, allow_cpu);
        let dtype = self.activation_dtype();
        let mut gdn_prefill = None;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Qwen38Mixer::Linear(linear) => {
                    linear.resolve_fused_gdn(switches, &self.device, dtype, allow_cpu);
                    linear.resolve_alpha_beta_gemv(switches, &self.device, dtype, allow_cpu);
                    linear.resolve_gdn_prefill(switches, &self.device, allow_cpu, &mut gdn_prefill);
                }
                Qwen38Mixer::Full(attention) => {
                    attention.resolve_fused_attention(switches, &self.device, dtype, allow_cpu)
                }
            }
        }
    }

    /// Resolve every fast path against the portable CPU references (tests
    /// only): production resolution enables them on CUDA alone.
    #[cfg(test)]
    pub(super) fn enable_fast_paths_for_tests(&mut self) {
        self.resolve_fast_paths(&Qwen38CudaSwitches::default(), true);
    }

    /// Fast-path resolutions for the admin diagnostics.
    pub(super) fn fast_path_diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "rms_norm": summarize([&self.fused_norm_path]),
            "gdn_decode": summarize(self.layers.iter().filter_map(|layer| match &layer.mixer {
                Qwen38Mixer::Linear(linear) => Some(&linear.fused_gdn_path),
                Qwen38Mixer::Full(_) => None,
            })),
            "gdn_prefill": summarize(self.layers.iter().filter_map(|layer| match &layer.mixer {
                Qwen38Mixer::Linear(linear) => Some(&linear.gdn_prefill_path),
                Qwen38Mixer::Full(_) => None,
            })),
            "gdn_alpha_beta_gemv": summarize(self.layers.iter().filter_map(|layer| match &layer.mixer {
                Qwen38Mixer::Linear(linear) => Some(&linear.alpha_beta_gemv_path),
                Qwen38Mixer::Full(_) => None,
            })),
            "qk_rope": summarize(self.full_attention_layers().map(|layer| layer.fast_paths()[0])),
            "kv_write": summarize(self.full_attention_layers().map(|layer| layer.fast_paths()[1])),
            "attn_gate": summarize(self.full_attention_layers().map(|layer| layer.fast_paths()[2])),
        })
    }

    /// Resolve graph decode after the fast paths it needs, with the
    /// whole-step graph when `whole_step` asks for it. When graph decode
    /// resolves, the `TensorIsland` regions retire: a capture cannot nest
    /// inside another.
    fn resolve_decode_graphs(
        &mut self,
        switches: &Qwen38CudaSwitches,
        allow_eager: bool,
        whole_step: bool,
    ) {
        let (graphs, path) =
            decode_graph::Qwen38DecodeGraphs::resolve(self, switches, allow_eager, whole_step);
        if graphs.is_some() {
            for layer in &mut self.layers {
                layer.graphs_enabled = false;
                layer.mlp.graphs_enabled = false;
            }
        }
        self.decode_graphs = graphs;
        self.decode_graphs_path = path;
    }

    /// Run single-row decode through the graph segments eagerly off CUDA
    /// (tests of the segmentation; CUDA resolves this at load). Call after
    /// the fast paths resolve.
    #[cfg(test)]
    pub(super) fn enable_segmented_decode_for_tests(&mut self) {
        self.resolve_decode_graphs(&Qwen38CudaSwitches::default(), true, false);
    }

    /// Run single-row decode through the whole-step body eagerly off CUDA
    /// (tests of the step body; CUDA resolves this at load). Call after the
    /// fast paths resolve.
    #[cfg(test)]
    pub(super) fn enable_step_graph_for_tests(&mut self) {
        self.resolve_decode_graphs(&Qwen38CudaSwitches::default(), true, true);
    }

    /// Whether single-row decode runs through the decode graphs. Such steps
    /// go through [`Self::forward_token_ids_batch_at_physical`], where the
    /// graphs live.
    pub(super) fn graph_decode(&self) -> bool {
        self.decode_graphs.is_some()
    }

    /// Whether a single-row step can be queued with a device-resident input
    /// token ([`Self::enqueue_graph_step`]): the whole-step graph is
    /// replaying.
    pub(super) fn graph_step_ready(&self) -> bool {
        self.decode_graphs
            .as_ref()
            .is_some_and(|graphs| graphs.whole_step_replaying())
    }

    /// The whole-step graph's token slot on the device (U32 `[1]`), where
    /// token selection writes the next step's input token.
    pub(super) fn graph_step_token(&self) -> Option<Result<Tensor>> {
        self.decode_graphs
            .as_ref()
            .and_then(|graphs| graphs.whole_step_token_slot())
    }

    /// Queue one single-row decode step at `position` through the replaying
    /// whole-step graph without waiting for it. `token` `None` takes the token
    /// the previous step's selection wrote into the step buffer. The step's
    /// DeltaNet state is published into `state`; its KV write is committed
    /// later, in order, by [`Self::commit_enqueued_step`]. `Ok(None)`: the
    /// graph did not take the step, and nothing was published.
    pub(super) fn enqueue_graph_step(
        &self,
        token: Option<u32>,
        position: usize,
        state: &mut Qwen38TextRuntimeState,
        cache: &PhysicalPagedKvCache,
    ) -> Result<Option<EnqueuedGraphStep>> {
        let Some(graphs) = &self.decode_graphs else {
            return Ok(None);
        };
        self.validate_runtime_state(state)?;
        let slot = cache.slots_for_queued_append(position, 1)?[0];
        let lowered = cache.arena().lower_slots(&[slot])?;
        let metadata = KvDecodeBatchMetadata {
            sequences: vec![cache.sequence_table(position + 1)?],
        };
        let mut completions =
            KvWriteCompletionCollector::new(cache.arena().config(), lowered.logical_slots())?;
        let output = graphs.forward_whole(
            self,
            token,
            position,
            slot,
            state,
            cache,
            lowered.as_ref(),
            &metadata,
            &mut completions,
        );
        let logits = match output {
            Ok(Some(decode_graph::GraphDecodeOutput::Logits(logits))) => Ok(logits),
            Ok(Some(decode_graph::GraphDecodeOutput::Hidden(hidden))) => {
                self.project_target_hidden_span(&hidden)
            }
            Ok(None) => {
                completions.drain()?;
                return Ok(None);
            }
            Err(error) => Err(error),
        };
        match logits {
            Ok(logits) => Ok(Some(EnqueuedGraphStep {
                logits,
                position,
                completions: Some(completions),
            })),
            Err(error) => match completions.drain() {
                Ok(()) => Err(error),
                Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3.8 queued graph step failed: {error}; write-fence drain also failed: {drain}"
                ))),
            },
        }
    }

    /// Commit a queued step's KV write (after its token was read back, so the
    /// write fences have passed), in order.
    pub(super) fn commit_enqueued_step(
        &self,
        mut step: EnqueuedGraphStep,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<()> {
        let completions = step
            .completions
            .take()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 graph step committed twice".into()))?;
        cache.commit_shared_completion(step.position, 1, Arc::new(completions.seal()?))
    }

    /// `cuda_graphs` in the admin diagnostics: resolution, phase and counters.
    pub(super) fn cuda_graphs_summary(&self) -> serde_json::Value {
        match &self.decode_graphs {
            Some(graphs) => graphs.summary(),
            None => {
                let mut summary = summarize([&self.decode_graphs_path]);
                summary["phase"] = serde_json::json!("off");
                summary
            }
        }
    }

    /// Packing outcome of every block-FP8 projection group of the trunk.
    pub(super) fn fp8_packing(&self) -> &[Qwen38FastPath] {
        &self.fp8_packing
    }

    /// Metal: the GEMV check outcome of every FP8 projection kind.
    pub(super) fn fp8_gemv_checks(&self) -> &[Qwen38FastPath] {
        &self.fp8_gemv_checks
    }

    /// CUDA: the dequantized prefill GEMM per checked FP8 projection kind.
    pub(super) fn fp8_prefill_gemm_checks(&self) -> &[Qwen38FastPath] {
        &self.fp8_prefill_gemm_checks
    }

    /// Whether any trunk projection kind runs the dequantized prefill GEMM.
    pub(super) fn prefill_gemm_active(&self) -> bool {
        self.fp8_prefill_gemm_checks
            .iter()
            .any(Qwen38FastPath::is_fused)
    }

    /// `fast_paths.fp8_lm_head`: the head's form and its observed error.
    pub(super) fn fp8_lm_head_diagnostics(&self) -> serde_json::Value {
        let (path, gate_error) = &self.fp8_lm_head;
        let mut summary = summarize([path]);
        summary["gate_relative_l2"] = serde_json::json!(gate_error);
        summary["precision"] = serde_json::json!(match &self.output {
            Qwen38Projection::NativeFp8 { .. } => "deliberately reduced: the BF16 head is quantized to block FP8 (E4M3, F32 scale per 128x128 block); logits stay in the activation dtype",
            Qwen38Projection::Quantized(_) => "reduced: the BF16 head is requantized to Q8_0 (F16 scale per 32 elements); logits stay F32",
            Qwen38Projection::Dense(_) => "checkpoint BF16 head in the activation dtype",
        });
        summary
    }

    fn full_attention_layers(&self) -> impl Iterator<Item = &Qwen38FullAttention> {
        self.layers.iter().filter_map(|layer| match &layer.mixer {
            Qwen38Mixer::Full(attention) => Some(attention),
            Qwen38Mixer::Linear(_) => None,
        })
    }

    /// The plan's activation dtype (the embedding table's).
    fn activation_dtype(&self) -> DType {
        self.token_embeddings.embeddings().dtype()
    }

    /// Fused residual add + RMSNorm on every trunk norm (layer, q/k and
    /// output norms), after a self-check of layer 0's input norm and the first
    /// q norm against the composition. `IZWI_QWEN38_FUSED_NORM=legacy` keeps
    /// the cast, norm, cast chain.
    fn resolve_fused_norms(&mut self, switches: &Qwen38CudaSwitches, allow_cpu: bool) {
        let dtype = self.activation_dtype();
        let path = resolve_norm_path(switches, &self.device, allow_cpu, dtype, || {
            let mut norms = Vec::with_capacity(2);
            if let Some(layer) = self.layers.first() {
                norms.push(&layer.attn_norm);
            }
            if let Some(attention) = self.layers.iter().find_map(|layer| match &layer.mixer {
                Qwen38Mixer::Full(attention) => Some(attention),
                Qwen38Mixer::Linear(_) => None,
            }) {
                norms.push(&attention.q_norm);
            }
            norms
                .into_iter()
                .try_for_each(|norm| norm_self_check(norm, &self.device, dtype))
        });
        let fused = path.is_fused();
        self.output_norm.fused = fused;
        for layer in &mut self.layers {
            layer.attn_norm.fused = fused;
            layer.post_attention_norm.fused = fused;
            if let Qwen38Mixer::Full(attention) = &mut layer.mixer {
                attention.q_norm.fused = fused;
                attention.k_norm.fused = fused;
            }
        }
        self.fused_norm_path = path;
    }

    fn validate_runtime_state(&self, state: &Qwen38TextRuntimeState) -> Result<()> {
        if state.layers.len() != self.layers.len() {
            return Err(Error::InferenceError(format!(
                "Qwen3.8 runtime state layer mismatch: state has {}, model has {}",
                state.layers.len(),
                self.layers.len()
            )));
        }
        Ok(())
    }
}

impl Qwen38Layer {
    fn decode_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen38Mixer::Linear(_) => "decode.linear_layer_output",
            Qwen38Mixer::Full(_) => "decode.full_attention_layer_output",
        }
    }

    fn prefill_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen38Mixer::Linear(_) => "prefill.linear_layer_output",
            Qwen38Mixer::Full(_) => "prefill.full_attention_layer_output",
        }
    }

    fn new_state(&self) -> Qwen38LayerRuntimeState {
        match self.mixer {
            Qwen38Mixer::Linear(_) => Qwen38LayerRuntimeState::Linear {
                conv_state: None,
                recurrent_state: None,
            },
            Qwen38Mixer::Full(_) => Qwen38LayerRuntimeState::Full,
        }
    }

    /// Pre-initialize lazy state tensors so the first-use allocation cost
    /// does not happen inside the per-token hot loop during prefill.
    fn ensure_state_initialized(
        &self,
        state: &mut Qwen38LayerRuntimeState,
        device: &Device,
    ) -> Result<()> {
        if let (
            Qwen38Mixer::Linear(mixer),
            Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            },
        ) = (&self.mixer, state)
        {
            if conv_state.is_none() && mixer.kernel_size > 1 {
                // Persistent runtime state cannot outlive a released scratch-pool
                // lease. Independent exact-size slots allow O(1) replacement
                // without retaining dead regions of a shared backing buffer.
                let history_len = mixer.kernel_size - 1;
                let mut slots = Vec::with_capacity(history_len);
                for _ in 0..history_len {
                    slots.push(owned_zero_tensor(&[mixer.conv_dim, 1], DType::F32, device)?);
                }
                *conv_state = Some(ConvRingState::from_slots(slots));
            }
            if recurrent_state.is_none() {
                *recurrent_state = Some(owned_zero_tensor(
                    &[1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim],
                    DType::F32,
                    device,
                )?);
            }
            // Full attention layers don't need pre-initialization.
        }
        Ok(())
    }

    fn residual_norm(&self, residual: &Tensor, mixed: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.post_attention_norm.fused_applies(residual) {
            // One launch already; the graph region would only wrap it.
            return self.post_attention_norm.add_forward(residual, mixed);
        }
        if self.graphs_enabled
            && residual
                .dims3()
                .is_ok_and(|(batch, rows, _)| batch * rows <= 4)
        {
            // Both outputs escape: the residual is needed after the FFN. The
            // RMS custom op has no growable external workspace; its only
            // allocations are its output, the explicitly retained sum and,
            // under a half-precision plan, the F32 input and output of the
            // F32-gain norm, retained as intermediates.
            let bound = residual
                .dtype()
                .size_in_bytes()
                .checked_add(DType::F32.size_in_bytes())
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| bytes.checked_mul(residual.elem_count()));
            if let Some(bound) = bound {
                if let Some(outputs) = self.graphs.run_multi(
                    &[residual, mixed],
                    std::slice::from_ref(&self.post_attention_norm.weight),
                    self.graph_generation,
                    "qwen38_residual_rms",
                    GRAPH_CACHE_BYTES,
                    bound,
                    |inputs| {
                        let sum = (&inputs[0] + &inputs[1])?;
                        let (normalized, intermediates) =
                            self.post_attention_norm.forward_retaining(&sum)?;
                        Ok(IslandOutput {
                            outputs: vec![sum, normalized],
                            intermediates,
                        })
                    },
                )? {
                    return Ok((outputs[0].clone(), outputs[1].clone()));
                }
            }
        }
        let sum = (residual + mixed)?;
        let normalized = self.post_attention_norm.forward(&sum)?;
        Ok((sum, normalized))
    }

    /// The layer tail after its mixer: residual add + post-attention norm,
    /// then the MLP, whose output the next norm adds.
    fn post_mixer(&self, residual: &Tensor, mixed: &Tensor) -> Result<Qwen38Hidden> {
        let (residual, hidden_states) = self.residual_norm(residual, mixed)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        Ok(Qwen38Hidden {
            residual,
            pending: Some(mlp_output),
        })
    }

    fn forward_physical(
        &self,
        hidden: Qwen38Hidden,
        state: &mut Qwen38LayerRuntimeState,
        rope: &Qwen38RopeInputs<'_>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: &mut usize,
        verification: Option<&mut Option<Qwen38LinearVerification>>,
    ) -> Result<Qwen38Hidden> {
        let (residual, normalized) = hidden.normalized(&self.attn_norm)?;
        let mixed = match &self.mixer {
            Qwen38Mixer::Linear(mixer) => {
                if normalized.dim(1)? == 1 {
                    mixer.forward(&normalized, state)?
                } else {
                    mixer.forward_sequence_recording(&normalized, state, verification)?
                }
            }
            Qwen38Mixer::Full(mixer) => {
                let output =
                    mixer.forward_physical(&normalized, rope, cache, prepared, *physical_layer)?;
                *physical_layer = physical_layer.checked_add(1).ok_or_else(|| {
                    Error::InvalidInput("Qwen3.8 physical layer ordinal overflow".into())
                })?;
                output
            }
        };
        self.post_mixer(&residual, &mixed)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_physical_decode_batch(
        &self,
        hidden: Qwen38Hidden,
        states: &mut [&mut Qwen38LayerRuntimeState],
        position_ids: &[[usize; 3]],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: &mut usize,
    ) -> Result<Qwen38Hidden> {
        let batch_size = hidden.residual.dim(0)?;
        if hidden.residual.dim(1)? != 1
            || batch_size == 0
            || states.len() != batch_size
            || position_ids.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 layer decode batch dimensions do not match".into(),
            ));
        }
        let (residual, normalized) = hidden.normalized(&self.attn_norm)?;
        let mixed = match &self.mixer {
            Qwen38Mixer::Linear(mixer) => mixer.forward_decode_batch(&normalized, states)?,
            Qwen38Mixer::Full(mixer) => {
                let output = mixer.forward_physical_decode_batch(
                    &normalized,
                    position_ids,
                    caches,
                    slots,
                    metadata,
                    completions,
                    *physical_layer,
                )?;
                *physical_layer = physical_layer.checked_add(1).ok_or_else(|| {
                    Error::InvalidInput("Qwen3.8 physical layer ordinal overflow".into())
                })?;
                output
            }
        };
        self.post_mixer(&residual, &mixed)
    }
}

impl Qwen38Mlp {
    pub(super) fn load_native(
        loader: &Qwen38WeightLoader<'_>,
        prefix: &str,
        cfg: &Qwen38TextConfig,
        performance: &CudaPerformanceConfig,
    ) -> Result<Self> {
        let device = loader.device;
        let gate_name = format!("{prefix}.mlp.gate_proj.weight");
        let up_name = format!("{prefix}.mlp.up_proj.weight");
        Ok(Self {
            graphs: Arc::new(TensorIsland::default()),
            graphs_enabled: device.is_cuda()
                && performance.enabled()
                && performance.decode_graphs.enabled(),
            graph_generation: 0,
            fused_decode: device.is_cuda()
                && performance.enabled()
                && performance.fused_decode.enabled(),
            gate_up: load_native_projection_group(
                loader,
                &[
                    (&gate_name, [cfg.feed_forward_length, cfg.embedding_length]),
                    (&up_name, [cfg.feed_forward_length, cfg.embedding_length]),
                ],
            )?,
            down: load_native_projection(
                loader,
                &format!("{prefix}.mlp.down_proj.weight"),
                [cfg.embedding_length, cfg.feed_forward_length],
            )?,
        })
    }

    pub(super) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if self.graphs_enabled
            && hidden_states
                .dims3()
                .is_ok_and(|(batch, rows, _)| batch * rows <= 4)
        {
            if let Some(owners) = self.native_graph_owners() {
                let rows = hidden_states.dim(0)? * hidden_states.dim(1)?;
                let ff = owners[0].dim(0)?;
                let bound = rows
                    .checked_mul(ff)
                    .and_then(|v| v.checked_mul(8))
                    .and_then(|v| v.checked_add(hidden_states.elem_count()))
                    .and_then(|v| v.checked_mul(hidden_states.dtype().size_in_bytes()));
                if let Some(bound) = bound {
                    if let Some(outputs) = self.graphs.run_multi(
                        &[hidden_states],
                        &owners,
                        self.graph_generation,
                        "qwen38_native_fp8_mlp",
                        GRAPH_CACHE_BYTES,
                        bound,
                        |inputs| self.native_graph_forward(&inputs[0]),
                    )? {
                        return Ok(outputs[0].clone());
                    }
                }
            }
        }
        self.forward_eager(hidden_states)
    }

    pub(super) fn graph_diagnostics(&self) -> serde_json::Value {
        self.graphs.diagnostics()
    }

    fn native_graph_owners(&self) -> Option<Vec<Tensor>> {
        // Never capture Candle QMatMul: its hidden device-global MMVQ scratch
        // can be reallocated by an unrelated projection after capture.
        let Qwen38ProjectionGroup::Separate(projections) = &self.gate_up else {
            return None;
        };
        if projections.len() != 2 {
            return None;
        }
        let mut owners = Vec::with_capacity(6);
        for projection in projections.iter().chain(std::iter::once(&self.down)) {
            let Qwen38Projection::NativeFp8 { raw, .. } = projection else {
                return None;
            };
            owners.push(raw.weights.clone());
            owners.push(raw.scales.clone());
        }
        Some(owners)
    }

    fn native_graph_forward(&self, input: &Tensor) -> candle_core::Result<IslandOutput> {
        let Qwen38ProjectionGroup::Separate(projections) = &self.gate_up else {
            candle_core::bail!("native FP8 graph requires separate projections")
        };
        let gate = projections[0].forward(input)?;
        let up = projections[1].forward(input)?;
        // Keep the explicit SiLU intermediate alive: capture owns every
        // allocation, including when a generic epilogue has no fused route.
        let activated = ops::silu(&gate)?;
        let product = (&activated * &up)?;
        let output = self.down.forward(&product)?;
        Ok(IslandOutput {
            outputs: vec![output],
            intermediates: vec![gate, up, activated, product],
        })
    }

    fn forward_eager(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Use fused SiLU-gate-up if available (reduces memory bandwidth)
        let mut gate_up = self.gate_up.forward(hidden_states)?.into_iter();
        let gate_proj_out = gate_up.next().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 gate/up projection omitted gate output".into())
        })?;
        let up_proj_out = gate_up.next().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 gate/up projection omitted up output".into())
        })?;

        let candidate = if self.fused_decode {
            let result = try_qwen38_silu_mul_decode(&gate_proj_out, &up_proj_out);
            record_cuda_kernel(CudaKernelPath::SiluMul, result.is_some());
            result
        } else {
            None
        };
        let fused = candidate.or_else(|| try_fused_silu_mul(&gate_proj_out, &up_proj_out));
        let hidden = if let Some(fused) = fused {
            fused
        } else {
            let gate = ops::silu(&gate_proj_out)?;
            (&gate * &up_proj_out)?
        };

        self.down.forward(&hidden).map_err(Error::from)
    }
}

impl Qwen38FullAttention {
    pub(super) fn load_native(
        loader: &Qwen38WeightLoader<'_>,
        prefix: &str,
        cfg: &Qwen38TextConfig,
    ) -> Result<Self> {
        let (tensors, device) = (loader.tensors, loader.device);
        let q_width = cfg
            .attention_head_count
            .checked_mul(cfg.attention_key_length)
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| Error::ModelLoadError("Qwen3.8 Q projection width overflow".into()))?;
        let kv_width = cfg
            .attention_head_count_kv
            .checked_mul(cfg.attention_key_length)
            .ok_or_else(|| Error::ModelLoadError("Qwen3.8 KV projection width overflow".into()))?;
        let q_name = format!("{prefix}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}.self_attn.k_proj.weight");
        let v_name = format!("{prefix}.self_attn.v_proj.weight");
        Ok(Self {
            qkv_proj: load_native_projection_group(
                loader,
                &[
                    (&q_name, [q_width, cfg.embedding_length]),
                    (&k_name, [kv_width, cfg.embedding_length]),
                    (&v_name, [kv_width, cfg.embedding_length]),
                ],
            )?,
            o_proj: load_native_projection(
                loader,
                &format!("{prefix}.self_attn.o_proj.weight"),
                [
                    cfg.embedding_length,
                    cfg.attention_head_count * cfg.attention_value_length,
                ],
            )?,
            q_norm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.self_attn.q_norm.weight"),
                cfg.attention_key_length,
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?,
            k_norm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.self_attn.k_norm.weight"),
                cfg.attention_key_length,
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?,
            num_heads: cfg.attention_head_count,
            num_kv_heads: cfg.attention_head_count_kv,
            head_dim: cfg.attention_key_length,
            rope_dim: cfg.rope_dimension_count.min(cfg.attention_key_length),
            rope_theta: cfg.rope_freq_base,
            mrope_sections: cfg
                .rope_dimension_sections
                .iter()
                .copied()
                .filter(|section| *section > 0)
                .take(3)
                .collect(),
            rope_kernel_enabled: qwen38_rope_kernel_enabled(device),
            rope_inv_freqs: build_rope_inv_freqs(
                cfg.rope_dimension_count.min(cfg.attention_key_length),
                cfg.rope_freq_base,
            )?,
            fused_qk: None,
            fused_qk_path: Qwen38FastPath::default(),
            fused_kv_write: false,
            fused_kv_write_path: Qwen38FastPath::default(),
            fused_gate: false,
            fused_gate_path: Qwen38FastPath::default(),
        })
    }

    /// Resolve the fused attention paths at load, each after a self-check on
    /// this layer's own head-norm gains and rotary configuration:
    /// - q/k norm + partial M-RoPE (`kernels/cuda/rope.rs`), against the
    ///   norm + plan + rotary chain at short, long and image positions;
    /// - the K/V write from that launch, into a scratch paged cache (it also
    ///   needs, per request, K/V stored in the activation dtype);
    /// - the attention gate (`kernels/cuda/gate.rs`), against the composition.
    pub(super) fn resolve_fused_attention(
        &mut self,
        switches: &Qwen38CudaSwitches,
        device: &Device,
        dtype: DType,
        allow_cpu: bool,
    ) {
        self.fused_qk = None;
        self.fused_qk_path = if let Some(path) =
            decode_gate(FUSED_QK_ROPE_ENV, switches, device, allow_cpu)
        {
            path
        } else if !self.fused_qk_fits(device, dtype) {
            Qwen38FastPath::legacy(
                "fused q/k norm + RoPE needs SM80+, 16-bit activations, F32 head-norm gains and rotary dims",
            )
        } else {
            let resolved = Tensor::from_vec(
                self.rope_inv_freqs.clone(),
                self.rope_inv_freqs.len(),
                device,
            )
            .map_err(Error::from)
            .and_then(|inv_freq| {
                let spec = QkNormRopeSpec {
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    head_dim: self.head_dim,
                    rope_dim: self.rope_dim,
                    eps: self.q_norm.eps as f32,
                };
                self.fused_qk_self_check(&spec, &inv_freq, device, dtype)?;
                Ok((spec, inv_freq))
            });
            match resolved {
                Ok(fused) => {
                    self.fused_qk = Some(fused);
                    Qwen38FastPath::Fused
                }
                Err(error) => Qwen38FastPath::self_check_failed("q/k norm + RoPE", &error),
            }
        };

        self.fused_kv_write = false;
        self.fused_kv_write_path =
            if let Some(path) = decode_gate(FUSED_KV_WRITE_ENV, switches, device, allow_cpu) {
                path
            } else if let Some((spec, inv_freq)) = &self.fused_qk {
                match self.fused_kv_write_self_check(spec, inv_freq, device, dtype) {
                    Ok(()) => {
                        self.fused_kv_write = true;
                        Qwen38FastPath::Fused
                    }
                    Err(error) => Qwen38FastPath::self_check_failed("K/V write", &error),
                }
            } else {
                Qwen38FastPath::legacy("the fused K/V write needs the fused q/k norm + RoPE")
            };

        self.fused_gate = false;
        self.fused_gate_path = if let Some(path) =
            decode_gate(FUSED_ATTN_GATE_ENV, switches, device, allow_cpu)
        {
            path
        } else if !matches!(dtype, DType::F16 | DType::BF16) || !gate::supported(device, dtype) {
            Qwen38FastPath::legacy("the fused attention gate needs SM80+ and 16-bit activations")
        } else {
            match self.fused_gate_self_check(device, dtype) {
                Ok(()) => {
                    self.fused_gate = true;
                    Qwen38FastPath::Fused
                }
                Err(error) => Qwen38FastPath::self_check_failed("attention gate", &error),
            }
        };
    }

    /// The fast-path resolutions `[q/k norm + RoPE, K/V write, gate]`.
    pub(super) fn fast_paths(&self) -> [&Qwen38FastPath; 3] {
        [
            &self.fused_qk_path,
            &self.fused_kv_write_path,
            &self.fused_gate_path,
        ]
    }

    fn fused_qk_fits(&self, device: &Device, dtype: DType) -> bool {
        matches!(dtype, DType::F16 | DType::BF16)
            && self.rope_dim > 0
            && self.rope_inv_freqs.len() == self.rope_dim / 2
            && rope::supported(device, dtype, self.head_dim, self.rope_dim)
            && self.q_norm.eps == self.k_norm.eps
            && [&self.q_norm, &self.k_norm].iter().all(|norm| {
                norm.weight.dtype() == DType::F32 && norm.weight.dims() == [self.head_dim]
            })
    }

    /// Random `dtype` values of `shape` on `device` for the self-checks.
    fn self_check_tensor(
        shape: &[usize],
        seed: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self_check_wave(shape.iter().product(), seed, 2.0),
            shape,
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?)
    }

    /// The fused q/k against today's chain (head norms, cos/sin plan from F32
    /// host angles, rotary op, pass-through concat) for one token at
    /// positions 5, 3000 and 200000, and at the image position [7, 3, 11]
    /// when the sections allow it.
    fn fused_qk_self_check(
        &self,
        spec: &QkNormRopeSpec,
        inv_freq: &Tensor,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let (heads, kv_heads, dim) = (self.num_heads, self.num_kv_heads, self.head_dim);
        for (index, position) in [[5, 5, 5], [3000, 3000, 3000], [200_000; 3], [7, 3, 11]]
            .into_iter()
            .enumerate()
        {
            let Some(mrope) = self.mrope_position(position) else {
                continue;
            };
            let seed = 10.0 * index as f32;
            let q_proj = Self::self_check_tensor(&[1, 1, heads, 2 * dim], seed, device, dtype)?;
            let k_proj =
                Self::self_check_tensor(&[1, 1, kv_heads, dim], seed + 1.0, device, dtype)?;
            let query = self
                .q_norm
                .forward(&q_proj.narrow(3, 0, dim)?.contiguous()?)?;
            let key = self.k_norm.forward(&k_proj)?;
            let plan = self.prepare_mrope_plan(&[position], device, dtype)?;
            let (q_ref, k_ref) = self.apply_rope_plan(&query, &key, &plan)?;
            let (q, k) = rope::qk_norm_rope(
                &q_proj,
                &k_proj,
                &self.q_norm.weight,
                &self.k_norm.weight,
                inv_freq,
                spec,
                mrope,
            )?;
            compare_values(
                &format!("fused q at {position:?}"),
                &host_values(&q)?,
                &host_values(&q_ref)?,
                0.01,
                0.03,
            )?;
            compare_values(
                &format!("fused k at {position:?}"),
                &host_values(&k)?,
                &host_values(&k_ref)?,
                0.01,
                0.03,
            )?;
        }
        Ok(())
    }

    /// The K/V-writing variant into a scratch `[2, 16, kv_heads, head_dim]`
    /// cache at flat slot 21, position 3000: the returned queries and the
    /// slot's key row match the eager kernel, the value row is the value, and
    /// every other slot stays zero.
    fn fused_kv_write_self_check(
        &self,
        spec: &QkNormRopeSpec,
        inv_freq: &Tensor,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        const PAGES: usize = 2;
        const PAGE_TOKENS: usize = 16;
        const SLOT: usize = 21;
        const POSITION: usize = 3000;
        let (heads, kv_heads, dim) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let q_proj = Self::self_check_tensor(&[1, 1, heads, 2 * dim], 1.0, device, dtype)?;
        let k_proj = Self::self_check_tensor(&[1, 1, kv_heads, dim], 2.0, device, dtype)?;
        let v_proj = Self::self_check_tensor(&[1, 1, kv_heads, dim], 3.0, device, dtype)?;
        let keys = Tensor::zeros((PAGES, PAGE_TOKENS, kv_heads, dim), dtype, device)?;
        let values = Tensor::zeros((PAGES, PAGE_TOKENS, kv_heads, dim), dtype, device)?;
        let step = Tensor::from_vec(vec![0u32, POSITION as u32, SLOT as u32], 3, device)?;
        let queries = rope::qk_norm_rope_kv(
            &q_proj,
            &k_proj,
            &v_proj,
            &self.q_norm.weight,
            &self.k_norm.weight,
            inv_freq,
            spec,
            &step,
            &keys,
            &values,
        )?;
        let (q_ref, k_ref) = rope::qk_norm_rope(
            &q_proj,
            &k_proj,
            &self.q_norm.weight,
            &self.k_norm.weight,
            inv_freq,
            spec,
            MropePosition {
                temporal: POSITION,
                height: POSITION,
                width: POSITION,
                height_section: 0,
                width_section: 0,
            },
        )?;
        compare_values(
            "fused K/V write queries",
            &host_values(&queries)?,
            &host_values(&q_ref)?,
            0.01,
            0.02,
        )?;
        let row = kv_heads * dim;
        let (keys, values) = (host_values(&keys)?, host_values(&values)?);
        for slot in 0..PAGES * PAGE_TOKENS {
            let (key, value) = (&keys[slot * row..][..row], &values[slot * row..][..row]);
            if slot == SLOT {
                compare_values(
                    "fused K/V write key",
                    key,
                    &host_values(&k_ref)?,
                    0.02,
                    0.02,
                )?;
                compare_values(
                    "fused K/V write value",
                    value,
                    &host_values(&v_proj)?,
                    0.02,
                    0.02,
                )?;
            } else if key.iter().chain(value).any(|x| *x != 0.0) {
                return Err(Error::InferenceError(format!(
                    "fused K/V write touched slot {slot} instead of only {SLOT}"
                )));
            }
        }
        Ok(())
    }

    /// The fused gate against the composition over two rows.
    fn fused_gate_self_check(&self, device: &Device, dtype: DType) -> Result<()> {
        let (heads, dim) = (self.num_heads, self.head_dim);
        let attn = Self::self_check_tensor(&[2, 1, heads * dim], 4.0, device, dtype)?;
        let q_proj = Self::self_check_tensor(&[2, 1, heads, 2 * dim], 5.0, device, dtype)?;
        compare_values(
            "fused attention gate",
            &host_values(&gate::attn_gate(&attn, &q_proj, heads, dim)?)?,
            &host_values(&gate::reference(&attn, &q_proj, heads, dim)?)?,
            0.01,
            0.02,
        )
    }

    pub(super) fn forward_physical(
        &self,
        hidden_states: &Tensor,
        rope: &Qwen38RopeInputs<'_>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 0 || seq_len != rope.positions.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 physical attention received {} tokens and {} rotary positions",
                seq_len,
                rope.positions.len()
            )));
        }
        let mut qkv = self.qkv_proj.forward(hidden_states)?.into_iter();
        let q_proj = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted Q".into()))?
            .reshape((1, seq_len, self.num_heads, self.head_dim * 2))?;
        let key_states = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted K".into()))?
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?;
        let value_states = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted V".into()))?
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?;
        let output_dtype = q_proj.dtype();
        let fused = match rope.positions {
            [position] => self.fused_qk_for(&q_proj, &key_states, *position),
            _ => None,
        };
        let output = if let Some((spec, inv_freq, position)) = fused {
            self.fused_qk_attend(
                spec,
                inv_freq,
                position,
                [&q_proj, &key_states, &value_states],
                rope.step.as_ref(),
                cache,
                prepared,
                physical_layer,
            )?
        } else {
            if seq_len == 1 && q_proj.device().is_cuda() {
                record_cuda_qk_rope_chain();
            }
            let query_states = q_proj.narrow(3, 0, self.head_dim)?;
            let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
            let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
            let (query_states, key_states) =
                self.apply_rope_plan(&query_states, &key_states, rope.plan(self)?)?;
            let queries = query_states
                .reshape((seq_len, self.num_heads, self.head_dim))?
                .contiguous()?;
            let keys = key_states
                .reshape((seq_len, self.num_kv_heads, self.head_dim))?
                .contiguous()?;
            let values = value_states
                .reshape((seq_len, self.num_kv_heads, self.head_dim))?
                .contiguous()?;
            self.write_and_attend(cache, prepared, physical_layer, &queries, &keys, &values)?
        };
        let output =
            output
                .to_dtype(output_dtype)?
                .reshape((1, seq_len, self.num_heads * self.head_dim))?;
        let output = self.gate_output(&output, &q_proj)?;
        self.o_proj.forward(&output).map_err(Error::from)
    }

    /// Cast token-major Q/K/V to the cache's storage dtype and write and
    /// attend through the arena.
    fn write_and_attend(
        &self,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: usize,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<Tensor> {
        let storage_dtype = cache.arena().config().dtype;
        if queries.device().is_cuda() && storage_dtype != queries.dtype() {
            // Q, K, V enter storage dtype and the attention result returns to
            // the activation dtype.
            record_cuda_attention_dtype_casts(4);
        }
        cache.write_and_attend(
            physical_layer,
            prepared,
            &queries.to_dtype(storage_dtype)?,
            &keys.to_dtype(storage_dtype)?,
            &values.to_dtype(storage_dtype)?,
            1.0 / (self.head_dim as f32).sqrt(),
        )
    }

    /// The fused q/k norm + RoPE for one token at `position`, when resolved
    /// and the projections fit it.
    fn fused_qk_for(
        &self,
        q_proj: &Tensor,
        key_states: &Tensor,
        position: [usize; 3],
    ) -> Option<(&QkNormRopeSpec, &Tensor, MropePosition)> {
        let (spec, inv_freq) = self.fused_qk.as_ref()?;
        let fits = matches!(q_proj.dtype(), DType::F16 | DType::BF16)
            && key_states.dtype() == q_proj.dtype();
        if !fits {
            return None;
        }
        Some((spec, inv_freq, self.mrope_position(position)?))
    }

    /// The kernel's view of one token's M-RoPE position: text positions need
    /// no sections; an image position takes the height and width sections,
    /// and the composition refuses it too when they are not valid.
    fn mrope_position(&self, [temporal, height, width]: [usize; 3]) -> Option<MropePosition> {
        let text = temporal == height && temporal == width;
        let sections = match self.mrope_sections.as_slice() {
            [_, height, width]
                if self.mrope_sections.iter().sum::<usize>() == self.rope_dim / 2 =>
            {
                Some((*height, *width))
            }
            _ => None,
        };
        let (height_section, width_section) = if text { (0, 0) } else { sections? };
        Some(MropePosition {
            temporal,
            height,
            width,
            height_section,
            width_section,
        })
    }

    /// One token through the fused q/k norm + partial M-RoPE. With the fused
    /// K/V write resolved, a text position, a step buffer for this forward and
    /// K/V stored in the activation dtype, the same launch writes the rotated
    /// key and the value into the paged cache; otherwise the arena writes the
    /// kernel's K/V. Returns the attention output `[1, heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn fused_qk_attend(
        &self,
        spec: &QkNormRopeSpec,
        inv_freq: &Tensor,
        position: MropePosition,
        [q_proj, key_states, value_states]: [&Tensor; 3],
        step: Option<&Tensor>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let text = position.temporal == position.height && position.temporal == position.width;
        if let Some(step) = step.filter(|_| {
            self.fused_kv_write && text && cache.arena().config().dtype == q_proj.dtype()
        }) {
            let output = cache.write_fused_and_attend(
                physical_layer,
                prepared,
                1.0 / (self.head_dim as f32).sqrt(),
                &mut |keys, values| {
                    let queries = rope::qk_norm_rope_kv(
                        q_proj,
                        key_states,
                        value_states,
                        &self.q_norm.weight,
                        &self.k_norm.weight,
                        inv_freq,
                        spec,
                        step,
                        keys,
                        values,
                    )?;
                    record_fused_qk_norm_rope(true);
                    Ok(queries.reshape((1, self.num_heads, self.head_dim))?)
                },
            )?;
            if let Some(output) = output {
                return Ok(output);
            }
        }
        let (queries, keys) = rope::qk_norm_rope(
            q_proj,
            key_states,
            &self.q_norm.weight,
            &self.k_norm.weight,
            inv_freq,
            spec,
            position,
        )?;
        record_fused_qk_norm_rope(false);
        self.write_and_attend(
            cache,
            prepared,
            physical_layer,
            &queries.reshape((1, self.num_heads, self.head_dim))?,
            &keys.reshape((1, self.num_kv_heads, self.head_dim))?,
            &value_states
                .reshape((1, self.num_kv_heads, self.head_dim))?
                .contiguous()?,
        )
    }

    /// `attn * sigmoid(gate)`, the gate being the second half of each head's
    /// slice of `q_proj` `[.., heads, 2 * head_dim]`: one launch when fused.
    fn gate_output(&self, output: &Tensor, q_proj: &Tensor) -> Result<Tensor> {
        if self.fused_gate
            && output.dtype() == q_proj.dtype()
            && matches!(output.dtype(), DType::F16 | DType::BF16)
        {
            record_fused_attn_gate();
            return Ok(gate::attn_gate(
                output,
                q_proj,
                self.num_heads,
                self.head_dim,
            )?);
        }
        if output.device().is_cuda() {
            record_cuda_attn_gate_composition();
        }
        let gate = q_proj
            .narrow(3, self.head_dim, self.head_dim)?
            .reshape(output.shape())?;
        (output * ops::sigmoid(&gate)?).map_err(Error::from)
    }

    pub(super) fn forward_physical_decode_batch(
        &self,
        hidden_states: &Tensor,
        position_ids: &[[usize; 3]],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: usize,
    ) -> Result<Tensor> {
        if hidden_states.dim(1)? != 1 {
            return Err(Error::InvalidInput(
                "Qwen3.8 full-attention decode batch dimensions do not match".into(),
            ));
        }
        let (q_proj, key_states, value_states) = self.decode_projections(hidden_states)?;
        let output = self.decode_attend(
            &q_proj,
            &key_states,
            &value_states,
            position_ids,
            caches,
            slots,
            metadata,
            completions,
            physical_layer,
        )?;
        self.decode_output(&output, &q_proj)
    }

    /// The q/k/v projections of decode rows `[B, 1, hidden]`: `q_proj`
    /// `[B, 1, heads, 2 * head_dim]` (query and gate halves of each head),
    /// keys and values `[B, 1, kv_heads, head_dim]`.
    pub(super) fn decode_projections(
        &self,
        hidden_states: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let batch_size = hidden_states.dim(0)?;
        let mut qkv = self.qkv_proj.forward(hidden_states)?.into_iter();
        let q_proj = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted Q".into()))?
            .reshape((batch_size, 1, self.num_heads, self.head_dim * 2))?;
        let key_states = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted K".into()))?
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        let value_states = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted V".into()))?
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        Ok((q_proj, key_states, value_states))
    }

    /// Attention for one decode token per row from its projections: q/k
    /// norm and RoPE at each row's position, the K/V write through the arena
    /// and the paged decode. Returns the attention output `[B, 1, heads *
    /// head_dim]` in the activation dtype, before the output gate.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_attend(
        &self,
        q_proj: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        position_ids: &[[usize; 3]],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let batch_size = q_proj.dim(0)?;
        if batch_size == 0 || position_ids.len() != batch_size || caches.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.8 full-attention decode batch dimensions do not match".into(),
            ));
        }
        let first = caches[0];
        if caches.iter().any(|cache| {
            !Arc::ptr_eq(cache.arena(), first.arena())
                || cache.layer_binding(physical_layer).ok()
                    != first.layer_binding(physical_layer).ok()
        }) {
            return Err(Error::InvalidInput(
                "Qwen3.8 decode rows must share one arena and sparse layer binding".into(),
            ));
        }
        if slots.arena_id() != first.arena().id() || slots.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.8 decode received an incompatible prepared slot map".into(),
            ));
        }

        let mut queries = Vec::with_capacity(batch_size);
        let mut keys = Vec::with_capacity(batch_size);
        let mut values = Vec::with_capacity(batch_size);
        let fused = position_ids
            .iter()
            .map(|position| self.fused_qk_for(q_proj, key_states, *position))
            .collect::<Option<Vec<_>>>();
        if let Some(rows) = fused {
            // One launch per row with host positions; the arena writes K/V.
            for (row, (spec, inv_freq, position)) in rows.into_iter().enumerate() {
                let (q_row, k_row) = rope::qk_norm_rope(
                    &q_proj.i(row)?,
                    &key_states.i(row)?,
                    &self.q_norm.weight,
                    &self.k_norm.weight,
                    inv_freq,
                    spec,
                    position,
                )?;
                record_fused_qk_norm_rope(false);
                queries.push(q_row.reshape((self.num_heads, self.head_dim))?);
                keys.push(k_row.reshape((self.num_kv_heads, self.head_dim))?);
                values.push(
                    value_states
                        .i(row)?
                        .reshape((self.num_kv_heads, self.head_dim))?,
                );
            }
        } else {
            if q_proj.device().is_cuda() {
                for _ in 0..batch_size {
                    record_cuda_qk_rope_chain();
                }
            }
            let query_states = q_proj.narrow(3, 0, self.head_dim)?;
            let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
            let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
            for row in 0..batch_size {
                let plan =
                    self.prepare_mrope_plan(&[position_ids[row]], q_proj.device(), q_proj.dtype())?;
                let q_row = query_states.i(row)?.unsqueeze(0)?;
                let k_row = key_states.i(row)?.unsqueeze(0)?;
                let (q_row, k_row) = self.apply_rope_plan(&q_row, &k_row, &plan)?;
                queries.push(q_row.reshape((self.num_heads, self.head_dim))?);
                keys.push(k_row.reshape((self.num_kv_heads, self.head_dim))?);
                values.push(
                    value_states
                        .i(row)?
                        .reshape((self.num_kv_heads, self.head_dim))?,
                );
            }
        }
        let query_refs = queries.iter().collect::<Vec<_>>();
        let key_refs = keys.iter().collect::<Vec<_>>();
        let value_refs = values.iter().collect::<Vec<_>>();
        let queries = Tensor::stack(&query_refs, 0)?.contiguous()?;
        let keys = Tensor::stack(&key_refs, 0)?.contiguous()?;
        let values = Tensor::stack(&value_refs, 0)?.contiguous()?;
        let storage_dtype = first.arena().config().dtype;
        let output_dtype = queries.dtype();
        if queries.device().is_cuda() && storage_dtype != output_dtype {
            record_cuda_attention_dtype_casts(4);
        }
        let queries = queries.to_dtype(storage_dtype)?;
        let keys = keys.to_dtype(storage_dtype)?;
        let values = values.to_dtype(storage_dtype)?;
        let binding = first.layer_binding(physical_layer)?;
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
                    batch: metadata,
                    softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                    softcap: None,
                },
            )
        })?;
        completions.collect(completion)?;
        let output = output.to_dtype(output_dtype)?.reshape((
            batch_size,
            1,
            self.num_heads * self.head_dim,
        ))?;
        Ok(output)
    }

    /// The attention tail: the output gate, then `o_proj`.
    pub(super) fn decode_output(&self, attended: &Tensor, q_proj: &Tensor) -> Result<Tensor> {
        let output = self.gate_output(attended, q_proj)?;
        self.o_proj.forward(&output).map_err(Error::from)
    }

    pub(super) fn prepare_mrope_plan(
        &self,
        position_ids: &[[usize; 3]],
        device: &Device,
        dtype: DType,
    ) -> Result<Qwen38MropePlan> {
        build_mrope_plan(
            self.rope_dim,
            position_ids,
            &self.mrope_sections,
            &self.rope_inv_freqs,
            device,
            dtype,
        )
    }

    pub(super) fn cache_geometry(&self) -> (usize, usize, usize) {
        (self.num_kv_heads, self.head_dim, self.head_dim)
    }

    pub(super) fn q_norm(&self) -> &Qwen38RmsNorm {
        &self.q_norm
    }

    pub(super) fn set_fused_norms(&mut self, fused: bool) {
        self.q_norm.set_fused(fused);
        self.k_norm.set_fused(fused);
    }

    fn apply_rope_plan(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        plan: &Qwen38MropePlan,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = query_states.dim(1)?;
        if seq_len != plan.sequence_len || self.rope_dim != plan.rope_dim {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 rotary plan mismatch: attention has seq_len={} and rope_dim={}, plan has seq_len={} and rope_dim={}",
                seq_len, self.rope_dim, plan.sequence_len, plan.rope_dim
            )));
        }
        if self.rope_dim == 0 {
            return Ok((query_states.clone(), key_states.clone()));
        }
        let cos = plan.cos.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 rotary plan omitted cosine values".into())
        })?;
        let sin = plan.sin.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 rotary plan omitted sine values".into())
        })?;

        let query_rot = query_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let key_rot = key_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let (query_rot, key_rot) = if self.should_try_rope_kernel(query_states.dtype()) {
            match try_apply_rope_thd(&query_rot, &key_rot, cos, sin)? {
                Some((query_rot, key_rot)) => {
                    for _ in 0..plan.sequence_len {
                        record_rope_kernel();
                        if query_states.device().is_cuda() {
                            record_cuda_rope(true);
                        }
                    }
                    (query_rot, key_rot)
                }
                None => {
                    for _ in 0..plan.sequence_len {
                        record_rope_manual();
                        if query_states.device().is_cuda() {
                            record_cuda_rope(false);
                        }
                    }
                    (
                        apply_rotary_emb(&query_rot, cos, sin)?,
                        apply_rotary_emb(&key_rot, cos, sin)?,
                    )
                }
            }
        } else {
            for _ in 0..plan.sequence_len {
                record_rope_manual();
                if query_states.device().is_cuda() {
                    record_cuda_rope(false);
                }
            }
            (
                apply_rotary_emb(&query_rot, cos, sin)?,
                apply_rotary_emb(&key_rot, cos, sin)?,
            )
        };

        if self.rope_dim == self.head_dim {
            return Ok((query_rot, key_rot));
        }

        let query_pass = query_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        let key_pass = key_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        Ok((
            Tensor::cat(&[&query_rot, &query_pass], 3)?,
            Tensor::cat(&[&key_rot, &key_pass], 3)?,
        ))
    }

    fn should_try_rope_kernel(&self, dtype: DType) -> bool {
        if !self.rope_kernel_enabled {
            return false;
        }
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) {
            return false;
        }
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
    }
}

impl Qwen38LinearAttention {
    fn load_native(
        loader: &Qwen38WeightLoader<'_>,
        prefix: &str,
        cfg: &Qwen38TextConfig,
        performance: &CudaPerformanceConfig,
    ) -> Result<Self> {
        let (tensors, device) = (loader.tensors, loader.device);
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let head_k_dim = cfg.ssm_state_size;
        let head_v_dim = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
        let linear = format!("{prefix}.linear_attn");
        let dt_bias = tensors
            .materialize_dense_tensor(
                &format!("{linear}.dt_bias"),
                &[num_v_heads],
                ProjectionMaterialization::F32,
                device,
            )?
            .reshape((1, 1, num_v_heads))?;
        let a_log = tensors.materialize_dense_tensor(
            &format!("{linear}.A_log"),
            &[num_v_heads],
            ProjectionMaterialization::F32,
            device,
        )?;
        let a = a_log.exp()?.neg()?.reshape((1, 1, num_v_heads))?;
        let conv_kernel = normalize_conv_kernel(
            tensors.materialize_dense_tensor(
                &format!("{linear}.conv1d.weight"),
                &[conv_dim, 1, cfg.ssm_conv_kernel],
                ProjectionMaterialization::F32,
                device,
            )?,
            conv_dim,
            cfg.ssm_conv_kernel,
        )?;
        let conv_kernel_slices = pre_slice_conv_kernel(&conv_kernel, cfg.ssm_conv_kernel)?;
        let norm = Qwen38GatedRmsNorm {
            fused_decode: device.is_cuda()
                && performance.enabled()
                && performance.fused_decode.enabled(),
            weight: tensors.materialize_dense_tensor(
                &format!("{linear}.norm.weight"),
                &[head_v_dim],
                ProjectionMaterialization::F32,
                device,
            )?,
            eps: cfg.attention_layer_norm_rms_epsilon,
        };
        let qkv_name = format!("{linear}.in_proj_qkv.weight");
        let z_name = format!("{linear}.in_proj_z.weight");
        let alpha_name = format!("{linear}.in_proj_a.weight");
        let beta_name = format!("{linear}.in_proj_b.weight");
        Ok(Self {
            fused_decode: device.is_cuda()
                && performance.enabled()
                && performance.fused_decode.enabled(),
            qkv_z_proj: load_native_projection_group(
                loader,
                &[
                    (&qkv_name, [conv_dim, cfg.embedding_length]),
                    (&z_name, [cfg.ssm_inner_size, cfg.embedding_length]),
                ],
            )?,
            alpha_beta_proj: load_native_projection_group(
                loader,
                &[
                    (&alpha_name, [num_v_heads, cfg.embedding_length]),
                    (&beta_name, [num_v_heads, cfg.embedding_length]),
                ],
            )?,
            dt_bias,
            a,
            conv_kernel,
            conv_kernel_slices,
            norm,
            out_proj: load_native_projection(
                loader,
                &format!("{linear}.out_proj.weight"),
                [cfg.embedding_length, cfg.ssm_inner_size],
            )?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            kernel_size: cfg.ssm_conv_kernel,
            tiled_recurrence_enabled: qwen38_tiled_recurrence_enabled(),
            tiled_recurrence_tile_size_override: qwen38_tiled_recurrence_tile_size_override(),
            fused_gdn: None,
            fused_gdn_path: Qwen38FastPath::default(),
            alpha_beta_rows: None,
            alpha_beta_gemv_path: Qwen38FastPath::default(),
            gdn_prefill: false,
            gdn_prefill_path: Qwen38FastPath::default(),
        })
    }

    /// Resolve the fused single-token decode at load. It needs 128-dim heads,
    /// a 4-tap conv and value heads grouped over key heads (HF order), and
    /// turns on only after a two-step self-check against today's decode on
    /// this layer's own conv kernel, `dt_bias`, `A` and norm gain.
    /// `IZWI_QWEN38_FUSED_GDN=legacy` keeps today's decode.
    fn resolve_fused_gdn(
        &mut self,
        switches: &Qwen38CudaSwitches,
        device: &Device,
        dtype: DType,
        allow_cpu: bool,
    ) {
        self.fused_gdn = None;
        self.fused_gdn_path = if let Some(path) =
            decode_gate(FUSED_GDN_ENV, switches, device, allow_cpu)
        {
            path
        } else if let Some(spec) = self.gdn_spec(device) {
            match self.fused_gdn_self_check(&spec, device, dtype) {
                Ok(()) => {
                    self.fused_gdn = Some(spec);
                    Qwen38FastPath::Fused
                }
                Err(error) => Qwen38FastPath::self_check_failed("DeltaNet decode", &error),
            }
        } else {
            Qwen38FastPath::legacy(
                    "fused DeltaNet decode needs SM80+, 128-dim heads, a 4-tap conv and value heads grouped over key heads",
                )
        };
    }

    /// The fused decode's view of this layer, when the generic kernel serves
    /// its geometry on `device`.
    fn gdn_spec(&self, device: &Device) -> Option<GdnDecodeSpec> {
        let spec = GdnDecodeSpec {
            key_heads: self.num_k_heads,
            value_heads: self.num_v_heads,
            grouped: true,
            norm_eps: self.norm.eps as f32,
        };
        let fits = gdn::supported(device, self.head_k_dim, self.head_v_dim, self.kernel_size)
            && self.num_k_heads > 0
            && self.num_v_heads.is_multiple_of(self.num_k_heads)
            && spec.conv_dim() == self.conv_dim
            && self.conv_kernel.dims() == [self.conv_dim, gdn::CONV_TAPS]
            && self.conv_kernel.dtype() == DType::F32
            && self.norm.weight.dims() == [self.head_v_dim]
            && self.norm.weight.dtype() == DType::F32;
        fits.then_some(spec)
    }

    fn fused_gdn_self_check(
        &self,
        spec: &GdnDecodeSpec,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        self.fused_gdn_self_check_with(device, dtype, |row, conv_state, recurrent_state| {
            self.fused_decode_row(
                spec,
                &row.mixed_qkv,
                &row.z,
                &row.beta,
                &row.alpha,
                conv_state,
                recurrent_state,
            )
        })
    }

    /// Two consecutive synthetic decode steps through today's decode and
    /// through `fused`, from identical states (a rotated portable ring and a
    /// random F32 recurrent state): the outputs, the recurrent states and the
    /// conv histories must agree after each step.
    fn fused_gdn_self_check_with(
        &self,
        device: &Device,
        dtype: DType,
        fused: impl Fn(
            &GdnSelfCheckRow,
            &mut Option<ConvRingState>,
            &mut Option<Tensor>,
        ) -> Result<Tensor>,
    ) -> Result<()> {
        let value_width = self.num_v_heads * self.head_v_dim;
        let row = |len: usize, seed: f32, scale: f32| -> Result<Tensor> {
            Ok(
                Tensor::from_vec(self_check_wave(len, seed, scale), (1, 1, len), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(device)?,
            )
        };
        let slot = |seed: f32| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                self_check_wave(self.conv_dim, seed, 1.0),
                (self.conv_dim, 1),
                &Device::Cpu,
            )?
            .to_device(device)?)
        };
        let state_len = self.num_v_heads * self.head_k_dim * self.head_v_dim;
        let conv_state = Some(ConvRingState {
            storage: ConvHistoryStorage::Ring {
                slots: vec![slot(1.0)?, slot(2.0)?, slot(3.0)?],
                next_idx: 1,
            },
        });
        let recurrent_state = Some(
            Tensor::from_vec(
                self_check_wave(state_len, 4.0, 0.3),
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                &Device::Cpu,
            )?
            .to_device(device)?,
        );
        let (mut legacy_conv, mut legacy_recurrent) = (conv_state.clone(), recurrent_state.clone());
        let (mut fused_conv, mut fused_recurrent) = (conv_state, recurrent_state);
        for step in 0..2 {
            let seed = 10.0 * (step as f32 + 1.0);
            let inputs = GdnSelfCheckRow {
                mixed_qkv: row(self.conv_dim, seed, 1.0)?,
                z: row(value_width, seed + 1.0, 1.5)?,
                beta: row(self.num_v_heads, seed + 2.0, 2.0)?,
                alpha: row(self.num_v_heads, seed + 3.0, 2.0)?,
            };
            let expected = self.legacy_decode_row(
                &inputs.mixed_qkv,
                &inputs.z,
                &inputs.alpha,
                &inputs.beta,
                &mut legacy_conv,
                &mut legacy_recurrent,
                dtype,
            )?;
            let actual = fused(&inputs, &mut fused_conv, &mut fused_recurrent)?;
            compare_values(
                &format!("DeltaNet decode output, step {step}"),
                &host_values(&actual)?,
                &host_values(&expected)?,
                0.01,
                0.03,
            )?;
            let state = |state: &Option<Tensor>| -> Result<Vec<f32>> {
                host_values(state.as_ref().ok_or_else(|| {
                    Error::InferenceError("DeltaNet self-check lost its recurrent state".into())
                })?)
            };
            compare_values(
                &format!("DeltaNet recurrent state, step {step}"),
                &state(&fused_recurrent)?,
                &state(&legacy_recurrent)?,
                1e-4,
                1e-3,
            )?;
            let history = |ring: &Option<ConvRingState>| -> Result<Vec<f32>> {
                host_values(
                    &ring
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InferenceError(
                                "DeltaNet self-check lost its conv history".into(),
                            )
                        })?
                        .contiguous_history()?,
                )
            };
            compare_values(
                &format!("DeltaNet conv history, step {step}"),
                &history(&fused_conv)?,
                &history(&legacy_conv)?,
                1e-6,
                1e-6,
            )?;
        }
        Ok(())
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen38LayerRuntimeState,
    ) -> Result<Tensor> {
        let (conv_state, recurrent_state) = linear_state(state)?;
        let residual_dtype = hidden_states.dtype();
        let (mixed_qkv, z, alpha, beta) = self.decode_projections(hidden_states)?;
        let value_width = self.num_v_heads * self.head_v_dim;
        let output = match self.fused_gdn_for(
            &mixed_qkv,
            &z,
            &alpha,
            &beta,
            [(&*conv_state, &*recurrent_state)],
        ) {
            Some(spec) => self
                .fused_decode_row(
                    spec,
                    &mixed_qkv,
                    &z,
                    &beta,
                    &alpha,
                    conv_state,
                    recurrent_state,
                )?
                .reshape((1, 1, value_width))?
                .to_dtype(residual_dtype)?,
            None => self.legacy_decode_row(
                &mixed_qkv,
                &z,
                &alpha,
                &beta,
                conv_state,
                recurrent_state,
                residual_dtype,
            )?,
        };
        self.out_proj.forward(&output).map_err(Error::from)
    }

    /// The decode projections in their own dtype: `(mixed_qkv, z, alpha,
    /// beta)`, the last two pre-activation. With the alpha/beta GEMV resolved,
    /// a 16-bit row takes alpha and beta from one router-GEMV launch, rounded
    /// through the activation dtype as the dense projection's output is.
    fn decode_projections(
        &self,
        hidden_states: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut qkv_z = self.qkv_z_proj.forward(hidden_states)?.into_iter();
        let mixed_qkv = required_group_output(&mut qkv_z, "DeltaNet QKV")?;
        let z = required_group_output(&mut qkv_z, "DeltaNet Z")?;
        if let Some(rows) = self
            .alpha_beta_rows
            .as_ref()
            .filter(|_| matches!(hidden_states.dtype(), DType::F16 | DType::BF16))
        {
            let (alpha, beta) = self.alpha_beta_gemv(hidden_states, rows)?;
            return Ok((mixed_qkv, z, alpha, beta));
        }
        let mut alpha_beta = self.alpha_beta_proj.forward(hidden_states)?.into_iter();
        let alpha = required_group_output(&mut alpha_beta, "DeltaNet alpha")?;
        let beta = required_group_output(&mut alpha_beta, "DeltaNet beta")?;
        Ok((mixed_qkv, z, alpha, beta))
    }

    /// `(alpha, beta)` `[batch, rows, value_heads]` in the input's dtype from
    /// the F32 `rows` through the router GEMV.
    fn alpha_beta_gemv(&self, hidden_states: &Tensor, rows: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, sequence, hidden) = hidden_states.dims3()?;
        let heads = self.num_v_heads;
        let packed = crate::kernels::cuda::moe::router_logits(
            &hidden_states.reshape((batch * sequence, hidden))?,
            rows,
        )?
        .to_dtype(hidden_states.dtype())?
        .reshape((batch, sequence, 2 * heads))?;
        Ok((packed.narrow(2, 0, heads)?, packed.narrow(2, heads, heads)?))
    }

    /// Resolve the alpha/beta GEMV at load: the dense `in_proj_a`/`in_proj_b`
    /// weights become F32 rows, kept only when a two-row probe through the
    /// router GEMV matches the dense projection on this device. The rows cost
    /// an F32 copy of the two small matrices (about 94 MB over 48 layers of
    /// the 27B checkpoint). `IZWI_QWEN38_FUSED_AB_GEMV=legacy` keeps the dense
    /// projection.
    fn resolve_alpha_beta_gemv(
        &mut self,
        switches: &Qwen38CudaSwitches,
        device: &Device,
        dtype: DType,
        allow_cpu: bool,
    ) {
        self.alpha_beta_rows = None;
        self.alpha_beta_gemv_path = if let Some(path) =
            decode_gate(FUSED_AB_GEMV_ENV, switches, device, allow_cpu)
        {
            path
        } else if !matches!(dtype, DType::F16 | DType::BF16) {
            Qwen38FastPath::legacy("the alpha/beta GEMV needs 16-bit activations")
        } else if let Some(rows) = self.alpha_beta_dense_rows() {
            match self.alpha_beta_gemv_self_check(&rows, device, dtype) {
                Ok(()) => {
                    self.alpha_beta_rows = Some(rows);
                    Qwen38FastPath::Fused
                }
                Err(error) => Qwen38FastPath::self_check_failed("alpha/beta GEMV", &error),
            }
        } else {
            Qwen38FastPath::legacy(
                "the alpha/beta GEMV needs dense in_proj_a/in_proj_b weights and a hidden width divisible by 4",
            )
        };
    }

    /// The dense alpha/beta weights as contiguous F32 rows `[2 * value_heads,
    /// hidden]`, alpha first, when both projections are dense.
    fn alpha_beta_dense_rows(&self) -> Option<Tensor> {
        let dense = |projection: &Qwen38Projection| match projection {
            Qwen38Projection::Dense(linear) => Some(linear.weight().clone()),
            _ => None,
        };
        let weights = match &self.alpha_beta_proj {
            Qwen38ProjectionGroup::Packed { projection, .. } => vec![dense(projection)?],
            Qwen38ProjectionGroup::Separate(parts) => {
                parts.iter().map(dense).collect::<Option<Vec<_>>>()?
            }
        };
        let rows = Tensor::cat(&weights, 0)
            .and_then(|rows| rows.to_dtype(DType::F32))
            .and_then(|rows| rows.contiguous())
            .ok()?;
        let (count, hidden) = rows.dims2().ok()?;
        (count == 2 * self.num_v_heads && hidden > 0 && hidden.is_multiple_of(4)).then_some(rows)
    }

    /// The router GEMV over `rows` against the dense alpha/beta projection
    /// for two `dtype` rows on `device`.
    fn alpha_beta_gemv_self_check(
        &self,
        rows: &Tensor,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let hidden = rows.dim(1)?;
        let probe = Tensor::from_vec(
            self_check_wave(2 * hidden, 13.0, 1.5),
            (2, 1, hidden),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?;
        let mut expected = self.alpha_beta_proj.forward(&probe)?.into_iter();
        let expected_alpha = required_group_output(&mut expected, "DeltaNet alpha")?;
        let expected_beta = required_group_output(&mut expected, "DeltaNet beta")?;
        let (alpha, beta) = self.alpha_beta_gemv(&probe, rows)?;
        compare_values(
            "DeltaNet alpha GEMV",
            &host_values(&alpha)?,
            &host_values(&expected_alpha)?,
            0.01,
            0.02,
        )?;
        compare_values(
            "DeltaNet beta GEMV",
            &host_values(&beta)?,
            &host_values(&expected_beta)?,
            0.01,
            0.02,
        )
    }

    /// Today's single-token decode from one row of projection outputs: F32
    /// casts, the Candle discretization, q38's conv and recurrence kernels (or
    /// the portable chain) and the gated norm. Returns the out-projection
    /// input `[1, 1, value_heads * head_v_dim]` in `out_dtype`.
    #[allow(clippy::too_many_arguments)]
    fn legacy_decode_row(
        &self,
        mixed_qkv: &Tensor,
        z: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        conv_state: &mut Option<ConvRingState>,
        recurrent_state: &mut Option<Tensor>,
        out_dtype: DType,
    ) -> Result<Tensor> {
        let mixed_qkv = mixed_qkv.to_dtype(DType::F32)?;
        let z = z.to_dtype(DType::F32)?;
        let alpha = alpha.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(&beta.to_dtype(DType::F32)?)?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        let mixed_qkv = self.depthwise_conv_step(&mixed_qkv, conv_state)?;

        if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
            return Err(Error::InferenceError(format!(
                "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }
        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            if mixed_qkv.device().is_cuda() {
                record_cuda_state_initial_allocation();
            }
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?
        };

        let beta = beta.reshape((1, self.num_v_heads))?;
        let g = g.reshape((1, self.num_v_heads))?;
        let specialized = if self.fused_decode {
            let result = try_qwen38_deltanet_decode(
                &mixed_qkv,
                &g,
                &beta,
                &current_state,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            );
            record_cuda_kernel(CudaKernelPath::DeltaNetSpecializedDecode, result.is_some());
            result
        } else {
            None
        };
        let (output, next_state) = if let Some(result) = specialized {
            result
        } else {
            let query = mixed_qkv.narrow(2, 0, key_width)?.reshape((
                1,
                self.num_k_heads,
                self.head_k_dim,
            ))?;
            let key = mixed_qkv.narrow(2, key_width, key_width)?.reshape((
                1,
                self.num_k_heads,
                self.head_k_dim,
            ))?;
            let value = mixed_qkv.narrow(2, key_width * 2, value_width)?.reshape((
                1,
                self.num_v_heads,
                self.head_v_dim,
            ))?;
            let mut query = l2norm(&query, 1e-6, self.fused_decode)?;
            let mut key = l2norm(&key, 1e-6, self.fused_decode)?;
            if self.num_v_heads != self.num_k_heads {
                let repeats = self.num_v_heads / self.num_k_heads;
                if query.device().is_cuda() {
                    record_cuda_head_expansion_materialization();
                    record_cuda_head_expansion_materialization();
                }
                query = repeat_interleave_head_states(&query, repeats)?;
                key = repeat_interleave_head_states(&key, repeats)?;
            }
            recurrent_gated_delta(&query, &key, &value, &g, &beta, current_state)?
        };
        // The recurrent decode output owns a fresh Candle-managed allocation;
        // retaining it avoids a full state-sized copy on every layer/token.
        *recurrent_state = Some(next_state);

        let output = output.reshape((self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z, true)?;
        output
            .reshape((1, 1, self.num_v_heads * self.head_v_dim))?
            .to_dtype(out_dtype)
            .map_err(Error::from)
    }

    /// The resolved fused decode, when every row's projections and state fit
    /// it: 16-bit or F32 projections with beta and alpha in z's dtype, a 3-slot
    /// F32 conv history and an F32 (or absent) recurrent state. A resolved path
    /// that cannot take the rows counts a fallback per row.
    fn fused_gdn_for<'a, 'b>(
        &'a self,
        mixed_qkv: &Tensor,
        z: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        rows: impl IntoIterator<Item = (&'b Option<ConvRingState>, &'b Option<Tensor>)>,
    ) -> Option<&'a GdnDecodeSpec> {
        let spec = self.fused_gdn.as_ref()?;
        let floating =
            |tensor: &Tensor| matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16);
        let projections = floating(mixed_qkv)
            && floating(z)
            && alpha.dtype() == z.dtype()
            && beta.dtype() == z.dtype();
        let mut count = 0usize;
        let mut states = true;
        for (conv_state, recurrent_state) in rows {
            count += 1;
            states &= conv_state.as_ref().is_some_and(|ring| {
                ring.history_len().ok() == Some(gdn::CONV_TAPS - 1) && ring.is_f32()
            }) && recurrent_state
                .as_ref()
                .is_none_or(|state| state.dtype() == DType::F32);
        }
        let accepted = projections && states;
        for _ in 0..count {
            record_cuda_kernel(CudaKernelPath::FusedGdnDecode, accepted);
        }
        accepted.then_some(spec)
    }

    /// One token's causal conv, recurrence and gated norm through two
    /// launches: q38's own conv kernel (channel-major `[C, 3]` history, input
    /// in the projection dtype) and the generic `gdn::recurrent_decode`.
    /// `beta_raw`/`alpha` are the pre-activation `in_proj_b`/`in_proj_a`
    /// outputs. The previous history and state are read-only and the kernels
    /// write fresh outputs, which are published only after both succeed: a
    /// failed step leaves the layer state untouched, and snapshots sharing
    /// the previous tensors stay valid. Returns `[value_heads * head_v_dim]`
    /// in z's dtype.
    #[allow(clippy::too_many_arguments)]
    fn fused_decode_row(
        &self,
        spec: &GdnDecodeSpec,
        mixed_qkv: &Tensor,
        z: &Tensor,
        beta_raw: &Tensor,
        alpha: &Tensor,
        conv_state: &mut Option<ConvRingState>,
        recurrent_state: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let ring = conv_state.as_mut().ok_or_else(|| {
            Error::InferenceError(
                "Qwen3.8 fused DeltaNet decode needs an initialized convolution history".into(),
            )
        })?;
        let history = ring.contiguous_history()?;
        let (conv, next_history) = qwen38_conv_decode(mixed_qkv, &self.conv_kernel, &history)?;
        let zeros;
        let previous = match recurrent_state.as_ref() {
            Some(previous) => previous,
            None => {
                if conv.device().is_cuda() {
                    record_cuda_state_initial_allocation();
                }
                zeros = Tensor::zeros(
                    (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    DType::F32,
                    conv.device(),
                )?;
                &zeros
            }
        };
        let (y, next) = gdn::recurrent_decode(
            &conv,
            z,
            beta_raw,
            alpha,
            &self.dt_bias,
            &self.a,
            &self.norm.weight,
            previous,
            spec,
        )?;
        ring.replace_contiguous(next_history, gdn::CONV_TAPS - 1)?;
        *recurrent_state = Some(next);
        Ok(y)
    }

    fn forward_decode_batch(
        &self,
        hidden_states: &Tensor,
        states: &mut [&mut Qwen38LayerRuntimeState],
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        if hidden_states.dim(1)? != 1 || batch_size == 0 || states.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.8 DeltaNet decode batch dimensions do not match".into(),
            ));
        }
        let residual_dtype = hidden_states.dtype();
        let (mixed_qkv, z, alpha, beta) = self.decode_projections(hidden_states)?;
        let value_width = self.num_v_heads * self.head_v_dim;
        let rows = states
            .iter()
            .map(|state| match &**state {
                Qwen38LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => Ok((conv_state, recurrent_state)),
                Qwen38LayerRuntimeState::Full => Err(Error::InferenceError(
                    "Qwen3.8 layer runtime state does not match linear-attention layer".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(spec) = self.fused_gdn_for(&mixed_qkv, &z, &alpha, &beta, rows) {
            // Two launches per row; the projections stay batched.
            let mut outputs = Vec::with_capacity(batch_size);
            for (row, state) in states.iter_mut().enumerate() {
                let (conv_state, recurrent_state) = linear_state(state)?;
                outputs.push(self.fused_decode_row(
                    spec,
                    &mixed_qkv.i(row)?,
                    &z.i(row)?,
                    &beta.i(row)?,
                    &alpha.i(row)?,
                    conv_state,
                    recurrent_state,
                )?);
            }
            let refs = outputs.iter().collect::<Vec<_>>();
            let output = Tensor::stack(&refs, 0)?
                .reshape((batch_size, 1, value_width))?
                .to_dtype(residual_dtype)?;
            return self.out_proj.forward(&output).map_err(Error::from);
        }
        let mixed_qkv = mixed_qkv.to_dtype(DType::F32)?;
        let z = z.to_dtype(DType::F32)?;
        let alpha = alpha.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(&beta.to_dtype(DType::F32)?)?;
        if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
            return Err(Error::InferenceError(format!(
                "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }
        let key_width = self.num_k_heads * self.head_k_dim;
        let mut output_rows = Vec::with_capacity(batch_size);
        let mut gate_rows = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let (conv_state, recurrent_state) = match &mut *states[row] {
                Qwen38LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => (conv_state, recurrent_state),
                _ => {
                    return Err(Error::InferenceError(
                        "Qwen3.8 layer runtime state does not match linear-attention layer".into(),
                    ))
                }
            };
            let mixed_qkv = mixed_qkv.i(row)?.unsqueeze(0)?;
            let z = z.i(row)?.unsqueeze(0)?;
            let alpha = alpha.i(row)?.unsqueeze(0)?;
            let beta = beta.i(row)?.unsqueeze(0)?;
            let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;
            let mixed_qkv = self.depthwise_conv_step(&mixed_qkv, conv_state)?;
            let current_state = if let Some(state) = recurrent_state.take() {
                state
            } else {
                if mixed_qkv.device().is_cuda() {
                    record_cuda_state_initial_allocation();
                }
                Tensor::zeros(
                    (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    mixed_qkv.dtype(),
                    mixed_qkv.device(),
                )?
            };
            let beta = beta.reshape((1, self.num_v_heads))?;
            let g = g.reshape((1, self.num_v_heads))?;
            let specialized = if self.fused_decode {
                let result = try_qwen38_deltanet_decode(
                    &mixed_qkv,
                    &g,
                    &beta,
                    &current_state,
                    self.num_k_heads,
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                );
                record_cuda_kernel(CudaKernelPath::DeltaNetSpecializedDecode, result.is_some());
                result
            } else {
                None
            };
            let (output, next_state) = if let Some(result) = specialized {
                result
            } else {
                let query = mixed_qkv.narrow(2, 0, key_width)?.reshape((
                    1,
                    self.num_k_heads,
                    self.head_k_dim,
                ))?;
                let key = mixed_qkv.narrow(2, key_width, key_width)?.reshape((
                    1,
                    self.num_k_heads,
                    self.head_k_dim,
                ))?;
                let value = mixed_qkv.narrow(2, key_width * 2, value_width)?.reshape((
                    1,
                    self.num_v_heads,
                    self.head_v_dim,
                ))?;
                let mut query = l2norm(&query, 1e-6, self.fused_decode)?;
                let mut key = l2norm(&key, 1e-6, self.fused_decode)?;
                if self.num_v_heads != self.num_k_heads {
                    let repeats = self.num_v_heads / self.num_k_heads;
                    if query.device().is_cuda() {
                        record_cuda_head_expansion_materialization();
                        record_cuda_head_expansion_materialization();
                    }
                    query = repeat_interleave_head_states(&query, repeats)?;
                    key = repeat_interleave_head_states(&key, repeats)?;
                }
                recurrent_gated_delta(&query, &key, &value, &g, &beta, current_state)?
            };
            *recurrent_state = Some(next_state);
            output_rows.push(output.reshape((self.num_v_heads, self.head_v_dim))?);
            gate_rows.push(z.reshape((self.num_v_heads, self.head_v_dim))?);
        }
        let output_refs = output_rows.iter().collect::<Vec<_>>();
        let gate_refs = gate_rows.iter().collect::<Vec<_>>();
        let output = Tensor::stack(&output_refs, 0)?;
        let gate = Tensor::stack(&gate_refs, 0)?;
        let output = self.norm.forward(&output, &gate, false)?;
        let output = output
            .reshape((batch_size, 1, self.num_v_heads * self.head_v_dim))?
            .to_dtype(residual_dtype)?;
        self.out_proj.forward(&output).map_err(Error::from)
    }

    fn forward_sequence(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen38LayerRuntimeState,
    ) -> Result<Tensor> {
        self.forward_sequence_recording(hidden_states, state, None)
    }

    fn forward_sequence_recording(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen38LayerRuntimeState,
        verification: Option<&mut Option<Qwen38LinearVerification>>,
    ) -> Result<Tensor> {
        let initial = verification.as_ref().map(|_| state.clone());
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 1 {
            return self.forward(hidden_states, state);
        }

        let (conv_state, recurrent_state) = match state {
            Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.8 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        let residual_dtype = hidden_states.dtype();
        let mut qkv_z = self.qkv_z_proj.forward(hidden_states)?.into_iter();
        let mixed_qkv = required_group_output(&mut qkv_z, "DeltaNet QKV")?.to_dtype(DType::F32)?;
        let z = required_group_output(&mut qkv_z, "DeltaNet Z")?.to_dtype(DType::F32)?;
        let mut alpha_beta = self.alpha_beta_proj.forward(hidden_states)?.into_iter();
        let alpha =
            required_group_output(&mut alpha_beta, "DeltaNet alpha")?.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(
            &required_group_output(&mut alpha_beta, "DeltaNet beta")?.to_dtype(DType::F32)?,
        )?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        let convolution_input = verification.as_ref().map(|_| mixed_qkv.clone());
        let mixed_qkv = self.depthwise_conv_sequence(&mixed_qkv, conv_state)?;
        if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
            return Err(Error::InferenceError(format!(
                "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }

        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            if mixed_qkv.device().is_cuda() {
                record_cuda_state_initial_allocation();
            }
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                mixed_qkv.device(),
            )?
        };

        let beta = beta.reshape((1, seq_len, self.num_v_heads))?;
        let g = g.reshape((1, seq_len, self.num_v_heads))?;
        // The grouped kernel reads the conv output as is. Verification keeps
        // today's path: it retains the normalized q/k for prefix recovery.
        let grouped = if self.gdn_prefill && verification.is_none() {
            let result = qwen38_deltanet_sequence(
                &mixed_qkv,
                &g,
                &beta,
                &current_state,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            );
            if mixed_qkv.device().is_cuda() {
                record_cuda_kernel(CudaKernelPath::DeltaNetGroupedPrefill, result.is_ok());
            }
            match result {
                Ok(result) => Some(result),
                Err(error) => {
                    tracing::warn!(%error, "Qwen3.8 grouped DeltaNet prefill failed; using the legacy recurrence");
                    None
                }
            }
        } else {
            None
        };
        let (output, next_state) = match grouped {
            Some(result) => result,
            None => {
                let (query, key, value) = self.normalized_qkv(&mixed_qkv, seq_len)?;
                if let Some(retained) = verification {
                    *retained = Some(Qwen38LinearVerification {
                        initial: initial.expect("recording retains initial state"),
                        convolution_input: convolution_input.expect("recording retains conv input"),
                        query: query.clone(),
                        key: key.clone(),
                        value: value.clone(),
                        g: g.clone(),
                        beta: beta.clone(),
                    });
                }
                self.legacy_sequence_recurrence(query, key, value, &g, &beta, current_state)?
            }
        };
        // Sequence kernels may return the final state as a view into their
        // packed output. Detach only this fixed-size escape value, using
        // Candle-managed storage rather than an application private buffer.
        *recurrent_state = Some(deep_copy_tensor_storage(&next_state)?);

        let output = output.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z, false)?;
        let output = output
            .reshape((1, seq_len, self.num_v_heads * self.head_v_dim))?
            .to_dtype(residual_dtype)?;
        self.out_proj.forward(&output).map_err(Error::from)
    }

    /// Split a conv output `[1, seq, conv_dim]` into L2-normalized query and
    /// key `[1, seq, key heads, key dim]` and the value `[1, seq, value
    /// heads, value dim]`, as the legacy prefill recurrence consumes them.
    fn normalized_qkv(
        &self,
        mixed_qkv: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let query = mixed_qkv.narrow(2, 0, key_width)?.reshape((
            1,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let key = mixed_qkv.narrow(2, key_width, key_width)?.reshape((
            1,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let value = mixed_qkv.narrow(2, key_width * 2, value_width)?.reshape((
            1,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ))?;
        Ok((
            l2norm(&query, 1e-6, self.fused_decode)?,
            l2norm(&key, 1e-6, self.fused_decode)?,
            value,
        ))
    }

    /// Today's prefill recurrence over normalized q/k: the tiled sequence
    /// kernel when enabled, the portable recurrence otherwise. Both take one
    /// q/k head per value head, so a grouped layout (16 key / 48 value heads)
    /// is first expanded in Transformers' repeat-interleave order; the
    /// kernel is never tried on the unexpanded heads it rejects.
    fn legacy_sequence_recurrence(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = query.dim(1)?;
        let (query, key) = if self.num_v_heads == self.num_k_heads {
            (query, key)
        } else {
            let repeats = self.num_v_heads / self.num_k_heads;
            if query.device().is_cuda() {
                record_cuda_head_expansion_materialization();
                record_cuda_head_expansion_materialization();
            }
            (
                repeat_interleave_head_states_seq(&query, repeats)?,
                repeat_interleave_head_states_seq(&key, repeats)?,
            )
        };
        if self.tiled_recurrence_enabled {
            let tile_size = qwen38_tiled_recurrence_tile_size(
                seq_len,
                self.tiled_recurrence_tile_size_override,
            );
            let fused =
                try_tiled_deltanet_recurrence(&query, &key, &value, g, beta, &state, tile_size);
            if query.device().is_cuda() {
                record_cuda_kernel(CudaKernelPath::DeltaNetPrefill, fused.is_some());
            }
            if let Some(fused) = fused {
                return Ok(fused);
            }
        }
        recurrent_gated_delta_sequence(&query, &key, &value, g, beta, state)
    }

    /// Resolve the grouped DeltaNet prefill (`kernels::cuda::
    /// qwen38_deltanet_sequence`) at load. It needs 128-dim key heads, value
    /// heads grouped over key heads and a value width divisible by 16, and
    /// turns on only after [`Self::gdn_prefill_self_check`] passes on this
    /// device. The kernel has no weights, so one verdict per geometry
    /// (`shared`) serves every layer. `IZWI_QWEN38_GDN_PREFILL=legacy`
    /// keeps today's recurrence.
    fn resolve_gdn_prefill(
        &mut self,
        switches: &Qwen38CudaSwitches,
        device: &Device,
        allow_cpu: bool,
        shared: &mut Option<([usize; 4], Qwen38FastPath)>,
    ) {
        let geometry = [
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        ];
        self.gdn_prefill_path = if let Some(path) =
            decode_gate(GDN_PREFILL_ENV, switches, device, allow_cpu)
        {
            path
        } else if !qwen38_deltanet_sequence_supported(
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        ) {
            Qwen38FastPath::legacy(
                "the grouped DeltaNet prefill needs 128-dim key heads, value heads grouped over key heads and a value width divisible by 16",
            )
        } else if let Some((_, path)) = shared.as_ref().filter(|(seen, _)| *seen == geometry) {
            path.clone()
        } else {
            let path = match self.gdn_prefill_self_check(device) {
                Ok(()) => Qwen38FastPath::Fused,
                Err(error) => Qwen38FastPath::self_check_failed("DeltaNet prefill", &error),
            };
            *shared = Some((geometry, path.clone()));
            path
        };
        self.gdn_prefill = self.gdn_prefill_path.is_fused();
    }

    /// The grouped prefill kernel against today's recurrence (split, L2
    /// norms, head expansion, the tiled or portable recurrence) on synthetic
    /// inputs at this layer's geometry: 37 tokens, crossing the legacy
    /// path's 16-token tiles and the kernel's one-token prefetch, from a
    /// non-zero state. Outputs and final states must agree to 1e-4 relative
    /// L2 and 1e-3 of the largest value per element (NaN-safe).
    fn gdn_prefill_self_check(&self, device: &Device) -> Result<()> {
        const TOKENS: usize = 37;
        let (heads, key_dim, value_dim) = (self.num_v_heads, self.head_k_dim, self.head_v_dim);
        let conv_dim = 2 * self.num_k_heads * key_dim + heads * value_dim;
        let tensor = |values: Vec<f32>, shape: &[usize]| -> Result<Tensor> {
            Ok(Tensor::from_vec(values, shape, &Device::Cpu)?.to_device(device)?)
        };
        // Pseudo-random (xorshift) inputs: a sinusoid's structure lets a
        // wrong key-head mapping through at a fraction of a percent.
        let noise =
            |rows: usize, width: usize, seed: u64| probe_activations(rows, width, DType::F32, seed);
        let mixed_qkv = tensor(noise(TOKENS, conv_dim, 31), &[1, TOKENS, conv_dim])?;
        let g = tensor(
            noise(TOKENS, heads, 32)
                .iter()
                .map(|v| -(0.02 + 0.5 * v.abs()))
                .collect(),
            &[1, TOKENS, heads],
        )?;
        let beta = tensor(
            noise(TOKENS, heads, 33)
                .iter()
                .map(|v| 0.5 + 0.45 * v.clamp(-1.0, 1.0))
                .collect(),
            &[1, TOKENS, heads],
        )?;
        let state = tensor(
            noise(heads * key_dim, value_dim, 34)
                .iter()
                .map(|v| 0.1 * v)
                .collect(),
            &[1, heads, key_dim, value_dim],
        )?;
        let (output, next_state) = qwen38_deltanet_sequence(
            &mixed_qkv,
            &g,
            &beta,
            &state,
            self.num_k_heads,
            heads,
            key_dim,
            value_dim,
        )?;
        let (query, key, value) = self.normalized_qkv(&mixed_qkv, TOKENS)?;
        let (expected, expected_state) =
            self.legacy_sequence_recurrence(query, key, value, &g, &beta, state)?;
        compare_values(
            "DeltaNet prefill output",
            &host_values(&output)?,
            &host_values(&expected)?,
            1e-4,
            1e-3,
        )?;
        compare_values(
            "DeltaNet prefill state",
            &host_values(&next_state)?,
            &host_values(&expected_state)?,
            1e-4,
            1e-3,
        )
    }

    fn depthwise_conv_sequence(
        &self,
        mixed_qkv: &Tensor,
        conv_state: &mut Option<ConvRingState>,
    ) -> Result<Tensor> {
        let seq_len = mixed_qkv.dim(1)?;
        if seq_len == 1 {
            return self.depthwise_conv_step(mixed_qkv, conv_state);
        }

        if self.kernel_size > 1 {
            let history_len = self.kernel_size - 1;
            let buffer = conv_state.as_mut().ok_or_else(|| {
                Error::InferenceError("conv_state not initialized but kernel_size > 1".to_string())
            })?;
            if buffer.history_len()? != history_len {
                return Err(Error::InferenceError(format!(
                    "Invalid Qwen3.8 convolution history: actual_history={}, expected_history={history_len}",
                    buffer.history_len()?
                )));
            }
            let history = buffer.contiguous_history()?;
            let fused = if !mixed_qkv.device().is_cuda() || self.fused_decode {
                try_qwen38_causal_conv_sequence(mixed_qkv, &self.conv_kernel, &history)
            } else {
                None
            };
            if mixed_qkv.device().is_cuda() {
                record_cuda_kernel(CudaKernelPath::CausalConvPrefill, fused.is_some());
            }
            if let Some((output, final_history)) = fused {
                // Prefill output packing may retain the entire sequence, so
                // detach its fixed history at the sequence boundary.
                buffer
                    .replace_contiguous(deep_copy_tensor_storage(&final_history)?, history_len)?;
                return Ok(output);
            }
        }

        let mut outputs = Vec::with_capacity(seq_len);
        for idx in 0..seq_len {
            let token = mixed_qkv.narrow(1, idx, 1)?;
            outputs.push(self.depthwise_conv_step(&token, conv_state)?);
        }
        if let Some(conv_state) = conv_state.as_mut() {
            conv_state.compact_owned()?;
        }
        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&output_refs, 1).map_err(Error::from)
    }

    fn depthwise_conv_step(
        &self,
        mixed_qkv: &Tensor,
        conv_state: &mut Option<ConvRingState>,
    ) -> Result<Tensor> {
        if self.kernel_size == 4 && mixed_qkv.device().is_cuda() && self.fused_decode {
            let buffer = conv_state.as_mut().ok_or_else(|| {
                Error::InferenceError(
                    "conv_state not initialized but Qwen3.8 decode convolution needs history"
                        .to_string(),
                )
            })?;
            if buffer.history_len()? != 3 {
                return Err(Error::InferenceError(format!(
                    "Invalid Qwen3.8 decode convolution history: actual_history={}, expected_history=3",
                    buffer.history_len()?
                )));
            }
            let history = buffer.contiguous_history()?;
            let fused = try_qwen38_causal_conv_decode(mixed_qkv, &self.conv_kernel, &history);
            record_cuda_kernel(CudaKernelPath::CausalConvDecode, fused.is_some());
            if let Some((output, next_history)) = fused {
                // The dedicated kernel returns oldest-to-newest contiguous
                // history. Retain it directly so subsequent decode tokens and
                // transaction staging do not stack/copy three slot views.
                buffer.replace_contiguous(next_history, 3)?;
                return Ok(output);
            }
        }

        let current = mixed_qkv.i((0, 0))?;
        let current = if current.dtype() != self.conv_kernel.dtype() {
            current.to_dtype(self.conv_kernel.dtype())?
        } else {
            current
        };
        let current = current.reshape((self.conv_dim, 1))?;

        let convolved = if self.kernel_size <= 1 {
            (&current * &self.conv_kernel)?.sum(D::Minus1)?
        } else {
            let buffer = if let Some(state) = conv_state.as_mut() {
                state
            } else {
                return Err(Error::InferenceError(
                    "conv_state not initialized but kernel_size > 1".to_string(),
                ));
            };

            // Compute convolution as sum of elementwise products
            // self.conv_kernel shape: (conv_dim, kernel_size)
            // ring buffer contains kernel_size - 1 past states of shape (conv_dim, 1)

            // Start with current * conv_kernel[:, kernel_size - 1]
            let k_slice = &self.conv_kernel_slices[self.kernel_size - 1];
            let mut convolved = (&current * k_slice)?;

            // Add previous tokens * their respective kernel weights.
            // Read history in oldest -> newest order from the circular ring.
            let history_len = self.kernel_size - 1;
            let (slots, next_idx) = buffer.ensure_ring()?;
            for i in 0..(self.kernel_size - 1) {
                let ring_idx = (*next_idx + i) % history_len;
                let prev_token = &slots[ring_idx];
                let k_slice = &self.conv_kernel_slices[i];
                convolved = (&convolved + &(prev_token * k_slice)?)?;
            }

            // Update the ring buffer in O(1): overwrite oldest and advance cursor.
            buffer.push_decode(&current)?;

            convolved.squeeze(1)?
        };

        let convolved = ops::silu(&convolved)?;
        convolved
            .reshape((1, 1, self.conv_dim))
            .map_err(Error::from)
    }
}

impl Qwen38GatedRmsNorm {
    fn forward(&self, hidden_states: &Tensor, gate: &Tensor, _decode: bool) -> Result<Tensor> {
        if hidden_states.dtype() == DType::F32 {
            if self.fused_decode {
                let candidate =
                    try_qwen38_gated_rms_norm_decode(hidden_states, gate, &self.weight, self.eps);
                record_cuda_kernel(CudaKernelPath::GatedRmsNorm, candidate.is_some());
                if let Some(result) = candidate {
                    return Ok(result);
                }
            }
            let fused = try_fused_gated_rms_norm(hidden_states, gate, &self.weight, self.eps);
            if let Some(result) = fused {
                return Ok(result);
            }
        }

        let normalized = candle_nn::ops::rms_norm(hidden_states, &self.weight, self.eps as f32)?;
        (&normalized * &ops::silu(gate)?).map_err(Error::from)
    }
}

/// What one load's block-FP8 checks resolved, for the diagnostics.
#[derive(Debug, Default)]
pub(super) struct Qwen38Fp8Outcomes {
    /// One entry per block-FP8 projection group: packed, or why not.
    pub(super) packing: Vec<Qwen38FastPath>,
    /// Metal: one entry per checked FP8 projection kind (GEMV or rerouted).
    pub(super) gemv: Vec<Qwen38FastPath>,
    /// CUDA: one entry per checked FP8 projection kind: the dequantized
    /// prefill GEMM, or why the kind keeps the compact kernel.
    pub(super) prefill_gemm: Vec<Qwen38FastPath>,
}

/// Load-scoped state shared by every projection one model load
/// materializes: the checkpoint, the residency policy, the device, and the
/// bookkeeping of the load-time FP8 checks and packing.
pub(super) struct Qwen38WeightLoader<'a> {
    pub(super) tensors: &'a IndexedSafetensors,
    pub(super) policy: &'a Qwen38ResidencyPolicy,
    pub(super) device: &'a Device,
    pub(super) block: [usize; 2],
    pub(super) target: ProjectionMaterialization,
    /// Projection kinds (layer index removed) whose resident FP8 form has
    /// passed the check against the original checkpoint bytes.
    checked: std::cell::RefCell<std::collections::BTreeSet<String>>,
    /// One entry per block-FP8 projection group: packed, or why not.
    packing: std::cell::RefCell<Vec<Qwen38FastPath>>,
    /// Metal: one entry per checked FP8 projection kind: the GEMV passed,
    /// or the kind was rerouted through dequantize + matmul.
    gemv: std::cell::RefCell<Vec<Qwen38FastPath>>,
    /// Metal: projection kinds whose GEMV failed its check.
    dequant_kinds: std::cell::RefCell<std::collections::BTreeSet<String>>,
    /// CUDA: one entry per checked FP8 projection kind: the dequantized
    /// prefill GEMM passed its check, or why the kind keeps `mm`.
    prefill_gemm: std::cell::RefCell<Vec<Qwen38FastPath>>,
    /// The prefill GEMM's verdict per checked projection kind.
    prefill_gemm_kinds: std::cell::RefCell<std::collections::BTreeMap<String, bool>>,
    /// Tests: corrupt the scales of every FP8 projection this loader
    /// materializes, to prove the load-time check fails closed.
    #[cfg(test)]
    pub(super) corrupt_fp8_scales: std::cell::Cell<bool>,
}

/// Activation rows of the load-time FP8 check: decode, verification and a
/// prefill-sized span, so every kernel the projection runs is compared.
const FP8_CHECK_ROWS: [usize; 3] = [1, 4, 16];
/// Output-row stride of the load-time FP8 check (plus every pack boundary).
const FP8_CHECK_ROW_STRIDE: usize = 61;

impl<'a> Qwen38WeightLoader<'a> {
    pub(super) fn new(
        tensors: &'a IndexedSafetensors,
        policy: &'a Qwen38ResidencyPolicy,
        device: &'a Device,
        block: [usize; 2],
        target: ProjectionMaterialization,
    ) -> Self {
        Self {
            tensors,
            policy,
            device,
            block,
            target,
            checked: Default::default(),
            packing: Default::default(),
            gemv: Default::default(),
            dequant_kinds: Default::default(),
            prefill_gemm: Default::default(),
            prefill_gemm_kinds: Default::default(),
            #[cfg(test)]
            corrupt_fp8_scales: Default::default(),
        }
    }

    pub(super) fn into_packing(self) -> Vec<Qwen38FastPath> {
        self.into_outcomes().packing
    }

    /// What this load's FP8 checks resolved.
    pub(super) fn into_outcomes(self) -> Qwen38Fp8Outcomes {
        Qwen38Fp8Outcomes {
            packing: self.packing.into_inner(),
            gemv: self.gemv.into_inner(),
            prefill_gemm: self.prefill_gemm.into_inner(),
        }
    }

    /// Whether a freshly materialized projection of `kind` runs the
    /// dequantized prefill GEMM: the policy resolved it and the kind has not
    /// failed its check in this load.
    fn prefill_gemm_for(&self, kind: &str) -> bool {
        self.policy.fp8_prefill_gemm() && self.prefill_gemm_kinds.borrow().get(kind) != Some(&false)
    }

    fn fp8_kernel(&self) -> Qwen38Fp8Kernel {
        if self.device.is_cuda() && self.policy.fp8_vectorized() {
            Qwen38Fp8Kernel::Vectorized
        } else if self.device.is_metal() && self.policy.fp8_dequant_only() {
            Qwen38Fp8Kernel::DequantOnly
        } else {
            Qwen38Fp8Kernel::Reference
        }
    }

    fn kind(parts: &[(&str, [usize; 2])]) -> String {
        parts
            .iter()
            .map(|(name, _)| projection_kind(name))
            .collect::<Vec<_>>()
            .join("+")
    }

    /// Check a freshly materialized FP8 projection (see
    /// [`Self::fp8_self_check`]). On Metal a mismatch first reroutes the
    /// projection kind through dequantize + matmul (the GEMV may be the
    /// culprit) and checks again; only a second mismatch fails.
    pub(super) fn checked_fp8(
        &self,
        parts: &[(&str, [usize; 2])],
        projection: Qwen38Projection,
    ) -> Result<Qwen38Projection> {
        let projection = self.checked_fp8_kernels(parts, projection)?;
        Ok(self.checked_prefill_gemm(parts, projection))
    }

    /// The dequantized prefill GEMM of a projection that passed its kernel
    /// checks: once per projection kind, [`fp8_prefill_gemm_check`] compares
    /// it with the compact kernel and the original checkpoint bytes. A
    /// mismatch keeps the kind on `mm` (with a diagnostics reason) instead of
    /// failing the load.
    fn checked_prefill_gemm(
        &self,
        parts: &[(&str, [usize; 2])],
        projection: Qwen38Projection,
    ) -> Qwen38Projection {
        if !matches!(
            projection,
            Qwen38Projection::NativeFp8 {
                prefill_gemm: true,
                ..
            }
        ) {
            return projection;
        }
        let kind = Self::kind(parts);
        if self.prefill_gemm_kinds.borrow().contains_key(&kind) {
            return projection;
        }
        match fp8_prefill_gemm_check(
            self.tensors,
            parts,
            self.block,
            &projection,
            self.target.dtype(),
            self.device,
        ) {
            Ok(()) => {
                self.prefill_gemm_kinds.borrow_mut().insert(kind, true);
                self.prefill_gemm.borrow_mut().push(Qwen38FastPath::Fused);
                projection
            }
            Err(error) => {
                record_fast_path_self_check_failure();
                tracing::warn!(%error, kind, "Qwen3.8 dequantized FP8 prefill GEMM failed its load-time check; this projection kind keeps the compact FP8 kernel");
                self.prefill_gemm
                    .borrow_mut()
                    .push(Qwen38FastPath::legacy(format!(
                        "self_check_failed: {kind}: {error}"
                    )));
                self.prefill_gemm_kinds.borrow_mut().insert(kind, false);
                projection
                    .with_prefill_gemm(false)
                    .expect("checked projections are native block FP8")
            }
        }
    }

    /// The kernel checks of [`Self::checked_fp8`] (the decode GEMVs and the
    /// compact `mm` kernel); see [`Self::fp8_self_check`].
    fn checked_fp8_kernels(
        &self,
        parts: &[(&str, [usize; 2])],
        projection: Qwen38Projection,
    ) -> Result<Qwen38Projection> {
        let kind = Self::kind(parts);
        let fresh = !self.checked.borrow().contains(&kind);
        match self.fp8_self_check(parts, &projection) {
            Ok(()) => {
                if fresh && self.device.is_metal() {
                    self.gemv.borrow_mut().push(match &projection {
                        Qwen38Projection::NativeFp8 {
                            kernel: Qwen38Fp8Kernel::DequantOnly,
                            ..
                        } => Qwen38FastPath::legacy(format!(
                            "{}=legacy",
                            super::residency::FP8_GEMV_ENV
                        )),
                        _ => Qwen38FastPath::Fused,
                    });
                }
                Ok(projection)
            }
            Err(error) if self.device.is_metal() => {
                let Qwen38Projection::NativeFp8 {
                    raw,
                    kernel,
                    prefill_gemm,
                } = projection
                else {
                    return Err(error);
                };
                if kernel == Qwen38Fp8Kernel::DequantOnly {
                    return Err(error);
                }
                let rerouted = Qwen38Projection::NativeFp8 {
                    raw,
                    kernel: Qwen38Fp8Kernel::DequantOnly,
                    prefill_gemm,
                };
                self.fp8_self_check(parts, &rerouted).map_err(|second| {
                    Error::ModelLoadError(format!(
                        "{error}; with dequantize + matmul only: {second}"
                    ))
                })?;
                record_fast_path_self_check_failure();
                tracing::warn!(%error, kind, "Qwen3.8 Metal FP8 GEMV failed its check; this projection kind runs dequantize + matmul");
                self.dequant_kinds.borrow_mut().insert(kind);
                self.gemv.borrow_mut().push(Qwen38FastPath::legacy(format!(
                    "self_check_failed: {error}"
                )));
                Ok(rerouted)
            }
            Err(error) => Err(error),
        }
    }

    /// The parts' bytes as one raw block-FP8 projection on the device.
    fn materialize_fp8(&self, parts: &[(&str, [usize; 2])]) -> Result<Qwen38Projection> {
        let raw = self
            .tensors
            .materialize_block_fp8_raw_group(parts, self.block, self.device)?;
        #[cfg(test)]
        let raw = if self.corrupt_fp8_scales.get() {
            super::native::RawBlockFp8Projection {
                scales: (&raw.scales * 1.5)?,
                ..raw
            }
        } else {
            raw
        };
        let kind = Self::kind(parts);
        let kernel = if self.dequant_kinds.borrow().contains(&kind) {
            Qwen38Fp8Kernel::DequantOnly
        } else {
            self.fp8_kernel()
        };
        Ok(Qwen38Projection::NativeFp8 {
            raw,
            kernel,
            prefill_gemm: self.prefill_gemm_for(&kind),
        })
    }

    /// Check a resident FP8 projection (one part, or a pack of `parts` in
    /// row order) against a host reference built from the ORIGINAL per-part
    /// checkpoint bytes and scales, never from the uploaded copy: sampled
    /// output rows, every pack boundary, at 1, 4 and 16 activation rows,
    /// through the compact kernels (the prefill GEMM has its own check).
    /// Runs once per projection kind (layer index removed).
    pub(super) fn fp8_self_check(
        &self,
        parts: &[(&str, [usize; 2])],
        projection: &Qwen38Projection,
    ) -> Result<()> {
        let kind = Self::kind(parts);
        if self.checked.borrow().contains(&kind) {
            return Ok(());
        }
        let compact = projection.with_prefill_gemm(false);
        fp8_projection_check(
            self.tensors,
            parts,
            self.block,
            compact.as_ref().unwrap_or(projection),
            self.target.dtype(),
            self.device,
        )
        .map_err(|error| Error::ModelLoadError(format!("{kind}: {error}")))?;
        self.checked.borrow_mut().insert(kind);
        Ok(())
    }
}

/// `model.language_model.layers.7.mlp.down_proj.weight` -> `mlp.down_proj`;
/// MTP names keep an `mtp:` tag.
fn projection_kind(name: &str) -> String {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let (tag, rest) = if let Some(rest) = stem.strip_prefix("mtp.layers.") {
        ("mtp:", rest)
    } else if let Some(rest) = stem.strip_prefix("model.language_model.layers.") {
        ("", rest)
    } else {
        return stem.to_string();
    };
    format!(
        "{tag}{}",
        rest.split_once('.').map_or(rest, |(_, rest)| rest)
    )
}

/// Output rows a load-time FP8 check samples and their original
/// checkpoint bytes: every [`FP8_CHECK_ROW_STRIDE`]th row, both sides of every
/// pack and scale-block boundary, the last row and `extra` rows, each with
/// `(bytes, the part's scale grid, row block in the part)`.
struct Fp8CheckReference {
    k: usize,
    total: usize,
    sampled: Vec<usize>,
    rows: Vec<(Vec<u8>, std::sync::Arc<Vec<f32>>, usize)>,
}

impl Fp8CheckReference {
    fn new(
        tensors: &IndexedSafetensors,
        parts: &[(&str, [usize; 2])],
        block: [usize; 2],
        extra: impl IntoIterator<Item = usize>,
    ) -> Result<Self> {
        let k = parts
            .first()
            .map(|(_, shape)| shape[1])
            .ok_or_else(|| Error::ModelLoadError("empty FP8 check".into()))?;
        let total: usize = parts.iter().map(|(_, shape)| shape[0]).sum();
        let mut rows: std::collections::BTreeSet<usize> =
            (0..total).step_by(FP8_CHECK_ROW_STRIDE).collect();
        let mut offset = 0usize;
        for (_, shape) in parts {
            for row in [
                offset.wrapping_sub(1),
                offset,
                offset + 1,
                offset + 127,
                offset + 128,
            ] {
                if row < total {
                    rows.insert(row);
                }
            }
            offset += shape[0];
        }
        rows.insert(total - 1);
        rows.extend(extra.into_iter().filter(|row| *row < total));
        let mut references = Vec::with_capacity(rows.len());
        let mut offset = 0usize;
        for (name, shape) in parts {
            let local: Vec<usize> = rows
                .range(offset..offset + shape[0])
                .map(|row| row - offset)
                .collect();
            if !local.is_empty() {
                let (bytes, scales) = tensors.block_fp8_rows(name, *shape, block, &local)?;
                let scales = std::sync::Arc::new(scales);
                for (index, row) in local.iter().enumerate() {
                    references.push((
                        bytes[index * k..(index + 1) * k].to_vec(),
                        scales.clone(),
                        row / block[0],
                    ));
                }
            }
            offset += shape[0];
        }
        Ok(Self {
            k,
            total,
            sampled: rows.into_iter().collect(),
            rows: references,
        })
    }

    /// `(actual, expected)` at the sampled rows for `m` activation rows `x`
    /// and the projection's `[m, total]` output.
    fn sampled(
        &self,
        block: [usize; 2],
        x: &[f32],
        output: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let m = x.len() / self.k;
        if output.len() != m * self.total {
            return Err(Error::ModelLoadError(format!(
                "FP8 check produced {} values, expected {}",
                output.len(),
                m * self.total
            )));
        }
        let scale_cols = self.k.div_ceil(block[1]);
        let mut actual = Vec::with_capacity(m * self.sampled.len());
        let mut expected = Vec::with_capacity(m * self.sampled.len());
        for (row_index, activations) in x.chunks(self.k).enumerate() {
            for (row, (bytes, scales, row_block)) in self.sampled.iter().zip(&self.rows) {
                actual.push(output[row_index * self.total + row]);
                expected.push(fp8_row_dot(
                    bytes,
                    scales,
                    *row_block,
                    scale_cols,
                    block[1],
                    activations,
                ));
            }
        }
        Ok((actual, expected))
    }
}

/// `projection` on `m` probe rows (seeded) of the activation dtype, on the
/// host as F32 `(activations, output)`.
fn fp8_check_forward(
    projection: &Qwen38Projection,
    m: usize,
    k: usize,
    activation: DType,
    seed: u64,
    device: &Device,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let x = probe_activations(m, k, activation, seed);
    let input = Tensor::from_vec(x.clone(), (1, m, k), &Device::Cpu)?
        .to_dtype(activation)?
        .to_device(device)?;
    let output = projection
        .forward(&input)?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((x, output))
}

/// The comparison behind [`Qwen38WeightLoader::fp8_self_check`]. Tolerance:
/// 1% relative L2 and 2% of the largest reference magnitude per element,
/// NaN-safe; the reference multiplies the same activation-dtype-rounded
/// inputs the device sees.
pub(super) fn fp8_projection_check(
    tensors: &IndexedSafetensors,
    parts: &[(&str, [usize; 2])],
    block: [usize; 2],
    projection: &Qwen38Projection,
    activation: DType,
    device: &Device,
) -> Result<()> {
    let reference = Fp8CheckReference::new(tensors, parts, block, [])?;
    for (seed, m) in FP8_CHECK_ROWS.into_iter().enumerate() {
        let (x, output) = fp8_check_forward(
            projection,
            m,
            reference.k,
            activation,
            101 + seed as u64,
            device,
        )?;
        let (actual, expected) = reference.sampled(block, &x, &output)?;
        compare_values(
            &format!("block-FP8 check at {m} rows"),
            &actual,
            &expected,
            0.01,
            0.02,
        )?;
    }
    Ok(())
}

/// Activation rows of the load-time prefill GEMM check: the switch-over (one
/// `mm` tile) and a multi-tile span.
const FP8_PREFILL_CHECK_ROWS: [usize; 2] = [FP8_PREFILL_GEMM_MIN_ROWS, 64];

/// The check behind [`Qwen38WeightLoader::checked_prefill_gemm`], at the
/// projection's real geometry and at 16 and 64 activation rows: the
/// dequantized GEMM's COMPLETE output against the compact `mm` kernel on
/// the same resident tensors, and its sampled rows (plus both sides of every
/// dequantization tile boundary) against a host reference from the original
/// checkpoint bytes. Same tolerances as [`fp8_projection_check`], NaN-safe.
pub(super) fn fp8_prefill_gemm_check(
    tensors: &IndexedSafetensors,
    parts: &[(&str, [usize; 2])],
    block: [usize; 2],
    projection: &Qwen38Projection,
    activation: DType,
    device: &Device,
) -> Result<()> {
    let (gemm, compact) = match (
        projection.with_prefill_gemm(true),
        projection.with_prefill_gemm(false),
    ) {
        (Some(gemm), Some(compact)) => (gemm, compact),
        _ => {
            return Err(Error::ModelLoadError(
                "the prefill GEMM check needs a native block-FP8 projection".into(),
            ))
        }
    };
    let total: usize = parts.iter().map(|(_, shape)| shape[0]).sum();
    let k = parts.first().map_or(0, |(_, shape)| shape[1]);
    let tile = crate::kernels::cuda::fp8::dequant_gemm_tile_rows(total, k).max(1);
    let boundaries = (tile..total).step_by(tile).flat_map(|row| [row - 1, row]);
    let reference = Fp8CheckReference::new(tensors, parts, block, boundaries)?;
    for (seed, m) in FP8_PREFILL_CHECK_ROWS.into_iter().enumerate() {
        let seed = 211 + seed as u64;
        let (x, output) = fp8_check_forward(&gemm, m, reference.k, activation, seed, device)?;
        let (_, compact) = fp8_check_forward(&compact, m, reference.k, activation, seed, device)?;
        compare_values(
            &format!("prefill GEMM vs the compact FP8 kernel at {m} rows"),
            &output,
            &compact,
            0.01,
            0.02,
        )?;
        let (actual, expected) = reference.sampled(block, &x, &output)?;
        compare_values(
            &format!("prefill GEMM vs the checkpoint bytes at {m} rows"),
            &actual,
            &expected,
            0.01,
            0.02,
        )?;
    }
    Ok(())
}

/// Load one projection in the form the residency policy chose. A native FP8
/// projection is checked against the original checkpoint bytes before it is
/// published; a mismatch fails the load and poisons the policy, so the retry
/// resolves (and admission prices) the fallback.
pub(super) fn load_native_projection(
    loader: &Qwen38WeightLoader<'_>,
    name: &str,
    shape: [usize; 2],
) -> Result<Qwen38Projection> {
    let tensors = loader.tensors;
    let info = tensors.tensor_info(name)?;
    match loader
        .policy
        .projection_form(info.dtype, shape, loader.block)
    {
        Qwen38WeightForm::RawFp8 => {
            let projection = loader.materialize_fp8(&[(name, shape)])?;
            loader
                .checked_fp8(&[(name, shape)], projection)
                .map_err(|error| {
                    record_fast_path_self_check_failure();
                    let reason = error.to_string();
                    loader.policy.poison_projections(&reason);
                    tracing::warn!(%reason, "Qwen3.8 native FP8 load-time check failed");
                    let remedy = if loader.device.is_metal() {
                        "The next load expands projections to F16; set IZWI_METAL_PROJECTION_BACKEND=expanded to skip native FP8 on Metal"
                    } else {
                        "The next load keeps Q8_0 projections; set IZWI_QWEN38_FP8_PROJECTIONS=legacy (or IZWI_CUDA_PROJECTION_BACKEND=q8) to skip native FP8"
                    };
                    Error::ModelLoadError(format!(
                        "Qwen3.8 native block-FP8 self_check_failed: {reason}. {remedy}"
                    ))
                })
        }
        Qwen38WeightForm::Q8_0 => tensors
            .materialize_q8_projection_parallel(&[(name, shape)], loader.block, loader.device)
            .map(Qwen38Projection::Quantized),
        Qwen38WeightForm::Dense(_) => tensors
            .materialize_projection(name, shape, loader.block, loader.target, loader.device)
            .map(|weight| Qwen38Projection::Dense(Linear::new(weight, None))),
    }
}

/// Load the LM head in the form the residency policy chose. A block-FP8 head
/// that fails its quality gate or device check fails the load closed and
/// poisons the head's FP8 form, so the retry admits and loads the dense head.
fn load_lm_head(
    loader: &Qwen38WeightLoader<'_>,
    shape: [usize; 2],
) -> Result<(Qwen38Projection, (Qwen38FastPath, Option<f64>))> {
    let name = "lm_head.weight";
    if loader.policy.lm_head_form(shape) == Qwen38WeightForm::Q8_0 {
        // The CPU's Q8_0 head, requantized from the BF16 mapping in row tiles.
        return Ok((
            loader
                .tensors
                .materialize_dense_as_q8(name, shape, loader.device)
                .map(Qwen38Projection::Quantized)?,
            (
                Qwen38FastPath::legacy("the CPU LM head is Q8_0, not block FP8"),
                None,
            ),
        ));
    }
    if loader.policy.lm_head_form(shape) != Qwen38WeightForm::RawFp8 {
        let reason = match loader.policy.fp8_lm_head() {
            Qwen38FastPath::Legacy { reason } => reason.clone(),
            Qwen38FastPath::Fused => "outside the FP8 kernel contract".into(),
        };
        return Ok((
            load_native_projection(loader, name, shape)?,
            (Qwen38FastPath::legacy(reason), None),
        ));
    }
    match super::lm_head::quantize_lm_head_from_checkpoint(
        loader.tensors,
        name,
        shape,
        loader.device,
        loader.target.dtype(),
        loader.fp8_kernel(),
    )? {
        Ok(head) => Ok((
            head.projection,
            (Qwen38FastPath::Fused, Some(head.gate_relative_l2)),
        )),
        Err(reason) => {
            record_fast_path_self_check_failure();
            loader.policy.poison_lm_head(&reason);
            tracing::warn!(%reason, "Qwen3.8 block-FP8 LM head failed its checks");
            Err(Error::ModelLoadError(format!(
                "Qwen3.8 block-FP8 LM head self_check_failed: {reason}. The next load keeps the dense head; set IZWI_QWEN38_FP8_LM_HEAD=legacy to skip it"
            )))
        }
    }
}

/// Load row-compatible projections as one packed matrix when the residency
/// policy packs their (shared) form, else separately. A block-FP8 pack needs
/// whole 128-row scale blocks in every part but the last; it is checked
/// against the original per-part bytes, and a pack that fails is dropped and
/// rebuilt from separate parts (the same bytes).
fn load_native_projection_group(
    loader: &Qwen38WeightLoader<'_>,
    projections: &[(&str, [usize; 2])],
) -> Result<Qwen38ProjectionGroup> {
    let tensors = loader.tensors;
    let load_separate = || {
        projections
            .iter()
            .map(|(name, shape)| load_native_projection(loader, name, *shape))
            .collect::<Result<Vec<_>>>()
            .map(Qwen38ProjectionGroup::Separate)
    };
    let forms = projections
        .iter()
        .map(|(name, shape)| {
            tensors.tensor_info(name).map(|info| {
                loader
                    .policy
                    .projection_form(info.dtype, *shape, loader.block)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let fp8 = forms.first() == Some(&Qwen38WeightForm::RawFp8);
    let packable =
        loader.policy.packing_allowed(&forms) && projection_group_geometry_compatible(projections);
    if !packable {
        if forms.contains(&Qwen38WeightForm::RawFp8) {
            loader.packing.borrow_mut().push(Qwen38FastPath::legacy(
                if forms.iter().all(|f| *f == forms[0]) {
                    "packing off"
                } else {
                    "mixed resident forms"
                },
            ));
        }
        return load_separate();
    }
    let widths = projections.iter().map(|(_, shape)| shape[0]).collect();
    let projection = match forms[0] {
        Qwen38WeightForm::RawFp8 => {
            let aligned = projections
                .iter()
                .take(projections.len() - 1)
                .all(|(_, shape)| shape[0].is_multiple_of(loader.block[0]));
            if !aligned {
                loader.packing.borrow_mut().push(Qwen38FastPath::legacy(
                    "a part does not end on a 128-row scale block",
                ));
                return load_separate();
            }
            let packed = loader
                .materialize_fp8(projections)
                .and_then(|projection| loader.checked_fp8(projections, projection));
            match packed {
                Ok(projection) => {
                    loader.packing.borrow_mut().push(Qwen38FastPath::Fused);
                    projection
                }
                Err(error) => {
                    record_fast_path_self_check_failure();
                    tracing::warn!(
                        %error,
                        "Qwen3.8 packed block-FP8 projection failed its check; loading the parts separately"
                    );
                    loader
                        .packing
                        .borrow_mut()
                        .push(Qwen38FastPath::legacy(format!(
                            "self_check_failed: {error}"
                        )));
                    return load_separate();
                }
            }
        }
        Qwen38WeightForm::Q8_0 => tensors
            .materialize_q8_projection_parallel(projections, loader.block, loader.device)
            .map(Qwen38Projection::Quantized)?,
        Qwen38WeightForm::Dense(_) => tensors
            .materialize_projection_group(projections, loader.block, loader.target, loader.device)
            .map(|weight| Qwen38Projection::Dense(Linear::new(weight, None)))?,
    };
    debug_assert!(fp8 == matches!(projection, Qwen38Projection::NativeFp8 { .. }));
    Ok(Qwen38ProjectionGroup::Packed { projection, widths })
}

fn projection_group_geometry_compatible(projections: &[(&str, [usize; 2])]) -> bool {
    let Some((_, first)) = projections.first() else {
        return false;
    };
    projections
        .iter()
        .all(|(_, shape)| shape[0] > 0 && shape[1] == first[1])
}

fn split_projection_output(output: &Tensor, widths: &[usize]) -> Result<Vec<Tensor>> {
    let output_axis = output.rank().checked_sub(1).ok_or_else(|| {
        Error::InferenceError("Qwen3.8 projection output cannot be scalar".into())
    })?;
    let output_width = output.dim(output_axis)?;
    let expected_width = widths.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width)
            .ok_or_else(|| Error::InferenceError("Qwen3.8 projection split width overflow".into()))
    })?;
    if output_width != expected_width {
        return Err(Error::InferenceError(format!(
            "Qwen3.8 packed projection produced width {output_width}, expected {expected_width}"
        )));
    }
    let mut offset = 0usize;
    widths
        .iter()
        .map(|width| {
            let split = output
                .narrow(output_axis, offset, *width)
                .map_err(Error::from)?;
            offset += *width;
            Ok(split)
        })
        .collect()
}

fn required_group_output(outputs: &mut std::vec::IntoIter<Tensor>, label: &str) -> Result<Tensor> {
    outputs.next().ok_or_else(|| {
        Error::InferenceError(format!("Qwen3.8 packed projection omitted {label} output"))
    })
}

pub(super) fn load_native_zero_centered_norm(
    tensors: &IndexedSafetensors,
    name: &str,
    length: usize,
    eps: f64,
    device: &Device,
) -> Result<Qwen38RmsNorm> {
    // The shifted gain stays F32 whatever the plan's activation dtype: `1 + w`
    // rounded to BF16 loses up to ~0.4% per channel. `Qwen38RmsNorm` runs the
    // norm in the gain's dtype, as HF does.
    let weight = tensors.materialize_dense_tensor(
        name,
        &[length],
        ProjectionMaterialization::F32,
        device,
    )?;
    let weight = (weight + 1.0)?;
    Ok(Qwen38RmsNorm::new(weight, eps))
}

/// The convolution and recurrent state of a linear-attention layer.
fn linear_state(
    state: &mut Qwen38LayerRuntimeState,
) -> Result<(&mut Option<ConvRingState>, &mut Option<Tensor>)> {
    match state {
        Qwen38LayerRuntimeState::Linear {
            conv_state,
            recurrent_state,
        } => Ok((conv_state, recurrent_state)),
        Qwen38LayerRuntimeState::Full => Err(Error::InferenceError(
            "Qwen3.8 layer runtime state does not match linear-attention layer".to_string(),
        )),
    }
}

/// Qwen3.8's channel-major causal-conv decode step. `x` holds `conv_dim`
/// values (F32, BF16 or F16), `weight` is `[C, 4]` F32 and `history` the
/// `[C, 3]` F32 history, oldest first. Returns fresh `(silu(conv) [C] F32,
/// next history [C, 3] F32)`. CUDA runs q38's own conv kernel; the CPU
/// reference (`gdn::conv_decode`, the same tap formula) serves tests.
fn qwen38_conv_decode(x: &Tensor, weight: &Tensor, history: &Tensor) -> Result<(Tensor, Tensor)> {
    let conv_dim = x.elem_count();
    if x.device().is_cuda() {
        let (output, next) = crate::kernels::cuda::try_qwen38_causal_conv_decode(
            &x.reshape((1, 1, conv_dim))?,
            weight,
            history,
        )
        .ok_or_else(|| {
            Error::InferenceError("Qwen3.8 CUDA conv decode rejected its inputs".into())
        })?;
        return Ok((output.flatten_all()?, next));
    }
    if !x.device().is_cpu() {
        return Err(Error::InferenceError(
            "Qwen3.8 fused conv decode runs on CUDA (and the CPU reference) only".into(),
        ));
    }
    let slots = (0..gdn::CONV_TAPS - 1)
        .map(|slot| history.narrow(1, slot, 1))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let (output, next) = gdn::conv_decode(x, weight, [&slots[0], &slots[1], &slots[2]])?;
    Ok((output, next.t()?.contiguous()?))
}

/// Resolve the fused norm path for `dtype` activations on `device`, running
/// `self_check` only when everything else allows the kernel.
pub(super) fn resolve_norm_path(
    switches: &Qwen38CudaSwitches,
    device: &Device,
    allow_cpu: bool,
    dtype: DType,
    self_check: impl FnOnce() -> Result<()>,
) -> Qwen38FastPath {
    if let Some(path) = decode_gate(FUSED_NORM_ENV, switches, device, allow_cpu) {
        path
    } else if !matches!(dtype, DType::F16 | DType::BF16) {
        Qwen38FastPath::legacy(format!(
            "fused RMSNorm needs 16-bit activations, the plan computes in {dtype:?}"
        ))
    } else if !norm::supported(device, dtype) {
        Qwen38FastPath::legacy("fused RMSNorm needs SM80+")
    } else {
        match self_check() {
            Ok(()) => Qwen38FastPath::Fused,
            Err(error) => Qwen38FastPath::self_check_failed("RMSNorm", &error),
        }
    }
}

/// Compare the fused add + RMSNorm and plain RMSNorm with `norm`'s own gain
/// against the composition (`forward_retaining`) on `device`, over three
/// `dtype` rows: the rounded residual sum exactly, the outputs to rounding.
pub(super) fn norm_self_check(norm: &Qwen38RmsNorm, device: &Device, dtype: DType) -> Result<()> {
    norm_self_check_with_gain(norm, &norm.weight, device, dtype)
}

/// [`norm_self_check`] with `gain` handed to the fused kernels; the
/// composition always uses the norm's own gain.
fn norm_self_check_with_gain(
    norm: &Qwen38RmsNorm,
    gain: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    if norm.weight.dtype() != DType::F32 {
        return Err(Error::InferenceError(
            "the norm gain is not F32; nothing to fuse".into(),
        ));
    }
    let hidden = norm.weight.elem_count();
    let tensor = |seed: f32| -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self_check_wave(3 * hidden, seed, 3.0),
            (3, 1, hidden),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?)
    };
    let (residual, delta) = (tensor(1.0)?, tensor(2.0)?);
    let eps = norm.eps as f32;
    let (sum, fused) = norm::add_rms_norm(&residual, &delta, gain, eps)?;
    let expected_sum = (&residual + &delta)?;
    let (expected, _) = norm.forward_retaining(&expected_sum)?;
    compare_values(
        "fused residual sum",
        &host_values(&sum)?,
        &host_values(&expected_sum)?,
        1e-6,
        1e-6,
    )?;
    compare_values(
        "fused add+RMSNorm",
        &host_values(&fused)?,
        &host_values(&expected)?,
        0.01,
        0.02,
    )?;
    let plain = norm::rms_norm(&residual, gain, eps)?;
    let (plain_expected, _) = norm.forward_retaining(&residual)?;
    compare_values(
        "fused RMSNorm",
        &host_values(&plain)?,
        &host_values(&plain_expected)?,
        0.01,
        0.02,
    )
}

/// Deterministic self-check input: `scale * sin((i + seed) * phi)`.
fn self_check_wave(len: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * scale)
        .collect()
}

/// All values of `tensor` as host F32.
fn host_values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn normalize_conv_kernel(
    tensor: Tensor,
    expected_channels: usize,
    expected_kernel: usize,
) -> Result<Tensor> {
    match tensor.rank() {
        2 => {
            let (d0, d1) = tensor.dims2()?;
            if d0 == expected_channels && d1 == expected_kernel {
                Ok(tensor)
            } else if d0 == expected_kernel && d1 == expected_channels {
                tensor.transpose(0, 1)?.contiguous().map_err(Error::from)
            } else {
                Err(Error::ModelLoadError(format!(
                    "Unexpected Qwen3.8 conv kernel shape: ({d0}, {d1}) for expected ({expected_channels}, {expected_kernel})"
                )))
            }
        }
        3 => {
            let dims = tensor.dims();
            if dims == [expected_channels, 1, expected_kernel] {
                tensor.squeeze(1).map_err(Error::from)
            } else if dims == [expected_kernel, 1, expected_channels] {
                tensor
                    .squeeze(1)?
                    .transpose(0, 1)?
                    .contiguous()
                    .map_err(Error::from)
            } else {
                Err(Error::ModelLoadError(format!(
                    "Unexpected rank-3 Qwen3.8 conv kernel shape: {:?}",
                    dims
                )))
            }
        }
        rank => Err(Error::ModelLoadError(format!(
            "Unexpected Qwen3.8 conv kernel rank {rank}"
        ))),
    }
}

fn pre_slice_conv_kernel(conv_kernel: &Tensor, kernel_size: usize) -> Result<Vec<Tensor>> {
    let mut slices = Vec::with_capacity(kernel_size);
    for idx in 0..kernel_size {
        slices.push(conv_kernel.narrow(1, idx, 1)?);
    }
    Ok(slices)
}

fn build_mrope(
    rope_dim: usize,
    position_ids: [usize; 3],
    mrope_sections: &[usize],
    inv_freqs: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = rope_dim / 2;
    if inv_freqs.len() != half_dim {
        return Err(Error::InferenceError(format!(
            "Invalid Qwen3.8 rotary dimension {rope_dim}"
        )));
    }

    let mut temporal = vec![0f32; half_dim];
    let mut height = vec![0f32; half_dim];
    let mut width = vec![0f32; half_dim];
    for (idx, inv_freq) in inv_freqs.iter().enumerate() {
        temporal[idx] = position_ids[0] as f32 * inv_freq;
        height[idx] = position_ids[1] as f32 * inv_freq;
        width[idx] = position_ids[2] as f32 * inv_freq;
    }

    let mut interleaved = temporal.clone();
    if position_ids[0] != position_ids[1] || position_ids[0] != position_ids[2] {
        if mrope_sections.iter().sum::<usize>() != half_dim || mrope_sections.len() < 3 {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.8 multimodal RoPE sections {:?} for rotary dim {}",
                mrope_sections, rope_dim
            )));
        }

        for (offset, source, section_len) in [
            (1usize, &height, mrope_sections[1]),
            (2usize, &width, mrope_sections[2]),
        ] {
            let stop = section_len * 3;
            for idx in (offset..stop.min(half_dim)).step_by(3) {
                interleaved[idx] = source[idx];
            }
        }
    }

    // Take cos/sin of the F32 angles and cast only the results, as HF does
    // (`emb.cos()` / `emb.sin()` in fp32, then `.to(x.dtype)`). Rounding the
    // angle to the plan dtype first puts BF16's 8-bit mantissa on
    // `position * inv_freq` (at theta 1e7 the fastest pairs drift ~0.1 rad
    // within a few hundred tokens and rotate randomly by ~4K) and overflows
    // an F16 angle to inf (NaN cos/sin) past position 65504. The trigonometry
    // runs on the host because Candle compiles its Metal kernels with fast
    // math, whose F32 cos/sin miss by ~1e-4 at position 4K and ~1e-2 at 262K.
    let trig = interleaved
        .iter()
        .map(|angle| angle.cos())
        .chain(interleaved.iter().map(|angle| angle.sin()))
        .collect::<Vec<_>>();
    let trig = Tensor::from_vec(trig, (2, 1, 1, half_dim), device)?.to_dtype(dtype)?;
    Ok((trig.i(0)?, trig.i(1)?))
}

fn build_mrope_plan(
    rope_dim: usize,
    position_ids: &[[usize; 3]],
    mrope_sections: &[usize],
    inv_freqs: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<Qwen38MropePlan> {
    let sequence_len = position_ids.len();
    if rope_dim == 0 {
        return Ok(Qwen38MropePlan {
            cos: None,
            sin: None,
            sequence_len,
            rope_dim: 0,
        });
    }
    if sequence_len == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.8 cannot prepare an empty rotary plan".into(),
        ));
    }

    // Retain the established token-wise construction and concatenate only
    // after each token has produced its original [1, 1, half_dim] tensors.
    // This keeps multimodal section selection and device trigonometry exactly
    // aligned with the pre-plan implementation.
    let mut cos_tokens = Vec::with_capacity(sequence_len);
    let mut sin_tokens = Vec::with_capacity(sequence_len);
    for &position_id in position_ids {
        let (cos, sin) = build_mrope(
            rope_dim,
            position_id,
            mrope_sections,
            inv_freqs,
            device,
            dtype,
        )?;
        cos_tokens.push(cos);
        sin_tokens.push(sin);
    }
    let cos_refs: Vec<&Tensor> = cos_tokens.iter().collect();
    let sin_refs: Vec<&Tensor> = sin_tokens.iter().collect();

    Ok(Qwen38MropePlan {
        cos: Some(Tensor::cat(&cos_refs, 1)?.contiguous()?),
        sin: Some(Tensor::cat(&sin_refs, 1)?.contiguous()?),
        sequence_len,
        rope_dim,
    })
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos.unsqueeze(2)?;
    let sin = sin.unsqueeze(2)?;
    let out_first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let out_second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&out_first, &out_second], 3).map_err(Error::from)
}

fn try_apply_rope_thd(
    query_rot: &Tensor,
    key_rot: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let query_rot = rotary_emb::rope_thd(query_rot, cos, sin)?;
        let key_rot = rotary_emb::rope_thd(key_rot, cos, sin)?;
        candle_core::Result::<(Tensor, Tensor)>::Ok((query_rot, key_rot))
    }));

    match kernel_result {
        Ok(Ok((query_rot, key_rot))) => Ok(Some((query_rot, key_rot))),
        failure => {
            crate::models::shared::telemetry::report_kernel_fallback("qwen38.rope", &failure);
            Ok(None)
        }
    }
}

fn build_rope_inv_freqs(rope_dim: usize, rope_theta: f64) -> Result<Vec<f32>> {
    let half_dim = rope_dim / 2;
    let inv_freqs: Vec<f32> = (0..rope_dim)
        .step_by(2)
        .map(|idx| (1.0f64 / rope_theta.powf(idx as f64 / rope_dim as f64)) as f32)
        .collect();
    if inv_freqs.len() != half_dim {
        return Err(Error::InferenceError(format!(
            "Invalid Qwen3.8 rotary dimension {rope_dim}"
        )));
    }
    Ok(inv_freqs)
}

/// Allocate persistent runtime state with independent backing storage.
///
/// Candle tensor views can outlive a scratch-pool checkout, so state that
/// survives a forward call must never escape after its pool lease is released.
fn owned_zero_tensor(shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    Tensor::zeros(shape.to_vec(), dtype, device).map_err(Error::from)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NonFiniteCounts {
    nan: usize,
    positive_infinity: usize,
    negative_infinity: usize,
}

impl NonFiniteCounts {
    fn total(self) -> usize {
        self.nan
            .saturating_add(self.positive_infinity)
            .saturating_add(self.negative_infinity)
    }
}

/// Collect diagnostics without exposing model inputs, outputs, or token data.
/// This host readback is intentionally guarded by an opt-in environment flag
/// at inference call sites because checking every layer synchronizes the GPU.
fn non_finite_counts(tensor: &Tensor) -> Result<NonFiniteCounts> {
    let values = tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut counts = NonFiniteCounts {
        nan: 0,
        positive_infinity: 0,
        negative_infinity: 0,
    };
    for value in values {
        if value.is_nan() {
            counts.nan = counts.nan.saturating_add(1);
        } else if value == f32::INFINITY {
            counts.positive_infinity = counts.positive_infinity.saturating_add(1);
        } else if value == f32::NEG_INFINITY {
            counts.negative_infinity = counts.negative_infinity.saturating_add(1);
        }
    }
    Ok(counts)
}

fn validate_qwen38_finite_tensor(
    tensor: &Tensor,
    layer_idx: usize,
    path: &str,
    enabled: bool,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }

    let counts = non_finite_counts(tensor)?;
    if counts.total() == 0 {
        return Ok(());
    }

    Err(Error::InferenceError(format!(
        "Qwen3.8 first non-finite tensor at {path}, layer {layer_idx}: \
         {} of {} values ({} NaN, {} +Inf, {} -Inf), shape {:?}, dtype {:?}",
        counts.total(),
        tensor.elem_count(),
        counts.nan,
        counts.positive_infinity,
        counts.negative_infinity,
        tensor.dims(),
        tensor.dtype(),
    )))
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // Evaluate DeltaNet discretization in F32 with the stable identity
    // max(x, 0) + log(1 + exp(-abs(x))). The direct log(exp(x) + 1)
    // formulation overflows for valid large positive activations.
    let x = x.to_dtype(DType::F32)?;
    let positive = x.relu()?;
    let correction = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    (&positive + &correction).map_err(Error::from)
}

fn l2norm(x: &Tensor, eps: f64, fused_decode: bool) -> Result<Tensor> {
    if x.dtype() == DType::F32 {
        if fused_decode && x.dims().len() == 3 {
            let candidate = try_qwen38_l2_norm_decode(x, eps);
            record_cuda_kernel(CudaKernelPath::L2Norm, candidate.is_some());
            if let Some(result) = candidate {
                return Ok(result);
            }
        }
        let fused = try_fused_l2_norm(x, eps);
        if let Some(result) = fused {
            return Ok(result);
        }
    }

    // Fallback to standard implementation
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?)
        .map_err(Error::from)
}

fn repeat_interleave_head_states(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, heads, dim) = x.dims3()?;
    x.unsqueeze(2)?
        .broadcast_as((batch, heads, repeats, dim))?
        .reshape((batch, heads * repeats, dim))
        .map_err(Error::from)
}

fn repeat_interleave_head_states_seq(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, seq, heads, dim) = x.dims4()?;
    x.unsqueeze(3)?
        .broadcast_as((batch, seq, heads, repeats, dim))?
        .reshape((batch, seq, heads * repeats, dim))
        .map_err(Error::from)
}

fn qwen38_env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn qwen38_decode_epilogues_policy(is_cuda: bool, requested: bool) -> bool {
    is_cuda && requested
}

fn qwen38_projection_packing_policy(is_cuda: bool, requested: bool) -> bool {
    is_cuda && requested
}

fn qwen38_deltanet_decode_policy(is_cuda: bool, requested: bool) -> bool {
    is_cuda && requested
}

fn qwen38_tiled_recurrence_enabled() -> bool {
    qwen38_env_bool("IZWI_QWEN38_TILED_RECURRENCE", true)
}

fn qwen38_rope_kernel_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_QWEN38_ROPE_KERNEL")
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        });
    qwen38_rope_kernel_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn qwen38_rope_kernel_policy(
    is_metal: bool,
    is_cuda: bool,
    override_enabled: Option<bool>,
) -> bool {
    if is_metal {
        return override_enabled.unwrap_or(true);
    }
    if is_cuda {
        return override_enabled.unwrap_or(true);
    }
    false
}

fn qwen38_tiled_recurrence_tile_size_override() -> Option<usize> {
    if let Ok(raw) = std::env::var("IZWI_QWEN38_TILED_RECURRENCE_TILE_SIZE") {
        if let Ok(parsed) = raw.trim().parse::<usize>() {
            return Some(parsed.max(1));
        }
    }
    None
}

fn qwen38_tiled_recurrence_tile_size(seq_len: usize, override_size: Option<usize>) -> usize {
    if let Some(override_size) = override_size {
        return override_size.min(seq_len.max(1));
    }

    if seq_len >= 256 {
        64
    } else if seq_len >= 64 {
        32
    } else if seq_len >= 16 {
        16
    } else {
        seq_len.max(1)
    }
}

fn recurrent_gated_delta_sequence(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
) -> Result<(Tensor, Tensor)> {
    let seq_len = query.dim(1)?;
    let mut outputs = Vec::with_capacity(seq_len);
    let mut state = state;

    for idx in 0..seq_len {
        let q_t = query.narrow(1, idx, 1)?.squeeze(1)?;
        let k_t = key.narrow(1, idx, 1)?.squeeze(1)?;
        let v_t = value.narrow(1, idx, 1)?.squeeze(1)?;
        let g_t = g.narrow(1, idx, 1)?.squeeze(1)?;
        let beta_t = beta.narrow(1, idx, 1)?.squeeze(1)?;

        let (output_t, next_state) = recurrent_gated_delta(&q_t, &k_t, &v_t, &g_t, &beta_t, state)?;
        outputs.push(output_t.unsqueeze(1)?);
        state = next_state;
    }

    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let output = Tensor::cat(&output_refs, 1)?;
    Ok((output, state))
}

fn recurrent_gated_delta(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
) -> Result<(Tensor, Tensor)> {
    // Try fused Metal kernel first (for F32 on Metal devices)
    if query.dtype() == DType::F32 {
        let fused = try_fused_gated_delta_recurrent(query, key, value, g, beta, &state);
        if query.device().is_cuda() {
            record_cuda_kernel(CudaKernelPath::DeltaNetDecode, fused.is_some());
        }
        if let Some(result) = fused {
            return Ok(result);
        }
    }

    // Optimized implementation using matmul for batched reductions.
    // Shapes: query/key (1, H, Dk), value (1, H, Dv), state (1, H, Dk, Dv)
    // g (1, H), beta (1, H)
    let dim = query.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let query = (query * scale)?;
    let g = g.exp()?.reshape((1, g.dim(1)?, 1, 1))?;
    let beta = beta.reshape((1, beta.dim(1)?, 1))?;

    // Gate the state: state = state * exp(g)
    let state = state.broadcast_mul(&g)?;

    // kv_mem = sum(state * key[..., None], dim=2) = matmul(key[:, :, None, :], state).squeeze(2)
    // key: (1, H, Dk) -> (1, H, 1, Dk)  matmul  state: (1, H, Dk, Dv) -> (1, H, 1, Dv) -> squeeze -> (1, H, Dv)
    let kv_mem = key.unsqueeze(2)?.matmul(&state)?.squeeze(2)?;

    // delta = (value - kv_mem) * beta
    let delta = (value - &kv_mem)?.broadcast_mul(&beta)?;

    // state += key[:, :, :, None] * delta[:, :, None, :]  (outer product)
    // = matmul(key.unsqueeze(3), delta.unsqueeze(2)) + state
    let state = (&state + &key.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?)?;

    // output = sum(state * query[..., None], dim=2) = matmul(query[:, :, None, :], state).squeeze(2)
    let output = query.unsqueeze(2)?.matmul(&state)?.squeeze(2)?;
    Ok((output, state))
}

#[cfg(test)]
mod tests {
    use super::{
        apply_rotary_emb, build_mrope, build_mrope_plan, build_rope_inv_freqs,
        convolution_domain_v2, non_finite_counts, owned_zero_tensor,
        projection_group_geometry_compatible, qwen38_decode_epilogues_policy,
        qwen38_deltanet_decode_policy, qwen38_projection_packing_policy, qwen38_rope_kernel_policy,
        recurrent_domain_v2, repeat_interleave_head_states, repeat_interleave_head_states_seq,
        softplus, ConvHistoryStorage, ConvRingState, Linear, Qwen38GatedRmsNorm,
        Qwen38LayerRuntimeState, Qwen38LinearAttention, Qwen38Projection, Qwen38ProjectionGroup,
        Qwen38TextRuntimeState,
    };
    use crate::models::architectures::qwen38::cache::{
        CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN,
    };
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::rotary_emb;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};

    fn tensor_storage_address(tensor: &Tensor) -> usize {
        let (storage, _) = tensor.storage_and_layout();
        std::ptr::from_ref(&*storage) as usize
    }

    #[test]
    fn retained_state_access_uses_the_contracts_canonical_domain_ids() {
        assert_eq!(recurrent_domain_v2(), RECURRENT_STATE_DOMAIN);
        assert_eq!(convolution_domain_v2(), CONVOLUTION_STATE_DOMAIN);
    }

    #[test]
    fn owned_recurrent_states_do_not_alias_across_concurrent_sessions() {
        const SESSION_COUNT: usize = 8;
        let barrier = Arc::new(Barrier::new(SESSION_COUNT));
        let handles = (0..SESSION_COUNT)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    owned_zero_tensor(&[1, 2, 4, 4], DType::F32, &Device::Cpu)
                        .expect("persistent recurrent state should allocate")
                })
            })
            .collect::<Vec<_>>();
        let states = handles
            .into_iter()
            .map(|handle| handle.join().expect("allocation thread should finish"))
            .collect::<Vec<_>>();

        let storage_addresses = states
            .iter()
            .map(tensor_storage_address)
            .collect::<HashSet<_>>();
        assert_eq!(storage_addresses.len(), SESSION_COUNT);
        assert!(states.iter().all(|state| state.dims() == [1, 2, 4, 4]));
        assert!(states.iter().all(|state| {
            state
                .flatten_all()
                .and_then(|state| state.to_vec1::<f32>())
                .is_ok_and(|values| values.iter().all(|value| *value == 0.0))
        }));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn owned_recurrent_states_do_not_alias_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "owned_recurrent_states_do_not_alias_on_metal",
        ) else {
            return;
        };
        let first = owned_zero_tensor(&[1, 2, 4, 4], DType::F32, &device)
            .expect("first Metal recurrent state should allocate");
        let second = owned_zero_tensor(&[1, 2, 4, 4], DType::F32, &device)
            .expect("second Metal recurrent state should allocate");

        assert_ne!(
            tensor_storage_address(&first),
            tensor_storage_address(&second)
        );
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 32]
        );
        assert_eq!(
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 32]
        );
    }

    #[test]
    fn native_head_repeat_uses_transformers_repeat_interleave_order() {
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu)
            .expect("tensor should build");
        let repeated = repeat_interleave_head_states(&x, 2).expect("repeat should succeed");
        assert_eq!(
            repeated.to_vec3::<f32>().expect("values"),
            vec![vec![
                vec![1.0, 2.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![3.0, 4.0]
            ]]
        );

        let sequence = x.unsqueeze(1).expect("sequence");
        let repeated =
            repeat_interleave_head_states_seq(&sequence, 2).expect("repeat should succeed");
        assert_eq!(
            repeated
                .reshape((1, 1, 8))
                .unwrap()
                .to_vec3::<f32>()
                .unwrap(),
            vec![vec![vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]]]
        );
    }

    #[test]
    fn dense_projection_supports_batched_sequence_inputs() {
        use candle_nn::Module;

        let projection = Qwen38Projection::Dense(Linear::new(
            Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &Device::Cpu).unwrap(),
            None,
        ));
        let input = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], (1, 2, 2), &Device::Cpu).unwrap();
        let output = projection.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (1, 2, 3));
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0, 3.0, 5.0, 4.0, 5.0, 9.0]
        );
    }

    #[test]
    fn batched_deltanet_decode_matches_independent_scalar_rows() {
        let device = &Device::Cpu;
        let dense = |out: usize, input: usize, scale: f32| {
            let values = (0..out * input)
                .map(|index| scale * ((index % input) + 1) as f32)
                .collect::<Vec<_>>();
            Qwen38Projection::Dense(Linear::new(
                Tensor::from_vec(values, (out, input), device).unwrap(),
                None,
            ))
        };
        let mixer = Qwen38LinearAttention {
            fused_decode: false,
            qkv_z_proj: Qwen38ProjectionGroup::Separate(vec![dense(6, 4, 0.01), dense(2, 4, 0.02)]),
            alpha_beta_proj: Qwen38ProjectionGroup::Separate(vec![
                dense(1, 4, 0.03),
                dense(1, 4, -0.02),
            ]),
            dt_bias: Tensor::zeros((1, 1, 1), DType::F32, device).unwrap(),
            a: Tensor::full(-0.5f32, (1, 1, 1), device).unwrap(),
            conv_kernel: Tensor::ones((6, 1), DType::F32, device).unwrap(),
            conv_kernel_slices: Vec::new(),
            norm: Qwen38GatedRmsNorm {
                fused_decode: false,
                weight: Tensor::ones(2, DType::F32, device).unwrap(),
                eps: 1e-6,
            },
            out_proj: dense(4, 2, 0.04),
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_dim: 6,
            kernel_size: 1,
            tiled_recurrence_enabled: false,
            tiled_recurrence_tile_size_override: None,
            fused_gdn: None,
            fused_gdn_path: Default::default(),
            alpha_beta_rows: None,
            alpha_beta_gemv_path: Default::default(),
            gdn_prefill: false,
            gdn_prefill_path: Default::default(),
        };
        let input = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.3, 0.4, -0.3, 0.5, 0.1, 0.2],
            (2, 1, 4),
            device,
        )
        .unwrap();
        let empty_state = || Qwen38LayerRuntimeState::Linear {
            conv_state: None,
            recurrent_state: None,
        };
        let mut scalar_states = [empty_state(), empty_state()];
        let scalar_rows = (0..2)
            .map(|row| {
                mixer
                    .forward(
                        &input.i(row).unwrap().unsqueeze(0).unwrap(),
                        &mut scalar_states[row],
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let scalar_refs = scalar_rows.iter().collect::<Vec<_>>();
        let scalar = Tensor::cat(&scalar_refs, 0).unwrap();

        let mut batch_states = [empty_state(), empty_state()];
        let mut batch_refs = batch_states.iter_mut().collect::<Vec<_>>();
        let batched = mixer.forward_decode_batch(&input, &mut batch_refs).unwrap();
        let scalar_values = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let batch_values = batched.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (scalar, batch) in scalar_values.iter().zip(&batch_values) {
            assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
        }
        for (scalar, batch) in scalar_states.iter().zip(&batch_states) {
            let Qwen38LayerRuntimeState::Linear {
                recurrent_state: Some(scalar),
                ..
            } = scalar
            else {
                panic!("scalar row omitted recurrent state")
            };
            let Qwen38LayerRuntimeState::Linear {
                recurrent_state: Some(batch),
                ..
            } = batch
            else {
                panic!("batch row omitted recurrent state")
            };
            let scalar = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let batch = batch.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (scalar, batch) in scalar.iter().zip(&batch) {
                assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
            }
        }
    }

    #[test]
    fn packed_projection_group_matches_separate_outputs_exactly() {
        let first =
            Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &Device::Cpu).unwrap();
        let second = Tensor::from_vec(vec![2f32, -1.0, -2.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let packed = Tensor::cat(&[&first, &second], 0).unwrap();
        let separate = Qwen38ProjectionGroup::Separate(vec![
            Qwen38Projection::Dense(Linear::new(first, None)),
            Qwen38Projection::Dense(Linear::new(second, None)),
        ]);
        let packed = Qwen38ProjectionGroup::Packed {
            projection: Qwen38Projection::Dense(Linear::new(packed, None)),
            widths: vec![3, 2],
        };
        let input = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], (1, 2, 2), &Device::Cpu).unwrap();

        let separate = separate.forward(&input).unwrap();
        let packed = packed.forward(&input).unwrap();
        assert_eq!(packed.len(), separate.len());
        for (packed, separate) in packed.iter().zip(&separate) {
            assert_eq!(packed.dims(), separate.dims());
            assert_eq!(
                packed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                separate.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn projection_packing_is_cuda_only_with_explicit_opt_out() {
        assert!(!qwen38_projection_packing_policy(false, false));
        assert!(!qwen38_projection_packing_policy(false, true));
        assert!(!qwen38_projection_packing_policy(true, false));
        assert!(qwen38_projection_packing_policy(true, true));
    }

    #[test]
    fn decode_epilogues_are_cuda_only_with_explicit_opt_out() {
        assert!(!qwen38_decode_epilogues_policy(false, false));
        assert!(!qwen38_decode_epilogues_policy(false, true));
        assert!(!qwen38_decode_epilogues_policy(true, false));
        assert!(qwen38_decode_epilogues_policy(true, true));
    }

    #[test]
    fn deltanet_decode_specialization_is_cuda_only_with_explicit_opt_out() {
        assert!(!qwen38_deltanet_decode_policy(false, false));
        assert!(!qwen38_deltanet_decode_policy(false, true));
        assert!(!qwen38_deltanet_decode_policy(true, false));
        assert!(qwen38_deltanet_decode_policy(true, true));
    }

    #[test]
    fn projection_group_rejects_incompatible_inner_dimensions() {
        assert!(projection_group_geometry_compatible(&[
            ("q", [3, 8]),
            ("k", [2, 8]),
            ("v", [2, 8]),
        ]));
        assert!(!projection_group_geometry_compatible(&[
            ("q", [3, 8]),
            ("k", [2, 4]),
        ]));
        assert!(!projection_group_geometry_compatible(&[]));
    }

    #[test]
    fn q8_projection_supports_model_rank_batched_sequence_inputs() {
        use candle_nn::Module;

        let mut weights = vec![1.0_f32; 32];
        weights.extend(std::iter::repeat_n(2.0, 32));
        weights.extend(std::iter::repeat_n(-1.0, 32));
        let weight = Tensor::from_vec(weights, (3, 32), &Device::Cpu).unwrap();
        let quantized = QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap();
        let projection = Qwen38Projection::Quantized(QMatMul::QTensor(Arc::new(quantized)));
        let input = Tensor::ones((2, 3, 32), DType::F32, &Device::Cpu).unwrap();
        let output = projection.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (2, 3, 3));
        for batch in output.to_vec3::<f32>().unwrap() {
            for token in batch {
                assert!((token[0] - 32.0).abs() < 0.5, "{} != 32", token[0]);
                assert!((token[1] - 64.0).abs() < 0.5, "{} != 64", token[1]);
                assert!((token[2] + 32.0).abs() < 0.5, "{} != -32", token[2]);
            }
        }
    }

    #[test]
    fn native_projection_inner_dimensions_satisfy_q8_block_geometry() {
        // Every projection K dimension is one of the model width, MLP width,
        // or attention/DeltaNet output width. The loader also validates each
        // concrete tensor before quantizing it.
        for inner_dim in [5_120_usize, 17_408, 6_144] {
            assert!(inner_dim.is_multiple_of(GgmlDType::Q8_0.block_size()));
        }
    }

    #[test]
    fn compact_conv_ring_drops_full_prefill_backing() {
        let backing = Tensor::zeros((1, 40, 32), DType::F32, &Device::Cpu).unwrap();
        let mut slots = Vec::new();
        for token_idx in 37..40 {
            slots.push(backing.i((0, token_idx)).unwrap().reshape((32, 1)).unwrap());
        }
        let mut state = Qwen38TextRuntimeState {
            layers: vec![Qwen38LayerRuntimeState::Linear {
                conv_state: Some(ConvRingState::from_slots(slots)),
                recurrent_state: None,
            }],
        };

        let retained_prefill_bytes = state.allocated_session_bytes().unwrap();
        assert!(retained_prefill_bytes >= 40 * 32 * 4);

        let Qwen38LayerRuntimeState::Linear { conv_state, .. } = &mut state.layers[0] else {
            unreachable!("test state is linear")
        };
        conv_state
            .as_mut()
            .unwrap()
            .compact_owned()
            .expect("compaction should succeed");

        assert_eq!(state.allocated_session_bytes(), Some(3 * 32 * 4));
    }

    #[test]
    fn conv_ring_decode_push_reuses_one_token_projection_backing() {
        let slots = (0..3)
            .map(|_| Tensor::zeros((32, 1), DType::F32, &Device::Cpu).unwrap())
            .collect();
        let mut ring = ConvRingState::from_slots(slots);
        let projection = Tensor::zeros((1, 1, 32), DType::F32, &Device::Cpu).unwrap();
        let current = projection.i((0, 0)).unwrap().reshape((32, 1)).unwrap();
        let projection_storage = tensor_storage_address(&current);

        ring.push_decode(&current)
            .expect("ring push should succeed");
        let ConvHistoryStorage::Ring { slots, .. } = &ring.storage else {
            panic!("generic push should retain ring representation")
        };
        assert_eq!(tensor_storage_address(&slots[0]), projection_storage);
        drop(current);
        drop(projection);

        let state = Qwen38TextRuntimeState {
            layers: vec![Qwen38LayerRuntimeState::Linear {
                conv_state: Some(ring),
                recurrent_state: None,
            }],
        };
        assert_eq!(state.allocated_session_bytes(), Some(3 * 32 * 4));
    }

    /// A projection in a foreign dtype must fail at the ring, leaving it
    /// untouched, instead of poisoning it with a mixed-dtype slot that a later
    /// `cat` or multiply fails on, whichever storage form the ring is in.
    #[test]
    fn conv_ring_decode_push_rejects_dtype_drift() {
        let slots = (0..3)
            .map(|_| Tensor::zeros((4, 1), DType::F32, &Device::Cpu).unwrap())
            .collect();
        let staged = Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap();
        for mut ring in [
            ConvRingState::from_slots(slots),
            ConvRingState::from_staged(staged).unwrap(),
        ] {
            for dtype in [DType::BF16, DType::F16] {
                let current = Tensor::ones((4, 1), dtype, &Device::Cpu).unwrap();
                let error = ring.push_decode(&current).unwrap_err().to_string();
                assert!(error.contains("dtype drift"), "{error}");
            }
            let history = ring.contiguous_history().unwrap();
            assert_eq!(history.dtype(), DType::F32);
            assert_eq!(history.to_vec2::<f32>().unwrap(), vec![vec![0.0; 3]; 4]);

            let current = Tensor::ones((4, 1), DType::F32, &Device::Cpu).unwrap();
            ring.push_decode(&current)
                .expect("a matching projection is accepted");
            assert_eq!(
                ring.contiguous_history().unwrap().to_vec2::<f32>().unwrap(),
                vec![vec![0.0, 0.0, 1.0]; 4]
            );
        }
    }

    #[test]
    fn contiguous_conv_history_stages_without_new_storage() {
        let history =
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 10.0, 20.0, 30.0], (2, 3), &Device::Cpu).unwrap();
        let history_storage = tensor_storage_address(&history);
        let mut state = ConvRingState::from_slots(vec![
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
        ]);

        state
            .replace_contiguous(history, 3)
            .expect("specialized history should be accepted");
        let staged = state
            .staged_tensor()
            .expect("contiguous state should stage");
        let next_decode = state
            .contiguous_history()
            .expect("contiguous state should feed decode");

        assert_eq!(staged.dims(), &[2, 3]);
        assert_eq!(tensor_storage_address(&staged), history_storage);
        assert_eq!(tensor_storage_address(&next_decode), history_storage);
        assert_eq!(
            staged.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]
        );
    }

    #[test]
    fn contiguous_conv_history_accounts_for_its_complete_kernel_backing() {
        // The CUDA custom op packs the current output before its next-history
        // slice. Retaining the slice deliberately avoids a device copy, and
        // accounting must still charge the complete backing allocation.
        let packed = Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap();
        let history = packed.narrow(0, 2, 6).unwrap().reshape((2, 3)).unwrap();
        let state = Qwen38TextRuntimeState {
            layers: vec![Qwen38LayerRuntimeState::Linear {
                conv_state: Some(ConvRingState {
                    storage: ConvHistoryStorage::Contiguous(history),
                }),
                recurrent_state: None,
            }],
        };

        assert_eq!(state.allocated_session_bytes(), Some(8 * 4));
    }

    #[test]
    fn conv_history_restore_accepts_canonical_and_legacy_representations() {
        let canonical =
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 10.0, 20.0, 30.0], (2, 3), &Device::Cpu).unwrap();
        let restored = ConvRingState::from_staged(canonical.clone()).unwrap();
        assert_eq!(
            restored.staged_tensor().unwrap().to_vec2::<f32>().unwrap(),
            canonical.to_vec2::<f32>().unwrap()
        );

        let legacy = canonical.transpose(0, 1).unwrap().unsqueeze(2).unwrap();
        let restored = ConvRingState::from_staged(legacy).unwrap();
        assert_eq!(
            restored.staged_tensor().unwrap().to_vec2::<f32>().unwrap(),
            canonical.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn persistent_zero_states_have_independent_storage() {
        let first = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let second = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let state = Qwen38TextRuntimeState {
            layers: vec![
                Qwen38LayerRuntimeState::Linear {
                    conv_state: None,
                    recurrent_state: Some(first),
                },
                Qwen38LayerRuntimeState::Linear {
                    conv_state: None,
                    recurrent_state: Some(second),
                },
            ],
        };

        assert_eq!(state.allocated_session_bytes(), Some(2 * 2 * 3 * 4 * 4));
    }

    #[test]
    fn softplus_is_f32_and_stable_for_large_magnitudes() {
        let input = Tensor::from_vec(vec![-1000f32, 0.0, 1000.0], 3, &Device::Cpu).unwrap();
        let output = softplus(&input).expect("softplus");
        assert_eq!(output.dtype(), DType::F32);
        let values = output.to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values[0].abs() < 1e-6);
        assert!((values[1] - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((values[2] - 1000.0).abs() < 1e-4);
    }

    #[test]
    fn non_finite_diagnostics_count_without_exposing_values() {
        let tensor = Tensor::from_vec(
            vec![0.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            4,
            &Device::Cpu,
        )
        .unwrap();
        let counts = non_finite_counts(&tensor).unwrap();
        assert_eq!(counts.nan, 1);
        assert_eq!(counts.positive_infinity, 1);
        assert_eq!(counts.negative_infinity, 1);
        assert_eq!(counts.total(), 3);
    }

    #[test]
    fn build_mrope_uses_half_dim_layout_and_sections() {
        let (cos, sin) = build_mrope(
            12,
            [3, 5, 7],
            &[2, 2, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &Device::Cpu,
            DType::F32,
        )
        .expect("mrope should build");

        assert_eq!(cos.dims(), &[1, 1, 6]);
        assert_eq!(sin.dims(), &[1, 1, 6]);

        let cos_vals = cos.to_vec3::<f32>().expect("cos values");
        let sin_vals = sin.to_vec3::<f32>().expect("sin values");
        let expected = [3.0f32, 5.0, 7.0, 3.0, 5.0, 7.0];
        for (idx, expected_theta) in expected.iter().enumerate() {
            assert!((cos_vals[0][0][idx] - expected_theta.cos()).abs() < 1e-5);
            assert!((sin_vals[0][0][idx] - expected_theta.sin()).abs() < 1e-5);
        }
    }

    /// Half-precision plans (BF16 CUDA, F16 Metal) must round cos/sin, never
    /// the angle: at long positions a BF16 angle is phase noise and an F16
    /// angle overflows to inf. Each `(dtype, tolerance)` plan built on
    /// `device` must match the cos/sin of the exact F32 angle.
    fn assert_mrope_plan_rounds_only_cos_sin(device: &Device, plans: &[(DType, f64)]) {
        // Qwen3.8-27B rotary geometry: 64 rotary dims, theta 1e7.
        let rope_dim = 64;
        let inv_freqs = build_rope_inv_freqs(rope_dim, 10_000_000.0).unwrap();
        let positions = [4_096usize, 30_000, 70_000, 262_143];
        let position_ids = positions.map(|position| [position; 3]);
        for &(dtype, tolerance) in plans {
            let plan = build_mrope_plan(
                rope_dim,
                &position_ids,
                &[11, 11, 10],
                &inv_freqs,
                device,
                dtype,
            )
            .unwrap();
            for (values, trig) in [
                (plan.cos.as_ref().unwrap(), f64::cos as fn(f64) -> f64),
                (plan.sin.as_ref().unwrap(), f64::sin),
            ] {
                assert_eq!(values.dtype(), dtype);
                let values = values
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (index, value) in values.iter().enumerate() {
                    let (row, pair) = (index / inv_freqs.len(), index % inv_freqs.len());
                    // The exact F32 angle HF and the plan both compute.
                    let angle = positions[row] as f32 * inv_freqs[pair];
                    let expected = trig(f64::from(angle));
                    assert!(
                        (f64::from(*value) - expected).abs() < tolerance,
                        "{dtype:?} position {} pair {pair}: {value} vs {expected}",
                        positions[row]
                    );
                }
            }
        }
    }

    #[test]
    fn mrope_plan_keeps_long_position_angles_in_f32() {
        assert_mrope_plan_rounds_only_cos_sin(
            &Device::Cpu,
            &[
                (DType::F32, 1e-5),
                // Half an ulp near 1, plus slack: 2^-9 for BF16, 2^-12 for F16.
                (DType::BF16, 4e-3),
                (DType::F16, 1e-3),
            ],
        );
    }

    /// The Metal plan's tables must match too: Candle's fast-math Metal
    /// cos/sin would already miss the F32 tolerance at position 4096.
    #[cfg(feature = "metal")]
    #[test]
    fn mrope_plan_keeps_long_position_angles_in_f32_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "mrope_plan_keeps_long_position_angles_in_f32_on_metal",
        ) else {
            return;
        };
        assert_mrope_plan_rounds_only_cos_sin(&device, &[(DType::F32, 1e-5), (DType::F16, 1e-3)]);
    }

    #[test]
    fn shared_mrope_plan_matches_tokenwise_construction_at_long_positions() {
        let positions = [
            [262_140, 262_140, 262_140],
            [262_141, 131_071, 65_535],
            [262_143, 262_142, 262_141],
        ];
        let sections = [2, 2, 2];
        let inv_freqs = build_rope_inv_freqs(12, 10_000_000.0).expect("inverse frequencies");
        let plan = build_mrope_plan(
            12,
            &positions,
            &sections,
            &inv_freqs,
            &Device::Cpu,
            DType::F32,
        )
        .expect("shared mrope plan");

        let mut legacy_cos = Vec::new();
        let mut legacy_sin = Vec::new();
        for position in positions {
            let (cos, sin) = build_mrope(
                12,
                position,
                &sections,
                &inv_freqs,
                &Device::Cpu,
                DType::F32,
            )
            .expect("tokenwise mrope");
            legacy_cos.push(cos);
            legacy_sin.push(sin);
        }
        let legacy_cos_refs = legacy_cos.iter().collect::<Vec<_>>();
        let legacy_sin_refs = legacy_sin.iter().collect::<Vec<_>>();
        let legacy_cos = Tensor::cat(&legacy_cos_refs, 1).expect("legacy cosine span");
        let legacy_sin = Tensor::cat(&legacy_sin_refs, 1).expect("legacy sine span");

        assert_eq!(plan.sequence_len, positions.len());
        assert_eq!(plan.rope_dim, 12);
        assert_eq!(
            plan.cos
                .expect("plan cosine")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            legacy_cos.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            plan.sin
                .expect("plan sine")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            legacy_sin.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn shared_mrope_plan_manual_matches_rope_kernel_at_long_positions() {
        let positions = [
            [262_141, 262_141, 262_141],
            [262_142, 131_071, 65_535],
            [262_143, 262_142, 262_141],
        ];
        let inv_freqs = build_rope_inv_freqs(8, 10_000_000.0).expect("inverse frequencies");
        let plan = build_mrope_plan(
            8,
            &positions,
            &[2, 1, 1],
            &inv_freqs,
            &Device::Cpu,
            DType::F32,
        )
        .expect("shared mrope plan");
        let x = Tensor::from_vec(
            (0..(positions.len() * 2 * 8))
                .map(|value| value as f32 / 31.0 - 0.5)
                .collect::<Vec<_>>(),
            (1, positions.len(), 2, 8),
            &Device::Cpu,
        )
        .expect("rotary input");
        let cos = plan.cos.as_ref().expect("plan cosine");
        let sin = plan.sin.as_ref().expect("plan sine");
        let manual = apply_rotary_emb(&x, cos, sin).expect("manual rotary");
        let kernel = rotary_emb::rope_thd(&x, cos, sin).expect("rotary kernel");
        let manual = manual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let kernel = kernel.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (manual, kernel) in manual.iter().zip(kernel.iter()) {
            assert!((manual - kernel).abs() < 1e-5);
        }
    }

    #[test]
    fn rotary_emb_manual_matches_rope_thd() {
        let x = Tensor::from_vec(
            (0..(3 * 2 * 8))
                .map(|v| v as f32 / 10.0)
                .collect::<Vec<_>>(),
            (1, 3, 2, 8),
            &Device::Cpu,
        )
        .expect("x");
        let theta = [
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];
        let cos = Tensor::from_vec(
            theta.iter().map(|v| v.cos()).collect::<Vec<_>>(),
            (1, 3, 4),
            &Device::Cpu,
        )
        .expect("cos");
        let sin = Tensor::from_vec(
            theta.iter().map(|v| v.sin()).collect::<Vec<_>>(),
            (1, 3, 4),
            &Device::Cpu,
        )
        .expect("sin");

        let manual = apply_rotary_emb(&x, &cos, &sin).expect("manual");
        let kernel = rotary_emb::rope_thd(&x, &cos, &sin).expect("kernel");

        let manual_vals = manual
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("manual vals");
        let kernel_vals = kernel
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("kernel vals");
        assert_eq!(manual_vals.len(), kernel_vals.len());
        for (manual, kernel) in manual_vals.iter().zip(kernel_vals.iter()) {
            assert!((manual - kernel).abs() < 1e-5);
        }
    }

    #[test]
    fn qwen38_cuda_rope_kernel_defaults_on_with_explicit_rollback() {
        assert!(qwen38_rope_kernel_policy(true, false, None));
        assert!(qwen38_rope_kernel_policy(false, true, None));
        assert!(qwen38_rope_kernel_policy(false, true, Some(true)));
        assert!(!qwen38_rope_kernel_policy(true, false, Some(false)));
        assert!(!qwen38_rope_kernel_policy(false, false, Some(true)));
    }

    fn synthetic_decode_state(
        device: &Device,
        layer_count: usize,
        num_v_heads: usize,
    ) -> Qwen38TextRuntimeState {
        let mut layers = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            if (layer_idx + 1).is_multiple_of(4) {
                layers.push(Qwen38LayerRuntimeState::Full);
            } else {
                let conv_width = num_v_heads * 2;
                layers.push(Qwen38LayerRuntimeState::Linear {
                    conv_state: Some(ConvRingState::from_slots(
                        (0..3)
                            .map(|_| Tensor::zeros((conv_width, 1), DType::F32, device).unwrap())
                            .collect(),
                    )),
                    recurrent_state: Some(
                        Tensor::zeros((1, num_v_heads, 2, 2), DType::F32, device).unwrap(),
                    ),
                });
            }
        }
        Qwen38TextRuntimeState { layers }
    }

    fn advance_synthetic_decode_state(
        state: &mut Qwen38TextRuntimeState,
        device: &Device,
        num_v_heads: usize,
    ) {
        for layer in &mut state.layers {
            match layer {
                Qwen38LayerRuntimeState::Linear {
                    conv_state: Some(conv_state),
                    recurrent_state,
                } => {
                    let conv_width = num_v_heads * 2;
                    let projection = Tensor::zeros((1, 1, conv_width), DType::F32, device).unwrap();
                    let current = projection
                        .i((0, 0))
                        .unwrap()
                        .reshape((conv_width, 1))
                        .unwrap();
                    conv_state.push_decode(&current).unwrap();
                    *recurrent_state =
                        Some(Tensor::zeros((1, num_v_heads, 2, 2), DType::F32, device).unwrap());
                }
                Qwen38LayerRuntimeState::Full => {}
                _ => panic!("synthetic state must initialize every linear cache"),
            }
        }
    }

    #[test]
    fn persistent_decode_storage_plateaus_for_small_and_large_topologies() {
        for (layer_count, num_v_heads) in [(24, 16), (32, 32)] {
            let mut state = synthetic_decode_state(&Device::Cpu, layer_count, num_v_heads);
            let baseline = state.allocated_session_bytes().unwrap();

            for _ in 0..96 {
                advance_synthetic_decode_state(&mut state, &Device::Cpu, num_v_heads);
                assert_eq!(
                    state.allocated_session_bytes(),
                    Some(baseline),
                    "{layer_count}-layer/{num_v_heads}V state retained growing backing storage"
                );
            }
        }
    }
}

#[cfg(test)]
mod fast_path_tests;
#[cfg(test)]
mod verification_tests;

#[cfg(test)]
mod golden_tests;

#[cfg(test)]
mod fp8_residency_tests;

#[cfg(test)]
mod graph_tests;
