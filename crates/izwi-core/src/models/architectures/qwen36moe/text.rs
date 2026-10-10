//! Qwen3.6-MoE hybrid trunk (Gated DeltaNet + gated full attention). Forked
//! from the dense `qwen35` trunk and owned by `qwen36moe` alone.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{ops, rotary_emb, Embedding};
use candle_core::quantized::QMatMul;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::kernels::cuda::gdn::{self, GdnDecodeSpec};
use crate::kernels::cuda::norm;
use crate::kernels::cuda::rope::{self, MropePosition, QkNormRopeSpec};
use crate::kernels::{
    try_fused_gated_delta_recurrent, try_fused_gated_rms_norm, try_fused_l2_norm,
    try_fused_silu_mul, try_qwen35_causal_conv_sequence, try_tiled_deltanet_recurrence,
};
use crate::kv::v2::{StateComponentId, StateDomainId};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::memory::accounting::{
    deep_copy_tensor_storage, TensorStorageAccounting,
};
use crate::models::shared::telemetry::{
    record_prefill_sequence_span, record_rope_kernel, record_rope_manual,
};
use crate::models::shared::weights::gguf::GgufLoader;

use super::cache::{CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN};
use super::exec::Qwen36TextConfig;
use crate::models::architectures::qwen36moe::fast_path::{
    compare_values, legacy_requested, Qwen36FusedPath,
};
use crate::models::architectures::qwen36moe::sparse::Qwen36MoeSparseMlp;

pub struct Qwen36TextModel {
    device: Device,
    token_embeddings: Embedding,
    layers: Vec<Qwen36Layer>,
    output_norm: Qwen36RmsNorm,
    output: Qwen36Projection,
    finite_diagnostics_enabled: bool,
    fused_norm_path: Qwen36FusedPath,
    fp8_gemv_path: Qwen36FusedPath,
}

/// One replay-prefill span's outputs: every row's pre-norm hidden (the MTP
/// pair rebuild consumes them) plus the optional final-row logits.
pub(crate) struct Qwen36PrefillSpanOutput {
    pub hidden_states: Tensor,
    pub logits: Option<Tensor>,
}

#[derive(Clone)]
pub struct Qwen36TextRuntimeState {
    layers: Vec<Qwen36LayerRuntimeState>,
}

impl Qwen36TextRuntimeState {
    /// Backing allocations retained by the per-request text runtime state.
    ///
    /// This intentionally excludes model-global caches (notably full-attention
    /// RoPE windows), so callers requiring a complete scheduler claim must keep
    /// Qwen3.5 fail-closed until those caches are independently bounded.
    pub fn allocated_session_bytes(&self) -> Option<u64> {
        let mut accounting = TensorStorageAccounting::default();
        self.account_storage(&mut accounting)?;
        Some(accounting.bytes())
    }

    pub(crate) fn account_storage(&self, accounting: &mut TensorStorageAccounting) -> Option<()> {
        for layer in &self.layers {
            match layer {
                Qwen36LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    if let Some(conv_state) = conv_state {
                        for slot in &conv_state.slots {
                            accounting.add_tensor(slot)?;
                        }
                    }
                    if let Some(recurrent_state) = recurrent_state {
                        accounting.add_tensor(recurrent_state)?;
                    }
                }
                Qwen36LayerRuntimeState::Full => {}
            }
        }
        Some(())
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
            Error::InferenceError("Qwen3.5 recurrent state is missing its convolution peer".into())
        })?;
        let convolution = convolution.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 convolution state is missing its recurrent peer".into())
        })?;
        let mut recurrent_components = recurrent.components.iter();
        let mut convolution_components = convolution.components.iter();
        for layer in &mut self.layers {
            let Qwen36LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent = recurrent_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 recurrent component coverage is incomplete".into())
            })?;
            let convolution = convolution_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 convolution component coverage is incomplete".into())
            })?;
            let recurrent_tensor = recurrent.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 recurrent component is absent".into())
            })?;
            let convolution_tensor = convolution.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 convolution component is absent".into())
            })?;
            *recurrent_state = Some(recurrent_tensor.clone());
            let history_len = convolution_tensor.dim(0)?;
            let slots = (0..history_len)
                .map(|index| convolution_tensor.i(index).map_err(Error::from))
                .collect::<Result<Vec<_>>>()?;
            *conv_state = Some(ConvRingState { slots, next_idx: 0 });
        }
        if recurrent_components.next().is_some() || convolution_components.next().is_some() {
            return Err(Error::InferenceError(
                "Qwen3.5 tensor state has components for unknown layers".into(),
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
            let Qwen36LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent_tensor = recurrent_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 recurrent state was not initialized".into())
            })?;
            let ring = conv_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 convolution state was not initialized".into())
            })?;
            if ring.slots.is_empty() || ring.next_idx >= ring.slots.len() {
                return Err(Error::InferenceError(
                    "Qwen3.5 convolution ring is invalid at the physical boundary".into(),
                ));
            }
            let ordered = ring.ordered_slots().collect::<Vec<_>>();
            let ring_tensor = Tensor::stack(&ordered, 0)?;
            let component = u32::try_from(recurrent.len() + 1)
                .map_err(|_| Error::InvalidInput("Qwen3.5 state component overflow".into()))?;
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
            if let Qwen36LayerRuntimeState::Linear {
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
    slots: Vec<Tensor>,
    next_idx: usize,
}

impl ConvRingState {
    /// History slots oldest first — the logical order every serialized form
    /// (arena staging, MTP rollback snapshots) stores, so a restore can
    /// rebuild the ring with `next_idx = 0`.
    fn ordered_slots(&self) -> impl Iterator<Item = &Tensor> {
        let len = self.slots.len();
        (0..len).map(move |offset| &self.slots[(self.next_idx + offset) % len])
    }

    /// Move every logical history slot into independent fixed-history storage.
    ///
    /// Sequence-prefill slots are views into the entire projected token span.
    /// Keeping those views in runtime state would retain the full projection
    /// instead of the fixed `kernel_size - 1` history required by the conv.
    fn compact_owned(&mut self) -> Result<()> {
        if self.slots.is_empty() {
            return Ok(());
        }

        self.slots = self
            .slots
            .iter()
            .map(deep_copy_tensor_storage)
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(())
    }

    /// Retain the current one-token projection as the newest decode slot.
    ///
    /// Decode projects exactly one token, so this view cannot retain a
    /// sequence-sized backing tensor. Sequence prefill detaches its final fixed
    /// history once at the sequence boundary instead.
    fn push_decode(&mut self, current: &Tensor) -> Result<()> {
        if self.slots.is_empty() || self.next_idx >= self.slots.len() {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.5 convolution ring: slots={}, next_idx={}",
                self.slots.len(),
                self.next_idx
            )));
        }
        // The ring holds the conv state arena's F32 dtype for the whole
        // session; an incoming projection in another dtype would poison the
        // ring with mixed-dtype slots that later `cat`/multiply fail on.
        let slot_dtype = self.slots[self.next_idx].dtype();
        if current.dtype() != slot_dtype {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 convolution ring dtype drift: ring {:?}, incoming {:?}",
                slot_dtype,
                current.dtype()
            )));
        }
        self.slots[self.next_idx] = current.clone();
        self.next_idx = (self.next_idx + 1) % self.slots.len();
        Ok(())
    }
}

#[derive(Clone)]
enum Qwen36LayerRuntimeState {
    Linear {
        conv_state: Option<ConvRingState>,
        recurrent_state: Option<Tensor>,
    },
    Full,
}

struct Qwen36Layer {
    attn_norm: Qwen36RmsNorm,
    mixer: Qwen36Mixer,
    post_attention_norm: Qwen36RmsNorm,
    ffn: Qwen36FeedForward,
}

enum Qwen36Mixer {
    Linear(Qwen36LinearAttention),
    Full(Qwen36FullAttention),
}

pub(crate) struct Qwen36Mlp {
    gate: Qwen36Projection,
    up: Qwen36Projection,
    down: Qwen36Projection,
}

pub(crate) struct Qwen36FullAttention {
    q_proj: Qwen36Projection,
    k_proj: Qwen36Projection,
    v_proj: Qwen36Projection,
    o_proj: Qwen36Projection,
    q_norm: Qwen36RmsNorm,
    k_norm: Qwen36RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    mrope_sections: Vec<usize>,
    rope_kernel_enabled: bool,
    rope_inv_freqs: Vec<f32>,
    /// Fused q/k norm + RoPE for single-token decode with its device-resident
    /// inverse frequencies, when resolved at load.
    fused_qk: Option<(QkNormRopeSpec, Tensor)>,
    fused_qk_path: Qwen36FusedPath,
    /// `q_proj` + `k_proj` + `v_proj` packed for one decode GEMV.
    packed_qkv: Option<Qwen36Projection>,
}

struct Qwen36LinearAttention {
    qkv_proj: Qwen36Projection,
    gate_proj: Qwen36Projection,
    beta_proj: Qwen36Projection,
    alpha_proj: Qwen36Projection,
    dt_bias: Tensor,
    a: Tensor,
    conv_kernel: Tensor,
    conv_kernel_slices: Vec<Tensor>,
    norm: Qwen36GatedRmsNorm,
    out_proj: Qwen36Projection,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_dim: usize,
    kernel_size: usize,
    v_head_order: Qwen36LinearVHeadOrder,
    tiled_recurrence_enabled: bool,
    tiled_recurrence_tile_size_override: Option<usize>,
    /// Fused single-token decode kernels, when resolved at load (see
    /// [`Qwen36LinearAttention::resolve_fused_decode`]).
    fused_decode: Option<GdnDecodeSpec>,
    fused_decode_path: Qwen36FusedPath,
    /// `qkv_proj` + `gate_proj` (and `beta_proj` + `alpha_proj`) packed for
    /// one GEMV each on the fused decode path.
    packed_in: Option<Qwen36Projection>,
    packed_beta_alpha: Option<Qwen36Projection>,
}

/// Environment switch for the fused DeltaNet decode (`legacy`/`off`/`0`
/// keeps the Candle op chain).
const FUSED_DECODE_ENV: &str = "IZWI_QWEN36_FUSED_DECODE";

/// Whether the fused decode kernels run on `device` in production (CUDA, and
/// Metal when built with it); `allow_cpu` lets tests use the portable paths.
fn fused_device(device: &Device, allow_cpu: bool) -> bool {
    device.is_cuda()
        || (cfg!(feature = "metal") && device.is_metal())
        || (allow_cpu && device.is_cpu())
}

/// 16-bit activation dtype of the native plan for the fused norms: F16 on
/// Metal, BF16 elsewhere.
fn norm_dtype(device: &Device) -> DType {
    if device.is_metal() {
        DType::F16
    } else {
        DType::BF16
    }
}

/// How a checkpoint orders the DeltaNet value heads relative to the shared
/// key heads when `num_v_heads > num_k_heads` (`r = num_v_heads /
/// num_k_heads` value heads per key head).
///
/// HF safetensors store value heads GROUPED by key head
/// (`[K0v0..K0v{r-1}, K1v0, ...]`) and expand q/k with `repeat_interleave`:
/// value head `j` reads key head `j / r`. llama.cpp's Qwen3.5/3.6 conversion
/// (`_LinearAttentionVReorderBase`) permutes every value-head-indexed tensor
/// into TILED order (`[K0v0, K1v0, ..., K0v1, ...]`) so `ggml_repeat` can do
/// the expansion: value head `j` reads key head `j % num_k_heads`. The two
/// are the same model under a value-side permutation; what matters is that
/// the expansion matches the checkpoint's storage order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen36LinearVHeadOrder {
    /// llama.cpp-converted GGUF.
    Tiled,
    /// Native HF safetensors.
    Grouped,
}

struct Qwen36GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

/// RMS norm over the shared trunk's own weight tensor.
///
/// `quantized_nn::RmsNorm` always dequantizes its weight to F32, which
/// breaks every plan whose activations are not F32: candle's rmsnorm op
/// requires x and weight in the same dtype on all backends, and a mixed
/// pair dies inside the op (on CUDA through Map2's "dtype mismatch in
/// binary op"). Sources therefore hand over the weight already materialized
/// in the plan's activation dtype (BF16 CUDA, F16 Metal, F32 CPU/GGUF).
///
/// A source may instead hand over an F32 weight under a lower-precision plan
/// when the weight carries a load-time transform whose rounding matters (the
/// native checkpoint's zero-centered `1 + w` gains): the norm then runs in F32
/// and casts back, which is exactly HF `Qwen3_5MoeRMSNorm`'s
/// `(norm(x.float()) * (1 + w.float())).type_as(x)`.
#[derive(Debug, Clone)]
pub(crate) struct Qwen36RmsNorm {
    weight: Tensor,
    eps: f64,
    /// Fused cast-free kernel for 16-bit activations with this F32 gain,
    /// enabled by [`Qwen36TextModel::resolve_fused_norms`].
    fused: bool,
}

impl Qwen36RmsNorm {
    pub(crate) fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            eps,
            fused: false,
        }
    }

    fn fused_applies(&self, x: &Tensor) -> bool {
        self.fused
            && self.weight.dtype() == DType::F32
            && matches!(x.dtype(), DType::F16 | DType::BF16)
    }

    /// `(residual + delta, norm(residual + delta))`, fused into one launch
    /// when enabled; the sum is rounded to the activation dtype either way.
    fn add_forward(&self, residual: &Tensor, delta: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.fused_applies(residual)
            && delta.dtype() == residual.dtype()
            && delta.dims() == residual.dims()
        {
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

    #[cfg(test)]
    pub(crate) fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Qwen36RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if self.fused_applies(x) {
            return norm::rms_norm(x, &self.weight, self.eps as f32);
        }
        if x.dtype() == self.weight.dtype() {
            return candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32);
        }
        candle_nn::ops::rms_norm(
            &x.to_dtype(self.weight.dtype())?,
            &self.weight,
            self.eps as f32,
        )?
        .to_dtype(x.dtype())
    }
}

/// Sparse-expert feed-forward geometry shared by every layer of a
/// sparse-MoE variant of the Qwen3.5 hybrid trunk (Qwen3.5-35B-A3B: 256
/// routed experts, 8 active, plus one always-on shared expert).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen36MoeFfnGeometry {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

/// Checkpoint-format seam for the shared Qwen3.5 hybrid trunk. The dense
/// GGUF family and the qwen36moe loaders (native block-FP8 safetensors and
/// the synthetic GGUF fixture) build the identical model through this
/// interface; only tensor naming, residency, and MoE weight layout differ.
///
/// Names handed to the source are the logical GGUF-style names
/// (`token_embd.weight`, `blk.{i}.attn_q.weight`, ...); implementations
/// translate to their own checkpoint layout internally.
/// Persistent residency form of one trunk projection. `Quantized` keeps
/// candle quantized-matmul residency (GGUF tensors, packed Q8_0 requants,
/// expanded F16/BF16); `CompactFp8` keeps the checkpoint's raw block-FP8
/// bytes plus F32 block scales resident and decodes per GEMM inside the CUDA
/// fp8 projection kernel — no expanded persistent weight tensor exists.
#[derive(Clone)]
pub(crate) enum Qwen36Projection {
    Quantized(QMatMul),
    CompactFp8 { weights: Tensor, scales: Tensor },
}

/// Activation dtype a self-check probe should use for projections like
/// `projection`: a dense weight's own dtype (its matmul requires it),
/// otherwise BF16 on CUDA (the native plan) and F32 elsewhere.
fn probe_dtype(projection: &Qwen36Projection, device: &Device) -> DType {
    match projection {
        Qwen36Projection::Quantized(QMatMul::Tensor(weight)) => weight.dtype(),
        _ if device.is_cuda() => DType::BF16,
        _ => DType::F32,
    }
}

/// Largest dense pack (elements) worth duplicating: the DeltaNet beta/alpha
/// pair. Larger dense projections are not packed so expanded F16/BF16
/// residencies never double.
const MAX_DENSE_PACK_ELEMENTS: usize = 1 << 20;

/// Pack projections that read the same input into one along their output
/// rows, so one GEMV computes them all. Block-FP8 parts must sit on 128-row
/// block boundaries and are replaced by views into the packed weights
/// (residency does not grow); small dense parts are concatenated. The pack is
/// verified (one synthetic row through the packed weights must reproduce every
/// part) before any part is replaced. Returns `None` with the parts untouched
/// when they cannot be packed.
fn pack_projections(parts: &mut [&mut Qwen36Projection]) -> Option<Qwen36Projection> {
    let compact = parts
        .iter()
        .map(|part| match &**part {
            Qwen36Projection::CompactFp8 { weights, scales } => Some((weights, scales)),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    let built = if let Some(compact) = compact {
        let dims = compact
            .iter()
            .map(|(w, _)| w.dims2().ok())
            .collect::<Option<Vec<_>>>()?;
        let cols = dims.first()?.1;
        if dims
            .iter()
            .any(|&(rows, k)| k != cols || !rows.is_multiple_of(128))
        {
            return None;
        }
        let weights = Tensor::cat(&compact.iter().map(|(w, _)| *w).collect::<Vec<_>>(), 0).ok()?;
        let scales = Tensor::cat(&compact.iter().map(|(_, s)| *s).collect::<Vec<_>>(), 0).ok()?;
        let mut views = Vec::with_capacity(dims.len());
        let mut offset = 0;
        for &(rows, _) in &dims {
            views.push(Qwen36Projection::CompactFp8 {
                weights: weights.narrow(0, offset, rows).ok()?,
                scales: scales.narrow(0, offset / 128, rows / 128).ok()?,
            });
            offset += rows;
        }
        (
            Qwen36Projection::CompactFp8 { weights, scales },
            Some(views),
            cols,
        )
    } else {
        let dense = parts
            .iter()
            .map(|part| match &**part {
                Qwen36Projection::Quantized(QMatMul::Tensor(tensor)) => Some(tensor),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let cols = dense.first()?.dims2().ok()?.1;
        let total: usize = dense.iter().map(|t| t.elem_count()).sum();
        if total > MAX_DENSE_PACK_ELEMENTS
            || dense.iter().any(|t| {
                t.dims2().map(|(_, k)| k).ok() != Some(cols) || t.dtype() != dense[0].dtype()
            })
        {
            return None;
        }
        let packed = Tensor::cat(&dense, 0).ok()?;
        (
            Qwen36Projection::Quantized(QMatMul::Tensor(packed)),
            None,
            cols,
        )
    };
    let (packed, views, cols) = built;
    let device = match &packed {
        Qwen36Projection::CompactFp8 { weights, .. } => weights.device().clone(),
        Qwen36Projection::Quantized(QMatMul::Tensor(t)) => t.device().clone(),
        _ => return None,
    };
    let dtype = probe_dtype(&packed, &device);
    let probe = Tensor::from_vec(
        (0..cols)
            .map(|i| ((i as f32) * 0.754_877_7).sin())
            .collect::<Vec<_>>(),
        (1, cols),
        &Device::Cpu,
    )
    .and_then(|t| t.to_dtype(dtype))
    .and_then(|t| t.to_device(&device))
    .ok()?;
    let packed_out = packed.forward(&probe).ok()?;
    let mut offset = 0;
    for part in parts.iter() {
        let expected = part.forward(&probe).ok()?;
        let width = expected.dim(1).ok()?;
        let actual = packed_out.narrow(1, offset, width).ok()?;
        let diff = (actual.to_dtype(DType::F32).ok()? - expected.to_dtype(DType::F32).ok()?)
            .ok()?
            .abs()
            .ok()?
            .max_all()
            .ok()?
            .to_scalar::<f32>()
            .ok()?;
        let scale = expected
            .to_dtype(DType::F32)
            .ok()?
            .abs()
            .ok()?
            .max_all()
            .ok()?
            .to_scalar::<f32>()
            .ok()?;
        if diff.is_nan() || diff > 1e-3 * scale.max(1e-6) {
            return None;
        }
        offset += width;
    }
    if let Some(views) = views {
        for (part, view) in parts.iter_mut().zip(views) {
            **part = view;
        }
    }
    Some(packed)
}

impl Qwen36Projection {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(qmatmul) => Ok(qmatmul.forward(x)?),
            Self::CompactFp8 { weights, scales } => {
                crate::kernels::cuda::fp8::block_fp8_projection(x, weights, scales)
                    .map_err(Error::from)
            }
        }
    }
}

pub(crate) trait Qwen36WeightSource {
    fn has(&self, name: &str) -> bool;

    fn projection(&self, name: &str, device: &Device) -> Result<Qwen36Projection>;

    fn rms_norm(&self, name: &str, eps: f64, device: &Device) -> Result<Qwen36RmsNorm>;

    /// Storage order of the DeltaNet value heads (see
    /// [`Qwen36LinearVHeadOrder`]). GGUF sources are tiled by conversion.
    fn linear_v_head_order(&self) -> Qwen36LinearVHeadOrder {
        Qwen36LinearVHeadOrder::Tiled
    }

    /// Dense tensor, coerced to `dtype` when requested (always F32 when
    /// `Some(F32)`).
    fn dense(&self, name: &str, dtype: Option<DType>, device: &Device) -> Result<Tensor>;

    /// Sparse-expert feed-forward weights for one decoder layer.
    fn moe_ffn(
        &self,
        layer: usize,
        geometry: &Qwen36MoeFfnGeometry,
        device: &Device,
    ) -> Result<Qwen36MoeSparseMlp>;

    /// Sparse-expert feed-forward weights at an arbitrary logical prefix.
    /// The MTP draft head builds its FFN under `mtpblk.{layer}.mlp`, which
    /// sits outside the trunk's `model.layers.{n}` indexing, so sources
    /// whose name resolution can address that scope override this; the
    /// default fails closed (a GGUF bundle has no such tensors).
    fn moe_ffn_prefix(
        &self,
        _prefix: &str,
        _geometry: &Qwen36MoeFfnGeometry,
        _device: &Device,
    ) -> Result<Qwen36MoeSparseMlp> {
        Err(Error::ModelLoadError(
            "this weight source cannot address an MoE FFN outside the trunk layer indexing"
                .to_string(),
        ))
    }

    /// Token embedding matrix `[vocab, hidden]`.
    fn token_embeddings(&self, device: &Device) -> Result<Tensor>;
}

/// GGUF-backed source for the dense Qwen3.5 family and the qwen36moe
/// synthetic fixture checkpoints (fused `ffn_*_exps` expert tensors).
pub(crate) struct GgufSource<'a> {
    loader: &'a GgufLoader,
}

impl<'a> GgufSource<'a> {
    pub(crate) fn new(loader: &'a GgufLoader) -> Self {
        Self { loader }
    }
}

impl Qwen36WeightSource for GgufSource<'_> {
    fn has(&self, name: &str) -> bool {
        self.loader.has_tensor(name)
    }

    fn projection(&self, name: &str, device: &Device) -> Result<Qwen36Projection> {
        Ok(Qwen36Projection::Quantized(load_qmatmul(
            self.loader, device, name,
        )?))
    }

    fn rms_norm(&self, name: &str, eps: f64, device: &Device) -> Result<Qwen36RmsNorm> {
        load_rms_norm(self.loader, device, name, eps)
    }

    fn dense(&self, name: &str, dtype: Option<DType>, device: &Device) -> Result<Tensor> {
        load_dense(self.loader, device, name, dtype)
    }

    fn moe_ffn(
        &self,
        layer: usize,
        geometry: &Qwen36MoeFfnGeometry,
        device: &Device,
    ) -> Result<Qwen36MoeSparseMlp> {
        crate::models::architectures::qwen36moe::sparse::load_gguf_sparse_mlp(
            self.loader,
            layer,
            geometry,
            device,
        )
    }

    fn token_embeddings(&self, device: &Device) -> Result<Tensor> {
        self.loader
            .load_qtensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(Error::from)
    }
}

/// Feed-forward branch of a trunk layer: dense SwiGLU MLP or the sparse
/// expert block. Mirrors `Qwen3FeedForward` on the qwen3 family.
enum Qwen36FeedForward {
    Dense(Qwen36Mlp),
    Sparse(Qwen36MoeSparseMlp),
}

impl Qwen36FeedForward {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden_states),
            Self::Sparse(moe) => moe.forward(hidden_states),
        }
    }
}

impl Qwen36TextModel {
    pub fn load(loader: &GgufLoader, cfg: &Qwen36TextConfig, device: &Device) -> Result<Self> {
        Self::load_with_source(&GgufSource::new(loader), cfg, device)
    }

    pub(crate) fn load_with_source(
        source: &dyn Qwen36WeightSource,
        cfg: &Qwen36TextConfig,
        device: &Device,
    ) -> Result<Self> {
        if cfg.attention_key_length != cfg.attention_value_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 full attention currently requires key/value head dims to match, found {} and {}",
                cfg.attention_key_length, cfg.attention_value_length
            )));
        }
        if cfg.ssm_time_step_rank == 0 || !cfg.ssm_inner_size.is_multiple_of(cfg.ssm_time_step_rank)
        {
            return Err(Error::ModelLoadError(format!(
                "Invalid Qwen3.5 linear attention dims: inner_size={}, time_step_rank={}",
                cfg.ssm_inner_size, cfg.ssm_time_step_rank
            )));
        }

        let embedding_weights = source.token_embeddings(device)?;
        let (vocab_size, hidden_size) = embedding_weights.dims2()?;
        if hidden_size != cfg.embedding_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 token embedding width mismatch: checkpoint has {hidden_size}, metadata says {}",
                cfg.embedding_length
            )));
        }
        let _ = vocab_size;

        let token_embeddings = Embedding::new(embedding_weights, hidden_size);
        let output_norm = source.rms_norm("output_norm.weight", cfg.attention_layer_norm_rms_epsilon, device)?;
        let output = if source.has("output.weight") {
            source.projection("output.weight", device)?
        } else {
            source.projection("token_embd.weight", device)?
        };
        let finite_diagnostics_enabled = qwen35_env_bool("IZWI_QWEN35_FINITE_DIAGNOSTICS", false);

        let mut layers = Vec::with_capacity(cfg.block_count);
        for layer_idx in 0..cfg.block_count {
            let prefix = format!("blk.{layer_idx}");
            let attn_norm = source.rms_norm(
                &format!("{prefix}.attn_norm.weight"),
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?;
            let post_attention_norm = source.rms_norm(
                &format!("{prefix}.post_attention_norm.weight"),
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?;
            let ffn = match &cfg.moe_ffn {
                Some(geometry) => {
                    Qwen36FeedForward::Sparse(source.moe_ffn(layer_idx, geometry, device)?)
                }
                None => Qwen36FeedForward::Dense(Qwen36Mlp::load_via(source, device, &prefix)?),
            };
            let mixer = if is_full_attention_layer(layer_idx, cfg.full_attention_interval) {
                Qwen36Mixer::Full(Qwen36FullAttention::load_via(source, device, &prefix, cfg)?)
            } else {
                Qwen36Mixer::Linear(Qwen36LinearAttention::load_via(source, device, &prefix, cfg)?)
            };

            layers.push(Qwen36Layer {
                attn_norm,
                mixer,
                post_attention_norm,
                ffn,
            });
        }
        let mut model = Self {
            device: device.clone(),
            token_embeddings,
            layers,
            output_norm,
            output,
            finite_diagnostics_enabled,
            fused_norm_path: Qwen36FusedPath::legacy("unresolved"),
            fp8_gemv_path: Qwen36FusedPath::legacy("unresolved"),
        };
        model.resolve_fused_norms(false);
        model.resolve_vectorized_fp8_decode();
        Ok(model)
    }

    /// Enable the vectorized block-FP8 decode GEMV (process-wide) after it
    /// matches the reference decode kernel on this device. Compact-FP8
    /// projections keep the reference kernel otherwise.
    fn resolve_vectorized_fp8_decode(&mut self) {
        self.fp8_gemv_path = if legacy_requested(FUSED_DECODE_ENV) {
            Qwen36FusedPath::legacy(format!("{FUSED_DECODE_ENV}=legacy"))
        } else if !self.device.is_cuda() {
            Qwen36FusedPath::legacy("vectorized FP8 decode GEMV runs on CUDA only")
        } else {
            match crate::kernels::cuda::fp8::verify_and_enable_vectorized_decode(&self.device) {
                Ok(()) => Qwen36FusedPath::Fused,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "vectorized FP8 decode GEMV self-check failed; keeping the reference kernel"
                    );
                    Qwen36FusedPath::legacy(format!("self-check failed: {error}"))
                }
            }
        };
    }

    /// Vectorized FP8 decode GEMV resolution, for the admin diagnostics.
    pub(crate) fn fp8_gemv_summary(&self) -> serde_json::Value {
        super::fast_path::summarize([&self.fp8_gemv_path])
    }

    /// Enable the fused RMSNorm kernels on every trunk norm (layer, q/k and
    /// output norms) after a self-check of the first layer's norm against the
    /// Candle composition. CUDA only in production (`allow_cpu` exercises the
    /// portable path in tests); `IZWI_QWEN36_FUSED_DECODE=legacy` keeps the
    /// cast → rms_norm → cast chain.
    fn resolve_fused_norms(&mut self, allow_cpu: bool) {
        let device = self.device.clone();
        let path = if legacy_requested(FUSED_DECODE_ENV) {
            Qwen36FusedPath::legacy(format!("{FUSED_DECODE_ENV}=legacy"))
        } else if !fused_device(&device, allow_cpu) {
            Qwen36FusedPath::legacy("fused RMSNorm runs on CUDA and Metal only")
        } else if !norm::supported(&device, norm_dtype(&device)) {
            Qwen36FusedPath::legacy("fused RMSNorm needs SM80+")
        } else {
            match self.fused_norm_self_check() {
                Ok(()) => Qwen36FusedPath::Fused,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Qwen3.6 fused RMSNorm self-check failed; using the Candle composition"
                    );
                    Qwen36FusedPath::legacy(format!("self-check failed: {error}"))
                }
            }
        };
        let fused = path.is_fused();
        self.output_norm.fused = fused;
        for layer in &mut self.layers {
            layer.attn_norm.fused = fused;
            layer.post_attention_norm.fused = fused;
            if let Qwen36Mixer::Full(attention) = &mut layer.mixer {
                attention.q_norm.fused = fused;
                attention.k_norm.fused = fused;
            }
        }
        self.fused_norm_path = path;
    }

    fn fused_norm_self_check(&self) -> Result<()> {
        let Some(layer) = self.layers.first() else {
            return Ok(());
        };
        let norm = &layer.attn_norm;
        if norm.weight.dtype() != DType::F32 {
            return Err(Error::InferenceError(
                "trunk norm gains are not F32; nothing to fuse".into(),
            ));
        }
        let hidden = norm.weight.elem_count();
        let wave = |seed: f32| {
            (0..3 * hidden)
                .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * 3.0)
                .collect::<Vec<_>>()
        };
        let dtype = norm_dtype(&self.device);
        let tensor = |seed: f32| -> Result<Tensor> {
            Ok(Tensor::from_vec(wave(seed), (3, 1, hidden), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&self.device)?)
        };
        let host = |tensor: &Tensor| -> Result<Vec<f32>> {
            Ok(tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        let (residual, delta) = (tensor(1.0)?, tensor(2.0)?);
        let eps = norm.eps as f32;
        let (sum, fused) = norm::add_rms_norm(&residual, &delta, &norm.weight, eps)?;
        let expected_sum = (&residual + &delta)?;
        let unfused = Qwen36RmsNorm {
            fused: false,
            ..norm.clone()
        };
        let expected = unfused.forward(&expected_sum)?;
        compare_values(
            "fused residual sum",
            &host(&sum)?,
            &host(&expected_sum)?,
            1e-6,
            1e-6,
        )?;
        compare_values(
            "fused add+RMSNorm",
            &host(&fused)?,
            &host(&expected)?,
            0.01,
            0.02,
        )?;
        let plain = norm::rms_norm(&residual, &norm.weight, eps)?;
        let plain_expected = unfused.forward(&residual)?;
        compare_values(
            "fused RMSNorm",
            &host(&plain)?,
            &host(&plain_expected)?,
            0.01,
            0.02,
        )?;
        Ok(())
    }

    /// Fused q/k norm + RoPE across the full-attention layers, for the admin
    /// diagnostics.
    pub(crate) fn qk_rope_summary(&self) -> serde_json::Value {
        super::fast_path::summarize(self.layers.iter().filter_map(|layer| match &layer.mixer {
            Qwen36Mixer::Full(attention) => Some(&attention.fused_qk_path),
            Qwen36Mixer::Linear(_) => None,
        }))
    }

    /// Fused RMSNorm resolution, for the admin diagnostics.
    pub(crate) fn rms_norm_summary(&self) -> serde_json::Value {
        super::fast_path::summarize([&self.fused_norm_path])
    }

    /// The device the trunk (and any MTP head) executes on.
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_state(&self) -> Qwen36TextRuntimeState {
        Qwen36TextRuntimeState {
            layers: self.layers.iter().map(Qwen36Layer::new_state).collect(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.token_embeddings.hidden_size()
    }

    /// Per-sparse-layer expert activation histograms (DS10 A6); empty for
    /// dense checkpoints.
    pub(crate) fn expert_activation_counters(
        &self,
    ) -> Vec<std::sync::Arc<crate::models::shared::moe::ExpertActivationCounters>> {
        self.layers
            .iter()
            .filter_map(|layer| match &layer.ffn {
                Qwen36FeedForward::Sparse(moe) => Some(moe.counters()),
                Qwen36FeedForward::Dense(_) => None,
            })
            .collect()
    }

    /// Sparse-expert execution paths across the trunk's MoE layers for the
    /// admin diagnostics: how many layers run the device-routed fused kernels
    /// and why any others stayed on the per-expert loop.
    pub(crate) fn moe_backend_summary(&self) -> serde_json::Value {
        super::fast_path::summarize(self.layers.iter().filter_map(|layer| match &layer.ffn {
            Qwen36FeedForward::Sparse(moe) => Some(moe.backend()),
            Qwen36FeedForward::Dense(_) => None,
        }))
    }

    /// Resolve every fused path against the portable CPU references (tests
    /// only): production resolution enables them on CUDA alone.
    #[cfg(test)]
    pub(crate) fn enable_fused_paths_for_tests(&mut self) {
        let device = self.device.clone();
        let hidden = self.hidden_size();
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Qwen36Mixer::Linear(linear) => {
                    linear.pack_projections();
                    linear.resolve_fused_decode(hidden, &device, true);
                }
                Qwen36Mixer::Full(attention) => {
                    attention.packed_qkv = pack_projections(&mut [
                        &mut attention.q_proj,
                        &mut attention.k_proj,
                        &mut attention.v_proj,
                    ]);
                    attention.resolve_fused_qk(&device, true);
                }
            }
        }
        self.resolve_fused_norms(true);
    }

    /// Fused single-token DeltaNet decode across the linear-attention layers,
    /// for the admin diagnostics.
    pub(crate) fn gdn_decode_summary(&self) -> serde_json::Value {
        super::fast_path::summarize(self.layers.iter().filter_map(|layer| match &layer.mixer {
            Qwen36Mixer::Linear(linear) => Some(&linear.fused_decode_path),
            Qwen36Mixer::Full(_) => None,
        }))
    }

    pub(crate) fn forward_token_id_at_physical(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen36TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_token_id_hidden_at_physical(token_id, position_ids, state, cache)?;
        self.forward_hidden_to_logits(&hidden)
    }

    /// Forward one token and return its PRE-norm hidden — the MTP
    /// verification pass derives both the logits (`project_hidden_span`)
    /// and the post-`output_norm` hidden (`normalize_hidden`) from it.
    pub(crate) fn forward_token_id_hidden_at_physical(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen36TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        self.forward_hidden_physical(&hidden, &[position_ids], state, cache)
    }

    /// Apply the trunk's `output_norm` — the MTP pair consumes the
    /// post-norm hidden while logits flow through the LM head directly.
    pub(crate) fn normalize_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output_norm.forward(hidden).map_err(Error::from)
    }

    pub(crate) fn forward_token_ids_batch_at_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        states: &mut [&mut Qwen36TextRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let batch_size = token_ids.len();
        if batch_size == 0
            || position_ids.len() != batch_size
            || states.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 hybrid decode batch rows do not match".into(),
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
                matches!(layer.mixer, Qwen36Mixer::Full(_)).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let first_full = self
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Qwen36Mixer::Full(attention) => Some(attention),
                Qwen36Mixer::Linear(_) => None,
            })
            .ok_or_else(|| {
                Error::InferenceError("Qwen3.5 model has no full-attention layer".into())
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
        let slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(start_positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let lowered = first.arena().lower_slots(&slots)?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(start_positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;
        let execution = (|| -> Result<Tensor> {
            let input = Tensor::from_slice(token_ids, (batch_size, 1), &self.device)?;
            let mut hidden = Qwen36Hidden::new(self.token_embeddings.forward(&input)?);
            let mut physical_layer = 0usize;
            for (layer_index, layer) in self.layers.iter().enumerate() {
                let mut layer_states = states
                    .iter_mut()
                    .map(|state| &mut state.layers[layer_index])
                    .collect::<Vec<_>>();
                for state in layer_states.iter_mut() {
                    layer.ensure_state_initialized(state, &self.device)?;
                }
                let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
                hidden = layer.forward_physical_decode_batch(
                    hidden,
                    &mut layer_states,
                    position_ids,
                    &cache_refs,
                    lowered.as_ref(),
                    &metadata,
                    &mut completions,
                    &mut physical_layer,
                )?;
                if self.finite_diagnostics_enabled {
                    let full = hidden.materialize()?;
                    validate_qwen35_finite_tensor(
                        &full,
                        layer_index,
                        layer.decode_diagnostic_path(),
                        true,
                    )?;
                    hidden = Qwen36Hidden::new(full);
                }
            }
            let hidden = hidden.materialize()?;
            if physical_layer != sparse_layers.len() {
                return Err(Error::InferenceError(
                    "Qwen3.5 batched attention did not cover every sparse layer".into(),
                ));
            }
            self.project_hidden_span(&hidden)
        })();
        let logits = match execution {
            Ok(logits) => logits,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3.5 target batch failed: {error}; write-fence drain also failed: {drain}"
                ))),
                }
            }
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(start_positions[row], 1, completion.clone())?;
        }
        Ok(logits)
    }

    pub(crate) fn prefill_token_ids_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen36TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 physical prefill span mismatch: {} token ids for {} position ids",
                token_ids.len(),
                position_ids.len()
            )));
        }
        record_prefill_sequence_span(token_ids.len());
        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        let hidden = self.forward_hidden_physical(&hidden, position_ids, state, cache)?;
        if !compute_logits {
            return Ok(None);
        }
        let last = hidden.narrow(1, token_ids.len() - 1, 1)?;
        self.forward_hidden_to_logits(&last).map(Some)
    }

    pub(crate) fn forward_input_embedding_at_physical(
        &self,
        input_embedding: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen36TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_hidden_physical(input_embedding, &[position_ids], state, cache)?;
        self.forward_hidden_to_logits(&hidden)
    }

    /// Prefill one span and return every row's PRE-`output_norm` hidden plus,
    /// when requested, the final row's logits. Replay spans rebuild the MTP
    /// draft domain from the hidden rows; ordinary prefill ignores them.
    pub(crate) fn prefill_token_ids_with_hidden_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen36TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Qwen36PrefillSpanOutput> {
        if token_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen3.5 replay prefill requires a non-empty span".into(),
            ));
        }
        if token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 replay prefill span mismatch: {} token ids for {} position ids",
                token_ids.len(),
                position_ids.len()
            )));
        }
        record_prefill_sequence_span(token_ids.len());
        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        let hidden_states = self.forward_hidden_physical(&hidden, position_ids, state, cache)?;
        let logits = if compute_logits {
            let last = hidden_states.narrow(1, token_ids.len() - 1, 1)?;
            Some(self.forward_hidden_to_logits(&last)?)
        } else {
            None
        };
        Ok(Qwen36PrefillSpanOutput {
            hidden_states,
            logits,
        })
    }

    pub(crate) fn prefill_input_embeddings_physical(
        &self,
        input_embeddings: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen36TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        let (_, sequence_len, _) = input_embeddings.dims3()?;
        if sequence_len == 0 || sequence_len != position_ids.len() {
            return Err(Error::InvalidInput(
                "Qwen3.5 embedding prefill span does not match its positions".into(),
            ));
        }
        record_prefill_sequence_span(sequence_len);
        let hidden = self.forward_hidden_physical(input_embeddings, position_ids, state, cache)?;
        if !compute_logits {
            return Ok(None);
        }
        let last = hidden.narrow(1, sequence_len - 1, 1)?;
        self.forward_hidden_to_logits(&last).map(Some)
    }

    fn forward_hidden_physical(
        &self,
        input: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen36TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        self.validate_runtime_state(state)?;
        let (_, sequence_len, hidden_size) = input.dims3()?;
        if sequence_len == 0
            || sequence_len != position_ids.len()
            || hidden_size != self.hidden_size()
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 physical hidden span does not match its positions or model width".into(),
            ));
        }
        let sparse_layers = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                matches!(layer.mixer, Qwen36Mixer::Full(_)).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let first_full = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen36Mixer::Full(attention) => Some(attention),
            Qwen36Mixer::Linear(_) => None,
        });
        let first_full = first_full.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 model has no full-attention layer".into())
        })?;
        cache.validate_sparse_model(
            &sparse_layers,
            first_full.num_kv_heads,
            first_full.head_dim,
            first_full.head_dim,
        )?;
        let start_pos = cache.context_len();
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.ensure_state_initialized(layer_state, &self.device)?;
        }
        let mut hidden = Qwen36Hidden::new(input.clone());
        let mut physical_layer = 0usize;
        for (layer_index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            hidden = layer.forward_physical(
                hidden,
                layer_state,
                position_ids,
                cache,
                &mut prepared,
                &mut physical_layer,
            )?;
            if self.finite_diagnostics_enabled {
                let full = hidden.materialize()?;
                validate_qwen35_finite_tensor(
                    &full,
                    layer_index,
                    if sequence_len == 1 {
                        layer.decode_diagnostic_path()
                    } else {
                        layer.prefill_diagnostic_path()
                    },
                    true,
                )?;
                hidden = Qwen36Hidden::new(full);
            }
        }
        let hidden = hidden.materialize()?;
        if physical_layer != sparse_layers.len() {
            return Err(Error::InferenceError(
                "Qwen3.5 physical attention did not cover every sparse layer".into(),
            ));
        }
        cache.commit_prepared(prepared)?;
        Ok(hidden)
    }

    pub fn forward_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.project_hidden_span(hidden)?
            .i((0, 0))
            .map_err(Error::from)
    }

    /// Embed one token row: `[1, 1, hidden]` in the embedding table's dtype.
    /// The MTP draft head reuses the target embeddings for its continuations.
    pub(crate) fn embed_token_ids(&self, token_ids: &[u32]) -> Result<Tensor> {
        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        self.token_embeddings.forward(&input).map_err(Error::from)
    }

    /// Project post-norm trunk hidden states through the raw LM head — no
    /// `output_norm`. The MTP draft head shares the target's LM head exactly
    /// this way: its outputs are already normalized by `mtp.norm`.
    pub(crate) fn project_with_shared_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output.forward(hidden)
    }

    pub(crate) fn project_hidden_span(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.output_norm.forward(hidden)?;
        validate_qwen35_finite_tensor(
            &hidden,
            self.layers.len(),
            "output.norm",
            self.finite_diagnostics_enabled,
        )?;
        let logits = self.output.forward(&hidden)?;
        validate_qwen35_finite_tensor(
            &logits,
            self.layers.len(),
            "output.logits",
            self.finite_diagnostics_enabled,
        )?;
        Ok(logits)
    }

    fn validate_runtime_state(&self, state: &Qwen36TextRuntimeState) -> Result<()> {
        if state.layers.len() != self.layers.len() {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 runtime state layer mismatch: state has {}, model has {}",
                state.layers.len(),
                self.layers.len()
            )));
        }
        Ok(())
    }
}

/// One linear layer's deep-copied runtime state — the restore unit for
/// MTP verification rollback. Tensors are small (conv history slots plus
/// one recurrent matrix) and cloned with independent storage.
#[derive(Clone)]
pub(crate) struct Qwen36LinearStateSnapshot {
    conv_slots: Vec<Tensor>,
    recurrent: Option<Tensor>,
}

impl Qwen36TextRuntimeState {
    /// Deep-copy every linear layer's runtime state. Used by the MTP verify
    /// pass: one snapshot per verify position bounds the rollback unit.
    pub(crate) fn snapshot_linear_states(&self) -> Result<Vec<Qwen36LinearStateSnapshot>> {
        self.layers
            .iter()
            .map(|layer| match layer {
                Qwen36LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    // Logical (oldest-first) order: the restore rebuilds the
                    // ring with `next_idx = 0`, so copying the physical slot
                    // order would rotate the history whenever the ring had
                    // wrapped.
                    let conv_slots = match conv_state {
                        Some(ring) => ring
                            .ordered_slots()
                            .map(deep_copy_tensor_storage)
                            .collect::<candle_core::Result<Vec<_>>>()?,
                        None => Vec::new(),
                    };
                    let recurrent = match recurrent_state {
                        Some(tensor) => Some(deep_copy_tensor_storage(tensor)?),
                        None => None,
                    };
                    Ok(Qwen36LinearStateSnapshot {
                        conv_slots,
                        recurrent,
                    })
                }
                Qwen36LayerRuntimeState::Full => Ok(Qwen36LinearStateSnapshot {
                    conv_slots: Vec::new(),
                    recurrent: None,
                }),
            })
            .collect()
    }

    /// Restore linear states from a snapshot, replacing current tensors with
    /// freshly detached copies so the snapshot stays reusable.
    pub(crate) fn restore_linear_states(
        &mut self,
        snapshot: &[Qwen36LinearStateSnapshot],
    ) -> Result<()> {
        if snapshot.len() != self.layers.len() {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 linear state snapshot covers {} layers, state has {}",
                snapshot.len(),
                self.layers.len()
            )));
        }
        for (layer, snap) in self.layers.iter_mut().zip(snapshot) {
            let Qwen36LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            if let Some(ring) = conv_state.as_mut() {
                if ring.slots.len() == snap.conv_slots.len() {
                    ring.slots = snap
                        .conv_slots
                        .iter()
                        .map(deep_copy_tensor_storage)
                        .collect::<candle_core::Result<Vec<_>>>()?;
                    ring.next_idx = 0;
                }
            }
            if let Some(snapshot_tensor) = &snap.recurrent {
                *recurrent_state = Some(deep_copy_tensor_storage(snapshot_tensor)?);
            }
        }
        Ok(())
    }
}

/// The residual stream at a layer boundary, with the previous layer's FFN
/// output optionally not yet added: the next layer's input norm folds that add
/// into one fused launch (`Qwen36RmsNorm::add_forward`).
struct Qwen36Hidden {
    residual: Tensor,
    pending: Option<Tensor>,
}

impl Qwen36Hidden {
    fn new(residual: Tensor) -> Self {
        Self {
            residual,
            pending: None,
        }
    }

    fn materialize(self) -> Result<Tensor> {
        match self.pending {
            Some(delta) => (&self.residual + &delta).map_err(Error::from),
            None => Ok(self.residual),
        }
    }

    /// `(residual stream, norm(residual stream))` for the next layer.
    fn normalized(self, norm: &Qwen36RmsNorm) -> Result<(Tensor, Tensor)> {
        match self.pending {
            Some(delta) => norm.add_forward(&self.residual, &delta),
            None => {
                let normalized = norm.forward(&self.residual)?;
                Ok((self.residual, normalized))
            }
        }
    }
}

impl Qwen36Layer {
    fn decode_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen36Mixer::Linear(_) => "decode.linear_layer_output",
            Qwen36Mixer::Full(_) => "decode.full_attention_layer_output",
        }
    }

    fn prefill_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen36Mixer::Linear(_) => "prefill.linear_layer_output",
            Qwen36Mixer::Full(_) => "prefill.full_attention_layer_output",
        }
    }

    fn new_state(&self) -> Qwen36LayerRuntimeState {
        match self.mixer {
            Qwen36Mixer::Linear(_) => Qwen36LayerRuntimeState::Linear {
                conv_state: None,
                recurrent_state: None,
            },
            Qwen36Mixer::Full(_) => Qwen36LayerRuntimeState::Full,
        }
    }

    /// Pre-initialize lazy state tensors so the first-use allocation cost
    /// does not happen inside the per-token hot loop during prefill.
    fn ensure_state_initialized(
        &self,
        state: &mut Qwen36LayerRuntimeState,
        device: &Device,
    ) -> Result<()> {
        if let (
            Qwen36Mixer::Linear(mixer),
            Qwen36LayerRuntimeState::Linear {
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
                *conv_state = Some(ConvRingState { slots, next_idx: 0 });
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

    fn forward_physical(
        &self,
        hidden: Qwen36Hidden,
        state: &mut Qwen36LayerRuntimeState,
        position_ids: &[[usize; 3]],
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: &mut usize,
    ) -> Result<Qwen36Hidden> {
        let (residual, normalized) = hidden.normalized(&self.attn_norm)?;
        let mixed = match &self.mixer {
            Qwen36Mixer::Linear(mixer) => {
                if normalized.dim(1)? == 1 {
                    mixer.forward(&normalized, state)?
                } else {
                    mixer.forward_sequence(&normalized, state)?
                }
            }
            Qwen36Mixer::Full(mixer) => {
                let output = mixer.forward_physical(
                    &normalized,
                    position_ids,
                    cache,
                    prepared,
                    *physical_layer,
                )?;
                *physical_layer = physical_layer.checked_add(1).ok_or_else(|| {
                    Error::InvalidInput("Qwen3.5 physical layer ordinal overflow".into())
                })?;
                output
            }
        };
        let (residual, hidden_states) = self.post_attention_norm.add_forward(&residual, &mixed)?;
        let ffn_output = self.ffn.forward(&hidden_states)?;
        Ok(Qwen36Hidden {
            residual,
            pending: Some(ffn_output),
        })
    }

    fn forward_physical_decode_batch(
        &self,
        hidden: Qwen36Hidden,
        states: &mut [&mut Qwen36LayerRuntimeState],
        position_ids: &[[usize; 3]],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: &mut usize,
    ) -> Result<Qwen36Hidden> {
        let batch_size = hidden.residual.dim(0)?;
        if hidden.residual.dim(1)? != 1
            || batch_size == 0
            || states.len() != batch_size
            || position_ids.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 layer decode batch dimensions do not match".into(),
            ));
        }
        let (residual, normalized) = hidden.normalized(&self.attn_norm)?;
        let mixed = match &self.mixer {
            Qwen36Mixer::Linear(mixer) => mixer.forward_decode_batch(&normalized, states)?,
            Qwen36Mixer::Full(mixer) => {
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
                    Error::InvalidInput("Qwen3.5 physical layer ordinal overflow".into())
                })?;
                output
            }
        };
        let (residual, hidden_states) = self.post_attention_norm.add_forward(&residual, &mixed)?;
        let ffn_output = self.ffn.forward(&hidden_states)?;
        Ok(Qwen36Hidden {
            residual,
            pending: Some(ffn_output),
        })
    }
}

impl Qwen36Mlp {
    pub(crate) fn load_via(
        source: &dyn Qwen36WeightSource,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            gate: source.projection(&format!("{prefix}.ffn_gate.weight"), device)?,
            up: source.projection(&format!("{prefix}.ffn_up.weight"), device)?,
            down: source.projection(&format!("{prefix}.ffn_down.weight"), device)?,
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Use fused SiLU-gate-up if available (reduces memory bandwidth)
        let gate_proj_out = self.gate.forward(hidden_states)?;
        let up_proj_out = self.up.forward(hidden_states)?;

        let hidden = if let Some(fused) = try_fused_silu_mul(&gate_proj_out, &up_proj_out) {
            fused
        } else {
            let gate = ops::silu(&gate_proj_out)?;
            (&gate * &up_proj_out)?
        };

        self.down.forward(&hidden)
    }
}

impl Qwen36FullAttention {
    pub(crate) fn load_via(
        source: &dyn Qwen36WeightSource,
        device: &Device,
        prefix: &str,
        cfg: &Qwen36TextConfig,
    ) -> Result<Self> {
        let mut attention = Self {
            q_proj: source.projection(&format!("{prefix}.attn_q.weight"), device)?,
            k_proj: source.projection(&format!("{prefix}.attn_k.weight"), device)?,
            v_proj: source.projection(&format!("{prefix}.attn_v.weight"), device)?,
            o_proj: source.projection(&format!("{prefix}.attn_output.weight"), device)?,
            q_norm: source.rms_norm(
                &format!("{prefix}.attn_q_norm.weight"),
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?,
            k_norm: source.rms_norm(
                &format!("{prefix}.attn_k_norm.weight"),
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
            rope_kernel_enabled: qwen35_rope_kernel_enabled(device),
            rope_inv_freqs: build_rope_inv_freqs(
                cfg.rope_dimension_count.min(cfg.attention_key_length),
                cfg.rope_freq_base,
            )?,
            fused_qk: None,
            fused_qk_path: Qwen36FusedPath::legacy("unresolved"),
            packed_qkv: None,
        };
        if fused_device(device, false) && !legacy_requested(FUSED_DECODE_ENV) {
            attention.packed_qkv = pack_projections(&mut [
                &mut attention.q_proj,
                &mut attention.k_proj,
                &mut attention.v_proj,
            ]);
        }
        attention.resolve_fused_qk(device, false);
        Ok(attention)
    }

    /// Decode-step `(q_proj [b, 1, heads, 2 * head_dim], k [b, 1, kv, head_dim],
    /// v [b, 1, kv, head_dim])`, through the packed GEMV when available.
    fn decode_projections(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let batch = hidden_states.dim(0)?;
        let q_width = self.num_heads * self.head_dim * 2;
        let kv_width = self.num_kv_heads * self.head_dim;
        let (q, k, v) = match &self.packed_qkv {
            Some(packed) => {
                let out = packed.forward(hidden_states)?;
                (
                    out.narrow(D::Minus1, 0, q_width)?,
                    out.narrow(D::Minus1, q_width, kv_width)?,
                    out.narrow(D::Minus1, q_width + kv_width, kv_width)?,
                )
            }
            None => (
                self.q_proj.forward(hidden_states)?,
                self.k_proj.forward(hidden_states)?,
                self.v_proj.forward(hidden_states)?,
            ),
        };
        Ok((
            q.reshape((batch, 1, self.num_heads, self.head_dim * 2))?,
            k.reshape((batch, 1, self.num_kv_heads, self.head_dim))?,
            v.reshape((batch, 1, self.num_kv_heads, self.head_dim))?,
        ))
    }

    /// Resolve the fused q/k norm + RoPE at load: CUDA only in production
    /// (`allow_cpu` exercises the portable path in tests), F32 head-norm gains,
    /// and a self-check against the norm + rotary chain at text and
    /// multimodal positions.
    fn resolve_fused_qk(&mut self, device: &Device, allow_cpu: bool) {
        self.fused_qk = None;
        let spec = QkNormRopeSpec {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            eps: self.q_norm.eps as f32,
        };
        self.fused_qk_path = if legacy_requested(FUSED_DECODE_ENV) {
            Qwen36FusedPath::legacy(format!("{FUSED_DECODE_ENV}=legacy"))
        } else if !fused_device(device, allow_cpu) {
            Qwen36FusedPath::legacy("fused q/k norm + RoPE runs on CUDA and Metal only")
        } else if !rope::supported(device, norm_dtype(device), self.head_dim, self.rope_dim)
            || self.q_norm.weight.dtype() != DType::F32
            || self.k_norm.weight.dtype() != DType::F32
            || self.q_norm.eps != self.k_norm.eps
            || self.rope_inv_freqs.len() != self.rope_dim / 2
        {
            Qwen36FusedPath::legacy(
                "fused q/k norm + RoPE needs SM80+ on CUDA, F32 head-norm gains and an even rotary dim",
            )
        } else {
            let inv_freq = Tensor::from_vec(
                self.rope_inv_freqs.clone(),
                self.rope_inv_freqs.len(),
                &Device::Cpu,
            )
            .and_then(|t| t.to_device(device));
            match inv_freq.map_err(Error::from).and_then(|inv_freq| {
                self.fused_qk_self_check(&spec, &inv_freq, device)
                    .map(|()| inv_freq)
            }) {
                Ok(inv_freq) => {
                    self.fused_qk = Some((spec, inv_freq));
                    Qwen36FusedPath::Fused
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Qwen3.6 fused q/k norm + RoPE self-check failed; using the rotary chain"
                    );
                    Qwen36FusedPath::legacy(format!("self-check failed: {error}"))
                }
            }
        };
    }

    /// The M-RoPE position the fused kernel would use, or `None` when the
    /// positions need sections this layer does not declare (the reference
    /// chain reports that error).
    fn fused_position(&self, position_ids: [usize; 3]) -> Option<MropePosition> {
        let sectioned = position_ids[0] != position_ids[1] || position_ids[0] != position_ids[2];
        if sectioned
            && (self.mrope_sections.len() < 3
                || self.mrope_sections.iter().sum::<usize>() != self.rope_dim / 2)
        {
            return None;
        }
        Some(MropePosition {
            temporal: position_ids[0],
            height: position_ids[1],
            width: position_ids[2],
            height_section: self.mrope_sections.get(1).copied().unwrap_or(0),
            width_section: self.mrope_sections.get(2).copied().unwrap_or(0),
        })
    }

    /// Normalized, rotated `(q, k)` for one token, `[1, 1, heads, head_dim]`,
    /// through the fused kernel when resolved and applicable, else the chain.
    /// `q_proj` is the gated projection `[1, 1, heads, 2 * head_dim]`, `k_proj`
    /// is `[1, 1, kv_heads, head_dim]`.
    fn token_qk(
        &self,
        q_proj: &Tensor,
        k_proj: &Tensor,
        position_ids: [usize; 3],
    ) -> Result<(Tensor, Tensor)> {
        if let Some((spec, inv_freq)) = &self.fused_qk {
            if let Some(position) = self.fused_position(position_ids) {
                if matches!(q_proj.dtype(), DType::F16 | DType::BF16) {
                    return Ok(rope::qk_norm_rope(
                        q_proj,
                        k_proj,
                        &self.q_norm.weight,
                        &self.k_norm.weight,
                        inv_freq,
                        spec,
                        position,
                    )?);
                }
            }
        }
        let query_states = q_proj.narrow(3, 0, self.head_dim)?;
        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&k_proj.contiguous()?)?;
        self.apply_rope(&query_states, &key_states, position_ids)
    }

    fn fused_qk_self_check(
        &self,
        spec: &QkNormRopeSpec,
        inv_freq: &Tensor,
        device: &Device,
    ) -> Result<()> {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        };
        let wave = |n: usize, seed: f32| {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * 2.5)
                .collect::<Vec<_>>()
        };
        let host = |tensor: &Tensor| -> Result<Vec<f32>> {
            Ok(tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        let q_proj = Tensor::from_vec(
            wave(self.num_heads * 2 * self.head_dim, 1.0),
            (1, 1, self.num_heads, 2 * self.head_dim),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?;
        let k_proj = Tensor::from_vec(
            wave(self.num_kv_heads * self.head_dim, 2.0),
            (1, 1, self.num_kv_heads, self.head_dim),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?;
        let mut positions = vec![[5usize, 5, 5], [3000, 3000, 3000]];
        if self.fused_position([7, 3, 11]).is_some() {
            positions.push([7, 3, 11]);
        }
        for position_ids in positions {
            let position = self.fused_position(position_ids).ok_or_else(|| {
                Error::InferenceError("self-check position needs M-RoPE sections".into())
            })?;
            let (q, k) = rope::qk_norm_rope(
                &q_proj,
                &k_proj,
                &self.q_norm.weight,
                &self.k_norm.weight,
                inv_freq,
                spec,
                position,
            )?;
            let query_states = self
                .q_norm
                .forward(&q_proj.narrow(3, 0, self.head_dim)?.contiguous()?)?;
            let key_states = self.k_norm.forward(&k_proj)?;
            let (q_ref, k_ref) = self.apply_rope(&query_states, &key_states, position_ids)?;
            compare_values(
                "fused query norm + RoPE",
                &host(&q)?,
                &host(&q_ref)?,
                0.01,
                0.03,
            )?;
            compare_values(
                "fused key norm + RoPE",
                &host(&k)?,
                &host(&k_ref)?,
                0.01,
                0.03,
            )?;
        }
        Ok(())
    }

    pub(crate) fn forward_physical(
        &self,
        hidden_states: &Tensor,
        position_ids: &[[usize; 3]],
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 0 || seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 physical attention received {} tokens and {} positions",
                seq_len,
                position_ids.len()
            )));
        }
        let (q_proj, key_states, value_states) = if seq_len == 1 {
            self.decode_projections(hidden_states)?
        } else {
            (
                self.q_proj.forward(hidden_states)?.reshape((
                    1,
                    seq_len,
                    self.num_heads,
                    self.head_dim * 2,
                ))?,
                self.k_proj.forward(hidden_states)?.reshape((
                    1,
                    seq_len,
                    self.num_kv_heads,
                    self.head_dim,
                ))?,
                self.v_proj.forward(hidden_states)?.reshape((
                    1,
                    seq_len,
                    self.num_kv_heads,
                    self.head_dim,
                ))?,
            )
        };
        let gate = q_proj.narrow(3, self.head_dim, self.head_dim)?.reshape((
            1,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        let (query_states, key_states) = if seq_len == 1 {
            self.token_qk(&q_proj, &key_states, position_ids[0])?
        } else {
            let query_states = q_proj.narrow(3, 0, self.head_dim)?;
            let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
            let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
            self.apply_rope_sequence(&query_states, &key_states, position_ids)?
        };
        let queries = query_states
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .contiguous()?;
        let keys = key_states
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let values = value_states
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let storage_dtype = cache.arena().config().dtype;
        let output_dtype = queries.dtype();
        let queries = queries.to_dtype(storage_dtype)?;
        let keys = keys.to_dtype(storage_dtype)?;
        let values = values.to_dtype(storage_dtype)?;
        let output = cache.write_and_attend(
            physical_layer,
            prepared,
            &queries,
            &keys,
            &values,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let output =
            output
                .to_dtype(output_dtype)?
                .reshape((1, seq_len, self.num_heads * self.head_dim))?;
        let output = (&output * &ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&output)
    }

    pub(crate) fn forward_physical_decode_batch(
        &self,
        hidden_states: &Tensor,
        position_ids: &[[usize; 3]],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        if hidden_states.dim(1)? != 1
            || batch_size == 0
            || position_ids.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 full-attention decode batch dimensions do not match".into(),
            ));
        }
        let first = caches[0];
        if caches.iter().any(|cache| {
            !Arc::ptr_eq(cache.arena(), first.arena())
                || cache.layer_binding(physical_layer).ok()
                    != first.layer_binding(physical_layer).ok()
        }) {
            return Err(Error::InvalidInput(
                "Qwen3.5 decode rows must share one arena and sparse layer binding".into(),
            ));
        }
        if slots.arena_id() != first.arena().id() || slots.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.5 decode received an incompatible prepared slot map".into(),
            ));
        }
        let (q_proj, key_states, value_states) = self.decode_projections(hidden_states)?;
        let gate = q_proj.narrow(3, self.head_dim, self.head_dim)?.reshape((
            batch_size,
            1,
            self.num_heads * self.head_dim,
        ))?;
        let mut queries = Vec::with_capacity(batch_size);
        let mut keys = Vec::with_capacity(batch_size);
        let mut values = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let q_row = q_proj.i(row)?.unsqueeze(0)?;
            let k_row = key_states.i(row)?.unsqueeze(0)?;
            let (q_row, k_row) = self.token_qk(&q_row, &k_row, position_ids[row])?;
            queries.push(q_row.reshape((self.num_heads, self.head_dim))?);
            keys.push(k_row.reshape((self.num_kv_heads, self.head_dim))?);
            values.push(
                value_states
                    .i(row)?
                    .reshape((self.num_kv_heads, self.head_dim))?,
            );
        }
        let query_refs = queries.iter().collect::<Vec<_>>();
        let key_refs = keys.iter().collect::<Vec<_>>();
        let value_refs = values.iter().collect::<Vec<_>>();
        let queries = Tensor::stack(&query_refs, 0)?.contiguous()?;
        let keys = Tensor::stack(&key_refs, 0)?.contiguous()?;
        let values = Tensor::stack(&value_refs, 0)?.contiguous()?;
        let storage_dtype = first.arena().config().dtype;
        let output_dtype = queries.dtype();
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
        let output = (&output * &ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&output)
    }

    fn apply_rope(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        position_ids: [usize; 3],
    ) -> Result<(Tensor, Tensor)> {
        if self.rope_dim == 0 {
            return Ok((query_states.clone(), key_states.clone()));
        }
        let (cos, sin) = self.mrope(position_ids, query_states.device(), query_states.dtype())?;

        let query_rot = query_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let key_rot = key_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let (query_rot, key_rot) = if self.should_try_rope_kernel(query_states.dtype()) {
            match try_apply_rope_thd(&query_rot, &key_rot, &cos, &sin)? {
                Some((query_rot, key_rot)) => {
                    record_rope_kernel();
                    (query_rot, key_rot)
                }
                None => {
                    record_rope_manual();
                    (
                        apply_rotary_emb(&query_rot, &cos, &sin)?,
                        apply_rotary_emb(&key_rot, &cos, &sin)?,
                    )
                }
            }
        } else {
            record_rope_manual();
            (
                apply_rotary_emb(&query_rot, &cos, &sin)?,
                apply_rotary_emb(&key_rot, &cos, &sin)?,
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

    fn apply_rope_sequence(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        position_ids: &[[usize; 3]],
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = query_states.dim(1)?;
        if seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 rotary sequence mismatch: seq_len={}, position_ids={}",
                seq_len,
                position_ids.len()
            )));
        }
        if self.rope_dim == 0 {
            return Ok((query_states.clone(), key_states.clone()));
        }

        let mut cos_tokens = Vec::with_capacity(seq_len);
        let mut sin_tokens = Vec::with_capacity(seq_len);
        for &position_id in position_ids {
            let (cos, sin) =
                self.mrope(position_id, query_states.device(), query_states.dtype())?;
            cos_tokens.push(cos);
            sin_tokens.push(sin);
        }
        let cos_refs: Vec<&Tensor> = cos_tokens.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_tokens.iter().collect();
        let cos = Tensor::cat(&cos_refs, 1)?.contiguous()?;
        let sin = Tensor::cat(&sin_refs, 1)?.contiguous()?;

        let query_rot = query_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let key_rot = key_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let (query_rot, key_rot) = if self.should_try_rope_kernel(query_states.dtype()) {
            match try_apply_rope_thd(&query_rot, &key_rot, &cos, &sin)? {
                Some((query_rot, key_rot)) => {
                    for _ in 0..seq_len {
                        record_rope_kernel();
                    }
                    (query_rot, key_rot)
                }
                None => {
                    for _ in 0..seq_len {
                        record_rope_manual();
                    }
                    (
                        apply_rotary_emb(&query_rot, &cos, &sin)?,
                        apply_rotary_emb(&key_rot, &cos, &sin)?,
                    )
                }
            }
        } else {
            for _ in 0..seq_len {
                record_rope_manual();
            }
            (
                apply_rotary_emb(&query_rot, &cos, &sin)?,
                apply_rotary_emb(&key_rot, &cos, &sin)?,
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

    fn mrope(
        &self,
        position_ids: [usize; 3],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        build_mrope(
            self.rope_dim,
            position_ids,
            &self.mrope_sections,
            &self.rope_inv_freqs,
            device,
            dtype,
        )
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

impl Qwen36LinearAttention {
    fn load_via(
        source: &dyn Qwen36WeightSource,
        device: &Device,
        prefix: &str,
        cfg: &Qwen36TextConfig,
    ) -> Result<Self> {
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let head_k_dim = cfg.ssm_state_size;
        let head_v_dim = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

        let dt_bias_name = if source.has(&format!("{prefix}.ssm_dt.bias")) {
            format!("{prefix}.ssm_dt.bias")
        } else {
            format!("{prefix}.ssm_dt")
        };
        let dt_bias = source
            .dense(&dt_bias_name, Some(DType::F32), device)?
            .reshape((num_v_heads,))?
            .reshape((1, 1, num_v_heads))?;
        if dt_bias.elem_count() != num_v_heads {
            return Err(Error::ModelLoadError(format!(
                "Unexpected tensor size for {dt_bias_name}: expected {num_v_heads} elements, found {}",
                dt_bias.elem_count()
            )));
        }
        let a_name = format!("{prefix}.ssm_a");
        let a = source
            .dense(&a_name, Some(DType::F32), device)?
            .reshape((num_v_heads,))?
            .reshape((1, 1, num_v_heads))?;
        if a.elem_count() != num_v_heads {
            return Err(Error::ModelLoadError(format!(
                "Unexpected tensor size for {a_name}: expected {num_v_heads} elements, found {}",
                a.elem_count()
            )));
        }
        let conv_kernel = normalize_conv_kernel(
            source.dense(
                &format!("{prefix}.ssm_conv1d.weight"),
                Some(DType::F32),
                device,
            )?,
            conv_dim,
            cfg.ssm_conv_kernel,
        )?;
        let conv_kernel_slices = pre_slice_conv_kernel(&conv_kernel, cfg.ssm_conv_kernel)?;
        let norm_weight_name = format!("{prefix}.ssm_norm.weight");
        let norm_weight = source
            .dense(&norm_weight_name, Some(DType::F32), device)?
            .reshape((head_v_dim,))?;
        if norm_weight.elem_count() != head_v_dim {
            return Err(Error::ModelLoadError(format!(
                "Unexpected tensor size for {norm_weight_name}: expected {head_v_dim} elements, found {}",
                norm_weight.elem_count()
            )));
        }
        let norm = Qwen36GatedRmsNorm {
            weight: norm_weight,
            eps: cfg.attention_layer_norm_rms_epsilon,
        };

        let mut mixer = Self {
            qkv_proj: source.projection(&format!("{prefix}.attn_qkv.weight"), device)?,
            gate_proj: source.projection(&format!("{prefix}.attn_gate.weight"), device)?,
            beta_proj: source.projection(&format!("{prefix}.ssm_beta.weight"), device)?,
            alpha_proj: source.projection(&format!("{prefix}.ssm_alpha.weight"), device)?,
            dt_bias,
            a,
            conv_kernel,
            conv_kernel_slices,
            norm,
            out_proj: source.projection(&format!("{prefix}.ssm_out.weight"), device)?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            kernel_size: cfg.ssm_conv_kernel,
            v_head_order: source.linear_v_head_order(),
            tiled_recurrence_enabled: qwen35_tiled_recurrence_enabled(),
            tiled_recurrence_tile_size_override: qwen35_tiled_recurrence_tile_size_override(),
            fused_decode: None,
            fused_decode_path: Qwen36FusedPath::legacy("unresolved"),
            packed_in: None,
            packed_beta_alpha: None,
        };
        if fused_device(device, false) && !legacy_requested(FUSED_DECODE_ENV) {
            mixer.pack_projections();
        }
        mixer.resolve_fused_decode(cfg.embedding_length, device, false);
        Ok(mixer)
    }

    fn pack_projections(&mut self) {
        self.packed_in = pack_projections(&mut [&mut self.qkv_proj, &mut self.gate_proj]);
        self.packed_beta_alpha = pack_projections(&mut [&mut self.beta_proj, &mut self.alpha_proj]);
    }

    /// `(mixed_qkv, z, beta_raw, alpha)` projection outputs, through the packed
    /// GEMVs when available.
    fn decode_projections(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (mixed_qkv, z) = match &self.packed_in {
            Some(packed) => {
                let out = packed.forward(x)?;
                (
                    out.narrow(D::Minus1, 0, self.conv_dim)?,
                    out.narrow(D::Minus1, self.conv_dim, self.num_v_heads * self.head_v_dim)?,
                )
            }
            None => (self.qkv_proj.forward(x)?, self.gate_proj.forward(x)?),
        };
        let (beta_raw, alpha) = match &self.packed_beta_alpha {
            Some(packed) => {
                let out = packed.forward(x)?;
                (
                    out.narrow(D::Minus1, 0, self.num_v_heads)?,
                    out.narrow(D::Minus1, self.num_v_heads, self.num_v_heads)?,
                )
            }
            None => (self.beta_proj.forward(x)?, self.alpha_proj.forward(x)?),
        };
        Ok((mixed_qkv, z, beta_raw, alpha))
    }

    /// Resolve the fused single-token decode at load. Production enables it on
    /// CUDA only (`allow_cpu` lets tests run the portable reference), for
    /// 128-dim heads and a 4-tap conv, after a self-check against the Candle
    /// op chain on this layer's own weights.
    fn resolve_fused_decode(&mut self, hidden: usize, device: &Device, allow_cpu: bool) {
        self.fused_decode = None;
        self.fused_decode_path = if legacy_requested(FUSED_DECODE_ENV) {
            Qwen36FusedPath::legacy(format!("{FUSED_DECODE_ENV}=legacy"))
        } else if !fused_device(device, allow_cpu) {
            Qwen36FusedPath::legacy("fused DeltaNet decode runs on CUDA and Metal only")
        } else if !gdn::supported(device, self.head_k_dim, self.head_v_dim, self.kernel_size)
            || self.num_k_heads == 0
            || !self.num_v_heads.is_multiple_of(self.num_k_heads)
        {
            Qwen36FusedPath::legacy(
                "fused DeltaNet decode needs SM80+, 128-dim heads, a 4-tap conv and value heads divisible by key heads",
            )
        } else {
            let spec = GdnDecodeSpec {
                key_heads: self.num_k_heads,
                value_heads: self.num_v_heads,
                grouped: self.v_head_order == Qwen36LinearVHeadOrder::Grouped,
                norm_eps: self.norm.eps as f32,
            };
            match self.fused_decode_self_check(&spec, hidden, device) {
                Ok(()) => {
                    self.fused_decode = Some(spec);
                    Qwen36FusedPath::Fused
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Qwen3.6 fused DeltaNet decode self-check failed; using the Candle op chain"
                    );
                    Qwen36FusedPath::legacy(format!("self-check failed: {error}"))
                }
            }
        };
    }

    /// Run one synthetic decode step through both paths from identical states
    /// and require matching outputs, recurrent states and conv rings.
    fn fused_decode_self_check(
        &self,
        spec: &GdnDecodeSpec,
        hidden: usize,
        device: &Device,
    ) -> Result<()> {
        let dtype = probe_dtype(&self.beta_proj, device);
        let wave = |n: usize, seed: f32, scale: f32| {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * scale)
                .collect::<Vec<_>>()
        };
        let host = |tensor: &Tensor| -> Result<Vec<f32>> {
            Ok(tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        let x = Tensor::from_vec(wave(hidden, 0.0, 1.5), (1, 1, hidden), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let slot = |seed: f32| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                wave(self.conv_dim, seed, 1.0),
                (self.conv_dim, 1),
                &Device::Cpu,
            )?
            .to_device(device)?)
        };
        let state_dims = (1, self.num_v_heads, self.head_k_dim, self.head_v_dim);
        let state_len = self.num_v_heads * self.head_k_dim * self.head_v_dim;
        let initial = Qwen36LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState {
                slots: vec![slot(1.0)?, slot(2.0)?, slot(3.0)?],
                next_idx: 1,
            }),
            recurrent_state: Some(
                Tensor::from_vec(wave(state_len, 4.0, 0.3), state_dims, &Device::Cpu)?
                    .to_device(device)?,
            ),
        };
        let mut legacy_state = initial.clone();
        let mut fused_state = initial;
        let expected = self.forward_legacy(&x, &mut legacy_state)?;
        let actual = self.forward_fused(spec, &x, &mut fused_state)?;
        compare_values(
            "DeltaNet decode output",
            &host(&actual)?,
            &host(&expected)?,
            0.03,
            0.08,
        )?;
        let (
            Qwen36LayerRuntimeState::Linear {
                conv_state: Some(legacy_ring),
                recurrent_state: Some(legacy_recurrent),
            },
            Qwen36LayerRuntimeState::Linear {
                conv_state: Some(fused_ring),
                recurrent_state: Some(fused_recurrent),
            },
        ) = (&legacy_state, &fused_state)
        else {
            return Err(Error::InferenceError(
                "DeltaNet decode self-check lost its layer state".into(),
            ));
        };
        compare_values(
            "DeltaNet recurrent state",
            &host(fused_recurrent)?,
            &host(legacy_recurrent)?,
            0.01,
            0.05,
        )?;
        if legacy_ring.next_idx != fused_ring.next_idx {
            return Err(Error::InferenceError(
                "DeltaNet decode self-check: conv ring cursor diverged".into(),
            ));
        }
        for (fused, legacy) in fused_ring.slots.iter().zip(&legacy_ring.slots) {
            compare_values(
                "DeltaNet conv ring",
                &host(fused)?,
                &host(legacy)?,
                1e-6,
                1e-6,
            )?;
        }
        Ok(())
    }

    /// Whether a single-token decode can take the fused kernels with this
    /// layer state (3-slot F32 ring, F32 or absent recurrent state).
    fn fused_decode_accepts(hidden_states: &Tensor, state: &Qwen36LayerRuntimeState) -> bool {
        matches!(hidden_states.dtype(), DType::F32 | DType::F16 | DType::BF16)
            && matches!(
                state,
                Qwen36LayerRuntimeState::Linear {
                    conv_state: Some(ring),
                    recurrent_state,
                } if ring.slots.len() == gdn::CONV_TAPS - 1
                    && ring.next_idx < ring.slots.len()
                    && ring.slots.iter().all(|slot| slot.dtype() == DType::F32)
                    && recurrent_state.as_ref().is_none_or(|s| s.dtype() == DType::F32)
            )
    }

    /// One token's conv + recurrence + gated norm through the fused kernels,
    /// from projection outputs `mixed_qkv`, `z`, `beta_raw`, `alpha` (one row
    /// each). Advances the ring and replaces the recurrent state; returns the
    /// out-projection input `[value_heads * head_v_dim]` in `z`'s dtype.
    fn fused_decode_row(
        &self,
        spec: &GdnDecodeSpec,
        mixed_qkv: &Tensor,
        z: &Tensor,
        beta_raw: &Tensor,
        alpha: &Tensor,
        state: &mut Qwen36LayerRuntimeState,
    ) -> Result<Tensor> {
        let Qwen36LayerRuntimeState::Linear {
            conv_state: Some(ring),
            recurrent_state,
        } = state
        else {
            return Err(Error::InferenceError(
                "Qwen3.6 fused DeltaNet decode needs an initialized linear-attention state".into(),
            ));
        };
        let history = ring.ordered_slots().cloned().collect::<Vec<_>>();
        let (conv, current) = gdn::conv_decode(
            mixed_qkv,
            &self.conv_kernel,
            [&history[0], &history[1], &history[2]],
        )?;
        ring.push_decode(&current.reshape((self.conv_dim, 1))?)?;
        let previous = match recurrent_state.take() {
            Some(previous) => previous,
            None => Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                conv.device(),
            )?,
        };
        let (y, next) = gdn::recurrent_decode(
            &conv,
            z,
            beta_raw,
            alpha,
            &self.dt_bias,
            &self.a,
            &self.norm.weight,
            &previous,
            spec,
        )?;
        *recurrent_state = Some(next);
        Ok(y)
    }

    fn forward_fused(
        &self,
        spec: &GdnDecodeSpec,
        hidden_states: &Tensor,
        state: &mut Qwen36LayerRuntimeState,
    ) -> Result<Tensor> {
        let (mixed_qkv, z, beta_raw, alpha) = self.decode_projections(hidden_states)?;
        let y = self.fused_decode_row(spec, &mixed_qkv, &z, &beta_raw, &alpha, state)?;
        self.out_proj
            .forward(&y.reshape((1, 1, self.num_v_heads * self.head_v_dim))?)
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen36LayerRuntimeState,
    ) -> Result<Tensor> {
        match &self.fused_decode {
            Some(spec) if Self::fused_decode_accepts(hidden_states, state) => {
                self.forward_fused(spec, hidden_states, state)
            }
            _ => self.forward_legacy(hidden_states, state),
        }
    }

    fn forward_legacy(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen36LayerRuntimeState,
    ) -> Result<Tensor> {
        let (conv_state, recurrent_state) = match state {
            Qwen36LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        // The DeltaNet block computes in F32 regardless of the trunk
        // activation dtype: the recurrent/conv state arena, the softplus
        // decay gates, and every fused CUDA kernel (causal conv, tiled
        // DeltaNet recurrence, gated delta decode, gated RMS norm) are
        // F32-only, while activations may be BF16/F16 under the CUDA/Metal
        // native plans. Upcast at entry, downcast before `out_proj` — the
        // same contract the qwen3.8 DeltaNet applies. Without the island a
        // CUDA forward fails inside the conv/recurrence with candle's
        // "dtype mismatch in binary op".
        let residual_dtype = hidden_states.dtype();
        let mixed_qkv = self
            .qkv_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        let z = self.gate_proj.forward(hidden_states)?.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(&self.beta_proj.forward(hidden_states)?)?.to_dtype(DType::F32)?;
        let alpha = self
            .alpha_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        let mixed_qkv = self.depthwise_conv_step(&mixed_qkv, conv_state)?;

        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let query =
            mixed_qkv
                .narrow(2, 0, key_width)?
                .reshape((1, self.num_k_heads, self.head_k_dim))?;
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

        let mut query = l2norm(&query, 1e-6)?;
        let mut key = l2norm(&key, 1e-6)?;
        if self.num_v_heads != self.num_k_heads {
            if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
                return Err(Error::InferenceError(format!(
                    "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                    self.num_v_heads, self.num_k_heads
                )));
            }
            query = self.expand_key_heads(&query)?;
            key = self.expand_key_heads(&key)?;
        }

        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            // The recurrent arena is F32 by contract (see
            // `ensure_state_initialized`); a lazily-created state must agree
            // with the pre-initialized dtype rather than inherit the
            // activation dtype.
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                value.device(),
            )?
        };

        let beta = beta.reshape((1, self.num_v_heads))?;
        let g = g.reshape((1, self.num_v_heads))?;
        let (output, next_state) =
            recurrent_gated_delta(&query, &key, &value, &g, &beta, current_state)?;
        // The recurrent decode output owns a fresh Candle-managed allocation;
        // retaining it avoids a full state-sized copy on every layer/token.
        *recurrent_state = Some(next_state);

        let output = output.reshape((self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output
            .reshape((1, 1, self.num_v_heads * self.head_v_dim))?
            .to_dtype(residual_dtype)?;
        self.out_proj.forward(&output)
    }

    fn forward_decode_batch(
        &self,
        hidden_states: &Tensor,
        states: &mut [&mut Qwen36LayerRuntimeState],
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        if hidden_states.dim(1)? != 1 || batch_size == 0 || states.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.5 linear-attention decode batch dimensions do not match".into(),
            ));
        }
        if let Some(spec) = &self.fused_decode {
            if states
                .iter()
                .all(|state| Self::fused_decode_accepts(hidden_states, state))
            {
                let (mixed_qkv, z, beta_raw, alpha) = self.decode_projections(hidden_states)?;
                let rows = states
                    .iter_mut()
                    .enumerate()
                    .map(|(row, state)| {
                        self.fused_decode_row(
                            spec,
                            &mixed_qkv.i(row)?,
                            &z.i(row)?,
                            &beta_raw.i(row)?,
                            &alpha.i(row)?,
                            state,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let output = Tensor::stack(&rows, 0)?.reshape((
                    batch_size,
                    1,
                    self.num_v_heads * self.head_v_dim,
                ))?;
                return self.out_proj.forward(&output);
            }
        }
        let residual_dtype = hidden_states.dtype();
        // F32 compute island — see `forward_legacy` for the contract.
        let mixed_qkv = self
            .qkv_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        let z = self.gate_proj.forward(hidden_states)?.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(&self.beta_proj.forward(hidden_states)?)?.to_dtype(DType::F32)?;
        let alpha = self
            .alpha_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
            return Err(Error::InferenceError(format!(
                "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }
        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let mut output_rows = Vec::with_capacity(batch_size);
        let mut gate_rows = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let (conv_state, recurrent_state) = match &mut *states[row] {
                Qwen36LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => (conv_state, recurrent_state),
                _ => {
                    return Err(Error::InferenceError(
                        "Qwen3.5 layer runtime state does not match linear-attention layer".into(),
                    ))
                }
            };
            let mixed_qkv = mixed_qkv.i(row)?.unsqueeze(0)?;
            let z = z.i(row)?.unsqueeze(0)?;
            let alpha = alpha.i(row)?.unsqueeze(0)?;
            let beta = beta.i(row)?.unsqueeze(0)?;
            let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;
            let mixed_qkv = self.depthwise_conv_step(&mixed_qkv, conv_state)?;
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
            let mut query = l2norm(&query, 1e-6)?;
            let mut key = l2norm(&key, 1e-6)?;
            if self.num_v_heads != self.num_k_heads {
                query = self.expand_key_heads(&query)?;
                key = self.expand_key_heads(&key)?;
            }
            let current_state = if let Some(state) = recurrent_state.take() {
                state
            } else {
                // F32 by contract — see `ensure_state_initialized`.
                Tensor::zeros(
                    (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    DType::F32,
                    value.device(),
                )?
            };
            let beta = beta.reshape((1, self.num_v_heads))?;
            let g = g.reshape((1, self.num_v_heads))?;
            let (output, next_state) =
                recurrent_gated_delta(&query, &key, &value, &g, &beta, current_state)?;
            *recurrent_state = Some(next_state);
            output_rows.push(output.reshape((self.num_v_heads, self.head_v_dim))?);
            gate_rows.push(z.reshape((self.num_v_heads, self.head_v_dim))?);
        }
        let output_refs = output_rows.iter().collect::<Vec<_>>();
        let gate_refs = gate_rows.iter().collect::<Vec<_>>();
        let output = Tensor::stack(&output_refs, 0)?;
        let gate = Tensor::stack(&gate_refs, 0)?;
        let output = self.norm.forward(&output, &gate)?.reshape((
            batch_size,
            1,
            self.num_v_heads * self.head_v_dim,
        ))?;
        let output = output.to_dtype(residual_dtype)?;
        self.out_proj.forward(&output)
    }

    fn forward_sequence(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen36LayerRuntimeState,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 1 {
            return self.forward(hidden_states, state);
        }

        let (conv_state, recurrent_state) = match state {
            Qwen36LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        // F32 compute island — see `forward` for the contract.
        let residual_dtype = hidden_states.dtype();
        let mixed_qkv = self
            .qkv_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        let z = self.gate_proj.forward(hidden_states)?.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(&self.beta_proj.forward(hidden_states)?)?.to_dtype(DType::F32)?;
        let alpha = self
            .alpha_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        let mixed_qkv = self.depthwise_conv_sequence(&mixed_qkv, conv_state)?;

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

        let query = l2norm(&query, 1e-6)?;
        let key = l2norm(&key, 1e-6)?;
        if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
            return Err(Error::InferenceError(format!(
                "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }

        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            // F32 by contract — see `ensure_state_initialized`.
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                value.device(),
            )?
        };

        let beta = beta.reshape((1, seq_len, self.num_v_heads))?;
        let g = g.reshape((1, seq_len, self.num_v_heads))?;
        let tile_size =
            qwen35_tiled_recurrence_tile_size(seq_len, self.tiled_recurrence_tile_size_override);
        // The compact (un-expanded) Metal sequence kernel hard-codes the tiled
        // pairing (`key_head = v_head % num_k_heads`), so a grouped checkpoint
        // may only reach it once q/k are expanded below.
        let compact_heads_compatible = self.num_v_heads == self.num_k_heads
            || self.v_head_order == Qwen36LinearVHeadOrder::Tiled;
        let fused_sequence = if self.tiled_recurrence_enabled && compact_heads_compatible {
            try_tiled_deltanet_recurrence(
                &query,
                &key,
                &value,
                &g,
                &beta,
                &current_state,
                tile_size,
            )
        } else {
            None
        };
        let (output, next_state) = if let Some(fused_sequence) = fused_sequence {
            fused_sequence
        } else {
            // CUDA's equal-head kernel and the portable Candle reference consume
            // q/k already expanded to one head per value head. The Metal
            // sequence op above consumes the compact converted-GGUF 16K layout
            // directly for both 16V and 32V tiled models.
            let (query, key) = if self.num_v_heads == self.num_k_heads {
                (query, key)
            } else {
                (
                    self.expand_key_heads_seq(&query)?,
                    self.expand_key_heads_seq(&key)?,
                )
            };
            if self.tiled_recurrence_enabled {
                if let Some(fused_sequence) = try_tiled_deltanet_recurrence(
                    &query,
                    &key,
                    &value,
                    &g,
                    &beta,
                    &current_state,
                    tile_size,
                ) {
                    fused_sequence
                } else {
                    recurrent_gated_delta_sequence(&query, &key, &value, &g, &beta, current_state)?
                }
            } else {
                recurrent_gated_delta_sequence(&query, &key, &value, &g, &beta, current_state)?
            }
        };
        // Sequence kernels may return the final state as a view into their
        // packed output. Detach only this fixed-size escape value, using
        // Candle-managed storage rather than an application private buffer.
        *recurrent_state = Some(deep_copy_tensor_storage(&next_state)?);

        let output = output.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output
            .reshape((1, seq_len, self.num_v_heads * self.head_v_dim))?
            .to_dtype(residual_dtype)?;
        self.out_proj.forward(&output)
    }

    /// Expand `[batch, num_k_heads, dim]` q/k to one head per value head in
    /// the checkpoint's value-head order.
    fn expand_key_heads(&self, x: &Tensor) -> Result<Tensor> {
        let repeats = self.num_v_heads / self.num_k_heads;
        match self.v_head_order {
            Qwen36LinearVHeadOrder::Tiled => repeat_head_states(x, repeats),
            Qwen36LinearVHeadOrder::Grouped => repeat_interleave_head_states(x, repeats),
        }
    }

    /// Sequence form of [`Self::expand_key_heads`] over
    /// `[batch, seq, num_k_heads, dim]`.
    fn expand_key_heads_seq(&self, x: &Tensor) -> Result<Tensor> {
        let repeats = self.num_v_heads / self.num_k_heads;
        match self.v_head_order {
            Qwen36LinearVHeadOrder::Tiled => repeat_head_states_seq(x, repeats),
            Qwen36LinearVHeadOrder::Grouped => repeat_interleave_head_states_seq(x, repeats),
        }
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
            if buffer.slots.len() != history_len || buffer.next_idx >= history_len {
                return Err(Error::InferenceError(format!(
                    "Invalid Qwen3.5 convolution history: slots={}, next_idx={}, expected_slots={history_len}",
                    buffer.slots.len(),
                    buffer.next_idx
                )));
            }
            let logical_slots: Vec<&Tensor> = (0..history_len)
                .map(|idx| &buffer.slots[(buffer.next_idx + idx) % history_len])
                .collect();
            let history = Tensor::cat(&logical_slots, 1)?;
            if let Some((output, final_history)) =
                try_qwen35_causal_conv_sequence(mixed_qkv, &self.conv_kernel, &history)
            {
                let final_history = deep_copy_tensor_storage(&final_history)?;
                buffer.slots = (0..history_len)
                    .map(|idx| final_history.narrow(1, idx, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                buffer.next_idx = 0;
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
            for i in 0..(self.kernel_size - 1) {
                let ring_idx = (buffer.next_idx + i) % history_len;
                let prev_token = &buffer.slots[ring_idx];
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

impl Qwen36GatedRmsNorm {
    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        if hidden_states.dtype() == DType::F32 {
            if let Some(result) =
                try_fused_gated_rms_norm(hidden_states, gate, &self.weight, self.eps)
            {
                return Ok(result);
            }
        }

        let normalized = candle_nn::ops::rms_norm(hidden_states, &self.weight, self.eps as f32)?;
        (&normalized * &ops::silu(gate)?).map_err(Error::from)
    }
}

fn is_full_attention_layer(layer_idx: usize, full_attention_interval: usize) -> bool {
    full_attention_interval > 0 && (layer_idx + 1).is_multiple_of(full_attention_interval)
}

fn load_qmatmul(loader: &GgufLoader, device: &Device, name: &str) -> Result<QMatMul> {
    let weights = Arc::new(loader.load_qtensor(name, device)?);
    QMatMul::from_arc(weights).map_err(Error::from)
}

fn load_rms_norm(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    eps: f64,
) -> Result<Qwen36RmsNorm> {
    Ok(Qwen36RmsNorm::new(
        loader
            .load_qtensor(name, device)?
            .dequantize(device)
            .map_err(Error::from)?,
        eps,
    ))
}

fn load_dense(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    dtype: Option<DType>,
) -> Result<Tensor> {
    let mut tensor = loader
        .load_qtensor(name, device)?
        .dequantize(device)
        .map_err(Error::from)?;
    if let Some(dtype) = dtype {
        if tensor.dtype() != dtype {
            tensor = tensor.to_dtype(dtype)?;
        }
    }
    Ok(tensor)
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
                    "Unexpected Qwen3.5 conv kernel shape: ({d0}, {d1}) for expected ({expected_channels}, {expected_kernel})"
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
                    "Unexpected rank-3 Qwen3.5 conv kernel shape: {:?}",
                    dims
                )))
            }
        }
        rank => Err(Error::ModelLoadError(format!(
            "Unexpected Qwen3.5 conv kernel rank {rank}"
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
            "Invalid Qwen3.5 rotary dimension {rope_dim}"
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
                "Invalid Qwen3.5 multimodal RoPE sections {:?} for rotary dim {}",
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

    // Take cos/sin of the F32 angles and cast only the results (HF computes
    // `freqs.cos()` / `.sin()` in fp32 before `.to(x.dtype)`). Rounding the
    // angle first puts BF16's 8-bit mantissa on `position * inv_freq`: ~0.15
    // rad of phase error by position 128 and effectively random rotations by
    // ~4K, and F16 overflows the angle to inf (NaN cos/sin) past 65504.
    let emb = Tensor::from_vec(interleaved, (1, 1, half_dim), device)?;
    Ok((emb.cos()?.to_dtype(dtype)?, emb.sin()?.to_dtype(dtype)?))
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
        Ok(Err(_)) | Err(_) => Ok(None),
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
            "Invalid Qwen3.5 rotary dimension {rope_dim}"
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

fn validate_qwen35_finite_tensor(
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
        "Qwen3.5 first non-finite tensor at {path}, layer {layer_idx}: \
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

fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    // Try fused Metal kernel first for F32 tensors
    if x.dtype() == DType::F32 {
        if let Some(result) = try_fused_l2_norm(x, eps) {
            return Ok(result);
        }
    }

    // Fallback to standard implementation
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?)
        .map_err(Error::from)
}

/// Tiled key-head expansion `[h0..hK, h0..hK, ...]`: value head `j` reads key
/// head `j % K`. Matches llama.cpp-converted GGUF, whose conversion permutes
/// the value heads into tiled order (see [`Qwen36LinearVHeadOrder`]).
fn repeat_head_states(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, heads, dim) = x.dims3()?;
    let expanded = x.unsqueeze(1)?.broadcast_as((batch, repeats, heads, dim))?;
    expanded
        .reshape((batch, repeats * heads, dim))
        .map_err(Error::from)
}

/// Grouped key-head expansion `[h0, h0, h1, h1, ...]` (HF `repeat_interleave`):
/// value head `j` reads key head `j / repeats`. Matches native HF safetensors.
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

fn repeat_head_states_seq(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, seq, heads, dim) = x.dims4()?;
    let expanded = x
        .unsqueeze(2)?
        .broadcast_as((batch, seq, repeats, heads, dim))?;
    expanded
        .reshape((batch, seq, repeats * heads, dim))
        .map_err(Error::from)
}

fn qwen35_env_bool(name: &str, default: bool) -> bool {
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

fn qwen35_tiled_recurrence_enabled() -> bool {
    qwen35_env_bool("IZWI_QWEN35_TILED_RECURRENCE", true)
}

fn qwen35_rope_kernel_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_QWEN35_ROPE_KERNEL")
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        });
    qwen35_rope_kernel_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn qwen35_rope_kernel_policy(
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

fn qwen35_tiled_recurrence_tile_size_override() -> Option<usize> {
    if let Ok(raw) = std::env::var("IZWI_QWEN35_TILED_RECURRENCE_TILE_SIZE") {
        if let Ok(parsed) = raw.trim().parse::<usize>() {
            return Some(parsed.max(1));
        }
    }
    None
}

fn qwen35_tiled_recurrence_tile_size(seq_len: usize, override_size: Option<usize>) -> usize {
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
        if let Some(result) = try_fused_gated_delta_recurrent(query, key, value, g, beta, &state) {
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
        apply_rotary_emb, build_mrope, convolution_domain_v2, non_finite_counts, owned_zero_tensor,
        qwen35_rope_kernel_policy, recurrent_domain_v2, repeat_head_states, repeat_head_states_seq,
        repeat_interleave_head_states, repeat_interleave_head_states_seq, softplus, ConvRingState,
        Qwen36GatedRmsNorm, Qwen36LayerRuntimeState, Qwen36LinearAttention, Qwen36LinearVHeadOrder,
        Qwen36Projection, Qwen36TextRuntimeState,
    };
    use crate::kernels::cuda::gdn::GdnDecodeSpec;
    use crate::models::architectures::qwen36moe::cache::{
        CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN,
    };
    use crate::models::architectures::qwen36moe::fast_path::Qwen36FusedPath;
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
    fn batched_linear_decode_matches_independent_scalar_rows_with_ring_history() {
        let device = &Device::Cpu;
        let dense = |out: usize, input: usize, scale: f32| {
            let weights = Tensor::from_vec(
                (0..out * input)
                    .map(|index| scale * ((index % input) + 1) as f32)
                    .collect::<Vec<_>>(),
                (out, input),
                device,
            )
            .unwrap();
            let weights = QTensor::quantize(&weights, GgmlDType::F32).unwrap();
            Qwen36Projection::Quantized(QMatMul::from_arc(Arc::new(weights)).unwrap())
        };
        let conv_kernel = Tensor::from_vec(
            (0..24)
                .map(|index| 0.01 * ((index % 4) + 1) as f32)
                .collect::<Vec<_>>(),
            (6, 4),
            device,
        )
        .unwrap();
        let conv_kernel_slices = super::pre_slice_conv_kernel(&conv_kernel, 4).unwrap();
        let mixer = Qwen36LinearAttention {
            qkv_proj: dense(6, 4, 0.01),
            gate_proj: dense(2, 4, 0.02),
            beta_proj: dense(1, 4, -0.02),
            alpha_proj: dense(1, 4, 0.03),
            dt_bias: Tensor::zeros((1, 1, 1), DType::F32, device).unwrap(),
            a: Tensor::full(-0.5f32, (1, 1, 1), device).unwrap(),
            conv_kernel,
            conv_kernel_slices,
            norm: Qwen36GatedRmsNorm {
                weight: Tensor::ones(2, DType::F32, device).unwrap(),
                eps: 1e-6,
            },
            out_proj: dense(4, 2, 0.04),
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_dim: 6,
            kernel_size: 4,
            v_head_order: Qwen36LinearVHeadOrder::Tiled,
            tiled_recurrence_enabled: false,
            tiled_recurrence_tile_size_override: None,
            fused_decode: None,
            fused_decode_path: Qwen36FusedPath::legacy("test"),
            packed_in: None,
            packed_beta_alpha: None,
        };
        let initial_state = |row: usize| Qwen36LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState {
                slots: (0..3)
                    .map(|slot| {
                        Tensor::full((row * 3 + slot + 1) as f32 * 0.01, (6, 1), device).unwrap()
                    })
                    .collect(),
                next_idx: row,
            }),
            recurrent_state: None,
        };
        let initial = [initial_state(0), initial_state(1)];
        let input = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.3, 0.4, -0.3, 0.5, 0.1, 0.2],
            (2, 1, 4),
            device,
        )
        .unwrap();
        let mut scalar_states = initial.clone();
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
        let mut batch_states = initial;
        let mut batch_refs = batch_states.iter_mut().collect::<Vec<_>>();
        let batched = mixer.forward_decode_batch(&input, &mut batch_refs).unwrap();
        let scalar_values = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let batch_values = batched.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (scalar, batch) in scalar_values.iter().zip(&batch_values) {
            assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
        }
        for (scalar, batch) in scalar_states.iter().zip(&batch_states) {
            let Qwen36LayerRuntimeState::Linear {
                conv_state: Some(scalar_conv),
                recurrent_state: Some(scalar_recurrent),
            } = scalar
            else {
                panic!("scalar row omitted hybrid state")
            };
            let Qwen36LayerRuntimeState::Linear {
                conv_state: Some(batch_conv),
                recurrent_state: Some(batch_recurrent),
            } = batch
            else {
                panic!("batch row omitted hybrid state")
            };
            assert_eq!(scalar_conv.next_idx, batch_conv.next_idx);
            assert_eq!(scalar_conv.slots.len(), batch_conv.slots.len());
            for (scalar, batch) in scalar_conv.slots.iter().zip(&batch_conv.slots) {
                let scalar = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let batch = batch.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for (scalar, batch) in scalar.iter().zip(&batch) {
                    assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
                }
            }
            let scalar = scalar_recurrent
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let batch = batch_recurrent
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (scalar, batch) in scalar.iter().zip(&batch) {
                assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
            }
        }
    }

    /// A real-geometry-per-head DeltaNet mixer (128-dim heads, 4-tap conv)
    /// with 2 key heads, 4 value heads and dense F32 projections.
    fn gdn_mixer(order: Qwen36LinearVHeadOrder) -> Qwen36LinearAttention {
        let device = &Device::Cpu;
        let (hidden, hk, hv, d) = (128usize, 2usize, 4usize, 128usize);
        let conv_dim = (2 * hk + hv) * d;
        let wave = |n: usize, seed: f32, scale: f32| {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.618_034).sin() * scale)
                .collect::<Vec<_>>()
        };
        let dense = |rows: usize, cols: usize, seed: f32, scale: f32| {
            Qwen36Projection::Quantized(QMatMul::Tensor(
                Tensor::from_vec(wave(rows * cols, seed, scale), (rows, cols), device).unwrap(),
            ))
        };
        let conv_kernel =
            Tensor::from_vec(wave(conv_dim * 4, 3.0, 0.4), (conv_dim, 4), device).unwrap();
        let conv_kernel_slices = super::pre_slice_conv_kernel(&conv_kernel, 4).unwrap();
        Qwen36LinearAttention {
            qkv_proj: dense(conv_dim, hidden, 1.0, 0.12),
            gate_proj: dense(hv * d, hidden, 2.0, 0.12),
            beta_proj: dense(hv, hidden, 4.0, 0.12),
            alpha_proj: dense(hv, hidden, 5.0, 0.12),
            dt_bias: Tensor::from_vec(wave(hv, 6.0, 1.0), (1, 1, hv), device).unwrap(),
            a: Tensor::from_vec(vec![-0.5f32, -1.5, -0.25, -3.0], (1, 1, hv), device).unwrap(),
            conv_kernel,
            conv_kernel_slices,
            norm: Qwen36GatedRmsNorm {
                weight: Tensor::from_vec(
                    wave(d, 7.0, 0.2)
                        .iter()
                        .map(|v| 1.0 + v)
                        .collect::<Vec<_>>(),
                    d,
                    device,
                )
                .unwrap(),
                eps: 1e-6,
            },
            out_proj: dense(hidden, hv * d, 8.0, 0.05),
            num_k_heads: hk,
            num_v_heads: hv,
            head_k_dim: d,
            head_v_dim: d,
            conv_dim,
            kernel_size: 4,
            v_head_order: order,
            tiled_recurrence_enabled: false,
            tiled_recurrence_tile_size_override: None,
            fused_decode: None,
            fused_decode_path: Qwen36FusedPath::legacy("test"),
            packed_in: None,
            packed_beta_alpha: None,
        }
    }

    fn gdn_state(seed: f32) -> Qwen36LayerRuntimeState {
        let conv_dim = 8 * 128;
        Qwen36LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState {
                slots: (0..3)
                    .map(|slot| {
                        Tensor::from_vec(
                            (0..conv_dim)
                                .map(|i| ((i as f32 + seed + slot as f32) * 0.31).cos() * 0.5)
                                .collect::<Vec<_>>(),
                            (conv_dim, 1),
                            &Device::Cpu,
                        )
                        .unwrap()
                    })
                    .collect(),
                next_idx: 2,
            }),
            recurrent_state: None,
        }
    }

    fn flat(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol * scale,
                "{label} index {index}: {a} vs {e}"
            );
        }
    }

    fn attention_for_qk(sections: Vec<usize>) -> super::Qwen36FullAttention {
        let device = &Device::Cpu;
        let unused = || {
            Qwen36Projection::Quantized(QMatMul::Tensor(
                Tensor::zeros((1, 1), DType::F32, device).unwrap(),
            ))
        };
        let gain = |seed: f32| {
            super::Qwen36RmsNorm::new(
                Tensor::from_vec(
                    (0..64)
                        .map(|i| 1.0 + ((i as f32 + seed) * 0.3).sin() * 0.2)
                        .collect::<Vec<_>>(),
                    64,
                    device,
                )
                .unwrap(),
                1e-6,
            )
        };
        super::Qwen36FullAttention {
            q_proj: unused(),
            k_proj: unused(),
            v_proj: unused(),
            o_proj: unused(),
            q_norm: gain(1.0),
            k_norm: gain(2.0),
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            rope_dim: 16,
            rope_theta: 10_000_000.0,
            mrope_sections: sections,
            rope_kernel_enabled: false,
            rope_inv_freqs: super::build_rope_inv_freqs(16, 10_000_000.0).unwrap(),
            fused_qk: None,
            fused_qk_path: Qwen36FusedPath::legacy("test"),
            packed_qkv: None,
        }
    }

    #[test]
    fn fused_qk_norm_rope_matches_the_rotary_chain() {
        let legacy = attention_for_qk(vec![3, 3, 2]);
        let mut fused = attention_for_qk(vec![3, 3, 2]);
        fused.resolve_fused_qk(&Device::Cpu, true);
        assert_eq!(fused.fused_qk_path, Qwen36FusedPath::Fused);
        let q_proj = Tensor::from_vec(
            (0..4 * 128)
                .map(|i| (i as f32 * 0.754_877_7).sin() * 2.0)
                .collect::<Vec<_>>(),
            (1, 1, 4, 128),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
        let k_proj = Tensor::from_vec(
            (0..2 * 64)
                .map(|i| (i as f32 * 0.31).cos() * 2.0)
                .collect::<Vec<_>>(),
            (1, 1, 2, 64),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
        for position in [[0usize, 0, 0], [17, 17, 17], [9000, 9000, 9000], [7, 3, 11]] {
            let (q, k) = fused.token_qk(&q_proj, &k_proj, position).unwrap();
            let (q_ref, k_ref) = legacy.token_qk(&q_proj, &k_proj, position).unwrap();
            assert_eq!(q.dims(), q_ref.dims());
            assert_eq!(k.dims(), k_ref.dims());
            assert_close(
                &flat(&q),
                &flat(&q_ref),
                2e-3,
                &format!("q at {position:?}"),
            );
            assert_close(
                &flat(&k),
                &flat(&k_ref),
                2e-3,
                &format!("k at {position:?}"),
            );
        }
    }

    #[test]
    fn fused_qk_defers_unsectioned_multimodal_positions_to_the_chain() {
        let mut attention = attention_for_qk(vec![]);
        attention.resolve_fused_qk(&Device::Cpu, true);
        assert!(attention.fused_qk.is_some());
        assert!(attention.fused_position([4, 4, 4]).is_some());
        assert!(
            attention.fused_position([4, 2, 9]).is_none(),
            "sectionless layers leave M-RoPE validation to the reference chain"
        );
    }

    /// A tiny Qwen3.6 trunk at block-FP8-friendly geometry (every projection a
    /// multiple of 128) with BF16 activations, so every fused path has a
    /// portable reference to run on the CPU.
    struct SyntheticFp8Source {
        moe: crate::models::architectures::qwen36moe::fused_moe::Qwen36MoeBackendRequest,
    }

    const E2E_HIDDEN: usize = 256;
    const E2E_VOCAB: usize = 64;

    fn e2e_config() -> super::Qwen36TextConfig {
        super::Qwen36TextConfig {
            architecture: "qwen36moe".into(),
            block_count: 4,
            context_length: 256,
            embedding_length: E2E_HIDDEN,
            feed_forward_length: 128,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            attention_key_length: 128,
            attention_value_length: 128,
            rope_dimension_sections: vec![8, 4, 4],
            rope_dimension_count: 32,
            rope_freq_base: 10_000_000.0,
            attention_layer_norm_rms_epsilon: 1e-6,
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 2,
            ssm_time_step_rank: 4,
            ssm_inner_size: 512,
            full_attention_interval: 4,
            moe_ffn: Some(super::Qwen36MoeFfnGeometry {
                num_experts: 8,
                num_experts_per_tok: 2,
                expert_intermediate_size: 128,
                shared_expert_intermediate_size: 128,
            }),
        }
    }

    fn e2e_seed(name: &str) -> u64 {
        name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        })
    }

    fn e2e_wave(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let phase = (seed % 1000) as f32 * 0.001;
        (0..n)
            .map(|i| ((i as f32 + phase * 977.0) * 0.618_034).sin() * scale)
            .collect()
    }

    fn e2e_fp8(rows: usize, cols: usize, name: &str) -> Qwen36Projection {
        let mut state = e2e_seed(name) | 1;
        let bytes = (0..rows * cols)
            .map(|_| loop {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let byte = state as u8;
                // Keep magnitudes modest: exponent field <= 7 (|w| < 2).
                if byte & 0x7f != 0x7f && (byte >> 3) & 15 <= 7 {
                    break byte;
                }
            })
            .collect::<Vec<_>>();
        Qwen36Projection::CompactFp8 {
            weights: Tensor::from_vec(bytes, (rows, cols), &Device::Cpu).unwrap(),
            scales: Tensor::from_vec(
                vec![0.6 / (cols as f32).sqrt(); (rows / 128) * (cols / 128)],
                (rows / 128, cols / 128),
                &Device::Cpu,
            )
            .unwrap(),
        }
    }

    fn e2e_dense(rows: usize, cols: usize, name: &str, dtype: DType) -> Tensor {
        Tensor::from_vec(
            e2e_wave(rows * cols, e2e_seed(name), 1.0 / (cols as f32).sqrt()),
            (rows, cols),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
    }

    /// Synthetic weights are built on the host; the source hands them over on
    /// the load device like the real sources do.
    fn e2e_on(projection: Qwen36Projection, device: &Device) -> Qwen36Projection {
        match projection {
            Qwen36Projection::CompactFp8 { weights, scales } => Qwen36Projection::CompactFp8 {
                weights: weights.to_device(device).unwrap(),
                scales: scales.to_device(device).unwrap(),
            },
            Qwen36Projection::Quantized(QMatMul::Tensor(weight)) => {
                Qwen36Projection::Quantized(QMatMul::Tensor(weight.to_device(device).unwrap()))
            }
            other => other,
        }
    }

    impl super::Qwen36WeightSource for SyntheticFp8Source {
        fn has(&self, _name: &str) -> bool {
            true
        }

        fn projection(
            &self,
            name: &str,
            device: &Device,
        ) -> crate::error::Result<Qwen36Projection> {
            let (h, conv_dim, value_width) = (E2E_HIDDEN, 1024usize, 512usize);
            let suffix = name.rsplit_once('.').map(|(head, _)| head).unwrap_or(name);
            let kind = suffix.rsplit_once('.').map(|(_, k)| k).unwrap_or(suffix);
            let projection = match kind {
                "attn_qkv" => e2e_fp8(conv_dim, h, name),
                "attn_gate" => e2e_fp8(value_width, h, name),
                "ssm_out" => e2e_fp8(h, value_width, name),
                "ssm_beta" | "ssm_alpha" => {
                    Qwen36Projection::Quantized(QMatMul::Tensor(e2e_dense(4, h, name, DType::F16)))
                }
                "attn_q" => e2e_fp8(2 * 2 * 128, h, name),
                "attn_k" | "attn_v" => e2e_fp8(128, h, name),
                "attn_output" => e2e_fp8(h, 2 * 128, name),
                "output" => Qwen36Projection::Quantized(QMatMul::Tensor(e2e_dense(
                    E2E_VOCAB,
                    h,
                    name,
                    DType::F16,
                ))),
                other => panic!("unexpected synthetic projection {name} ({other})"),
            };
            Ok(e2e_on(projection, device))
        }

        fn rms_norm(
            &self,
            name: &str,
            eps: f64,
            device: &Device,
        ) -> crate::error::Result<super::Qwen36RmsNorm> {
            let width = if name.contains("attn_q_norm") || name.contains("attn_k_norm") {
                128
            } else {
                E2E_HIDDEN
            };
            let gains = e2e_wave(width, e2e_seed(name), 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>();
            Ok(super::Qwen36RmsNorm::new(
                Tensor::from_vec(gains, width, device).unwrap(),
                eps,
            ))
        }

        fn linear_v_head_order(&self) -> Qwen36LinearVHeadOrder {
            Qwen36LinearVHeadOrder::Grouped
        }

        fn dense(
            &self,
            name: &str,
            dtype: Option<DType>,
            device: &Device,
        ) -> crate::error::Result<Tensor> {
            let tensor = if name.ends_with("ssm_conv1d.weight") {
                e2e_dense(1024, 4, name, DType::F32)
            } else if name.ends_with("ssm_norm.weight") {
                Tensor::from_vec(
                    e2e_wave(128, e2e_seed(name), 0.2)
                        .iter()
                        .map(|v| 1.0 + v)
                        .collect::<Vec<_>>(),
                    128,
                    &Device::Cpu,
                )
                .unwrap()
            } else if name.ends_with("ssm_a") {
                Tensor::from_vec(vec![-0.5f32, -1.0, -2.0, -0.25], 4, &Device::Cpu).unwrap()
            } else {
                Tensor::from_vec(e2e_wave(4, e2e_seed(name), 0.5), 4, &Device::Cpu).unwrap()
            };
            Ok(match dtype {
                Some(dtype) => tensor.to_dtype(dtype).unwrap(),
                None => tensor,
            }
            .to_device(device)?)
        }

        fn moe_ffn(
            &self,
            layer: usize,
            geometry: &super::Qwen36MoeFfnGeometry,
            device: &Device,
        ) -> crate::error::Result<crate::models::architectures::qwen36moe::sparse::Qwen36MoeSparseMlp>
        {
            use crate::models::architectures::qwen36moe::sparse::{
                Qwen36MoeExpertWeights, Qwen36MoeLinear, Qwen36MoeSharedExpertWeights,
                Qwen36MoeSparseMlp,
            };
            let linear = |rows: usize, cols: usize, name: String| match e2e_on(
                e2e_fp8(rows, cols, &name),
                device,
            ) {
                Qwen36Projection::CompactFp8 { weights, scales } => {
                    Qwen36MoeLinear::CompactFp8 { weights, scales }
                }
                Qwen36Projection::Quantized(_) => unreachable!(),
            };
            let (h, i) = (E2E_HIDDEN, geometry.expert_intermediate_size);
            let experts = (0..geometry.num_experts)
                .map(|e| Qwen36MoeExpertWeights {
                    gate: linear(i, h, format!("{layer}.{e}.gate")),
                    up: linear(i, h, format!("{layer}.{e}.up")),
                    down: linear(h, i, format!("{layer}.{e}.down")),
                })
                .collect();
            let shared = Qwen36MoeSharedExpertWeights {
                gate: linear(i, h, format!("{layer}.shared.gate")),
                up: linear(i, h, format!("{layer}.shared.up")),
                down: linear(h, i, format!("{layer}.shared.down")),
                output_gate: Some(Qwen36MoeLinear::from_dense(
                    e2e_dense(1, h, &format!("{layer}.shared.output_gate"), DType::F32)
                        .to_device(device)?,
                )),
            };
            let router = Qwen36MoeLinear::from_dense(
                (e2e_dense(
                    geometry.num_experts,
                    h,
                    &format!("{layer}.router"),
                    DType::F32,
                ) * 4.0)?
                    .to_device(device)?,
            );
            Qwen36MoeSparseMlp::from_weights_with_backend(
                router, experts, shared, geometry, self.moe,
            )
        }

        fn token_embeddings(&self, device: &Device) -> crate::error::Result<Tensor> {
            Ok(e2e_dense(E2E_VOCAB, E2E_HIDDEN, "token_embd", DType::F16).to_device(device)?)
        }
    }

    fn e2e_cache(
        device: &Device,
    ) -> crate::models::shared::attention::physical::PhysicalPagedKvCache {
        use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
        use crate::backends::BackendKind;
        use crate::engine::ModelInstanceId;
        use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
        use crate::models::shared::attention::physical::PhysicalPagedKvCache;
        use candle_core::DeviceLocation;
        let bindings = vec![KvLayerBinding {
            model_layer: 3,
            physical_layer: 0,
        }];
        let (backend, device_ordinal) = match device.location() {
            DeviceLocation::Cpu => (BackendKind::Cpu, None),
            DeviceLocation::Cuda { gpu_id } => (BackendKind::Cuda, Some(gpu_id as u32)),
            DeviceLocation::Metal { gpu_id } => {
                let gpu_id = gpu_id as u64;
                (BackendKind::Metal, Some((gpu_id ^ (gpu_id >> 32)) as u32))
            }
        };
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4251),
            backend,
            device_ordinal,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let config = KvArenaConfig {
            id,
            group,
            page_tokens: 8,
            capacity_pages: 8,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding: bindings[0],
                num_kv_heads: 1,
                key_head_dim: 128,
                value_head_dim: 128,
            }],
        };
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let arena: Arc<dyn KvArena> = if device.is_cpu() {
            Arc::new(CpuKvArena::new(config).unwrap())
        } else {
            Arc::new(
                crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                    config,
                    device.clone(),
                )
                .unwrap(),
            )
        };
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let arena: Arc<dyn KvArena> = Arc::new(CpuKvArena::new(config).unwrap());
        let blocks = (0..8u32)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(arena, bindings, blocks, 0).unwrap()
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap()
    }

    fn e2e_logits(model: &super::Qwen36TextModel, tokens: &[u32], prompt: usize) -> Vec<Vec<f32>> {
        let positions: Vec<[usize; 3]> = (0..tokens.len()).map(|p| [p, p, p]).collect();
        let mut state = model.new_state();
        let mut cache = e2e_cache(&model.device);
        let mut rows = vec![flat(
            &model
                .prefill_token_ids_physical(
                    &tokens[..prompt],
                    &positions[..prompt],
                    &mut state,
                    &mut cache,
                    true,
                )
                .unwrap()
                .expect("prefill logits"),
        )];
        for position in prompt..tokens.len() {
            rows.push(flat(
                &model
                    .forward_token_id_at_physical(
                        tokens[position],
                        positions[position],
                        &mut state,
                        &mut cache,
                    )
                    .unwrap(),
            ));
        }
        rows
    }

    #[test]
    fn every_fused_path_tracks_the_reference_trunk_end_to_end() {
        use crate::models::architectures::qwen36moe::fused_moe::Qwen36MoeBackendRequest;
        let cfg = e2e_config();
        let legacy = super::Qwen36TextModel::load_with_source(
            &SyntheticFp8Source {
                moe: Qwen36MoeBackendRequest::Legacy,
            },
            &cfg,
            &Device::Cpu,
        )
        .unwrap();
        let mut fused = super::Qwen36TextModel::load_with_source(
            &SyntheticFp8Source {
                moe: Qwen36MoeBackendRequest::Auto,
            },
            &cfg,
            &Device::Cpu,
        )
        .unwrap();
        fused.enable_fused_paths_for_tests();
        assert_eq!(fused.moe_backend_summary()["backend"], "fused");
        assert_eq!(
            fused.gdn_decode_summary()["backend"],
            "fused",
            "{}",
            fused.gdn_decode_summary()
        );
        assert_eq!(
            fused.qk_rope_summary()["backend"],
            "fused",
            "{}",
            fused.qk_rope_summary()
        );
        assert_eq!(
            fused.rms_norm_summary()["backend"],
            "fused",
            "{}",
            fused.rms_norm_summary()
        );
        assert_eq!(legacy.moe_backend_summary()["backend"], "legacy");

        assert_tracks_reference(&fused, &legacy);
    }

    /// Prefill 6 tokens, then decode 8: every step within 1% relative L2 of
    /// the reference model's logits and on its argmax.
    fn assert_tracks_reference(fused: &super::Qwen36TextModel, reference: &super::Qwen36TextModel) {
        let tokens: Vec<u32> = (0..14).map(|i| (i * 7 + 3) % E2E_VOCAB as u32).collect();
        let expected = e2e_logits(reference, &tokens, 6);
        let actual = e2e_logits(fused, &tokens, 6);
        for (step, (a, e)) in actual.iter().zip(&expected).enumerate() {
            let err: f32 = a
                .iter()
                .zip(e)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = e.iter().map(|y| y * y).sum::<f32>().sqrt();
            // Observed ~1e-3 (rounding-order noise); a wrong expert, head
            // mapping or state hand-off is far above 1%.
            assert!(
                err <= 0.01 * norm,
                "step {step}: fused logits diverge (relative L2 {})",
                err / norm
            );
            assert_eq!(
                argmax(a),
                argmax(e),
                "step {step}: fused decode picks a different token"
            );
        }
    }

    /// The Metal leg loads through production resolution (no test hook): each
    /// fast path must pass its self-check on the GPU and resolve to fused, and
    /// the GPU trunk must track the CPU reference.
    #[cfg(feature = "metal")]
    #[test]
    fn every_fused_path_resolves_and_tracks_the_reference_trunk_on_metal() {
        use crate::models::architectures::qwen36moe::fused_moe::Qwen36MoeBackendRequest;
        let Some(device) = crate::backends::metal_device_if_available(0) else {
            eprintln!("metal device unavailable; metal end-to-end leg not run");
            return;
        };
        let cfg = e2e_config();
        let reference = super::Qwen36TextModel::load_with_source(
            &SyntheticFp8Source {
                moe: Qwen36MoeBackendRequest::Legacy,
            },
            &cfg,
            &Device::Cpu,
        )
        .unwrap();
        let fused = super::Qwen36TextModel::load_with_source(
            &SyntheticFp8Source {
                moe: Qwen36MoeBackendRequest::Auto,
            },
            &cfg,
            &device,
        )
        .unwrap();
        for (path, summary) in [
            ("moe", fused.moe_backend_summary()),
            ("gdn_decode", fused.gdn_decode_summary()),
            ("qk_rope", fused.qk_rope_summary()),
            ("rms_norm", fused.rms_norm_summary()),
        ] {
            assert_eq!(summary["backend"], "fused", "{path}: {summary}");
        }
        assert_tracks_reference(&fused, &reference);
    }

    #[test]
    fn fused_rms_norm_keeps_the_casting_composition_semantics() {
        use candle_core::Module;
        let device = &Device::Cpu;
        let weight = Tensor::from_vec(
            (0..64)
                .map(|i| 1.0 + (i as f32 * 0.3).sin() * 0.2)
                .collect::<Vec<_>>(),
            64,
            device,
        )
        .unwrap();
        let mut norm = super::Qwen36RmsNorm::new(weight, 1e-6);
        let input = |seed: f32| {
            Tensor::from_vec(
                (0..128)
                    .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * 3.0)
                    .collect::<Vec<_>>(),
                (2, 1, 64),
                device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
        };
        let (residual, delta) = (input(1.0), input(2.0));
        let (legacy_sum, legacy) = norm.add_forward(&residual, &delta).unwrap();
        norm.fused = true;
        let (fused_sum, fused) = norm.add_forward(&residual, &delta).unwrap();
        assert_eq!(fused.dtype(), DType::BF16, "activation dtype is preserved");
        assert_eq!(flat(&fused_sum), flat(&legacy_sum));
        assert_eq!(flat(&fused), flat(&legacy));
        // F32 activations keep the plain path even when fusion is enabled.
        let f32_input = residual.to_dtype(DType::F32).unwrap();
        assert_eq!(norm.forward(&f32_input).unwrap().dtype(), DType::F32);
    }

    #[test]
    fn fused_gdn_decode_tracks_the_candle_chain_across_steps() {
        for order in [
            Qwen36LinearVHeadOrder::Grouped,
            Qwen36LinearVHeadOrder::Tiled,
        ] {
            let legacy = gdn_mixer(order);
            let mut fused = gdn_mixer(order);
            fused.resolve_fused_decode(128, &Device::Cpu, true);
            assert_eq!(fused.fused_decode_path, Qwen36FusedPath::Fused, "{order:?}");
            let (mut legacy_state, mut fused_state) = (gdn_state(0.0), gdn_state(0.0));
            for step in 0..4 {
                let x = Tensor::from_vec(
                    (0..128)
                        .map(|i| ((i + 37 * step) as f32 * 0.754_877_7).sin() * 1.5)
                        .collect::<Vec<_>>(),
                    (1, 1, 128),
                    &Device::Cpu,
                )
                .unwrap();
                let expected = legacy.forward(&x, &mut legacy_state).unwrap();
                let actual = fused.forward(&x, &mut fused_state).unwrap();
                assert_close(
                    &flat(&actual),
                    &flat(&expected),
                    1e-4,
                    &format!("{order:?} step {step}"),
                );
            }
            let (
                Qwen36LayerRuntimeState::Linear {
                    conv_state: Some(legacy_ring),
                    recurrent_state: Some(legacy_recurrent),
                },
                Qwen36LayerRuntimeState::Linear {
                    conv_state: Some(fused_ring),
                    recurrent_state: Some(fused_recurrent),
                },
            ) = (&legacy_state, &fused_state)
            else {
                panic!("decode must keep the hybrid state");
            };
            assert_eq!(legacy_ring.next_idx, fused_ring.next_idx);
            for (l, f) in legacy_ring.slots.iter().zip(&fused_ring.slots) {
                assert_eq!(flat(l), flat(f));
            }
            assert_close(
                &flat(fused_recurrent),
                &flat(legacy_recurrent),
                1e-5,
                "state",
            );
        }
    }

    fn compact_projection(rows: usize, seed: u64) -> Qwen36Projection {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let bytes = (0..rows * 128)
            .map(|_| loop {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let byte = state as u8;
                if byte & 0x7f != 0x7f {
                    break byte;
                }
            })
            .collect::<Vec<_>>();
        Qwen36Projection::CompactFp8 {
            weights: Tensor::from_vec(bytes, (rows, 128), &Device::Cpu).unwrap(),
            scales: Tensor::from_vec(
                (0..rows / 128)
                    .map(|i| 0.01 + i as f32 * 0.003)
                    .collect::<Vec<_>>(),
                (rows / 128, 1),
                &Device::Cpu,
            )
            .unwrap(),
        }
    }

    #[test]
    fn packed_projections_reproduce_each_part_through_views() {
        let (mut a, mut b) = (compact_projection(256, 1), compact_projection(128, 2));
        let (a0, b0) = (a.clone(), b.clone());
        let packed = super::pack_projections(&mut [&mut a, &mut b])
            .expect("block-aligned block-FP8 parts must pack");
        let x = Tensor::from_vec(
            (0..2 * 128)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            (2, 128),
            &Device::Cpu,
        )
        .unwrap();
        let out = packed.forward(&x).unwrap();
        assert_eq!(out.dims(), [2, 384]);
        assert_eq!(
            flat(&out.narrow(1, 0, 256).unwrap()),
            flat(&a0.forward(&x).unwrap())
        );
        assert_eq!(
            flat(&out.narrow(1, 256, 128).unwrap()),
            flat(&b0.forward(&x).unwrap())
        );
        // The parts are now views into the packed weights and still agree.
        assert_eq!(
            flat(&a.forward(&x).unwrap()),
            flat(&a0.forward(&x).unwrap())
        );
        assert_eq!(
            flat(&b.forward(&x).unwrap()),
            flat(&b0.forward(&x).unwrap())
        );
        let Qwen36Projection::CompactFp8 { weights, .. } = &b else {
            panic!("packed part must stay block-FP8");
        };
        assert_eq!(weights.layout().start_offset(), 256 * 128);

        // Parts off the 128-row block grid are refused and left untouched.
        let (mut c, mut d) = (compact_projection(128, 3), compact_projection(128, 4));
        let Qwen36Projection::CompactFp8 { weights, scales } = &d else {
            unreachable!()
        };
        d = Qwen36Projection::CompactFp8 {
            weights: weights.narrow(0, 0, 64).unwrap(),
            scales: scales.clone(),
        };
        let d0 = d.clone();
        assert!(super::pack_projections(&mut [&mut c, &mut d]).is_none());
        let (
            Qwen36Projection::CompactFp8 { weights: left, .. },
            Qwen36Projection::CompactFp8 { weights: right, .. },
        ) = (&d, &d0)
        else {
            unreachable!()
        };
        assert_eq!(left.dims(), right.dims());
    }

    #[test]
    fn fused_gdn_decode_with_packed_projections_tracks_the_chain() {
        let legacy = gdn_mixer(Qwen36LinearVHeadOrder::Grouped);
        let mut fused = gdn_mixer(Qwen36LinearVHeadOrder::Grouped);
        fused.pack_projections();
        assert!(fused.packed_in.is_some() && fused.packed_beta_alpha.is_some());
        fused.resolve_fused_decode(128, &Device::Cpu, true);
        assert_eq!(fused.fused_decode_path, Qwen36FusedPath::Fused);
        let (mut legacy_state, mut fused_state) = (gdn_state(3.0), gdn_state(3.0));
        for step in 0..3 {
            let x = Tensor::from_vec(
                (0..128)
                    .map(|i| ((i + 11 * step) as f32 * 0.754_877_7).sin())
                    .collect::<Vec<_>>(),
                (1, 1, 128),
                &Device::Cpu,
            )
            .unwrap();
            let expected = legacy.forward(&x, &mut legacy_state).unwrap();
            let actual = fused.forward(&x, &mut fused_state).unwrap();
            assert_close(
                &flat(&actual),
                &flat(&expected),
                1e-4,
                &format!("step {step}"),
            );
        }
    }

    #[test]
    fn fused_gdn_batched_decode_matches_scalar_rows() {
        let mut mixer = gdn_mixer(Qwen36LinearVHeadOrder::Grouped);
        mixer.resolve_fused_decode(128, &Device::Cpu, true);
        assert!(mixer.fused_decode.is_some());
        let input = Tensor::from_vec(
            (0..256)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            (2, 1, 128),
            &Device::Cpu,
        )
        .unwrap();
        let mut scalar_states = [gdn_state(1.0), gdn_state(2.0)];
        let scalar = (0..2)
            .map(|row| {
                flat(
                    &mixer
                        .forward(
                            &input.i(row).unwrap().unsqueeze(0).unwrap(),
                            &mut scalar_states[row],
                        )
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
            .concat();
        let mut batch_states = [gdn_state(1.0), gdn_state(2.0)];
        let mut refs = batch_states.iter_mut().collect::<Vec<_>>();
        let batched = flat(&mixer.forward_decode_batch(&input, &mut refs).unwrap());
        assert_close(&batched, &scalar, 1e-4, "batch vs scalar");
    }

    #[test]
    fn fused_gdn_self_check_rejects_the_wrong_head_order() {
        let mixer = gdn_mixer(Qwen36LinearVHeadOrder::Grouped);
        let wrong = GdnDecodeSpec {
            key_heads: 2,
            value_heads: 4,
            grouped: false,
            norm_eps: 1e-6,
        };
        let error = mixer
            .fused_decode_self_check(&wrong, 128, &Device::Cpu)
            .expect_err("a mismatched head order must fail the self-check");
        assert!(format!("{error}").contains("diverges"), "{error}");
    }

    #[test]
    fn fused_gdn_decode_stays_off_outside_cuda_in_production() {
        let mut mixer = gdn_mixer(Qwen36LinearVHeadOrder::Grouped);
        mixer.resolve_fused_decode(128, &Device::Cpu, false);
        assert!(mixer.fused_decode.is_none());
        assert!(matches!(
            &mixer.fused_decode_path,
            Qwen36FusedPath::Legacy { reason } if reason.contains("CUDA and Metal only")
        ));
    }

    #[test]
    fn linear_attention_computes_in_f32_under_bf16_activations() {
        // Mirror the CUDA native plan in miniature: CompactFp8 projections
        // (the CPU fp8 decode kernel returns the activation's dtype, so the
        // block input is BF16) against an F32 state arena. The whole conv →
        // recurrence → gated-norm island must run in F32 and hand the trunk
        // back its BF16 activation dtype; before the F32 island this failed
        // with a mixed-dtype binary op inside the conv step.
        let device = &Device::Cpu;
        let fp8_projection = |rows: usize, cols: usize| {
            Qwen36Projection::CompactFp8 {
                weights: Tensor::from_vec(vec![0x38u8; rows * cols], (rows, cols), device).unwrap(),
                scales: Tensor::from_vec(vec![1f32], (1, 1), device).unwrap(),
            }
        };
        let dense_f32 = |values: &[f32], shape: (usize, usize)| {
            Tensor::from_vec(values.to_vec(), shape, device).unwrap()
        };
        let conv_kernel = dense_f32(
            &(0..24)
                .map(|index| 0.01 * ((index % 4) + 1) as f32)
                .collect::<Vec<_>>(),
            (6, 4),
        );
        let conv_kernel_slices = super::pre_slice_conv_kernel(&conv_kernel, 4).unwrap();
        let mixer = Qwen36LinearAttention {
            qkv_proj: fp8_projection(6, 4),
            gate_proj: fp8_projection(2, 4),
            beta_proj: fp8_projection(1, 4),
            alpha_proj: fp8_projection(1, 4),
            dt_bias: Tensor::zeros((1, 1, 1), DType::F32, device).unwrap(),
            a: Tensor::full(-0.5f32, (1, 1, 1), device).unwrap(),
            conv_kernel,
            conv_kernel_slices,
            norm: Qwen36GatedRmsNorm {
                weight: Tensor::ones(2, DType::F32, device).unwrap(),
                eps: 1e-6,
            },
            out_proj: fp8_projection(4, 2),
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_dim: 6,
            kernel_size: 4,
            v_head_order: Qwen36LinearVHeadOrder::Tiled,
            tiled_recurrence_enabled: false,
            tiled_recurrence_tile_size_override: None,
            fused_decode: None,
            fused_decode_path: Qwen36FusedPath::legacy("test"),
            packed_in: None,
            packed_beta_alpha: None,
        };

        let bf16_input = |values: &[f32], seq: usize| {
            Tensor::from_vec(values.to_vec(), (1, seq, 4), device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let new_state = || Qwen36LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState {
                slots: (0..3)
                    .map(|_| Tensor::zeros((6, 1), DType::F32, device).unwrap())
                    .collect(),
                next_idx: 0,
            }),
            recurrent_state: None,
        };

        // Decode step (seq == 1).
        let mut state = new_state();
        let output = mixer
            .forward(&bf16_input(&[0.2, -0.1, 0.3, 0.4], 1), &mut state)
            .unwrap();
        assert_eq!(output.dtype(), DType::BF16, "trunk dtype must be restored");
        assert!(
            output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()),
            "decode output must be finite"
        );
        let Qwen36LayerRuntimeState::Linear {
            recurrent_state: Some(recurrent),
            ..
        } = &state
        else {
            panic!("decode must leave a recurrent state")
        };
        assert_eq!(recurrent.dtype(), DType::F32, "state arena stays F32");

        // Prefill (seq > 1) exercises the sequence path and its conv ring.
        let mut state = new_state();
        let output = mixer
            .forward_sequence(
                &bf16_input(&[0.2, -0.1, 0.3, 0.4, -0.3, 0.5, 0.1, 0.2], 2),
                &mut state,
            )
            .unwrap();
        assert_eq!(output.dtype(), DType::BF16);
        let values = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        // Prefill must be numerically stable under the F32 island: identical
        // prefixes of the same sequence agree token-for-token with the decode
        // step on the same prefix.
        let mut decode_state = new_state();
        let decode_output = mixer
            .forward(&bf16_input(&[0.2, -0.1, 0.3, 0.4], 1), &mut decode_state)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (prefill, decode) in values[..2].iter().zip(&decode_output) {
            assert!((prefill - decode).abs() < 1e-4, "{prefill} != {decode}");
        }
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
        let Some(device) = crate::backends::metal_device_if_available(0) else {
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
    fn repeat_head_states_uses_tiled_order() {
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu)
            .expect("tensor should build");
        let repeated = repeat_head_states(&x, 2).expect("repeat should succeed");
        let values = repeated.to_vec3::<f32>().expect("values");

        assert_eq!(
            values,
            vec![vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0]
            ]]
        );
    }

    #[test]
    fn repeat_interleave_head_states_uses_grouped_order() {
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
        let seq = x.unsqueeze(1).expect("seq axis");
        let repeated_seq =
            repeat_interleave_head_states_seq(&seq, 2).expect("repeat should succeed");
        assert_eq!(
            repeated_seq
                .squeeze(1)
                .expect("squeeze")
                .to_vec3::<f32>()
                .expect("values"),
            repeated.to_vec3::<f32>().expect("values")
        );
    }

    /// The two value-head orders are the same model under llama.cpp's
    /// grouped→tiled value permutation (`_LinearAttentionVReorderBase`): new
    /// tiled head `n` holds grouped head `(n % K) * r + n / K`, and must read
    /// the same key head under tiled expansion that the grouped head reads
    /// under `repeat_interleave`. Pins the equivalence that lets the trunk
    /// serve both GGUF (tiled) and native HF (grouped) checkpoints.
    #[test]
    fn tiled_and_grouped_expansions_agree_under_the_value_permutation() {
        let (num_k, repeats, dim) = (16usize, 2usize, 3usize);
        let keys: Vec<f32> = (0..num_k * dim).map(|v| v as f32).collect();
        let x = Tensor::from_vec(keys, (1, num_k, dim), &Device::Cpu).expect("tensor");
        let tiled = repeat_head_states(&x, repeats)
            .expect("tiled")
            .to_vec3::<f32>()
            .unwrap();
        let grouped = repeat_interleave_head_states(&x, repeats)
            .expect("grouped")
            .to_vec3::<f32>()
            .unwrap();
        for n in 0..num_k * repeats {
            let grouped_head = (n % num_k) * repeats + n / num_k;
            assert_eq!(tiled[0][n], grouped[0][grouped_head], "tiled head {n}");
        }
    }

    /// MTP rollback must restore the conv history in logical order: a ring
    /// whose `next_idx` has wrapped mid-buffer comes back with the same
    /// oldest-to-newest sequence (the restore rebuilds it at `next_idx = 0`).
    #[test]
    fn linear_state_snapshot_restores_a_wrapped_conv_ring_in_order() {
        let slot = |value: f32| Tensor::full(value, (1, 2), &Device::Cpu).unwrap();
        let logical = |state: &Qwen36TextRuntimeState| -> Vec<f32> {
            let Qwen36LayerRuntimeState::Linear {
                conv_state: Some(ring),
                ..
            } = &state.layers[0]
            else {
                panic!("linear layer with a conv ring");
            };
            ring.ordered_slots()
                .map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0])
                .collect()
        };
        // Physical [3, 1, 2] with next_idx 1 = logical oldest→newest [1, 2, 3].
        let mut state = Qwen36TextRuntimeState {
            layers: vec![Qwen36LayerRuntimeState::Linear {
                conv_state: Some(ConvRingState {
                    slots: vec![slot(3.0), slot(1.0), slot(2.0)],
                    next_idx: 1,
                }),
                recurrent_state: Some(slot(9.0)),
            }],
        };
        assert_eq!(logical(&state), vec![1.0, 2.0, 3.0]);
        let snapshot = state.snapshot_linear_states().unwrap();

        // Advance the ring past the snapshot, then roll back.
        if let Qwen36LayerRuntimeState::Linear {
            conv_state: Some(ring),
            ..
        } = &mut state.layers[0]
        {
            ring.push_decode(&slot(4.0)).unwrap();
        }
        assert_eq!(logical(&state), vec![2.0, 3.0, 4.0]);
        state.restore_linear_states(&snapshot).unwrap();
        assert_eq!(logical(&state), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_head_states_seq_uses_tiled_order() {
        let x = Tensor::from_vec(
            vec![
                // seq 0
                1f32, 2.0, 3.0, 4.0, // seq 1
                5.0, 6.0, 7.0, 8.0,
            ],
            (1, 2, 2, 2),
            &Device::Cpu,
        )
        .expect("tensor should build");

        let repeated = repeat_head_states_seq(&x, 2).expect("repeat should succeed");
        let values = repeated
            .reshape((1, 2, 8))
            .expect("reshape")
            .to_vec3::<f32>()
            .expect("values");

        assert_eq!(
            values,
            vec![vec![
                vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]
            ]]
        );
    }

    #[test]
    fn compact_conv_ring_drops_full_prefill_backing() {
        let backing = Tensor::zeros((1, 40, 32), DType::F32, &Device::Cpu).unwrap();
        let mut slots = Vec::new();
        for token_idx in 37..40 {
            slots.push(backing.i((0, token_idx)).unwrap().reshape((32, 1)).unwrap());
        }
        let mut state = Qwen36TextRuntimeState {
            layers: vec![Qwen36LayerRuntimeState::Linear {
                conv_state: Some(ConvRingState { slots, next_idx: 0 }),
                recurrent_state: None,
            }],
        };

        let retained_prefill_bytes = state.allocated_session_bytes().unwrap();
        assert!(retained_prefill_bytes >= 40 * 32 * 4);

        let Qwen36LayerRuntimeState::Linear { conv_state, .. } = &mut state.layers[0] else {
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
        let mut ring = ConvRingState { slots, next_idx: 0 };
        let projection = Tensor::zeros((1, 1, 32), DType::F32, &Device::Cpu).unwrap();
        let current = projection.i((0, 0)).unwrap().reshape((32, 1)).unwrap();
        let projection_storage = tensor_storage_address(&current);

        ring.push_decode(&current)
            .expect("ring push should succeed");
        assert_eq!(tensor_storage_address(&ring.slots[0]), projection_storage);
        drop(current);
        drop(projection);

        let state = Qwen36TextRuntimeState {
            layers: vec![Qwen36LayerRuntimeState::Linear {
                conv_state: Some(ring),
                recurrent_state: None,
            }],
        };
        assert_eq!(state.allocated_session_bytes(), Some(3 * 32 * 4));
    }

    #[test]
    fn persistent_zero_states_have_independent_storage() {
        let first = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let second = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let state = Qwen36TextRuntimeState {
            layers: vec![
                Qwen36LayerRuntimeState::Linear {
                    conv_state: None,
                    recurrent_state: Some(first),
                },
                Qwen36LayerRuntimeState::Linear {
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

    /// Half-precision plans must round cos/sin, never the angle: at long
    /// positions a BF16 angle is phase noise and an F16 angle overflows.
    #[test]
    fn build_mrope_keeps_long_position_angles_in_f32() {
        // Qwen3.6-35B-A3B rotary geometry: 64 rotary dims, theta 1e7.
        let half_dim = 32;
        let inv_freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / 10_000_000f32.powf(2.0 * i as f32 / 64.0))
            .collect();
        for position in [4_096usize, 70_000] {
            let (cos_ref, sin_ref) = build_mrope(
                64,
                [position; 3],
                &[11, 11, 10],
                &inv_freqs,
                &Device::Cpu,
                DType::F32,
            )
            .expect("f32 mrope");
            let cos_ref = cos_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let sin_ref = sin_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for dtype in [DType::BF16, DType::F16] {
                let (cos, sin) = build_mrope(
                    64,
                    [position; 3],
                    &[11, 11, 10],
                    &inv_freqs,
                    &Device::Cpu,
                    dtype,
                )
                .expect("half-precision mrope");
                assert_eq!(cos.dtype(), dtype);
                for (values, reference) in [(cos, &cos_ref), (sin, &sin_ref)] {
                    let values = values
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    for (value, expected) in values.iter().zip(reference.iter()) {
                        assert!(
                            (value - expected).abs() < 1e-2,
                            "{dtype:?} position {position}: {value} vs {expected}"
                        );
                    }
                }
            }
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
    fn qwen35_cuda_rope_kernel_defaults_on_with_explicit_rollback() {
        assert!(qwen35_rope_kernel_policy(true, false, None));
        assert!(qwen35_rope_kernel_policy(false, true, None));
        assert!(qwen35_rope_kernel_policy(false, true, Some(true)));
        assert!(!qwen35_rope_kernel_policy(true, false, Some(false)));
        assert!(!qwen35_rope_kernel_policy(false, false, Some(true)));
    }

    fn synthetic_decode_state(
        device: &Device,
        layer_count: usize,
        num_v_heads: usize,
    ) -> Qwen36TextRuntimeState {
        let mut layers = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            if (layer_idx + 1).is_multiple_of(4) {
                layers.push(Qwen36LayerRuntimeState::Full);
            } else {
                let conv_width = num_v_heads * 2;
                layers.push(Qwen36LayerRuntimeState::Linear {
                    conv_state: Some(ConvRingState {
                        slots: (0..3)
                            .map(|_| Tensor::zeros((conv_width, 1), DType::F32, device).unwrap())
                            .collect(),
                        next_idx: 0,
                    }),
                    recurrent_state: Some(
                        Tensor::zeros((1, num_v_heads, 2, 2), DType::F32, device).unwrap(),
                    ),
                });
            }
        }
        Qwen36TextRuntimeState { layers }
    }

    fn advance_synthetic_decode_state(
        state: &mut Qwen36TextRuntimeState,
        device: &Device,
        num_v_heads: usize,
    ) {
        for layer in &mut state.layers {
            match layer {
                Qwen36LayerRuntimeState::Linear {
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
                Qwen36LayerRuntimeState::Full => {}
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
