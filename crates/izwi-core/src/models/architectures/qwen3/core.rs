//! Minimal Qwen3 decoder implementation for native inference.
//!
//! Adapted from the Qwen3 architecture to allow embedding overrides
//! (used for audio-conditioned ASR).

use candle_core::quantized::{ggml_file, GgmlDType, QMatMul, QTensor};

use crate::models::shared::memory::exact::exact_concat_rows;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{ops, rotary_emb, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv as candle_repeat_kv;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::decode_graph::{GraphKey, Qwen3DecodeGraph};
#[cfg(test)]
use crate::backends::kv::KvArena;
use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::error::{Error, Result};
use crate::kernels::cuda::decode_step::{self, DecodeStepLayout, DecodeStepRow};
use crate::kernels::cuda::kquant::{self, KQuantWeight};
use crate::kernels::cuda::split_decode;
use crate::kernels::{try_fused_qk_rms_norm, try_fused_silu_mul_with_status};
use crate::kv::v2::{
    AttentionMask, AttentionPattern, CheckpointPolicy, InferenceStateContract, KeyEncoding,
    PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec, PlacementPolicy,
    PrefixPolicy, StateClock, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec,
    StateGroupId, StateGroupSpec, StateScope, CURRENT_INFERENCE_STATE_ABI,
};
use crate::kv::KvDecodeBatchMetadata;
use crate::kv::KvSlotRef;
#[cfg(test)]
use crate::kv::{CacheBlockRef, KvLayerBinding};
use crate::models::shared::attention::flash::try_fused_self_attention;
use crate::models::shared::attention::geometry::AttentionGeometry;
use crate::models::shared::attention::gqa::{compact_gqa_sdpa_bhsd, CompactGqaMask};
use crate::models::shared::moe::{
    ExpertSet, ExpertActivationCounters, SparseMoeConfig, SparseMoeDispatcher,
};
use crate::models::shared::attention::mixed::{MixedRowLayout, PreparedPhysicalMixedStep};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::telemetry::{record_rope_kernel, record_rope_manual};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::models::shared::weights::mlx;
use crate::models::shared::weights::token_table::QuantizedTokenTable;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeScalingConfig {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub lm_head_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond_dim: usize,
    // Sparse (MoE) feed-forward geometry. Absent/zero on dense checkpoints;
    // when present, every decoder layer's FFN becomes a sparse expert block.
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
}

impl Qwen3Config {
    pub(crate) fn attention_geometry(&self) -> Result<AttentionGeometry> {
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::ModelLoadError(format!(
                "Qwen3 attention geometry is invalid: rope_theta must be finite and positive, got {}",
                self.rope_theta
            )));
        }
        let geometry = AttentionGeometry::from_hidden_size(
            "Qwen3",
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )?;
        geometry.validate_rotary_dim("Qwen3", geometry.key_head_dim())?;
        Ok(geometry)
    }

    /// Compatibility accessor for callers that only inspect a parsed config.
    /// Model construction uses [`Self::attention_geometry`] and returns a typed
    /// error for invalid geometry rather than relying on this sentinel value.
    pub fn head_dim(&self) -> usize {
        match self.head_dim {
            Some(head_dim) => head_dim,
            None if self.num_attention_heads != 0
                && self.hidden_size.is_multiple_of(self.num_attention_heads) =>
            {
                self.hidden_size / self.num_attention_heads
            }
            None => 0,
        }
    }

    pub fn kv_groups(&self) -> usize {
        if self.num_key_value_heads != 0
            && self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            self.num_attention_heads / self.num_key_value_heads
        } else {
            0
        }
    }

    pub fn context_length(&self) -> Option<usize> {
        self.max_position_embeddings.filter(|length| *length > 0)
    }

    pub fn sliding_window(&self) -> Option<usize> {
        self.use_sliding_window
            .then_some(self.sliding_window)
            .flatten()
            .filter(|window| *window > 0)
    }

    /// Resolved sparse-expert geometry, or `None` for a dense checkpoint.
    /// `num_experts > 0` opts the whole decoder into sparse FFNs; the remaining
    /// geometry must then be complete (fail-closed rather than defaulted, so a
    /// mis-keyed MoE checkpoint cannot load half-configured).
    pub fn sparse_moe(&self) -> Result<Option<SparseMoeConfig>> {
        let Some(num_experts) = self.num_experts.filter(|count| *count > 0) else {
            return Ok(None);
        };
        let Some(num_experts_per_tok) = self.num_experts_per_tok else {
            return Err(Error::ModelLoadError(
                "Qwen3 MoE checkpoint sets num_experts without num_experts_per_tok".into(),
            ));
        };
        let Some(moe_intermediate_size) = self.moe_intermediate_size else {
            return Err(Error::ModelLoadError(
                "Qwen3 MoE checkpoint sets num_experts without moe_intermediate_size".into(),
            ));
        };
        if moe_intermediate_size == 0 {
            return Err(Error::ModelLoadError(
                "Qwen3 MoE moe_intermediate_size must be positive".into(),
            ));
        }
        Ok(Some(SparseMoeConfig {
            num_experts,
            num_experts_per_tok,
            norm_topk_prob: self.norm_topk_prob.unwrap_or(true),
        }))
    }

    pub fn moe_intermediate_size(&self) -> Option<usize> {
        self.moe_intermediate_size
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3DecoderCacheGeometry {
    pub domain: StateDomainId,
    pub clock: StateClock,
    pub num_layers: usize,
    pub num_query_heads: usize,
    pub num_kv_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub sliding_window: Option<usize>,
    pub storage_dtype: DType,
    pub preferred_page_tokens: usize,
    pub prefix: PrefixPolicy,
}

pub(crate) fn qwen3_decoder_cache_domain(
    geometry: Qwen3DecoderCacheGeometry,
) -> Result<PagedAttentionDomainSpec> {
    let num_layers = u32::try_from(geometry.num_layers)
        .map_err(|_| Error::InvalidInput("Qwen3 layer count exceeds u32".into()))?;
    let num_query_heads = u32::try_from(geometry.num_query_heads)
        .map_err(|_| Error::InvalidInput("Qwen3 query head count exceeds u32".into()))?;
    let num_kv_heads = u32::try_from(geometry.num_kv_heads)
        .map_err(|_| Error::InvalidInput("Qwen3 KV head count exceeds u32".into()))?;
    let key_head_dim = u32::try_from(geometry.key_head_dim)
        .map_err(|_| Error::InvalidInput("Qwen3 key head dimension exceeds u32".into()))?;
    let value_head_dim = u32::try_from(geometry.value_head_dim)
        .map_err(|_| Error::InvalidInput("Qwen3 value head dimension exceeds u32".into()))?;
    let preferred = u32::try_from(geometry.preferred_page_tokens.max(1))
        .map_err(|_| Error::InvalidInput("Qwen3 page size exceeds u32".into()))?;
    let max_page_tokens = preferred.max(256);
    let attention = match geometry.sliding_window {
        Some(window_tokens) => AttentionPattern::SlidingWindow {
            window_tokens: u32::try_from(window_tokens)
                .map_err(|_| Error::InvalidInput("Qwen3 sliding window exceeds u32".into()))?,
        },
        None => AttentionPattern::Full,
    };
    let storage_dtype = match geometry.storage_dtype {
        DType::F32 => StateDType::F32,
        DType::F16 => StateDType::F16,
        DType::BF16 => StateDType::Bf16,
        dtype => {
            return Err(Error::InvalidInput(format!(
                "Qwen3 managed KV does not support {dtype:?} storage"
            )))
        }
    };

    let layers = (0..num_layers)
        .map(|model_layer| PagedAttentionLayerSpec {
            model_layer,
            query_heads: num_query_heads,
            kv_heads: num_kv_heads,
            key_head_dim,
            value_head_dim,
            pattern: attention,
            mask: AttentionMask::Causal,
            key_encoding: KeyEncoding::Rotary {
                rotary_dim: key_head_dim,
            },
            attention_logit_softcap: None,
        })
        .collect();
    Ok(PagedAttentionDomainSpec {
        header: StateDomainHeader {
            id: geometry.domain,
            scope: StateScope::Retained,
            clock: geometry.clock,
            placement: PlacementPolicy::BackendLocalWithHostOffload,
            checkpoint: if matches!(geometry.prefix, PrefixPolicy::Disabled) {
                CheckpointPolicy::Transactional
            } else {
                CheckpointPolicy::CopyOnWrite
            },
            prefix: geometry.prefix,
        },
        layers,
        page_size: PageSizeConstraint {
            min_tokens: 1,
            preferred_tokens: preferred,
            max_tokens: max_page_tokens,
            multiple_of: 1,
        },
        accepted_dtypes: vec![storage_dtype],
    })
}

pub(crate) fn qwen3_dense_projection_fusion_enabled(device: &Device) -> bool {
    let override_enabled = crate::env::flag("IZWI_QWEN3_DENSE_PROJECTION_FUSION");
    qwen3_dense_projection_fusion_policy(device.is_metal(), override_enabled)
}

pub(crate) fn qwen3_dense_projection_fusion_policy(
    is_metal: bool,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn qwen3_qk_rms_norm_fusion_enabled(device: &Device) -> bool {
    static OVERRIDE: crate::env::ReadOnce<Option<bool>> =
        crate::env::ReadOnce::new(|| crate::env::flag("IZWI_QWEN3_QK_RMS_NORM_FUSION"));
    let override_enabled = OVERRIDE.get();
    qwen3_qk_rms_norm_fusion_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn qwen3_qk_rms_norm_fusion_policy(
    is_metal: bool,
    is_cuda: bool,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal || is_cuda)
}

fn qwen3_quantized_projection_grouping_enabled(device: &Device, cfg: &Qwen3Config) -> bool {
    qwen3_quantized_projection_grouping_for_hidden_size(device, cfg.hidden_size)
}

/// Whether the quantized Q/K/V and gate/up projections of a trunk with
/// `hidden_size` are packed into one `QMatMul` each on `device`. Load admission
/// prices the packed form (`runtime/lifecycle/asr_memory.rs`).
pub(crate) fn qwen3_quantized_projection_grouping_for_hidden_size(
    device: &Device,
    hidden_size: usize,
) -> bool {
    let override_enabled = crate::env::flag("IZWI_QWEN3_QUANTIZED_PROJECTION_GROUPING");
    qwen3_quantized_projection_grouping_policy(device.is_metal(), hidden_size, override_enabled)
}

pub(crate) fn qwen3_quantized_projection_grouping_policy(
    is_metal: bool,
    hidden_size: usize,
    override_enabled: Option<bool>,
) -> bool {
    is_metal && override_enabled.unwrap_or(hidden_size <= 1024)
}

struct Qwen3DenseProjection {
    linear: Linear,
    has_bias: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen3RuntimeProfileSnapshot {
    pub qmatmul_calls: u64,
    pub qmatmul_ns: u64,
    pub qmatmul_input_casts: u64,
    pub qmatmul_input_cast_ns: u64,
    pub qmatmul_output_casts: u64,
    pub qmatmul_output_cast_ns: u64,
    pub lm_head_calls: u64,
    pub lm_head_ns: u64,
    pub silu_mul_fused_calls: u64,
    pub silu_mul_fallback_calls: u64,
}

impl Qwen3RuntimeProfileSnapshot {
    fn saturating_delta(self, baseline: Self) -> Self {
        Self {
            qmatmul_calls: self.qmatmul_calls.saturating_sub(baseline.qmatmul_calls),
            qmatmul_ns: self.qmatmul_ns.saturating_sub(baseline.qmatmul_ns),
            qmatmul_input_casts: self
                .qmatmul_input_casts
                .saturating_sub(baseline.qmatmul_input_casts),
            qmatmul_input_cast_ns: self
                .qmatmul_input_cast_ns
                .saturating_sub(baseline.qmatmul_input_cast_ns),
            qmatmul_output_casts: self
                .qmatmul_output_casts
                .saturating_sub(baseline.qmatmul_output_casts),
            qmatmul_output_cast_ns: self
                .qmatmul_output_cast_ns
                .saturating_sub(baseline.qmatmul_output_cast_ns),
            lm_head_calls: self.lm_head_calls.saturating_sub(baseline.lm_head_calls),
            lm_head_ns: self.lm_head_ns.saturating_sub(baseline.lm_head_ns),
            silu_mul_fused_calls: self
                .silu_mul_fused_calls
                .saturating_sub(baseline.silu_mul_fused_calls),
            silu_mul_fallback_calls: self
                .silu_mul_fallback_calls
                .saturating_sub(baseline.silu_mul_fallback_calls),
        }
    }
}

static QWEN3_PROFILE_QMATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_QMATMUL_NS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_QMATMUL_INPUT_CASTS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_QMATMUL_INPUT_CAST_NS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_QMATMUL_OUTPUT_CASTS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_QMATMUL_OUTPUT_CAST_NS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_LM_HEAD_CALLS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_LM_HEAD_NS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_SILU_MUL_FUSED_CALLS: AtomicU64 = AtomicU64::new(0);
static QWEN3_PROFILE_SILU_MUL_FALLBACK_CALLS: AtomicU64 = AtomicU64::new(0);

/// Device-to-host readbacks of MRoPE position ids (each one is a sync).
static QWEN3_MROPE_POSITION_READBACKS: AtomicU64 = AtomicU64::new(0);
/// RoPE tables built on the host for sequential positions under MRoPE.
static QWEN3_SEQUENTIAL_ROPE_HOST_TABLES: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static QWEN3_MROPE_POSITION_READBACKS_THREAD: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    static QWEN3_ROPE_TABLE_BUILDS_THREAD: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    static QWEN3_HEAD_ROWS_THREAD: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static QWEN3_SDPA_QUERY_BLOCKS_THREAD: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

fn record_mrope_position_readback() {
    QWEN3_MROPE_POSITION_READBACKS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    QWEN3_MROPE_POSITION_READBACKS_THREAD.with(|count| count.set(count.get() + 1));
}

/// Process-wide count of MRoPE position-id readbacks (device-to-host syncs).
pub fn qwen3_mrope_position_readbacks() -> u64 {
    QWEN3_MROPE_POSITION_READBACKS.load(Ordering::Relaxed)
}

/// Process-wide count of host RoPE tables built for sequential MRoPE positions.
pub fn qwen3_sequential_rope_host_tables() -> u64 {
    QWEN3_SEQUENTIAL_ROPE_HOST_TABLES.load(Ordering::Relaxed)
}

#[cfg(test)]
fn thread_mrope_position_readbacks() -> u64 {
    QWEN3_MROPE_POSITION_READBACKS_THREAD.with(std::cell::Cell::get)
}

#[cfg(test)]
fn thread_rope_table_builds() -> u64 {
    QWEN3_ROPE_TABLE_BUILDS_THREAD.with(std::cell::Cell::get)
}

#[cfg(test)]
fn thread_head_rows() -> u64 {
    QWEN3_HEAD_ROWS_THREAD.with(std::cell::Cell::get)
}

#[cfg(test)]
fn thread_sdpa_query_blocks() -> u64 {
    QWEN3_SDPA_QUERY_BLOCKS_THREAD.with(std::cell::Cell::get)
}

pub fn qwen3_runtime_profiling_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_QWEN3_PROFILE", false));
    ENABLED.get()
}

pub fn qwen3_runtime_profile_snapshot() -> Qwen3RuntimeProfileSnapshot {
    Qwen3RuntimeProfileSnapshot {
        qmatmul_calls: QWEN3_PROFILE_QMATMUL_CALLS.load(Ordering::Relaxed),
        qmatmul_ns: QWEN3_PROFILE_QMATMUL_NS.load(Ordering::Relaxed),
        qmatmul_input_casts: QWEN3_PROFILE_QMATMUL_INPUT_CASTS.load(Ordering::Relaxed),
        qmatmul_input_cast_ns: QWEN3_PROFILE_QMATMUL_INPUT_CAST_NS.load(Ordering::Relaxed),
        qmatmul_output_casts: QWEN3_PROFILE_QMATMUL_OUTPUT_CASTS.load(Ordering::Relaxed),
        qmatmul_output_cast_ns: QWEN3_PROFILE_QMATMUL_OUTPUT_CAST_NS.load(Ordering::Relaxed),
        lm_head_calls: QWEN3_PROFILE_LM_HEAD_CALLS.load(Ordering::Relaxed),
        lm_head_ns: QWEN3_PROFILE_LM_HEAD_NS.load(Ordering::Relaxed),
        silu_mul_fused_calls: QWEN3_PROFILE_SILU_MUL_FUSED_CALLS.load(Ordering::Relaxed),
        silu_mul_fallback_calls: QWEN3_PROFILE_SILU_MUL_FALLBACK_CALLS.load(Ordering::Relaxed),
    }
}

pub fn qwen3_runtime_profile_delta(
    baseline: Qwen3RuntimeProfileSnapshot,
) -> Qwen3RuntimeProfileSnapshot {
    qwen3_runtime_profile_snapshot().saturating_delta(baseline)
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn record_qwen3_silu_mul_profile(fused: bool) {
    if !qwen3_runtime_profiling_enabled() {
        return;
    }
    let counter = if fused {
        &QWEN3_PROFILE_SILU_MUL_FUSED_CALLS
    } else {
        &QWEN3_PROFILE_SILU_MUL_FALLBACK_CALLS
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

enum Qwen3Projection {
    Dense(Qwen3DenseProjection),
    Quantized(Qwen3QuantizedProjection),
}

/// A quantized projection, with how its activations reach the kernel
/// decided once at load.
struct Qwen3QuantizedProjection {
    qmatmul: QMatMul,
    /// On CUDA, Candle's GGUF kernels take this weight's BF16/F16
    /// activations as they are ([`qwen3_cuda_qmatmul_takes_half_activations`]),
    /// so no F32 casts wrap the call.
    cuda_half_activations: bool,
    /// On CUDA, the same weight for the izwi-owned K-quant kernels, which
    /// the decode graphs run instead of `QMatMul` (whose process-global
    /// scratch a graph cannot capture); `None` for other types and devices.
    kquant: Option<KQuantWeight>,
}

impl Qwen3QuantizedProjection {
    fn new(qmatmul: QMatMul) -> Self {
        let on_cuda = match &qmatmul {
            QMatMul::QTensor(weight) => weight.device().is_cuda(),
            QMatMul::Tensor(weight) | QMatMul::TensorF16(weight) => weight.device().is_cuda(),
        };
        let cuda_half_activations = on_cuda && qwen3_cuda_qmatmul_takes_half_activations(&qmatmul);
        let kquant = KQuantWeight::from_qmatmul(&qmatmul);
        Self {
            qmatmul,
            cuda_half_activations,
            kquant,
        }
    }

    /// `x · weightᵀ` through the K-quant kernels, as the decode graphs run
    /// it: the activations cast to F32 (exact), quantized to Q8_1 once,
    /// MMVQ up to eight rows and MMQ above, the F32 result cast back to the
    /// activations' dtype. Bit for bit `QMatMul`'s result (`kquant`), which
    /// for 16-bit activations rounds its F32 sums to them the same way.
    fn kquant_forward(&self, weight: &KQuantWeight, x: &Tensor) -> candle_core::Result<Tensor> {
        let activation_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let rows = x.dims()[..x.rank().saturating_sub(1)]
            .iter()
            .product::<usize>();
        let out = if rows <= kquant::MAX_ROWS {
            kquant::mat_vec(weight, &kquant::quantize(&x)?)?
        } else {
            kquant::mat_mul(weight, &kquant::quantize_tiles(&x, weight.tile_scales())?)?
        };
        if out.dtype() == activation_dtype {
            Ok(out)
        } else {
            out.to_dtype(activation_dtype)
        }
    }
}

thread_local! {
    /// Set while a decode graph's body runs on this thread: quantized
    /// projections then take the K-quant kernels ([`GraphProjections`]).
    static GRAPH_PROJECTIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Routes this thread's quantized projections through the K-quant kernels
/// until dropped: the decode graphs' body, eager or captured.
struct GraphProjections {
    previous: bool,
}

impl GraphProjections {
    fn enter() -> Self {
        Self {
            previous: GRAPH_PROJECTIONS.with(|on| on.replace(true)),
        }
    }
}

impl Drop for GraphProjections {
    fn drop(&mut self) {
        GRAPH_PROJECTIONS.with(|on| on.set(self.previous));
    }
}

/// The text model's token embedding: a dense table (safetensors and MLX
/// checkpoints), or a GGUF table kept in its stored quantized form, whose
/// gathered rows are cast to the model dtype. Dequantizing the whole GGUF
/// table instead kept a dense copy beside the stored one (Qwen3-ASR-1.7B:
/// 0.62 GB of F16, rounded to 1 GiB on Metal, next to a byte-identical Q8_0
/// head).
enum Qwen3TokenEmbedding {
    Dense(Embedding),
    Quantized {
        table: QuantizedTokenTable,
        dtype: DType,
    },
}

impl Qwen3TokenEmbedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(embedding) => embedding.forward(ids).map_err(Error::from),
            Self::Quantized { table, dtype } => {
                let rows = table.embed(ids)?;
                if rows.dtype() == *dtype {
                    Ok(rows)
                } else {
                    rows.to_dtype(*dtype).map_err(Error::from)
                }
            }
        }
    }

    fn vocab_size(&self) -> Result<usize> {
        match self {
            Self::Dense(embedding) => Ok(embedding.embeddings().dim(0)?),
            Self::Quantized { table, .. } => Ok(table.vocab_size()),
        }
    }

    /// A head tied to this table, sharing its storage.
    fn tied_head(&self) -> Result<Qwen3Projection> {
        match self {
            Self::Dense(embedding) => {
                Ok(Qwen3Projection::tied_dense(embedding.embeddings().clone()))
            }
            Self::Quantized { table, .. } => Ok(Qwen3Projection::from_qmatmul(table.tied_head()?)),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Dense(embedding) => embedding.embeddings().dtype(),
            Self::Quantized { dtype, .. } => *dtype,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen3ProjectionDiagnostics {
    pub dense_projection_count: usize,
    pub dense_bias_projection_count: usize,
    pub quantized_projection_count: usize,
}

impl Qwen3ProjectionDiagnostics {
    fn add(self, other: Self) -> Self {
        Self {
            dense_projection_count: self
                .dense_projection_count
                .saturating_add(other.dense_projection_count),
            dense_bias_projection_count: self
                .dense_bias_projection_count
                .saturating_add(other.dense_bias_projection_count),
            quantized_projection_count: self
                .quantized_projection_count
                .saturating_add(other.quantized_projection_count),
        }
    }
}

/// Whether MLX 4-bit weights stay quantized on `device`, repacked into Q4_1
/// (`mlx::read_q4_1_blocks`): on Metal, whose ggml kernels take F32
/// activations as they are. CPU and CUDA quantize the activations of a
/// quantized matmul, so they keep the dense dequantization.
pub(crate) fn qwen3_mlx_q4_1_enabled(device: &Device) -> bool {
    device.is_metal()
}

/// The Q4_1 blocks of the MLX 4-bit weights `parts` (each `(prefix,
/// out_dim)` under `vb`, an empty prefix for `vb` itself, `in_dim` wide,
/// without a bias) stacked by rows into one quantized projection, built on
/// the host and uploaded once; `None` when any part loads dense.
fn qwen3_mlx_q4_1_projection(
    vb: &VarBuilder,
    parts: &[(&str, usize)],
    in_dim: usize,
) -> Result<Option<Qwen3Projection>> {
    if !qwen3_mlx_q4_1_enabled(vb.device()) {
        return Ok(None);
    }
    let mut blocks = Vec::new();
    let mut rows = 0;
    for &(prefix, out_dim) in parts {
        let part = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(prefix)
        };
        if part.contains_tensor("bias") {
            return Ok(None);
        }
        let Some(part_blocks) = mlx::read_q4_1_blocks(&part, out_dim, in_dim)? else {
            return Ok(None);
        };
        blocks.extend_from_slice(&part_blocks);
        rows += out_dim;
    }
    // Free the packed words and parameters just read back before the upload:
    // Candle hands a pooled buffer to the next allocation it fits, so the
    // resident blocks would otherwise keep a larger released read buffer
    // (11 MiB over the 4-bit aligner).
    vb.device().synchronize()?;
    let qtensor =
        ggml_file::qtensor_from_ggml(GgmlDType::Q4_1, &blocks, vec![rows, in_dim], vb.device())?;
    Qwen3Projection::from_qtensor(qtensor).map(Some)
}

impl Qwen3Projection {
    fn dense(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        if let Some(quantized) = qwen3_mlx_q4_1_projection(&vb, &[("", out_dim)], in_dim)? {
            return Ok(quantized);
        }
        let has_bias = vb.contains_tensor("bias");
        let linear = if has_bias {
            mlx::load_linear(in_dim, out_dim, vb)
        } else {
            mlx::load_linear_no_bias(in_dim, out_dim, vb)
        }?;
        Ok(Self::Dense(Qwen3DenseProjection { linear, has_bias }))
    }

    fn quantized(loader: &GgufLoader, device: &Device, name: &str) -> Result<Self> {
        let resolved = qwen3_gguf_tensor_name(loader, name).unwrap_or_else(|| name.to_string());
        let weights = Arc::new(loader.load_qtensor(&resolved, device)?);
        QMatMul::from_arc(weights)
            .map(Self::from_qmatmul)
            .map_err(Error::from)
    }

    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        QMatMul::from_qtensor(qtensor)
            .map(Self::from_qmatmul)
            .map_err(Error::from)
    }

    fn from_qmatmul(qmatmul: QMatMul) -> Self {
        Self::Quantized(Qwen3QuantizedProjection::new(qmatmul))
    }

    fn packed_quantized(loader: &GgufLoader, device: &Device, names: &[&str]) -> Result<Self> {
        let resolved = names
            .iter()
            .map(|name| qwen3_gguf_tensor_name(loader, name).unwrap_or_else(|| (*name).to_string()))
            .collect::<Vec<_>>();
        let resolved = resolved.iter().map(String::as_str).collect::<Vec<_>>();
        packed_quantized_qmatmul(loader, device, &resolved).map(Self::from_qmatmul)
    }

    fn tied_dense(weight: Tensor) -> Self {
        Self::Dense(Qwen3DenseProjection {
            linear: Linear::new(weight, None),
            has_bias: false,
        })
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        match self {
            Self::Dense(dense) => Qwen3ProjectionDiagnostics {
                dense_projection_count: 1,
                dense_bias_projection_count: usize::from(dense.has_bias),
                quantized_projection_count: 0,
            },
            Self::Quantized(_) => Qwen3ProjectionDiagnostics {
                dense_projection_count: 0,
                dense_bias_projection_count: 0,
                quantized_projection_count: 1,
            },
        }
    }

    fn logits_for_tokens(&self, hidden: &Tensor, token_ids: &[u32]) -> Result<Vec<(u32, f32)>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let logits = match self {
            Self::Dense(dense) => dense_logits_for_tokens(dense, hidden, token_ids)?,
            Self::Quantized(_) => logits_for_tokens_from_full(&self.forward(hidden)?, token_ids)?,
        };
        let values = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(token_ids.iter().copied().zip(values).collect())
    }
}

fn qwen3_metal_qmatmul_f32_input_enabled(device: &Device) -> bool {
    static OVERRIDE: crate::env::ReadOnce<Option<bool>> =
        crate::env::ReadOnce::new(|| crate::env::flag("IZWI_QWEN3_METAL_QMATMUL_F32_INPUT"));
    let override_enabled = OVERRIDE.get();
    qwen3_metal_qmatmul_f32_input_policy(device.is_metal(), override_enabled)
}

fn qwen3_metal_qmatmul_f32_input_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    is_metal && override_enabled.unwrap_or(true)
}

fn qwen3_qmatmul_input_dtype(
    device: &Device,
    cuda_half_activations: bool,
    activation_dtype: DType,
) -> Option<DType> {
    qwen3_qmatmul_input_dtype_policy(
        device.is_cuda(),
        cuda_half_activations,
        qwen3_metal_qmatmul_f32_input_enabled(device),
        activation_dtype,
    )
}

/// The dtype a quantized projection's activations are cast to before the
/// call (and its output cast back from), or `None` to pass them as they are.
fn qwen3_qmatmul_input_dtype_policy(
    is_cuda: bool,
    cuda_half_activations: bool,
    metal_f32_input_enabled: bool,
    activation_dtype: DType,
) -> Option<DType> {
    if activation_dtype == DType::F32 {
        return None;
    }
    if is_cuda {
        let native = cuda_half_activations && matches!(activation_dtype, DType::BF16 | DType::F16);
        return (!native).then_some(DType::F32);
    }
    metal_f32_input_enabled.then_some(DType::F32)
}

/// Whether Candle 0.11.1's CUDA `QMatMul` takes BF16/F16 activations for
/// `qmatmul` and gives bit-for-bit what casting them to F32, running the F32
/// kernel and casting the result back gives:
/// - a `QTensor` of a type its fast GGUF kernels cover (Q4_0, Q4_1, Q5_0,
///   Q5_1, Q8_0, Q2_K to Q6_K), whose rows are whole blocks. Up to 8 rows
///   MMVQ quantizes the activations to Q8_1 from their exact F32 values,
///   sums in F32 and rounds each sum to the output dtype to nearest even,
///   as the cast kernel does; more rows run MMQ, which makes those casts
///   itself.
/// - not a dequantized `Tensor` (an F16/BF16/F32 GGUF weight, an F32
///   matmul), `TensorF16`, or another quantized type: their fallback kernels
///   read F32 only.
fn qwen3_cuda_qmatmul_takes_half_activations(qmatmul: &QMatMul) -> bool {
    let QMatMul::QTensor(weight) = qmatmul else {
        return false;
    };
    let dtype = weight.dtype();
    let fast_kernels = matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    );
    fast_kernels
        && weight
            .shape()
            .dims2()
            .is_ok_and(|(_, cols)| cols.is_multiple_of(dtype.block_size()))
}

impl Module for Qwen3Projection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(dense) => dense.linear.forward(x),
            Self::Quantized(projection) => {
                if let Some(weight) = projection
                    .kquant
                    .as_ref()
                    .filter(|_| GRAPH_PROJECTIONS.with(std::cell::Cell::get))
                {
                    return projection.kquant_forward(weight, x);
                }
                let activation_dtype = x.dtype();
                let profile = qwen3_runtime_profiling_enabled();
                let x = if let Some(dtype) = qwen3_qmatmul_input_dtype(
                    x.device(),
                    projection.cuda_half_activations,
                    activation_dtype,
                ) {
                    let started = profile.then(Instant::now);
                    let casted = x.to_dtype(dtype)?;
                    if let Some(started) = started {
                        QWEN3_PROFILE_QMATMUL_INPUT_CASTS.fetch_add(1, Ordering::Relaxed);
                        QWEN3_PROFILE_QMATMUL_INPUT_CAST_NS
                            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
                    }
                    casted
                } else if projection.cuda_half_activations {
                    // The fast kernels read contiguous rows; the cast this
                    // replaces always produced them.
                    x.contiguous()?
                } else {
                    x.clone()
                };
                let qmatmul_started = profile.then(Instant::now);
                let out = projection.qmatmul.forward(&x)?;
                if let Some(started) = qmatmul_started {
                    QWEN3_PROFILE_QMATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
                    QWEN3_PROFILE_QMATMUL_NS.fetch_add(elapsed_ns(started), Ordering::Relaxed);
                }
                if out.dtype() != activation_dtype {
                    let started = profile.then(Instant::now);
                    let casted = out.to_dtype(activation_dtype);
                    if let Some(started) = started {
                        QWEN3_PROFILE_QMATMUL_OUTPUT_CASTS.fetch_add(1, Ordering::Relaxed);
                        QWEN3_PROFILE_QMATMUL_OUTPUT_CAST_NS
                            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
                    }
                    casted
                } else {
                    Ok(out)
                }
            }
        }
    }
}

fn packed_quantized_qmatmul(
    loader: &GgufLoader,
    device: &Device,
    names: &[&str],
) -> Result<QMatMul> {
    let mut tensors = Vec::with_capacity(names.len());
    for name in names {
        tensors.push(loader.load_qtensor(name, device)?);
    }
    packed_qmatmul_from_qtensors(&tensors, device)
}

fn packed_qmatmul_from_qtensors(tensors: &[QTensor], device: &Device) -> Result<QMatMul> {
    if tensors.is_empty() {
        return Err(Error::InferenceError(
            "Cannot pack empty Qwen3 quantized projection group".to_string(),
        ));
    }

    let dtype = tensors[0].dtype();
    let (mut total_rows, common_cols) = qtensor_dims2(&tensors[0])?;
    let mut raw = tensors[0].data()?.into_owned();

    for tensor in &tensors[1..] {
        let (rows, cols) = qtensor_dims2(tensor)?;
        if cols != common_cols {
            return Err(Error::InferenceError(format!(
                "Qwen3 quantized projection group input mismatch: expected {common_cols}, got {cols}"
            )));
        }
        if tensor.dtype() != dtype {
            return Err(Error::InferenceError(format!(
                "Qwen3 quantized projection group dtype mismatch: expected {dtype:?}, got {:?}",
                tensor.dtype()
            )));
        }
        raw.extend_from_slice(tensor.data()?.as_ref());
        total_rows = total_rows.saturating_add(rows);
    }

    let expected_bytes = quantized_storage_bytes(total_rows, common_cols, dtype)?;
    if raw.len() != expected_bytes {
        return Err(Error::InferenceError(format!(
            "Qwen3 packed quantized projection has {} bytes, expected {expected_bytes}",
            raw.len()
        )));
    }

    let qtensor = ggml_file::qtensor_from_ggml(dtype, &raw, vec![total_rows, common_cols], device)
        .map_err(Error::from)?;
    QMatMul::from_qtensor(qtensor).map_err(Error::from)
}

fn qtensor_dims2(tensor: &QTensor) -> Result<(usize, usize)> {
    tensor.shape().dims2().map_err(Error::from)
}

fn quantized_storage_bytes(rows: usize, cols: usize, dtype: GgmlDType) -> Result<usize> {
    let elems = rows.checked_mul(cols).ok_or_else(|| {
        Error::InferenceError("Qwen3 quantized projection shape overflows usize".to_string())
    })?;
    if !elems.is_multiple_of(dtype.block_size()) {
        return Err(Error::InferenceError(format!(
            "Qwen3 quantized projection element count {elems} is not divisible by block size {}",
            dtype.block_size()
        )));
    }
    elems
        .checked_div(dtype.block_size())
        .and_then(|blocks| blocks.checked_mul(dtype.type_size()))
        .ok_or_else(|| {
            Error::InferenceError(
                "Qwen3 quantized projection byte size overflows usize".to_string(),
            )
        })
}

fn concat_linear_biases(name: &str, linears: &[&Linear]) -> Result<Option<Tensor>> {
    let bias_count = linears
        .iter()
        .filter(|linear| linear.bias().is_some())
        .count();
    if bias_count == 0 {
        return Ok(None);
    }
    if bias_count != linears.len() {
        return Err(Error::InferenceError(format!(
            "{name} cannot fuse projections with mixed bias presence"
        )));
    }
    let biases = linears
        .iter()
        .map(|linear| linear.bias().expect("bias checked"))
        .collect::<Vec<_>>();
    Tensor::cat(&biases, 0).map(Some).map_err(Error::from)
}

/// Drops the projections a fusion replaced and synchronizes the device so
/// Candle frees their buffers at once. Until a synchronization Candle keeps
/// released buffers pooled and hands each to the next allocation it fits,
/// however much smaller: a norm gain loaded after the fusion kept a whole
/// pre-fusion projection's buffer resident (the dense forced aligner held
/// 0.55 GiB of them on Metal).
fn release_fused_parts<T>(parts: T, device: &Device) -> Result<()> {
    drop(parts);
    device.synchronize().map_err(Error::from)
}

struct Qwen3FusedQkvProjection {
    fused: Linear,
    q_out: usize,
    k_out: usize,
    v_out: usize,
    bias_count: usize,
}

impl Qwen3FusedQkvProjection {
    fn new(
        q_proj: &Qwen3DenseProjection,
        k_proj: &Qwen3DenseProjection,
        v_proj: &Qwen3DenseProjection,
    ) -> Result<Self> {
        let q_weight = q_proj.linear.weight();
        let k_weight = k_proj.linear.weight();
        let v_weight = v_proj.linear.weight();
        let q_out = q_weight.dim(0)?;
        let k_out = k_weight.dim(0)?;
        let v_out = v_weight.dim(0)?;
        let in_dim = q_weight.dim(1)?;
        if k_weight.dim(1)? != in_dim || v_weight.dim(1)? != in_dim {
            return Err(Error::InferenceError(
                "Qwen3 fused QKV projection input dimensions do not match".to_string(),
            ));
        }
        // Joined at its exact size: `Tensor::cat` rounds the joined weight up
        // to a power of two on Metal.
        let weight = exact_concat_rows(&[q_weight, k_weight, v_weight])?;
        let bias = concat_linear_biases(
            "Qwen3 fused QKV projection",
            &[&q_proj.linear, &k_proj.linear, &v_proj.linear],
        )?;
        let bias_count = usize::from(q_proj.has_bias)
            .saturating_add(usize::from(k_proj.has_bias))
            .saturating_add(usize::from(v_proj.has_bias));
        Ok(Self {
            fused: Linear::new(weight, bias),
            q_out,
            k_out,
            v_out,
            bias_count,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.fused.forward(x)?;
        let last_dim = qkv.rank().saturating_sub(1);
        let q = qkv.narrow(last_dim, 0, self.q_out)?;
        let k = qkv.narrow(last_dim, self.q_out, self.k_out)?;
        let v = qkv.narrow(last_dim, self.q_out + self.k_out, self.v_out)?;
        Ok((q, k, v))
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        Qwen3ProjectionDiagnostics {
            dense_projection_count: 3,
            dense_bias_projection_count: self.bias_count,
            quantized_projection_count: 0,
        }
    }
}

struct Qwen3FusedQuantizedQkvProjection {
    fused: Qwen3Projection,
    q_out: usize,
    k_out: usize,
    v_out: usize,
}

impl Qwen3FusedQuantizedQkvProjection {
    fn load_gguf(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let geometry = cfg.attention_geometry()?;
        let q_out = geometry.query_width();
        let k_out = geometry.key_width();
        let v_out = geometry.value_width();
        let names = [
            format!("{prefix}.q_proj.weight"),
            format!("{prefix}.k_proj.weight"),
            format!("{prefix}.v_proj.weight"),
        ];
        let fused = Qwen3Projection::packed_quantized(
            loader,
            device,
            &[names[0].as_str(), names[1].as_str(), names[2].as_str()],
        )?;
        Ok(Self {
            fused,
            q_out,
            k_out,
            v_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.fused.forward(x)?;
        let last_dim = qkv.rank().saturating_sub(1);
        let q = qkv.narrow(last_dim, 0, self.q_out)?;
        let k = qkv.narrow(last_dim, self.q_out, self.k_out)?;
        let v = qkv.narrow(last_dim, self.q_out + self.k_out, self.v_out)?;
        Ok((q, k, v))
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        Qwen3ProjectionDiagnostics {
            dense_projection_count: 0,
            dense_bias_projection_count: 0,
            quantized_projection_count: 3,
        }
    }
}

enum Qwen3QkvProjection {
    Fused(Qwen3FusedQkvProjection),
    FusedQuantized(Qwen3FusedQuantizedQkvProjection),
    Separate {
        q_proj: Qwen3Projection,
        k_proj: Qwen3Projection,
        v_proj: Qwen3Projection,
    },
}

impl Qwen3QkvProjection {
    fn new_dense(
        q_proj: Qwen3Projection,
        k_proj: Qwen3Projection,
        v_proj: Qwen3Projection,
        device: &Device,
    ) -> Result<Self> {
        if qwen3_dense_projection_fusion_enabled(device) {
            if let (
                Qwen3Projection::Dense(q_dense),
                Qwen3Projection::Dense(k_dense),
                Qwen3Projection::Dense(v_dense),
            ) = (&q_proj, &k_proj, &v_proj)
            {
                let fused = Qwen3FusedQkvProjection::new(q_dense, k_dense, v_dense)?;
                release_fused_parts((q_proj, k_proj, v_proj), device)?;
                return Ok(Self::Fused(fused));
            }
        }
        Ok(Self::Separate {
            q_proj,
            k_proj,
            v_proj,
        })
    }

    fn new_separate(
        q_proj: Qwen3Projection,
        k_proj: Qwen3Projection,
        v_proj: Qwen3Projection,
    ) -> Self {
        Self::Separate {
            q_proj,
            k_proj,
            v_proj,
        }
    }

    fn new_quantized_grouped(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        Qwen3FusedQuantizedQkvProjection::load_gguf(loader, cfg, device, prefix)
            .map(Self::FusedQuantized)
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        match self {
            Self::Fused(fused) => fused.diagnostics(),
            Self::FusedQuantized(fused) => fused.diagnostics(),
            Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => q_proj
                .diagnostics()
                .add(k_proj.diagnostics())
                .add(v_proj.diagnostics()),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Fused(fused) => fused.forward(x),
            Self::FusedQuantized(fused) => fused.forward(x),
            Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => Ok((q_proj.forward(x)?, k_proj.forward(x)?, v_proj.forward(x)?)),
        }
    }
}

struct Qwen3FusedGateUpProjection {
    fused: Linear,
    intermediate_size: usize,
    bias_count: usize,
}

impl Qwen3FusedGateUpProjection {
    fn new(gate_proj: &Qwen3DenseProjection, up_proj: &Qwen3DenseProjection) -> Result<Self> {
        let gate_weight = gate_proj.linear.weight();
        let up_weight = up_proj.linear.weight();
        let intermediate_size = gate_weight.dim(0)?;
        if up_weight.dim(0)? != intermediate_size || up_weight.dim(1)? != gate_weight.dim(1)? {
            return Err(Error::InferenceError(
                "Qwen3 fused MLP projection dimensions do not match".to_string(),
            ));
        }
        let weight = exact_concat_rows(&[gate_weight, up_weight])?;
        let bias = concat_linear_biases(
            "Qwen3 fused MLP projection",
            &[&gate_proj.linear, &up_proj.linear],
        )?;
        let bias_count =
            usize::from(gate_proj.has_bias).saturating_add(usize::from(up_proj.has_bias));
        Ok(Self {
            fused: Linear::new(weight, bias),
            intermediate_size,
            bias_count,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let gate_up = self.fused.forward(x)?;
        let last_dim = gate_up.rank().saturating_sub(1);
        let gate = gate_up.narrow(last_dim, 0, self.intermediate_size)?;
        let up = gate_up.narrow(last_dim, self.intermediate_size, self.intermediate_size)?;
        Ok((gate, up))
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        Qwen3ProjectionDiagnostics {
            dense_projection_count: 2,
            dense_bias_projection_count: self.bias_count,
            quantized_projection_count: 0,
        }
    }
}

struct Qwen3FusedQuantizedGateUpProjection {
    fused: Qwen3Projection,
    intermediate_size: usize,
}

impl Qwen3FusedQuantizedGateUpProjection {
    fn load_gguf(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let names = [
            format!("{prefix}.gate_proj.weight"),
            format!("{prefix}.up_proj.weight"),
        ];
        let fused = Qwen3Projection::packed_quantized(
            loader,
            device,
            &[names[0].as_str(), names[1].as_str()],
        )?;
        Ok(Self {
            fused,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let gate_up = self.fused.forward(x)?;
        let last_dim = gate_up.rank().saturating_sub(1);
        let gate = gate_up.narrow(last_dim, 0, self.intermediate_size)?;
        let up = gate_up.narrow(last_dim, self.intermediate_size, self.intermediate_size)?;
        Ok((gate, up))
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        Qwen3ProjectionDiagnostics {
            dense_projection_count: 0,
            dense_bias_projection_count: 0,
            quantized_projection_count: 2,
        }
    }
}

enum Qwen3GateUpProjection {
    Fused(Qwen3FusedGateUpProjection),
    FusedQuantized(Qwen3FusedQuantizedGateUpProjection),
    Separate {
        gate_proj: Qwen3Projection,
        up_proj: Qwen3Projection,
    },
}

impl Qwen3GateUpProjection {
    fn new_dense(
        gate_proj: Qwen3Projection,
        up_proj: Qwen3Projection,
        device: &Device,
    ) -> Result<Self> {
        if qwen3_dense_projection_fusion_enabled(device) {
            if let (Qwen3Projection::Dense(gate_dense), Qwen3Projection::Dense(up_dense)) =
                (&gate_proj, &up_proj)
            {
                let fused = Qwen3FusedGateUpProjection::new(gate_dense, up_dense)?;
                release_fused_parts((gate_proj, up_proj), device)?;
                return Ok(Self::Fused(fused));
            }
        }
        Ok(Self::Separate { gate_proj, up_proj })
    }

    fn new_separate(gate_proj: Qwen3Projection, up_proj: Qwen3Projection) -> Self {
        Self::Separate { gate_proj, up_proj }
    }

    fn new_quantized_grouped(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        Qwen3FusedQuantizedGateUpProjection::load_gguf(loader, cfg, device, prefix)
            .map(Self::FusedQuantized)
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        match self {
            Self::Fused(fused) => fused.diagnostics(),
            Self::FusedQuantized(fused) => fused.diagnostics(),
            Self::Separate { gate_proj, up_proj } => {
                gate_proj.diagnostics().add(up_proj.diagnostics())
            }
        }
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fused(fused) => fused.forward(x),
            Self::FusedQuantized(fused) => fused.forward(x),
            Self::Separate { gate_proj, up_proj } => {
                Ok((gate_proj.forward(x)?, up_proj.forward(x)?))
            }
        }
    }
}

fn load_optional_rms_norm_from_gguf(
    loader: &GgufLoader,
    device: &Device,
    dtype: DType,
    name: &str,
    eps: f64,
) -> Result<Option<RmsNorm>> {
    let Some(name) = qwen3_gguf_tensor_name(loader, name) else {
        return Ok(None);
    };
    let weight = loader.load_tensor(&name, dtype, device)?;
    Ok(Some(RmsNorm::new(weight, eps)))
}

fn qwen3_gguf_tensor_name(loader: &GgufLoader, logical_name: &str) -> Option<String> {
    if loader.has_tensor(logical_name) {
        return Some(logical_name.to_string());
    }

    let canonical = qwen3_canonical_gguf_tensor_name(logical_name)?;
    loader.has_tensor(&canonical).then_some(canonical)
}

fn qwen3_canonical_gguf_tensor_name(logical_name: &str) -> Option<String> {
    match logical_name {
        "model.embed_tokens.weight" => Some("token_embd.weight".to_string()),
        "model.norm.weight" => Some("output_norm.weight".to_string()),
        "lm_head.weight" => Some("output.weight".to_string()),
        _ => {
            let layer = logical_name.strip_prefix("model.layers.")?;
            let (layer_index, suffix) = layer.split_once('.')?;
            let suffix = match suffix {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "mlp.gate.weight" => "ffn_gate_inp.weight",
                "mlp.ffn_gate_exps.weight" => "ffn_gate_exps.weight",
                "mlp.ffn_up_exps.weight" => "ffn_up_exps.weight",
                "mlp.ffn_down_exps.weight" => "ffn_down_exps.weight",
                _ => return None,
            };
            Some(format!("blk.{layer_index}.{suffix}"))
        }
    }
}

fn qk_norm_weight(q_norm: &Option<RmsNorm>, k_norm: &Option<RmsNorm>) -> Result<Option<Tensor>> {
    match (q_norm, k_norm) {
        (Some(q_norm), Some(k_norm)) if q_norm.weight().dtype() == k_norm.weight().dtype() => {
            Tensor::cat(&[q_norm.weight(), k_norm.weight()], 0)
                .map(Some)
                .map_err(Error::from)
        }
        _ => Ok(None),
    }
}

/// Whether a decode graph can capture each projection: dense weights, or
/// quantized ones the K-quant kernels serve (never `QMatMul`).
impl Qwen3Projection {
    fn graph_capturable(&self) -> bool {
        match self {
            Self::Dense(_) => true,
            Self::Quantized(projection) => projection.kquant.is_some(),
        }
    }
}

impl Qwen3QkvProjection {
    fn graph_capturable(&self) -> bool {
        match self {
            Self::Fused(_) => true,
            Self::FusedQuantized(fused) => fused.fused.graph_capturable(),
            Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => {
                q_proj.graph_capturable() && k_proj.graph_capturable() && v_proj.graph_capturable()
            }
        }
    }
}

impl Qwen3GateUpProjection {
    fn graph_capturable(&self) -> bool {
        match self {
            Self::Fused(_) => true,
            Self::FusedQuantized(fused) => fused.fused.graph_capturable(),
            Self::Separate { gate_proj, up_proj } => {
                gate_proj.graph_capturable() && up_proj.graph_capturable()
            }
        }
    }
}

/// What a decode graph's attention layers read: the arena's `(keys,
/// values)` per layer and the step buffer's slots and split-decode
/// metadata ([`decode_step`]).
struct Qwen3GraphAttention<'a> {
    caches: &'a [(Tensor, Tensor)],
    slots: Tensor,
    metadata: Tensor,
    page_tokens: usize,
    max_blocks: usize,
    max_partitions: usize,
}

struct Qwen3Attention {
    qkv_proj: Qwen3QkvProjection,
    o_proj: Qwen3Projection,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    qk_norm_weight: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    use_mrope: bool,
    mrope_section: Option<Vec<usize>>,
    rope_inv_freqs: Vec<f32>,
    rope_kernel_enabled: bool,
}

impl Qwen3Attention {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let geometry = cfg.attention_geometry()?;
        let head_dim = geometry.key_head_dim();
        let (q_out, k_out, v_out) = (
            geometry.query_width(),
            geometry.key_width(),
            geometry.value_width(),
        );
        let grouped =
            qwen3_quantized_projection_grouping_for_hidden_size(vb.device(), cfg.hidden_size)
                .then(|| {
                    qwen3_mlx_q4_1_projection(
                        &vb,
                        &[("q_proj", q_out), ("k_proj", k_out), ("v_proj", v_out)],
                        cfg.hidden_size,
                    )
                })
                .transpose()?
                .flatten();
        let qkv_proj = match grouped {
            // MLX 4-bit Q/K/V stacked as one Q4_1 projection.
            Some(fused) => Qwen3QkvProjection::FusedQuantized(Qwen3FusedQuantizedQkvProjection {
                fused,
                q_out,
                k_out,
                v_out,
            }),
            None => {
                let q_proj = Qwen3Projection::dense(cfg.hidden_size, q_out, vb.pp("q_proj"))?;
                let k_proj = Qwen3Projection::dense(cfg.hidden_size, k_out, vb.pp("k_proj"))?;
                let v_proj = Qwen3Projection::dense(cfg.hidden_size, v_out, vb.pp("v_proj"))?;
                Qwen3QkvProjection::new_dense(q_proj, k_proj, v_proj, vb.device())?
            }
        };
        let o_proj = Qwen3Projection::dense(q_out, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm")).ok();
        let qk_norm_weight = qk_norm_weight(&q_norm, &k_norm)?;
        let (use_mrope, mrope_section) = cfg
            .rope_scaling
            .as_ref()
            .map(|scaling| {
                let use_mrope = scaling.mrope_interleaved.unwrap_or(false)
                    || scaling.interleaved.unwrap_or(false);
                (use_mrope, scaling.mrope_section.clone())
            })
            .unwrap_or((false, None));

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            qk_norm_weight,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            use_mrope,
            mrope_section,
            rope_inv_freqs: build_rope_inv_freqs(head_dim, cfg.rope_theta),
            rope_kernel_enabled: qwen3_rope_kernel_enabled(vb.device()),
        })
    }

    fn load_gguf(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        dtype: DType,
        prefix: &str,
    ) -> Result<Self> {
        let head_dim = cfg.attention_geometry()?.key_head_dim();
        let qkv_proj = if qwen3_quantized_projection_grouping_enabled(device, cfg) {
            Qwen3QkvProjection::new_quantized_grouped(loader, cfg, device, prefix)?
        } else {
            let q_proj =
                Qwen3Projection::quantized(loader, device, &format!("{prefix}.q_proj.weight"))?;
            let k_proj =
                Qwen3Projection::quantized(loader, device, &format!("{prefix}.k_proj.weight"))?;
            let v_proj =
                Qwen3Projection::quantized(loader, device, &format!("{prefix}.v_proj.weight"))?;
            Qwen3QkvProjection::new_separate(q_proj, k_proj, v_proj)
        };
        let o_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.o_proj.weight"))?;

        let q_norm = load_optional_rms_norm_from_gguf(
            loader,
            device,
            dtype,
            &format!("{prefix}.q_norm.weight"),
            cfg.rms_norm_eps,
        )?;
        let k_norm = load_optional_rms_norm_from_gguf(
            loader,
            device,
            dtype,
            &format!("{prefix}.k_norm.weight"),
            cfg.rms_norm_eps,
        )?;
        let qk_norm_weight = qk_norm_weight(&q_norm, &k_norm)?;
        let (use_mrope, mrope_section) = cfg
            .rope_scaling
            .as_ref()
            .map(|scaling| {
                let use_mrope = scaling.mrope_interleaved.unwrap_or(false)
                    || scaling.interleaved.unwrap_or(false);
                (use_mrope, scaling.mrope_section.clone())
            })
            .unwrap_or((false, None));

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            qk_norm_weight,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            use_mrope,
            mrope_section,
            rope_inv_freqs: build_rope_inv_freqs(head_dim, cfg.rope_theta),
            rope_kernel_enabled: qwen3_rope_kernel_enabled(device),
        })
    }

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.qkv_proj.diagnostics().add(self.o_proj.diagnostics())
    }

    fn apply_qk_norm(
        &self,
        x: Tensor,
        norm: &Option<RmsNorm>,
        heads: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        if let Some(norm) = norm {
            let bsz = x.dim(0)?;
            let reshaped = x.reshape((bsz * seq_len * heads, self.head_dim))?;
            let normed = norm.forward(&reshaped)?;
            normed
                .reshape((bsz, seq_len, heads, self.head_dim))
                .map_err(Error::from)
        } else {
            Ok(x)
        }
    }

    fn apply_qk_norm_pair(&self, q: Tensor, k: Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if let (Some(q_norm), Some(_k_norm), Some(qk_weight)) =
            (&self.q_norm, &self.k_norm, &self.qk_norm_weight)
        {
            if seq_len == 1 && qwen3_qk_rms_norm_fusion_enabled(q.device()) {
                if let Some((q, k)) = try_fused_qk_rms_norm(&q, &k, qk_weight, q_norm.eps()) {
                    return Ok((q, k));
                }
            }
        }

        let q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, seq_len)?;
        let k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, seq_len)?;
        Ok((q, k))
    }

    /// Whether sequential MRoPE positions take the host 1-D table path. With
    /// fewer than three sections MRoPE falls back to positions from zero, which
    /// stays on the original path.
    fn sequential_mrope_uses_host_tables(&self) -> bool {
        self.use_mrope && self.mrope_section.as_ref().is_some_and(|s| s.len() >= 3)
    }

    fn apply_rope_pair(
        &self,
        q: Tensor,
        k: Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        self.apply_rope_pair_cached(q, k, start_pos, position_ids, &mut None)
    }

    /// [`Self::apply_rope_pair`] with the cos/sin tables kept in `tables`
    /// across calls: a stateless forward builds them once for every layer
    /// (the layers share one RoPE geometry) instead of once per layer
    /// (QFA-9). The tables stay host-built and uploaded.
    fn apply_rope_pair_cached(
        &self,
        q: Tensor,
        k: Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        tables: &mut Option<(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        if k.dim(1)? != seq_len {
            return Err(Error::InferenceError(format!(
                "Qwen3 RoPE sequence mismatch: q_seq={}, k_seq={}",
                seq_len,
                k.dim(1)?
            )));
        }
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let reusable = tables
            .as_ref()
            .is_some_and(|(cos, _)| cos.dtype() == q.dtype() && cos.dim(0).ok() == Some(seq_len));
        if !reusable {
            *tables = Some(self.rope_tables(seq_len, start_pos, position_ids, &q)?);
        }
        let Some((cos_half, sin_half)) = tables.as_ref() else {
            return Err(Error::InferenceError("Qwen3 RoPE tables missing".into()));
        };
        self.apply_rope_pair_with_tables(&q, &k, cos_half, sin_half)
    }

    /// Rotates contiguous `[batch, seq, heads, head_dim]` `q` and `k` by the
    /// `[seq, head_dim / 2]` `cos_half`/`sin_half` rows: the fused kernel
    /// where enabled, else the composed ops. Each output element reads only
    /// its own row's cos/sin, so the rows may come from any positions.
    fn apply_rope_pair_with_tables(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos_half: &Tensor,
        sin_half: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.should_try_rope_kernel(q.dtype()) {
            let cos = cos_half.unsqueeze(0)?.contiguous()?;
            let sin = sin_half.unsqueeze(0)?.contiguous()?;
            if let Some((q_out, k_out)) = try_apply_rope_pair_thd(q, k, &cos, &sin)? {
                record_rope_kernel();
                record_rope_kernel();
                return Ok((q_out, k_out));
            }
        }
        record_rope_manual();
        record_rope_manual();

        Ok((
            apply_rotary_emb(q, cos_half, sin_half)?,
            apply_rotary_emb(k, cos_half, sin_half)?,
        ))
    }

    /// RoPE over a whole decode batch in one call per tensor: `[rows, 1,
    /// heads, head_dim]` `q`/`k` rotate as one `[1, rows, heads, head_dim]`
    /// sequence whose token `i` takes `rope`'s row `i`. Each element sees
    /// exactly the cos/sin and ops a one-row call at that row's position
    /// does. Returns the `[rows, heads, head_dim]` queries and keys.
    fn apply_decode_rope(
        &self,
        q: Tensor,
        k: Tensor,
        rope: &Qwen3DecodeRopeRows,
    ) -> Result<(Tensor, Tensor)> {
        let rows = q.dim(0)?;
        if rope.cos.dtype() != q.dtype() {
            return Err(Error::InferenceError(format!(
                "Qwen3 decode RoPE rows are {:?} but the queries are {:?}",
                rope.cos.dtype(),
                q.dtype()
            )));
        }
        let q = q
            .reshape((1, rows, self.num_heads, self.head_dim))?
            .contiguous()?;
        let k = k
            .reshape((1, rows, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let (q, k) = self.apply_rope_pair_with_tables(&q, &k, &rope.cos, &rope.sin)?;
        Ok((
            q.reshape((rows, self.num_heads, self.head_dim))?
                .contiguous()?,
            k.reshape((rows, self.num_kv_heads, self.head_dim))?
                .contiguous()?,
        ))
    }

    /// The position whose angles a one-token decode row at `start_pos`
    /// rotates by in [`Self::rope_tables`]: MRoPE with fewer than three
    /// sections builds its table from position zero.
    fn decode_rope_position(&self, start_pos: usize) -> usize {
        if self.use_mrope && !self.sequential_mrope_uses_host_tables() {
            0
        } else {
            start_pos
        }
    }

    /// cos/sin rows for positions `0..rows`, built by the builder
    /// [`Self::rope_tables`] runs for one decode token. Every element is
    /// computed on its own (`pos as f32 * inv`, then cos/sin elementwise),
    /// so row `p` is bit-identical to the table that path builds at `p`.
    fn decode_rope_table(
        &self,
        rows: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        if self.sequential_mrope_uses_host_tables() {
            build_sequential_rope_host_tables(rows, 0, device, dtype, &self.rope_inv_freqs)
        } else {
            build_rope_cache_with_inv_freqs(rows, 0, device, dtype, &self.rope_inv_freqs)
        }
    }

    /// Each decode row's `[rows, head_dim / 2]` cos/sin: one upload of the
    /// rows' positions and one gather per table from the model's decode
    /// table, which grows (rebuilt from zero by the same builder) when a
    /// position passes its end.
    fn decode_rope_rows(
        &self,
        table: &Mutex<Option<Qwen3DecodeRopeTable>>,
        start_positions: &[usize],
        device: &Device,
        dtype: DType,
    ) -> Result<Qwen3DecodeRopeRows> {
        let positions = start_positions
            .iter()
            .map(|&pos| {
                u32::try_from(self.decode_rope_position(pos))
                    .map_err(|_| Error::InvalidInput("Qwen3 RoPE position exceeds u32".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let needed = positions.iter().max().map_or(1, |&pos| pos as usize + 1);
        let (cos, sin, _) = self.decode_rope_table_rows(table, needed, device, dtype)?;
        let rows = positions.len();
        let positions = Tensor::from_vec(positions, rows, device)?;
        Ok(Qwen3DecodeRopeRows {
            cos: cos.index_select(&positions, 0)?,
            sin: sin.index_select(&positions, 0)?,
        })
    }

    /// The model's decode `(cos, sin)` tables, grown (rebuilt from zero by
    /// the same builder) to hold at least `needed` positions, and their row
    /// count.
    fn decode_rope_table_rows(
        &self,
        table: &Mutex<Option<Qwen3DecodeRopeTable>>,
        needed: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor, usize)> {
        let mut table = table
            .lock()
            .map_err(|_| Error::InferenceError("Qwen3 decode RoPE table lock poisoned".into()))?;
        let stale = !table
            .as_ref()
            .is_some_and(|table| table.dtype == dtype && table.rows >= needed);
        if stale {
            let rows = needed
                .next_power_of_two()
                .max(QWEN3_DECODE_ROPE_TABLE_MIN_ROWS);
            let (cos, sin) = self.decode_rope_table(rows, device, dtype)?;
            *table = Some(Qwen3DecodeRopeTable {
                dtype,
                rows,
                cos,
                sin,
            });
        }
        let Some(table) = table.as_ref() else {
            return Err(Error::InferenceError(
                "Qwen3 decode RoPE table missing".into(),
            ));
        };
        Ok((table.cos.clone(), table.sin.clone(), table.rows))
    }

    /// The `[seq, head_dim / 2]` cos/sin tables for `q`'s positions, in its
    /// dtype and on its device.
    fn rope_tables(
        &self,
        seq_len: usize,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        q: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(test)]
        QWEN3_ROPE_TABLE_BUILDS_THREAD.with(|count| count.set(count.get() + 1));
        if let Some(position_ids) = position_ids {
            if self.use_mrope {
                build_mrope_cache_with_inv_freqs(
                    seq_len,
                    self.head_dim,
                    q.device(),
                    q.dtype(),
                    position_ids,
                    self.mrope_section.as_deref().unwrap_or(&[]),
                    &self.rope_inv_freqs,
                )
            } else {
                build_rope_cache_with_inv_freqs(
                    seq_len,
                    start_pos,
                    q.device(),
                    q.dtype(),
                    &self.rope_inv_freqs,
                )
            }
        } else if self.sequential_mrope_uses_host_tables() {
            // Equal sequential positions on all three axes are plain 1-D RoPE:
            // build the same host tables directly instead of uploading the
            // positions and reading them back (a sync per layer per row).
            build_sequential_rope_host_tables(
                seq_len,
                start_pos,
                q.device(),
                q.dtype(),
                &self.rope_inv_freqs,
            )
        } else if self.use_mrope {
            let mut data = Vec::with_capacity(3 * seq_len);
            let base = start_pos as i64;
            for _axis in 0..3 {
                for idx in 0..seq_len {
                    data.push(base + idx as i64);
                }
            }
            let position_ids = Tensor::from_vec(data, (3, seq_len), q.device())?;
            build_mrope_cache_with_inv_freqs(
                seq_len,
                self.head_dim,
                q.device(),
                q.dtype(),
                &position_ids,
                self.mrope_section.as_deref().unwrap_or(&[]),
                &self.rope_inv_freqs,
            )
        } else {
            build_rope_cache_with_inv_freqs(
                seq_len,
                start_pos,
                q.device(),
                q.dtype(),
                &self.rope_inv_freqs,
            )
        }
    }

    fn should_try_rope_kernel(&self, dtype: DType) -> bool {
        if !self.rope_kernel_enabled {
            return false;
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return false;
        }
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
    }

    fn forward_stateless(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        rope_tables: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let sequence = x.dim(1)?;
        let (query, key, value) = self.qkv_proj.forward(x)?;
        let query = query.reshape((batch, sequence, self.num_heads, self.head_dim))?;
        let key = key.reshape((batch, sequence, self.num_kv_heads, self.head_dim))?;
        let value = value.reshape((batch, sequence, self.num_kv_heads, self.head_dim))?;
        let (query, key) = self.apply_qk_norm_pair(query, key, sequence)?;
        let (query, key) =
            self.apply_rope_pair_cached(query, key, start_pos, position_ids, rope_tables)?;

        let query = query.transpose(1, 2)?;
        let key_heads = key.transpose(1, 2)?;
        let value_heads = value.transpose(1, 2)?;
        if start_pos == 0 {
            if let Some(output) = try_fused_self_attention(
                &query,
                &key_heads,
                &value_heads,
                None,
                self.head_dim,
                true,
            )? {
                return self
                    .o_proj
                    .forward(&output.transpose(1, 2)?.reshape((
                        batch,
                        sequence,
                        self.num_heads * self.head_dim,
                    ))?)
                    .map_err(Error::from);
            }
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let output = if start_pos == 0 {
            causal_sdpa_by_query_blocks(
                &query,
                &key_heads,
                &value_heads,
                scale,
                QWEN3_STATELESS_QUERY_BLOCK,
            )?
        } else {
            let mask = (sequence > 1)
                .then(|| causal_mask(sequence, sequence, start_pos, query.device(), query.dtype()))
                .transpose()?;
            compact_gqa_sdpa_bhsd(
                &query,
                &key_heads,
                &value_heads,
                mask.as_ref().map(CompactGqaMask::Additive),
                scale,
            )?
        }
        .transpose(1, 2)?
        .reshape((batch, sequence, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&output).map_err(Error::from)
    }

    fn forward_managed(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        if bsz != 1 {
            return Err(Error::InvalidInput(
                "Qwen3 managed prefill expects one sequence per call".to_string(),
            ));
        }
        let (q, k, v) = self.qkv_proj.forward(x)?;
        let q = q.reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;
        let (q, k) = self.apply_qk_norm_pair(q, k, seq_len)?;
        let (q, k) = self.apply_rope_pair(q, k, start_pos, position_ids)?;

        let queries = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .contiguous()?;
        let keys = k
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let values = v
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let out = cache.write_and_attend(
            layer_idx,
            prepared,
            &queries,
            &keys,
            &values,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }

    fn forward_managed_decode_batch(
        &self,
        x: &Tensor,
        rope: &Qwen3DecodeRopeRows,
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        if x.dim(1)? != 1 || bsz == 0 || rope.cos.dim(0)? != bsz || caches.len() != bsz {
            return Err(Error::InvalidInput(
                "Qwen3 managed decode batch dimensions do not match".to_string(),
            ));
        }
        let first = caches[0];
        if caches.iter().any(|cache| {
            !Arc::ptr_eq(&cache.arena, &first.arena)
                || cache.layer_binding(layer_idx).ok() != first.layer_binding(layer_idx).ok()
        }) {
            return Err(Error::InvalidInput(
                "Qwen3 managed decode rows must share one arena and layer binding".to_string(),
            ));
        }

        let (q, k, v) = self.qkv_proj.forward(x)?;
        let q = q.reshape((bsz, 1, self.num_heads, self.head_dim))?;
        let k = k.reshape((bsz, 1, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((bsz, 1, self.num_kv_heads, self.head_dim))?;
        let (q, k) = self.apply_qk_norm_pair(q, k, 1)?;
        let (queries, keys) = self.apply_decode_rope(q, k, rope)?;
        let values = v
            .reshape((bsz, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        if slots.arena_id() != first.arena.id() || slots.len() != bsz {
            return Err(Error::InvalidInput(
                "Qwen3 managed decode received an incompatible prepared slot map".into(),
            ));
        }
        let binding = first.layer_binding(layer_idx)?;
        let completion = first.arena.write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let (out, completion) = submit_ordered_after_write(completion, || {
            first.arena.paged_decode(
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
        let out = out.reshape((bsz, 1, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }

    /// [`Self::forward_managed_decode_batch`] as a decode graph runs it:
    /// each row's K/V rows written at its step-buffer slot, and the split
    /// decode over its step-buffer block table with row-invariant
    /// partitions. No host value is read or uploaded.
    fn forward_graph_decode(
        &self,
        x: &Tensor,
        rope: &Qwen3DecodeRopeRows,
        attention: &Qwen3GraphAttention<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let (q, k, v) = self.qkv_proj.forward(x)?;
        let q = q.reshape((bsz, 1, self.num_heads, self.head_dim))?;
        let k = k.reshape((bsz, 1, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((bsz, 1, self.num_kv_heads, self.head_dim))?;
        let (q, k) = self.apply_qk_norm_pair(q, k, 1)?;
        let (queries, keys) = self.apply_decode_rope(q, k, rope)?;
        let values = v
            .reshape((bsz, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let (cache_keys, cache_values) = attention.caches.get(layer_idx).ok_or_else(|| {
            Error::InferenceError(format!(
                "Qwen3 decode graph has no KV cache for layer {layer_idx}"
            ))
        })?;
        decode_step::write_kv_rows(cache_keys, cache_values, &keys, &values, &attention.slots)?;
        let out = split_decode::paged_decode_adaptive(
            &queries,
            cache_keys,
            cache_values,
            &attention.metadata,
            bsz,
            self.num_heads,
            self.num_kv_heads,
            attention.page_tokens,
            attention.max_blocks,
            1.0 / (self.head_dim as f32).sqrt(),
            None,
            attention.max_partitions,
        )?;
        let out = out.reshape((bsz, 1, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }

    fn graph_capturable(&self) -> bool {
        self.qkv_proj.graph_capturable() && self.o_proj.graph_capturable()
    }

    /// One mixed step: `x` is `[1, tokens, hidden]` in `step`'s row layout.
    /// Projections run on every token at once; QK-norm and RoPE take the
    /// shapes the split paths give them (one-token decode rows, then each
    /// prefill span from its own position).
    fn forward_managed_mixed(
        &self,
        x: &Tensor,
        caches: &[&PhysicalPagedKvCache],
        step: &mut PreparedPhysicalMixedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let tokens = x.dim(1)?;
        let layout = step.layout();
        let decode = layout.decode_rows();
        let (q, k, v) = self.qkv_proj.forward(x)?;
        let q = q.reshape((tokens, self.num_heads, self.head_dim))?;
        let k = k.reshape((tokens, self.num_kv_heads, self.head_dim))?;
        let mut queries = Vec::with_capacity(layout.rows().len());
        let mut keys = Vec::with_capacity(layout.rows().len());
        if decode > 0 {
            let (q, k) = self.apply_qk_norm_pair(
                q.narrow(0, 0, decode)?.unsqueeze(1)?,
                k.narrow(0, 0, decode)?.unsqueeze(1)?,
                1,
            )?;
            for (row, span) in layout.rows()[..decode].iter().enumerate() {
                let (q, k) = self.apply_rope_pair(
                    q.i(row)?.unsqueeze(0)?,
                    k.i(row)?.unsqueeze(0)?,
                    span.start_pos,
                    None,
                )?;
                queries.push(q.reshape((1, self.num_heads, self.head_dim))?);
                keys.push(k.reshape((1, self.num_kv_heads, self.head_dim))?);
            }
        }
        if decode < tokens {
            let prefill = tokens - decode;
            let (q, k) = self.apply_qk_norm_pair(
                q.narrow(0, decode, prefill)?.unsqueeze(0)?,
                k.narrow(0, decode, prefill)?.unsqueeze(0)?,
                prefill,
            )?;
            for (row, span) in layout.rows().iter().enumerate().skip(decode) {
                let offset = layout.token_offset(row) - decode;
                let (q, k) = self.apply_rope_pair(
                    q.narrow(1, offset, span.q_len)?,
                    k.narrow(1, offset, span.q_len)?,
                    span.start_pos,
                    None,
                )?;
                queries.push(q.reshape((span.q_len, self.num_heads, self.head_dim))?);
                keys.push(k.reshape((span.q_len, self.num_kv_heads, self.head_dim))?);
            }
        }
        let queries = Tensor::cat(&queries, 0)?.contiguous()?;
        let keys = Tensor::cat(&keys, 0)?.contiguous()?;
        let values = v
            .reshape((tokens, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let out = step.write_and_attend(
            caches,
            layer_idx,
            &queries,
            &keys,
            &values,
            1.0 / (self.head_dim as f32).sqrt(),
            None,
            None,
        )?;
        let out = out.reshape((1, tokens, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

struct Qwen3Mlp {
    gate_up_proj: Qwen3GateUpProjection,
    down_proj: Qwen3Projection,
}

impl Qwen3Mlp {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let grouped =
            qwen3_quantized_projection_grouping_for_hidden_size(vb.device(), cfg.hidden_size)
                .then(|| {
                    qwen3_mlx_q4_1_projection(
                        &vb,
                        &[
                            ("gate_proj", cfg.intermediate_size),
                            ("up_proj", cfg.intermediate_size),
                        ],
                        cfg.hidden_size,
                    )
                })
                .transpose()?
                .flatten();
        let gate_up_proj = match grouped {
            // MLX 4-bit gate/up stacked as one Q4_1 projection.
            Some(fused) => {
                Qwen3GateUpProjection::FusedQuantized(Qwen3FusedQuantizedGateUpProjection {
                    fused,
                    intermediate_size: cfg.intermediate_size,
                })
            }
            None => {
                let gate_proj = Qwen3Projection::dense(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    vb.pp("gate_proj"),
                )?;
                let up_proj = Qwen3Projection::dense(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    vb.pp("up_proj"),
                )?;
                Qwen3GateUpProjection::new_dense(gate_proj, up_proj, vb.device())?
            }
        };
        let down_proj =
            Qwen3Projection::dense(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn load_gguf(
        cfg: &Qwen3Config,
        loader: &GgufLoader,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let gate_up_proj = if qwen3_quantized_projection_grouping_enabled(device, cfg) {
            Qwen3GateUpProjection::new_quantized_grouped(loader, cfg, device, prefix)?
        } else {
            let gate_proj =
                Qwen3Projection::quantized(loader, device, &format!("{prefix}.gate_proj.weight"))?;
            let up_proj =
                Qwen3Projection::quantized(loader, device, &format!("{prefix}.up_proj.weight"))?;
            Qwen3GateUpProjection::new_separate(gate_proj, up_proj)
        };
        let down_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.down_proj.weight"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.gate_up_proj
            .diagnostics()
            .add(self.down_proj.diagnostics())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (gate, up) = self.gate_up_proj.forward(x)?;
        let hidden = if let Some(fused) = try_fused_silu_mul_with_status(&gate, &up) {
            record_qwen3_silu_mul_profile(fused.used_custom_kernel);
            fused.tensor
        } else {
            record_qwen3_silu_mul_profile(false);
            let act = ops::silu(&gate)?;
            act.broadcast_mul(&up)?
        };
        let out = self.down_proj.forward(&hidden)?;
        Ok(out)
    }
}

/// One routed expert of a sparse Qwen3 feed-forward block. Projection
/// machinery and dtype policies match the dense MLP exactly, so an expert is
/// a dense FFN that the router addresses by index.
struct Qwen3SparseExpert {
    gate_up_proj: Qwen3GateUpProjection,
    down_proj: Qwen3Projection,
}

impl ExpertSet for Vec<Qwen3SparseExpert> {
    fn num_experts(&self) -> usize {
        self.len()
    }

    fn apply_expert(&self, expert: usize, tokens: &Tensor) -> Result<Tensor> {
        let expert = self.get(expert).ok_or_else(|| {
            Error::InferenceError(format!("Qwen3 sparse expert {expert} is out of range"))
        })?;
        let (gate, up) = expert.gate_up_proj.forward(tokens)?;
        // Per-expert silu-mul is intentionally not profile-recorded: the
        // fused-kernel telemetry would multiply by expert count per step.
        let hidden = match try_fused_silu_mul_with_status(&gate, &up) {
            Some(fused) => fused.tensor,
            None => ops::silu(&gate)?.broadcast_mul(&up)?,
        };
        expert.down_proj.forward(&hidden).map_err(Error::from)
    }
}

/// Sparse Qwen3 feed-forward: router over routed experts, combined by the
/// shared sparse dispatch seam (`models::shared::moe`). Expert parallelism
/// would replace the `Vec<Qwen3SparseExpert>` ExpertSet implementation, not
/// this block.
struct Qwen3SparseMlp {
    dispatcher: SparseMoeDispatcher,
    router: Qwen3Projection,
    experts: Vec<Qwen3SparseExpert>,
    /// Per-step routing histogram (EPLB-style balancer input; DS10 A6).
    activation_counters: Arc<ExpertActivationCounters>,
}

impl Qwen3SparseMlp {
    fn counters(&self) -> Arc<ExpertActivationCounters> {
        self.activation_counters.clone()
    }

    fn load(cfg: &Qwen3Config, moe: SparseMoeConfig, vb: VarBuilder) -> Result<Self> {
        let moe_intermediate_size = cfg
            .moe_intermediate_size()
            .ok_or_else(|| Error::ModelLoadError("sparse MoE intermediate size missing".into()))?;
        let counters = Arc::new(ExpertActivationCounters::new(moe.num_experts));
        let router = Qwen3Projection::dense(cfg.hidden_size, moe.num_experts, vb.pp("gate"))?;
        let vb_experts = vb.pp("experts");
        let experts = (0..moe.num_experts)
            .map(|idx| {
                let expert_vb = vb_experts.pp(idx);
                let gate_proj = Qwen3Projection::dense(
                    cfg.hidden_size,
                    moe_intermediate_size,
                    expert_vb.pp("gate_proj"),
                )?;
                let up_proj = Qwen3Projection::dense(
                    cfg.hidden_size,
                    moe_intermediate_size,
                    expert_vb.pp("up_proj"),
                )?;
                let down_proj = Qwen3Projection::dense(
                    moe_intermediate_size,
                    cfg.hidden_size,
                    expert_vb.pp("down_proj"),
                )?;
                Ok(Qwen3SparseExpert {
                    gate_up_proj: Qwen3GateUpProjection::new_dense(
                        gate_proj,
                        up_proj,
                        vb.device(),
                    )?,
                    down_proj,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dispatcher: SparseMoeDispatcher::new(moe)?
                .with_counters(counters.clone()),
            router,
            experts,
            activation_counters: counters,
        })
    }

    fn load_gguf(
        cfg: &Qwen3Config,
        moe: SparseMoeConfig,
        loader: &GgufLoader,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let moe_intermediate_size = cfg
            .moe_intermediate_size()
            .ok_or_else(|| Error::ModelLoadError("sparse MoE intermediate size missing".into()))?;
        let counters = Arc::new(ExpertActivationCounters::new(moe.num_experts));
        let router = Qwen3Projection::quantized(loader, device, &format!("{prefix}.gate.weight"))?;
        let gate_experts = fused_expert_qtensors(
            loader,
            device,
            &format!("{prefix}.ffn_gate_exps.weight"),
            moe.num_experts,
            moe_intermediate_size,
            cfg.hidden_size,
        )?;
        let up_experts = fused_expert_qtensors(
            loader,
            device,
            &format!("{prefix}.ffn_up_exps.weight"),
            moe.num_experts,
            moe_intermediate_size,
            cfg.hidden_size,
        )?;
        let down_experts = fused_expert_qtensors(
            loader,
            device,
            &format!("{prefix}.ffn_down_exps.weight"),
            moe.num_experts,
            cfg.hidden_size,
            moe_intermediate_size,
        )?;
        let experts = gate_experts
            .into_iter()
            .zip(up_experts)
            .zip(down_experts)
            .map(|((gate, up), down)| {
                Ok(Qwen3SparseExpert {
                    gate_up_proj: Qwen3GateUpProjection::new_separate(
                        Qwen3Projection::from_qtensor(gate)?,
                        Qwen3Projection::from_qtensor(up)?,
                    ),
                    down_proj: Qwen3Projection::from_qtensor(down)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dispatcher: SparseMoeDispatcher::new(moe)?
                .with_counters(counters.clone()),
            router,
            experts,
            activation_counters: counters,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // The dispatch seam operates on [tokens, hidden]; attention may hand
        // the FFN [rows, hidden] directly or a [batch, seq, hidden] prefill
        // block, so flatten and restore around the router.
        let original_shape = x.shape().clone();
        let (tokens, hidden_dim) = match x.rank() {
            2 => x.dims2()?,
            3 => {
                let (batch, seq, hidden) = x.dims3()?;
                (batch * seq, hidden)
            }
            rank => {
                return Err(Error::InferenceError(format!(
                    "Qwen3 sparse FFN expects rank-2 or rank-3 input, got rank {rank}"
                )))
            }
        };
        let flat = x.reshape((tokens, hidden_dim))?;
        let router_logits = self.router.forward(&flat)?;
        let output = self
            .dispatcher
            .dispatch(&flat, &router_logits, &self.experts)?;
        output.reshape(&original_shape).map_err(Error::from)
    }

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        let expert_diagnostics = self.experts.iter().fold(
            Qwen3ProjectionDiagnostics::default(),
            |acc, expert| {
                acc.add(expert.gate_up_proj.diagnostics())
                    .add(expert.down_proj.diagnostics())
            },
        );
        self.router.diagnostics().add(expert_diagnostics)
    }
}

fn fused_expert_qtensors(
    loader: &GgufLoader,
    device: &Device,
    logical_name: &str,
    num_experts: usize,
    expert_rows: usize,
    expert_cols: usize,
) -> Result<Vec<QTensor>> {
    let resolved = qwen3_gguf_tensor_name(loader, logical_name).unwrap_or_else(|| logical_name.to_string());
    let fused = loader.load_qtensor(&resolved, device)?;
    crate::models::shared::weights::gguf::split_fused_expert_qtensor(
        &fused,
        num_experts,
        expert_rows,
        expert_cols,
    )
}

enum Qwen3FeedForward {
    Dense(Qwen3Mlp),
    Sparse(Qwen3SparseMlp),
}

impl Qwen3FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(mlp) => mlp.forward(x),
        }
    }

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        match self {
            Self::Dense(mlp) => mlp.projection_diagnostics(),
            Self::Sparse(mlp) => mlp.projection_diagnostics(),
        }
    }
}

struct Qwen3Layer {
    input_layernorm: RmsNorm,
    self_attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Qwen3FeedForward,
}

impl Qwen3Layer {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Qwen3Attention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = match cfg.sparse_moe()? {
            Some(moe) => Qwen3FeedForward::Sparse(Qwen3SparseMlp::load(cfg, moe, vb.pp("mlp"))?),
            None => Qwen3FeedForward::Dense(Qwen3Mlp::load(cfg, vb.pp("mlp"))?),
        };
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn load_gguf(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        dtype: DType,
        prefix: &str,
    ) -> Result<Self> {
        let input_layernorm = RmsNorm::new(
            loader.load_tensor(
                &qwen3_gguf_tensor_name(loader, &format!("{prefix}.input_layernorm.weight"))
                    .unwrap_or_else(|| format!("{prefix}.input_layernorm.weight")),
                dtype,
                device,
            )?,
            cfg.rms_norm_eps,
        );
        let self_attn =
            Qwen3Attention::load_gguf(loader, cfg, device, dtype, &format!("{prefix}.self_attn"))?;
        let post_attention_layernorm = RmsNorm::new(
            loader.load_tensor(
                &qwen3_gguf_tensor_name(
                    loader,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )
                .unwrap_or_else(|| format!("{prefix}.post_attention_layernorm.weight")),
                dtype,
                device,
            )?,
            cfg.rms_norm_eps,
        );
        let mlp = match cfg.sparse_moe()? {
            Some(moe) => {
                Qwen3FeedForward::Sparse(Qwen3SparseMlp::load_gguf(cfg, moe, loader, device, &format!("{prefix}.mlp"))?)
            }
            None => Qwen3FeedForward::Dense(Qwen3Mlp::load_gguf(cfg, loader, device, &format!("{prefix}.mlp"))?),
        };
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.self_attn
            .projection_diagnostics()
            .add(self.mlp.projection_diagnostics())
    }

    fn forward_stateless(
        &self,
        input: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        rope_tables: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let attention = self.self_attn.forward_stateless(
            &self.input_layernorm.forward(input)?,
            start_pos,
            position_ids,
            rope_tables,
        )?;
        let hidden = input.broadcast_add(&attention)?;
        let mlp = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden)?)?;
        hidden.broadcast_add(&mlp).map_err(Error::from)
    }

    fn forward_managed(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_managed(
            &normed,
            start_pos,
            position_ids,
            cache,
            prepared,
            layer_idx,
        )?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }

    fn forward_managed_decode_batch(
        &self,
        x: &Tensor,
        rope: &Qwen3DecodeRopeRows,
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_managed_decode_batch(
            &normed,
            rope,
            caches,
            slots,
            metadata,
            completions,
            layer_idx,
        )?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }

    /// [`Self::forward_managed_decode_batch`] as a decode graph runs it.
    fn forward_graph_decode(
        &self,
        x: &Tensor,
        rope: &Qwen3DecodeRopeRows,
        attention: &Qwen3GraphAttention<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_graph_decode(&normed, rope, attention, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }

    fn graph_capturable(&self) -> bool {
        self.self_attn.graph_capturable()
            && match &self.mlp {
                Qwen3FeedForward::Dense(mlp) => {
                    mlp.gate_up_proj.graph_capturable() && mlp.down_proj.graph_capturable()
                }
                // Expert routing reads the host.
                Qwen3FeedForward::Sparse(_) => false,
            }
    }

    fn forward_managed_mixed(
        &self,
        x: &Tensor,
        caches: &[&PhysicalPagedKvCache],
        step: &mut PreparedPhysicalMixedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_managed_mixed(&normed, caches, step, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

pub struct Qwen3Model {
    embed_tokens: Qwen3TokenEmbedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    lm_head: Qwen3Projection,
    device: Device,
    cfg: Qwen3Config,
    use_mrope: bool,
    /// Decode cos/sin by position, built at the first batched decode step.
    decode_rope_table: Mutex<Option<Qwen3DecodeRopeTable>>,
    /// The ASR families' whole-step CUDA decode graphs, or why there are
    /// none ([`Self::enable_asr_decode_graph`]).
    decode_graph: std::result::Result<Qwen3DecodeGraph, String>,
}

/// Rows a decode RoPE table starts with. A position past its end rebuilds it
/// at the next power of two (same builder from zero, so the old rows are
/// unchanged).
const QWEN3_DECODE_ROPE_TABLE_MIN_ROWS: usize = 4096;

/// A model's decode cos/sin, `[rows, head_dim / 2]` each, for positions
/// `0..rows` in one dtype.
struct Qwen3DecodeRopeTable {
    dtype: DType,
    rows: usize,
    cos: Tensor,
    sin: Tensor,
}

/// One decode step's `[rows, head_dim / 2]` cos/sin, row `i` for decode row
/// `i`'s position; every layer reads the same rows.
struct Qwen3DecodeRopeRows {
    cos: Tensor,
    sin: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3WeightLayout {
    pub model_prefix: &'static str,
    pub lm_head_prefix: Option<&'static str>,
}

impl Qwen3WeightLayout {
    pub const STANDARD: Self = Self {
        model_prefix: "model",
        lm_head_prefix: Some("lm_head"),
    };

    pub const VIBEVOICE: Self = Self {
        model_prefix: "model.language_model",
        lm_head_prefix: Some("lm_head"),
    };
}

impl Default for Qwen3WeightLayout {
    fn default() -> Self {
        Self::STANDARD
    }
}

impl Qwen3Model {
    pub fn context_length(&self) -> Option<usize> {
        self.cfg.context_length()
    }

    pub(crate) fn supports_managed_kv_execution(&self) -> bool {
        self.cfg.sliding_window().is_none()
    }

    pub(crate) fn managed_inference_state_contract(
        &self,
        domain: StateDomainId,
        storage_dtype: DType,
        preferred_page_tokens: usize,
        prefix_reuse: bool,
    ) -> Result<InferenceStateContract> {
        let attention = self.cfg.attention_geometry()?;
        let cache_domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
            domain,
            clock: StateClock::DecoderTokens,
            num_layers: self.cfg.num_hidden_layers,
            num_query_heads: self.cfg.num_attention_heads,
            num_kv_heads: self.cfg.num_key_value_heads,
            key_head_dim: attention.key_head_dim(),
            value_head_dim: attention.value_head_dim(),
            sliding_window: self.cfg.sliding_window(),
            storage_dtype,
            preferred_page_tokens,
            prefix: if prefix_reuse {
                PrefixPolicy::CommittedPages {
                    positions: crate::kv::v2::PositionSemantics::Absolute,
                }
            } else {
                PrefixPolicy::Disabled
            },
        })?;
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::PagedAttention(cache_domain)],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(domain.get()),
                domains: vec![domain],
                prefix_shareable: prefix_reuse,
            }],
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn load(cfg: Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Self::load_with_layout(cfg, vb, Qwen3WeightLayout::STANDARD)
    }

    pub fn load_with_layout(
        cfg: Qwen3Config,
        vb: VarBuilder,
        layout: Qwen3WeightLayout,
    ) -> Result<Self> {
        cfg.attention_geometry()?;
        let lm_head_size = cfg.lm_head_size.unwrap_or(cfg.vocab_size);
        let embed_vb = vb.pp(qwen3_join_prefix(layout.model_prefix, "embed_tokens"));
        // An MLX 4-bit table stays quantized where its projections do: rows
        // are gathered from the Q4_1 blocks and cast to the model dtype,
        // exactly the dense table's rows.
        let embed_tokens =
            match qwen3_mlx_q4_1_projection(&embed_vb, &[("", cfg.vocab_size)], cfg.hidden_size)? {
                Some(Qwen3Projection::Quantized(Qwen3QuantizedProjection {
                    qmatmul: QMatMul::QTensor(table),
                    ..
                })) => Qwen3TokenEmbedding::Quantized {
                    table: QuantizedTokenTable::from_arc(table)?,
                    dtype: vb.dtype(),
                },
                _ => Qwen3TokenEmbedding::Dense(mlx::load_embedding(
                    cfg.vocab_size,
                    cfg.hidden_size,
                    embed_vb,
                )?),
            };
        // A tied head is the token table, even where the checkpoint also
        // stores a copy (Qwen/Qwen3-0.6B and -1.7B do): the copy is unread.
        let tied = cfg.tie_word_embeddings && lm_head_size == cfg.vocab_size;
        let lm_head = if let Some(lm_head_prefix) = layout.lm_head_prefix {
            let lm_head_weight = qwen3_join_prefix(lm_head_prefix, "weight");
            if vb.contains_tensor(&lm_head_weight) && !tied {
                Qwen3Projection::dense(cfg.hidden_size, lm_head_size, vb.pp(lm_head_prefix))?
            } else {
                if lm_head_size != cfg.vocab_size {
                    return Err(Error::InvalidInput(format!(
                        "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but {} is missing",
                        cfg.vocab_size, lm_head_weight
                    )));
                }
                embed_tokens.tied_head()?
            }
        } else {
            if lm_head_size != cfg.vocab_size {
                return Err(Error::InvalidInput(format!(
                    "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but no lm_head prefix was configured",
                    cfg.vocab_size
                )));
            }
            embed_tokens.tied_head()?
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Qwen3Layer::load(
                &cfg,
                vb.pp(qwen3_join_prefix(
                    layout.model_prefix,
                    &format!("layers.{idx}"),
                )),
            )?;
            layers.push(layer);
        }
        let norm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp(qwen3_join_prefix(layout.model_prefix, "norm")),
        )?;
        let use_mrope = cfg
            .rope_scaling
            .as_ref()
            .map(|scaling| {
                scaling.mrope_interleaved.unwrap_or(false) || scaling.interleaved.unwrap_or(false)
            })
            .unwrap_or(false);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            cfg,
            use_mrope,
            decode_rope_table: Mutex::new(None),
            decode_graph: Err("not enabled for this model".into()),
        })
    }

    pub fn load_gguf_text(
        cfg: Qwen3Config,
        loader: &GgufLoader,
        device: &Device,
        dtype: DType,
        prefix: &str,
    ) -> Result<Self> {
        cfg.attention_geometry()?;
        let lm_head_size = cfg.lm_head_size.unwrap_or(cfg.vocab_size);
        let model_prefix = qwen3_join_prefix(prefix, "model");
        let embed_name = qwen3_join_prefix(&model_prefix, "embed_tokens.weight");
        let embed_name = qwen3_gguf_tensor_name(loader, &embed_name).unwrap_or(embed_name);
        let table = QuantizedTokenTable::new(loader.load_qtensor(&embed_name, device)?)?;
        if table.hidden_size() != cfg.hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Qwen3 token embedding width {} differs from hidden_size {}",
                table.hidden_size(),
                cfg.hidden_size
            )));
        }
        let lm_head_name = qwen3_join_prefix(prefix, "lm_head.weight");
        let stored_head = qwen3_gguf_tensor_name(loader, &lm_head_name).is_some();
        let lm_head = if stored_head && !cfg.tie_word_embeddings {
            Qwen3Projection::quantized(loader, device, &lm_head_name)?
        } else {
            if lm_head_size != cfg.vocab_size {
                return Err(Error::InvalidInput(format!(
                    "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but the head is tied to the embedding",
                    cfg.vocab_size
                )));
            }
            // A tied head (a stored copy of the table, as Qwen3-ASR
            // checkpoints carry, is not read) shares the token table.
            Qwen3Projection::from_qmatmul(table.tied_head()?)
        };
        let embed_tokens = Qwen3TokenEmbedding::Quantized { table, dtype };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer_prefix = qwen3_join_prefix(&model_prefix, &format!("layers.{idx}"));
            let layer = Qwen3Layer::load_gguf(loader, &cfg, device, dtype, &layer_prefix)?;
            layers.push(layer);
        }
        let norm_name = qwen3_join_prefix(&model_prefix, "norm.weight");
        let norm_name = qwen3_gguf_tensor_name(loader, &norm_name).unwrap_or(norm_name);
        let norm = RmsNorm::new(
            loader.load_tensor(&norm_name, dtype, device)?,
            cfg.rms_norm_eps,
        );
        let use_mrope = cfg
            .rope_scaling
            .as_ref()
            .map(|scaling| {
                scaling.mrope_interleaved.unwrap_or(false) || scaling.interleaved.unwrap_or(false)
            })
            .unwrap_or(false);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            cfg,
            use_mrope,
            decode_rope_table: Mutex::new(None),
            decode_graph: Err("not enabled for this model".into()),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn attention_head_dim(&self) -> usize {
        self.cfg.head_dim()
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Rows of the language-model head: the logits width.
    pub fn lm_head_size(&self) -> usize {
        self.cfg.lm_head_size.unwrap_or(self.cfg.vocab_size)
    }

    pub fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.layers
            .iter()
            .fold(self.lm_head.diagnostics(), |acc, layer| {
                acc.add(layer.projection_diagnostics())
            })
    }

    /// Per-layer expert-activation counters of the sparse layers (empty when
    /// the model is dense). Engine-side telemetry only: protocol exposure is
    /// deferred to expert-parallel activation (DS10 A6).
    pub fn expert_activation_counters(&self) -> Vec<Arc<ExpertActivationCounters>> {
        self.layers
            .iter()
            .filter_map(|layer| match &layer.mlp {
                Qwen3FeedForward::Sparse(sparse) => Some(sparse.counters()),
                Qwen3FeedForward::Dense(_) => None,
            })
            .collect()
    }

    /// Execute native Qwen3 against backend-owned authoritative KV pages.
    ///
    /// Initial prefill writes each layer directly into the supplied arena and
    /// advances the logical context only after every layer succeeds. Decode
    /// consumes those same pages without constructing a `Qwen3Cache`.
    pub fn forward_managed(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len) = input_ids.dims2()?;
        if batch_size != 1 || sequence_len == 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3 managed forward expects [1,sequence], got {:?}",
                input_ids.dims()
            )));
        }
        let embeds = self.embeddings(input_ids)?;
        self.forward_managed_with_embeds(&embeds, start_pos, cache, None)
    }

    /// Execute caller-provided embeddings against authoritative KV pages.
    ///
    /// Multimodal adapters use this entry point after replacing placeholder
    /// token embeddings with modality features. The cache behavior is exactly
    /// the same as [`Self::forward_managed`]: every layer writes the supplied
    /// arena and the logical context advances only after the full pass succeeds.
    pub fn forward_managed_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (hidden, prepared) =
            self.prepare_managed_hidden_with_embeds(embeds, start_pos, cache, position_ids)?;
        let logits = self.logits_from_hidden(&hidden)?;
        cache.commit_prepared(prepared)?;
        Ok(logits)
    }

    /// [`Self::forward_managed_with_embeds`] for a prefill that only samples
    /// the next token: `lm_head` runs on the last row alone and the result is
    /// `[1, 1, vocab]`. A 30 s Qwen3-ASR chunk otherwise projects ~400 rows
    /// of 151,936 logits (~243 MB F32) to read one. The prepared KV writes
    /// still commit only after the logits succeed.
    pub fn forward_managed_last_logits_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (hidden, prepared) =
            self.prepare_managed_hidden_with_embeds(embeds, start_pos, cache, position_ids)?;
        let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
        let logits = self.logits_from_hidden(&last)?;
        cache.commit_prepared(prepared)?;
        Ok(logits)
    }

    /// Execute caller-provided embeddings against authoritative KV pages and
    /// return the normalized decoder hidden states without applying `lm_head`.
    ///
    /// Diffusion-conditioned adapters consume these states directly. The
    /// physical page transition remains atomic: the prepared write set is
    /// committed only after all decoder layers and the final norm succeed.
    pub fn forward_managed_hidden_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (hidden, prepared) =
            self.prepare_managed_hidden_with_embeds(embeds, start_pos, cache, position_ids)?;
        cache.commit_prepared(prepared)?;
        Ok(hidden)
    }

    /// Commit caller-provided embeddings to managed KV without retaining a
    /// decoder output or projecting logits.
    ///
    /// Resumable multimodal prefill uses this for every non-final scheduler
    /// span. Only the current span's transient layer tensor exists during the
    /// call; processed spans are represented solely by their committed KV.
    pub fn forward_managed_prefill_only_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
    ) -> Result<()> {
        let (_last_layer, prepared) =
            self.prepare_managed_layers_with_embeds(embeds, start_pos, cache, position_ids)?;
        cache.commit_prepared(prepared)?;
        Ok(())
    }

    fn prepare_managed_hidden_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, PreparedPhysicalPagedStep)> {
        let (x, prepared) =
            self.prepare_managed_layers_with_embeds(embeds, start_pos, cache, position_ids)?;
        let hidden = self.norm.forward(&x)?;
        Ok((hidden, prepared))
    }

    fn prepare_managed_layers_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, PreparedPhysicalPagedStep)> {
        let (batch_size, sequence_len, hidden_size) = embeds.dims3()?;
        if batch_size != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Qwen3 managed embeddings expect [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                embeds.dims()
            )));
        }
        if self.cfg.sliding_window().is_some() {
            return Err(Error::InvalidInput(
                "Qwen3 managed execution does not yet support sliding-window block tables"
                    .to_string(),
            ));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.attention_geometry()?.key_head_dim(),
        )?;
        cache.slots_for_append(start_pos, sequence_len)?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let position_ids =
            self.elide_sequential_mrope_positions(start_pos, sequence_len, position_ids)?;

        let mut x = embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_managed(
                &x,
                start_pos,
                position_ids,
                cache,
                &mut prepared,
                layer_idx,
            )?;
        }
        Ok((x, prepared))
    }

    /// One direct paged-attention call per layer for a ragged batch of native
    /// Qwen3 decode rows sharing the same physical arena.
    pub fn forward_managed_decode_batch(
        &self,
        input_ids: &Tensor,
        start_positions: &[usize],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let (batch_size, sequence_len) = input_ids.dims2()?;
        if sequence_len != 1
            || batch_size == 0
            || start_positions.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3 managed decode expects matching [batch,1] rows, got {:?}",
                input_ids.dims()
            )));
        }
        if self.cfg.sliding_window().is_some() {
            return Err(Error::InvalidInput(
                "Qwen3 managed execution does not yet support sliding-window block tables"
                    .to_string(),
            ));
        }
        for (row, cache) in caches.iter().enumerate() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.attention_geometry()?.key_head_dim(),
            )?;
            cache.slots_for_append(start_positions[row], 1)?;
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
        let lowered = first.arena.lower_slots(&combined_slots)?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(start_positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let mut completions =
            KvWriteCompletionCollector::new(first.arena.config(), lowered.logical_slots())?;
        let cache_refs = caches
            .iter()
            .map(|cache| &**cache)
            .collect::<Vec<&PhysicalPagedKvCache>>();
        let graphed = match self
            .decode_graph
            .as_ref()
            .ok()
            .filter(|graph| graph.is_on())
        {
            Some(graph) => self.graph_step(
                graph,
                input_ids,
                start_positions,
                &combined_slots,
                lowered.as_ref(),
                &metadata,
                &cache_refs,
                &mut completions,
            )?,
            None => None,
        };
        let logits = match graphed {
            Some(logits) => logits,
            None => self.standard_decode_logits(
                input_ids,
                start_positions,
                &cache_refs,
                lowered.as_ref(),
                &metadata,
                &mut completions,
            )?,
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(start_positions[row], 1, completion.clone())?;
        }
        Ok(logits)
    }

    /// The standard batched decode step: the rows' logits, their K/V
    /// writes collected in `completions`.
    fn standard_decode_logits(
        &self,
        input_ids: &Tensor,
        start_positions: &[usize],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<Tensor> {
        let mut x = self.embeddings(input_ids)?;
        if let Some(first_layer) = self.layers.first() {
            // Every layer shares one RoPE geometry: gather the rows' cos/sin
            // once for the whole step.
            let rope = first_layer.self_attn.decode_rope_rows(
                &self.decode_rope_table,
                start_positions,
                &self.device,
                x.dtype(),
            )?;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                x = layer.forward_managed_decode_batch(
                    &x,
                    &rope,
                    caches,
                    slots,
                    metadata,
                    completions,
                    layer_idx,
                )?;
            }
        }
        let hidden = self.norm.forward(&x)?;
        self.logits_from_hidden(&hidden)
    }

    /// Turns on `label`'s whole-step CUDA decode graphs behind `switch`
    /// ([`super::decode_graph`]): CUDA only, and only for a dense
    /// feed-forward trunk whose every projection a graph can capture.
    pub(crate) fn enable_asr_decode_graph(&mut self, label: &'static str, switch: &'static str) {
        self.decode_graph = if self.cfg.sliding_window().is_some() {
            Err("sliding-window attention has no graph".into())
        } else if !self.layers.iter().all(Qwen3Layer::graph_capturable)
            || !self.lm_head.graph_capturable()
        {
            Err("a projection has no K-quant kernel or the feed-forward is sparse".into())
        } else {
            Qwen3DecodeGraph::resolve(label, switch, &self.device)
        };
    }

    /// Whether decode steps go through the CUDA decode graphs: a lone row
    /// then decodes as a batch of one, through its width-1 graph.
    pub(crate) fn decode_graph_on(&self) -> bool {
        self.decode_graph
            .as_ref()
            .is_ok_and(Qwen3DecodeGraph::is_on)
    }

    pub(crate) fn decode_graph_diagnostics(&self) -> serde_json::Value {
        match &self.decode_graph {
            Ok(graph) => graph.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        }
    }

    /// One decode step through `graph`, its K/V write completions collected
    /// in `completions`: `None` when the step does not fit a graph (more
    /// rows than the widest, a KV arena or geometry the split decode does
    /// not serve), so the standard decode runs it. A width's first graphed
    /// step is checked against the standard decode; a mismatch turns the
    /// graphs off and the step keeps the standard logits.
    #[allow(clippy::too_many_arguments)]
    fn graph_step(
        &self,
        graph: &Qwen3DecodeGraph,
        input_ids: &Tensor,
        start_positions: &[usize],
        slots: &[KvSlotRef],
        slot_map: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        caches: &[&PhysicalPagedKvCache],
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<Option<Tensor>> {
        let rows = start_positions.len();
        let (Some(width), Some(first_layer)) =
            (decode_step::graph_width(rows), self.layers.first())
        else {
            return Ok(None);
        };
        let attention = &first_layer.self_attn;
        let first = caches[0];
        let arena = first.arena.clone();
        let config = arena.config();
        let dtype = self.embed_tokens.dtype();
        if caches
            .iter()
            .any(|cache| !Arc::ptr_eq(&cache.arena, &arena))
            || config.dtype != dtype
            || !split_decode::supported(
                dtype,
                attention.num_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.head_dim,
            )
        {
            return Ok(None);
        }
        let page_tokens = config.page_tokens;
        let positions = start_positions
            .iter()
            .map(|&pos| {
                u32::try_from(attention.decode_rope_position(pos))
                    .map_err(|_| Error::InvalidInput("Qwen3 RoPE position exceeds u32".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let needed = positions.iter().max().map_or(1, |&pos| pos as usize + 1);
        let (cos, sin, rope_rows) = attention.decode_rope_table_rows(
            &self.decode_rope_table,
            needed,
            &self.device,
            dtype,
        )?;
        let step_rows = positions
            .iter()
            .zip(slots)
            .zip(&metadata.sequences)
            .map(|((position, slot), sequence)| {
                DecodeStepRow::new(*position, slot, sequence, page_tokens)
            })
            .collect::<Result<Vec<_>>>()?;
        let longest = step_rows
            .iter()
            .map(|row| row.blocks.len())
            .max()
            .unwrap_or(1);
        let layout = DecodeStepLayout::for_blocks(width, longest);
        let step = layout.upload(&step_rows, &self.device)?;
        let ids = input_ids.reshape((rows, 1))?;
        let ids = if width > rows {
            ids.pad_with_zeros(0, 0, width - rows)?
        } else {
            ids
        };
        let bindings = (0..self.layers.len())
            .map(|layer| first.layer_binding(layer))
            .collect::<Result<Vec<_>>>()?;
        let page_tokens = page_tokens as usize;
        let max_partitions = decode_step::max_partitions(
            width,
            attention.num_kv_heads,
            layout.max_context(page_tokens),
        );
        // The graph keeps stable copies of its inputs and its logits.
        let tensor_bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        let retained = tensor_bytes(&ids)
            + tensor_bytes(&step)
            + (width * self.cfg.vocab_size * std::mem::size_of::<f32>()) as u64;
        let inputs = [ids, step];
        let mut output = None;
        let graph_completions = arena.with_graph_writes(
            &bindings,
            slot_map,
            self.layers.len() as u64,
            &mut |layers| {
                let key = GraphKey {
                    width,
                    arena: arena.id(),
                    generation: layers.generation,
                    max_blocks: layout.max_blocks,
                    rope_rows,
                };
                let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                    let [ids, step] = inputs else {
                        return Err(Error::InferenceError(format!(
                            "Qwen3 decode graph received {} inputs, expected 2",
                            inputs.len()
                        )));
                    };
                    let attention = Qwen3GraphAttention {
                        caches: &layers.layers,
                        slots: step.narrow(0, layout.slots(), width)?,
                        metadata: step.narrow(0, layout.metadata(), layout.metadata_len())?,
                        page_tokens,
                        max_blocks: layout.max_blocks,
                        max_partitions,
                    };
                    let positions = step.narrow(0, layout.positions(), width)?;
                    Ok(vec![self.graph_decode_body(
                        ids, &positions, &cos, &sin, &attention,
                    )?])
                };
                output = graph.graphs.run(key, retained, &inputs, &body)?.pop();
                Ok(())
            },
        )?;
        for completion in graph_completions {
            completions.collect(completion)?;
        }
        let logits = output
            .ok_or_else(|| Error::InferenceError("Qwen3 decode graph returned nothing".into()))?;
        let logits = if width > rows {
            logits.narrow(0, 0, rows)?
        } else {
            logits
        };
        graph.record_step(rows, width);
        if !graph.needs_check(width) {
            return Ok(Some(logits));
        }
        // The check: the standard decode of the same rows, its writes (the
        // same K/V) collected and sealed apart.
        let mut reference_completions =
            KvWriteCompletionCollector::new(config, slot_map.logical_slots())?;
        let reference = self.standard_decode_logits(
            input_ids,
            start_positions,
            caches,
            slot_map,
            metadata,
            &mut reference_completions,
        );
        let sealed = reference_completions.seal();
        let reference = reference?;
        sealed?;
        match decode_step::logits_agree(&logits, &reference) {
            Ok(_) => {
                graph.mark_checked(width);
                Ok(Some(logits))
            }
            Err(reason) => {
                graph.disable(format!("a {rows}-row step (width {width}): {reason}"));
                Ok(Some(reference))
            }
        }
    }

    /// The decode graphs' body over a padded step: token embeddings, every
    /// layer through [`Qwen3Layer::forward_graph_decode`], the final norm
    /// and the logits, quantized projections on the K-quant kernels.
    fn graph_decode_body(
        &self,
        ids: &Tensor,
        positions: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention: &Qwen3GraphAttention<'_>,
    ) -> Result<Tensor> {
        let _projections = GraphProjections::enter();
        let rope = Qwen3DecodeRopeRows {
            cos: cos.index_select(positions, 0)?,
            sin: sin.index_select(positions, 0)?,
        };
        let mut x = self.embeddings(ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_graph_decode(&x, &rope, attention, layer_idx)?;
        }
        let hidden = self.norm.forward(&x)?;
        self.logits_from_hidden(&hidden)
    }

    /// One forward over a mixed step: `input_ids` holds every row's new
    /// tokens in `layout` order (one-token decode rows first, then prefill
    /// spans), and `caches[i]` is row `i`'s cache. Every token runs through
    /// the projections and MLP together; each layer writes all K/V once and
    /// attends in at most two paged calls. Returns `[sample_rows, vocab]`
    /// logits of the last token of each row in `sample_rows`, or `None` when
    /// no row samples. Caches advance only when the whole forward succeeds.
    pub(crate) fn forward_managed_mixed(
        &self,
        input_ids: &[u32],
        layout: MixedRowLayout,
        caches: &mut [&mut PhysicalPagedKvCache],
        sample_rows: &[usize],
    ) -> Result<Option<Tensor>> {
        if input_ids.len() != layout.total_tokens()
            || caches.len() != layout.rows().len()
            || sample_rows.iter().any(|row| *row >= caches.len())
        {
            return Err(Error::InvalidInput(
                "Qwen3 mixed step inputs do not match its row layout".to_string(),
            ));
        }
        if self.cfg.sliding_window().is_some() {
            return Err(Error::InvalidInput(
                "Qwen3 managed execution does not yet support sliding-window block tables"
                    .to_string(),
            ));
        }
        let key_head_dim = self.cfg.attention_geometry()?.key_head_dim();
        for cache in caches.iter() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                key_head_dim,
            )?;
        }
        let sample_tokens = sample_rows
            .iter()
            .map(|row| u32::try_from(layout.last_token(*row)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::InvalidInput("Qwen3 mixed step exceeds u32 tokens".into()))?;
        let (step, logits) = {
            let refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            let mut step = PreparedPhysicalMixedStep::new(&refs, layout)?;
            let logits = (|| -> Result<Option<Tensor>> {
                let ids = Tensor::from_slice(input_ids, (1, input_ids.len()), &self.device)?;
                let mut x = self.embeddings(&ids)?;
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    x = layer.forward_managed_mixed(&x, &refs, &mut step, layer_idx)?;
                }
                if sample_tokens.is_empty() {
                    return Ok(None);
                }
                let hidden = self.norm.forward(&x)?.squeeze(0)?;
                let sample_tokens =
                    Tensor::from_vec(sample_tokens, sample_rows.len(), &self.device)?;
                self.logits_from_hidden(&hidden.index_select(&sample_tokens, 0)?)
                    .map(Some)
            })();
            (step, logits)
        };
        step.finish(caches, logits)
    }

    /// MRoPE position ids that hold `start_pos..start_pos + seq_len` on every
    /// axis (all of Qwen3-ASR's: it has no vision spans) are plain 1-D RoPE.
    /// Check them with one readback per forward and drop them, so every layer
    /// builds host tables from `start_pos` instead of reading the ids back
    /// itself. Any other ids (vision spans) pass through unchanged.
    fn elide_sequential_mrope_positions<'a>(
        &self,
        start_pos: usize,
        seq_len: usize,
        position_ids: Option<&'a Tensor>,
    ) -> Result<Option<&'a Tensor>> {
        let Some(ids) = position_ids else {
            return Ok(None);
        };
        let host_tables = self
            .layers
            .first()
            .is_some_and(|layer| layer.self_attn.sequential_mrope_uses_host_tables());
        if !host_tables || ids.dims() != [3, seq_len] || ids.dtype() != DType::I64 {
            return Ok(Some(ids));
        }
        record_mrope_position_readback();
        let sequential = ids.to_vec2::<i64>()?.iter().all(|axis| {
            axis.iter()
                .enumerate()
                .all(|(t, &pos)| u64::try_from(pos).ok() == Some((start_pos + t) as u64))
        });
        Ok(if sequential { None } else { Some(ids) })
    }

    pub fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// A load-time numeric probe: `tokens` xorshift-chosen token ids
    /// through every layer and the head with stateless attention. Returns
    /// the logits for a range self-check.
    pub(crate) fn stateless_numeric_probe(
        &self,
        tokens: usize,
        seed: u64,
    ) -> Result<Vec<(String, Tensor)>> {
        let vocab = self.embed_tokens.vocab_size()?.max(1);
        let ids = crate::kernels::numeric_check::xorshift_values(tokens, seed, 1.0)
            .into_iter()
            .map(|unit| (((unit + 1.0) * 0.5 * vocab as f32) as usize).min(vocab - 1) as u32)
            .collect::<Vec<_>>();
        let device = self.device.clone();
        let ids = Tensor::from_vec(ids, (1, tokens), &device)?;
        let logits = self.forward_stateless_with_embeds(&self.embeddings(&ids)?, 0, None)?;
        Ok(vec![("logits".to_string(), logits)])
    }

    pub fn forward_stateless_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden = self.stateless_normed_hidden(embeds, start_pos, position_ids)?;
        self.logits_from_hidden(&hidden)
    }

    /// A stateless forward that scores only the prompt `rows`: the normed
    /// hidden state is gathered at those positions before the head, so the
    /// logits are `[1, rows.len(), vocab]` instead of one row per prompt
    /// position (QFA-9: the forced aligner reads only its timestamp rows).
    pub fn forward_stateless_rows_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        rows: &[usize],
    ) -> Result<Tensor> {
        let (batch, sequence, _) = embeds.dims3()?;
        if batch != 1 {
            return Err(Error::InvalidInput(format!(
                "Qwen3 row-scored forward takes one sequence, got {batch}"
            )));
        }
        if let Some(row) = rows.iter().find(|row| **row >= sequence) {
            return Err(Error::InferenceError(format!(
                "Qwen3 scored row {row} is out of range for sequence length {sequence}"
            )));
        }
        let rows = rows
            .iter()
            .map(|row| u32::try_from(*row).map_err(|_| Error::InvalidInput("row index".into())))
            .collect::<Result<Vec<_>>>()?;
        let index = Tensor::from_vec(rows.clone(), rows.len(), embeds.device())?;
        let hidden = self.stateless_normed_hidden(embeds, start_pos, position_ids)?;
        self.logits_from_hidden(&hidden.index_select(&index, 1)?)
    }

    /// The final-normed hidden state of a stateless forward over `embeds`.
    /// The RoPE tables are built once and shared by every layer.
    fn stateless_normed_hidden(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let position_ids =
            self.elide_sequential_mrope_positions(start_pos, embeds.dim(1)?, position_ids)?;
        let mut rope_tables = None;
        let mut hidden = embeds.clone();
        for layer in &self.layers {
            hidden = layer.forward_stateless(&hidden, start_pos, position_ids, &mut rope_tables)?;
        }
        self.norm.forward(&hidden).map_err(Error::from)
    }

    pub fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        #[cfg(test)]
        QWEN3_HEAD_ROWS_THREAD.with(|count| {
            let rows = hidden.dims()[..hidden.rank().saturating_sub(1)]
                .iter()
                .product::<usize>();
            count.set(count.get() + rows as u64)
        });
        if !qwen3_runtime_profiling_enabled() {
            return self.lm_head.forward(hidden).map_err(Error::from);
        }
        let started = Instant::now();
        let logits = self.lm_head.forward(hidden).map_err(Error::from);
        QWEN3_PROFILE_LM_HEAD_CALLS.fetch_add(1, Ordering::Relaxed);
        if logits.is_ok() {
            QWEN3_PROFILE_LM_HEAD_NS.fetch_add(elapsed_ns(started), Ordering::Relaxed);
        }
        logits
    }

    pub fn logits_from_hidden_for_tokens(
        &self,
        hidden: &Tensor,
        token_ids: &[u32],
    ) -> Result<Vec<(u32, f32)>> {
        self.lm_head.logits_for_tokens(hidden, token_ids)
    }

    /// [`Self::logits_from_hidden_for_tokens`] for every row of `hidden`
    /// (`[rows, hidden]` or `[rows, 1, hidden]`): `[rows, token_ids.len()]`
    /// logits left on the device, so one readback scores every row.
    pub(crate) fn logits_rows_for_tokens(
        &self,
        hidden: &Tensor,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        let hidden = hidden.flatten_to(hidden.rank().saturating_sub(2))?;
        match &self.lm_head {
            Qwen3Projection::Dense(dense) => dense_logits_for_tokens(dense, &hidden, token_ids),
            Qwen3Projection::Quantized(_) => self
                .lm_head
                .forward(&hidden)?
                .index_select(&token_index_tensor(token_ids, hidden.device())?, 1)
                .map_err(Error::from),
        }
    }

    /// [`Self::forward_managed_decode_batch`] for caller-provided embeddings
    /// (`[batch, 1, hidden]`), returning the normalized hidden states
    /// (`[batch, 1, hidden]`) without `lm_head`: diffusion-conditioned
    /// decode rows (VibeVoice TTS) step every row in one launch per op. The
    /// rows share one physical arena; every cache advances only when the
    /// whole forward succeeds.
    pub(crate) fn forward_managed_decode_batch_hidden_with_embeds(
        &self,
        embeds: &Tensor,
        start_positions: &[usize],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, hidden_size) = embeds.dims3()?;
        if sequence_len != 1
            || batch_size == 0
            || hidden_size != self.cfg.hidden_size
            || start_positions.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3 managed embedding decode expects matching [batch,1,{}] rows, got {:?}",
                self.cfg.hidden_size,
                embeds.dims()
            )));
        }
        if self.cfg.sliding_window().is_some() {
            return Err(Error::InvalidInput(
                "Qwen3 managed execution does not yet support sliding-window block tables"
                    .to_string(),
            ));
        }
        let key_head_dim = self.cfg.attention_geometry()?.key_head_dim();
        let mut combined_slots = Vec::with_capacity(batch_size);
        for (cache, &start) in caches.iter().zip(start_positions) {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                key_head_dim,
            )?;
            combined_slots.push(cache.slots_for_append(start, 1)?[0]);
        }
        let first = &*caches[0];
        let lowered = first.arena.lower_slots(&combined_slots)?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .zip(start_positions)
                .map(|(cache, &start)| cache.sequence_table(start + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let mut completions =
            KvWriteCompletionCollector::new(first.arena.config(), lowered.logical_slots())?;
        let mut x = embeds.clone();
        if let Some(first_layer) = self.layers.first() {
            let rope = first_layer.self_attn.decode_rope_rows(
                &self.decode_rope_table,
                start_positions,
                &self.device,
                x.dtype(),
            )?;
            let cache_refs = caches
                .iter()
                .map(|cache| &**cache)
                .collect::<Vec<&PhysicalPagedKvCache>>();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                x = layer.forward_managed_decode_batch(
                    &x,
                    &rope,
                    &cache_refs,
                    lowered.as_ref(),
                    &metadata,
                    &mut completions,
                    layer_idx,
                )?;
            }
        }
        let hidden = self.norm.forward(&x)?;
        let completion = Arc::new(completions.seal()?);
        for (cache, &start) in caches.iter_mut().zip(start_positions) {
            cache.commit_shared_completion(start, 1, completion.clone())?;
        }
        Ok(hidden)
    }

    pub fn uses_mrope(&self) -> bool {
        self.use_mrope
    }
}

fn dense_logits_for_tokens(
    dense: &Qwen3DenseProjection,
    hidden: &Tensor,
    token_ids: &[u32],
) -> Result<Tensor> {
    let token_indices = token_index_tensor(token_ids, hidden.device())?;
    let weight_rows = dense.linear.weight().index_select(&token_indices, 0)?;
    let mut logits = hidden.matmul(&weight_rows.t()?)?;
    if let Some(bias) = dense.linear.bias() {
        let bias_rows = bias.index_select(&token_indices, 0)?;
        logits = logits.broadcast_add(&bias_rows)?;
    }
    Ok(logits)
}

fn logits_for_tokens_from_full(logits: &Tensor, token_ids: &[u32]) -> Result<Tensor> {
    let token_indices = token_index_tensor(token_ids, logits.device())?;
    match logits.dims() {
        [1, _] => logits
            .i(0)?
            .index_select(&token_indices, 0)
            .map_err(Error::from),
        [1, 1, _] => logits
            .i((0, 0))?
            .index_select(&token_indices, 0)
            .map_err(Error::from),
        _ => Err(Error::InferenceError(format!(
            "Qwen3 selected-token logits expected [1,vocab] or [1,1,vocab], got {:?}",
            logits.dims()
        ))),
    }
}

fn token_index_tensor(token_ids: &[u32], device: &Device) -> Result<Tensor> {
    let indices = token_ids
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    Tensor::from_vec(indices, (token_ids.len(),), device).map_err(Error::from)
}

fn qwen3_join_prefix(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

pub fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_heads == num_kv_heads {
        return Ok(x.clone());
    }
    if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidInput(format!(
            "Invalid GQA head config: num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )));
    }
    let repeats = num_heads / num_kv_heads;
    let seq_len = x.dim(1)?;
    let x = x.transpose(1, 2)?; // [b, kv, seq, d]
    let repeated = candle_repeat_kv(x, repeats)?;
    match repeated.rank() {
        4 => repeated.transpose(1, 2).map_err(Error::from),
        5 => {
            let (b, _kv_or_rep0, a2, a3, d) = repeated.dims5()?;
            let repeated = if a2 == seq_len && a3 == repeats {
                repeated
            } else if a3 == seq_len && a2 == repeats {
                repeated.transpose(2, 3)?
            } else {
                return Err(Error::InferenceError(format!(
                    "Unexpected repeat_kv rank-5 shape: {:?}, seq_len={seq_len}, repeats={repeats}",
                    repeated.dims()
                )));
            };
            let repeated = repeated.transpose(1, 2)?; // [b, seq, kv, repeats, d]
            repeated
                .reshape((b, seq_len, num_heads, d))
                .map_err(Error::from)
        }
        rank => Err(Error::InferenceError(format!(
            "Unexpected repeat_kv rank {rank}: {:?}",
            repeated.dims()
        ))),
    }
}

pub(crate) fn build_rope_inv_freqs(head_dim: usize, rope_theta: f64) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let power = (2.0 * i as f64) / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(power)) as f32);
    }
    inv_freq
}

fn build_rope_cache_with_inv_freqs(
    seq_len: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
    inv_freqs: &[f32],
) -> Result<(Tensor, Tensor)> {
    let half_dim = inv_freqs.len();
    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for &inv in inv_freqs {
            angles.push(pos as f32 * inv);
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
    Ok((
        angles.cos()?.to_dtype(dtype)?,
        angles.sin()?.to_dtype(dtype)?,
    ))
}

/// Host cos/sin tables for positions `start_pos..start_pos + seq_len`.
///
/// Bit-identical to [`build_mrope_cache_with_inv_freqs`] when every axis holds
/// those positions (each angle is `pos as f32 * inv` and its `cos`/`sin` are
/// taken on the host), so precision does not depend on the device's
/// transcendental functions (Metal compiles them with fast math).
fn build_sequential_rope_host_tables(
    seq_len: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
    inv_freqs: &[f32],
) -> Result<(Tensor, Tensor)> {
    let half_dim = inv_freqs.len();
    let mut cos_data = Vec::with_capacity(seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        let pos = pos as f32;
        for &inv in inv_freqs {
            let angle = pos * inv;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }
    QWEN3_SEQUENTIAL_ROPE_HOST_TABLES.fetch_add(1, Ordering::Relaxed);
    Ok((
        Tensor::from_vec(cos_data, (seq_len, half_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin_data, (seq_len, half_dim), device)?.to_dtype(dtype)?,
    ))
}

fn build_mrope_cache_with_inv_freqs(
    seq_len: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
    position_ids: &Tensor,
    mrope_section: &[usize],
    inv_freqs: &[f32],
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    if mrope_section.len() < 3 {
        return build_rope_cache_with_inv_freqs(seq_len, 0, device, dtype, inv_freqs);
    }

    record_mrope_position_readback();
    let positions = position_ids.to_vec2::<i64>()?;
    if positions.len() != 3 || positions.iter().any(|axis| axis.len() < seq_len) {
        return build_rope_cache_with_inv_freqs(seq_len, 0, device, dtype, inv_freqs);
    }

    // Match Qwen3 Omni's interleaved MRoPE layout:
    // T,H,W,T,H,W,... for the first 3*section dims, then T for the tail.
    let h_limit = mrope_section[1].saturating_mul(3).min(half_dim);
    let w_limit = mrope_section[2].saturating_mul(3).min(half_dim);

    let mut cos_data = Vec::with_capacity(seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(seq_len * half_dim);
    for t in 0..seq_len {
        let p0 = positions[0][t] as f32;
        let p1 = positions[1][t] as f32;
        let p2 = positions[2][t] as f32;
        for (dim, &inv) in inv_freqs.iter().enumerate() {
            let pos = if dim % 3 == 1 && dim < h_limit {
                p1
            } else if dim % 3 == 2 && dim < w_limit {
                p2
            } else {
                p0
            };
            let angle = pos * inv;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }

    Ok((
        Tensor::from_vec(cos_data, (seq_len, half_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin_data, (seq_len, half_dim), device)?.to_dtype(dtype)?,
    ))
}

pub fn build_rope_cache(
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let inv_freqs = build_rope_inv_freqs(head_dim, rope_theta);
    build_rope_cache_with_inv_freqs(seq_len, start_pos, device, dtype, &inv_freqs)
}

pub fn build_mrope_cache(
    seq_len: usize,
    head_dim: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
    position_ids: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let inv_freqs = build_rope_inv_freqs(head_dim, rope_theta);
    build_mrope_cache_with_inv_freqs(
        seq_len,
        head_dim,
        device,
        dtype,
        position_ids,
        mrope_section,
        &inv_freqs,
    )
}

/// Query rows per block of the portable causal prefill (CPU, and CUDA
/// without FlashAttention; Metal runs fused SDPA). The portable SDPA keeps
/// the scaled, masked and softmaxed `[heads, rows, keys]` scores alive at
/// once: about 3 GB per layer for a 4,000-token prompt (a 180 s aligner
/// window of ordinary speech). A block bounds that at `block / prompt` of it.
pub(crate) const QWEN3_STATELESS_QUERY_BLOCK: usize = 256;

/// Causal attention over a whole prompt from position 0, `block` query rows
/// at a time. Row `i` attends keys `0..=i`, so a block reads only the keys up
/// to its last row; the rows past it are masked in an unblocked pass anyway.
/// One block is exactly the unblocked computation.
fn causal_sdpa_by_query_blocks(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f64,
    block: usize,
) -> Result<Tensor> {
    let sequence = query.dim(2)?;
    let block = block.max(1);
    let mut outputs = Vec::with_capacity(sequence.div_ceil(block));
    // The last block attends the most keys. Running it first lets the host
    // allocator reuse its freed buffers for every later, smaller block;
    // ascending order grows each allocation and strands the freed ones.
    for start in (0..sequence).step_by(block).rev() {
        let end = (start + block).min(sequence);
        let rows = end - start;
        #[cfg(test)]
        QWEN3_SDPA_QUERY_BLOCKS_THREAD.with(|count| count.set(count.get() + 1));
        let mask = (end > 1)
            .then(|| causal_mask(rows, end, start, query.device(), query.dtype()))
            .transpose()?;
        let (query, key, value) = if rows == sequence {
            (query.clone(), key.clone(), value.clone())
        } else {
            (
                query.narrow(2, start, rows)?,
                key.narrow(2, 0, end)?,
                value.narrow(2, 0, end)?,
            )
        };
        outputs.push(compact_gqa_sdpa_bhsd(
            &query,
            &key,
            &value,
            mask.as_ref().map(CompactGqaMask::Additive),
            scale,
        )?);
    }
    if outputs.len() == 1 {
        return Ok(outputs.remove(0));
    }
    outputs.reverse();
    Tensor::cat(&outputs, 2).map_err(Error::from)
}

pub fn causal_mask(
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * total_len];
    for i in 0..seq_len {
        let limit = start_pos + i;
        for j in 0..total_len {
            if j > limit {
                data[i * total_len + j] = -1e4;
            }
        }
    }
    Tensor::from_vec(data, (1, seq_len, total_len), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn apply_rotary_emb(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos_half.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin_half.unsqueeze(0)?.unsqueeze(2)?;
    let out_first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let out_second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&out_first, &out_second], 3).map_err(Error::from)
}

fn try_apply_rope_pair_thd(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let q_out = rotary_emb::rope_thd(q, cos, sin)?;
        let k_out = rotary_emb::rope_thd(k, cos, sin)?;
        candle_core::Result::<(Tensor, Tensor)>::Ok((q_out, k_out))
    }));
    match kernel_result {
        Ok(Ok((q_out, k_out))) => Ok(Some((q_out, k_out))),
        failure => {
            crate::models::shared::telemetry::report_kernel_fallback("qwen3.rope", &failure);
            Ok(None)
        }
    }
}

fn qwen3_rope_kernel_enabled(device: &Device) -> bool {
    let override_enabled = crate::env::flag("IZWI_QWEN3_ROPE_KERNEL");
    qwen3_rope_kernel_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn qwen3_rope_kernel_policy(is_metal: bool, is_cuda: bool, override_enabled: Option<bool>) -> bool {
    if is_metal {
        return override_enabled.unwrap_or(true);
    }
    if is_cuda {
        return override_enabled.unwrap_or(true);
    }
    false
}

#[cfg(test)]
pub(crate) fn tiny_qwen3_model_for_test(device: &Device) -> Qwen3Model {
    tests::tiny_qwen3_model(device)
}

/// [`tiny_qwen3_model_for_test`] with non-finite embeddings, so its
/// load-time numeric probe fails.
#[cfg(test)]
pub(crate) fn tiny_non_finite_qwen3_model_for_test(device: &Device) -> Qwen3Model {
    let mut model = tests::tiny_qwen3_model(device);
    let shape = (
        model.embed_tokens.vocab_size().unwrap(),
        model.cfg.hidden_size,
    );
    let hidden = model.cfg.hidden_size;
    model.embed_tokens = Qwen3TokenEmbedding::Dense(Embedding::new(
        Tensor::full(f32::NAN, shape, device).unwrap(),
        hidden,
    ));
    model
}

#[cfg(test)]
#[path = "core_mixed_tests.rs"]
mod mixed_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{KvArenaId, KvGroupId};
    use candle_nn::rotary_emb;
    use std::collections::HashMap;

    fn attention_config(
        hidden_size: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: Option<usize>,
    ) -> Qwen3Config {
        Qwen3Config {
            hidden_size,
            intermediate_size: hidden_size.saturating_mul(2),
            num_attention_heads: query_heads,
            num_hidden_layers: 1,
            num_key_value_heads: kv_heads,
            max_position_embeddings: Some(128),
            head_dim,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            vocab_size: 32,
            lm_head_size: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            ada_rms_norm_t_cond: false,
            ada_rms_norm_t_cond_dim: 0,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            norm_topk_prob: None,
        }
    }

    #[test]
    fn qwen3_attention_geometry_accepts_explicit_gqa_width() {
        let config = attention_config(3072, 32, 8, Some(128));
        let geometry = config.attention_geometry().unwrap();
        assert_eq!(geometry.query_width(), 4096);
        assert_eq!(geometry.key_width(), 1024);
        assert_eq!(geometry.kv_groups(), 4);
    }

    #[test]
    fn qwen3_malformed_heads_return_errors_without_division_panics() {
        let zero_query = attention_config(4096, 0, 8, Some(128));
        assert_eq!(zero_query.head_dim(), 128);
        assert!(zero_query
            .attention_geometry()
            .unwrap_err()
            .to_string()
            .contains("head counts must be non-zero"));

        let inferred_zero_query = attention_config(4096, 0, 8, None);
        assert_eq!(inferred_zero_query.head_dim(), 0);
        assert!(inferred_zero_query.attention_geometry().is_err());

        let zero_kv = attention_config(4096, 32, 0, Some(128));
        assert_eq!(zero_kv.kv_groups(), 0);
        assert!(zero_kv.attention_geometry().is_err());

        let non_divisible = attention_config(1536, 12, 5, Some(128));
        assert!(non_divisible
            .attention_geometry()
            .unwrap_err()
            .to_string()
            .contains("query heads must be divisible"));

        let mut invalid_theta = attention_config(4096, 32, 8, Some(128));
        invalid_theta.rope_theta = f64::NAN;
        assert!(invalid_theta
            .attention_geometry()
            .unwrap_err()
            .to_string()
            .contains("rope_theta must be finite and positive"));
    }

    #[test]
    fn canonical_qwen3_gguf_names_cover_the_native_decoder_graph() {
        for (logical, canonical) in [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            (
                "model.layers.7.input_layernorm.weight",
                "blk.7.attn_norm.weight",
            ),
            (
                "model.layers.7.self_attn.q_proj.weight",
                "blk.7.attn_q.weight",
            ),
            (
                "model.layers.7.self_attn.k_norm.weight",
                "blk.7.attn_k_norm.weight",
            ),
            (
                "model.layers.7.self_attn.o_proj.weight",
                "blk.7.attn_output.weight",
            ),
            (
                "model.layers.7.post_attention_layernorm.weight",
                "blk.7.ffn_norm.weight",
            ),
            (
                "model.layers.7.mlp.gate_proj.weight",
                "blk.7.ffn_gate.weight",
            ),
            (
                "model.layers.7.mlp.down_proj.weight",
                "blk.7.ffn_down.weight",
            ),
        ] {
            assert_eq!(
                qwen3_canonical_gguf_tensor_name(logical).as_deref(),
                Some(canonical)
            );
        }
        assert!(qwen3_canonical_gguf_tensor_name("model.layers.7.unknown.weight").is_none());
    }

    #[test]
    fn cache_domain_is_derived_from_loaded_qwen3_geometry() {
        let domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
            domain: StateDomainId::new(4),
            clock: StateClock::DecoderTokens,
            num_layers: 3,
            num_query_heads: 12,
            num_kv_heads: 4,
            key_head_dim: 80,
            value_head_dim: 64,
            sliding_window: Some(1024),
            storage_dtype: DType::BF16,
            preferred_page_tokens: 32,
            prefix: PrefixPolicy::Disabled,
        })
        .unwrap();
        assert_eq!(domain.layers.len(), 3);
        assert_eq!(domain.layers[2].query_heads, 12);
        assert_eq!(domain.layers[2].kv_heads, 4);
        assert_eq!(domain.layers[2].key_head_dim, 80);
        assert_eq!(domain.layers[2].value_head_dim, 64);
        assert_eq!(domain.accepted_dtypes, vec![StateDType::Bf16]);
        assert_eq!(domain.page_size.preferred_tokens, 32);
    }

    fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor) {
        assert_eq!(lhs.dims(), rhs.dims());
        let lhs = lhs
            .to_dtype(DType::F32)
            .expect("lhs dtype")
            .flatten_all()
            .expect("lhs flat")
            .to_vec1::<f32>()
            .expect("lhs values");
        let rhs = rhs
            .to_dtype(DType::F32)
            .expect("rhs dtype")
            .flatten_all()
            .expect("rhs flat")
            .to_vec1::<f32>()
            .expect("rhs values");
        for (idx, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "tensor mismatch at {idx}: {lhs} != {rhs}"
            );
        }
    }

    fn test_projection(
        output_size: usize,
        input_size: usize,
        offset: usize,
        device: &Device,
    ) -> Qwen3Projection {
        let values = (0..output_size * input_size)
            .map(|idx| {
                let value = (idx.saturating_mul(7).saturating_add(offset)) % 23;
                (value as f32 - 11.0) / 32.0
            })
            .collect::<Vec<_>>();
        Qwen3Projection::Dense(Qwen3DenseProjection {
            linear: Linear::new(
                Tensor::from_vec(values, (output_size, input_size), device).unwrap(),
                None,
            ),
            has_bias: false,
        })
    }

    pub(super) fn tiny_qwen3_model(device: &Device) -> Qwen3Model {
        let cfg = Qwen3Config {
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            num_key_value_heads: 1,
            max_position_embeddings: Some(32),
            head_dim: Some(2),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            vocab_size: 8,
            lm_head_size: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            ada_rms_norm_t_cond: false,
            ada_rms_norm_t_cond_dim: 0,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            norm_topk_prob: None,
        };
        let embeddings = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|idx| ((idx * 5 % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let q_proj = test_projection(4, 4, 1, device);
        let k_proj = test_projection(2, 4, 2, device);
        let v_proj = test_projection(2, 4, 3, device);
        let attention = Qwen3Attention {
            qkv_proj: Qwen3QkvProjection::new_separate(q_proj, k_proj, v_proj),
            o_proj: test_projection(4, 4, 4, device),
            q_norm: None,
            k_norm: None,
            qk_norm_weight: None,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            use_mrope: false,
            mrope_section: None,
            rope_inv_freqs: build_rope_inv_freqs(2, cfg.rope_theta),
            rope_kernel_enabled: false,
        };
        let layer = Qwen3Layer {
            input_layernorm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            self_attn: attention,
            post_attention_layernorm: RmsNorm::new(
                Tensor::ones(4, DType::F32, device).unwrap(),
                1e-5,
            ),
            mlp: Qwen3FeedForward::Dense(Qwen3Mlp {
                gate_up_proj: Qwen3GateUpProjection::new_separate(
                    test_projection(8, 4, 5, device),
                    test_projection(8, 4, 6, device),
                ),
                down_proj: test_projection(4, 8, 7, device),
            }),
        };
        Qwen3Model {
            embed_tokens: Qwen3TokenEmbedding::Dense(Embedding::new(
                Tensor::from_vec(embeddings, (cfg.vocab_size, cfg.hidden_size), device).unwrap(),
                cfg.hidden_size,
            )),
            layers: vec![layer],
            norm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            lm_head: test_projection(cfg.vocab_size, cfg.hidden_size, 8, device),
            device: device.clone(),
            cfg,
            use_mrope: false,
            decode_rope_table: Mutex::new(None),
            decode_graph: Err("not enabled for this model".into()),
        }
    }

    fn bitwise(tensor: &Tensor) -> Vec<u32> {
        tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn equal_axis_positions(seq_len: usize, start_pos: usize, device: &Device) -> Tensor {
        let axis = (0..seq_len).map(|t| (start_pos + t) as i64);
        let data = axis
            .clone()
            .chain(axis.clone())
            .chain(axis)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (3, seq_len), device).unwrap()
    }

    #[test]
    fn sequential_host_rope_tables_match_mrope_with_equal_axes_bit_for_bit() {
        let device = Device::Cpu;
        // Qwen3-ASR text geometry: head_dim 128, theta 1e6, interleaved [24, 20, 20].
        let inv_freqs = build_rope_inv_freqs(128, 1_000_000.0);
        for (seq_len, start_pos) in [(1, 0), (1, 1), (37, 0), (37, 777), (5, 65_000)] {
            let ids = equal_axis_positions(seq_len, start_pos, &device);
            let (mrope_cos, mrope_sin) = build_mrope_cache_with_inv_freqs(
                seq_len,
                128,
                &device,
                DType::F32,
                &ids,
                &[24, 20, 20],
                &inv_freqs,
            )
            .unwrap();
            let (cos, sin) = build_sequential_rope_host_tables(
                seq_len,
                start_pos,
                &device,
                DType::F32,
                &inv_freqs,
            )
            .unwrap();
            assert_eq!(bitwise(&mrope_cos), bitwise(&cos), "cos at {start_pos}");
            assert_eq!(bitwise(&mrope_sin), bitwise(&sin), "sin at {start_pos}");
        }
    }

    /// A two-layer tiny model whose attention uses interleaved MRoPE.
    fn tiny_mrope_qwen3_model(device: &Device) -> Qwen3Model {
        let mut model = tiny_qwen3_model(device);
        model.layers.extend(tiny_qwen3_model(device).layers);
        model.cfg.num_hidden_layers = 2;
        model.use_mrope = true;
        for layer in &mut model.layers {
            layer.self_attn.use_mrope = true;
            layer.self_attn.mrope_section = Some(vec![1, 1, 1]);
        }
        model
    }

    /// The pre-change path: every layer receives the ids and reads them back.
    fn per_layer_mrope_logits(
        model: &Qwen3Model,
        embeds: &Tensor,
        start_pos: usize,
        ids: &Tensor,
    ) -> Tensor {
        let mut hidden = embeds.clone();
        for layer in &model.layers {
            hidden = layer
                .forward_stateless(&hidden, start_pos, Some(ids), &mut None)
                .unwrap();
        }
        model
            .logits_from_hidden(&model.norm.forward(&hidden).unwrap())
            .unwrap()
    }

    #[test]
    fn sequential_mrope_positions_read_back_once_per_forward_and_match_per_layer_path() {
        let device = Device::Cpu;
        let model = tiny_mrope_qwen3_model(&device);
        let input_ids = Tensor::new(&[[1u32, 5, 2, 7, 3]], &device).unwrap();
        let embeds = model.embeddings(&input_ids).unwrap();
        for start_pos in [0usize, 9] {
            let ids = equal_axis_positions(5, start_pos, &device);
            let before = thread_mrope_position_readbacks();
            let expected = per_layer_mrope_logits(&model, &embeds, start_pos, &ids);
            assert_eq!(thread_mrope_position_readbacks() - before, 2);

            let before = thread_mrope_position_readbacks();
            let actual = model
                .forward_stateless_with_embeds(&embeds, start_pos, Some(&ids))
                .unwrap();
            assert_eq!(thread_mrope_position_readbacks() - before, 1);
            assert_eq!(bitwise(&expected), bitwise(&actual));

            // Decode-style calls pass no ids: no readback at all.
            let before = thread_mrope_position_readbacks();
            let decoded = model
                .forward_stateless_with_embeds(&embeds, start_pos, None)
                .unwrap();
            assert_eq!(thread_mrope_position_readbacks() - before, 0);
            assert_eq!(bitwise(&expected), bitwise(&decoded));
        }

        // Ids that differ across axes (vision spans) keep the MRoPE path.
        let data = vec![0i64, 1, 2, 3, 4, 0, 0, 1, 1, 2, 0, 1, 0, 1, 0];
        let ids = Tensor::from_vec(data, (3, 5), &device).unwrap();
        let expected = per_layer_mrope_logits(&model, &embeds, 0, &ids);
        let before = thread_mrope_position_readbacks();
        let actual = model
            .forward_stateless_with_embeds(&embeds, 0, Some(&ids))
            .unwrap();
        // The sequential check, then one table build shared by both layers.
        assert_eq!(thread_mrope_position_readbacks() - before, 2);
        assert_eq!(bitwise(&expected), bitwise(&actual));
    }

    /// QFA-9: a stateless forward builds its RoPE tables once for all
    /// layers, and the result equals building them in every layer.
    #[test]
    fn stateless_forward_builds_rope_tables_once() {
        let device = Device::Cpu;
        let model = tiny_mrope_qwen3_model(&device);
        let embeds = Tensor::from_vec(
            crate::kernels::numeric_check::xorshift_values(23 * 4, 3, 1.0),
            (1, 23, 4),
            &device,
        )
        .unwrap();
        let mixed_axes = (0..69).map(|v| v % 7).collect::<Vec<i64>>();
        for ids in [
            None,
            Some(equal_axis_positions(23, 0, &device)),
            Some(Tensor::from_vec(mixed_axes, (3, 23), &device).unwrap()),
        ] {
            let reference_ids = ids
                .clone()
                .unwrap_or_else(|| equal_axis_positions(23, 0, &device));
            let before = thread_rope_table_builds();
            let expected = per_layer_mrope_logits(&model, &embeds, 0, &reference_ids);
            assert_eq!(thread_rope_table_builds() - before, 2);
            let before = thread_rope_table_builds();
            let actual = model
                .forward_stateless_with_embeds(&embeds, 0, ids.as_ref())
                .unwrap();
            assert_eq!(thread_rope_table_builds() - before, 1);
            assert_eq!(bitwise(&expected), bitwise(&actual));
        }
    }

    /// QFA-9: the portable causal prefill attends by blocks of query rows
    /// and matches one unblocked pass; a prompt longer than a block runs
    /// one block per 256 rows.
    #[test]
    fn query_blocked_causal_attention_matches_one_pass() {
        let device = Device::Cpu;
        let random = |heads: usize, seed: u64| {
            Tensor::from_vec(
                crate::kernels::numeric_check::xorshift_values(heads * 300 * 8, seed, 1.0),
                (1, heads, 300, 8),
                &device,
            )
            .unwrap()
        };
        let (query, key, value) = (random(4, 1), random(2, 2), random(2, 3));
        let scale = 1.0 / 8f64.sqrt();
        let whole = causal_sdpa_by_query_blocks(&query, &key, &value, scale, 300).unwrap();
        let mask = causal_mask(300, 300, 0, &device, DType::F32).unwrap();
        let reference = compact_gqa_sdpa_bhsd(
            &query,
            &key,
            &value,
            Some(CompactGqaMask::Additive(&mask)),
            scale,
        )
        .unwrap();
        assert_eq!(bitwise(&whole), bitwise(&reference));
        for block in [1usize, 7, 64, 299] {
            let before = thread_sdpa_query_blocks();
            let blocked = causal_sdpa_by_query_blocks(&query, &key, &value, scale, block).unwrap();
            assert_eq!(
                thread_sdpa_query_blocks() - before,
                300usize.div_ceil(block) as u64
            );
            let delta = (&blocked - &reference)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(delta <= 1e-6, "block {block} moved attention by {delta}");
        }

        let model = tiny_qwen3_model(&device);
        let embeds = Tensor::from_vec(
            crate::kernels::numeric_check::xorshift_values(300 * 4, 5, 1.0),
            (1, 300, 4),
            &device,
        )
        .unwrap();
        let before = thread_sdpa_query_blocks();
        model
            .forward_stateless_with_embeds(&embeds, 0, None)
            .unwrap();
        assert_eq!(thread_sdpa_query_blocks() - before, 2);
    }

    /// QFA-9: scoring only the requested rows runs the head over those rows
    /// alone and gives their logits and classes; a row past the sequence is
    /// rejected.
    #[test]
    fn row_scored_forward_matches_the_full_logits_at_those_rows() {
        let device = Device::Cpu;
        let model = tiny_mrope_qwen3_model(&device);
        let embeds = Tensor::from_vec(
            crate::kernels::numeric_check::xorshift_values(31 * 4, 11, 1.0),
            (1, 31, 4),
            &device,
        )
        .unwrap();
        let rows = [30usize, 2, 17, 17, 0, 9];
        let full = model
            .forward_stateless_with_embeds(&embeds, 0, None)
            .unwrap();
        let index =
            Tensor::from_vec(rows.map(|row| row as u32).to_vec(), rows.len(), &device).unwrap();
        let expected = full.index_select(&index, 1).unwrap();
        let (tables, head_rows) = (thread_rope_table_builds(), thread_head_rows());
        let scored = model
            .forward_stateless_rows_with_embeds(&embeds, 0, None, &rows)
            .unwrap();
        assert_eq!(thread_rope_table_builds() - tables, 1);
        assert_eq!(thread_head_rows() - head_rows, rows.len() as u64);
        assert_eq!(scored.dims(), [1, rows.len(), 8]);
        let classes = |logits: &Tensor| {
            logits
                .argmax(candle_core::D::Minus1)
                .unwrap()
                .to_vec2::<u32>()
                .unwrap()
        };
        assert_eq!(classes(&expected), classes(&scored));
        let delta = (expected - &scored)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(delta <= 1e-6, "row logits moved by {delta}");
        let error = model
            .forward_stateless_rows_with_embeds(&embeds, 0, None, &[31])
            .unwrap_err();
        assert!(error.to_string().contains("out of range"), "{error}");
    }

    /// Cache-free reference: logits for every position of `ids`, computed by
    /// the production stateless path with a causal mask over the whole span.
    fn stateless_logits(model: &Qwen3Model, ids: &[u32]) -> Tensor {
        let ids = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &model.device).unwrap();
        model
            .forward_stateless_with_embeds(&model.embeddings(&ids).unwrap(), 0, None)
            .unwrap()
    }

    /// Reference logits for the last token of `ids`, shaped like a decode step.
    fn stateless_last_logits(model: &Qwen3Model, ids: &[u32]) -> Tensor {
        stateless_logits(model, ids)
            .narrow(1, ids.len() - 1, 1)
            .unwrap()
    }

    /// The ASR decode graphs never resolve off CUDA, so a CPU or Metal model
    /// keeps the standard decode and its lone-row route; a sparse
    /// feed-forward (host expert routing) never has graphs.
    #[test]
    fn asr_decode_graphs_stay_off_off_cuda_and_for_sparse_trunks() {
        use super::super::decode_graph::QWEN3_ASR_DECODE_GRAPHS_ENV;
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut dense = tiny_qwen3_model(&Device::Cpu);
        assert!(!dense.decode_graph_on());
        assert_eq!(
            dense.decode_graph_diagnostics()["off"],
            "not enabled for this model"
        );
        dense.enable_asr_decode_graph("test", QWEN3_ASR_DECODE_GRAPHS_ENV);
        assert!(!dense.decode_graph_on());
        let reason = dense.decode_graph_diagnostics()["off"].to_string();
        assert!(
            reason.contains("CUDA only") || reason.contains(QWEN3_ASR_DECODE_GRAPHS_ENV),
            "{reason}"
        );
        let mut sparse = tiny_qwen3_moe_model(&Device::Cpu, 2);
        sparse.enable_asr_decode_graph("test", QWEN3_ASR_DECODE_GRAPHS_ENV);
        assert!(sparse.decode_graph_diagnostics()["off"]
            .to_string()
            .contains("sparse"));
    }

    /// Tiny sparse-MoE twin of [`tiny_qwen3_model`]: identical embeddings,
    /// attention, norms, and head, with the FFN replaced by `num_experts`
    /// experts that all share the dense MLP's exact weights. With normalized
    /// top-k routing the expert mixture is a convex combination, so the sparse
    /// model must compute the same function as the dense model.
    pub(super) fn tiny_qwen3_moe_model(device: &Device, num_experts: usize) -> Qwen3Model {
        let mut cfg = Qwen3Config {
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            num_key_value_heads: 1,
            max_position_embeddings: Some(32),
            head_dim: Some(2),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            vocab_size: 8,
            lm_head_size: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            ada_rms_norm_t_cond: false,
            ada_rms_norm_t_cond_dim: 0,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            norm_topk_prob: None,
        };
        cfg.num_experts = Some(num_experts);
        cfg.num_experts_per_tok = Some(2);
        cfg.moe_intermediate_size = Some(8);
        cfg.norm_topk_prob = Some(true);

        let embeddings = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|idx| ((idx * 5 % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let q_proj = test_projection(4, 4, 1, device);
        let k_proj = test_projection(2, 4, 2, device);
        let v_proj = test_projection(2, 4, 3, device);
        let attention = Qwen3Attention {
            qkv_proj: Qwen3QkvProjection::new_separate(q_proj, k_proj, v_proj),
            o_proj: test_projection(4, 4, 4, device),
            q_norm: None,
            k_norm: None,
            qk_norm_weight: None,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            use_mrope: false,
            mrope_section: None,
            rope_inv_freqs: build_rope_inv_freqs(2, cfg.rope_theta),
            rope_kernel_enabled: false,
        };
        // Every expert reuses the dense MLP's projection seeds (5, 6, 7), so
        // all experts compute the identical function.
        let experts = (0..num_experts)
            .map(|_| Qwen3SparseExpert {
                gate_up_proj: Qwen3GateUpProjection::new_separate(
                    test_projection(8, 4, 5, device),
                    test_projection(8, 4, 6, device),
                ),
                down_proj: test_projection(4, 8, 7, device),
            })
            .collect::<Vec<_>>();
        let activation_counters = Arc::new(ExpertActivationCounters::new(num_experts));
        let sparse = Qwen3SparseMlp {
            dispatcher: SparseMoeDispatcher::new(
                cfg.sparse_moe().unwrap().expect("MoE geometry configured"),
            )
            .unwrap()
            .with_counters(activation_counters.clone()),
            router: test_projection(num_experts, 4, 9, device),
            experts,
            activation_counters,
        };
        let layer = Qwen3Layer {
            input_layernorm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            self_attn: attention,
            post_attention_layernorm: RmsNorm::new(
                Tensor::ones(4, DType::F32, device).unwrap(),
                1e-5,
            ),
            mlp: Qwen3FeedForward::Sparse(sparse),
        };
        Qwen3Model {
            embed_tokens: Qwen3TokenEmbedding::Dense(Embedding::new(
                Tensor::from_vec(embeddings, (cfg.vocab_size, cfg.hidden_size), device).unwrap(),
                cfg.hidden_size,
            )),
            layers: vec![layer],
            norm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            lm_head: test_projection(cfg.vocab_size, cfg.hidden_size, 8, device),
            device: device.clone(),
            cfg,
            use_mrope: false,
            decode_rope_table: Mutex::new(None),
            decode_graph: Err("not enabled for this model".into()),
        }
    }

    fn test_managed_arena() -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = CpuKvArena::new(KvArenaConfig {
            id: KvArenaId {
                model_instance: ModelInstanceId::new(93),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            page_tokens: 2,
            capacity_pages: 8,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })
        .unwrap();
        (Arc::new(arena), vec![binding])
    }

    fn test_managed_cache(
        arena: Arc<dyn KvArena>,
        bindings: Vec<KvLayerBinding>,
        first_page: u32,
    ) -> PhysicalPagedKvCache {
        let blocks = (first_page..first_page + 4)
            .map(|index| CacheBlockRef {
                arena: arena.id(),
                group: arena.config().group,
                index,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(arena, bindings, blocks, 0).unwrap()
    }

    /// Prefill that projects only the last row returns exactly the last row of
    /// the full projection and commits the same KV.
    #[test]
    fn last_row_prefill_logits_match_full_projection_last_row() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let mut state = 0x510E_527F_ADE6_82D1u64;
        let values = (0..7 * 4)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect::<Vec<_>>();
        let embeds = Tensor::from_vec(values, (1, 7, 4), &device).unwrap();
        for (start, len) in [(0usize, 7usize), (3, 4), (6, 1)] {
            let (full_arena, full_bindings) = test_managed_arena();
            let (last_arena, last_bindings) = test_managed_arena();
            let mut full = test_managed_cache(full_arena, full_bindings, 0);
            let mut last = test_managed_cache(last_arena, last_bindings, 0);
            if start > 0 {
                let prefix = embeds.narrow(1, 0, start).unwrap();
                model
                    .forward_managed_prefill_only_with_embeds(&prefix, 0, &mut full, None)
                    .unwrap();
                model
                    .forward_managed_prefill_only_with_embeds(&prefix, 0, &mut last, None)
                    .unwrap();
            }
            let span = embeds.narrow(1, start, len).unwrap();
            let full_logits = model
                .forward_managed_with_embeds(&span, start, &mut full, None)
                .unwrap();
            let last_logits = model
                .forward_managed_last_logits_with_embeds(&span, start, &mut last, None)
                .unwrap();
            assert_eq!(last_logits.dims(), &[1, 1, 8]);
            assert_eq!(full.context_len(), last.context_len());
            // A one-row matmul may round differently from the multi-row one in
            // the last ulp (CPU gemm blocking), so compare at 1e-6 relative.
            let expected = full_logits
                .narrow(1, len - 1, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = last_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (e, a) in expected.iter().zip(&actual) {
                let diff = (e - a).abs();
                assert!(
                    diff.is_finite() && diff <= 1e-6 * e.abs().max(1.0),
                    "span {start}+{len}: {e} vs {a}"
                );
            }
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
            };
            assert_eq!(argmax(&expected), argmax(&actual));
        }
    }

    #[test]
    fn managed_qwen3_prefill_and_decode_match_stateless_reference() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let (arena, bindings) = test_managed_arena();
        let mut managed = test_managed_cache(arena.clone(), bindings, 0);
        let prompt = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device).unwrap();

        let managed_prefill = model.forward_managed(&prompt, 0, &mut managed).unwrap();
        assert_tensor_close(&stateless_logits(&model, &[0, 1, 2]), &managed_prefill);
        assert_eq!(managed.context_len(), 3);
        assert_eq!(arena.operation_stats().slot_write_dispatches, 1);
        assert_eq!(arena.operation_stats().paged_prefill_dispatches, 1);
        assert_eq!(arena.operation_stats().paged_decode_dispatches, 0);
        assert_eq!(
            arena.operation_stats().cpu_reference_attention_dispatches,
            1
        );
        let prefill_completions = managed.take_completed_writes();
        assert_eq!(prefill_completions.len(), 1);
        assert_eq!(prefill_completions[0].slots_per_layer(), 3);

        let token = Tensor::from_vec(vec![5u32], (1, 1), &device).unwrap();
        let managed_decode = model.forward_managed(&token, 3, &mut managed).unwrap();
        assert_tensor_close(
            &stateless_last_logits(&model, &[0, 1, 2, 5]),
            &managed_decode,
        );
        assert_eq!(managed.context_len(), 4);
        assert_eq!(arena.operation_stats().slot_write_dispatches, 2);
        assert_eq!(arena.operation_stats().paged_prefill_dispatches, 1);
        assert_eq!(arena.operation_stats().paged_decode_dispatches, 1);
        assert_eq!(
            arena.operation_stats().cpu_reference_attention_dispatches,
            2
        );
        let decode_completions = managed.take_completed_writes();
        assert_eq!(decode_completions.len(), 1);
        assert_eq!(decode_completions[0].slots_per_layer(), 1);
    }

    #[test]
    fn managed_qwen3_sparse_moe_with_identical_experts_matches_dense() {
        let device = Device::Cpu;
        let dense = tiny_qwen3_model(&device);
        let sparse = tiny_qwen3_moe_model(&device, 4);
        let (arena, bindings) = test_managed_arena();
        let mut managed = test_managed_cache(arena, bindings, 0);
        let mut history = vec![0u32, 1, 2, 5];
        let prompt = Tensor::from_vec(history.clone(), (1, 4), &device).unwrap();

        // Prefill: the sparse layer's convex combination of identical experts
        // must reduce to the dense MLP's function.
        let sparse_prefill = sparse
            .forward_managed(&prompt, 0, &mut managed)
            .unwrap();
        assert_tensor_close(&stateless_logits(&dense, &history), &sparse_prefill);
        assert_eq!(managed.context_len(), 4);

        // Decode steps continue to agree on fresh tokens against the dense
        // model's cache-free reference over the same history.
        for token in [3u32, 6] {
            history.push(token);
            let dense_step = stateless_last_logits(&dense, &history);
            let sparse_step = sparse
                .forward_managed(
                    &Tensor::from_vec(vec![token], (1, 1), &device).unwrap(),
                    managed.context_len(),
                    &mut managed,
                )
                .unwrap();
            assert_tensor_close(&dense_step, &sparse_step);
        }
    }

    #[test]
    fn managed_qwen3_moe_records_expert_activation_histograms() {
        let device = Device::Cpu;
        let model = tiny_qwen3_moe_model(&device, 4);
        let counters = model.expert_activation_counters();
        assert_eq!(counters.len(), 1, "one sparse layer exposes one counter set");
        assert_eq!(counters[0].num_experts(), 4);
        assert_eq!(counters[0].total_selections(), 0);

        let (arena, bindings) = test_managed_arena();
        let mut managed = test_managed_cache(arena, bindings, 0);
        // 4 prompt tokens + 2 decode steps, top-2 routing: 12 selections.
        let prompt = Tensor::from_vec(vec![0u32, 1, 2, 5], (1, 4), &device).unwrap();
        model.forward_managed(&prompt, 0, &mut managed).unwrap();
        for token in [3u32, 6] {
            model
                .forward_managed(
                    &Tensor::from_vec(vec![token], (1, 1), &device).unwrap(),
                    managed.context_len(),
                    &mut managed,
                )
                .unwrap();
        }
        assert_eq!(
            counters[0].total_selections(),
            12,
            "every routed token selects exactly experts_per_tok entries"
        );
        let snapshot = counters[0].snapshot();
        assert_eq!(snapshot.iter().sum::<u64>(), 12);
    }

    #[test]
    fn qwen3_moe_config_resolves_sparse_geometry_fail_closed() {
        let moe_json = serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "vocab_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "norm_topk_prob": true,
        });
        let config: Qwen3Config = serde_json::from_value(moe_json).unwrap();
        let moe = config.sparse_moe().unwrap().expect("MoE geometry resolves");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.num_experts_per_tok, 2);
        assert!(moe.norm_topk_prob);

        // Dense defaults: no expert fields, no sparse geometry.
        let dense_json = serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "vocab_size": 32,
        });
        let config: Qwen3Config = serde_json::from_value(dense_json).unwrap();
        assert!(config.sparse_moe().unwrap().is_none());

        // num_experts without its companion geometry fails closed.
        let incomplete_json = serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "vocab_size": 32,
            "num_experts": 8,
        });
        let config: Qwen3Config = serde_json::from_value(incomplete_json).unwrap();
        let error = config.sparse_moe().unwrap_err();
        assert!(format!("{error}").contains("num_experts_per_tok"));
    }

    #[test]
    fn managed_qwen3_chunked_prefill_matches_monolithic_final_logits() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let (full_arena, full_bindings) = test_managed_arena();
        let (chunk_arena, chunk_bindings) = test_managed_arena();
        let mut full_cache = test_managed_cache(full_arena, full_bindings, 0);
        let mut chunk_cache = test_managed_cache(chunk_arena, chunk_bindings, 0);
        let full_prompt = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let first_chunk = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        let final_chunk = Tensor::from_vec(vec![2u32, 3], (1, 2), &device).unwrap();

        let full = model
            .forward_managed(&full_prompt, 0, &mut full_cache)
            .unwrap();
        model
            .forward_managed(&first_chunk, 0, &mut chunk_cache)
            .unwrap();
        let chunked = model
            .forward_managed(&final_chunk, 2, &mut chunk_cache)
            .unwrap();

        assert_tensor_close(
            &full.narrow(1, 3, 1).unwrap(),
            &chunked.narrow(1, 1, 1).unwrap(),
        );
        assert_eq!(full_cache.context_len(), 4);
        assert_eq!(chunk_cache.context_len(), 4);
        assert_eq!(full_cache.take_completed_writes().len(), 1);
        assert_eq!(chunk_cache.take_completed_writes().len(), 2);
    }

    #[test]
    fn managed_qwen3_caller_embeddings_match_token_prefill() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let (token_arena, token_bindings) = test_managed_arena();
        let (embed_arena, embed_bindings) = test_managed_arena();
        let mut token_cache = test_managed_cache(token_arena, token_bindings, 0);
        let mut embed_cache = test_managed_cache(embed_arena, embed_bindings, 0);
        let prompt = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device).unwrap();
        let embeds = model.embeddings(&prompt).unwrap();

        let token_logits = model.forward_managed(&prompt, 0, &mut token_cache).unwrap();
        let embed_logits = model
            .forward_managed_with_embeds(&embeds, 0, &mut embed_cache, None)
            .unwrap();

        assert_tensor_close(&token_logits, &embed_logits);
        assert_eq!(token_cache.context_len(), 3);
        assert_eq!(embed_cache.context_len(), 3);
    }

    #[test]
    fn managed_qwen3_decode_batches_ragged_rows_in_one_arena() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let (arena, bindings) = test_managed_arena();
        let mut managed_a = test_managed_cache(arena.clone(), bindings.clone(), 0);
        let mut managed_b = test_managed_cache(arena.clone(), bindings, 4);
        let prompt_a = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        let prompt_b = Tensor::from_vec(vec![2u32, 3, 4], (1, 3), &device).unwrap();
        model.forward_managed(&prompt_a, 0, &mut managed_a).unwrap();
        model.forward_managed(&prompt_b, 0, &mut managed_b).unwrap();
        assert_eq!(managed_a.take_completed_writes().len(), 1);
        assert_eq!(managed_b.take_completed_writes().len(), 1);

        let scalar_a = stateless_last_logits(&model, &[0, 1, 5]);
        let scalar_b = stateless_last_logits(&model, &[2, 3, 4, 6]);
        let mut caches = [&mut managed_a, &mut managed_b];
        let batched = model
            .forward_managed_decode_batch(
                &Tensor::from_vec(vec![5u32, 6], (2, 1), &device).unwrap(),
                &[2, 3],
                &mut caches,
            )
            .unwrap();

        assert_tensor_close(&scalar_a, &batched.i(0).unwrap().unsqueeze(0).unwrap());
        assert_tensor_close(&scalar_b, &batched.i(1).unwrap().unsqueeze(0).unwrap());
        assert_eq!(managed_a.context_len(), 3);
        assert_eq!(managed_b.context_len(), 4);
        let completion_a = managed_a.take_completed_writes();
        let completion_b = managed_b.take_completed_writes();
        assert_eq!(completion_a.len(), 1);
        assert_eq!(completion_b.len(), 1);
        assert!(Arc::ptr_eq(&completion_a[0], &completion_b[0]));
        assert_eq!(completion_a[0].slots_per_layer(), 2);
        // Two prompt prefill calls plus one batched decode attention call per
        // layer, not one arena operation per ragged decode row.
        assert_eq!(arena.operation_stats().slot_write_dispatches, 3);
        assert_eq!(arena.operation_stats().paged_prefill_dispatches, 2);
        assert_eq!(arena.operation_stats().paged_decode_dispatches, 1);
    }

    #[derive(Clone, Copy, Debug)]
    enum RopeTestMode {
        Plain,
        /// Qwen3-ASR: three MRoPE sections, sequential positions on host tables.
        Mrope,
        /// MRoPE with fewer than three sections: tables from position zero.
        ShortMrope,
    }

    /// The tiny model's attention with 3 query heads, 2 KV heads and a
    /// head_dim of 8 (four frequencies), for RoPE-only tests.
    fn rope_test_attention(mode: RopeTestMode, kernel: bool) -> Qwen3Attention {
        let mut attention = tiny_qwen3_model(&Device::Cpu).layers.remove(0).self_attn;
        attention.num_heads = 3;
        attention.num_kv_heads = 2;
        attention.head_dim = 8;
        attention.rope_inv_freqs = build_rope_inv_freqs(8, 1_000_000.0);
        attention.rope_kernel_enabled = kernel;
        (attention.use_mrope, attention.mrope_section) = match mode {
            RopeTestMode::Plain => (false, None),
            RopeTestMode::Mrope => (true, Some(vec![24, 20, 20])),
            RopeTestMode::ShortMrope => (true, Some(vec![2])),
        };
        attention
    }

    fn f32_bits(tensor: &Tensor) -> Vec<u32> {
        bitwise(&tensor.to_dtype(DType::F32).unwrap())
    }

    fn decode_rope_table_rows(table: &Mutex<Option<Qwen3DecodeRopeTable>>) -> usize {
        table.lock().unwrap().as_ref().map_or(0, |table| table.rows)
    }

    /// Row `p` of the decode table is exactly the table the one-token path
    /// builds at position `p`, in every RoPE mode and dtype.
    #[test]
    fn decode_rope_table_rows_match_per_token_tables_bit_for_bit() {
        let device = Device::Cpu;
        for mode in [
            RopeTestMode::Plain,
            RopeTestMode::Mrope,
            RopeTestMode::ShortMrope,
        ] {
            let attention = rope_test_attention(mode, false);
            for dtype in [DType::F32, DType::BF16] {
                let probe = Tensor::zeros((1, 1, 3, 8), dtype, &device).unwrap();
                let (cos, sin) = attention.decode_rope_table(300, &device, dtype).unwrap();
                for pos in [0, 1, 2, 17, 255, 299] {
                    let (expected_cos, expected_sin) =
                        attention.rope_tables(1, pos, None, &probe).unwrap();
                    let row = attention.decode_rope_position(pos);
                    assert_eq!(
                        f32_bits(&cos.narrow(0, row, 1).unwrap()),
                        f32_bits(&expected_cos),
                        "{mode:?} {dtype:?} cos at {pos}"
                    );
                    assert_eq!(
                        f32_bits(&sin.narrow(0, row, 1).unwrap()),
                        f32_bits(&expected_sin),
                        "{mode:?} {dtype:?} sin at {pos}"
                    );
                }
            }
        }
    }

    /// One RoPE call over a ragged decode batch equals rotating each row on
    /// its own (the pre-batching loop: `apply_rope_pair` per row, then a
    /// stack), bit for bit, through the fused kernel (Metal and CUDA) and
    /// the composed ops (CPU), including a step that grows the table.
    ///
    /// The composed ops run only at F32, the CPU activation dtype: Candle's
    /// CPU BF16 binary ops round a NEON vector body and a scalar tail
    /// differently, so their bits depend on the tensor's length.
    #[test]
    fn batched_decode_rope_matches_per_row_rope_bit_for_bit() {
        let device = Device::Cpu;
        let positions = [0usize, 9, 3, 4_100, 9];
        let rows = positions.len();
        for mode in [
            RopeTestMode::Plain,
            RopeTestMode::Mrope,
            RopeTestMode::ShortMrope,
        ] {
            for (kernel, dtype) in [
                (false, DType::F32),
                (true, DType::F32),
                (true, DType::BF16),
                (true, DType::F16),
            ] {
                let attention = rope_test_attention(mode, kernel);
                let values =
                    |count, seed| crate::kernels::numeric_check::xorshift_values(count, seed, 1.0);
                let q = Tensor::from_vec(values(rows * 3 * 8, 11), (rows, 1, 3, 8), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let k = Tensor::from_vec(values(rows * 2 * 8, 12), (rows, 1, 2, 8), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let mut expected_q = Vec::new();
                let mut expected_k = Vec::new();
                for (row, &pos) in positions.iter().enumerate() {
                    let (q_row, k_row) = attention
                        .apply_rope_pair(
                            q.i(row).unwrap().unsqueeze(0).unwrap(),
                            k.i(row).unwrap().unsqueeze(0).unwrap(),
                            pos,
                            None,
                        )
                        .unwrap();
                    expected_q.push(q_row.reshape((3, 8)).unwrap());
                    expected_k.push(k_row.reshape((2, 8)).unwrap());
                }
                let expected_q = Tensor::stack(&expected_q, 0).unwrap();
                let expected_k = Tensor::stack(&expected_k, 0).unwrap();

                let table = Mutex::new(None);
                attention
                    .decode_rope_rows(&table, &[1], &device, dtype)
                    .unwrap();
                assert_eq!(decode_rope_table_rows(&table), 4_096);
                let rope = attention
                    .decode_rope_rows(&table, &positions, &device, dtype)
                    .unwrap();
                let grown = match mode {
                    RopeTestMode::ShortMrope => 4_096,
                    RopeTestMode::Plain | RopeTestMode::Mrope => 8_192,
                };
                assert_eq!(decode_rope_table_rows(&table), grown);
                let (actual_q, actual_k) = attention
                    .apply_decode_rope(q.clone(), k.clone(), &rope)
                    .unwrap();
                let label = format!("{mode:?} kernel={kernel} {dtype:?}");
                assert_eq!(f32_bits(&expected_q), f32_bits(&actual_q), "q {label}");
                assert_eq!(f32_bits(&expected_k), f32_bits(&actual_k), "k {label}");
            }
        }
    }

    /// A batched decode step gathers its RoPE rows from the model's table
    /// (no per-row table builds; the table is built once and reused), and
    /// each row matches that row decoded alone, with plain RoPE and with
    /// Qwen3-ASR's MRoPE. Within a tolerance: Candle's CPU GEMM rounds a
    /// one-row product differently from a two-row one.
    #[test]
    fn managed_decode_batch_gathers_rope_once_and_matches_single_rows() {
        let device = Device::Cpu;
        for mrope in [false, true] {
            let mut model = tiny_qwen3_model(&device);
            if mrope {
                model.use_mrope = true;
                for layer in &mut model.layers {
                    layer.self_attn.use_mrope = true;
                    layer.self_attn.mrope_section = Some(vec![1, 1, 1]);
                }
            }
            let prefilled = || {
                let (arena, bindings) = test_managed_arena();
                let mut a = test_managed_cache(arena.clone(), bindings.clone(), 0);
                let mut b = test_managed_cache(arena, bindings, 4);
                for (cache, prompt) in [(&mut a, vec![0u32, 1]), (&mut b, vec![2u32, 3, 4])] {
                    let len = prompt.len();
                    let prompt = Tensor::from_vec(prompt, (1, len), &device).unwrap();
                    model.forward_managed(&prompt, 0, cache).unwrap();
                }
                (a, b)
            };
            let (mut solo_a, mut solo_b) = prefilled();
            let (mut batch_a, mut batch_b) = prefilled();
            let mut solo = Vec::new();
            for (cache, token, pos) in [(&mut solo_a, 5u32, 2usize), (&mut solo_b, 6, 3)] {
                let token = Tensor::from_vec(vec![token], (1, 1), &device).unwrap();
                solo.push(
                    model
                        .forward_managed_decode_batch(&token, &[pos], &mut [cache])
                        .unwrap(),
                );
            }
            let table_id = |model: &Qwen3Model| {
                let table = model.decode_rope_table.lock().unwrap();
                table.as_ref().unwrap().cos.id()
            };
            let table = table_id(&model);
            let builds = thread_rope_table_builds();
            let batched = model
                .forward_managed_decode_batch(
                    &Tensor::from_vec(vec![5u32, 6], (2, 1), &device).unwrap(),
                    &[2, 3],
                    &mut [&mut batch_a, &mut batch_b],
                )
                .unwrap();
            assert_eq!(thread_rope_table_builds(), builds, "mrope={mrope}");
            assert_eq!(table_id(&model), table, "mrope={mrope}");
            for (row, solo) in solo.iter().enumerate() {
                assert_tensor_close(solo, &batched.narrow(0, row, 1).unwrap());
            }
        }
    }

    #[test]
    fn managed_hidden_forward_matches_stateless_and_exposes_early_release_receipt() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let input_ids = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device).unwrap();
        let embeds = model.embeddings(&input_ids).unwrap();

        let (arena, bindings) = test_managed_arena();
        let mut managed = test_managed_cache(arena, bindings, 0);
        let managed_hidden = model
            .forward_managed_hidden_with_embeds(&embeds, 0, &mut managed, None)
            .unwrap();

        // The managed path returns final-norm hidden states; projecting them
        // must reproduce the cache-free reference logits.
        assert_tensor_close(
            &stateless_logits(&model, &[0, 1, 2]),
            &model.logits_from_hidden(&managed_hidden).unwrap(),
        );
        assert_eq!(managed.context_len(), 3);
        let completions = managed.take_completed_writes();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].slots_per_layer(), 3);
        assert!(managed.take_completed_writes().is_empty());
    }

    #[test]
    fn managed_full_page_prefix_reuse_matches_full_prefill_logits() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let full_logits = stateless_logits(&model, &[0, 1, 2, 3, 4]);

        let (arena, bindings) = test_managed_arena();
        let mut publisher = test_managed_cache(arena.clone(), bindings.clone(), 0);
        let prefix = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        model.forward_managed(&prefix, 0, &mut publisher).unwrap();
        assert_eq!(publisher.context_len(), 4);

        let mut reused = PhysicalPagedKvCache::new(
            arena,
            bindings,
            publisher.blocks.clone(),
            publisher.context_len(),
        )
        .unwrap();
        let suffix = Tensor::from_vec(vec![4u32], (1, 1), &device).unwrap();
        let reused_logits = model.forward_managed(&suffix, 4, &mut reused).unwrap();
        let expected_last = full_logits.i((.., 4..5, ..)).unwrap();
        assert_tensor_close(&expected_last, &reused_logits);
    }

    #[test]
    fn managed_multi_token_prefix_extension_matches_full_prefill_logits() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model(&device);
        let full_logits = stateless_logits(&model, &[0, 1, 2, 3, 4]);

        let (arena, bindings) = test_managed_arena();
        let mut managed = test_managed_cache(arena, bindings, 0);
        let prefix = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        model.forward_managed(&prefix, 0, &mut managed).unwrap();
        let suffix = Tensor::from_vec(vec![2u32, 3, 4], (1, 3), &device).unwrap();
        let suffix_logits = model.forward_managed(&suffix, 2, &mut managed).unwrap();

        assert_tensor_close(&full_logits.i((.., 2.., ..)).unwrap(), &suffix_logits);
        assert_eq!(managed.context_len(), 5);
    }

    #[test]
    fn qwen3_dense_projection_loads_optional_bias() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).expect("weight"),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.5f32, -1.0], (2,), &device).expect("bias"),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let projection = Qwen3Projection::dense(2, 2, vb).expect("projection");
        assert_eq!(
            projection.diagnostics(),
            Qwen3ProjectionDiagnostics {
                dense_projection_count: 1,
                dense_bias_projection_count: 1,
                quantized_projection_count: 0,
            }
        );

        let input = Tensor::from_vec(vec![2.0f32, 3.0], (1, 2), &device).expect("input");
        let out = projection
            .forward(&input)
            .expect("projection forward")
            .to_vec2::<f32>()
            .expect("output values");
        assert_eq!(out, vec![vec![8.5, 17.0]]);
    }

    #[test]
    fn selected_dense_projection_logits_match_full_projection() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.5, -1.0, 0.25, 2.0, -0.5, 1.5, 0.75, -0.25, 1.25, 0.5,
            ],
            (4, 3),
            &device,
        )
        .expect("weight");
        let bias = Tensor::from_vec(vec![0.1f32, -0.2, 0.3, 0.4], (4,), &device).expect("bias");
        let projection = Qwen3Projection::Dense(Qwen3DenseProjection {
            linear: Linear::new(weight, Some(bias)),
            has_bias: true,
        });
        let hidden = Tensor::from_vec(vec![0.5f32, -1.0, 2.0], (1, 3), &device).expect("hidden");

        let selected = projection
            .logits_for_tokens(&hidden, &[3, 1])
            .expect("selected logits");
        let full = projection
            .forward(&hidden)
            .expect("full logits")
            .i(0)
            .expect("row")
            .to_vec1::<f32>()
            .expect("values");

        assert_eq!(selected.len(), 2);
        assert!((selected[0].1 - full[3]).abs() < 1e-6);
        assert!((selected[1].1 - full[1]).abs() < 1e-6);
    }

    #[test]
    fn qk_norm_weight_concatenates_query_and_key_weights() {
        let device = Device::Cpu;
        let q_norm = RmsNorm::new(
            Tensor::from_vec(vec![1.0f32, 1.1, 1.2], 3, &device).expect("q weight"),
            1e-6,
        );
        let k_norm = RmsNorm::new(
            Tensor::from_vec(vec![0.7f32, 0.8, 0.9], 3, &device).expect("k weight"),
            1e-6,
        );

        let weight = qk_norm_weight(&Some(q_norm), &Some(k_norm))
            .expect("qk weight")
            .expect("qk weight present")
            .to_vec1::<f32>()
            .expect("values");

        assert_eq!(weight, vec![1.0, 1.1, 1.2, 0.7, 0.8, 0.9]);
    }

    #[test]
    fn qwen3_dense_projection_keeps_no_bias_path() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).expect("weight"),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let projection = Qwen3Projection::dense(2, 2, vb).expect("projection");
        assert_eq!(
            projection.diagnostics(),
            Qwen3ProjectionDiagnostics {
                dense_projection_count: 1,
                dense_bias_projection_count: 0,
                quantized_projection_count: 0,
            }
        );

        let input = Tensor::from_vec(vec![2.0f32, 3.0], (1, 2), &device).expect("input");
        let out = projection
            .forward(&input)
            .expect("projection forward")
            .to_vec2::<f32>()
            .expect("output values");
        assert_eq!(out, vec![vec![8.0, 18.0]]);
    }

    #[test]
    fn fused_qwen3_qkv_projection_matches_separate_dense_projections() {
        let device = Device::Cpu;
        let q_proj = Qwen3DenseProjection {
            linear: Linear::new(
                Tensor::from_vec(
                    vec![
                        0.1f32, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2,
                    ],
                    (4, 3),
                    &device,
                )
                .expect("q weight"),
                None,
            ),
            has_bias: false,
        };
        let k_proj = Qwen3DenseProjection {
            linear: Linear::new(
                Tensor::from_vec(vec![0.3f32, -0.2, 0.1, 0.6, 0.4, -0.5], (2, 3), &device)
                    .expect("k weight"),
                None,
            ),
            has_bias: false,
        };
        let v_proj = Qwen3DenseProjection {
            linear: Linear::new(
                Tensor::from_vec(vec![-0.7f32, 0.8, 0.9, 0.2, -0.1, 0.4], (2, 3), &device)
                    .expect("v weight"),
                None,
            ),
            has_bias: false,
        };
        let fused = Qwen3FusedQkvProjection::new(&q_proj, &k_proj, &v_proj).expect("fused qkv");
        let input = Tensor::from_vec(
            vec![1.0f32, -2.0, 0.5, 0.25, 1.5, -0.75],
            (1, 2, 3),
            &device,
        )
        .expect("input");

        let (q, k, v) = fused.forward(&input).expect("fused forward");

        assert_tensor_close(&q, &q_proj.linear.forward(&input).expect("q"));
        assert_tensor_close(&k, &k_proj.linear.forward(&input).expect("k"));
        assert_tensor_close(&v, &v_proj.linear.forward(&input).expect("v"));
    }

    #[test]
    fn fused_qwen3_gate_up_projection_matches_separate_dense_projections() {
        let device = Device::Cpu;
        let gate_proj = Qwen3DenseProjection {
            linear: Linear::new(
                Tensor::from_vec(
                    vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9],
                    (3, 3),
                    &device,
                )
                .expect("gate weight"),
                Some(Tensor::from_vec(vec![0.1f32, -0.2, 0.3], (3,), &device).expect("gate bias")),
            ),
            has_bias: true,
        };
        let up_proj = Qwen3DenseProjection {
            linear: Linear::new(
                Tensor::from_vec(
                    vec![-0.3f32, 0.2, 0.1, 0.6, -0.4, 0.5, 0.7, 0.9, -0.8],
                    (3, 3),
                    &device,
                )
                .expect("up weight"),
                Some(Tensor::from_vec(vec![0.2f32, 0.4, -0.6], (3,), &device).expect("up bias")),
            ),
            has_bias: true,
        };
        let fused = Qwen3FusedGateUpProjection::new(&gate_proj, &up_proj).expect("fused gate/up");
        let input = Tensor::from_vec(
            vec![1.0f32, -2.0, 0.5, 0.25, 1.5, -0.75],
            (1, 2, 3),
            &device,
        )
        .expect("input");

        let (gate, up) = fused.forward(&input).expect("fused forward");

        assert_tensor_close(&gate, &gate_proj.linear.forward(&input).expect("gate"));
        assert_tensor_close(&up, &up_proj.linear.forward(&input).expect("up"));
    }

    #[test]
    fn qwen3_dense_projection_fusion_defaults_to_metal_only() {
        assert!(qwen3_dense_projection_fusion_policy(true, None));
        assert!(!qwen3_dense_projection_fusion_policy(false, None));
        assert!(qwen3_dense_projection_fusion_policy(false, Some(true)));
        assert!(!qwen3_dense_projection_fusion_policy(true, Some(false)));
    }

    #[test]
    fn qwen3_qk_rms_norm_fusion_defaults_to_accelerators() {
        assert!(qwen3_qk_rms_norm_fusion_policy(true, false, None));
        assert!(qwen3_qk_rms_norm_fusion_policy(false, true, None));
        assert!(!qwen3_qk_rms_norm_fusion_policy(false, false, None));
        assert!(qwen3_qk_rms_norm_fusion_policy(false, false, Some(true)));
        assert!(!qwen3_qk_rms_norm_fusion_policy(true, false, Some(false)));
    }

    #[test]
    fn repeat_kv_returns_rank4_heads_last() {
        let device = Device::Cpu;
        let input = Tensor::zeros((1, 63, 16, 128), DType::F32, &device).expect("tensor");
        let out = repeat_kv(&input, 32, 16).expect("repeat_kv");
        assert_eq!(out.rank(), 4);
        assert_eq!(out.dims4().expect("dims4"), (1, 63, 32, 128));
    }

    #[test]
    fn qwen3_manual_rotary_matches_rope_thd() {
        let device = Device::Cpu;
        let seq_len = 4usize;
        let head_dim = 8usize;
        let x = Tensor::from_vec(
            (0..(seq_len * 2 * head_dim))
                .map(|v| (v as f32) * 0.01)
                .collect::<Vec<_>>(),
            (1, seq_len, 2, head_dim),
            &device,
        )
        .expect("x");

        let (cos_half, sin_half) =
            build_rope_cache(seq_len, head_dim, 0, 10000.0, &device, DType::F32).expect("cache");
        let kernel = rotary_emb::rope_thd(
            &x,
            &cos_half.unsqueeze(0).expect("cos"),
            &sin_half.unsqueeze(0).expect("sin"),
        )
        .expect("kernel");

        let half_dim = head_dim / 2;
        let cos = Tensor::cat(&[cos_half.clone(), cos_half], 1)
            .expect("cat cos")
            .unsqueeze(0)
            .expect("unsqueeze cos")
            .unsqueeze(2)
            .expect("unsqueeze cos head");
        let sin = Tensor::cat(&[sin_half.clone(), sin_half], 1)
            .expect("cat sin")
            .unsqueeze(0)
            .expect("unsqueeze sin")
            .unsqueeze(2)
            .expect("unsqueeze sin head");

        let x1 = x.narrow(3, 0, half_dim).expect("x1");
        let x2 = x.narrow(3, half_dim, half_dim).expect("x2");
        let minus_one = Tensor::from_vec(vec![-1.0f32], (1,), &device).expect("minus one");
        let neg_x2 = x2.broadcast_mul(&minus_one).expect("neg");
        let rotated = Tensor::cat(&[neg_x2, x1], 3).expect("rotated");
        let manual = x
            .broadcast_mul(&cos)
            .expect("mul")
            .broadcast_add(&rotated.broadcast_mul(&sin).expect("rot mul"))
            .expect("add");

        let kernel_vals = kernel
            .flatten_all()
            .expect("flatten kernel")
            .to_vec1::<f32>()
            .expect("kernel vals");
        let manual_vals = manual
            .flatten_all()
            .expect("flatten manual")
            .to_vec1::<f32>()
            .expect("manual vals");
        assert_eq!(kernel_vals.len(), manual_vals.len());
        for (lhs, rhs) in kernel_vals.iter().zip(manual_vals.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn qwen3_cuda_rope_kernel_defaults_on_with_explicit_rollback() {
        assert!(qwen3_rope_kernel_policy(true, false, None));
        assert!(qwen3_rope_kernel_policy(false, true, None));
        assert!(qwen3_rope_kernel_policy(false, true, Some(true)));
        assert!(!qwen3_rope_kernel_policy(true, false, Some(false)));
        assert!(!qwen3_rope_kernel_policy(false, false, Some(true)));
    }

    #[test]
    fn qwen3_apply_rotary_emb_accepts_half_dim_cos_sin() {
        let device = Device::Cpu;
        let seq_len = 3usize;
        let head_dim = 8usize;
        let x = Tensor::from_vec(
            (0..(seq_len * 2 * head_dim))
                .map(|v| (v as f32) * 0.03 - 0.2)
                .collect::<Vec<_>>(),
            (1, seq_len, 2, head_dim),
            &device,
        )
        .expect("x");
        let (cos_half, sin_half) =
            build_rope_cache(seq_len, head_dim, 0, 10000.0, &device, DType::F32).expect("cache");

        let rotated = apply_rotary_emb(&x, &cos_half, &sin_half).expect("rotary");
        assert_eq!(rotated.dims4().expect("dims4"), (1, seq_len, 2, head_dim));
    }

    #[test]
    fn qwen3_qmatmul_uses_f32_input_for_accelerated_lower_precision_activations() {
        // CUDA weights without half-activation kernels, and Metal: cast.
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(true, false, false, DType::BF16),
            Some(DType::F32)
        );
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(true, false, false, DType::F16),
            Some(DType::F32)
        );
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(true, false, false, DType::F32),
            None
        );
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(false, false, true, DType::F16),
            Some(DType::F32)
        );
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(false, false, false, DType::BF16),
            None
        );
    }

    #[test]
    fn qwen3_cuda_qmatmul_passes_half_activations_only_to_weights_that_take_them() {
        for dtype in [DType::BF16, DType::F16] {
            assert_eq!(
                qwen3_qmatmul_input_dtype_policy(true, true, false, dtype),
                None
            );
        }
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(true, true, false, DType::F32),
            None
        );
        // Any other activation dtype still reaches the F32 kernels.
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(true, true, false, DType::F64),
            Some(DType::F32)
        );
        // The flag means nothing off CUDA: Metal keeps its own policy.
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(false, true, true, DType::BF16),
            Some(DType::F32)
        );
        assert_eq!(
            qwen3_qmatmul_input_dtype_policy(false, true, false, DType::BF16),
            None
        );
    }

    #[test]
    fn qwen3_cuda_half_activation_weights_are_fast_kernel_qtensors() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            crate::kernels::numeric_check::xorshift_values(4 * 256, 9, 1.0),
            (4, 256),
            &device,
        )
        .unwrap();
        let qmatmul =
            |dtype| QMatMul::QTensor(Arc::new(QTensor::quantize(&weight, dtype).unwrap()));
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            assert!(
                qwen3_cuda_qmatmul_takes_half_activations(&qmatmul(dtype)),
                "{dtype:?}"
            );
        }
        // Types with no fast kernel fall back to F32-only kernels.
        for dtype in [GgmlDType::Q8_1, GgmlDType::Q8K] {
            assert!(
                !qwen3_cuda_qmatmul_takes_half_activations(&qmatmul(dtype)),
                "{dtype:?}"
            );
        }
        // A dense (dequantized F16/BF16/F32 GGUF) weight is an F32 matmul.
        let dense =
            QMatMul::from_qtensor(QTensor::quantize(&weight, GgmlDType::F16).unwrap()).unwrap();
        assert!(matches!(dense, QMatMul::Tensor(_)));
        assert!(!qwen3_cuda_qmatmul_takes_half_activations(&dense));
        assert!(!qwen3_cuda_qmatmul_takes_half_activations(
            &QMatMul::TensorF16(weight.to_dtype(DType::F16).unwrap())
        ));
        // Decided at load, and only for a CUDA weight.
        let Qwen3Projection::Quantized(projection) =
            Qwen3Projection::from_qmatmul(qmatmul(GgmlDType::Q8_0))
        else {
            panic!("a QMatMul loads as a quantized projection");
        };
        assert!(!projection.cuda_half_activations);
    }

    #[test]
    fn qwen3_metal_qmatmul_f32_input_defaults_on_for_metal_only() {
        assert!(qwen3_metal_qmatmul_f32_input_policy(true, None));
        assert!(!qwen3_metal_qmatmul_f32_input_policy(false, None));
        assert!(!qwen3_metal_qmatmul_f32_input_policy(false, Some(true)));
        assert!(!qwen3_metal_qmatmul_f32_input_policy(true, Some(false)));
    }

    #[test]
    fn qwen3_quantized_projection_grouping_defaults_to_small_metal_models() {
        assert!(qwen3_quantized_projection_grouping_policy(true, 1024, None));
        assert!(!qwen3_quantized_projection_grouping_policy(
            true, 2048, None
        ));
        assert!(!qwen3_quantized_projection_grouping_policy(
            false, 1024, None
        ));
        assert!(qwen3_quantized_projection_grouping_policy(
            true,
            2048,
            Some(true)
        ));
        assert!(!qwen3_quantized_projection_grouping_policy(
            false,
            1024,
            Some(true)
        ));
        assert!(!qwen3_quantized_projection_grouping_policy(
            true,
            1024,
            Some(false)
        ));
    }

    #[test]
    fn packed_qmatmul_matches_separate_q8_0_on_cpu() {
        use candle_core::quantized::GgmlDType;

        let device = Device::Cpu;
        let w1_data = (0..64)
            .map(|idx| (idx as f32 - 31.0) / 8.0)
            .collect::<Vec<_>>();
        let w2_data = (0..32)
            .map(|idx| (15.0 - idx as f32) / 7.0)
            .collect::<Vec<_>>();
        let w1 = Tensor::from_vec(w1_data, (2, 32), &device).unwrap();
        let w2 = Tensor::from_vec(w2_data, (1, 32), &device).unwrap();
        let q1 = QTensor::quantize(&w1, GgmlDType::Q8_0).unwrap();
        let q2 = QTensor::quantize(&w2, GgmlDType::Q8_0).unwrap();

        let packed = packed_qmatmul_from_qtensors(&[q1, q2], &device).unwrap();
        let x = Tensor::from_vec(
            (0..32)
                .map(|idx| (idx as f32 + 1.0) / 16.0)
                .collect::<Vec<_>>(),
            (1, 32),
            &device,
        )
        .unwrap();

        let packed_out = packed.forward(&x).unwrap();
        let expected = Tensor::cat(
            &[
                &QMatMul::from_qtensor(QTensor::quantize(&w1, GgmlDType::Q8_0).unwrap())
                    .unwrap()
                    .forward(&x)
                    .unwrap(),
                &QMatMul::from_qtensor(QTensor::quantize(&w2, GgmlDType::Q8_0).unwrap())
                    .unwrap()
                    .forward(&x)
                    .unwrap(),
            ],
            1,
        )
        .unwrap();

        assert_tensor_close(&packed_out, &expected);
    }
}
