//! Engine configuration types.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::scheduler::SchedulingPolicy;
use crate::backends::{BackendKind, BackendRouter, BackendSelectionSource};
use crate::config::{
    resolve_kv_cache_policy, BatchSizePreference, PhysicalExecutionCapacity, PhysicalExecutionMode,
    PhysicalInFlightLimit, ResolvedKvCachePolicy, DEFAULT_MAX_PREFIX_CACHE_PAGES,
};
use crate::engine::ResourceAmount;
use crate::model::ModelVariant;
use crate::Result;

/// Largest native text context in the currently supported CUDA model catalog
/// (Qwen3.5). Longer rope-scaled modes are intentionally excluded until their
/// scaling semantics are implemented by the corresponding adapters.
pub(crate) const CUDA_MAX_NATIVE_CONTEXT_TOKENS: usize = 262_144;

pub(crate) fn resolve_backend_model_context(
    backend: BackendKind,
    configured_max_seq_len: usize,
    loaded_model_max: usize,
) -> Result<usize> {
    if loaded_model_max == 0 {
        return Err(crate::Error::ModelLoadError(
            "loaded model reported a zero context length".into(),
        ));
    }
    let configured = configured_max_seq_len.max(1);
    Ok(if backend == BackendKind::Cuda {
        loaded_model_max.min(CUDA_MAX_NATIVE_CONTEXT_TOKENS)
    } else {
        configured.min(loaded_model_max)
    })
}

pub(crate) fn tts_explicit_output_limit(
    backend: BackendKind,
    variant: ModelVariant,
    configured_max_seq_len: usize,
) -> usize {
    let configured = configured_max_seq_len.max(1);
    if backend == BackendKind::Cuda {
        return match variant.family() {
            crate::catalog::ModelFamily::Lfm25Audio => {
                ModelVariant::LFM25_AUDIO_NATIVE_CONTEXT_TOKENS
            }
            crate::catalog::ModelFamily::FishS2Tts => {
                ModelVariant::FISH_S2_PRO_NATIVE_CONTEXT_TOKENS
            }
            crate::catalog::ModelFamily::VoxtralTts => {
                ModelVariant::VOXTRAL_TTS_CUDA_MAX_OUTPUT_FRAMES
            }
            _ => variant.tts_max_output_frames_hint().unwrap_or(configured),
        };
    }
    variant.tts_max_output_frames_hint().unwrap_or(configured)
}

/// Managed-state row limits resolved by the runtime instead of the operator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AutomaticRowLimits {
    /// `max_retained_sequences` was zero and resolved to the administrative
    /// ceiling. CUDA resident fitting may then retain fewer rows.
    pub(crate) retained_sequences: bool,
    /// `max_staged_transactions` was zero and resolved to the step launch width.
    pub(crate) staged_transactions: bool,
}

/// Configuration for the engine core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreConfig {
    /// Performance policy, resolved once before model loading.
    #[serde(default)]
    pub performance: crate::performance::PerformanceConfig,
    /// Directory containing models
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Maximum logical rows selected by one scheduler step.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Automatic or explicitly fixed physical tensor invocation width.
    #[serde(default)]
    pub max_tensor_batch_size: BatchSizePreference,

    /// Rollout mode for overlapping separate physical inference launches.
    #[serde(default)]
    pub physical_execution_mode: PhysicalExecutionMode,

    /// Prospective engine-wide physical launch ceiling. Serial and Shadow
    /// modes still resolve actual physical execution to one in-flight launch.
    #[serde(default)]
    pub max_physical_in_flight: PhysicalInFlightLimit,

    /// Run one step driver per execution lane (loaded model instance)
    /// instead of one for the whole engine, so one model's long dispatch
    /// does not hold every other model's steps. Off by default; set with
    /// IZWI_LANE_STEP_DRIVERS=1.
    #[serde(default = "default_lane_step_drivers")]
    pub lane_step_drivers: bool,

    /// Maximum retained sequence/session rows in managed model state.
    #[serde(default = "default_max_retained_sequences")]
    pub max_retained_sequences: usize,

    /// Maximum simultaneously staged managed-state transactions.
    #[serde(default = "default_max_staged_transactions")]
    pub max_staged_transactions: usize,

    /// Row limits the operator left automatic. An automatic limit is a
    /// runtime-sized default rather than an operator contract, so a model
    /// that fits its own rows may differ from it.
    #[serde(skip)]
    pub(crate) automatic_row_limits: AutomaticRowLimits,

    /// Maximum admitted jobs in the runtime inference queue.
    #[serde(default = "default_max_queued_requests")]
    pub max_queued_requests: usize,

    /// Maximum sequence length (tokens)
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Resolve CPU/Metal context from the load-time memory plan. CUDA ignores
    /// this flag; CUDA families may independently seal a resource-fitted
    /// context when their physical growth contract requires it.
    #[serde(default)]
    pub(crate) portable_context_auto: bool,

    /// Memory kept outside fitted state plans for allocator and backend
    /// command-buffer overhead.
    #[serde(default = "default_portable_context_reserve_bytes")]
    pub(crate) portable_context_reserve_bytes: u64,

    /// Operator override for the CUDA context safety reserve (see
    /// [`Self::cuda_context_reserve_bytes`]), from
    /// IZWI_CUDA_CONTEXT_RESERVE_BYTES: a positive byte count.
    #[serde(default = "default_cuda_context_reserve_override_bytes")]
    pub(crate) cuda_context_reserve_override_bytes: Option<u64>,

    /// Maximum number of tokens per step (token budget)
    #[serde(default = "default_max_tokens_per_step")]
    pub max_tokens_per_step: usize,

    /// Block size for KV cache paged attention
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// KV cache storage dtype hint (float16, float32, int8, ...).
    #[serde(default = "default_kv_cache_dtype")]
    pub kv_cache_dtype: String,

    /// Aggregate number of KV cache blocks across all paged state groups.
    /// Heterogeneous groups receive capacities with equal token reach.
    #[serde(default = "default_max_blocks")]
    pub max_blocks: usize,

    /// Scheduling policy
    #[serde(default)]
    pub scheduling_policy: SchedulingPolicy,

    /// Enable prefix reuse only for executor-backed physical cache snapshots.
    #[serde(default = "default_enable_prefix_caching")]
    pub enable_prefix_caching: bool,

    /// Namespace salt for managed physical-prefix reuse. Required explicitly
    /// whenever prefix caching is enabled.
    #[serde(default = "default_managed_prefix_cache_salt")]
    pub managed_prefix_cache_salt: Option<String>,

    /// Hard upper bound for retained committed-prefix pages. `None` is
    /// automatic: [`DEFAULT_MAX_PREFIX_CACHE_PAGES`], raised per model to a
    /// quarter of a CUDA KV arena that is fully allocated at load.
    #[serde(default)]
    pub max_prefix_cache_pages: Option<usize>,

    /// Committed prefix snapshots of hybrid recurrent state (one row each) a
    /// snapshot-sharing model may hold beyond its live rows. They are fitted
    /// and reserved with the rows at load. Set with IZWI_MAX_PREFIX_SNAPSHOTS
    /// (0 keeps hybrid state private, so hybrid prefixes never attach);
    /// default 8.
    #[serde(default = "default_max_prefix_snapshots")]
    pub max_prefix_snapshots: u32,

    /// Host KV pool budget in bytes for hierarchical offload (DS4). Zero (the
    /// default) keeps the feature fully dormant.
    #[serde(default)]
    pub kv_host_pool_budget_bytes: u64,

    /// Device-arena pressure ratio at which demotion steps start.
    #[serde(default = "default_kv_offload_high_watermark")]
    pub kv_offload_high_watermark: f32,

    /// Pressure ratio demotion steps drain toward before stopping.
    #[serde(default = "default_kv_offload_low_watermark")]
    pub kv_offload_low_watermark: f32,

    /// Maximum pages with a demotion copy in flight at once.
    #[serde(default = "default_kv_offload_max_in_flight_pages")]
    pub kv_offload_max_in_flight_pages: usize,

    /// Maximum host-resident pages promoted back in one prepare.
    #[serde(default = "default_kv_offload_max_promotion_pages")]
    pub kv_offload_max_promotion_pages: usize,

    /// Catalog-gated prefix-reuse default (DS1.6). Honored once when the
    /// engine core is constructed: catalog-auto mode engages the managed
    /// prefix namespace with the operator's salt or a generated per-process
    /// one, and per-model engagement is then gated by the catalog cell for
    /// the loaded family on the active backend.
    #[serde(default)]
    pub prefix_reuse_catalog_auto: bool,

    /// Enable chunked prefill for long prompts
    #[serde(default = "default_chunked_prefill")]
    pub enable_chunked_prefill: bool,

    /// Default-on CUDA admission for replay-capable Qwen3.8 requests.
    /// Operators can disable it with IZWI_CUDA_INCREMENTAL_CHAT=0.
    #[serde(default = "default_cuda_incremental_chat")]
    pub enable_cuda_incremental_chat: bool,

    /// Share of the post-weight CUDA budget that retained recurrent state may
    /// take when retained rows are automatic; the rest backs KV pages. Set
    /// with IZWI_CUDA_STATE_BUDGET_FRACTION in (0, 1]; default 0.25.
    #[serde(default = "default_cuda_state_budget_fraction")]
    pub cuda_state_budget_fraction: f64,

    /// Threshold for chunked prefill (tokens)
    #[serde(default = "default_chunked_prefill_threshold")]
    pub chunked_prefill_threshold: usize,

    /// Output sample rate (Hz)
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Number of audio codebooks
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: usize,

    /// Selected backend for execution and device policy.
    #[serde(default = "default_backend_kind")]
    pub backend: BackendKind,

    /// Number of CPU threads
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,

    /// Defer lower-priority decode while higher-priority work is waiting.
    #[serde(default = "default_enable_preemption")]
    pub enable_preemption: bool,

    /// Recompute preemption under managed KV pressure: when every claim
    /// holder of a model waits to grow, suspend the lowest-priority, then
    /// youngest, session of a family that declares token replay and rebuild
    /// it by replaying its tokens once the protected request leaves, instead
    /// of failing a request. Operators can disable it with
    /// IZWI_CAPACITY_REPLAY=0 (CUDA incremental chat keeps its replay).
    #[serde(default = "default_capacity_replay")]
    pub enable_capacity_replay: bool,

    /// Output tokens a chat request claims in managed KV beyond its prompt and
    /// current step. Claims grow by this quantum as generation proceeds
    /// instead of reserving the whole output budget up front, so concurrent
    /// requests are admitted together; zero restores full up-front claims.
    /// Set with IZWI_MANAGED_CHAT_CLAIM_QUANTUM_TOKENS; default 1024.
    #[serde(default = "default_managed_chat_claim_quantum_tokens")]
    pub managed_chat_claim_quantum_tokens: usize,

    /// Largest chat context fitted automatically on CPU/Metal when no explicit
    /// `max_sequence_length` is configured. Portable arenas are allocated in
    /// full at load, so fitting the model's whole advertised context (262,144
    /// tokens for Qwen3.5) pinned gigabytes of unified memory for an idle
    /// model. An explicit `max_sequence_length` still selects any length the
    /// memory budget can hold; zero disables the cap.
    #[serde(default = "default_portable_auto_context_cap_tokens")]
    pub portable_auto_context_cap_tokens: usize,

    /// Enable adaptive scheduling heuristics driven by runtime latency feedback.
    #[serde(default = "default_enable_adaptive_batching")]
    pub enable_adaptive_batching: bool,

    /// Minimum token budget per scheduler step when adaptive batching is enabled.
    #[serde(default = "default_min_tokens_per_step")]
    pub min_tokens_per_step: usize,

    /// Target time-to-first-token for adaptive scheduling.
    #[serde(default = "default_target_ttft_ms")]
    pub target_ttft_ms: f64,

    /// Target time-per-output-token for adaptive scheduling.
    #[serde(default = "default_target_decode_tpot_ms")]
    pub target_decode_tpot_ms: f64,

    /// Waiting-time interval used for priority aging in adaptive scheduling.
    #[serde(default = "default_priority_aging_ms")]
    pub priority_aging_ms: u64,
    /// Enable deadline-aware scheduler boosts.
    #[serde(default = "default_enable_deadline_scheduling")]
    pub enable_deadline_scheduling: bool,
    /// Soft SLA budget for critical-priority requests.
    #[serde(default = "default_critical_sla_ms")]
    pub critical_sla_ms: u64,
    /// Soft SLA budget for high-priority requests.
    #[serde(default = "default_high_sla_ms")]
    pub high_sla_ms: u64,
    /// Soft SLA budget for normal-priority requests.
    #[serde(default = "default_normal_sla_ms")]
    pub normal_sla_ms: u64,
    /// Soft SLA budget for low-priority requests.
    #[serde(default = "default_low_sla_ms")]
    pub low_sla_ms: u64,
    /// Enable thermal/power-aware scheduler adaptation.
    #[serde(default = "default_enable_power_adaptive")]
    pub enable_power_adaptive: bool,
    /// External thermal pressure hint in [0, 1].
    #[serde(default = "default_thermal_pressure_hint")]
    pub thermal_pressure_hint: f64,
    /// Force power-save scheduling mode.
    #[serde(default = "default_power_save_mode")]
    pub power_save_mode: bool,
    /// Enable decode token quanta greater than 1 when safe.
    #[serde(default = "default_enable_decode_quanta")]
    pub enable_decode_quanta: bool,
    /// Maximum decode tokens per request in one scheduler step.
    #[serde(default = "default_max_decode_tokens_per_request")]
    pub max_decode_tokens_per_request: usize,
}

fn default_models_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

fn default_max_batch_size() -> usize {
    8
}
fn default_max_retained_sequences() -> usize {
    8
}
fn default_max_staged_transactions() -> usize {
    8
}
fn default_max_queued_requests() -> usize {
    128
}
fn default_max_seq_len() -> usize {
    4096
}
fn default_max_tokens_per_step() -> usize {
    384
}

/// Scheduler step sizes a backend uses when the operator leaves them
/// automatic (zero), SCH-F3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackendStepSizes {
    pub(crate) max_tokens_per_step: usize,
    pub(crate) chunked_prefill_threshold: usize,
}

/// Automatic step sizes per backend. An engine step launches one physical
/// dispatch in the default serial mode and prefill is one request per
/// dispatch, so with chunked prefill on, a step's prefill span is the chunk
/// threshold (or the model's preferred span for an isolated request).
///
/// - CUDA always chunks (incremental chat turns it on). A 512-token chunk
///   keeps the GPU's prefill matmuls wide; the 2,048-token budget covers a
///   chunk beside a full decode batch. A larger budget would not widen a
///   step until prefill rows batch across requests.
/// - CPU and Metal leave chunked prefill off, so prefill runs whole prompts
///   (full prefill ignores the chunk and the budget) and the budget only
///   bounds decode, which needs at most 64 tokens (8 rows of 8-token quanta).
///   The sizes apply when an operator turns chunking on, and stay at the
///   historical 384 and 192.
pub(crate) const fn backend_step_sizes(backend: BackendKind) -> BackendStepSizes {
    match backend {
        BackendKind::Cuda => BackendStepSizes {
            max_tokens_per_step: 2_048,
            chunked_prefill_threshold: 512,
        },
        BackendKind::Metal | BackendKind::Cpu => BackendStepSizes {
            max_tokens_per_step: 384,
            chunked_prefill_threshold: 192,
        },
    }
}
fn default_block_size() -> usize {
    64
}
fn default_kv_cache_dtype() -> String {
    "float16".to_string()
}
fn default_max_blocks() -> usize {
    1024
}
fn default_lane_step_drivers() -> bool {
    std::env::var("IZWI_LANE_STEP_DRIVERS")
        .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "on"))
}

fn default_capacity_replay() -> bool {
    capacity_replay_from_env(std::env::var("IZWI_CAPACITY_REPLAY").ok().as_deref())
}

fn capacity_replay_from_env(value: Option<&str>) -> bool {
    value.is_none_or(|value| !matches!(value.trim(), "0" | "false" | "off"))
}

fn default_cuda_incremental_chat() -> bool {
    cuda_incremental_chat_from_env(std::env::var("IZWI_CUDA_INCREMENTAL_CHAT").ok().as_deref())
}

fn cuda_incremental_chat_from_env(value: Option<&str>) -> bool {
    value.is_none_or(|value| matches!(value, "1" | "true" | "on"))
}

pub(crate) const DEFAULT_CUDA_STATE_BUDGET_FRACTION: f64 = 0.25;

fn default_max_prefix_snapshots() -> u32 {
    max_prefix_snapshots_from_env(std::env::var("IZWI_MAX_PREFIX_SNAPSHOTS").ok().as_deref())
}

fn max_prefix_snapshots_from_env(value: Option<&str>) -> u32 {
    let default = crate::engine::cache::managed::DEFAULT_PREFIX_SNAPSHOT_ROWS;
    let Some(value) = value else {
        return default;
    };
    match value.trim().parse::<u32>() {
        Ok(rows) if rows <= 1_024 => rows,
        _ => {
            tracing::warn!(
                value,
                default,
                "Ignoring IZWI_MAX_PREFIX_SNAPSHOTS that is not a snapshot count in 0..=1024"
            );
            default
        }
    }
}

fn default_cuda_state_budget_fraction() -> f64 {
    cuda_state_budget_fraction_from_env(
        std::env::var("IZWI_CUDA_STATE_BUDGET_FRACTION")
            .ok()
            .as_deref(),
    )
}

fn cuda_state_budget_fraction_from_env(value: Option<&str>) -> f64 {
    let Some(value) = value else {
        return DEFAULT_CUDA_STATE_BUDGET_FRACTION;
    };
    match value.trim().parse::<f64>() {
        Ok(fraction) if fraction > 0.0 && fraction <= 1.0 => fraction,
        _ => {
            tracing::warn!(
                value,
                default = DEFAULT_CUDA_STATE_BUDGET_FRACTION,
                "Ignoring IZWI_CUDA_STATE_BUDGET_FRACTION outside (0, 1]"
            );
            DEFAULT_CUDA_STATE_BUDGET_FRACTION
        }
    }
}

fn default_chunked_prefill() -> bool {
    false
}
fn default_chunked_prefill_threshold() -> usize {
    192
}
fn default_sample_rate() -> u32 {
    24000
}

fn default_num_codebooks() -> usize {
    8
}

fn default_backend_kind() -> BackendKind {
    BackendRouter::resolve_context(
        BackendRouter::unset_preference(),
        BackendSelectionSource::Default,
    )
    .backend_kind
}
fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8)
}
fn default_managed_chat_claim_quantum_tokens() -> usize {
    claim_quantum_from_env(
        std::env::var("IZWI_MANAGED_CHAT_CLAIM_QUANTUM_TOKENS")
            .ok()
            .as_deref(),
    )
}

/// The chat claim quantum, overridable with
/// IZWI_MANAGED_CHAT_CLAIM_QUANTUM_TOKENS (zero restores full claims); an
/// unparsable value keeps the default.
fn claim_quantum_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(1024)
}

fn default_portable_auto_context_cap_tokens() -> usize {
    32 * 1024
}

fn default_enable_preemption() -> bool {
    false
}
fn default_enable_adaptive_batching() -> bool {
    false
}
fn default_min_tokens_per_step() -> usize {
    96
}
fn default_target_ttft_ms() -> f64 {
    250.0
}
fn default_target_decode_tpot_ms() -> f64 {
    40.0
}
fn default_priority_aging_ms() -> u64 {
    1_000
}
fn default_enable_deadline_scheduling() -> bool {
    true
}
fn default_critical_sla_ms() -> u64 {
    200
}
fn default_high_sla_ms() -> u64 {
    400
}
fn default_normal_sla_ms() -> u64 {
    1_000
}
fn default_low_sla_ms() -> u64 {
    2_500
}
fn default_enable_power_adaptive() -> bool {
    false
}
fn default_thermal_pressure_hint() -> f64 {
    std::env::var("IZWI_THERMAL_PRESSURE")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}
fn default_power_save_mode() -> bool {
    std::env::var("IZWI_POWER_SAVE")
        .ok()
        .map(|raw| {
            let value = raw.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}
fn default_enable_decode_quanta() -> bool {
    false
}
fn default_max_decode_tokens_per_request() -> usize {
    2
}
fn default_portable_context_reserve_bytes() -> u64 {
    1024 * 1024 * 1024
}

/// Share of CUDA device memory, in percent, that the resident-context fit
/// keeps free when it exceeds `portable_context_reserve_bytes`.
pub(crate) const CUDA_CONTEXT_RESERVE_PERCENT: u64 = 3;

fn default_cuda_context_reserve_override_bytes() -> Option<u64> {
    cuda_context_reserve_override_from_env(
        std::env::var("IZWI_CUDA_CONTEXT_RESERVE_BYTES")
            .ok()
            .as_deref(),
    )
}

fn cuda_context_reserve_override_from_env(value: Option<&str>) -> Option<u64> {
    let value = value?;
    match value.trim().parse::<u64>() {
        Ok(bytes) if bytes > 0 => Some(bytes),
        _ => {
            tracing::warn!(
                value,
                "Ignoring IZWI_CUDA_CONTEXT_RESERVE_BYTES that is not a positive byte count"
            );
            None
        }
    }
}

/// The larger of `floor_bytes` and [`CUDA_CONTEXT_RESERVE_PERCENT`] of the
/// device, unless an operator set the reserve explicitly.
fn cuda_context_reserve_bytes(
    floor_bytes: u64,
    device_capacity: ResourceAmount,
    override_bytes: Option<u64>,
) -> u64 {
    if let Some(bytes) = override_bytes {
        return bytes;
    }
    match device_capacity {
        ResourceAmount::Known(capacity) => {
            floor_bytes.max(capacity.saturating_mul(CUDA_CONTEXT_RESERVE_PERCENT) / 100)
        }
        ResourceAmount::Unknown => floor_bytes,
    }
}
fn default_enable_prefix_caching() -> bool {
    false
}
fn default_managed_prefix_cache_salt() -> Option<String> {
    None
}
fn default_kv_offload_high_watermark() -> f32 {
    0.85
}

fn default_kv_offload_low_watermark() -> f32 {
    0.70
}

fn default_kv_offload_max_in_flight_pages() -> usize {
    8
}

fn default_kv_offload_max_promotion_pages() -> usize {
    64
}

impl Default for EngineCoreConfig {
    fn default() -> Self {
        Self {
            performance: Default::default(),
            models_dir: default_models_dir(),
            max_batch_size: default_max_batch_size(),
            max_tensor_batch_size: BatchSizePreference::Auto,
            physical_execution_mode: PhysicalExecutionMode::Serial,
            max_physical_in_flight: PhysicalInFlightLimit::default(),
            lane_step_drivers: default_lane_step_drivers(),
            max_retained_sequences: default_max_retained_sequences(),
            max_staged_transactions: default_max_staged_transactions(),
            automatic_row_limits: AutomaticRowLimits::default(),
            max_queued_requests: default_max_queued_requests(),
            max_seq_len: default_max_seq_len(),
            portable_context_auto: false,
            portable_context_reserve_bytes: default_portable_context_reserve_bytes(),
            cuda_context_reserve_override_bytes: default_cuda_context_reserve_override_bytes(),
            max_tokens_per_step: default_max_tokens_per_step(),
            block_size: default_block_size(),
            kv_cache_dtype: default_kv_cache_dtype(),
            max_blocks: default_max_blocks(),
            scheduling_policy: SchedulingPolicy::default(),
            enable_prefix_caching: default_enable_prefix_caching(),
            managed_prefix_cache_salt: default_managed_prefix_cache_salt(),
            max_prefix_cache_pages: None,
            max_prefix_snapshots: default_max_prefix_snapshots(),
            kv_host_pool_budget_bytes: 0,
            kv_offload_high_watermark: default_kv_offload_high_watermark(),
            kv_offload_low_watermark: default_kv_offload_low_watermark(),
            kv_offload_max_in_flight_pages: default_kv_offload_max_in_flight_pages(),
            kv_offload_max_promotion_pages: default_kv_offload_max_promotion_pages(),
            prefix_reuse_catalog_auto: false,
            enable_chunked_prefill: default_chunked_prefill(),
            enable_cuda_incremental_chat: default_cuda_incremental_chat(),
            cuda_state_budget_fraction: default_cuda_state_budget_fraction(),
            chunked_prefill_threshold: default_chunked_prefill_threshold(),
            sample_rate: default_sample_rate(),
            num_codebooks: default_num_codebooks(),
            backend: default_backend_kind(),
            num_threads: default_num_threads(),
            enable_preemption: default_enable_preemption(),
            enable_capacity_replay: default_capacity_replay(),
            managed_chat_claim_quantum_tokens: default_managed_chat_claim_quantum_tokens(),
            portable_auto_context_cap_tokens: default_portable_auto_context_cap_tokens(),
            enable_adaptive_batching: default_enable_adaptive_batching(),
            min_tokens_per_step: default_min_tokens_per_step(),
            target_ttft_ms: default_target_ttft_ms(),
            target_decode_tpot_ms: default_target_decode_tpot_ms(),
            priority_aging_ms: default_priority_aging_ms(),
            enable_deadline_scheduling: default_enable_deadline_scheduling(),
            critical_sla_ms: default_critical_sla_ms(),
            high_sla_ms: default_high_sla_ms(),
            normal_sla_ms: default_normal_sla_ms(),
            low_sla_ms: default_low_sla_ms(),
            enable_power_adaptive: default_enable_power_adaptive(),
            thermal_pressure_hint: default_thermal_pressure_hint(),
            power_save_mode: default_power_save_mode(),
            enable_decode_quanta: default_enable_decode_quanta(),
            max_decode_tokens_per_request: default_max_decode_tokens_per_request(),
        }
    }
}

impl EngineCoreConfig {
    /// Device memory the CUDA resident-context fit leaves outside every
    /// reservation (the decode workspace reserve is added on top).
    ///
    /// After the fit, CUDA keeps allocating memory that no reservation
    /// tracks: kernel modules and library workspaces (cuBLAS) loaded on first
    /// use, local memory grown for the kernels that need it, CUDA graph pools,
    /// and per-step transients. On the H100 field deploy of Qwen3.8 that grew
    /// by about 0.88 GB over the first turns, against a 1 GiB reserve, and
    /// left 132 MB for every later admission. Part of it scales with the
    /// device (SMs, graph pools for larger models), so the reserve is the
    /// larger of `portable_context_reserve_bytes` (1 GiB) and 3% of device
    /// memory: 2.55 GB on an 80 GB H100. CPU and Metal do not use it.
    pub(crate) fn cuda_context_reserve_bytes(&self, device_capacity: ResourceAmount) -> u64 {
        cuda_context_reserve_bytes(
            self.portable_context_reserve_bytes,
            device_capacity,
            self.cuda_context_reserve_override_bytes,
        )
    }

    pub fn cuda_incremental_chat_enabled(&self) -> bool {
        self.backend == BackendKind::Cuda && self.enable_cuda_incremental_chat
    }

    pub(crate) fn effective_chunked_prefill(&self) -> bool {
        self.enable_chunked_prefill || self.cuda_incremental_chat_enabled()
    }

    /// Set the step token budget and prefill chunk threshold from the
    /// operator's values, where zero takes this backend's automatic size.
    pub(crate) fn resolve_step_sizes(
        &mut self,
        max_tokens_per_step: usize,
        chunked_prefill_threshold: usize,
    ) {
        let automatic = backend_step_sizes(self.backend);
        self.max_tokens_per_step = match max_tokens_per_step {
            0 => automatic.max_tokens_per_step,
            tokens => tokens,
        };
        self.chunked_prefill_threshold = match chunked_prefill_threshold {
            0 => automatic.chunked_prefill_threshold,
            tokens => tokens,
        };
    }

    /// Resolve rollout-aware dispatch and physical-launch capacity axes.
    pub fn resolved_physical_execution_capacity(&self) -> PhysicalExecutionCapacity {
        self.physical_execution_mode
            .resolve_capacity(self.max_physical_in_flight)
    }

    /// Apply the CUDA-native context ceiling and ensure a single paged-state
    /// group has enough pages to reach it. CPU and Metal retain the configured
    /// sequence and cache defaults exactly.
    pub(crate) fn apply_backend_context_capacity(&mut self, configured_max_seq_len: usize) {
        if self.backend != BackendKind::Cuda {
            self.max_seq_len = if self.portable_context_auto {
                CUDA_MAX_NATIVE_CONTEXT_TOKENS
            } else {
                configured_max_seq_len.max(1)
            };
            return;
        }

        self.max_seq_len = CUDA_MAX_NATIVE_CONTEXT_TOKENS;
        let page_tokens = self.block_size.max(1);
        let required_pages = self.max_seq_len.div_ceil(page_tokens);
        // The prefix-cache budget must fit beside one maximum-length request
        // (the policy resolver reserves that request's pages). Without the
        // headroom, prefix reuse resolves to a zero budget on CUDA, and
        // explicit enablement fails startup. The arena grows on demand, so
        // this raises the logical cap only.
        let prefix_pages = if self.enable_prefix_caching {
            self.prefix_cache_pages_or_default()
        } else {
            0
        };
        self.max_blocks = self
            .max_blocks
            .max(required_pages.saturating_add(prefix_pages));
    }

    /// The configured prefix page budget, or the default when it is
    /// automatic (the floor each model's automatic budget starts from).
    pub(crate) fn prefix_cache_pages_or_default(&self) -> usize {
        self.max_prefix_cache_pages
            .unwrap_or(DEFAULT_MAX_PREFIX_CACHE_PAGES)
    }

    pub fn resolved_kv_cache_policy(&self) -> Result<ResolvedKvCachePolicy> {
        resolve_kv_cache_policy(
            self.block_size,
            &self.kv_cache_dtype,
            self.enable_prefix_caching,
            self.managed_prefix_cache_salt.as_deref(),
            self.prefix_cache_pages_or_default(),
            self.max_blocks,
            self.max_seq_len,
            // Catalog-auto reuse degrades instead of failing startup when the
            // safe page budget is zero; explicit enablement stays fail-closed.
            self.prefix_reuse_catalog_auto,
        )
    }

    /// Resolve prefix-reuse engagement once, in place. Explicit operator
    /// enablement keeps the fail-closed salt contract (a missing namespace
    /// stays absent so the policy resolver rejects the configuration);
    /// catalog-auto mode engages with the operator namespace or a generated
    /// per-process one. Call before `resolved_kv_cache_policy` so both agree
    /// on one namespace.
    pub fn apply_prefix_engagement(&mut self) {
        let (engaged, namespace) = crate::config::resolve_prefix_engagement(
            self.enable_prefix_caching,
            self.prefix_reuse_catalog_auto,
            self.managed_prefix_cache_salt.as_deref(),
        );
        if engaged {
            self.enable_prefix_caching = true;
            if let Some(namespace) = namespace {
                self.managed_prefix_cache_salt = Some(namespace);
            }
        }
        // The auto flag stays set for catalog-auto engagements so the policy
        // resolver can degrade instead of failing startup on a zero budget.
    }

    /// Create config for Qwen3-TTS model
    pub fn for_qwen3_tts() -> Self {
        Self {
            sample_rate: 24000,
            num_codebooks: 8,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod managed_kv_default_tests {
    use super::{
        capacity_replay_from_env, claim_quantum_from_env, resolve_backend_model_context,
        tts_explicit_output_limit, EngineCoreConfig, CUDA_MAX_NATIVE_CONTEXT_TOKENS,
    };
    use crate::backends::BackendKind;
    use crate::config::{
        KvCacheDtype, PhysicalExecutionMode, PhysicalInFlightLimit, PrefixCachePolicy,
    };
    use crate::model::ModelVariant;

    #[test]
    fn incremental_chat_defaults_on_and_preserves_explicit_opt_out() {
        assert!(super::cuda_incremental_chat_from_env(None));
        for value in ["1", "true", "on"] {
            assert!(super::cuda_incremental_chat_from_env(Some(value)));
        }
        for value in ["0", "false", "off", "", "invalid"] {
            assert!(!super::cuda_incremental_chat_from_env(Some(value)));
        }
    }

    #[test]
    fn prefix_snapshot_budget_parses_a_bounded_count() {
        use super::max_prefix_snapshots_from_env;
        assert_eq!(max_prefix_snapshots_from_env(None), 8);
        assert_eq!(max_prefix_snapshots_from_env(Some(" 16 ")), 16);
        assert_eq!(max_prefix_snapshots_from_env(Some("0")), 0);
        for invalid in ["-1", "many", "", "1025", "2.5"] {
            assert_eq!(max_prefix_snapshots_from_env(Some(invalid)), 8, "{invalid}");
        }
    }

    #[test]
    fn cuda_state_budget_fraction_accepts_only_a_positive_share() {
        use super::{cuda_state_budget_fraction_from_env, DEFAULT_CUDA_STATE_BUDGET_FRACTION};
        assert_eq!(
            cuda_state_budget_fraction_from_env(None),
            DEFAULT_CUDA_STATE_BUDGET_FRACTION
        );
        assert_eq!(cuda_state_budget_fraction_from_env(Some(" 0.4 ")), 0.4);
        assert_eq!(cuda_state_budget_fraction_from_env(Some("1")), 1.0);
        for value in ["0", "-0.1", "1.5", "NaN", "", "quarter"] {
            assert_eq!(
                cuda_state_budget_fraction_from_env(Some(value)),
                DEFAULT_CUDA_STATE_BUDGET_FRACTION,
                "{value:?}"
            );
        }
    }

    #[test]
    fn chat_claim_quantum_takes_the_operator_override() {
        assert_eq!(claim_quantum_from_env(None), 1024);
        assert_eq!(claim_quantum_from_env(Some(" 16 ")), 16);
        assert_eq!(claim_quantum_from_env(Some("0")), 0);
        assert_eq!(claim_quantum_from_env(Some("lots")), 1024);
    }

    #[test]
    fn capacity_replay_is_on_unless_the_operator_turns_it_off() {
        assert!(capacity_replay_from_env(None));
        for value in ["1", "true", "on", " on "] {
            assert!(capacity_replay_from_env(Some(value)), "{value}");
        }
        for value in ["0", "false", "off", " 0 "] {
            assert!(!capacity_replay_from_env(Some(value)), "{value}");
        }
    }

    #[test]
    fn incremental_chat_rollout_requires_cuda_and_enables_resumable_prefill() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let mut config = EngineCoreConfig {
                backend,
                enable_cuda_incremental_chat: true,
                enable_chunked_prefill: false,
                ..Default::default()
            };
            assert_eq!(
                config.cuda_incremental_chat_enabled(),
                backend == BackendKind::Cuda
            );
            assert_eq!(
                config.effective_chunked_prefill(),
                backend == BackendKind::Cuda
            );
            config.enable_cuda_incremental_chat = false;
            assert!(!config.cuda_incremental_chat_enabled());
            assert!(!config.effective_chunked_prefill());
            config.enable_chunked_prefill = true;
            assert!(config.effective_chunked_prefill());
        }
    }

    #[test]
    fn automatic_step_sizes_are_per_backend() {
        use super::backend_step_sizes;
        let cuda = backend_step_sizes(BackendKind::Cuda);
        assert_eq!(cuda.max_tokens_per_step, 2_048);
        assert_eq!(cuda.chunked_prefill_threshold, 512);
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let automatic = backend_step_sizes(backend);
            // A chunk always fits the budget beside a full CUDA decode batch.
            assert!(automatic.chunked_prefill_threshold + 32 <= automatic.max_tokens_per_step);
            let mut config = EngineCoreConfig {
                backend,
                ..Default::default()
            };
            config.resolve_step_sizes(0, 0);
            assert_eq!(config.max_tokens_per_step, automatic.max_tokens_per_step);
            assert_eq!(
                config.chunked_prefill_threshold,
                automatic.chunked_prefill_threshold
            );
            config.resolve_step_sizes(4_096, 256);
            assert_eq!(config.max_tokens_per_step, 4_096);
            assert_eq!(config.chunked_prefill_threshold, 256);
        }
    }

    #[test]
    fn managed_prefix_reuse_is_disabled_by_default() {
        let config = EngineCoreConfig::default();
        assert!(!config.enable_prefix_caching);
        assert!(config.managed_prefix_cache_salt.is_none());
        let policy = config.resolved_kv_cache_policy().unwrap();
        assert_eq!(policy.effective.page_size, 64);
        assert_eq!(policy.effective.dtype, KvCacheDtype::Float16);
        assert_eq!(policy.effective.prefix, PrefixCachePolicy::Disabled);
    }

    #[test]
    fn prefix_engagement_applies_once_and_preserves_explicit_fail_closed() {
        // Catalog-auto engages with the operator namespace when present and
        // clears the auto flag so the engine never generates a second one.
        let mut config = EngineCoreConfig {
            prefix_reuse_catalog_auto: true,
            managed_prefix_cache_salt: Some("tenant-a".to_string()),
            ..EngineCoreConfig::default()
        };
        config.apply_prefix_engagement();
        assert!(config.enable_prefix_caching);
        assert_eq!(
            config.managed_prefix_cache_salt.as_deref(),
            Some("tenant-a")
        );
        assert_eq!(
            config.resolved_kv_cache_policy().unwrap().effective.prefix,
            PrefixCachePolicy::Namespaced {
                namespace: "tenant-a".to_string(),
                max_pages: 128
            }
        );

        // Catalog-auto without a namespace generates one per process.
        let mut config = EngineCoreConfig {
            prefix_reuse_catalog_auto: true,
            ..EngineCoreConfig::default()
        };
        config.apply_prefix_engagement();
        assert!(config.enable_prefix_caching);
        let generated = config.managed_prefix_cache_salt.clone().expect("generated");
        assert!(!generated.is_empty());

        // Explicit enablement without a namespace stays fail-closed.
        let mut config = EngineCoreConfig {
            enable_prefix_caching: true,
            ..EngineCoreConfig::default()
        };
        config.apply_prefix_engagement();
        assert!(config.enable_prefix_caching);
        assert!(config.managed_prefix_cache_salt.is_none());
        assert!(config.resolved_kv_cache_policy().is_err());

        // Disabled stays disabled.
        let mut config = EngineCoreConfig::default();
        config.apply_prefix_engagement();
        assert!(!config.enable_prefix_caching);
        assert!(config.managed_prefix_cache_salt.is_none());
    }

    #[test]
    fn physical_execution_defaults_remain_serial_and_capacity_axes_are_distinct() {
        let config = EngineCoreConfig::default();
        assert_eq!(
            config.physical_execution_mode,
            PhysicalExecutionMode::Serial
        );
        assert_eq!(config.max_physical_in_flight.get(), 1);
        assert_eq!(config.max_batch_size, 8);
        assert!(config.max_tensor_batch_size.resolve(config.backend) >= 2);

        let mut shadow = config;
        shadow.physical_execution_mode = PhysicalExecutionMode::Shadow;
        shadow.max_physical_in_flight = PhysicalInFlightLimit::new(4).unwrap();
        let capacity = shadow.resolved_physical_execution_capacity();
        assert_eq!(capacity.candidate_dispatch_limit.get(), 4);
        assert_eq!(capacity.physical_launch_limit.get(), 1);
    }

    #[test]
    fn cuda_context_capacity_reaches_largest_native_model() {
        let mut config = EngineCoreConfig {
            backend: BackendKind::Cuda,
            block_size: 64,
            max_blocks: 1024,
            ..EngineCoreConfig::default()
        };

        config.apply_backend_context_capacity(4096);

        assert_eq!(config.max_seq_len, CUDA_MAX_NATIVE_CONTEXT_TOKENS);
        assert_eq!(config.max_blocks, 4096);
    }

    #[test]
    fn cuda_context_capacity_leaves_room_for_the_prefix_cache() {
        let mut config = EngineCoreConfig {
            backend: BackendKind::Cuda,
            block_size: 64,
            max_blocks: 1024,
            enable_prefix_caching: true,
            managed_prefix_cache_salt: Some("tenant".into()),
            max_prefix_cache_pages: Some(256),
            ..EngineCoreConfig::default()
        };

        config.apply_backend_context_capacity(4096);

        assert_eq!(config.max_blocks, 4096 + 256);
        let policy = config
            .resolved_kv_cache_policy()
            .expect("explicit prefix caching fits beside a max-length request");
        assert!(matches!(
            policy.effective.prefix,
            crate::config::PrefixCachePolicy::Namespaced { max_pages: 256, .. }
        ));
    }

    #[test]
    fn cpu_and_metal_context_capacity_preserve_configuration() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            let mut config = EngineCoreConfig {
                backend,
                block_size: 64,
                max_blocks: 1024,
                ..EngineCoreConfig::default()
            };

            config.apply_backend_context_capacity(4096);

            assert_eq!(config.max_seq_len, 4096);
            assert_eq!(config.max_blocks, 1024);
        }
    }

    #[test]
    fn portable_auto_keeps_a_native_upper_bound_until_model_fit() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            let mut config = EngineCoreConfig {
                backend,
                portable_context_auto: true,
                ..EngineCoreConfig::default()
            };
            config.apply_backend_context_capacity(4096);
            assert_eq!(config.max_seq_len, CUDA_MAX_NATIVE_CONTEXT_TOKENS);
            assert_eq!(config.max_blocks, 1024);
        }
    }

    #[test]
    fn loaded_model_context_is_cuda_only() {
        assert_eq!(
            resolve_backend_model_context(BackendKind::Cuda, 4096, 40_960).unwrap(),
            40_960
        );
        assert_eq!(
            resolve_backend_model_context(BackendKind::Cpu, 4096, 40_960).unwrap(),
            4096
        );
        assert_eq!(
            resolve_backend_model_context(BackendKind::Metal, 4096, 40_960).unwrap(),
            4096
        );
        assert!(resolve_backend_model_context(BackendKind::Cuda, 4096, 0).is_err());
    }

    #[test]
    fn cuda_unlocks_explicit_audio_generation_contexts_only() {
        assert_eq!(
            tts_explicit_output_limit(BackendKind::Cuda, ModelVariant::Lfm25Audio15BGguf, 4096,),
            32_768
        );
        assert_eq!(
            tts_explicit_output_limit(BackendKind::Cuda, ModelVariant::FishAudioS2Pro, 4096),
            32_768
        );
        assert_eq!(
            tts_explicit_output_limit(BackendKind::Cuda, ModelVariant::Voxtral4BTts2603, 4096),
            2048
        );
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(
                tts_explicit_output_limit(backend, ModelVariant::Lfm25Audio15BGguf, 4096),
                4096
            );
            assert_eq!(
                tts_explicit_output_limit(backend, ModelVariant::FishAudioS2Pro, 4096),
                4096
            );
            assert_eq!(
                tts_explicit_output_limit(backend, ModelVariant::Voxtral4BTts2603, 4096),
                1500
            );
        }
    }
}

#[cfg(test)]
mod cuda_context_reserve_tests {
    use super::{
        cuda_context_reserve_bytes, cuda_context_reserve_override_from_env, EngineCoreConfig,
    };
    use crate::engine::{
        BatchId, CapacitySource, ExecutionGroupId, PhysicalCapacityProvider,
        PhysicalCapacitySnapshot, ReservationClass, ReservationOwner, ResourceAmount,
        ResourceAuthority, ResourceLease, ResourceVector,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    const GIB: u64 = 1024 * 1024 * 1024;
    /// The H100 the field failure ran on (`physical_capacity` in its refusal).
    const H100_BYTES: u64 = 85_064_286_208;

    #[test]
    fn cuda_context_reserve_scales_with_the_device_above_a_floor() {
        // 3% of the H100, versus the 1 GiB the fit used to keep.
        assert_eq!(
            cuda_context_reserve_bytes(GIB, ResourceAmount::Known(H100_BYTES), None),
            2_551_928_586
        );
        // Small devices keep the floor; unknown capacity cannot scale it.
        assert_eq!(
            cuda_context_reserve_bytes(GIB, ResourceAmount::Known(24 * GIB), None),
            GIB
        );
        assert_eq!(
            cuda_context_reserve_bytes(GIB, ResourceAmount::Unknown, None),
            GIB
        );
        // An operator value wins in both directions.
        for bytes in [GIB / 2, 4 * GIB] {
            assert_eq!(
                cuda_context_reserve_bytes(GIB, ResourceAmount::Known(H100_BYTES), Some(bytes)),
                bytes
            );
        }
        let config = EngineCoreConfig {
            cuda_context_reserve_override_bytes: None,
            ..EngineCoreConfig::default()
        };
        assert_eq!(
            config.cuda_context_reserve_bytes(ResourceAmount::Known(H100_BYTES)),
            2_551_928_586
        );
    }

    #[test]
    fn cuda_context_reserve_override_accepts_only_a_positive_byte_count() {
        assert_eq!(cuda_context_reserve_override_from_env(None), None);
        assert_eq!(
            cuda_context_reserve_override_from_env(Some(" 2147483648 ")),
            Some(2 * GIB)
        );
        for value in ["0", "-1", "2GiB", "1.5", ""] {
            assert_eq!(
                cuda_context_reserve_override_from_env(Some(value)),
                None,
                "{value:?}"
            );
        }
    }

    #[derive(Debug)]
    struct DeviceFree {
        available: AtomicU64,
    }

    impl PhysicalCapacityProvider for DeviceFree {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            let device = |bytes| ResourceVector {
                device_bytes: ResourceAmount::Known(bytes),
                ..ResourceVector::zero()
            };
            PhysicalCapacitySnapshot {
                capacity: device(H100_BYTES),
                available: device(self.available.load(Ordering::SeqCst)),
                source: CapacitySource::Test,
            }
        }
    }

    fn device(bytes: u64) -> ResourceVector {
        ResourceVector {
            device_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        }
    }

    struct FieldDeploy {
        authority: Arc<ResourceAuthority>,
        provider: Arc<DeviceFree>,
        _leases: [ResourceLease; 2],
    }

    /// The H100 field deploy of Qwen3.8 on b1d51b89, replayed against a
    /// reserve. The fit fills KV until free memory minus every pending claim
    /// equals the reserve plus the decode workspace reserve. Then the first
    /// turns commit a state row (recorded as materialized) and grow memory
    /// no reservation tracks by `UNTRACKED_GROWTH`, and a 64 MiB request claim
    /// is pending.
    fn field_after_first_turns(reserve: u64) -> FieldDeploy {
        // Model pending: 86 state rows x 156,893,184 + graph decode.
        const MODEL_PENDING: u64 = 14_030_602_240;
        // In use before the fit: the CUDA context and its baseline pools.
        const CONTEXT_BASELINE: u64 = 615_918_464;
        // Eight decode rows x 401,408 B, added to the reserve by the fit.
        const DECODE_RESERVE: u64 = 3_211_264;
        const STATE_ROW: u64 = 156_893_184;
        const REQUEST_PENDING: u64 = 64 * 1024 * 1024;
        let free_at_fit = MODEL_PENDING + reserve + DECODE_RESERVE;
        let resident = H100_BYTES - CONTEXT_BASELINE - free_at_fit;
        let provider = Arc::new(DeviceFree {
            available: AtomicU64::new(free_at_fit),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let model = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Model, "qwen38"),
                device(resident + MODEL_PENDING),
                device(resident),
            )
            .unwrap();
        provider
            .available
            .fetch_sub(STATE_ROW + UNTRACKED_GROWTH, Ordering::SeqCst);
        model
            .record_materialized_usage(device(resident + STATE_ROW))
            .unwrap();
        let request = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "second-turn"),
                device(REQUEST_PENDING),
            )
            .unwrap();
        FieldDeploy {
            authority,
            provider,
            _leases: [model, request],
        }
    }

    /// Untracked device memory that grew after the fit in the field, from
    /// the refusal: with the row recorded, 1 GiB + 3,211,264 - 64 MiB -
    /// 877,920,256 leaves the 131,923,968 B margin the fixed build has.
    const UNTRACKED_GROWTH: u64 = 877_920_256;

    fn workspace_fits(authority: &Arc<ResourceAuthority>, bytes: u64) -> bool {
        authority
            .reserve_batch_workspace(ExecutionGroupId::new(1), BatchId::new(1), device(bytes))
            .is_ok()
    }

    #[test]
    fn scaled_cuda_reserve_absorbs_twice_the_untracked_growth_measured_in_the_field() {
        const DECODE_ROW_WORKSPACE: u64 = 401_408;
        let old_reserve = GIB;
        let new_reserve = EngineCoreConfig {
            cuda_context_reserve_override_bytes: None,
            ..EngineCoreConfig::default()
        }
        .cuda_context_reserve_bytes(ResourceAmount::Known(H100_BYTES));

        for (reserve, margin) in [(old_reserve, 131_923_968), (new_reserve, 1_610_110_730)] {
            let FieldDeploy {
                authority,
                provider,
                _leases,
            } = field_after_first_turns(reserve);
            // The margin is exact, and both reserves pass the field step.
            assert!(workspace_fits(&authority, DECODE_ROW_WORKSPACE));
            assert!(workspace_fits(&authority, margin));
            assert!(!workspace_fits(&authority, margin + 1));
            // What no reservation tracks: the baseline plus the growth.
            let snapshot = authority.snapshot();
            let (
                ResourceAmount::Known(capacity),
                ResourceAmount::Known(available),
                ResourceAmount::Known(materialized),
            ) = (
                snapshot.physical.capacity.device_bytes,
                snapshot.physical.available.device_bytes,
                snapshot.materialized.device_bytes,
            )
            else {
                panic!("the field authority reports known device bytes");
            };
            assert_eq!(capacity - available - materialized, 1_493_838_720);

            // The same growth again (another kernel set, larger library
            // workspaces, a graph recapture): 1 GiB cannot absorb it.
            provider
                .available
                .fetch_sub(UNTRACKED_GROWTH, Ordering::SeqCst);
            assert_eq!(
                workspace_fits(&authority, DECODE_ROW_WORKSPACE),
                reserve == new_reserve,
                "reserve {reserve}"
            );
        }
    }
}
