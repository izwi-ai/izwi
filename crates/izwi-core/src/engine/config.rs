//! Engine configuration types.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::scheduler::SchedulingPolicy;
use crate::backends::{BackendKind, BackendPreference, BackendRouter, BackendSelectionSource};
use crate::config::{
    resolve_kv_cache_policy, BatchSizePreference, PhysicalExecutionCapacity, PhysicalExecutionMode,
    PhysicalInFlightLimit, ResolvedKvCachePolicy,
};
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

    /// Maximum retained sequence/session rows in managed model state.
    #[serde(default = "default_max_retained_sequences")]
    pub max_retained_sequences: usize,

    /// Maximum simultaneously staged managed-state transactions.
    #[serde(default = "default_max_staged_transactions")]
    pub max_staged_transactions: usize,

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

    /// Hard upper bound for retained committed-prefix pages.
    #[serde(default = "default_max_prefix_cache_pages")]
    pub max_prefix_cache_pages: usize,

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

    /// Threshold for chunked prefill (tokens)
    #[serde(default = "default_chunked_prefill_threshold")]
    pub chunked_prefill_threshold: usize,

    /// Output sample rate (Hz)
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Number of audio codebooks
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: usize,

    /// Chunk size for streaming output (samples)
    #[serde(default = "default_streaming_chunk_size")]
    pub streaming_chunk_size: usize,

    /// Selected backend for execution and device policy.
    #[serde(default = "default_backend_kind")]
    pub backend: BackendKind,

    /// Number of CPU threads
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,

    /// Defer lower-priority decode while higher-priority work is waiting.
    #[serde(default = "default_enable_preemption")]
    pub enable_preemption: bool,

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
fn default_block_size() -> usize {
    64
}
fn default_kv_cache_dtype() -> String {
    "float16".to_string()
}
fn default_max_blocks() -> usize {
    1024
}
fn default_cuda_incremental_chat() -> bool {
    cuda_incremental_chat_from_env(std::env::var("IZWI_CUDA_INCREMENTAL_CHAT").ok().as_deref())
}

fn cuda_incremental_chat_from_env(value: Option<&str>) -> bool {
    value.is_none_or(|value| matches!(value, "1" | "true" | "on"))
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
fn default_streaming_chunk_size() -> usize {
    4800
} // 200ms at 24kHz

fn default_backend_kind() -> BackendKind {
    BackendRouter::resolve_context_from_env_or(
        BackendPreference::Auto,
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
fn default_enable_prefix_caching() -> bool {
    false
}
fn default_managed_prefix_cache_salt() -> Option<String> {
    None
}
fn default_max_prefix_cache_pages() -> usize {
    128
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
            max_retained_sequences: default_max_retained_sequences(),
            max_staged_transactions: default_max_staged_transactions(),
            max_queued_requests: default_max_queued_requests(),
            max_seq_len: default_max_seq_len(),
            portable_context_auto: false,
            portable_context_reserve_bytes: default_portable_context_reserve_bytes(),
            max_tokens_per_step: default_max_tokens_per_step(),
            block_size: default_block_size(),
            kv_cache_dtype: default_kv_cache_dtype(),
            max_blocks: default_max_blocks(),
            scheduling_policy: SchedulingPolicy::default(),
            enable_prefix_caching: default_enable_prefix_caching(),
            managed_prefix_cache_salt: default_managed_prefix_cache_salt(),
            max_prefix_cache_pages: default_max_prefix_cache_pages(),
            prefix_reuse_catalog_auto: false,
            enable_chunked_prefill: default_chunked_prefill(),
            enable_cuda_incremental_chat: default_cuda_incremental_chat(),
            chunked_prefill_threshold: default_chunked_prefill_threshold(),
            sample_rate: default_sample_rate(),
            num_codebooks: default_num_codebooks(),
            streaming_chunk_size: default_streaming_chunk_size(),
            backend: default_backend_kind(),
            num_threads: default_num_threads(),
            enable_preemption: default_enable_preemption(),
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
    pub fn cuda_incremental_chat_enabled(&self) -> bool {
        self.backend == BackendKind::Cuda && self.enable_cuda_incremental_chat
    }

    pub(crate) fn effective_chunked_prefill(&self) -> bool {
        self.enable_chunked_prefill || self.cuda_incremental_chat_enabled()
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
        self.max_blocks = self.max_blocks.max(required_pages);
    }

    pub fn resolved_kv_cache_policy(&self) -> Result<ResolvedKvCachePolicy> {
        resolve_kv_cache_policy(
            self.block_size,
            &self.kv_cache_dtype,
            self.enable_prefix_caching,
            self.managed_prefix_cache_salt.as_deref(),
            self.max_prefix_cache_pages,
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
        resolve_backend_model_context, tts_explicit_output_limit, EngineCoreConfig,
        CUDA_MAX_NATIVE_CONTEXT_TOKENS,
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
