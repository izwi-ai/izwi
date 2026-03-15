//! Engine configuration types.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::scheduler::SchedulingPolicy;
use crate::backends::{BackendKind, BackendPreference, BackendRouter, BackendSelectionSource};

/// Configuration for the engine core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreConfig {
    /// Directory containing models
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Maximum batch size for inference
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Maximum sequence length (tokens)
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Maximum number of tokens per step (token budget)
    #[serde(default = "default_max_tokens_per_step")]
    pub max_tokens_per_step: usize,

    /// Block size for KV cache paged attention
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// KV cache storage dtype hint (float16, float32, int8, ...).
    #[serde(default = "default_kv_cache_dtype")]
    pub kv_cache_dtype: String,

    /// Maximum number of KV cache blocks
    #[serde(default = "default_max_blocks")]
    pub max_blocks: usize,

    /// Scheduling policy
    #[serde(default)]
    pub scheduling_policy: SchedulingPolicy,

    /// Enable chunked prefill for long prompts
    #[serde(default = "default_chunked_prefill")]
    pub enable_chunked_prefill: bool,

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

    /// Enable request preemption when KV cache is full
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
    /// Enable KV residency tiering hints during scheduling.
    #[serde(default = "default_enable_kv_tiering")]
    pub enable_kv_tiering: bool,
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
fn default_max_seq_len() -> usize {
    4096
}
fn default_max_tokens_per_step() -> usize {
    384
}
fn default_block_size() -> usize {
    16
}
fn default_kv_cache_dtype() -> String {
    "float16".to_string()
}
fn default_max_blocks() -> usize {
    1024
}
fn default_chunked_prefill() -> bool {
    true
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
    true
}
fn default_enable_adaptive_batching() -> bool {
    true
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
    true
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
    true
}
fn default_max_decode_tokens_per_request() -> usize {
    2
}
fn default_enable_kv_tiering() -> bool {
    true
}

impl Default for EngineCoreConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            max_batch_size: default_max_batch_size(),
            max_seq_len: default_max_seq_len(),
            max_tokens_per_step: default_max_tokens_per_step(),
            block_size: default_block_size(),
            kv_cache_dtype: default_kv_cache_dtype(),
            max_blocks: default_max_blocks(),
            scheduling_policy: SchedulingPolicy::default(),
            enable_chunked_prefill: default_chunked_prefill(),
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
            enable_kv_tiering: default_enable_kv_tiering(),
        }
    }
}

impl EngineCoreConfig {
    /// Create config for Qwen3-TTS model
    pub fn for_qwen3_tts() -> Self {
        Self {
            sample_rate: 24000,
            num_codebooks: 8,
            ..Default::default()
        }
    }

    /// Calculate memory required for KV cache
    pub fn kv_cache_memory_bytes(&self) -> usize {
        // Approximate: 2 (K+V) * block_size * hidden_dim * num_layers * dtype_size
        // Using typical values for audio models
        let hidden_dim = 1024;
        let num_layers = 24;
        let requested_dtype_bytes = match self.kv_cache_dtype.trim().to_ascii_lowercase().as_str() {
            "float32" | "f32" => 4,
            "int8" | "i8" | "q8" | "q8_0" => 1,
            _ => 2,
        };
        let dtype_bytes = if self.backend == BackendKind::Metal && requested_dtype_bytes != 1 {
            4
        } else {
            requested_dtype_bytes
        };

        self.max_blocks * self.block_size * hidden_dim * num_layers * 2 * dtype_bytes
    }
}
