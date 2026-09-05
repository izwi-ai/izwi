//! Configuration types for the Izwi TTS engine

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::str::FromStr;

use crate::backends::{BackendKind, BackendPreference, BackendRouter};
use crate::{Error, Result};

/// Operator intent for selecting a model's maximum context length.
///
/// Automatic context is resolved only after the backend, model geometry, and
/// available physical memory are known. Positive numeric configuration values
/// retain the historical meaning of an explicit fixed token limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextLengthPreference {
    #[default]
    Auto,
    Explicit(NonZeroUsize),
}

impl ContextLengthPreference {
    pub fn explicit(tokens: usize) -> Result<Self> {
        NonZeroUsize::new(tokens)
            .map(Self::Explicit)
            .ok_or_else(|| Error::ConfigError("context length must be greater than zero".into()))
    }

    pub const fn explicit_tokens(self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::Explicit(tokens) => Some(tokens.get()),
        }
    }
}

impl fmt::Display for ContextLengthPreference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::Explicit(tokens) => tokens.fmt(formatter),
        }
    }
}

impl FromStr for ContextLengthPreference {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let tokens = value.parse::<usize>().map_err(|_| {
            Error::ConfigError(format!(
                "invalid context length {value:?}; expected 'auto' or a positive integer"
            ))
        })?;
        Self::explicit(tokens)
    }
}

impl Serialize for ContextLengthPreference {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Explicit(tokens) => serializer.serialize_u64(tokens.get() as u64),
        }
    }
}

impl<'de> Deserialize<'de> for ContextLengthPreference {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ContextLengthVisitor;

        impl serde::de::Visitor<'_> for ContextLengthVisitor {
            type Value = ContextLengthPreference;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("\"auto\" or a positive integer context length")
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.trim().eq_ignore_ascii_case("auto") {
                    Ok(ContextLengthPreference::Auto)
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Str(value), &self))
                }
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let value = usize::try_from(value)
                    .map_err(|_| E::invalid_value(serde::de::Unexpected::Unsigned(value), &self))?;
                NonZeroUsize::new(value)
                    .map(ContextLengthPreference::Explicit)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Unsigned(0), &self))
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let value = u64::try_from(value)
                    .map_err(|_| E::invalid_value(serde::de::Unexpected::Signed(value), &self))?;
                self.visit_u64(value)
            }
        }

        deserializer.deserialize_any(ContextLengthVisitor)
    }
}

/// Operator intent for the maximum width of one physical tensor invocation.
///
/// This is deliberately independent from scheduler, retained-session, staged
/// transaction, and request-queue capacities. Automatic width is resolved only
/// after the execution backend is known; positive numeric values retain the
/// historical meaning of an explicit fixed width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchSizePreference {
    #[default]
    Auto,
    Fixed(NonZeroUsize),
}

impl BatchSizePreference {
    pub fn fixed(rows: usize) -> Result<Self> {
        NonZeroUsize::new(rows)
            .map(Self::Fixed)
            .ok_or_else(|| Error::ConfigError("batch size must be greater than zero".into()))
    }

    pub const fn fixed_rows(self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::Fixed(rows) => Some(rows.get()),
        }
    }

    /// Resolve the effective native tensor width for a concrete backend.
    pub const fn resolve(self, backend: BackendKind) -> usize {
        match self {
            Self::Fixed(rows) => rows.get(),
            Self::Auto => match backend {
                // Both portable backends have complete width-two chat kernels;
                // keep the conservative hard cap while enabling useful
                // continuous tensor batching by default.
                BackendKind::Cpu | BackendKind::Metal => 2,
                BackendKind::Cuda => 8,
            },
        }
    }
}

impl fmt::Display for BatchSizePreference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::Fixed(rows) => rows.fmt(formatter),
        }
    }
}

impl FromStr for BatchSizePreference {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let rows = value.parse::<usize>().map_err(|_| {
            Error::ConfigError(format!(
                "invalid batch size {value:?}; expected 'auto' or a positive integer"
            ))
        })?;
        Self::fixed(rows)
    }
}

impl Serialize for BatchSizePreference {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Fixed(rows) => serializer.serialize_u64(rows.get() as u64),
        }
    }
}

impl<'de> Deserialize<'de> for BatchSizePreference {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BatchSizeVisitor;

        impl serde::de::Visitor<'_> for BatchSizeVisitor {
            type Value = BatchSizePreference;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("\"auto\" or a positive integer physical batch width")
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.trim().eq_ignore_ascii_case("auto") {
                    Ok(BatchSizePreference::Auto)
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Str(value), &self))
                }
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let value = usize::try_from(value)
                    .map_err(|_| E::invalid_value(serde::de::Unexpected::Unsigned(value), &self))?;
                NonZeroUsize::new(value)
                    .map(BatchSizePreference::Fixed)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Unsigned(0), &self))
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let value = u64::try_from(value)
                    .map_err(|_| E::invalid_value(serde::de::Unexpected::Signed(value), &self))?;
                self.visit_u64(value)
            }
        }

        deserializer.deserialize_any(BatchSizeVisitor)
    }
}

/// Rollout mode for overlapping distinct physical inference launches.
///
/// This does not enable or disable inference. `Serial` retains the current
/// one-launch-at-a-time behavior, `Shadow` may evaluate concurrent dispatch
/// decisions while still launching serially, and `Concurrent` permits the
/// resolved physical launch limit to exceed one. Tensor batch width, scheduler
/// rows, retained sessions, and queued requests are independent capacities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalExecutionMode {
    #[default]
    Serial,
    Shadow,
    Concurrent,
}

impl fmt::Display for PhysicalExecutionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Serial => "serial",
            Self::Shadow => "shadow",
            Self::Concurrent => "concurrent",
        })
    }
}

impl FromStr for PhysicalExecutionMode {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "serial" => Ok(Self::Serial),
            "shadow" => Ok(Self::Shadow),
            "concurrent" => Ok(Self::Concurrent),
            _ => Err(Error::ConfigError(format!(
                "invalid physical execution mode {value:?}; expected 'serial', 'shadow', or 'concurrent'"
            ))),
        }
    }
}

/// Non-zero engine-wide ceiling for simultaneously owned physical launches.
///
/// A distinct type prevents this capacity from being confused with the width
/// of one tensor invocation or with any logical scheduler/session capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct PhysicalInFlightLimit(NonZeroUsize);

impl PhysicalInFlightLimit {
    /// Operational ceiling well below Tokio's semaphore panic boundary. A
    /// physical launch count above this value is a configuration error rather
    /// than a meaningful inference setting.
    pub const MAX: usize = 1024;

    pub fn new(limit: usize) -> Result<Self> {
        let limit = NonZeroUsize::new(limit).ok_or_else(|| {
            Error::ConfigError("max physical in-flight must be greater than zero".into())
        })?;
        if limit.get() > Self::MAX {
            return Err(Error::ConfigError(format!(
                "max physical in-flight {} exceeds the supported limit {}",
                limit,
                Self::MAX
            )));
        }
        Ok(Self(limit))
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for PhysicalInFlightLimit {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let limit = usize::deserialize(deserializer)?;
        Self::new(limit).map_err(serde::de::Error::custom)
    }
}

impl Default for PhysicalInFlightLimit {
    fn default() -> Self {
        Self(NonZeroUsize::MIN)
    }
}

impl fmt::Display for PhysicalInFlightLimit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for PhysicalInFlightLimit {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let limit = value.trim().parse::<usize>().map_err(|_| {
            Error::ConfigError(format!(
                "invalid max physical in-flight {value:?}; expected a positive integer"
            ))
        })?;
        Self::new(limit)
    }
}

/// Resolved launch capacities for dispatch planning and physical execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalExecutionCapacity {
    /// Candidate dispatches the rollout may evaluate. Shadow mode can exceed
    /// one here without permitting overlapping model calls.
    pub candidate_dispatch_limit: PhysicalInFlightLimit,
    /// Physical model calls that may actually overlap. Only Concurrent mode
    /// can resolve this axis above one.
    pub physical_launch_limit: PhysicalInFlightLimit,
}

impl PhysicalExecutionMode {
    pub fn resolve_capacity(
        self,
        configured_limit: PhysicalInFlightLimit,
    ) -> PhysicalExecutionCapacity {
        let serial = PhysicalInFlightLimit::default();
        match self {
            Self::Serial => PhysicalExecutionCapacity {
                candidate_dispatch_limit: serial,
                physical_launch_limit: serial,
            },
            Self::Shadow => PhysicalExecutionCapacity {
                candidate_dispatch_limit: configured_limit,
                physical_launch_limit: serial,
            },
            Self::Concurrent => PhysicalExecutionCapacity {
                candidate_dispatch_limit: configured_limit,
                physical_launch_limit: configured_limit,
            },
        }
    }
}

/// Requested storage dtype for retained KV state.
///
/// Quantized variants remain deserializable so old configuration files fail
/// with an actionable startup error instead of an opaque enum parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheDtype {
    #[serde(alias = "fp16", alias = "f16")]
    Float16,
    #[serde(alias = "bf16")]
    Bfloat16,
    #[serde(alias = "fp32", alias = "f32")]
    Float32,
    Int8,
    #[serde(alias = "int4")]
    Q4,
}

impl KvCacheDtype {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "float16" | "fp16" | "f16" => Ok(Self::Float16),
            "bfloat16" | "bf16" => Ok(Self::Bfloat16),
            "float32" | "fp32" | "f32" => Ok(Self::Float32),
            "int8" => Ok(Self::Int8),
            "q4" | "int4" => Ok(Self::Q4),
            value => Err(Error::ConfigError(format!(
                "unsupported kv_cache_dtype `{value}`; expected float16, bfloat16, float32, int8, or q4"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Bfloat16 => "bfloat16",
            Self::Float32 => "float32",
            Self::Int8 => "int8",
            Self::Q4 => "q4",
        }
    }

    const fn is_production_supported(self) -> bool {
        matches!(self, Self::Float16 | Self::Bfloat16 | Self::Float32)
    }
}

impl std::fmt::Display for KvCacheDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Prefix-sharing intent after configuration parsing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PrefixCachePolicy {
    Disabled,
    Namespaced {
        /// Operational isolation value. Presence and mode are reported, but
        /// the raw tenant/deployment namespace is never serialized to health
        /// or diagnostics responses.
        #[serde(skip_serializing)]
        namespace: String,
        max_pages: usize,
    },
}

/// Cache policy exactly as requested by the deployment configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequestedKvCachePolicy {
    pub page_size: usize,
    pub dtype: KvCacheDtype,
    pub prefix: PrefixCachePolicy,
}

/// Cache policy that the runtime will actually enforce.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EffectiveKvCachePolicy {
    pub page_size: usize,
    pub dtype: KvCacheDtype,
    pub prefix: PrefixCachePolicy,
}

/// Requested/effective cache truth, including any safe capacity clamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedKvCachePolicy {
    pub requested: RequestedKvCachePolicy,
    pub effective: EffectiveKvCachePolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

/// Resolve and validate the cache policy at the configuration boundary.
pub(crate) fn resolve_kv_cache_policy(
    page_size: usize,
    dtype: &str,
    enable_prefix_caching: bool,
    prefix_namespace: Option<&str>,
    max_prefix_cache_pages: usize,
    total_capacity_pages: usize,
    max_sequence_length: usize,
) -> Result<ResolvedKvCachePolicy> {
    if page_size == 0 {
        return Err(Error::ConfigError(
            "kv_page_size/block_size must be greater than zero".to_string(),
        ));
    }
    let dtype = KvCacheDtype::parse(dtype)?;
    if !dtype.is_production_supported() {
        return Err(Error::ConfigError(format!(
            "kv_cache_dtype `{dtype}` is not production-ready; use float16, bfloat16, or float32 until quantized KV storage and model kernels are certified"
        )));
    }

    let prefix = if enable_prefix_caching {
        let namespace = prefix_namespace
            .map(str::trim)
            .filter(|namespace| !namespace.is_empty())
            .ok_or_else(|| {
                Error::ConfigError(
                    "enable_prefix_caching=true requires an explicit non-empty managed_prefix_cache_salt namespace"
                        .to_string(),
                )
            })?;
        if max_prefix_cache_pages == 0 {
            return Err(Error::ConfigError(
                "enable_prefix_caching=true requires max_prefix_cache_pages greater than zero"
                    .to_string(),
            ));
        }
        PrefixCachePolicy::Namespaced {
            namespace: namespace.to_string(),
            max_pages: max_prefix_cache_pages,
        }
    } else {
        PrefixCachePolicy::Disabled
    };
    let requested = RequestedKvCachePolicy {
        page_size,
        dtype,
        prefix,
    };

    let request_reserve_pages =
        max_sequence_length.max(1).saturating_add(page_size - 1) / page_size;
    let prefix_capacity = total_capacity_pages.saturating_sub(request_reserve_pages);
    let (effective_prefix, fallback_reason) = match &requested.prefix {
        PrefixCachePolicy::Disabled => (PrefixCachePolicy::Disabled, None),
        PrefixCachePolicy::Namespaced {
            namespace,
            max_pages,
        } => {
            let effective_pages = (*max_pages).min(prefix_capacity);
            if effective_pages == 0 {
                return Err(Error::ConfigError(format!(
                    "prefix cache has no safe page budget: capacity_pages={total_capacity_pages}, reserved_request_pages={request_reserve_pages}"
                )));
            }
            let fallback_reason = (effective_pages != *max_pages).then(|| {
                format!(
                    "prefix page budget clamped from {max_pages} to {effective_pages} to reserve {request_reserve_pages} request pages"
                )
            });
            (
                PrefixCachePolicy::Namespaced {
                    namespace: namespace.clone(),
                    max_pages: effective_pages,
                },
                fallback_reason,
            )
        }
    };

    Ok(ResolvedKvCachePolicy {
        effective: EffectiveKvCachePolicy {
            page_size,
            dtype,
            prefix: effective_prefix,
        },
        requested,
        fallback_reason,
    })
}

/// Main engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Performance policy, resolved once before model loading.
    #[serde(default)]
    pub performance: crate::performance::PerformanceConfig,
    /// Directory to store downloaded models
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Maximum simultaneously resident model variants. `None` leaves model
    /// residency unbounded for embedded runtimes; server profiles resolve an
    /// explicit value.
    #[serde(default)]
    pub max_loaded_models: Option<usize>,

    /// Automatic or explicitly fixed physical tensor batch width.
    #[serde(default)]
    pub max_batch_size: BatchSizePreference,

    /// Rollout mode for overlapping separate physical inference launches.
    #[serde(default)]
    pub physical_execution_mode: PhysicalExecutionMode,

    /// Prospective engine-wide physical launch ceiling. Serial and Shadow
    /// modes still resolve actual physical execution to one in-flight launch.
    #[serde(default)]
    pub max_physical_in_flight: PhysicalInFlightLimit,

    /// Maximum logical rows selected by one scheduler step.
    #[serde(default = "default_max_scheduler_batch_size")]
    pub max_scheduler_batch_size: usize,

    /// Maximum retained sequence/session rows in managed model state.
    #[serde(default = "default_max_retained_sequences")]
    pub max_retained_sequences: usize,

    /// Maximum simultaneously staged managed-state transactions.
    #[serde(default = "default_max_staged_transactions")]
    pub max_staged_transactions: usize,

    /// Maximum admitted jobs in the runtime inference queue.
    #[serde(default = "default_max_queued_requests")]
    pub max_queued_requests: usize,

    /// Automatic or explicitly fixed maximum sequence length.
    #[serde(default)]
    pub max_sequence_length: ContextLengthPreference,

    /// Memory kept outside fitted model/state plans for allocator and backend
    /// command-buffer overhead.
    #[serde(default = "default_portable_context_reserve_bytes")]
    pub portable_context_reserve_bytes: u64,

    /// Chunk size for streaming (in audio tokens)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Data type for KV cache
    #[serde(default = "default_kv_cache_dtype")]
    pub kv_cache_dtype: String,

    /// Number of tokens per KV page for decode-time paged cache.
    #[serde(default = "default_kv_page_size")]
    pub kv_page_size: usize,

    /// Preferred backend selection strategy.
    #[serde(default = "default_backend_preference")]
    pub backend: BackendPreference,

    /// Number of threads for CPU operations
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,

    /// Enable committed managed-prefix reuse.
    #[serde(default = "default_enable_prefix_caching")]
    pub enable_prefix_caching: bool,

    /// Deployment/tenant namespace salt for managed physical prefix pages.
    /// Required explicitly when prefix caching is enabled.
    #[serde(default = "default_managed_prefix_cache_salt")]
    pub managed_prefix_cache_salt: Option<String>,

    /// Hard upper bound for committed prefix pages. This is additionally
    /// clamped to preserve capacity for at least one maximum-length request.
    #[serde(default = "default_max_prefix_cache_pages")]
    pub max_prefix_cache_pages: usize,

    /// Enable scheduler-level chunked prefill so long prompts are admitted in
    /// bounded token quanta instead of one monolithic prefill transaction.
    #[serde(default = "default_enable_chunked_prefill")]
    pub enable_chunked_prefill: bool,

    /// Prompt length (tokens) above which chunked prefill applies.
    #[serde(default = "default_chunked_prefill_threshold")]
    pub chunked_prefill_threshold: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            performance: Default::default(),
            models_dir: default_models_dir(),
            max_loaded_models: None,
            max_batch_size: BatchSizePreference::Auto,
            physical_execution_mode: PhysicalExecutionMode::Serial,
            max_physical_in_flight: PhysicalInFlightLimit::default(),
            max_scheduler_batch_size: default_max_scheduler_batch_size(),
            max_retained_sequences: default_max_retained_sequences(),
            max_staged_transactions: default_max_staged_transactions(),
            max_queued_requests: default_max_queued_requests(),
            max_sequence_length: ContextLengthPreference::Auto,
            portable_context_reserve_bytes: default_portable_context_reserve_bytes(),
            chunk_size: default_chunk_size(),
            kv_cache_dtype: default_kv_cache_dtype(),
            kv_page_size: default_kv_page_size(),
            backend: default_backend_preference(),
            num_threads: default_num_threads(),
            enable_prefix_caching: default_enable_prefix_caching(),
            managed_prefix_cache_salt: default_managed_prefix_cache_salt(),
            max_prefix_cache_pages: default_max_prefix_cache_pages(),
            enable_chunked_prefill: default_enable_chunked_prefill(),
            chunked_prefill_threshold: default_chunked_prefill_threshold(),
        }
    }
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

fn default_enable_chunked_prefill() -> bool {
    false
}

fn default_chunked_prefill_threshold() -> usize {
    192
}

fn default_models_dir() -> PathBuf {
    if let Ok(from_env) = std::env::var("IZWI_MODELS_DIR") {
        let trimmed = from_env.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

fn default_max_scheduler_batch_size() -> usize {
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

fn default_portable_context_reserve_bytes() -> u64 {
    1024 * 1024 * 1024
}

fn default_chunk_size() -> usize {
    128
}

fn default_kv_cache_dtype() -> String {
    "float16".to_string()
}

fn default_kv_page_size() -> usize {
    64
}

impl EngineConfig {
    /// Resolve rollout-aware dispatch and physical-launch capacity axes.
    pub fn resolved_physical_execution_capacity(&self) -> PhysicalExecutionCapacity {
        self.physical_execution_mode
            .resolve_capacity(self.max_physical_in_flight)
    }

    /// Numeric portable ceiling used until model load resolves an automatic
    /// context against the concrete memory plan. Explicit operator intent is
    /// preserved; Auto starts from the historical safe portable baseline.
    pub(crate) fn portable_context_ceiling(&self) -> usize {
        self.max_sequence_length.explicit_tokens().unwrap_or(4096)
    }

    /// Validate cache settings and report requested versus effective policy.
    pub fn resolved_kv_cache_policy(
        &self,
        total_capacity_pages: usize,
    ) -> Result<ResolvedKvCachePolicy> {
        resolve_kv_cache_policy(
            self.kv_page_size,
            &self.kv_cache_dtype,
            self.enable_prefix_caching,
            self.managed_prefix_cache_salt.as_deref(),
            self.max_prefix_cache_pages,
            total_capacity_pages,
            // Portable automatic context is resolved during model loading. Keep
            // the historical reserve until that effective value is available.
            self.portable_context_ceiling(),
        )
    }
}

fn default_backend_preference() -> BackendPreference {
    BackendRouter::env_preference().unwrap_or(BackendPreference::Auto)
}

fn default_num_threads() -> usize {
    get_num_cpus().min(8)
}

/// Model-specific configuration from config.json (Qwen3-TTS format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub architectures: Vec<String>,

    #[serde(default)]
    pub model_type: Option<String>,

    #[serde(default)]
    pub tts_bos_token_id: Option<usize>,

    #[serde(default)]
    pub tts_eos_token_id: Option<usize>,

    #[serde(default)]
    pub tts_pad_token_id: Option<usize>,

    #[serde(default)]
    pub talker_config: Option<TalkerConfig>,

    #[serde(default)]
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TalkerConfig {
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub text_vocab_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub num_code_groups: usize,
    #[serde(default)]
    pub code_predictor_config: Option<CodePredictorConfig>,
}

fn default_rope_theta() -> f64 {
    1000000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodePredictorConfig {
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_code_groups: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpeakerEncoderConfig {
    #[serde(default)]
    pub enc_dim: usize,
    #[serde(default)]
    pub sample_rate: usize,
}

impl ModelConfig {
    /// Get the hidden size from talker_config
    pub fn hidden_size(&self) -> usize {
        self.talker_config
            .as_ref()
            .map(|c| c.hidden_size)
            .unwrap_or(1024)
    }

    /// Get the number of hidden layers from talker_config
    pub fn num_hidden_layers(&self) -> usize {
        self.talker_config
            .as_ref()
            .map(|c| c.num_hidden_layers)
            .unwrap_or(28)
    }

    /// Get the vocab size from talker_config
    pub fn vocab_size(&self) -> usize {
        self.talker_config
            .as_ref()
            .map(|c| c.text_vocab_size)
            .unwrap_or(151936)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["Qwen3TTSForConditionalGeneration".to_string()],
            model_type: Some("qwen3_tts".to_string()),
            tts_bos_token_id: Some(151672),
            tts_eos_token_id: Some(151673),
            tts_pad_token_id: Some(151671),
            talker_config: Some(TalkerConfig::default()),
            speaker_encoder_config: Some(SpeakerEncoderConfig::default()),
        }
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_cors_enabled")]
    pub cors_enabled: bool,

    #[serde(default)]
    pub cors_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            cors_enabled: default_cors_enabled(),
            cors_origins: vec!["*".to_string()],
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_cors_enabled() -> bool {
    true
}

fn get_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod managed_kv_default_tests {
    use super::{
        BatchSizePreference, ContextLengthPreference, EngineConfig, KvCacheDtype,
        PhysicalExecutionMode, PhysicalInFlightLimit, PrefixCachePolicy,
    };
    use crate::backends::BackendKind;

    #[test]
    fn physical_batch_defaults_are_backend_aware() {
        let preference = EngineConfig::default().max_batch_size;
        assert_eq!(preference, BatchSizePreference::Auto);
        assert_eq!(preference.resolve(BackendKind::Cpu), 2);
        assert_eq!(preference.resolve(BackendKind::Metal), 2);
        assert_eq!(preference.resolve(BackendKind::Cuda), 8);
    }

    #[test]
    fn fixed_physical_batch_provenance_survives_configuration_parsing() {
        let automatic: EngineConfig = serde_json::from_str(r#"{"max_batch_size":"auto"}"#).unwrap();
        assert_eq!(automatic.max_batch_size, BatchSizePreference::Auto);

        let fixed: EngineConfig = serde_json::from_str(r#"{"max_batch_size":4}"#).unwrap();
        assert_eq!(fixed.max_batch_size.fixed_rows(), Some(4));
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(fixed.max_batch_size.resolve(backend), 4);
        }

        let serialized = serde_json::to_value(fixed).unwrap();
        assert_eq!(serialized["max_batch_size"], 4);
    }

    #[test]
    fn physical_and_logical_capacity_defaults_are_independent() {
        let config = EngineConfig::default();
        assert_eq!(config.max_scheduler_batch_size, 8);
        assert_eq!(config.max_retained_sequences, 8);
        assert_eq!(config.max_staged_transactions, 8);
        assert_eq!(config.max_queued_requests, 128);
        assert_eq!(config.max_batch_size.resolve(BackendKind::Cpu), 2);
        assert_eq!(
            config.physical_execution_mode,
            PhysicalExecutionMode::Serial
        );
        assert_eq!(config.max_physical_in_flight.get(), 1);
        let capacity = config.resolved_physical_execution_capacity();
        assert_eq!(capacity.candidate_dispatch_limit.get(), 1);
        assert_eq!(capacity.physical_launch_limit.get(), 1);
    }

    #[test]
    fn physical_execution_modes_resolve_candidate_and_launch_axes() {
        let configured = PhysicalInFlightLimit::new(4).unwrap();
        let cases = [
            (PhysicalExecutionMode::Serial, 1, 1),
            (PhysicalExecutionMode::Shadow, 4, 1),
            (PhysicalExecutionMode::Concurrent, 4, 4),
        ];

        for (mode, expected_candidates, expected_launches) in cases {
            let capacity = mode.resolve_capacity(configured);
            assert_eq!(capacity.candidate_dispatch_limit.get(), expected_candidates);
            assert_eq!(capacity.physical_launch_limit.get(), expected_launches);
        }
    }

    #[test]
    fn physical_execution_configuration_is_fail_closed() {
        let shadow: EngineConfig = serde_json::from_str(
            r#"{"physical_execution_mode":"shadow","max_physical_in_flight":4}"#,
        )
        .unwrap();
        let capacity = shadow.resolved_physical_execution_capacity();
        assert_eq!(capacity.candidate_dispatch_limit.get(), 4);
        assert_eq!(capacity.physical_launch_limit.get(), 1);

        for invalid in [
            r#"{"physical_execution_mode":"on"}"#,
            r#"{"physical_execution_mode":"off"}"#,
            r#"{"physical_execution_mode":"parallel"}"#,
            r#"{"max_physical_in_flight":0}"#,
            r#"{"max_physical_in_flight":1025}"#,
        ] {
            assert!(
                serde_json::from_str::<EngineConfig>(invalid).is_err(),
                "{invalid}"
            );
        }

        assert!(PhysicalInFlightLimit::new(0).is_err());
        assert!(PhysicalInFlightLimit::new(PhysicalInFlightLimit::MAX + 1).is_err());
    }

    #[test]
    fn managed_prefix_reuse_is_fail_closed_for_normal_runtime_config() {
        let config = EngineConfig::default();
        assert!(!config.enable_prefix_caching);
        assert!(config.managed_prefix_cache_salt.is_none());
        let policy = config.resolved_kv_cache_policy(1024).unwrap();
        assert_eq!(policy.effective.page_size, 64);
        assert_eq!(policy.effective.dtype, KvCacheDtype::Float16);
        assert_eq!(policy.effective.prefix, PrefixCachePolicy::Disabled);
    }

    #[test]
    fn quantized_cache_requests_fail_before_model_readiness() {
        for dtype in [KvCacheDtype::Int8, KvCacheDtype::Q4] {
            let config = EngineConfig {
                kv_cache_dtype: dtype.to_string(),
                ..EngineConfig::default()
            };
            let error = config.resolved_kv_cache_policy(1024).unwrap_err();
            assert!(error.to_string().contains("not production-ready"));
        }
    }

    #[test]
    fn prefix_reuse_requires_namespace_and_preserves_request_capacity() {
        let missing_namespace = EngineConfig {
            enable_prefix_caching: true,
            ..EngineConfig::default()
        };
        assert!(missing_namespace
            .resolved_kv_cache_policy(1024)
            .unwrap_err()
            .to_string()
            .contains("explicit non-empty"));

        let config = EngineConfig {
            max_sequence_length: ContextLengthPreference::explicit(512).unwrap(),
            enable_prefix_caching: true,
            managed_prefix_cache_salt: Some("tenant-a".to_string()),
            max_prefix_cache_pages: 100,
            ..EngineConfig::default()
        };
        let policy = config.resolved_kv_cache_policy(64).unwrap();
        assert_eq!(
            policy.effective.prefix,
            PrefixCachePolicy::Namespaced {
                namespace: "tenant-a".to_string(),
                max_pages: 56,
            }
        );
        assert!(policy.fallback_reason.is_some());
    }

    #[test]
    fn legacy_json_cache_values_parse_then_fail_actionably() {
        let quantized: EngineConfig = serde_json::from_str(r#"{"kv_cache_dtype":"int8"}"#)
            .expect("legacy int8 config should remain parseable");
        assert!(quantized
            .resolved_kv_cache_policy(1024)
            .unwrap_err()
            .to_string()
            .contains("until quantized KV storage"));

        let implicit_namespace: EngineConfig =
            serde_json::from_str(r#"{"enable_prefix_caching":true}"#)
                .expect("legacy prefix flag should remain parseable");
        assert!(implicit_namespace
            .resolved_kv_cache_policy(1024)
            .unwrap_err()
            .to_string()
            .contains("managed_prefix_cache_salt"));
    }

    #[test]
    fn context_length_defaults_to_auto_when_absent() {
        let config: EngineConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.max_sequence_length, ContextLengthPreference::Auto);
    }

    #[test]
    fn context_length_accepts_auto_and_legacy_positive_numbers() {
        let automatic: EngineConfig =
            serde_json::from_str(r#"{"max_sequence_length":"auto"}"#).unwrap();
        assert_eq!(automatic.max_sequence_length, ContextLengthPreference::Auto);

        let explicit: EngineConfig =
            serde_json::from_str(r#"{"max_sequence_length":2048}"#).unwrap();
        assert_eq!(explicit.max_sequence_length.explicit_tokens(), Some(2048));

        let toml: EngineConfig = toml::from_str("max_sequence_length = 8192").unwrap();
        assert_eq!(toml.max_sequence_length.explicit_tokens(), Some(8192));
    }

    #[test]
    fn context_length_serializes_canonically() {
        let automatic = serde_json::to_value(EngineConfig::default()).unwrap();
        assert_eq!(automatic["max_sequence_length"], "auto");

        let explicit = EngineConfig {
            max_sequence_length: ContextLengthPreference::explicit(4096).unwrap(),
            ..EngineConfig::default()
        };
        let explicit = serde_json::to_value(explicit).unwrap();
        assert_eq!(explicit["max_sequence_length"], 4096);
    }

    #[test]
    fn context_length_rejects_invalid_values() {
        for invalid in [
            r#"{"max_sequence_length":0}"#,
            r#"{"max_sequence_length":-1}"#,
            r#"{"max_sequence_length":1.5}"#,
            r#"{"max_sequence_length":"4096"}"#,
            r#"{"max_sequence_length":"native"}"#,
        ] {
            assert!(
                serde_json::from_str::<EngineConfig>(invalid).is_err(),
                "{invalid}"
            );
        }
    }
}
