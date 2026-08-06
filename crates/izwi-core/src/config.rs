//! Configuration types for the Izwi TTS engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::backends::{BackendPreference, BackendRouter};
use crate::{Error, Result};

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
    /// Directory to store downloaded models
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Maximum batch size for inference
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Maximum sequence length (tokens)
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,

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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            max_batch_size: default_max_batch_size(),
            max_sequence_length: default_max_sequence_length(),
            chunk_size: default_chunk_size(),
            kv_cache_dtype: default_kv_cache_dtype(),
            kv_page_size: default_kv_page_size(),
            backend: default_backend_preference(),
            num_threads: default_num_threads(),
            enable_prefix_caching: default_enable_prefix_caching(),
            managed_prefix_cache_salt: default_managed_prefix_cache_salt(),
            max_prefix_cache_pages: default_max_prefix_cache_pages(),
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

fn default_max_batch_size() -> usize {
    8
}

fn default_max_sequence_length() -> usize {
    4096
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
            self.max_sequence_length,
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
    use super::{EngineConfig, KvCacheDtype, PrefixCachePolicy};

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
            max_sequence_length: 512,
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
}
