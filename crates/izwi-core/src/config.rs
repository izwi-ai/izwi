//! Configuration types for the Izwi TTS engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::backends::BackendPreference;

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
        }
    }
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
    std::env::var("IZWI_KV_PAGE_SIZE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64)
}

fn default_backend_preference() -> BackendPreference {
    if let Ok(raw) = std::env::var("IZWI_BACKEND") {
        if let Some(parsed) = BackendPreference::parse(&raw) {
            return parsed;
        }
    }

    if let Ok(raw) = std::env::var("IZWI_USE_METAL") {
        let value = raw.trim().to_ascii_lowercase();
        if matches!(value.as_str(), "1" | "true" | "yes" | "on") {
            return BackendPreference::Metal;
        }
    }

    BackendPreference::Auto
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
