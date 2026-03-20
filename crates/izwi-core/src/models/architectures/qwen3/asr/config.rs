//! Configuration parsing for the retained Qwen speech/aligner stack.

use serde::Deserialize;

use crate::models::architectures::qwen3::core::Qwen3Config;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3AsrConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub quantization: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
    pub thinker_config: ThinkerConfig,
    #[serde(default)]
    pub support_languages: Vec<String>,
    /// Timestamp token ID for forced alignment (present in ForcedAligner models)
    #[serde(default)]
    pub timestamp_token_id: Option<u32>,
    /// Timestamp segment time in ms (present in ForcedAligner models)
    #[serde(default)]
    pub timestamp_segment_time: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub classify_num: Option<usize>,
    pub audio_config: AudioConfig,
    pub text_config: Qwen3Config,
    #[serde(default)]
    pub audio_start_token_id: Option<u32>,
    #[serde(default)]
    pub audio_end_token_id: Option<u32>,
    #[serde(default)]
    pub audio_token_id: Option<u32>,
    #[serde(default)]
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_layers: usize,
    pub num_mel_bins: usize,
    pub downsample_hidden_size: usize,
    pub output_dim: usize,
    #[serde(default)]
    pub conv_chunksize: Option<usize>,
    #[serde(default)]
    pub n_window: Option<usize>,
    #[serde(default)]
    pub n_window_infer: Option<usize>,
}
