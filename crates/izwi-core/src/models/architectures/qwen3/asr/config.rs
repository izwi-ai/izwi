//! Configuration parsing for the retained Qwen speech/aligner stack.

use serde::Deserialize;

use crate::catalog::ModelFamily;
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

impl Qwen3AsrConfig {
    /// The forced aligner (a `forced_aligner` thinker or a classification
    /// head) or Qwen3-ASR.
    pub fn model_family(&self) -> ModelFamily {
        let forced_aligner = self
            .thinker_config
            .model_type
            .as_deref()
            .is_some_and(|name| name.to_ascii_lowercase().contains("forced_aligner"))
            || self.thinker_config.classify_num.is_some();
        if forced_aligner {
            ModelFamily::Qwen3ForcedAligner
        } else {
            ModelFamily::Qwen3Asr
        }
    }

    /// An MLX-quantized safetensors checkpoint.
    pub fn is_mlx_quantized(&self) -> bool {
        self.quantization.is_some() || self.quantization_config.is_some()
    }

    /// The MLX quantization settings, `quantization_config` first.
    pub fn mlx_quantization(&self) -> Option<&serde_json::Value> {
        self.quantization_config
            .as_ref()
            .or(self.quantization.as_ref())
    }

    /// The bit width of each packed MLX weight element.
    pub fn mlx_bits(&self) -> Option<u64> {
        self.mlx_quantization()?.get("bits")?.as_u64()
    }
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
