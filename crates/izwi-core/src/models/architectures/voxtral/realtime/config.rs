//! Configuration for Voxtral Realtime model.

use serde::Deserialize;

use crate::models::architectures::qwen3::core::Qwen3Config;

/// Voxtral params.json configuration (Mistral format)
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    /// Text model dimension (hidden_size)
    #[serde(rename = "dim")]
    pub text_dim: usize,
    /// Text model number of layers
    #[serde(rename = "n_layers")]
    pub text_n_layers: usize,
    pub head_dim: usize,
    /// Text model hidden dimension (intermediate_size)
    #[serde(rename = "hidden_dim")]
    pub text_hidden_dim: usize,
    /// Text model number of attention heads
    #[serde(rename = "n_heads")]
    pub text_n_heads: usize,
    /// Text model number of key/value heads
    #[serde(rename = "n_kv_heads")]
    pub text_n_kv_heads: usize,
    pub use_biases: bool,
    pub causal: bool,
    pub rope_theta: f64,
    #[serde(rename = "norm_eps")]
    pub norm_eps: f64,
    pub vocab_size: usize,
    pub model_parallel: usize,
    pub tied_embeddings: bool,
    pub sliding_window: usize,
    pub model_max_length: usize,
    /// Multimodal configuration (contains audio encoder)
    pub multimodal: MultimodalConfig,
    pub ada_rms_norm_t_cond: bool,
    pub ada_rms_norm_t_cond_dim: usize,
}

impl VoxtralConfig {
    /// Get text config as MistralConfig
    pub fn text_config(&self) -> MistralConfig {
        MistralConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.text_dim,
            intermediate_size: self.text_hidden_dim,
            num_hidden_layers: self.text_n_layers,
            num_attention_heads: self.text_n_heads,
            num_key_value_heads: self.text_n_kv_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.model_max_length,
            rms_norm_eps: self.norm_eps,
            rope_theta: self.rope_theta as f32,
            tie_word_embeddings: self.tied_embeddings,
            sliding_window: self.sliding_window,
            use_sliding_window: self.sliding_window > 0,
            ada_rms_norm_t_cond: self.ada_rms_norm_t_cond,
            ada_rms_norm_t_cond_dim: self.ada_rms_norm_t_cond_dim,
        }
    }

    /// Get audio encoder config
    pub fn audio_config(&self) -> AudioEncoderConfig {
        let whisper = &self.multimodal.whisper_model_args.encoder_args;
        AudioEncoderConfig {
            d_model: whisper.dim,
            encoder_layers: whisper.n_layers,
            encoder_attention_heads: whisper.n_heads,
            encoder_ffn_dim: whisper.hidden_dim,
            num_mel_bins: whisper.audio_encoding_args.num_mel_bins,
            max_source_positions: whisper.max_source_positions.unwrap_or(1500),
            window_size: whisper.audio_encoding_args.window_size,
            hop_length: whisper.audio_encoding_args.hop_length,
            sampling_rate: whisper.audio_encoding_args.sampling_rate,
            is_causal: whisper.causal,
            conv1_kernel_size: 3,
            conv1_stride: 1,
            conv2_kernel_size: 3,
            conv2_stride: 2,
            global_log_mel_max: whisper.audio_encoding_args.global_log_mel_max,
            head_dim: whisper.head_dim,
        }
    }

    /// Get downsample factor
    pub fn downsample_factor(&self) -> usize {
        self.multimodal
            .whisper_model_args
            .downsample_args
            .downsample_factor
    }

    /// Get frame rate
    pub fn frame_rate(&self) -> f32 {
        self.multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args
            .frame_rate
    }

    /// Get number of delay tokens
    pub fn num_delay_tokens(&self) -> usize {
        6
    }

    /// Get block pool size
    pub fn block_pool_size(&self) -> usize {
        // Default pooling factor
        4
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultimodalConfig {
    #[serde(rename = "whisper_model_args")]
    pub whisper_model_args: WhisperModelArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WhisperModelArgs {
    #[serde(rename = "encoder_args")]
    pub encoder_args: WhisperEncoderArgs,
    #[serde(rename = "downsample_args")]
    pub downsample_args: DownsampleArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WhisperEncoderArgs {
    #[serde(rename = "audio_encoding_args")]
    pub audio_encoding_args: AudioEncodingArgs,
    pub dim: usize,
    #[serde(rename = "n_layers")]
    pub n_layers: usize,
    pub head_dim: usize,
    #[serde(rename = "hidden_dim")]
    pub hidden_dim: usize,
    #[serde(rename = "n_heads")]
    pub n_heads: usize,
    pub vocab_size: usize,
    #[serde(rename = "n_kv_heads")]
    pub n_kv_heads: usize,
    pub use_biases: bool,
    pub use_cache: bool,
    pub rope_theta: f64,
    pub causal: bool,
    pub norm_eps: f64,
    pub pos_embed: String,
    pub max_source_positions: Option<usize>,
    pub ffn_type: String,
    pub norm_type: String,
    pub sliding_window: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncodingArgs {
    pub sampling_rate: usize,
    pub frame_rate: f32,
    pub num_mel_bins: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub chunk_length_s: Option<f32>,
    pub global_log_mel_max: Option<f32>,
    pub transcription_format: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DownsampleArgs {
    pub downsample_factor: usize,
}

/// Mistral text model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub sliding_window: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond_dim: usize,
}

/// Whisper-based audio encoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub window_size: usize,
    pub hop_length: usize,
    pub sampling_rate: usize,
    #[serde(default)]
    pub is_causal: bool,
    pub conv1_kernel_size: usize,
    pub conv1_stride: usize,
    pub conv2_kernel_size: usize,
    pub conv2_stride: usize,
    #[serde(default = "default_global_log_mel_max")]
    pub global_log_mel_max: Option<f32>,
    #[serde(default)]
    pub head_dim: usize,
}

fn default_global_log_mel_max() -> Option<f32> {
    None
}

impl From<MistralConfig> for Qwen3Config {
    fn from(cfg: MistralConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_attention_heads: cfg.num_attention_heads,
            num_hidden_layers: cfg.num_hidden_layers,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: if cfg.head_dim > 0 {
                Some(cfg.head_dim)
            } else {
                None
            },
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta as f64,
            vocab_size: cfg.vocab_size,
            lm_head_size: None,
            tie_word_embeddings: cfg.tie_word_embeddings,
            rope_scaling: None,
            sliding_window: cfg.use_sliding_window.then_some(cfg.sliding_window),
            use_sliding_window: cfg.use_sliding_window,
            ada_rms_norm_t_cond: cfg.ada_rms_norm_t_cond,
            ada_rms_norm_t_cond_dim: cfg.ada_rms_norm_t_cond_dim,
        }
    }
}
