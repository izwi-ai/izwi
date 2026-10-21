//! Configuration parsing for Qwen3-TTS models.

use serde::Deserialize;
use std::collections::HashMap;

/// Main TTS model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub tokenizer_type: String,
    pub tts_model_size: String,
    pub tts_model_type: String,

    /// Special token IDs
    pub assistant_token_id: u32,
    pub im_end_token_id: u32,
    pub im_start_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,

    /// Talker (main LLM) configuration
    pub talker_config: TalkerConfig,

    /// ECAPA x-vector encoder (Base checkpoints).
    #[serde(default)]
    pub speaker_encoder_config: Option<super::speaker_encoder::SpeakerEncoderConfig>,
}

/// Talker model configuration (the main Qwen3-based LLM)
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub text_hidden_size: usize,
    pub num_code_groups: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub use_cache: bool,
    pub position_id_per_seconds: usize,

    /// MRoPE configuration
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Sliding window attention
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Code predictor configuration (for multi-codebook generation)
    pub code_predictor_config: CodePredictorConfig,

    /// Speaker IDs mapping
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,

    /// Speaker dialect mapping
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, serde_json::Value>,

    /// Codec special token IDs
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,

    /// Language ID mapping
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
}

/// Code predictor configuration for multi-codebook RVQ generation
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub num_code_groups: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub use_cache: bool,
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Text hidden size for codec embeddings (may differ from hidden_size in larger models)
    #[serde(default)]
    pub text_hidden_size: Option<usize>,
}

/// RoPE scaling configuration for MRoPE (Multi-modal RoPE)
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    pub rope_type: String,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub interleaved: Option<bool>,
}

impl TalkerConfig {
    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get number of KV groups for GQA
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Check if MRoPE is enabled
    pub fn uses_mrope(&self) -> bool {
        self.rope_scaling
            .as_ref()
            .map(|s| s.interleaved.unwrap_or(false))
            .unwrap_or(false)
    }

    /// Get MRoPE section configuration
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .and_then(|s| s.mrope_section.clone())
            .unwrap_or_else(|| vec![24, 20, 20])
    }
}

impl CodePredictorConfig {
    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get number of KV groups for GQA
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Width of the predictor's inputs (talker hidden state and codec
    /// embeddings) before `small_to_mtp_projection` maps them to
    /// `hidden_size`: 2048 on the 1.7B checkpoints, `hidden_size` on 0.6B.
    pub fn input_hidden_size(&self) -> usize {
        self.text_hidden_size.unwrap_or(self.hidden_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let json = r#"{
            "architectures": ["Qwen3TTSForConditionalGeneration"],
            "model_type": "qwen3_tts",
            "tokenizer_type": "qwen3_tts_tokenizer_12hz",
            "tts_model_size": "0b6",
            "tts_model_type": "custom_voice",
            "assistant_token_id": 77091,
            "im_end_token_id": 151645,
            "im_start_token_id": 151644,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "tts_pad_token_id": 151671,
            "talker_config": {
                "model_type": "qwen3_tts_talker",
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 32768,
                "vocab_size": 3072,
                "text_vocab_size": 151936,
                "text_hidden_size": 2048,
                "num_code_groups": 16,
                "rms_norm_eps": 1e-06,
                "rope_theta": 1000000,
                "hidden_act": "silu",
                "use_cache": true,
                "position_id_per_seconds": 13,
                "code_predictor_config": {
                    "model_type": "qwen3_tts_talker_code_predictor",
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 5,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "max_position_embeddings": 65536,
                    "vocab_size": 2048,
                    "num_code_groups": 16,
                    "rms_norm_eps": 1e-06,
                    "rope_theta": 1000000,
                    "hidden_act": "silu",
                    "use_cache": true
                },
                "codec_bos_id": 2149,
                "codec_eos_token_id": 2150,
                "codec_think_id": 2154,
                "codec_nothink_id": 2155,
                "codec_pad_id": 2148,
                "codec_think_bos_id": 2156,
                "codec_think_eos_id": 2157
            }
        }"#;

        let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "qwen3_tts");
        assert_eq!(config.talker_config.hidden_size, 1024);
        assert_eq!(
            config.talker_config.code_predictor_config.num_hidden_layers,
            5
        );
    }
}
