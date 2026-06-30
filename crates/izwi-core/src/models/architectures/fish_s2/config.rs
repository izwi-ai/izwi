//! Fish Audio S2 Pro Hugging Face config parsing.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct FishS2Config {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    pub text_config: FishS2TextConfig,
    pub audio_decoder_config: FishS2AudioDecoderConfig,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub max_seq_len: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub audio_pad_token_id: u32,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct FishS2TextConfig {
    #[serde(alias = "hidden_size", alias = "dim")]
    pub hidden_size: usize,
    #[serde(alias = "num_hidden_layers", alias = "n_layers")]
    pub num_hidden_layers: usize,
    #[serde(alias = "num_attention_heads", alias = "n_heads")]
    pub num_attention_heads: usize,
    #[serde(alias = "num_key_value_heads", alias = "n_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default, alias = "head_dim")]
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    #[serde(alias = "max_position_embeddings")]
    pub max_seq_len: usize,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct FishS2AudioDecoderConfig {
    #[serde(alias = "hidden_size", alias = "dim")]
    pub hidden_size: usize,
    #[serde(alias = "num_hidden_layers", alias = "n_layers")]
    pub num_hidden_layers: usize,
    #[serde(alias = "num_attention_heads", alias = "n_heads")]
    pub num_attention_heads: usize,
    #[serde(alias = "num_key_value_heads", alias = "n_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default, alias = "head_dim")]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub max_seq_len: Option<usize>,
}

impl FishS2Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path).map_err(|err| {
            Error::ModelLoadError(format!("Failed to read {}: {err}", path.display()))
        })?;
        let config: Self = serde_json::from_str(&raw).map_err(|err| {
            Error::ModelLoadError(format!("Failed to parse {}: {err}", path.display()))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "fish_qwen3_omni" {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Fish S2 model_type `{}`",
                self.model_type
            )));
        }
        if !self
            .architectures
            .iter()
            .any(|name| name == "DualARTransformer")
        {
            return Err(Error::ModelLoadError(
                "Fish S2 config must advertise DualARTransformer".to_string(),
            ));
        }
        if self.num_codebooks == 0 || self.codebook_size == 0 {
            return Err(Error::ModelLoadError(
                "Fish S2 codebook dimensions must be non-zero".to_string(),
            ));
        }
        if self.semantic_end_token_id <= self.semantic_start_token_id {
            return Err(Error::ModelLoadError(
                "Fish S2 semantic token range is empty".to_string(),
            ));
        }
        if self.audio_pad_token_id == self.semantic_start_token_id {
            return Err(Error::ModelLoadError(
                "Fish S2 audio pad token must differ from semantic start token".to_string(),
            ));
        }
        self.text_config.validate("text_config")?;
        self.audio_decoder_config.validate("audio_decoder_config")?;
        Ok(())
    }

    pub fn semantic_vocab_size(&self) -> usize {
        (self.semantic_end_token_id - self.semantic_start_token_id) as usize
    }
}

impl FishS2TextConfig {
    fn validate(&self, label: &str) -> Result<()> {
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
            || self.max_seq_len == 0
        {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 {label} dimensions must be non-zero"
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 {label} attention heads must be divisible by KV heads"
            )));
        }
        Ok(())
    }
}

impl FishS2AudioDecoderConfig {
    fn validate(&self, label: &str) -> Result<()> {
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
        {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 {label} dimensions must be non-zero"
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 {label} attention heads must be divisible by KV heads"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_config_json() -> &'static str {
        r#"{
          "architectures": ["DualARTransformer"],
          "model_type": "fish_qwen3_omni",
          "torch_dtype": "bfloat16",
          "num_codebooks": 10,
          "codebook_size": 4096,
          "max_seq_len": 4096,
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "pad_token_id": 151643,
          "audio_pad_token_id": 151677,
          "semantic_start_token_id": 151678,
          "semantic_end_token_id": 155773,
          "sample_rate": null,
          "text_config": {
            "dim": 2560,
            "n_layers": 36,
            "n_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 155776,
            "max_seq_len": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001,
            "intermediate_size": 9728
          },
          "audio_decoder_config": {
            "dim": 5120,
            "n_layers": 6,
            "n_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "max_seq_len": 11
          }
        }"#
    }

    #[test]
    fn parses_current_fish_s2_config_shape() {
        let config: FishS2Config = serde_json::from_str(current_config_json()).unwrap();
        config.validate().unwrap();
        assert_eq!(config.num_codebooks, 10);
        assert_eq!(config.codebook_size, 4096);
        assert_eq!(config.semantic_vocab_size(), 4095);
        assert_eq!(config.text_config.hidden_size, 2560);
        assert_eq!(config.text_config.num_hidden_layers, 36);
        assert_eq!(config.text_config.num_attention_heads, 32);
        assert_eq!(config.text_config.num_key_value_heads, 8);
        assert_eq!(config.audio_decoder_config.hidden_size, 5120);
        assert_eq!(config.audio_decoder_config.num_hidden_layers, 6);
    }

    #[test]
    fn rejects_wrong_model_type() {
        let mut config: FishS2Config = serde_json::from_str(current_config_json()).unwrap();
        config.model_type = "qwen3".to_string();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("model_type"));
    }
}
