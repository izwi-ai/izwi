//! Fish Audio S2 Pro Hugging Face config parsing.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Deserializer};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2Config {
    pub architectures: Vec<String>,
    pub model_type: String,
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
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct FishS2RawConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: String,
    #[serde(default, alias = "dtype")]
    torch_dtype: Option<String>,
    text_config: FishS2TextConfig,
    audio_decoder_config: FishS2AudioDecoderConfig,
    #[serde(default)]
    num_codebooks: Option<usize>,
    #[serde(default)]
    codebook_size: Option<usize>,
    #[serde(default)]
    max_seq_len: Option<usize>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    eos_token_id: u32,
    pad_token_id: u32,
    audio_pad_token_id: u32,
    semantic_start_token_id: u32,
    semantic_end_token_id: u32,
    #[serde(default)]
    sample_rate: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct FishS2TextConfig {
    #[serde(alias = "hidden_size", alias = "dim")]
    pub hidden_size: usize,
    #[serde(alias = "num_hidden_layers", alias = "n_layer", alias = "n_layers")]
    pub num_hidden_layers: usize,
    #[serde(alias = "num_attention_heads", alias = "n_head", alias = "n_heads")]
    pub num_attention_heads: usize,
    #[serde(
        alias = "num_key_value_heads",
        alias = "n_local_heads",
        alias = "n_kv_heads"
    )]
    pub num_key_value_heads: usize,
    #[serde(default, alias = "head_dim")]
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    #[serde(alias = "max_position_embeddings")]
    pub max_seq_len: usize,
    #[serde(default, alias = "rope_base")]
    pub rope_theta: Option<f64>,
    #[serde(default, alias = "norm_eps")]
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
    #[serde(alias = "num_hidden_layers", alias = "n_layer", alias = "n_layers")]
    pub num_hidden_layers: usize,
    #[serde(alias = "num_attention_heads", alias = "n_head", alias = "n_heads")]
    pub num_attention_heads: usize,
    #[serde(
        alias = "num_key_value_heads",
        alias = "n_local_heads",
        alias = "n_kv_heads"
    )]
    pub num_key_value_heads: usize,
    #[serde(default, alias = "head_dim")]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    #[serde(default)]
    pub num_codebooks: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
}

impl<'de> Deserialize<'de> for FishS2Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = FishS2RawConfig::deserialize(deserializer)?;
        let semantic_vocab_size = raw
            .semantic_end_token_id
            .checked_sub(raw.semantic_start_token_id)
            .ok_or_else(|| {
                serde::de::Error::custom(
                    "semantic_end_token_id must be greater than semantic_start_token_id",
                )
            })?
            + 1;
        let codebook_size = raw
            .codebook_size
            .or(raw.audio_decoder_config.vocab_size)
            .unwrap_or(semantic_vocab_size as usize);
        let num_codebooks = raw
            .num_codebooks
            .or(raw.audio_decoder_config.num_codebooks)
            .ok_or_else(|| {
                serde::de::Error::custom(
                    "Fish S2 config missing num_codebooks/audio_decoder_config.num_codebooks",
                )
            })?;
        let max_seq_len = raw.max_seq_len.unwrap_or(raw.text_config.max_seq_len);

        Ok(Self {
            architectures: if raw.architectures.is_empty() {
                vec!["DualARTransformer".to_string()]
            } else {
                raw.architectures
            },
            model_type: raw.model_type,
            torch_dtype: raw.torch_dtype,
            text_config: raw.text_config,
            audio_decoder_config: raw.audio_decoder_config,
            num_codebooks,
            codebook_size,
            max_seq_len,
            bos_token_id: raw.bos_token_id.unwrap_or(raw.pad_token_id),
            eos_token_id: raw.eos_token_id,
            pad_token_id: raw.pad_token_id,
            audio_pad_token_id: raw.audio_pad_token_id,
            semantic_start_token_id: raw.semantic_start_token_id,
            semantic_end_token_id: raw.semantic_end_token_id,
            sample_rate: raw.sample_rate,
        })
    }
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
        if self.semantic_vocab_size() != self.codebook_size {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 semantic token range has {} entries but codebook_size is {}",
                self.semantic_vocab_size(),
                self.codebook_size
            )));
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
        (self.semantic_end_token_id - self.semantic_start_token_id + 1) as usize
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
pub(crate) fn current_config_json() -> &'static str {
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
            "n_layer": 36,
            "n_head": 32,
            "n_local_heads": 8,
            "head_dim": 128,
            "vocab_size": 155776,
            "max_seq_len": 32768,
            "rope_base": 1000000.0,
            "norm_eps": 0.000001,
            "intermediate_size": 9728
          },
          "audio_decoder_config": {
            "dim": 2560,
            "n_layer": 4,
            "n_head": 32,
            "n_local_heads": 8,
            "head_dim": 128,
            "intermediate_size": 9728,
            "max_seq_len": 11,
            "num_codebooks": 10,
            "vocab_size": 4096
          }
        }"#
}

#[cfg(test)]
pub(crate) fn current_config() -> FishS2Config {
    serde_json::from_str(current_config_json()).expect("current Fish S2 config")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_current_fish_s2_config_shape() {
        let config: FishS2Config = serde_json::from_str(current_config_json()).unwrap();
        config.validate().unwrap();
        assert_eq!(config.num_codebooks, 10);
        assert_eq!(config.codebook_size, 4096);
        assert_eq!(config.semantic_vocab_size(), 4096);
        assert_eq!(config.text_config.hidden_size, 2560);
        assert_eq!(config.text_config.num_hidden_layers, 36);
        assert_eq!(config.text_config.num_attention_heads, 32);
        assert_eq!(config.text_config.num_key_value_heads, 8);
        assert_eq!(config.audio_decoder_config.hidden_size, 2560);
        assert_eq!(config.audio_decoder_config.num_hidden_layers, 4);
    }

    #[test]
    fn parses_downloaded_hf_config_shape_without_flat_dualar_fields() {
        let raw = r#"{
          "audio_decoder_config": {
            "dim": 2560,
            "head_dim": 128,
            "intermediate_size": 9728,
            "max_seq_len": 11,
            "model_type": "fish_qwen3_audio_decoder",
            "n_head": 32,
            "n_layer": 4,
            "n_local_heads": 8,
            "num_codebooks": 10,
            "rope_base": 1000000,
            "text_dim": 2560,
            "vocab_size": 4096
          },
          "audio_pad_token_id": 151677,
          "dtype": "bfloat16",
          "eos_token_id": 151645,
          "model_type": "fish_qwen3_omni",
          "pad_token_id": 151669,
          "semantic_end_token_id": 155773,
          "semantic_start_token_id": 151678,
          "text_config": {
            "dim": 2560,
            "head_dim": 128,
            "intermediate_size": 9728,
            "max_seq_len": 32768,
            "model_type": "fish_qwen3",
            "n_head": 32,
            "n_layer": 36,
            "n_local_heads": 8,
            "norm_eps": 1e-06,
            "rope_base": 1000000,
            "vocab_size": 155776
          }
        }"#;
        let config: FishS2Config = serde_json::from_str(raw).unwrap();
        config.validate().unwrap();
        assert_eq!(config.architectures, vec!["DualARTransformer"]);
        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(config.bos_token_id, 151669);
        assert_eq!(config.num_codebooks, 10);
        assert_eq!(config.codebook_size, 4096);
        assert_eq!(config.max_seq_len, 32768);
        assert_eq!(config.audio_decoder_config.hidden_size, 2560);
        assert_eq!(config.audio_decoder_config.num_hidden_layers, 4);
    }

    #[test]
    fn rejects_wrong_model_type() {
        let mut config: FishS2Config = serde_json::from_str(current_config_json()).unwrap();
        config.model_type = "qwen3".to_string();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("model_type"));
    }
}
