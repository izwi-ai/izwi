//! Rumble's published configuration: the Transformers-native flat layout
//! (`audio_config` and `text_config` at the top level, RoPE under
//! `rope_parameters`).

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::Qwen3Config;

/// `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RumbleConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub audio_config: RumbleAudioConfig,
    pub text_config: RumbleTextConfig,
    pub audio_token_id: u32,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    #[serde(default)]
    pub eos_token_id: RumbleTokenIds,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub support_languages: Vec<String>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// One token id or a list of them (`eos_token_id`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(untagged)]
pub enum RumbleTokenIds {
    #[default]
    None,
    One(u32),
    Many(Vec<u32>),
}

impl RumbleTokenIds {
    pub fn ids(&self) -> Vec<u32> {
        match self {
            Self::None => Vec::new(),
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.clone(),
        }
    }
}

/// `audio_config`: the audio tower's geometry.
#[derive(Debug, Clone, Deserialize)]
pub struct RumbleAudioConfig {
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
    #[serde(default)]
    pub activation_function: Option<String>,
}

/// `text_config`: the Qwen3 text trunk's geometry.
#[derive(Debug, Clone, Deserialize)]
pub struct RumbleTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub rope_parameters: Option<RumbleRopeParameters>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RumbleRopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: Option<String>,
}

impl RumbleConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path).map_err(|error| {
            Error::ModelNotFound(format!("Missing Rumble config {}: {error}", path.display()))
        })?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(raw).map_err(|error| {
            Error::ModelLoadError(format!("Invalid Rumble config.json: {error}"))
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Reject a config this loader does not implement, before any weight is
    /// read.
    pub fn validate(&self) -> Result<()> {
        let fail = |reason: String| Err(Error::ModelLoadError(format!("Rumble config: {reason}")));
        if let Some(model_type) = self.model_type.as_deref() {
            if model_type != "qwen3_asr" {
                return fail(format!("unsupported model_type {model_type:?}"));
            }
        }
        let audio = &self.audio_config;
        let text = &self.text_config;
        if audio.output_dim != text.hidden_size {
            return fail(format!(
                "projector output ({}) differs from the decoder width ({})",
                audio.output_dim, text.hidden_size
            ));
        }
        if audio.encoder_attention_heads == 0
            || !audio.d_model.is_multiple_of(audio.encoder_attention_heads)
        {
            return fail("audio width is not divisible by its heads".into());
        }
        if audio.num_mel_bins == 0 || !audio.num_mel_bins.is_multiple_of(8) {
            return fail("mel bins must be a positive multiple of 8".into());
        }
        if let Some(activation) = audio.activation_function.as_deref() {
            if activation != "gelu" {
                return fail(format!("unsupported audio activation {activation:?}"));
            }
        }
        if let Some(activation) = text.hidden_act.as_deref() {
            if activation != "silu" {
                return fail(format!("unsupported decoder activation {activation:?}"));
            }
        }
        if !(self.tie_word_embeddings && text.tie_word_embeddings) {
            return fail("the output head must be tied to the token table".into());
        }
        if text.use_sliding_window {
            return fail("sliding-window attention is not supported".into());
        }
        if let Some(rope) = &text.rope_parameters {
            if rope
                .rope_type
                .as_deref()
                .is_some_and(|kind| kind != "default")
            {
                return fail(format!("unsupported rope_type {:?}", rope.rope_type));
            }
        }
        let theta = self.rope_theta()?;
        if !theta.is_finite() || theta <= 0.0 {
            return fail("rope_theta must be finite and positive".into());
        }
        if text.max_position_embeddings == 0 || text.vocab_size == 0 {
            return fail("the decoder needs a context and a vocabulary".into());
        }
        for (name, id) in [
            ("audio_token_id", self.audio_token_id),
            ("audio_start_token_id", self.audio_start_token_id),
            ("audio_end_token_id", self.audio_end_token_id),
        ] {
            if id as usize >= text.vocab_size {
                return fail(format!("{name} {id} is outside the vocabulary"));
            }
        }
        Ok(())
    }

    pub fn rope_theta(&self) -> Result<f64> {
        self.text_config
            .rope_parameters
            .as_ref()
            .map(|rope| rope.rope_theta)
            .or(self.text_config.rope_theta)
            .ok_or_else(|| Error::ModelLoadError("Rumble config: missing rope_theta".into()))
    }

    /// The decoder's head width.
    pub fn decoder_head_dim(&self) -> usize {
        let text = &self.text_config;
        text.head_dim
            .unwrap_or(text.hidden_size / text.num_attention_heads.max(1))
    }

    /// The shared Qwen3 trunk's config for this decoder.
    pub fn trunk_config(&self) -> Result<Qwen3Config> {
        let text = &self.text_config;
        serde_json::from_value(serde_json::json!({
            "hidden_size": text.hidden_size,
            "intermediate_size": text.intermediate_size,
            "num_attention_heads": text.num_attention_heads,
            "num_hidden_layers": text.num_hidden_layers,
            "num_key_value_heads": text.num_key_value_heads,
            "max_position_embeddings": text.max_position_embeddings,
            "head_dim": self.decoder_head_dim(),
            "rms_norm_eps": text.rms_norm_eps,
            "rope_theta": self.rope_theta()?,
            "vocab_size": text.vocab_size,
            "tie_word_embeddings": true,
        }))
        .map_err(|error| Error::ModelLoadError(format!("Rumble decoder config: {error}")))
    }
}

/// `preprocessor_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RumblePreprocessorConfig {
    #[serde(default = "default_feature_size")]
    pub feature_size: usize,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: u32,
    /// Clips shorter than this many samples are zero-padded to it.
    #[serde(default = "default_min_length")]
    pub min_length: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
}

fn default_feature_size() -> usize {
    128
}
fn default_n_fft() -> usize {
    400
}
fn default_hop_length() -> usize {
    160
}
fn default_sampling_rate() -> u32 {
    16_000
}
fn default_min_length() -> usize {
    8_000
}
fn default_n_window() -> usize {
    50
}

impl RumblePreprocessorConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        let raw = fs::read_to_string(&path).map_err(|error| {
            Error::ModelNotFound(format!(
                "Missing Rumble preprocessor config {}: {error}",
                path.display()
            ))
        })?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(raw).map_err(|error| {
            Error::ModelLoadError(format!("Invalid Rumble preprocessor_config.json: {error}"))
        })?;
        if config.n_fft == 0 || config.hop_length == 0 || config.feature_size == 0 {
            return Err(Error::ModelLoadError(
                "Rumble preprocessor config: n_fft, hop_length and feature_size must be positive"
                    .into(),
            ));
        }
        if config.sampling_rate != 16_000 {
            return Err(Error::ModelLoadError(format!(
                "Rumble preprocessor config: unsupported sampling rate {}",
                config.sampling_rate
            )));
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PUBLISHED_CONFIG: &str = include_str!("../../../../tests/fixtures/rumble/config.json");
    const PUBLISHED_PREPROCESSOR: &str =
        include_str!("../../../../tests/fixtures/rumble/preprocessor_config.json");

    #[test]
    fn the_published_config_parses_into_the_trunk_geometry() {
        let config = RumbleConfig::from_json(PUBLISHED_CONFIG).unwrap();
        assert_eq!(config.audio_config.encoder_layers, 18);
        assert_eq!(config.audio_config.d_model, 896);
        assert_eq!(config.audio_config.output_dim, 1024);
        assert_eq!(config.audio_token_id, 99_595);
        assert_eq!(config.eos_token_id.ids(), vec![99_562, 99_564]);
        assert_eq!(config.support_languages.len(), 30);
        let trunk = config.trunk_config().unwrap();
        assert_eq!(trunk.num_hidden_layers, 14);
        assert_eq!(trunk.hidden_size, 1024);
        assert_eq!(trunk.num_attention_heads, 16);
        assert_eq!(trunk.num_key_value_heads, 8);
        assert_eq!(trunk.head_dim, Some(128));
        assert_eq!(trunk.vocab_size, 99_630);
        assert_eq!(trunk.rope_theta, 1_000_000.0);
        assert!(trunk.tie_word_embeddings);
        assert_eq!(trunk.max_position_embeddings, Some(65_536));
    }

    #[test]
    fn the_published_preprocessor_config_parses() {
        let config = RumblePreprocessorConfig::from_json(PUBLISHED_PREPROCESSOR).unwrap();
        assert_eq!(
            (config.feature_size, config.n_fft, config.hop_length),
            (128, 400, 160)
        );
        assert_eq!(config.min_length, 8_000);
        assert_eq!(config.n_window, 50);
    }

    #[test]
    fn an_untied_head_or_another_model_type_is_refused() {
        let mut value: serde_json::Value = serde_json::from_str(PUBLISHED_CONFIG).unwrap();
        value["tie_word_embeddings"] = false.into();
        assert!(RumbleConfig::from_json(&value.to_string()).is_err());
        let mut value: serde_json::Value = serde_json::from_str(PUBLISHED_CONFIG).unwrap();
        value["model_type"] = "whisper".into();
        assert!(RumbleConfig::from_json(&value.to_string()).is_err());
        let mut value: serde_json::Value = serde_json::from_str(PUBLISHED_CONFIG).unwrap();
        value["text_config"]["rope_parameters"] = serde_json::Value::Null;
        assert!(RumbleConfig::from_json(&value.to_string()).is_err());
    }
}
