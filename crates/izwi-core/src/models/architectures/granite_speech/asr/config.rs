//! Hugging Face config parsing for Granite Speech 4.1 Plus.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeechConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub audio_token_index: u32,
    #[serde(default = "default_downsample_rate")]
    pub downsample_rate: usize,
    #[serde(default)]
    pub dtype: Option<String>,
    pub encoder_config: GraniteSpeechEncoderConfig,
    #[serde(default)]
    pub has_lora_adapter: bool,
    pub projector_config: GraniteSpeechProjectorConfig,
    pub text_config: GraniteTextConfig,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default)]
    pub model_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeechEncoderConfig {
    #[serde(default)]
    pub cat_hidden_layers: Vec<usize>,
    pub context_size: usize,
    pub conv_expansion_factor: usize,
    pub conv_kernel_size: usize,
    pub dim_head: usize,
    #[serde(default)]
    pub dropout: f32,
    pub feedforward_mult: usize,
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub max_pos_emb: usize,
    #[serde(default)]
    pub model_type: Option<String>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeechProjectorConfig {
    #[serde(default = "default_attention_probs_dropout_prob")]
    pub attention_probs_dropout_prob: f32,
    pub encoder_hidden_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_hidden_dropout_prob")]
    pub hidden_dropout_prob: f32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub model_type: Option<String>,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub use_qformer_text_input: bool,
    #[serde(default = "default_cross_attention_frequency")]
    pub cross_attention_frequency: usize,
    #[serde(default)]
    pub chunk_size_feed_forward: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteTextConfig {
    pub attention_multiplier: f32,
    pub bos_token_id: u32,
    #[serde(default)]
    pub dtype: Option<String>,
    pub embedding_multiplier: f32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub logits_scaling: f32,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub model_type: Option<String>,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pad_token_id: u32,
    pub residual_multiplier: f32,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeechProcessorConfig {
    pub audio_processor: GraniteSpeechAudioProcessorConfig,
    #[serde(default = "default_audio_token")]
    pub audio_token: String,
    #[serde(default)]
    pub processor_class: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeechAudioProcessorConfig {
    #[serde(default)]
    pub feature_extractor_type: Option<String>,
    pub melspec_kwargs: GraniteSpeechMelSpecConfig,
    #[serde(default = "default_downsample_rate")]
    pub projector_downsample_rate: usize,
    #[serde(default = "default_window_size")]
    pub projector_window_size: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: u32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeechMelSpecConfig {
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    #[serde(default = "default_sampling_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_win_length")]
    pub win_length: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GraniteSpeechGenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    #[serde(default)]
    pub use_cache: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GraniteSpeechTokenizerConfig {
    #[serde(default)]
    pub add_prefix_space: bool,
    #[serde(default = "default_audio_token")]
    pub audio_token: String,
    #[serde(default)]
    pub bos_token: Option<String>,
    #[serde(default)]
    pub eos_token: Option<String>,
    #[serde(default)]
    pub pad_token: Option<String>,
    #[serde(default)]
    pub tokenizer_class: Option<String>,
    #[serde(default)]
    pub processor_class: Option<String>,
}

impl GraniteSpeechConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        load_json_file(&model_dir.join("config.json"), "Granite Speech config")
    }

    pub fn validate_plus(&self) -> Result<()> {
        let model_type_ok = self
            .model_type
            .as_deref()
            .is_some_and(|model_type| model_type == "granite_speech_plus");
        let architecture_ok = self
            .architectures
            .iter()
            .any(|architecture| architecture == "GraniteSpeechPlusForConditionalGeneration");
        if !model_type_ok && !architecture_ok {
            return Err(Error::ModelLoadError(
                "Granite Speech loader expected a granite_speech_plus config".to_string(),
            ));
        }

        if self.audio_token_index >= self.text_config.vocab_size as u32 {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech audio_token_index {} exceeds text vocab size {}",
                self.audio_token_index, self.text_config.vocab_size
            )));
        }

        let projected_encoder_width =
            self.encoder_config.hidden_dim * (self.encoder_config.cat_hidden_layers.len() + 1);
        if projected_encoder_width != self.projector_config.encoder_hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech concatenated encoder width {} does not match projector encoder_hidden_size {}",
                projected_encoder_width, self.projector_config.encoder_hidden_size
            )));
        }

        Ok(())
    }

    pub fn target_dtype_hint(&self) -> Option<&str> {
        self.dtype
            .as_deref()
            .or(self.text_config.dtype.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    pub fn decoder_head_dim(&self) -> usize {
        self.text_config.hidden_size / self.text_config.num_attention_heads.max(1)
    }

    pub fn decoder_kv_groups(&self) -> usize {
        self.text_config.num_attention_heads / self.text_config.num_key_value_heads.max(1)
    }
}

impl GraniteSpeechProcessorConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        load_json_file(
            &model_dir.join("processor_config.json"),
            "Granite Speech processor config",
        )
    }

    pub fn sample_rate(&self) -> u32 {
        self.audio_processor
            .melspec_kwargs
            .sample_rate
            .max(self.audio_processor.sampling_rate)
            .max(1)
    }
}

impl GraniteSpeechGenerationConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        load_json_file(
            &model_dir.join("generation_config.json"),
            "Granite Speech generation config",
        )
    }
}

impl GraniteSpeechTokenizerConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        load_json_file(
            &model_dir.join("tokenizer_config.json"),
            "Granite Speech tokenizer config",
        )
    }
}

pub fn load_granite_speech_chat_template(model_dir: &Path) -> Result<String> {
    let path = model_dir.join("chat_template.jinja");
    fs::read_to_string(&path).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to read Granite Speech chat template {}: {err}",
            path.display()
        ))
    })
}

fn load_json_file<T>(path: &Path, label: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let raw = fs::read_to_string(path).map_err(|err| {
        Error::ModelLoadError(format!("Failed to read {label} {}: {err}", path.display()))
    })?;
    serde_json::from_str(&raw).map_err(|err| {
        Error::ModelLoadError(format!("Failed to parse {label} {}: {err}", path.display()))
    })
}

fn default_audio_token() -> String {
    "<|audio|>".to_string()
}

fn default_sampling_rate() -> u32 {
    16_000
}

fn default_hop_length() -> usize {
    160
}

fn default_n_fft() -> usize {
    512
}

fn default_n_mels() -> usize {
    80
}

fn default_win_length() -> usize {
    400
}

fn default_downsample_rate() -> usize {
    5
}

fn default_window_size() -> usize {
    15
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

fn default_cross_attention_frequency() -> usize {
    1
}

fn default_hidden_act() -> String {
    "gelu".to_string()
}

fn default_hidden_dropout_prob() -> f32 {
    0.1
}

fn default_attention_probs_dropout_prob() -> f32 {
    0.1
}

fn default_rope_theta() -> f64 {
    10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plus_config() -> GraniteSpeechConfig {
        GraniteSpeechConfig {
            architectures: vec!["GraniteSpeechPlusForConditionalGeneration".to_string()],
            audio_token_index: 100_352,
            downsample_rate: 5,
            dtype: Some("bfloat16".to_string()),
            encoder_config: GraniteSpeechEncoderConfig {
                cat_hidden_layers: vec![3],
                context_size: 200,
                conv_expansion_factor: 2,
                conv_kernel_size: 15,
                dim_head: 128,
                dropout: 0.0,
                feedforward_mult: 4,
                hidden_dim: 1024,
                input_dim: 160,
                max_pos_emb: 512,
                model_type: Some("granite_speech_plus_encoder".to_string()),
                num_heads: 8,
                num_layers: 16,
                output_dim: 348,
            },
            has_lora_adapter: false,
            projector_config: GraniteSpeechProjectorConfig {
                attention_probs_dropout_prob: default_attention_probs_dropout_prob(),
                encoder_hidden_size: 2048,
                hidden_act: default_hidden_act(),
                hidden_dropout_prob: default_hidden_dropout_prob(),
                hidden_size: 1024,
                intermediate_size: 4096,
                layer_norm_eps: default_layer_norm_eps(),
                max_position_embeddings: 512,
                model_type: Some("blip_2_qformer".to_string()),
                num_attention_heads: 16,
                num_hidden_layers: 2,
                pad_token_id: Some(0),
                use_qformer_text_input: false,
                cross_attention_frequency: 1,
                chunk_size_feed_forward: 0,
            },
            text_config: GraniteTextConfig {
                attention_multiplier: 0.0078125,
                bos_token_id: 100_257,
                dtype: Some("bfloat16".to_string()),
                embedding_multiplier: 12.0,
                eos_token_id: 100_257,
                hidden_size: 2048,
                intermediate_size: 4096,
                logits_scaling: 8.0,
                max_position_embeddings: 4096,
                model_type: Some("granite".to_string()),
                num_attention_heads: 16,
                num_hidden_layers: 40,
                num_key_value_heads: 4,
                pad_token_id: 100_256,
                residual_multiplier: 0.22,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                tie_word_embeddings: true,
                use_cache: true,
                vocab_size: 100_353,
            },
            window_size: 15,
            model_type: Some("granite_speech_plus".to_string()),
        }
    }

    #[test]
    fn validate_plus_accepts_concatenated_encoder_width() {
        let cfg = plus_config();
        cfg.validate_plus().unwrap();
    }

    #[test]
    fn validate_plus_rejects_projector_width_mismatch() {
        let mut cfg = plus_config();
        cfg.projector_config.encoder_hidden_size = 1024;
        let err = cfg.validate_plus().unwrap_err();
        assert!(err
            .to_string()
            .contains("concatenated encoder width 2048"));
    }

    #[test]
    fn projector_config_defaults_blip2_fields_when_omitted() {
        let raw = r#"{
            "encoder_hidden_size": 2048,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 2
        }"#;
        let cfg: GraniteSpeechProjectorConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.hidden_act, "gelu");
        assert_eq!(cfg.hidden_dropout_prob, 0.1);
        assert_eq!(cfg.attention_probs_dropout_prob, 0.1);
        assert_eq!(cfg.cross_attention_frequency, 1);
    }
}
