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
    pub encoder_hidden_size: usize,
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

        if self.encoder_config.output_dim != self.projector_config.encoder_hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech encoder output_dim {} does not match projector encoder_hidden_size {}",
                self.encoder_config.output_dim, self.projector_config.encoder_hidden_size
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

fn default_rope_theta() -> f64 {
    10_000.0
}
