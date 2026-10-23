//! Hugging Face config parsing for Granite Speech 5.0.
//!
//! 5.0 declares the same `model_type` and architecture as Granite Speech 4.1
//! base, so a config never tells the two apart: the catalog variant selects
//! this loader, and validation here requires what 5.0 adds (a YaRN language
//! model whose context is the scaled pretraining context).

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::models::shared::rope::{RopeScaling, YarnRopeScaling};

const MODEL_TYPE: &str = "granite_speech";
const ARCHITECTURE: &str = "GraniteSpeechForConditionalGeneration";

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeech5Config {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub audio_token_index: u32,
    #[serde(default = "default_downsample_rate")]
    pub downsample_rate: usize,
    #[serde(default)]
    pub dtype: Option<String>,
    pub encoder_config: GraniteSpeech5EncoderConfig,
    /// transformers' `GraniteSpeechConfig` defaults this to true, so an
    /// absent field means an adapter is expected.
    #[serde(default = "default_has_lora_adapter")]
    pub has_lora_adapter: bool,
    pub projector_config: GraniteSpeech5ProjectorConfig,
    pub text_config: GraniteSpeech5TextConfig,
    /// Composite-config tie flag: 5.0 publishes `false` here and an untied
    /// `language_model.lm_head.weight`.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default)]
    pub model_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeech5EncoderConfig {
    /// Granite Speech 4.1 plus concatenates intermediate layers into the
    /// projector input; 5.0 must not.
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
pub struct GraniteSpeech5ProjectorConfig {
    pub encoder_hidden_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub model_type: Option<String>,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub use_qformer_text_input: bool,
    #[serde(default = "default_cross_attention_frequency")]
    pub cross_attention_frequency: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeech5TextConfig {
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
    /// transformers v4 RoPE scaling (what 5.0 publishes) and its v5
    /// replacement; resolve through [`GraniteSpeech5TextConfig::rope`].
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    /// Language-model tie flag; resolve through
    /// [`GraniteSpeech5Config::tie_word_embeddings`].
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    pub vocab_size: usize,
}

/// The audio front end: `preprocessor_config.json`, with the audio token from
/// `processor_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeech5ProcessorConfig {
    pub audio_processor: GraniteSpeech5AudioProcessorConfig,
    pub audio_token: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct ProcessorConfigFile {
    #[serde(default = "default_audio_token")]
    audio_token: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeech5AudioProcessorConfig {
    #[serde(default)]
    pub feature_extractor_type: Option<String>,
    pub melspec_kwargs: GraniteSpeech5MelSpecConfig,
    #[serde(default = "default_downsample_rate")]
    pub projector_downsample_rate: usize,
    #[serde(default = "default_window_size")]
    pub projector_window_size: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: u32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraniteSpeech5MelSpecConfig {
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
pub struct GraniteSpeech5GenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GraniteSpeech5TokenizerConfig {
    #[serde(default = "default_audio_token")]
    pub audio_token: String,
}

impl GraniteSpeech5Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        load_json_file(&model_dir.join("config.json"), "Granite Speech 5.0 config")
    }

    /// Check the config against what this runtime implements.
    pub fn validate(&self) -> Result<()> {
        if let Some(model_type) = self.model_type.as_deref() {
            if model_type != MODEL_TYPE {
                return Err(Error::ModelLoadError(format!(
                    "Granite Speech 5.0 loader expected model_type {MODEL_TYPE}, got {model_type}"
                )));
            }
        }
        if !self.architectures.is_empty()
            && !self.architectures.iter().any(|name| name == ARCHITECTURE)
        {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 loader expected architecture {ARCHITECTURE}, got {:?}",
                self.architectures
            )));
        }
        // IBM publishes merged weights; the runtime has no LoRA path.
        if self.has_lora_adapter {
            return Err(Error::ModelLoadError(
                "Granite Speech 5.0 checkpoints with a LoRA adapter are not supported (has_lora_adapter must be false)"
                    .to_string(),
            ));
        }
        if self.audio_token_index >= self.text_config.vocab_size as u32 {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 audio_token_index {} exceeds text vocab size {}",
                self.audio_token_index, self.text_config.vocab_size
            )));
        }
        let encoder = &self.encoder_config;
        if !encoder.cat_hidden_layers.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 encoder must not concatenate hidden layers, got {:?}",
                encoder.cat_hidden_layers
            )));
        }
        if encoder.context_size == 0 || encoder.context_size > encoder.max_pos_emb {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 encoder context_size {} must be in [1, max_pos_emb {}]",
                encoder.context_size, encoder.max_pos_emb
            )));
        }
        if encoder.hidden_dim != self.projector_config.encoder_hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 encoder width {} does not match projector encoder_hidden_size {}",
                encoder.hidden_dim, self.projector_config.encoder_hidden_size
            )));
        }
        if self.window_size == 0
            || self.downsample_rate == 0
            || !self.window_size.is_multiple_of(self.downsample_rate)
        {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 projector window {} must be a positive multiple of downsample_rate {}",
                self.window_size, self.downsample_rate
            )));
        }
        self.text_config.yarn().map(|_| ())
    }

    /// Whether the output head shares the token embedding: the language
    /// model's own flag first, then the composite config's, then
    /// transformers' default (tied).
    pub fn tie_word_embeddings(&self) -> bool {
        self.text_config
            .tie_word_embeddings
            .or(self.tie_word_embeddings)
            .unwrap_or(true)
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

    /// Projected audio tokens per projector window.
    pub fn queries_per_window(&self) -> usize {
        self.window_size.max(1) / self.downsample_rate.max(1)
    }
}

impl GraniteSpeech5TextConfig {
    /// The RoPE the language model was trained with, from `rope_scaling` or,
    /// when only the v5 form is published, `rope_parameters`.
    pub fn rope(&self) -> Result<RopeScaling> {
        let declared = self
            .rope_scaling
            .as_ref()
            .filter(|value| !value.is_null())
            .or(self.rope_parameters.as_ref());
        if let Some(theta) = self
            .rope_parameters
            .as_ref()
            .and_then(|params| params.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
        {
            if theta != self.rope_theta {
                return Err(Error::ModelLoadError(format!(
                    "Granite Speech 5.0 rope_parameters.rope_theta {theta} disagrees with rope_theta {}",
                    self.rope_theta
                )));
            }
        }
        RopeScaling::from_config_value(declared, self.max_position_embeddings)
    }

    /// 5.0's language model is YaRN-extended: its context must be the
    /// pretraining context times the factor. Unscaled RoPE here means a
    /// different (or mis-published) checkpoint, which this loader refuses.
    pub fn yarn(&self) -> Result<YarnRopeScaling> {
        let RopeScaling::Yarn(yarn) = self.rope()? else {
            return Err(Error::ModelLoadError(
                "Granite Speech 5.0 expects a YaRN language model, but the config declares default RoPE"
                    .to_string(),
            ));
        };
        let scaled = yarn.factor * yarn.original_max_position_embeddings as f64;
        if (scaled - self.max_position_embeddings as f64).abs() > 0.5 {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 max_position_embeddings {} is not the YaRN factor {} times the original context {}",
                self.max_position_embeddings, yarn.factor, yarn.original_max_position_embeddings
            )));
        }
        Ok(yarn)
    }
}

impl GraniteSpeech5ProcessorConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let processor: ProcessorConfigFile = load_json_file(
            &model_dir.join("processor_config.json"),
            "Granite Speech 5.0 processor config",
        )?;
        let audio_processor = load_json_file(
            &model_dir.join("preprocessor_config.json"),
            "Granite Speech 5.0 preprocessor config",
        )?;
        Ok(Self {
            audio_processor,
            audio_token: processor.audio_token,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.audio_processor
            .melspec_kwargs
            .sample_rate
            .max(self.audio_processor.sampling_rate)
            .max(1)
    }
}

impl GraniteSpeech5GenerationConfig {
    /// `generation_config.json` when present, else the language model's own
    /// special tokens.
    pub fn load(model_dir: &Path, config: &GraniteSpeech5Config) -> Result<Self> {
        let path = model_dir.join("generation_config.json");
        if path.exists() {
            return load_json_file(&path, "Granite Speech 5.0 generation config");
        }
        Ok(Self::from_text_config(&config.text_config))
    }

    pub fn from_text_config(text: &GraniteSpeech5TextConfig) -> Self {
        Self {
            bos_token_id: text.bos_token_id,
            eos_token_id: text.eos_token_id,
            pad_token_id: text.pad_token_id,
        }
    }
}

impl GraniteSpeech5TokenizerConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        load_json_file(
            &model_dir.join("tokenizer_config.json"),
            "Granite Speech 5.0 tokenizer config",
        )
    }
}

pub fn load_granite_speech5_chat_template(model_dir: &Path) -> Result<String> {
    let path = model_dir.join("chat_template.jinja");
    fs::read_to_string(&path).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to read Granite Speech 5.0 chat template {}: {err}",
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

fn default_has_lora_adapter() -> bool {
    true
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

fn default_rope_theta() -> f64 {
    10_000.0
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/granite_speech5")
    }

    fn published() -> serde_json::Value {
        serde_json::from_str(&fs::read_to_string(fixture_dir().join("config.json")).unwrap())
            .unwrap()
    }

    fn parse(raw: serde_json::Value) -> GraniteSpeech5Config {
        serde_json::from_value(raw).unwrap()
    }

    fn census() -> serde_json::Value {
        serde_json::from_str(&fs::read_to_string(fixture_dir().join("tensor_census.json")).unwrap())
            .unwrap()
    }

    fn census_shape(census: &serde_json::Value, tensor: &str) -> Vec<u64> {
        census["tensors"][tensor]["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("census is missing {tensor}"))
            .iter()
            .map(|dim| dim.as_u64().unwrap())
            .collect()
    }

    /// The published config is the contract: it validates, declares the
    /// YaRN language model, and agrees with the tensors the checkpoint ships.
    #[test]
    fn published_config_validates_against_its_tensor_census() {
        let cfg = GraniteSpeech5Config::load(&fixture_dir()).unwrap();
        cfg.validate().unwrap();
        let yarn = cfg.text_config.yarn().unwrap();
        assert_eq!(yarn.factor, 4.0);
        assert_eq!(yarn.original_max_position_embeddings, 4096);
        assert_eq!(cfg.text_config.max_position_embeddings, 16_384);
        assert_eq!(cfg.decoder_head_dim(), 128);
        assert_eq!(cfg.queries_per_window(), 3);
        assert!(!cfg.tie_word_embeddings());

        let census = census();
        assert_eq!(census["tensor_count"], serde_json::json!(954));
        assert_eq!(census["has_lm_head"], serde_json::json!(true));
        assert_eq!(
            census_shape(&census, "language_model.lm_head.weight"),
            vec![
                cfg.text_config.vocab_size as u64,
                cfg.text_config.hidden_size as u64
            ]
        );
        assert_eq!(
            census_shape(
                &census,
                "projector.qformer.encoder.layer.0.crossattention.attention.key.weight"
            ),
            vec![1024, cfg.projector_config.encoder_hidden_size as u64]
        );
        assert_eq!(
            census_shape(
                &census,
                "language_model.model.layers.0.self_attn.k_proj.weight"
            ),
            vec![
                (cfg.text_config.num_key_value_heads * cfg.decoder_head_dim()) as u64,
                cfg.text_config.hidden_size as u64
            ]
        );
        assert_eq!(
            census_shape(&census, "encoder.layers.0.attn.rel_pos_emb.weight"),
            vec![
                (2 * cfg.encoder_config.max_pos_emb + 1) as u64,
                cfg.encoder_config.dim_head as u64
            ]
        );
    }

    #[test]
    fn validate_requires_the_yarn_context_5_0_was_trained_with() {
        let mut unscaled = published();
        unscaled["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("rope_scaling");
        let err = parse(unscaled).validate().unwrap_err().to_string();
        assert!(err.contains("expects a YaRN language model"), "{err}");

        let mut short = published();
        short["text_config"]["max_position_embeddings"] = serde_json::json!(4096);
        let err = parse(short).validate().unwrap_err().to_string();
        assert!(err.contains("is not the YaRN factor"), "{err}");

        let mut linear = published();
        linear["text_config"]["rope_scaling"] =
            serde_json::json!({"rope_type": "linear", "factor": 4.0});
        assert!(parse(linear).validate().is_err());

        // The v5 form of the same scaling also validates.
        let mut v5 = published();
        let scaling = v5["text_config"]["rope_scaling"].take();
        let mut params = scaling.as_object().unwrap().clone();
        params.insert("rope_theta".into(), serde_json::json!(10000));
        v5["text_config"]["rope_parameters"] = serde_json::Value::Object(params);
        parse(v5).validate().unwrap();

        let mut theta = published();
        theta["text_config"]["rope_parameters"] =
            serde_json::json!({"rope_type": "yarn", "factor": 4.0, "rope_theta": 500000});
        let err = parse(theta).validate().unwrap_err().to_string();
        assert!(err.contains("disagrees with rope_theta"), "{err}");
    }

    #[test]
    fn validate_rejects_4_1_plus_shapes_lora_and_foreign_configs() {
        let mut concatenated = published();
        concatenated["encoder_config"]["cat_hidden_layers"] = serde_json::json!([3]);
        concatenated["projector_config"]["encoder_hidden_size"] = serde_json::json!(2048);
        assert!(parse(concatenated)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not concatenate"));

        let mut lora = published();
        lora.as_object_mut().unwrap().remove("has_lora_adapter");
        assert!(parse(lora)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("LoRA"));

        let mut plus = published();
        plus["model_type"] = serde_json::json!("granite_speech_plus");
        assert!(parse(plus).validate().is_err());

        let mut architecture = published();
        architecture["architectures"] =
            serde_json::json!(["GraniteSpeechPlusForConditionalGeneration"]);
        assert!(parse(architecture).validate().is_err());

        let mut context = published();
        context["encoder_config"]["context_size"] = serde_json::json!(513);
        assert!(parse(context).validate().is_err());

        let mut window = published();
        window["window_size"] = serde_json::json!(16);
        assert!(parse(window).validate().is_err());
    }

    #[test]
    fn processor_generation_and_tokenizer_configs_load_from_the_published_files() {
        let processor = GraniteSpeech5ProcessorConfig::load(&fixture_dir()).unwrap();
        assert_eq!(processor.audio_token, "<|audio|>");
        assert_eq!(processor.audio_processor.melspec_kwargs.n_mels, 80);
        assert_eq!(processor.audio_processor.melspec_kwargs.n_fft, 512);
        assert_eq!(processor.audio_processor.melspec_kwargs.hop_length, 160);
        assert_eq!(processor.audio_processor.melspec_kwargs.win_length, 400);
        assert_eq!(processor.audio_processor.projector_window_size, 15);
        assert_eq!(processor.audio_processor.projector_downsample_rate, 5);
        assert_eq!(processor.sample_rate(), 16_000);

        let cfg = GraniteSpeech5Config::load(&fixture_dir()).unwrap();
        let generation = GraniteSpeech5GenerationConfig::load(&fixture_dir(), &cfg).unwrap();
        assert_eq!(
            generation,
            GraniteSpeech5GenerationConfig::from_text_config(&cfg.text_config)
        );
        assert_eq!(generation.eos_token_id, 100_257);
        assert_eq!(generation.pad_token_id, 100_256);
        assert_eq!(
            load_granite_speech5_chat_template(&fixture_dir()).unwrap(),
            "{% for message in messages %}{% if message['role'] == 'user' %}USER: {{ message['content'] }}\n ASSISTANT:{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"
        );
    }
}
