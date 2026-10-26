//! Hugging Face config parsing for the Granite Speech 4.1 checkpoints: the
//! punctuating base model (`granite_speech`) and the rich-transcription plus
//! model (`granite_speech_plus`).

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::rope::{rope_frequencies, RopeFrequencies, RopeScaling};

const BASE_MODEL_TYPE: &str = "granite_speech";
const PLUS_MODEL_TYPE: &str = "granite_speech_plus";
const BASE_ARCHITECTURE: &str = "GraniteSpeechForConditionalGeneration";
const PLUS_ARCHITECTURE: &str = "GraniteSpeechPlusForConditionalGeneration";

/// Which published Granite Speech checkpoint a config describes. The two share
/// one architecture; plus concatenates intermediate encoder layers into the
/// projector input and is trained for speaker turns and word timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteSpeechCheckpointKind {
    Base,
    Plus,
}

impl GraniteSpeechCheckpointKind {
    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            BASE_MODEL_TYPE => Some(Self::Base),
            PLUS_MODEL_TYPE => Some(Self::Plus),
            _ => None,
        }
    }

    fn from_architecture(architecture: &str) -> Option<Self> {
        match architecture {
            BASE_ARCHITECTURE => Some(Self::Base),
            PLUS_ARCHITECTURE => Some(Self::Plus),
            _ => None,
        }
    }

    /// The checkpoint a catalog variant must load.
    pub const fn for_variant(variant: ModelVariant) -> Option<Self> {
        match variant {
            ModelVariant::GraniteSpeech412B => Some(Self::Base),
            ModelVariant::GraniteSpeech412BPlus => Some(Self::Plus),
            _ => None,
        }
    }

    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Base => BASE_MODEL_TYPE,
            Self::Plus => PLUS_MODEL_TYPE,
        }
    }
}

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
    /// transformers' `GraniteSpeechConfig` defaults this to true, so an
    /// absent field means an adapter is expected.
    #[serde(default = "default_has_lora_adapter")]
    pub has_lora_adapter: bool,
    pub projector_config: GraniteSpeechProjectorConfig,
    pub text_config: GraniteTextConfig,
    /// Composite-config tie flag. Base publishes `false` here and an untied
    /// `language_model.lm_head.weight`; plus sets it on `text_config` instead.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
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
    /// transformers v4 RoPE scaling and its v5 replacement: default RoPE, or
    /// YaRN for plus v2610 (see [`GraniteTextConfig::rope_scaling`]). Any
    /// other `rope_type` is rejected instead of silently dropped.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    /// Language-model tie flag; resolve through
    /// [`GraniteSpeechConfig::tie_word_embeddings`], which also consults the
    /// composite config.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub use_cache: bool,
    pub vocab_size: usize,
}

/// Processor settings with the audio front end resolved. Plus nests the
/// feature extractor under `processor_config.json:audio_processor`; base
/// ships it as a separate `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechProcessorConfig {
    pub audio_processor: GraniteSpeechAudioProcessorConfig,
    pub audio_token: String,
    pub processor_class: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct GraniteSpeechProcessorConfigFile {
    #[serde(default)]
    audio_processor: Option<GraniteSpeechAudioProcessorConfig>,
    #[serde(default = "default_audio_token")]
    audio_token: String,
    #[serde(default)]
    processor_class: Option<String>,
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

    /// The checkpoint this config describes. `model_type` and the declared
    /// architecture must agree when both name a Granite Speech checkpoint.
    pub fn checkpoint_kind(&self) -> Result<GraniteSpeechCheckpointKind> {
        let from_model_type = match self.model_type.as_deref() {
            Some(model_type) => Some(
                GraniteSpeechCheckpointKind::from_model_type(model_type).ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "Granite Speech loader expected model_type {BASE_MODEL_TYPE} or {PLUS_MODEL_TYPE}, got {model_type}"
                    ))
                })?,
            ),
            None => None,
        };
        let mut from_architectures = self.architectures.iter().filter_map(|architecture| {
            GraniteSpeechCheckpointKind::from_architecture(architecture)
        });
        let from_architecture = from_architectures.next();
        if from_architectures.any(|other| Some(other) != from_architecture) {
            return Err(Error::ModelLoadError(
                "Granite Speech config declares both the base and the plus architecture"
                    .to_string(),
            ));
        }
        match (from_model_type, from_architecture) {
            (Some(model_type), Some(architecture)) if model_type != architecture => {
                Err(Error::ModelLoadError(format!(
                    "Granite Speech model_type {} disagrees with architecture {:?}",
                    model_type.model_type(),
                    self.architectures
                )))
            }
            (Some(kind), _) | (None, Some(kind)) => Ok(kind),
            (None, None) => Err(Error::ModelLoadError(format!(
                "Granite Speech loader expected a {BASE_MODEL_TYPE} or {PLUS_MODEL_TYPE} config"
            ))),
        }
    }

    /// Check the config against what the native runtime implements and
    /// return the checkpoint kind it describes.
    pub fn validate(&self) -> Result<GraniteSpeechCheckpointKind> {
        let kind = self.checkpoint_kind()?;

        // The runtime has no LoRA path; IBM's 4.1 checkpoints ship merged
        // weights and declare `has_lora_adapter: false`.
        if self.has_lora_adapter {
            return Err(Error::ModelLoadError(
                "Granite Speech checkpoints with a LoRA adapter are not supported (has_lora_adapter must be false)"
                    .to_string(),
            ));
        }

        if self.audio_token_index >= self.text_config.vocab_size as u32 {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech audio_token_index {} exceeds text vocab size {}",
                self.audio_token_index, self.text_config.vocab_size
            )));
        }

        self.text_config.rope_scaling()?;

        let encoder = &self.encoder_config;
        // transformers' base encoder has no concatenation; only plus exports
        // intermediate layers into the projector input.
        if kind == GraniteSpeechCheckpointKind::Base && !encoder.cat_hidden_layers.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech base config must not set encoder cat_hidden_layers, got {:?}",
                encoder.cat_hidden_layers
            )));
        }
        for (position, &layer) in encoder.cat_hidden_layers.iter().enumerate() {
            if layer >= encoder.num_layers {
                return Err(Error::ModelLoadError(format!(
                    "Granite Speech cat_hidden_layers index {layer} is out of range [0, {})",
                    encoder.num_layers
                )));
            }
            if encoder.cat_hidden_layers[..position].contains(&layer) {
                return Err(Error::ModelLoadError(format!(
                    "Granite Speech cat_hidden_layers repeats layer {layer}"
                )));
            }
        }
        if encoder.context_size == 0 || encoder.context_size > encoder.max_pos_emb {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech encoder context_size {} must be in [1, max_pos_emb {}]",
                encoder.context_size, encoder.max_pos_emb
            )));
        }

        let projected_encoder_width = encoder.hidden_dim * (encoder.cat_hidden_layers.len() + 1);
        if projected_encoder_width != self.projector_config.encoder_hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech concatenated encoder width {} does not match projector encoder_hidden_size {}",
                projected_encoder_width, self.projector_config.encoder_hidden_size
            )));
        }

        Ok(kind)
    }

    /// Whether the output head shares the token embedding: the language
    /// model's own flag first, then the composite config's, then
    /// transformers' `GraniteSpeechConfig` default (tied).
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

    pub fn decoder_kv_groups(&self) -> usize {
        self.text_config.num_attention_heads / self.text_config.num_key_value_heads.max(1)
    }
}

impl GraniteTextConfig {
    /// The language model's RoPE, from `rope_parameters` (transformers v5)
    /// or `rope_scaling` (v4): default for the checkpoints through June 2026,
    /// YaRN for plus v2610, which applies it at every length. Any other type
    /// is an error: run without its scaling, a checkpoint would load and
    /// quietly transcribe worse.
    pub fn rope_scaling(&self) -> Result<RopeScaling> {
        let declared = [
            ("rope_parameters", &self.rope_parameters),
            ("rope_scaling", &self.rope_scaling),
        ]
        .into_iter()
        .filter_map(|(field, value)| {
            value
                .as_ref()
                .filter(|value| !value.is_null())
                .map(|value| (field, value))
        })
        .collect::<Vec<_>>();
        let mut scaling = None;
        for (field, params) in declared {
            if let Some(theta) = params.get("rope_theta").and_then(serde_json::Value::as_f64) {
                if theta != self.rope_theta {
                    return Err(Error::ModelLoadError(format!(
                        "Granite Speech text_config.{field}.rope_theta {theta} disagrees with rope_theta {}",
                        self.rope_theta
                    )));
                }
            }
            let parsed = RopeScaling::from_config_value(Some(params), self.max_position_embeddings)
                .map_err(|err| {
                    Error::ModelLoadError(format!("Granite Speech text_config.{field}: {err}"))
                })?;
            match scaling {
                Some(earlier) if earlier != parsed => {
                    return Err(Error::ModelLoadError(format!(
                        "Granite Speech text_config.rope_parameters and rope_scaling disagree: {} vs {}",
                        RopeScaling::name(earlier),
                        parsed.name()
                    )));
                }
                _ => scaling = Some(parsed),
            }
        }
        Ok(scaling.unwrap_or(RopeScaling::Default))
    }

    /// The decoder context the checkpoint runs at: YaRN extends the
    /// pretrained context by its factor (16,384 for plus v2610, whose config
    /// keeps `max_position_embeddings` at the pretrained 4,096).
    pub fn decoder_context(&self) -> usize {
        match self.rope_scaling() {
            Ok(RopeScaling::Yarn(yarn)) => self.max_position_embeddings.max(
                (yarn.original_max_position_embeddings as f64 * yarn.factor).round() as usize,
            ),
            _ => self.max_position_embeddings,
        }
    }

    /// Per-pair RoPE frequencies for a `head_dim` head, scaling included.
    pub fn rope_frequencies(&self, head_dim: usize) -> Result<RopeFrequencies> {
        rope_frequencies(head_dim, self.rope_theta, &self.rope_scaling()?)
    }
}

impl GraniteSpeechProcessorConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let file: GraniteSpeechProcessorConfigFile = load_json_file(
            &model_dir.join("processor_config.json"),
            "Granite Speech processor config",
        )?;
        let audio_processor = match file.audio_processor {
            Some(audio_processor) => audio_processor,
            None => {
                let path = model_dir.join("preprocessor_config.json");
                if !path.exists() {
                    return Err(Error::ModelNotFound(format!(
                        "Granite Speech processor_config.json has no audio_processor and {} is missing",
                        path.display()
                    )));
                }
                load_json_file(&path, "Granite Speech preprocessor config")?
            }
        };
        Ok(Self {
            audio_processor,
            audio_token: file.audio_token,
            processor_class: file.processor_class,
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

impl GraniteSpeechGenerationConfig {
    /// `generation_config.json` when the checkpoint ships one (plus), else
    /// the language model's own special tokens (base ships none).
    pub fn load(model_dir: &Path, config: &GraniteSpeechConfig) -> Result<Self> {
        let path = model_dir.join("generation_config.json");
        if path.exists() {
            return load_json_file(&path, "Granite Speech generation config");
        }
        Ok(Self::from_text_config(&config.text_config))
    }

    pub fn from_text_config(text: &GraniteTextConfig) -> Self {
        Self {
            bos_token_id: text.bos_token_id,
            eos_token_id: text.eos_token_id,
            pad_token_id: text.pad_token_id,
            use_cache: text.use_cache,
        }
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
                rope_scaling: None,
                rope_parameters: None,
                tie_word_embeddings: Some(true),
                use_cache: true,
                vocab_size: 100_353,
            },
            tie_word_embeddings: None,
            window_size: 15,
            model_type: Some("granite_speech_plus".to_string()),
        }
    }

    fn fixture_dir(kind: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/granite_speech")
            .join(kind)
    }

    fn census(kind: &str) -> serde_json::Value {
        let raw = fs::read_to_string(fixture_dir(kind).join("tensor_census.json")).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    fn census_shape(census: &serde_json::Value, tensor: &str) -> Vec<u64> {
        census["tensors"][tensor]["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("census is missing {tensor}"))
            .iter()
            .map(|dim| dim.as_u64().unwrap())
            .collect()
    }

    const PROJECTOR_KEY: &str =
        "projector.qformer.encoder.layer.0.crossattention.attention.key.weight";

    #[test]
    fn validate_accepts_concatenated_plus_encoder_width() {
        let cfg = plus_config();
        assert_eq!(cfg.validate().unwrap(), GraniteSpeechCheckpointKind::Plus);
    }

    #[test]
    fn validate_rejects_projector_width_mismatch() {
        let mut cfg = plus_config();
        cfg.projector_config.encoder_hidden_size = 1024;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("concatenated encoder width 2048"));
    }

    /// The published configs are the contract: each must validate as its own
    /// checkpoint kind and agree with the tensors the checkpoint ships.
    #[test]
    fn published_plus_config_validates_against_its_tensor_census() {
        let cfg = GraniteSpeechConfig::load(&fixture_dir("plus")).unwrap();
        let census = census("plus");
        assert_eq!(cfg.validate().unwrap(), GraniteSpeechCheckpointKind::Plus);
        assert_eq!(cfg.encoder_config.cat_hidden_layers, vec![3]);
        assert_eq!(cfg.projector_config.encoder_hidden_size, 2048);
        assert_eq!(
            census_shape(&census, PROJECTOR_KEY),
            vec![1024, cfg.projector_config.encoder_hidden_size as u64]
        );
        assert!(cfg.tie_word_embeddings());
        assert_eq!(census["has_lm_head"], serde_json::json!(false));
    }

    #[test]
    fn published_base_config_validates_against_its_tensor_census() {
        let cfg = GraniteSpeechConfig::load(&fixture_dir("base")).unwrap();
        let census = census("base");
        assert_eq!(cfg.validate().unwrap(), GraniteSpeechCheckpointKind::Base);
        assert!(cfg.encoder_config.cat_hidden_layers.is_empty());
        assert_eq!(cfg.projector_config.encoder_hidden_size, 1024);
        assert_eq!(
            census_shape(&census, PROJECTOR_KEY),
            vec![1024, cfg.projector_config.encoder_hidden_size as u64]
        );
        // Base publishes `tie_word_embeddings: false` only on the composite
        // config, and ships the untied head the runtime must then load.
        assert_eq!(cfg.text_config.tie_word_embeddings, None);
        assert!(!cfg.tie_word_embeddings());
        assert_eq!(census["has_lm_head"], serde_json::json!(true));
        assert_eq!(
            census_shape(&census, "language_model.lm_head.weight"),
            vec![
                cfg.text_config.vocab_size as u64,
                cfg.text_config.hidden_size as u64
            ]
        );
    }

    #[test]
    fn tie_resolution_prefers_text_config_then_composite_then_tied() {
        let mut cfg = plus_config();
        cfg.text_config.tie_word_embeddings = Some(false);
        cfg.tie_word_embeddings = Some(true);
        assert!(!cfg.tie_word_embeddings());
        cfg.text_config.tie_word_embeddings = None;
        assert!(cfg.tie_word_embeddings());
        cfg.tie_word_embeddings = Some(false);
        assert!(!cfg.tie_word_embeddings());
        cfg.tie_word_embeddings = None;
        assert!(cfg.tie_word_embeddings());
    }

    #[test]
    fn checkpoint_kind_requires_agreeing_model_type_and_architecture() {
        let mut cfg = plus_config();
        cfg.model_type = None;
        assert_eq!(
            cfg.checkpoint_kind().unwrap(),
            GraniteSpeechCheckpointKind::Plus
        );
        cfg.model_type = Some("granite_speech".to_string());
        assert!(cfg
            .checkpoint_kind()
            .unwrap_err()
            .to_string()
            .contains("disagrees"));
        cfg.model_type = Some("granite".to_string());
        assert!(cfg
            .checkpoint_kind()
            .unwrap_err()
            .to_string()
            .contains("expected model_type"));
        cfg.model_type = None;
        cfg.architectures.clear();
        assert!(cfg.checkpoint_kind().is_err());
    }

    #[test]
    fn validate_rejects_lora_adapters_and_defaults_absent_flag_like_transformers() {
        let mut cfg = plus_config();
        cfg.has_lora_adapter = true;
        assert!(cfg.validate().unwrap_err().to_string().contains("LoRA"));

        let mut raw: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_dir("plus").join("config.json")).unwrap(),
        )
        .unwrap();
        raw.as_object_mut().unwrap().remove("has_lora_adapter");
        let cfg: GraniteSpeechConfig = serde_json::from_value(raw).unwrap();
        assert!(cfg.has_lora_adapter);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_malformed_encoder_concatenation_and_context() {
        let mut base = plus_config();
        base.model_type = Some("granite_speech".to_string());
        base.architectures = vec!["GraniteSpeechForConditionalGeneration".to_string()];
        assert!(base
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not set encoder cat_hidden_layers"));

        let mut out_of_range = plus_config();
        out_of_range.encoder_config.cat_hidden_layers = vec![16];
        assert!(out_of_range
            .validate()
            .unwrap_err()
            .to_string()
            .contains("out of range [0, 16)"));

        let mut repeated = plus_config();
        repeated.encoder_config.cat_hidden_layers = vec![3, 3];
        repeated.projector_config.encoder_hidden_size = 3072;
        assert!(repeated
            .validate()
            .unwrap_err()
            .to_string()
            .contains("repeats"));

        let mut context = plus_config();
        context.encoder_config.context_size = 513;
        assert!(context
            .validate()
            .unwrap_err()
            .to_string()
            .contains("context_size 513"));
    }

    /// The published 4.1 configs declare default RoPE in the v5 form; plus
    /// v2610 declares YaRN, which runs here with the decoder context it
    /// extends to. Other types, and a theta that disagrees, are rejected.
    #[test]
    fn validate_accepts_default_and_yarn_rope_and_rejects_the_rest() {
        let base = || -> serde_json::Value {
            serde_json::from_str(
                &fs::read_to_string(fixture_dir("base").join("config.json")).unwrap(),
            )
            .unwrap()
        };
        let parse = |raw: serde_json::Value| -> GraniteSpeechConfig {
            serde_json::from_value(raw).unwrap()
        };
        let published = parse(base());
        assert_eq!(
            published.text_config.rope_parameters,
            Some(serde_json::json!({"rope_theta": 10000, "rope_type": "default"}))
        );
        assert_eq!(
            published.validate().unwrap(),
            GraniteSpeechCheckpointKind::Base
        );
        assert_eq!(published.text_config.rope_scaling().unwrap(), RopeScaling::Default);
        assert_eq!(published.text_config.decoder_context(), 4096);

        // Plus v2610: YaRN in the v5 form, max_position_embeddings left at
        // the pretrained 4096.
        let mut plus_v2610 = base();
        plus_v2610["text_config"]["rope_parameters"] = serde_json::json!({
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "rope_theta": 10000,
            "rope_type": "yarn"
        });
        let plus_v2610 = parse(plus_v2610);
        assert!(plus_v2610.validate().is_ok());
        assert!(matches!(
            plus_v2610.text_config.rope_scaling().unwrap(),
            RopeScaling::Yarn(yarn) if yarn.factor == 4.0 && yarn.original_max_position_embeddings == 4096
        ));
        assert_eq!(plus_v2610.text_config.decoder_context(), 16_384);
        let frequencies = plus_v2610.text_config.rope_frequencies(128).unwrap();
        assert!((frequencies.attention_scale - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12);

        let mut legacy_key = base();
        legacy_key["text_config"]["rope_scaling"] =
            serde_json::json!({"type": "linear", "factor": 2.0});
        assert!(parse(legacy_key).validate().is_err());

        let mut theta = base();
        theta["text_config"]["rope_parameters"]["rope_theta"] = serde_json::json!(500000);
        let err = parse(theta).validate().unwrap_err().to_string();
        assert!(err.contains("disagrees with rope_theta"), "{err}");

        let mut conflicting = base();
        conflicting["text_config"]["rope_scaling"] = serde_json::json!({
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn"
        });
        let err = parse(conflicting).validate().unwrap_err().to_string();
        assert!(err.contains("disagree"), "{err}");
    }

    #[test]
    fn processor_config_reads_nested_plus_and_separate_base_audio_settings() {
        for kind in ["plus", "base"] {
            let processor = GraniteSpeechProcessorConfig::load(&fixture_dir(kind)).unwrap();
            assert_eq!(processor.audio_token, "<|audio|>");
            assert_eq!(processor.audio_processor.melspec_kwargs.n_mels, 80);
            assert_eq!(processor.audio_processor.melspec_kwargs.hop_length, 160);
            assert_eq!(processor.audio_processor.projector_window_size, 15);
            assert_eq!(processor.audio_processor.projector_downsample_rate, 5);
            assert_eq!(processor.sample_rate(), 16_000);
        }
    }

    #[test]
    fn processor_config_without_audio_settings_names_the_missing_file() {
        let dir =
            std::env::temp_dir().join(format!("granite-speech-processor-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("processor_config.json"),
            r#"{"audio_token":"<|audio|>"}"#,
        )
        .unwrap();
        let err = GraniteSpeechProcessorConfig::load(&dir).unwrap_err();
        assert!(err.to_string().contains("preprocessor_config.json"));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn generation_config_is_optional_and_falls_back_to_text_tokens() {
        let base = GraniteSpeechConfig::load(&fixture_dir("base")).unwrap();
        assert!(!fixture_dir("base").join("generation_config.json").exists());
        let generation = GraniteSpeechGenerationConfig::load(&fixture_dir("base"), &base).unwrap();
        assert_eq!(generation.bos_token_id, 100_257);
        assert_eq!(generation.eos_token_id, 100_257);
        assert_eq!(generation.pad_token_id, 100_256);

        let plus = GraniteSpeechConfig::load(&fixture_dir("plus")).unwrap();
        let generation = GraniteSpeechGenerationConfig::load(&fixture_dir("plus"), &plus).unwrap();
        assert_eq!(
            generation,
            GraniteSpeechGenerationConfig::from_text_config(&plus.text_config)
        );
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
