//! Native Granite Speech ASR facade.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::DType;
use serde_json::json;
use tracing::info;

use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

mod config;
mod preprocessor;
mod prompt;
mod transcript;

pub use config::{
    load_granite_speech_chat_template, GraniteSpeechAudioProcessorConfig, GraniteSpeechConfig,
    GraniteSpeechEncoderConfig, GraniteSpeechGenerationConfig, GraniteSpeechMelSpecConfig,
    GraniteSpeechProcessorConfig, GraniteSpeechProjectorConfig, GraniteSpeechTokenizerConfig,
    GraniteTextConfig,
};
pub use preprocessor::{GraniteSpeechAudioFeatures, GraniteSpeechPreprocessor};
pub use prompt::{
    GraniteSpeechPrompt, GraniteSpeechPromptOptions, GraniteSpeechPromptTokenizer,
    GraniteSpeechSpecialTokens, GraniteSpeechTask, GRANITE_SPEECH_ASR_PROMPT,
    GRANITE_SPEECH_SPEAKER_PROMPT, GRANITE_SPEECH_SYSTEM_PROMPT, GRANITE_SPEECH_TIMESTAMP_PROMPT,
};
pub use transcript::{
    parse_granite_speech_output, GraniteSpeechParsedTranscript, GraniteSpeechSegment,
    GraniteSpeechTimestampWord,
};

const REQUIRED_ARTIFACTS: &[&str] = &[
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

const DEFAULT_MAX_AUDIO_SECONDS: f32 = 9.0 * 60.0;
const TIMESTAMP_MAX_AUDIO_SECONDS: f32 = 5.0 * 60.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechAsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

pub struct GraniteSpeechAsrModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: GraniteSpeechConfig,
    processor: GraniteSpeechProcessorConfig,
    generation: GraniteSpeechGenerationConfig,
    tokenizer_config: GraniteSpeechTokenizerConfig,
    chat_template: String,
    prompt_tokenizer: GraniteSpeechPromptTokenizer,
    preprocessor: GraniteSpeechPreprocessor,
}

impl GraniteSpeechAsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::GraniteSpeech412BPlus {
            return Err(Error::InvalidInput(format!(
                "GraniteSpeechAsrModel cannot load non-Granite variant {variant}"
            )));
        }

        let shards = ensure_granite_speech_artifacts(model_dir)?;
        let config = GraniteSpeechConfig::load(model_dir)?;
        config.validate_plus()?;
        let processor = GraniteSpeechProcessorConfig::load(model_dir)?;
        let generation = GraniteSpeechGenerationConfig::load(model_dir)?;
        let tokenizer_config = GraniteSpeechTokenizerConfig::load(model_dir)?;
        let chat_template = load_granite_speech_chat_template(model_dir)?;
        let prompt_tokenizer =
            GraniteSpeechPromptTokenizer::load(model_dir, &config, &processor, &tokenizer_config)?;
        let preprocessor = GraniteSpeechPreprocessor::new(processor.clone())?;
        let dtype = std::env::var("IZWI_GRANITE_SPEECH_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
            .or(config.target_dtype_hint())
            .map(|raw| {
                device.select_model_dtype_checked(
                    ModelFamily::GraniteSpeechAsr,
                    Some(raw),
                    "Granite Speech ASR",
                )
            })
            .transpose()?
            .unwrap_or_else(|| device.select_model_dtype(ModelFamily::GraniteSpeechAsr, None));

        info!(
            "Validated Granite Speech ASR metadata in {:?} on {:?} with dtype {:?} ({} shard files)",
            model_dir,
            device.kind,
            dtype,
            shards.len()
        );

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            processor,
            generation,
            tokenizer_config,
            chat_template,
            prompt_tokenizer,
            preprocessor,
        })
    }

    pub fn config(&self) -> &GraniteSpeechConfig {
        &self.config
    }

    pub fn processor(&self) -> &GraniteSpeechProcessorConfig {
        &self.processor
    }

    pub fn generation_config(&self) -> &GraniteSpeechGenerationConfig {
        &self.generation
    }

    pub fn tokenizer_config(&self) -> &GraniteSpeechTokenizerConfig {
        &self.tokenizer_config
    }

    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn prepare_audio_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<GraniteSpeechAudioFeatures> {
        self.preprocessor.prepare(audio, sample_rate)
    }

    pub fn build_prompt(
        &self,
        options: &GraniteSpeechPromptOptions,
    ) -> Result<GraniteSpeechPrompt> {
        self.prompt_tokenizer.build_prompt(options)
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(DEFAULT_MAX_AUDIO_SECONDS)
    }

    pub fn max_timestamp_audio_seconds_hint(&self) -> Option<f32> {
        Some(TIMESTAMP_MAX_AUDIO_SECONDS)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let _ = (audio, sample_rate, language, prompt);
        Err(granite_speech_inference_not_ready())
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let _ = (audio, sample_rate, language, prompt, on_delta);
        Err(granite_speech_inference_not_ready())
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        json!({
            "family": "granite_speech_asr",
            "model_type": self.config.model_type,
            "audio_token_index": self.config.audio_token_index,
            "dtype": format!("{:?}", self.dtype),
            "device_kind": format!("{:?}", self.device.kind),
            "sample_rate": self.processor.sample_rate(),
            "n_mels": self.processor.audio_processor.melspec_kwargs.n_mels,
            "projector_downsample_rate": self.processor.audio_processor.projector_downsample_rate,
            "chat_template_bytes": self.chat_template.len(),
            "max_audio_seconds": DEFAULT_MAX_AUDIO_SECONDS,
            "max_timestamp_audio_seconds": TIMESTAMP_MAX_AUDIO_SECONDS,
        })
    }
}

fn granite_speech_inference_not_ready() -> Error {
    Error::InferenceError(
        "Granite Speech native inference is not wired yet; decoder forward/generation lands in the next implementation phase"
            .to_string(),
    )
}

pub fn ensure_granite_speech_artifacts(model_dir: &Path) -> Result<Vec<PathBuf>> {
    for file in REQUIRED_ARTIFACTS {
        let path = model_dir.join(file);
        if !path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Granite Speech artifact {}",
                path.display()
            )));
        }
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let raw = fs::read_to_string(&index_path).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to read Granite Speech safetensors index {}: {err}",
            index_path.display()
        ))
    })?;
    let index: serde_json::Value = serde_json::from_str(&raw).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to parse Granite Speech safetensors index {}: {err}",
            index_path.display()
        ))
    })?;
    let weight_map = index
        .get("weight_map")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            Error::ModelLoadError(
                "Invalid Granite Speech model.safetensors.index.json: missing weight_map"
                    .to_string(),
            )
        })?;

    let mut shard_files = BTreeSet::new();
    for value in weight_map.values() {
        let Some(file) = value.as_str() else {
            return Err(Error::ModelLoadError(
                "Invalid Granite Speech safetensors index: non-string shard filename".to_string(),
            ));
        };
        validate_shard_filename(file)?;
        shard_files.insert(file.to_string());
    }

    if shard_files.is_empty() {
        return Err(Error::ModelLoadError(
            "Granite Speech safetensors index contains no shard files".to_string(),
        ));
    }

    let mut shard_paths = Vec::with_capacity(shard_files.len());
    for file in shard_files {
        let path = model_dir.join(&file);
        if !path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Granite Speech safetensors shard {}",
                path.display()
            )));
        }
        shard_paths.push(path);
    }

    Ok(shard_paths)
}

fn validate_shard_filename(file: &str) -> Result<()> {
    let path = Path::new(file);
    let is_plain_relative = path
        .parent()
        .is_none_or(|parent| parent.as_os_str().is_empty())
        && path.file_name().is_some()
        && !path.is_absolute();
    if is_plain_relative {
        Ok(())
    } else {
        Err(Error::ModelLoadError(format!(
            "Invalid Granite Speech safetensors shard path '{file}'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp_model_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("granite-speech-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_required_non_weight_files(model_dir: &Path) {
        std::fs::write(model_dir.join("chat_template.jinja"), "{{ prompt }}").unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("generation_config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("processor_config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(model_dir.join("tokenizer_config.json"), "{}").unwrap();
    }

    #[test]
    fn artifact_validation_requires_index_weight_map() {
        let dir = temp_model_dir();
        write_required_non_weight_files(&dir);
        std::fs::write(dir.join("model.safetensors.index.json"), "{}").unwrap();
        let err = ensure_granite_speech_artifacts(&dir).unwrap_err();
        assert!(err.to_string().contains("weight_map"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn artifact_validation_returns_unique_shards() {
        let dir = temp_model_dir();
        write_required_non_weight_files(&dir);
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00003.safetensors","b":"model-00002-of-00003.safetensors","c":"model-00002-of-00003.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("model-00001-of-00003.safetensors"), []).unwrap();
        std::fs::write(dir.join("model-00002-of-00003.safetensors"), []).unwrap();
        let shards = ensure_granite_speech_artifacts(&dir).unwrap();
        assert_eq!(shards.len(), 2);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn artifact_validation_rejects_path_traversal_shards() {
        let dir = temp_model_dir();
        write_required_non_weight_files(&dir);
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"../model-00001-of-00003.safetensors"}}"#,
        )
        .unwrap();
        let err = ensure_granite_speech_artifacts(&dir).unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid Granite Speech safetensors shard path"));
        std::fs::remove_dir_all(dir).ok();
    }
}
