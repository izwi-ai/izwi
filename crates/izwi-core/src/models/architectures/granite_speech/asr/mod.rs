//! Native Granite Speech ASR facade.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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
mod runtime;
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
pub use runtime::{
    GraniteSpeechAudioEmbeddingStats, GraniteSpeechGeneration, GraniteSpeechGenerationStats,
    GraniteSpeechGenerationTimings, GraniteSpeechRuntime,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechAsrGenerationOptions {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for GraniteSpeechAsrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 768,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        }
    }
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
    runtime: GraniteSpeechRuntime,
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
        let runtime = GraniteSpeechRuntime::load(&shards, &config, &device, dtype)?;

        info!(
            "Loaded Granite Speech ASR in {:?} on {:?} with dtype {:?} ({} shard files)",
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
            runtime,
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
        self.transcribe_with_details_and_prompt_and_options(
            audio,
            sample_rate,
            language,
            prompt,
            GraniteSpeechAsrGenerationOptions::default(),
        )
    }

    pub fn transcribe_with_details_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            prompt,
            None,
            options,
            &mut no_op,
        )
    }

    pub fn transcribe_with_details_and_prompt_prefix_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            prompt,
            prefix_text,
            options,
            &mut no_op,
        )
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_with_callback_and_prompt_and_options(
            audio,
            sample_rate,
            language,
            prompt,
            GraniteSpeechAsrGenerationOptions::default(),
            on_delta,
        )
        .map(|output| output.text)
    }

    pub fn transcribe_with_callback_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            prompt,
            None,
            options,
            on_delta,
        )
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

    fn transcribe_internal(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let model_start = Instant::now();
        let prepare_start = Instant::now();
        let features = self.prepare_audio_features(audio, sample_rate)?;
        let mel_prepare = prepare_start.elapsed();
        validate_granite_audio_duration(features.audio_seconds)?;

        let prompt_options = GraniteSpeechPromptOptions {
            task: GraniteSpeechTask::Asr,
            language: language.map(str::to_string),
            custom_prompt: prompt.map(str::to_string),
            prefix_text: prefix_text.map(str::to_string),
            ..GraniteSpeechPromptOptions::default()
        };
        let granite_prompt = self.build_prompt(&prompt_options)?;
        let encoder_start = Instant::now();
        let (audio_embeds, audio_stats) = self.runtime.audio_embeddings_with_stats(&features)?;
        let encoder_forward = encoder_start.elapsed();
        let special_tokens = self.prompt_tokenizer.special_tokens().clone();
        let mut decode = |ids: &[u32]| self.prompt_tokenizer.decode(ids);
        let generation = self.runtime.generate(
            &granite_prompt,
            &special_tokens,
            &audio_embeds,
            options.max_new_tokens,
            &options.stop_token_ids,
            &options.stop_sequences,
            &mut decode,
            on_delta,
        )?;
        let model_total = model_start.elapsed();
        let parsed = parse_granite_speech_output(&generation.text);
        let text = parsed.text.clone();
        let timings = GraniteSpeechAsrTimings {
            mel_prepare,
            encoder_forward,
            prefill: generation.stats.timings.prefill,
            decode: generation.stats.timings.decode,
            model_total,
        };

        Ok(GraniteSpeechAsrTranscriptionOutput {
            text,
            language: language.map(str::to_string),
            diagnostics: Some(granite_diagnostics(
                &features,
                &granite_prompt,
                &generation,
                &parsed,
                self.dtype,
                &self.device,
                timings,
                audio_stats,
            )),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraniteSpeechAsrTimings {
    mel_prepare: Duration,
    encoder_forward: Duration,
    prefill: Duration,
    decode: Duration,
    model_total: Duration,
}

fn validate_granite_audio_duration(audio_seconds: f32) -> Result<()> {
    if audio_seconds <= DEFAULT_MAX_AUDIO_SECONDS {
        return Ok(());
    }
    Err(Error::InvalidInput(format!(
        "Granite Speech ASR supports audio up to {DEFAULT_MAX_AUDIO_SECONDS:.0}s, got {audio_seconds:.1}s"
    )))
}

fn granite_diagnostics(
    features: &GraniteSpeechAudioFeatures,
    prompt: &GraniteSpeechPrompt,
    generation: &GraniteSpeechGeneration,
    parsed: &GraniteSpeechParsedTranscript,
    dtype: DType,
    device: &DeviceProfile,
    timings: GraniteSpeechAsrTimings,
    audio_stats: GraniteSpeechAudioEmbeddingStats,
) -> serde_json::Value {
    json!({
        "family": "granite_speech_asr",
        "dtype": format!("{:?}", dtype),
        "device_kind": format!("{:?}", device.kind),
        "audio_seconds": features.audio_seconds,
        "sample_rate": features.sample_rate,
        "mel_frames": features.mel_frames,
        "mel_bins": features.mel_bins,
        "encoder_frames": features.encoder_frames,
        "encoder_dim": features.encoder_dim,
        "projected_audio_tokens": generation.stats.audio_tokens,
        "prompt_tokens": generation.stats.prompt_tokens,
        "prompt_prefix_tokens": prompt.prefix_text_token_count,
        "prompt_audio_placeholders": prompt.audio_token_positions.len(),
        "prompt": {
            "prompt_tokens": generation.stats.prompt_tokens,
            "prefix_tokens": prompt.prefix_text_token_count,
            "audio_placeholders": prompt.audio_token_positions.len(),
        },
        "audio": {
            "audio_tokens": generation.stats.audio_tokens,
            "mel_frames": features.mel_frames,
            "encoder_frames": features.encoder_frames,
            "encoder_dim": features.encoder_dim,
            "conformer_context_size": audio_stats.conformer_context_size,
            "conformer_blocks": audio_stats.conformer_blocks,
            "conformer_pad_frames": audio_stats.conformer_pad_frames,
            "conformer_layers": audio_stats.conformer_layers,
            "qformer_windows": audio_stats.qformer_windows,
            "qformer_window_size": audio_stats.qformer_window_size,
            "qformer_queries_per_window": audio_stats.qformer_queries_per_window,
            "qformer_layers": audio_stats.qformer_layers,
        },
        "generated_tokens": generation.stats.generated_tokens,
        "stop_reason": generation.stats.stop_reason,
        "stop_token": generation.stats.stop_token,
        "decode": {
            "generated_tokens": generation.stats.generated_tokens,
            "stop_reason": generation.stats.stop_reason,
            "stop_token": generation.stats.stop_token,
        },
        "execution": {
            "dense_decode_cache": generation.stats.dense_decode_cache_enabled,
            "dense_decode_cache_configured": generation.stats.dense_decode_cache_enabled,
            "dense_head_decode_enabled": generation.stats.dense_head_decode_enabled,
            "cuda_dense_decode_cache": generation.stats.dense_decode_cache_enabled,
            "dense_decode_max_tokens": generation.stats.dense_decode_max_tokens,
        },
        "timings_ms": {
            "mel_prepare": duration_ms(timings.mel_prepare),
            "encoder_forward": duration_ms(timings.encoder_forward),
            "audio_input_upload": duration_ms(audio_stats.upload),
            "audio_encoder": duration_ms(audio_stats.encoder),
            "audio_projector": duration_ms(audio_stats.projector),
            "audio_frontend_total": duration_ms(timings.mel_prepare + timings.encoder_forward),
            "prefill": duration_ms(timings.prefill),
            "decode": duration_ms(timings.decode),
            "generation_total": duration_ms(timings.prefill + timings.decode),
            "model_non_generation": duration_ms(
                timings.model_total.saturating_sub(timings.prefill + timings.decode),
            ),
            "model_total": duration_ms(timings.model_total),
        },
        "speaker_segments": parsed.segments.iter().map(|segment| {
            json!({
                "speaker": segment.speaker,
                "text": segment.text,
            })
        }).collect::<Vec<_>>(),
        "timestamp_words": parsed.timestamp_words.iter().map(|word| {
            json!({
                "word": word.word,
                "end_time_seconds": word.end_time_seconds,
            })
        }).collect::<Vec<_>>(),
    })
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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

    #[test]
    fn audio_duration_guard_allows_nine_minutes_only() {
        assert!(validate_granite_audio_duration(DEFAULT_MAX_AUDIO_SECONDS).is_ok());
        let err = validate_granite_audio_duration(DEFAULT_MAX_AUDIO_SECONDS + 0.1).unwrap_err();
        assert!(err.to_string().contains("supports audio up to 540s"));
    }

    #[test]
    fn diagnostics_include_prompt_audio_and_rich_transcript_metadata() {
        let features = GraniteSpeechAudioFeatures {
            samples: vec![0.0; 160],
            sample_rate: 16_000,
            audio_seconds: 0.01,
            mel_frames: 2,
            mel_bins: 80,
            encoder_frames: 1,
            encoder_dim: 160,
            projected_frames_hint: 3,
            log_mel: vec![vec![0.0; 80]; 2],
            input_features: vec![vec![0.0; 160]],
        };
        let prompt = GraniteSpeechPrompt {
            text: "<|audio|>".to_string(),
            input_ids: vec![100_352],
            audio_token_positions: vec![0],
            prefix_text_token_count: 0,
        };
        let generation = GraniteSpeechGeneration {
            token_ids: vec![1, 2],
            text: "[Speaker 1]: hello [T:045]".to_string(),
            stats: GraniteSpeechGenerationStats {
                prompt_tokens: 4,
                audio_tokens: 3,
                generated_tokens: 2,
                stop_reason: "stop_token".to_string(),
                stop_token: Some(100_257),
                dense_decode_cache_enabled: true,
                dense_head_decode_enabled: false,
                dense_decode_max_tokens: 8192,
                timings: GraniteSpeechGenerationTimings {
                    prefill: Duration::from_millis(7),
                    decode: Duration::from_millis(3),
                },
            },
        };
        let timings = GraniteSpeechAsrTimings {
            mel_prepare: Duration::from_millis(1),
            encoder_forward: Duration::from_millis(2),
            prefill: generation.stats.timings.prefill,
            decode: generation.stats.timings.decode,
            model_total: Duration::from_millis(12),
        };
        let audio_stats = GraniteSpeechAudioEmbeddingStats {
            upload: Duration::from_millis(1),
            encoder: Duration::from_millis(2),
            projector: Duration::from_millis(3),
            encoder_frames: 1,
            encoder_dim: 160,
            conformer_context_size: 200,
            conformer_blocks: 1,
            conformer_pad_frames: 199,
            conformer_layers: 16,
            qformer_windows: 1,
            qformer_window_size: 15,
            qformer_queries_per_window: 3,
            qformer_layers: 2,
        };
        let parsed = parse_granite_speech_output(&generation.text);
        let diagnostics = granite_diagnostics(
            &features,
            &prompt,
            &generation,
            &parsed,
            DType::F32,
            &DeviceProfile::cpu(),
            timings,
            audio_stats,
        );

        assert_eq!(diagnostics["projected_audio_tokens"], 3);
        assert_eq!(diagnostics["prompt_prefix_tokens"], 0);
        assert_eq!(diagnostics["prompt_audio_placeholders"], 1);
        assert_eq!(diagnostics["prompt"]["prompt_tokens"], 4);
        assert_eq!(diagnostics["audio"]["audio_tokens"], 3);
        assert_eq!(diagnostics["audio"]["conformer_context_size"], 200);
        assert_eq!(diagnostics["audio"]["conformer_pad_frames"], 199);
        assert_eq!(diagnostics["audio"]["qformer_queries_per_window"], 3);
        assert_eq!(diagnostics["decode"]["generated_tokens"], 2);
        assert_eq!(diagnostics["execution"]["dense_decode_cache"], true);
        assert_eq!(
            diagnostics["execution"]["dense_decode_cache_configured"],
            true
        );
        assert_eq!(diagnostics["execution"]["dense_head_decode_enabled"], false);
        assert_eq!(diagnostics["execution"]["dense_decode_max_tokens"], 8192);
        assert_eq!(diagnostics["timings_ms"]["prefill"], 7.0);
        assert_eq!(diagnostics["timings_ms"]["decode"], 3.0);
        assert_eq!(diagnostics["timings_ms"]["audio_input_upload"], 1.0);
        assert_eq!(diagnostics["timings_ms"]["audio_encoder"], 2.0);
        assert_eq!(diagnostics["timings_ms"]["audio_projector"], 3.0);
        assert_eq!(diagnostics["timings_ms"]["audio_frontend_total"], 3.0);
        assert_eq!(diagnostics["timings_ms"]["generation_total"], 10.0);
        assert_eq!(diagnostics["timings_ms"]["model_non_generation"], 2.0);
        assert_eq!(diagnostics["speaker_segments"][0]["speaker"], "Speaker 1");
        assert_eq!(diagnostics["timestamp_words"][0]["word"], "hello");
    }
}
