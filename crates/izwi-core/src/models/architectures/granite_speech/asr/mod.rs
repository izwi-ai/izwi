//! Native Granite Speech ASR facade.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use candle_core::{DType, Tensor};
use serde_json::json;
use tracing::info;

use crate::backends::{parse_dtype_name, DeviceKind, DeviceProfile};
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
    GraniteSpeechAttentionDecodeProfile, GraniteSpeechAudioEmbeddingStats,
    GraniteSpeechDecodeLoopProfile, GraniteSpeechDecodeProfile, GraniteSpeechForwardProfile,
    GraniteSpeechGeneration, GraniteSpeechGenerationStats, GraniteSpeechGenerationTimings,
    GraniteSpeechLayerDecodeProfile, GraniteSpeechMlpDecodeProfile, GraniteSpeechRuntime,
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
const AUDIO_EMBEDDING_CACHE_MAX_SECONDS: f32 = 60.0;

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
    audio_embedding_cache: Mutex<Option<GraniteSpeechAudioEmbeddingCacheEntry>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraniteSpeechAudioEmbeddingCacheKey {
    sample_rate: u32,
    sample_count: usize,
    hash: u64,
}

#[derive(Clone)]
struct GraniteSpeechAudioEmbeddingCacheEntry {
    key: GraniteSpeechAudioEmbeddingCacheKey,
    features: GraniteSpeechAudioFeatures,
    audio_embeds: Tensor,
    stats: GraniteSpeechAudioEmbeddingStats,
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
        let dtype = select_granite_speech_dtype(&device, config.target_dtype_hint())?;
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
            audio_embedding_cache: Mutex::new(None),
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
            GraniteSpeechTask::Asr,
            prompt,
            None,
            options,
            &mut no_op,
            false,
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
            GraniteSpeechTask::Asr,
            prompt,
            prefix_text,
            options,
            &mut no_op,
            false,
        )
    }

    pub fn transcribe_with_details_task_prefix_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            task,
            None,
            prefix_text,
            options,
            &mut no_op,
            false,
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
            GraniteSpeechTask::Asr,
            prompt,
            None,
            options,
            on_delta,
            true,
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
        task: GraniteSpeechTask,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
        emit_deltas: bool,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let model_start = Instant::now();
        let cache_eligible = granite_audio_embedding_cache_eligible(audio.len(), sample_rate);
        let cache_key =
            cache_eligible.then(|| granite_audio_embedding_cache_key(audio, sample_rate));
        let prepare_start = Instant::now();
        let cached = cache_key.and_then(|key| self.cached_audio_embedding(key));
        let (features, audio_embeds, audio_stats, mel_prepare, encoder_forward, audio_cache_hit) =
            if let Some(cached) = cached {
                let stats = granite_cached_audio_embedding_stats(cached.stats);
                (
                    cached.features,
                    cached.audio_embeds,
                    stats,
                    prepare_start.elapsed(),
                    Duration::ZERO,
                    true,
                )
            } else {
                let features = self.prepare_audio_features(audio, sample_rate)?;
                let mel_prepare = prepare_start.elapsed();
                validate_granite_audio_duration(features.audio_seconds)?;
                let encoder_start = Instant::now();
                let (audio_embeds, audio_stats) =
                    self.runtime.audio_embeddings_with_stats(&features)?;
                let encoder_forward = encoder_start.elapsed();
                if let Some(cache_key) = cache_key {
                    self.store_audio_embedding(
                        cache_key,
                        features.clone(),
                        audio_embeds.clone(),
                        audio_stats,
                    );
                }
                (
                    features,
                    audio_embeds,
                    audio_stats,
                    mel_prepare,
                    encoder_forward,
                    false,
                )
            };
        validate_granite_audio_duration(features.audio_seconds)?;

        let prompt_options = GraniteSpeechPromptOptions {
            task,
            language: language.map(str::to_string),
            custom_prompt: prompt.map(str::to_string),
            prefix_text: prefix_text.map(str::to_string),
            ..GraniteSpeechPromptOptions::default()
        };
        let granite_prompt = self.build_prompt(&prompt_options)?;
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
            emit_deltas,
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
            audio_cache_hit,
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

    fn cached_audio_embedding(
        &self,
        key: GraniteSpeechAudioEmbeddingCacheKey,
    ) -> Option<GraniteSpeechAudioEmbeddingCacheEntry> {
        let guard = self.audio_embedding_cache.lock().ok()?;
        guard.as_ref().filter(|entry| entry.key == key).cloned()
    }

    fn store_audio_embedding(
        &self,
        key: GraniteSpeechAudioEmbeddingCacheKey,
        features: GraniteSpeechAudioFeatures,
        audio_embeds: Tensor,
        stats: GraniteSpeechAudioEmbeddingStats,
    ) {
        if let Ok(mut guard) = self.audio_embedding_cache.lock() {
            *guard = Some(GraniteSpeechAudioEmbeddingCacheEntry {
                key,
                features,
                audio_embeds,
                stats,
            });
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraniteSpeechAsrTimings {
    mel_prepare: Duration,
    encoder_forward: Duration,
    prefill: Duration,
    decode: Duration,
    model_total: Duration,
    audio_cache_hit: bool,
}

fn granite_audio_embedding_cache_eligible(sample_count: usize, sample_rate: u32) -> bool {
    sample_rate > 0
        && sample_count > 0
        && sample_count as f32 / sample_rate as f32 <= AUDIO_EMBEDDING_CACHE_MAX_SECONDS
}

fn granite_audio_embedding_cache_key(
    audio: &[f32],
    sample_rate: u32,
) -> GraniteSpeechAudioEmbeddingCacheKey {
    let mut hasher = DefaultHasher::new();
    sample_rate.hash(&mut hasher);
    audio.len().hash(&mut hasher);
    for sample in audio {
        sample.to_bits().hash(&mut hasher);
    }
    GraniteSpeechAudioEmbeddingCacheKey {
        sample_rate,
        sample_count: audio.len(),
        hash: hasher.finish(),
    }
}

fn granite_cached_audio_embedding_stats(
    stats: GraniteSpeechAudioEmbeddingStats,
) -> GraniteSpeechAudioEmbeddingStats {
    GraniteSpeechAudioEmbeddingStats {
        upload: Duration::ZERO,
        encoder: Duration::ZERO,
        projector: Duration::ZERO,
        ..stats
    }
}

fn validate_granite_audio_duration(audio_seconds: f32) -> Result<()> {
    if audio_seconds <= DEFAULT_MAX_AUDIO_SECONDS {
        return Ok(());
    }
    Err(Error::InvalidInput(format!(
        "Granite Speech ASR supports audio up to {DEFAULT_MAX_AUDIO_SECONDS:.0}s, got {audio_seconds:.1}s"
    )))
}

fn select_granite_speech_dtype(device: &DeviceProfile, config_hint: Option<&str>) -> Result<DType> {
    let explicit = std::env::var("IZWI_GRANITE_SPEECH_DTYPE")
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty());

    if let Some(raw) = explicit.as_deref() {
        return resolve_explicit_granite_speech_dtype(device, raw);
    }

    config_hint
        .map(|raw| {
            device.select_model_dtype_checked(
                ModelFamily::GraniteSpeechAsr,
                Some(raw),
                "Granite Speech ASR",
            )
        })
        .transpose()
        .map(|dtype| {
            dtype.unwrap_or_else(|| device.select_model_dtype(ModelFamily::GraniteSpeechAsr, None))
        })
}

fn resolve_explicit_granite_speech_dtype(device: &DeviceProfile, raw: &str) -> Result<DType> {
    let dtype = parse_dtype_name(raw).ok_or_else(|| {
        Error::InvalidInput(format!(
            "Invalid Granite Speech ASR dtype override {raw:?}: expected one of f32, f16, or bf16"
        ))
    })?;
    match device.kind {
        DeviceKind::Metal => match dtype {
            DType::F32 => Ok(DType::F32),
            DType::F16 if device.capabilities.supports_f16 => Ok(DType::F16),
            DType::F16 => Err(Error::InvalidInput(
                "Invalid Granite Speech ASR dtype override \"f16\": Metal device does not report F16 support"
                    .to_string(),
            )),
            DType::BF16 => Err(Error::InvalidInput(
                "Invalid Granite Speech ASR dtype override \"bf16\": BF16 is not supported on Metal"
                    .to_string(),
            )),
            _ => Err(Error::InvalidInput(format!(
                "Invalid Granite Speech ASR dtype override {raw:?}: dtype is not supported"
            ))),
        },
        _ => device.select_model_dtype_checked(
            ModelFamily::GraniteSpeechAsr,
            Some(raw),
            "Granite Speech ASR",
        ),
    }
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
            "max_new_tokens": generation.stats.max_new_tokens,
            "stop_reason": generation.stats.stop_reason,
            "stop_token": generation.stats.stop_token,
        },
        "token_debug": granite_token_debug_enabled().then(|| json!({
            "token_ids": generation.token_ids,
            "token_count": generation.token_ids.len(),
            "decoded_text": generation.text,
        })),
        "execution": {
            "dense_decode_cache": generation.stats.dense_decode_cache_enabled,
            "dense_decode_cache_configured": generation.stats.dense_decode_cache_enabled,
            "dense_head_decode_enabled": generation.stats.dense_head_decode_enabled,
            "qkv_projection_fused": generation.stats.qkv_projection_fused,
            "gate_up_projection_fused": generation.stats.gate_up_projection_fused,
            "rope_cache_precomputed": generation.stats.rope_cache_precomputed,
            "cuda_dense_decode_cache": generation.stats.dense_decode_cache_enabled,
            "cuda_device_argmax": generation.stats.cuda_device_argmax,
            "residual_branches_prescaled": generation.stats.residual_branches_prescaled,
            "f16_lm_head": generation.stats.f16_lm_head,
            "dense_decode_preallocated": generation.stats.dense_decode_preallocated,
            "dense_decode_initial_capacity": generation.stats.dense_decode_initial_capacity,
            "deferred_stop_check": generation.stats.deferred_stop_check,
            "chunked_stop_check": generation.stats.chunked_stop_check,
            "stop_check_interval": generation.stats.stop_check_interval,
            "dense_decode_max_tokens": generation.stats.dense_decode_max_tokens,
            "audio_embedding_cache_hit": timings.audio_cache_hit,
        },
        "decode_profile": generation
            .stats
            .decode_profile
            .as_ref()
            .map(granite_decode_profile_json),
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

fn granite_decode_profile_json(profile: &GraniteSpeechDecodeProfile) -> serde_json::Value {
    let layer_totals = granite_layer_totals(&profile.layers);
    json!({
        "enabled": true,
        "timing_kind": profile.timing_kind,
        "steps": profile.steps,
        "layer_count": profile.layer_count,
        "step_total_ms": duration_stats_json(&profile.step_total_samples),
        "loop_totals_ms": decode_loop_profile_json(profile.totals),
        "forward_totals_ms": forward_profile_json(profile.forward),
        "decoder_totals_ms": layer_profile_json(layer_totals),
        "layers": profile.layers.iter().enumerate().map(|(idx, layer)| {
            json!({
                "index": idx,
                "timings_ms": layer_profile_json(*layer),
            })
        }).collect::<Vec<_>>(),
    })
}

fn granite_layer_totals(
    layers: &[GraniteSpeechLayerDecodeProfile],
) -> GraniteSpeechLayerDecodeProfile {
    let mut total = GraniteSpeechLayerDecodeProfile::default();
    for layer in layers {
        total.total += layer.total;
        total.input_norm += layer.input_norm;
        total.attention.qkv += layer.attention.qkv;
        total.attention.rope += layer.attention.rope;
        total.attention.cache += layer.attention.cache;
        total.attention.kernel += layer.attention.kernel;
        total.attention.output += layer.attention.output;
        total.post_attention_norm += layer.post_attention_norm;
        total.mlp.gate_up += layer.mlp.gate_up;
        total.mlp.activation += layer.mlp.activation;
        total.mlp.down += layer.mlp.down;
        total.residual += layer.residual;
    }
    total
}

fn decode_loop_profile_json(profile: GraniteSpeechDecodeLoopProfile) -> serde_json::Value {
    json!({
        "argmax": duration_ms(profile.argmax),
        "scalar_read": duration_ms(profile.scalar_read),
        "stop_check": duration_ms(profile.stop_check),
        "model_forward": duration_ms(profile.model_forward),
        "text_decode": duration_ms(profile.text_decode),
        "delta_emit": duration_ms(profile.delta_emit),
        "step_total": duration_ms(profile.step_total),
    })
}

fn forward_profile_json(profile: GraniteSpeechForwardProfile) -> serde_json::Value {
    json!({
        "token_embedding": duration_ms(profile.token_embedding),
        "rope_build": duration_ms(profile.rope_build),
        "layers_total": duration_ms(profile.layers_total),
        "final_norm": duration_ms(profile.final_norm),
        "lm_head": duration_ms(profile.lm_head),
    })
}

fn layer_profile_json(profile: GraniteSpeechLayerDecodeProfile) -> serde_json::Value {
    json!({
        "total": duration_ms(profile.total),
        "input_norm": duration_ms(profile.input_norm),
        "attention": attention_profile_json(profile.attention),
        "post_attention_norm": duration_ms(profile.post_attention_norm),
        "mlp": mlp_profile_json(profile.mlp),
        "residual": duration_ms(profile.residual),
    })
}

fn attention_profile_json(profile: GraniteSpeechAttentionDecodeProfile) -> serde_json::Value {
    json!({
        "qkv": duration_ms(profile.qkv),
        "rope": duration_ms(profile.rope),
        "cache": duration_ms(profile.cache),
        "kernel": duration_ms(profile.kernel),
        "output": duration_ms(profile.output),
    })
}

fn mlp_profile_json(profile: GraniteSpeechMlpDecodeProfile) -> serde_json::Value {
    json!({
        "gate_up": duration_ms(profile.gate_up),
        "activation": duration_ms(profile.activation),
        "down": duration_ms(profile.down),
    })
}

fn duration_stats_json(samples: &[Duration]) -> serde_json::Value {
    if samples.is_empty() {
        return json!({
            "count": 0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        });
    }
    let mut values = samples
        .iter()
        .map(|value| duration_ms(*value))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let sum = values.iter().sum::<f64>();
    json!({
        "count": values.len(),
        "avg": sum / values.len() as f64,
        "p50": percentile_sorted(&values, 0.50),
        "p95": percentile_sorted(&values, 0.95),
        "max": values.last().copied().unwrap_or(0.0),
    })
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[idx.min(values.len() - 1)]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn granite_token_debug_enabled() -> bool {
    std::env::var("IZWI_GRANITE_TOKEN_DEBUG")
        .ok()
        .or_else(|| std::env::var("IZWI_GRANITE_TOKEN_DIAGNOSTICS").ok())
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(false)
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
    use crate::backends::DeviceCapabilities;
    use uuid::Uuid;

    static GRANITE_DTYPE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

    fn test_profile(kind: DeviceKind, supports_f16: bool) -> DeviceProfile {
        DeviceProfile {
            device: candle_core::Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                supports_f16,
                prefers_f32: kind.is_metal(),
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    #[test]
    fn granite_speech_dtype_config_hint_keeps_existing_metal_f32_policy() {
        let _guard = GRANITE_DTYPE_ENV_LOCK.lock().unwrap();
        std::env::remove_var("IZWI_GRANITE_SPEECH_DTYPE");
        let metal = test_profile(DeviceKind::Metal, true);

        assert_eq!(
            select_granite_speech_dtype(&metal, Some("torch.bfloat16")).unwrap(),
            DType::F32
        );
    }

    #[test]
    fn granite_speech_dtype_explicit_f16_can_opt_into_metal_half_precision() {
        let _guard = GRANITE_DTYPE_ENV_LOCK.lock().unwrap();
        std::env::set_var("IZWI_GRANITE_SPEECH_DTYPE", "f16");
        let metal = test_profile(DeviceKind::Metal, true);

        assert_eq!(
            select_granite_speech_dtype(&metal, None).unwrap(),
            DType::F16
        );
        std::env::remove_var("IZWI_GRANITE_SPEECH_DTYPE");
    }

    #[test]
    fn granite_speech_dtype_rejects_bf16_on_metal() {
        let _guard = GRANITE_DTYPE_ENV_LOCK.lock().unwrap();
        std::env::set_var("IZWI_GRANITE_SPEECH_DTYPE", "bf16");
        let metal = test_profile(DeviceKind::Metal, true);

        let err = select_granite_speech_dtype(&metal, None).unwrap_err();
        assert!(err.to_string().contains("BF16 is not supported on Metal"));
        std::env::remove_var("IZWI_GRANITE_SPEECH_DTYPE");
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
                max_new_tokens: 128,
                stop_reason: "stop_token".to_string(),
                stop_token: Some(100_257),
                dense_decode_cache_enabled: true,
                dense_head_decode_enabled: false,
                qkv_projection_fused: true,
                gate_up_projection_fused: true,
                rope_cache_precomputed: true,
                cuda_device_argmax: false,
                residual_branches_prescaled: true,
                f16_lm_head: false,
                dense_decode_preallocated: true,
                dense_decode_initial_capacity: 512,
                deferred_stop_check: true,
                chunked_stop_check: false,
                stop_check_interval: 1,
                dense_decode_max_tokens: 8192,
                timings: GraniteSpeechGenerationTimings {
                    prefill: Duration::from_millis(7),
                    decode: Duration::from_millis(3),
                },
                decode_profile: Some(GraniteSpeechDecodeProfile {
                    timing_kind: "host_wall_clock_no_device_sync",
                    steps: 2,
                    layer_count: 1,
                    step_total_samples: vec![Duration::from_millis(2), Duration::from_millis(3)],
                    totals: GraniteSpeechDecodeLoopProfile {
                        argmax: Duration::from_millis(1),
                        scalar_read: Duration::from_millis(1),
                        stop_check: Duration::from_millis(1),
                        model_forward: Duration::from_millis(4),
                        text_decode: Duration::from_millis(1),
                        delta_emit: Duration::ZERO,
                        step_total: Duration::from_millis(5),
                    },
                    forward: GraniteSpeechForwardProfile {
                        token_embedding: Duration::from_millis(1),
                        rope_build: Duration::from_millis(1),
                        layers_total: Duration::from_millis(2),
                        final_norm: Duration::from_millis(1),
                        lm_head: Duration::from_millis(1),
                    },
                    layers: vec![GraniteSpeechLayerDecodeProfile {
                        total: Duration::from_millis(2),
                        input_norm: Duration::from_millis(1),
                        attention: GraniteSpeechAttentionDecodeProfile {
                            qkv: Duration::from_millis(1),
                            rope: Duration::from_millis(1),
                            cache: Duration::from_millis(1),
                            kernel: Duration::from_millis(1),
                            output: Duration::from_millis(1),
                        },
                        post_attention_norm: Duration::from_millis(1),
                        mlp: GraniteSpeechMlpDecodeProfile {
                            gate_up: Duration::from_millis(1),
                            activation: Duration::from_millis(1),
                            down: Duration::from_millis(1),
                        },
                        residual: Duration::from_millis(1),
                    }],
                }),
            },
        };
        let timings = GraniteSpeechAsrTimings {
            mel_prepare: Duration::from_millis(1),
            encoder_forward: Duration::from_millis(2),
            prefill: generation.stats.timings.prefill,
            decode: generation.stats.timings.decode,
            model_total: Duration::from_millis(12),
            audio_cache_hit: true,
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
        assert_eq!(diagnostics["decode"]["max_new_tokens"], 128);
        assert_eq!(diagnostics["execution"]["dense_decode_cache"], true);
        assert_eq!(
            diagnostics["execution"]["dense_decode_cache_configured"],
            true
        );
        assert_eq!(diagnostics["execution"]["dense_head_decode_enabled"], false);
        assert_eq!(diagnostics["execution"]["qkv_projection_fused"], true);
        assert_eq!(diagnostics["execution"]["gate_up_projection_fused"], true);
        assert_eq!(diagnostics["execution"]["rope_cache_precomputed"], true);
        assert_eq!(diagnostics["execution"]["cuda_device_argmax"], false);
        assert_eq!(
            diagnostics["execution"]["residual_branches_prescaled"],
            true
        );
        assert_eq!(diagnostics["execution"]["dense_decode_preallocated"], true);
        assert_eq!(
            diagnostics["execution"]["dense_decode_initial_capacity"],
            512
        );
        assert_eq!(diagnostics["execution"]["deferred_stop_check"], true);
        assert_eq!(diagnostics["execution"]["chunked_stop_check"], false);
        assert_eq!(diagnostics["execution"]["stop_check_interval"], 1);
        assert_eq!(diagnostics["execution"]["dense_decode_max_tokens"], 8192);
        assert_eq!(diagnostics["execution"]["audio_embedding_cache_hit"], true);
        assert_eq!(diagnostics["decode_profile"]["enabled"], true);
        assert_eq!(
            diagnostics["decode_profile"]["timing_kind"],
            "host_wall_clock_no_device_sync"
        );
        assert_eq!(diagnostics["decode_profile"]["steps"], 2);
        assert_eq!(diagnostics["decode_profile"]["step_total_ms"]["count"], 2);
        assert_eq!(diagnostics["decode_profile"]["step_total_ms"]["p50"], 3.0);
        assert_eq!(
            diagnostics["decode_profile"]["loop_totals_ms"]["model_forward"],
            4.0
        );
        assert_eq!(
            diagnostics["decode_profile"]["decoder_totals_ms"]["attention"]["cache"],
            1.0
        );
        assert_eq!(
            diagnostics["decode_profile"]["layers"][0]["timings_ms"]["mlp"]["down"],
            1.0
        );
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

    #[test]
    fn audio_embedding_cache_key_tracks_exact_audio_and_rate() {
        let audio = [0.0f32, 0.25, -0.5, f32::INFINITY];
        let same = granite_audio_embedding_cache_key(&audio, 16_000);
        assert_eq!(same, granite_audio_embedding_cache_key(&audio, 16_000));

        let mut changed_audio = audio;
        changed_audio[1] = 0.5;
        assert_ne!(
            same,
            granite_audio_embedding_cache_key(&changed_audio, 16_000)
        );
        assert_ne!(same, granite_audio_embedding_cache_key(&audio, 8_000));
    }

    #[test]
    fn audio_embedding_cache_is_bounded_to_short_audio() {
        assert!(granite_audio_embedding_cache_eligible(16_000, 16_000));
        assert!(granite_audio_embedding_cache_eligible(60 * 16_000, 16_000));
        assert!(!granite_audio_embedding_cache_eligible(
            60 * 16_000 + 1,
            16_000
        ));
        assert!(!granite_audio_embedding_cache_eligible(1, 0));
        assert!(!granite_audio_embedding_cache_eligible(0, 16_000));
    }

    #[test]
    fn cached_audio_embedding_stats_zero_current_run_timings() {
        let stats = GraniteSpeechAudioEmbeddingStats {
            upload: Duration::from_millis(1),
            encoder: Duration::from_millis(2),
            projector: Duration::from_millis(3),
            encoder_frames: 4,
            encoder_dim: 5,
            conformer_context_size: 6,
            conformer_blocks: 7,
            conformer_pad_frames: 8,
            conformer_layers: 9,
            qformer_windows: 10,
            qformer_window_size: 11,
            qformer_queries_per_window: 12,
            qformer_layers: 13,
        };
        let cached = granite_cached_audio_embedding_stats(stats);

        assert_eq!(cached.upload, Duration::ZERO);
        assert_eq!(cached.encoder, Duration::ZERO);
        assert_eq!(cached.projector, Duration::ZERO);
        assert_eq!(cached.encoder_frames, stats.encoder_frames);
        assert_eq!(cached.qformer_windows, stats.qformer_windows);
    }
}
