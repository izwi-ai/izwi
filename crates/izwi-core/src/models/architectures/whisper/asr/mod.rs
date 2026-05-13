//! Native Whisper Large v3 Turbo ASR model loader and inference.
//!
//! This implementation follows Whisper prompting/decoding conventions used in:
//! - `whisper.cpp` (llama.cpp ecosystem): SOT/lang/task/no-timestamps prefix and
//!   timestamp suppression for text-only decode.
//! - Hugging Face `transformers`: language/task prompt handling and suppress token
//!   masks from `generation_config.json`.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{
    self, model::Whisper as CandleWhisper, Config as WhisperConfig,
};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rand::Rng;
use serde::Deserialize;
use serde_json::json;
use tracing::info;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::backends::{DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

use super::model::Whisper as CudaWhisper;

const SAMPLE_RATE: u32 = whisper::SAMPLE_RATE as u32;
const DEFAULT_MAX_NEW_TOKENS: usize = 448;
const MAX_AUDIO_SECONDS_HINT: f32 = whisper::CHUNK_LENGTH as f32;
const DEFAULT_TEMPERATURE_FALLBACK_INC: f32 = 0.2;
const DEFAULT_MAX_FALLBACK_RETRIES: usize = 1;
const DEFAULT_ADAPTIVE_MAX_NEW_TOKENS_PER_SECOND: f32 = 12.0;
const DEFAULT_ADAPTIVE_MIN_NEW_TOKENS: usize = 32;
const DEFAULT_ADAPTIVE_BUDGET_BUFFER_TOKENS: usize = 8;
const DEFAULT_SILENCE_TRIM_THRESHOLD_SCALE: f32 = 0.02;
const DEFAULT_SILENCE_TRIM_MIN_ABS: f32 = 0.0015;
const DEFAULT_SILENCE_TRIM_MARGIN_MS: usize = 120;
const DEFAULT_SILENCE_TRIM_MIN_LEADING_MS: usize = 500;
const DEFAULT_SILENCE_TRIM_MIN_TRAILING_MS: usize = 160;
const DEFAULT_SILENCE_TRIM_MIN_CLIP_SECS: f32 = 0.8;
const DEFAULT_INITIAL_PROMPT_MAX_TOKENS: usize = 224;
const DEFAULT_LOGPROB_THRESHOLD: f32 = -1.0;
const DEFAULT_NO_SPEECH_THRESHOLD: f32 = 0.6;
const REPETITION_GUARD_MIN_SPAN_TOKENS: usize = 8;
const REPETITION_GUARD_MAX_SPAN_TOKENS: usize = 96;
const REPETITION_GUARD_MIN_TOTAL_TOKENS: usize = 20;

#[derive(Debug, Clone, Deserialize, Default)]
struct WhisperGenerationConfig {
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    lang_to_id: HashMap<String, u32>,
    #[serde(default)]
    task_to_id: HashMap<String, u32>,
    #[serde(default)]
    no_timestamps_token_id: Option<u32>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    temperature_increment_on_fallback: Option<f32>,
    #[serde(default)]
    compression_ratio_threshold: Option<f32>,
    #[serde(default)]
    logprob_threshold: Option<f32>,
    #[serde(default)]
    no_speech_threshold: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct WhisperSpecialTokens {
    sot: u32,
    sot_prev: Option<u32>,
    transcribe: u32,
    eot: u32,
    blank: Option<u32>,
    no_timestamps: Option<u32>,
    no_speech: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct WhisperDecodeAttempt {
    text: String,
    avg_logprob: f32,
    no_speech_prob: Option<f32>,
    ended_with_eot: bool,
    repetition_loop: bool,
    compression_ratio: Option<f32>,
    generated_token_count: usize,
    sampled_token_count: usize,
    decode_steps: usize,
}

#[derive(Debug, Clone)]
struct WhisperRuntimeTuning {
    no_fallback: bool,
    max_fallback_retries: usize,
    adaptive_decode_budget: bool,
    max_new_tokens_per_second: f32,
    min_new_tokens: usize,
    max_new_tokens_cap: usize,
    decode_budget_buffer_tokens: usize,
    default_language: Option<String>,
    reuse_detected_language: bool,
    trim_silence: bool,
    silence_trim_threshold_scale: f32,
    silence_trim_min_abs: f32,
    silence_trim_margin_ms: usize,
    silence_trim_min_leading_ms: usize,
    silence_trim_min_trailing_ms: usize,
    silence_trim_min_clip_secs: f32,
    suppress_blank: bool,
    suppress_numerals: bool,
    initial_prompt_max_tokens: usize,
}

impl WhisperRuntimeTuning {
    fn from_env() -> Self {
        Self {
            no_fallback: env_bool("IZWI_WHISPER_NO_FALLBACK").unwrap_or(false),
            max_fallback_retries: env_usize("IZWI_WHISPER_MAX_FALLBACK_RETRIES")
                .unwrap_or(DEFAULT_MAX_FALLBACK_RETRIES),
            adaptive_decode_budget: env_bool("IZWI_WHISPER_ADAPTIVE_MAX_NEW_TOKENS")
                .unwrap_or(true),
            max_new_tokens_per_second: env_f32("IZWI_WHISPER_MAX_NEW_TOKENS_PER_SECOND")
                .unwrap_or(DEFAULT_ADAPTIVE_MAX_NEW_TOKENS_PER_SECOND),
            min_new_tokens: env_usize("IZWI_WHISPER_MIN_NEW_TOKENS")
                .unwrap_or(DEFAULT_ADAPTIVE_MIN_NEW_TOKENS),
            max_new_tokens_cap: env_usize("IZWI_WHISPER_MAX_NEW_TOKENS_CAP")
                .unwrap_or(DEFAULT_MAX_NEW_TOKENS),
            decode_budget_buffer_tokens: env_usize("IZWI_WHISPER_MAX_NEW_TOKENS_BUFFER")
                .unwrap_or(DEFAULT_ADAPTIVE_BUDGET_BUFFER_TOKENS),
            default_language: env_nonempty_string("IZWI_WHISPER_DEFAULT_LANGUAGE"),
            reuse_detected_language: env_bool("IZWI_WHISPER_REUSE_DETECTED_LANGUAGE")
                .unwrap_or(true),
            trim_silence: env_bool("IZWI_WHISPER_TRIM_SILENCE").unwrap_or(true),
            silence_trim_threshold_scale: env_f32("IZWI_WHISPER_SILENCE_TRIM_THRESHOLD_SCALE")
                .unwrap_or(DEFAULT_SILENCE_TRIM_THRESHOLD_SCALE),
            silence_trim_min_abs: env_f32("IZWI_WHISPER_SILENCE_TRIM_MIN_ABS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_ABS),
            silence_trim_margin_ms: env_usize("IZWI_WHISPER_SILENCE_TRIM_MARGIN_MS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MARGIN_MS),
            silence_trim_min_leading_ms: env_usize("IZWI_WHISPER_SILENCE_TRIM_MIN_LEADING_MS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_LEADING_MS),
            silence_trim_min_trailing_ms: env_usize("IZWI_WHISPER_SILENCE_TRIM_MIN_TRAILING_MS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_TRAILING_MS),
            silence_trim_min_clip_secs: env_f32("IZWI_WHISPER_SILENCE_TRIM_MIN_CLIP_SECS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_CLIP_SECS),
            suppress_blank: env_bool("IZWI_WHISPER_SUPPRESS_BLANK").unwrap_or(true),
            suppress_numerals: env_bool("IZWI_WHISPER_SUPPRESS_NUMERALS").unwrap_or(false),
            initial_prompt_max_tokens: env_usize("IZWI_WHISPER_INITIAL_PROMPT_MAX_TOKENS")
                .unwrap_or(DEFAULT_INITIAL_PROMPT_MAX_TOKENS),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct WhisperPromptDiagnostics {
    initial_prompt_requested: bool,
    initial_prompt_token_count: usize,
    initial_prompt_tokens_used: usize,
    initial_prompt_tokens_truncated: usize,
    initial_prompt_max_tokens: usize,
    previous_context_token_id: Option<u32>,
    rolling_context_enabled: bool,
}

#[derive(Debug, Clone)]
struct WhisperPromptPrefix {
    ids: Vec<u32>,
    diagnostics: WhisperPromptDiagnostics,
}

#[derive(Debug, Clone)]
struct WhisperLanguageResolution {
    resolved: Option<(u32, String)>,
    hint_used: bool,
    detect_ms: f64,
    strategy: &'static str,
}

enum WhisperModel {
    Upstream(CandleWhisper),
    Cuda(CudaWhisper),
}

impl WhisperModel {
    fn load(vb: &VarBuilder, config: WhisperConfig, use_cuda_dtype_shim: bool) -> Result<Self> {
        if use_cuda_dtype_shim {
            CudaWhisper::load(vb, config)
                .map(Self::Cuda)
                .map_err(Error::from)
        } else {
            CandleWhisper::load(vb, config)
                .map(Self::Upstream)
                .map_err(Error::from)
        }
    }

    fn reset_kv_cache(&mut self) {
        match self {
            Self::Upstream(model) => model.reset_kv_cache(),
            Self::Cuda(model) => model.reset_kv_cache(),
        }
    }

    fn encoder_forward(&mut self, x: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        match self {
            Self::Upstream(model) => model
                .encoder
                .forward(x, flush_kv_cache)
                .map_err(Error::from),
            Self::Cuda(model) => model
                .encoder
                .forward(x, flush_kv_cache)
                .map_err(Error::from),
        }
    }

    fn decoder_forward(
        &mut self,
        x: &Tensor,
        audio_features: &Tensor,
        flush_kv_cache: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Upstream(model) => model
                .decoder
                .forward(x, audio_features, flush_kv_cache)
                .map_err(Error::from),
            Self::Cuda(model) => model
                .decoder
                .forward(x, audio_features, flush_kv_cache)
                .map_err(Error::from),
        }
    }

    fn decoder_final_linear(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Upstream(model) => model.decoder.final_linear(x).map_err(Error::from),
            Self::Cuda(model) => model.decoder.final_linear(x).map_err(Error::from),
        }
    }
}

fn use_cuda_whisper_dtype_shim(device: &candle_core::Device) -> bool {
    device.is_cuda()
}

fn whisper_device_diagnostics(
    device_kind: DeviceKind,
    model_dtype: DType,
    cuda_dtype_shim: bool,
) -> serde_json::Value {
    json!({
        "kind": format!("{device_kind:?}"),
        "model_dtype": format!("{model_dtype:?}"),
        "cuda_dtype_shim": cuda_dtype_shim,
        "whisper_impl": whisper_impl_name(cuda_dtype_shim),
    })
}

fn whisper_impl_name(cuda_dtype_shim: bool) -> &'static str {
    if cuda_dtype_shim {
        "cuda_dtype_shim"
    } else {
        "upstream_candle"
    }
}

pub struct WhisperTurboAsrModel {
    device: DeviceProfile,
    model_dtype: DType,
    whisper: Mutex<WhisperModel>,
    config: WhisperConfig,
    generation: WhisperGenerationConfig,
    tokenizer: Tokenizer,
    special: WhisperSpecialTokens,
    mel: MelSpectrogram,
    suppress_tokens: Vec<u32>,
    numeral_symbol_tokens: Vec<u32>,
    language_token_ids: Vec<u32>,
    token_id_to_language_code: HashMap<u32, String>,
    runtime_tuning: WhisperRuntimeTuning,
    cuda_dtype_shim: bool,
    cached_detected_language: Mutex<Option<(u32, String)>>,
}

impl WhisperTurboAsrModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_data = fs::read_to_string(config_path)?;
        let config: WhisperConfig = serde_json::from_str(&config_data)?;

        let generation = read_generation_config(model_dir)?;
        let tokenizer = Tokenizer::from_path(model_dir)?;

        let dtype_override = std::env::var("IZWI_WHISPER_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let model_dtype = match dtype_override.as_deref() {
            Some(raw) => {
                device.select_model_dtype_checked(ModelFamily::WhisperAsr, Some(raw), "Whisper")?
            }
            None => device.select_model_dtype(ModelFamily::WhisperAsr, None),
        };

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            let index_data = fs::read_to_string(index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|value| value.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid Whisper safetensors index format".to_string())
                })?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> = shard_files
                .iter()
                .map(|file| model_dir.join(file))
                .collect();

            unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, model_dtype, &device.device)?
            }
        } else {
            let model_path = model_dir.join("model.safetensors");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], model_dtype, &device.device)?
            }
        };

        let use_cuda_dtype_shim = use_cuda_whisper_dtype_shim(&device.device);
        let whisper = WhisperModel::load(&vb, config.clone(), use_cuda_dtype_shim)?;
        let special = resolve_special_tokens(&tokenizer, &generation)?;
        let (language_token_ids, token_id_to_language_code) =
            build_language_token_maps(&tokenizer, &generation);

        let mut suppress_tokens = generation.suppress_tokens.clone();
        suppress_tokens.sort_unstable();
        suppress_tokens.dedup();
        let runtime_tuning = WhisperRuntimeTuning::from_env();
        let numeral_symbol_tokens = if runtime_tuning.suppress_numerals {
            build_numeral_symbol_tokens(&tokenizer, &special)
        } else {
            Vec::new()
        };

        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: whisper::SAMPLE_RATE,
            n_fft: whisper::N_FFT,
            hop_length: whisper::HOP_LENGTH,
            n_mels: config.num_mel_bins,
            f_min: 0.0,
            f_max: (whisper::SAMPLE_RATE / 2) as f32,
            normalize: true,
        })?;

        info!(
            "Loaded Whisper Large v3 Turbo ASR on {:?} (dtype={:?}, cuda_dtype_shim={})",
            device.kind, model_dtype, use_cuda_dtype_shim
        );

        Ok(Self {
            device,
            model_dtype,
            whisper: Mutex::new(whisper),
            config,
            generation,
            tokenizer,
            special,
            mel,
            suppress_tokens,
            numeral_symbol_tokens,
            language_token_ids,
            token_id_to_language_code,
            runtime_tuning,
            cuda_dtype_shim: use_cuda_dtype_shim,
            cached_detected_language: Mutex::new(None),
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        self.transcribe_with_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        Ok(self
            .transcribe_impl(audio, sample_rate, language, initial_prompt, &mut no_op)?
            .text)
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        self.transcribe_with_details_and_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_impl(audio, sample_rate, language, initial_prompt, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_with_callback_and_prompt(audio, sample_rate, language, None, on_delta)
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_streaming(audio, sample_rate, language, initial_prompt, on_delta)
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(MAX_AUDIO_SECONDS_HINT)
    }

    fn transcribe_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        _on_delta: &mut dyn FnMut(&str),
    ) -> Result<AsrTranscriptionOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let trimmed_audio = self.trimmed_audio_slice(audio, sample_rate);
        let request_started = Instant::now();
        let mel_started = Instant::now();
        let mel = self.prepare_mel(trimmed_audio, sample_rate)?;
        let mel_prepare_ms = mel_started.elapsed().as_secs_f64() * 1000.0;
        let mut whisper = self
            .whisper
            .lock()
            .map_err(|_| Error::InferenceError("Whisper model mutex poisoned".to_string()))?;

        whisper.reset_kv_cache();
        let encoder_started = Instant::now();
        let audio_features = whisper.encoder_forward(&mel, true)?;
        let encoder_forward_ms = encoder_started.elapsed().as_secs_f64() * 1000.0;

        let language_resolution =
            self.resolve_request_language(&mut whisper, &audio_features, language)?;
        let resolved_language = language_resolution.resolved.clone();
        let language_hint_used = language_resolution.hint_used;
        let language_detect_ms = language_resolution.detect_ms;

        let initial_prompt_tokens = self.encode_initial_prompt_tokens(initial_prompt)?;
        let prompt_prefix = build_whisper_prompt_prefix(
            &self.special,
            resolved_language
                .as_ref()
                .map(|(language_token, _language_code)| *language_token),
            &initial_prompt_tokens,
            self.config.max_target_positions,
            self.runtime_tuning.initial_prompt_max_tokens,
        )?;
        let prompt = prompt_prefix.ids;
        let prompt_diagnostics = prompt_prefix.diagnostics;

        let max_steps = decode_step_budget(
            prompt.len(),
            self.config.max_target_positions,
            self.resolve_max_decode_tokens(audio.len(), sample_rate),
        )?;
        let temperatures = self.decode_temperatures();
        let logprob_threshold = self
            .generation
            .logprob_threshold
            .unwrap_or(DEFAULT_LOGPROB_THRESHOLD);
        let no_speech_threshold = self
            .generation
            .no_speech_threshold
            .unwrap_or(DEFAULT_NO_SPEECH_THRESHOLD);
        let compression_ratio_threshold = self.generation.compression_ratio_threshold;

        let decode_started = Instant::now();
        let mut attempted_temperatures = Vec::with_capacity(temperatures.len());
        let mut attempt_diagnostics = Vec::with_capacity(temperatures.len());
        let mut fallback_reasons = Vec::<&'static str>::new();
        let mut best_attempt: Option<WhisperDecodeAttempt> = None;
        let mut selected_temperature = temperatures.first().copied().unwrap_or(0.0);
        let mut best_temperature = selected_temperature;
        for (idx, temperature) in temperatures.iter().copied().enumerate() {
            attempted_temperatures.push(temperature);
            let attempt = self.decode_attempt(
                &mut whisper,
                &audio_features,
                &prompt,
                max_steps,
                temperature,
            )?;
            let no_speech_skip =
                should_skip_as_no_speech(&attempt, logprob_threshold, no_speech_threshold);
            let retry_reasons =
                decode_retry_reasons(&attempt, logprob_threshold, compression_ratio_threshold);
            attempt_diagnostics.push(whisper_attempt_diagnostics(
                temperature,
                &attempt,
                &retry_reasons,
                no_speech_skip,
            ));

            if no_speech_skip {
                record_unique_reason(&mut fallback_reasons, "no_speech");
                best_attempt = Some(WhisperDecodeAttempt {
                    text: String::new(),
                    ..attempt
                });
                selected_temperature = temperature;
                break;
            }

            let is_last_temperature = idx + 1 == temperatures.len();
            let should_retry = !is_last_temperature && !retry_reasons.is_empty();
            if !should_retry {
                if best_attempt
                    .as_ref()
                    .map(|best| is_better_attempt(&attempt, best))
                    .unwrap_or(true)
                {
                    best_attempt = Some(attempt);
                    best_temperature = temperature;
                }
                selected_temperature = best_temperature;
                break;
            }

            record_unique_reasons(&mut fallback_reasons, &retry_reasons);
            if best_attempt
                .as_ref()
                .map(|best| is_better_attempt(&attempt, best))
                .unwrap_or(true)
            {
                best_attempt = Some(attempt);
                best_temperature = temperature;
            }
        }
        let decode_secs = decode_started.elapsed().as_secs_f64();
        let decode_ms = decode_secs * 1000.0;

        let final_attempt = best_attempt.unwrap_or_else(|| WhisperDecodeAttempt {
            text: String::new(),
            avg_logprob: f32::NEG_INFINITY,
            no_speech_prob: None,
            ended_with_eot: false,
            repetition_loop: false,
            compression_ratio: None,
            generated_token_count: 0,
            sampled_token_count: 0,
            decode_steps: 0,
        });

        let text = final_attempt.text.trim().to_string();
        let language = resolved_language.map(|(_token_id, code)| code);
        let model_total_ms = request_started.elapsed().as_secs_f64() * 1000.0;
        let generated_tokens_per_second = if decode_secs > 0.0 {
            Some(final_attempt.generated_token_count as f64 / decode_secs)
        } else {
            None
        };
        let diagnostics = json!({
            "model_family": "whisper_asr",
            "device": whisper_device_diagnostics(
                self.device.kind,
                self.model_dtype,
                self.cuda_dtype_shim,
            ),
            "fallback_attempts": attempted_temperatures.len(),
            "attempted_temperatures": attempted_temperatures,
            "fallback_policy": {
                "no_fallback": self.runtime_tuning.no_fallback,
                "max_fallback_retries": self.runtime_tuning.max_fallback_retries,
                "max_attempts": self.runtime_tuning.max_fallback_retries.saturating_add(1),
            },
            "decode_budget": {
                "adaptive_enabled": self.runtime_tuning.adaptive_decode_budget,
                "max_new_tokens_per_second": self.runtime_tuning.max_new_tokens_per_second,
                "min_new_tokens": self.runtime_tuning.min_new_tokens,
                "max_new_tokens_cap": self.runtime_tuning.max_new_tokens_cap,
                "buffer_tokens": self.runtime_tuning.decode_budget_buffer_tokens,
                "resolved_max_new_tokens": self.resolve_max_decode_tokens(audio.len(), sample_rate),
                "audio_seconds": if sample_rate > 0 {
                    audio.len() as f32 / sample_rate as f32
                } else {
                    0.0
                },
            },
            "audio_window": {
                "trim_silence": self.runtime_tuning.trim_silence,
                "input_samples": audio.len(),
                "effective_samples": trimmed_audio.len(),
                "trimmed_samples": audio.len().saturating_sub(trimmed_audio.len()),
            },
            "language_resolution": {
                "strategy": language_resolution.strategy,
                "default_language": self.runtime_tuning.default_language,
                "reuse_detected_language": self.runtime_tuning.reuse_detected_language,
            },
            "prompt": {
                "initial_prompt_requested": prompt_diagnostics.initial_prompt_requested,
                "initial_prompt_token_count": prompt_diagnostics.initial_prompt_token_count,
                "initial_prompt_tokens_used": prompt_diagnostics.initial_prompt_tokens_used,
                "initial_prompt_tokens_truncated": prompt_diagnostics.initial_prompt_tokens_truncated,
                "initial_prompt_max_tokens": prompt_diagnostics.initial_prompt_max_tokens,
                "previous_context_token_id": prompt_diagnostics.previous_context_token_id,
                "rolling_context_enabled": prompt_diagnostics.rolling_context_enabled,
            },
            "logit_filters": {
                "suppress_blank": self.runtime_tuning.suppress_blank,
                "blank_token_id": self.special.blank,
                "suppress_numerals": self.runtime_tuning.suppress_numerals,
                "numeral_symbol_token_count": self.numeral_symbol_tokens.len(),
            },
            "selected_temperature": selected_temperature,
            "language_hint_used": language_hint_used,
            "fallback_reasons": fallback_reasons,
            "decode_attempts": attempt_diagnostics,
            "decode": {
                "ended_with_eot": final_attempt.ended_with_eot,
                "repetition_loop": final_attempt.repetition_loop,
                "avg_logprob": final_attempt.avg_logprob,
                "no_speech_prob": final_attempt.no_speech_prob,
                "compression_ratio": final_attempt.compression_ratio,
                "decode_steps": final_attempt.decode_steps,
                "generated_token_count": final_attempt.generated_token_count,
                "sampled_token_count": final_attempt.sampled_token_count,
                "generated_tokens_per_second": generated_tokens_per_second,
            },
            "timings_ms": {
                "mel_prepare": mel_prepare_ms,
                "encoder_forward": encoder_forward_ms,
                "language_detect": language_detect_ms,
                "decode": decode_ms,
                "model_total": model_total_ms,
            }
        });

        Ok(AsrTranscriptionOutput {
            text,
            language,
            diagnostics: Some(diagnostics),
        })
    }

    fn transcribe_streaming(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let trimmed_audio = self.trimmed_audio_slice(audio, sample_rate);
        let mel = self.prepare_mel(trimmed_audio, sample_rate)?;
        let mut whisper = self
            .whisper
            .lock()
            .map_err(|_| Error::InferenceError("Whisper model mutex poisoned".to_string()))?;

        whisper.reset_kv_cache();
        let audio_features = whisper.encoder_forward(&mel, true)?;

        let language_resolution =
            self.resolve_request_language(&mut whisper, &audio_features, language)?;
        let resolved_language = language_resolution.resolved;

        let initial_prompt_tokens = self.encode_initial_prompt_tokens(initial_prompt)?;
        let prompt = build_whisper_prompt_prefix(
            &self.special,
            resolved_language
                .as_ref()
                .map(|(language_token, _language_code)| *language_token),
            &initial_prompt_tokens,
            self.config.max_target_positions,
            self.runtime_tuning.initial_prompt_max_tokens,
        )?
        .ids;

        let max_steps = decode_step_budget(
            prompt.len(),
            self.config.max_target_positions,
            self.resolve_max_decode_tokens(audio.len(), sample_rate),
        )?;
        let temperatures = self.decode_temperatures();
        let logprob_threshold = self
            .generation
            .logprob_threshold
            .unwrap_or(DEFAULT_LOGPROB_THRESHOLD);
        let no_speech_threshold = self
            .generation
            .no_speech_threshold
            .unwrap_or(DEFAULT_NO_SPEECH_THRESHOLD);
        let compression_ratio_threshold = self.generation.compression_ratio_threshold;

        let first_temperature = temperatures.first().copied().unwrap_or(0.0);
        let first_attempt = self.decode_attempt_streaming(
            &mut whisper,
            &audio_features,
            &prompt,
            max_steps,
            first_temperature,
            on_delta,
        )?;

        if should_skip_as_no_speech(&first_attempt, logprob_threshold, no_speech_threshold) {
            return Ok(String::new());
        }

        let mut best_attempt = first_attempt;
        let mut should_retry = temperatures.len() > 1
            && (best_attempt.text.trim().is_empty()
                || should_retry_decode(
                    &best_attempt,
                    logprob_threshold,
                    compression_ratio_threshold,
                ));
        if !should_retry {
            return Ok(best_attempt.text.trim().to_string());
        }

        for (idx, temperature) in temperatures.iter().copied().enumerate().skip(1) {
            let attempt = self.decode_attempt(
                &mut whisper,
                &audio_features,
                &prompt,
                max_steps,
                temperature,
            )?;

            if should_skip_as_no_speech(&attempt, logprob_threshold, no_speech_threshold) {
                return Ok(String::new());
            }

            let is_last_temperature = idx + 1 == temperatures.len();
            should_retry = !is_last_temperature
                && (attempt.text.trim().is_empty()
                    || should_retry_decode(
                        &attempt,
                        logprob_threshold,
                        compression_ratio_threshold,
                    ));
            if !should_retry {
                if is_better_attempt(&attempt, &best_attempt) {
                    best_attempt = attempt;
                }
                break;
            }

            if is_better_attempt(&attempt, &best_attempt) {
                best_attempt = attempt;
            }
        }

        Ok(best_attempt.text.trim().to_string())
    }

    fn prepare_mel(&self, audio: &[f32], sample_rate: u32) -> Result<Tensor> {
        let mono_16khz = if sample_rate == SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, SAMPLE_RATE)
        };

        let mut mel_spec = self.mel.compute(&mono_16khz)?;
        if mel_spec.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        // Whisper encoder downsamples by 2 before positional embeddings.
        let max_input_frames = self.config.max_source_positions.saturating_mul(2).max(1);
        if mel_spec.len() > max_input_frames {
            mel_spec.truncate(max_input_frames);
        }

        let n_mels = self.config.num_mel_bins;
        let frames = mel_spec.len();
        let mut flat = vec![0f32; frames * n_mels];
        for (frame_idx, frame) in mel_spec.iter().enumerate() {
            for mel_idx in 0..n_mels {
                flat[mel_idx * frames + frame_idx] = frame[mel_idx];
            }
        }

        let mel = Tensor::from_vec(flat, (1, n_mels, frames), &self.device.device)?;
        if mel.dtype() != self.model_dtype {
            return Ok(mel.to_dtype(self.model_dtype)?);
        }
        Ok(mel)
    }

    fn resolve_language_token(&self, language: &str) -> Result<Option<(u32, String)>> {
        let normalized = language.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Ok(None);
        }

        let language_code = if let Some(code) = normalized
            .strip_prefix("<|")
            .and_then(|inner| inner.strip_suffix("|>"))
        {
            code.to_string()
        } else if has_whisper_language_token(
            &self.generation.lang_to_id,
            &normalized,
            &self.tokenizer,
        ) {
            normalized
        } else if let Some(code) = language_name_to_code(&normalized) {
            code.to_string()
        } else if let Some(code) = language_alias_to_code(&normalized) {
            code.to_string()
        } else {
            return Err(Error::InvalidInput(format!(
                "Unsupported Whisper language '{}'",
                language
            )));
        };

        let token = format!("<|{}|>", language_code);
        let token_id = self
            .generation
            .lang_to_id
            .get(&token)
            .copied()
            .or_else(|| self.tokenizer.token_to_id(&token))
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Whisper model does not support language token '{}'",
                    token
                ))
            })?;

        Ok(Some((token_id, language_code)))
    }

    fn detect_language_token(
        &self,
        whisper: &mut WhisperModel,
        audio_features: &Tensor,
    ) -> Result<Option<(u32, String)>> {
        if self.language_token_ids.is_empty() {
            return Ok(None);
        }

        let tokens = Tensor::new(&[[self.special.sot]], &self.device.device)?;
        let ys = whisper.decoder_forward(&tokens, audio_features, true)?;
        let logits = whisper.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let logits_vec = logits.to_vec1::<f32>()?;

        let mut best_token: Option<u32> = None;
        let mut best_score = f32::NEG_INFINITY;
        for token_id in &self.language_token_ids {
            let idx = *token_id as usize;
            if idx >= logits_vec.len() {
                continue;
            }
            let score = logits_vec[idx];
            if score > best_score {
                best_score = score;
                best_token = Some(*token_id);
            }
        }

        let Some(token_id) = best_token else {
            return Ok(None);
        };
        let Some(code) = self.token_id_to_language_code.get(&token_id).cloned() else {
            return Ok(None);
        };

        Ok(Some((token_id, code)))
    }

    fn decode_generated_text(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(token_ids)
    }

    fn encode_initial_prompt_tokens(&self, initial_prompt: Option<&str>) -> Result<Vec<u32>> {
        let Some(prompt) = initial_prompt
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
        else {
            return Ok(Vec::new());
        };
        self.tokenizer.encode(prompt)
    }

    fn decode_temperatures(&self) -> Vec<f32> {
        // Mirrors whisper.cpp/transformers temperature fallback ladder.
        let start = self.generation.temperature.unwrap_or(0.0).clamp(0.0, 1.0);
        let inc = self
            .generation
            .temperature_increment_on_fallback
            .unwrap_or(DEFAULT_TEMPERATURE_FALLBACK_INC);
        capped_decode_temperatures(
            start,
            inc,
            self.runtime_tuning.no_fallback,
            self.runtime_tuning.max_fallback_retries,
        )
    }

    fn resolve_max_decode_tokens(&self, audio_len_samples: usize, sample_rate: u32) -> usize {
        let configured_max = self.generation.max_length.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        let cap = configured_max.min(self.runtime_tuning.max_new_tokens_cap.max(1));
        if !self.runtime_tuning.adaptive_decode_budget || sample_rate == 0 {
            return cap;
        }

        let audio_secs = audio_len_samples as f32 / sample_rate as f32;
        adaptive_decode_budget(
            audio_secs,
            cap,
            self.runtime_tuning.max_new_tokens_per_second,
            self.runtime_tuning.min_new_tokens,
            self.runtime_tuning.decode_budget_buffer_tokens,
        )
    }

    fn resolve_request_language(
        &self,
        whisper: &mut WhisperModel,
        audio_features: &Tensor,
        language: Option<&str>,
    ) -> Result<WhisperLanguageResolution> {
        if let Some(language) = language {
            let resolved = self.resolve_language_token(language)?;
            if let Some((token, code)) = resolved.as_ref() {
                self.update_cached_language(*token, code.clone());
            }
            return Ok(WhisperLanguageResolution {
                resolved,
                hint_used: true,
                detect_ms: 0.0,
                strategy: "hint",
            });
        }

        if let Some(default_language) = self.runtime_tuning.default_language.as_deref() {
            let resolved = self.resolve_language_token(default_language)?;
            if let Some((token, code)) = resolved.as_ref() {
                self.update_cached_language(*token, code.clone());
            }
            return Ok(WhisperLanguageResolution {
                resolved,
                hint_used: false,
                detect_ms: 0.0,
                strategy: "default",
            });
        }

        if self.runtime_tuning.reuse_detected_language {
            if let Some(cached) = self.cached_language() {
                return Ok(WhisperLanguageResolution {
                    resolved: Some(cached),
                    hint_used: false,
                    detect_ms: 0.0,
                    strategy: "cached",
                });
            }
        }

        let detect_started = Instant::now();
        let resolved = self.detect_language_token(whisper, audio_features)?;
        let detect_ms = detect_started.elapsed().as_secs_f64() * 1000.0;
        if let Some((token, code)) = resolved.as_ref() {
            self.update_cached_language(*token, code.clone());
        }
        Ok(WhisperLanguageResolution {
            resolved,
            hint_used: false,
            detect_ms,
            strategy: "detected",
        })
    }

    fn cached_language(&self) -> Option<(u32, String)> {
        self.cached_detected_language
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().cloned())
    }

    fn update_cached_language(&self, token: u32, code: String) {
        if !self.runtime_tuning.reuse_detected_language {
            return;
        }
        if let Ok(mut guard) = self.cached_detected_language.lock() {
            *guard = Some((token, code));
        }
    }

    fn trimmed_audio_slice<'a>(&self, audio: &'a [f32], sample_rate: u32) -> &'a [f32] {
        if !self.runtime_tuning.trim_silence {
            return audio;
        }

        let (start, end) = trimmed_audio_bounds(
            audio,
            sample_rate,
            self.runtime_tuning.silence_trim_threshold_scale,
            self.runtime_tuning.silence_trim_min_abs,
            self.runtime_tuning.silence_trim_margin_ms,
            self.runtime_tuning.silence_trim_min_leading_ms,
            self.runtime_tuning.silence_trim_min_trailing_ms,
            self.runtime_tuning.silence_trim_min_clip_secs,
        );
        &audio[start..end]
    }
}

fn capped_decode_temperatures(
    start: f32,
    temperature_inc: f32,
    no_fallback: bool,
    max_fallback_retries: usize,
) -> Vec<f32> {
    if no_fallback {
        return vec![start];
    }

    let mut temperatures = Vec::new();
    if temperature_inc <= 0.0 {
        temperatures.push(start);
    } else {
        let mut t = start;
        while t <= 1.0 + 1e-6 {
            temperatures.push((t * 100.0).round() / 100.0);
            t += temperature_inc;
        }
    }

    if temperatures.is_empty() {
        temperatures.push(start);
    }

    let max_attempts = max_fallback_retries.saturating_add(1);
    if temperatures.len() > max_attempts {
        temperatures.truncate(max_attempts);
    }
    temperatures
}

fn adaptive_decode_budget(
    audio_secs: f32,
    configured_cap: usize,
    tokens_per_second: f32,
    min_new_tokens: usize,
    buffer_tokens: usize,
) -> usize {
    if configured_cap == 0 {
        return 1;
    }
    let tps = tokens_per_second.max(0.0);
    let scaled = (audio_secs.max(0.0) * tps).ceil() as usize;
    let proposed = scaled
        .saturating_add(buffer_tokens)
        .max(min_new_tokens)
        .max(1);
    proposed.min(configured_cap.max(1))
}

fn build_whisper_prompt_prefix(
    special: &WhisperSpecialTokens,
    language_token: Option<u32>,
    initial_prompt_tokens: &[u32],
    max_target_positions: usize,
    initial_prompt_max_tokens: usize,
) -> Result<WhisperPromptPrefix> {
    let mut controls = Vec::with_capacity(4);
    controls.push(special.sot);
    if let Some(language_token) = language_token {
        controls.push(language_token);
    }
    controls.push(special.transcribe);
    if let Some(no_timestamps) = special.no_timestamps {
        controls.push(no_timestamps);
    }

    if controls.len() >= max_target_positions {
        return Err(Error::InvalidInput(format!(
            "Whisper prompt controls length {} exceeds decoder context {}",
            controls.len(),
            max_target_positions
        )));
    }

    let initial_prompt_requested = !initial_prompt_tokens.is_empty();
    let base_context_budget = max_target_positions
        .saturating_sub(controls.len())
        .saturating_sub(1);
    let can_use_previous_context =
        initial_prompt_requested && special.sot_prev.is_some() && base_context_budget > 1;
    let previous_context_tokens = usize::from(can_use_previous_context);
    let available_for_context = base_context_budget.saturating_sub(previous_context_tokens);
    let prompt_budget = available_for_context.min(initial_prompt_max_tokens);
    let initial_prompt_tokens_used = initial_prompt_tokens.len().min(prompt_budget);
    let initial_prompt_tokens_truncated = initial_prompt_tokens
        .len()
        .saturating_sub(initial_prompt_tokens_used);
    let previous_context_token_id = if initial_prompt_tokens_used > 0 && can_use_previous_context {
        special.sot_prev
    } else {
        None
    };

    let mut ids =
        Vec::with_capacity(previous_context_tokens + initial_prompt_tokens_used + controls.len());
    if let Some(token_id) = previous_context_token_id {
        ids.push(token_id);
    }
    if initial_prompt_tokens_used > 0 {
        let start = initial_prompt_tokens.len() - initial_prompt_tokens_used;
        ids.extend_from_slice(&initial_prompt_tokens[start..]);
    }
    ids.extend(controls);

    Ok(WhisperPromptPrefix {
        ids,
        diagnostics: WhisperPromptDiagnostics {
            initial_prompt_requested,
            initial_prompt_token_count: initial_prompt_tokens.len(),
            initial_prompt_tokens_used,
            initial_prompt_tokens_truncated,
            initial_prompt_max_tokens,
            previous_context_token_id,
            rolling_context_enabled: false,
        },
    })
}

fn trimmed_audio_bounds(
    audio: &[f32],
    sample_rate: u32,
    threshold_scale: f32,
    min_abs: f32,
    margin_ms: usize,
    min_leading_ms: usize,
    min_trailing_ms: usize,
    min_clip_secs: f32,
) -> (usize, usize) {
    if audio.is_empty() || sample_rate == 0 {
        return (0, audio.len());
    }

    let clip_secs = audio.len() as f32 / sample_rate as f32;
    if clip_secs < min_clip_secs.max(0.0) {
        return (0, audio.len());
    }

    let peak = audio.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak <= 0.0 {
        return (0, audio.len());
    }

    let threshold = (peak * threshold_scale.max(0.0)).max(min_abs.max(0.0));
    let Some(first) = audio.iter().position(|sample| sample.abs() >= threshold) else {
        return (0, audio.len());
    };
    let Some(last) = audio.iter().rposition(|sample| sample.abs() >= threshold) else {
        return (0, audio.len());
    };

    let margin = sample_rate as usize * margin_ms / 1000;
    let mut start = first.saturating_sub(margin);
    let mut end = (last.saturating_add(margin).saturating_add(1)).min(audio.len());

    let min_leading_samples =
        ((min_leading_ms as u64).saturating_mul(sample_rate as u64) / 1000) as usize;
    let min_trailing_samples =
        ((min_trailing_ms as u64).saturating_mul(sample_rate as u64) / 1000) as usize;
    if start < min_leading_samples {
        start = 0;
    }
    if audio.len().saturating_sub(end) < min_trailing_samples {
        end = audio.len();
    }

    if end <= start {
        return (0, audio.len());
    }
    if start == 0 && end == audio.len() {
        return (0, audio.len());
    }
    (start, end)
}

impl WhisperTurboAsrModel {
    fn decode_attempt(
        &self,
        whisper: &mut WhisperModel,
        audio_features: &Tensor,
        prompt_prefix: &[u32],
        max_steps: usize,
        temperature: f32,
    ) -> Result<WhisperDecodeAttempt> {
        let mut rng = rand::thread_rng();
        let deterministic = temperature <= 0.0;
        let mut prompt = prompt_prefix.to_vec();
        let mut generated_tokens = Vec::<u32>::new();
        let mut sum_logprobs = 0.0f64;
        let mut sampled_token_count = 0usize;
        let mut no_speech_prob: Option<f32> = None;
        let mut ended_with_eot = false;
        let mut repetition_loop = false;
        let mut decode_steps = 0usize;

        for step_idx in 0..max_steps {
            decode_steps = decode_steps.saturating_add(1);
            let tokens_t = Tensor::new(prompt.as_slice(), &self.device.device)?.unsqueeze(0)?;
            let ys = whisper.decoder_forward(&tokens_t, audio_features, step_idx == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = whisper
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let mut logits_vec = logits.to_vec1::<f32>()?;
            self.apply_decode_constraints(&mut logits_vec, step_idx == 0);
            let (next, next_logprob, step_no_speech_prob) = if deterministic {
                let inv_temperature = 1.0f32 / temperature.max(1e-6);
                if let Some(logsumexp) = scaled_logsumexp(&logits_vec, inv_temperature) {
                    let (next, best_logit) = best_finite_logit(&logits_vec, self.special.eot);
                    let next_scaled_logit = best_logit * inv_temperature;
                    let next_logprob = if next_scaled_logit.is_finite() {
                        next_scaled_logit - logsumexp
                    } else {
                        f32::NEG_INFINITY
                    };
                    let no_speech_prob = self.special.no_speech.and_then(|token_id| {
                        probability_for_token_from_logits(
                            &logits_vec,
                            token_id,
                            logsumexp,
                            inv_temperature,
                        )
                    });
                    (next, next_logprob, no_speech_prob)
                } else {
                    (self.special.eot, f32::NEG_INFINITY, None)
                }
            } else {
                let log_probs_buf = logits_to_log_probs(&logits_vec, temperature);
                let no_speech_prob = self
                    .special
                    .no_speech
                    .and_then(|token_id| probability_for_token(&log_probs_buf, token_id));
                let (next, next_logprob) = sample_token_from_log_probs(
                    &log_probs_buf,
                    temperature,
                    self.special.eot,
                    &mut rng,
                );
                (next, next_logprob, no_speech_prob)
            };

            if step_idx == 0 {
                no_speech_prob = step_no_speech_prob;
            }
            sum_logprobs += next_logprob as f64;
            sampled_token_count = sampled_token_count.saturating_add(1);

            if next == self.special.eot {
                ended_with_eot = true;
                break;
            }

            generated_tokens.push(next);
            prompt.push(next);

            if let Some((span, repeats)) = find_suffix_token_repetition(&generated_tokens) {
                let trim = span.saturating_mul(repeats.saturating_sub(1));
                if trim > 0 && trim <= generated_tokens.len() {
                    generated_tokens.truncate(generated_tokens.len() - trim);
                }
                repetition_loop = true;
                break;
            }
        }

        let text = self
            .decode_generated_text(&generated_tokens)?
            .trim()
            .to_string();
        let avg_logprob = if sampled_token_count > 0 {
            (sum_logprobs / sampled_token_count as f64) as f32
        } else {
            f32::NEG_INFINITY
        };
        let compression_ratio = token_compression_ratio(&generated_tokens, self.config.vocab_size);

        Ok(WhisperDecodeAttempt {
            text,
            avg_logprob,
            no_speech_prob,
            ended_with_eot,
            repetition_loop,
            compression_ratio,
            generated_token_count: generated_tokens.len(),
            sampled_token_count,
            decode_steps,
        })
    }

    fn decode_attempt_streaming(
        &self,
        whisper: &mut WhisperModel,
        audio_features: &Tensor,
        prompt_prefix: &[u32],
        max_steps: usize,
        temperature: f32,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<WhisperDecodeAttempt> {
        let mut rng = rand::thread_rng();
        let deterministic = temperature <= 0.0;
        let mut prompt = prompt_prefix.to_vec();
        let mut generated_tokens = Vec::<u32>::new();
        let mut sum_logprobs = 0.0f64;
        let mut sampled_token_count = 0usize;
        let mut no_speech_prob: Option<f32> = None;
        let mut ended_with_eot = false;
        let mut repetition_loop = false;
        let mut streamed_text = String::new();
        let mut log_probs_buf = Vec::<f32>::new();
        let mut decode_steps = 0usize;

        for step_idx in 0..max_steps {
            decode_steps = decode_steps.saturating_add(1);
            let tokens_t = Tensor::new(prompt.as_slice(), &self.device.device)?.unsqueeze(0)?;
            let ys = whisper.decoder_forward(&tokens_t, audio_features, step_idx == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = whisper
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let mut logits_vec = logits.to_vec1::<f32>()?;
            self.apply_decode_constraints(&mut logits_vec, step_idx == 0);
            let (next, next_logprob, step_no_speech_prob) = if deterministic {
                let inv_temperature = 1.0f32 / temperature.max(1e-6);
                if let Some(logsumexp) = scaled_logsumexp(&logits_vec, inv_temperature) {
                    let (next, best_logit) = best_finite_logit(&logits_vec, self.special.eot);
                    let next_scaled_logit = best_logit * inv_temperature;
                    let next_logprob = if next_scaled_logit.is_finite() {
                        next_scaled_logit - logsumexp
                    } else {
                        f32::NEG_INFINITY
                    };
                    let no_speech_prob = self.special.no_speech.and_then(|token_id| {
                        probability_for_token_from_logits(
                            &logits_vec,
                            token_id,
                            logsumexp,
                            inv_temperature,
                        )
                    });
                    (next, next_logprob, no_speech_prob)
                } else {
                    (self.special.eot, f32::NEG_INFINITY, None)
                }
            } else {
                logits_to_log_probs_in_place(&logits_vec, temperature, &mut log_probs_buf);
                let no_speech_prob = self
                    .special
                    .no_speech
                    .and_then(|token_id| probability_for_token(&log_probs_buf, token_id));
                let (next, next_logprob) = sample_token_from_log_probs(
                    &log_probs_buf,
                    temperature,
                    self.special.eot,
                    &mut rng,
                );
                (next, next_logprob, no_speech_prob)
            };

            if step_idx == 0 {
                no_speech_prob = step_no_speech_prob;
            }
            sum_logprobs += next_logprob as f64;
            sampled_token_count = sampled_token_count.saturating_add(1);

            if next == self.special.eot {
                ended_with_eot = true;
                break;
            }

            generated_tokens.push(next);
            prompt.push(next);

            let decoded = self.decode_generated_text(&generated_tokens)?;
            let trimmed = decoded.trim();
            let delta = text_delta(&streamed_text, trimmed);
            if !delta.is_empty() {
                on_delta(delta);
                streamed_text.clear();
                streamed_text.push_str(trimmed);
            }

            if let Some((span, repeats)) = find_suffix_token_repetition(&generated_tokens) {
                let trim = span.saturating_mul(repeats.saturating_sub(1));
                if trim > 0 && trim <= generated_tokens.len() {
                    generated_tokens.truncate(generated_tokens.len() - trim);
                }
                repetition_loop = true;
                break;
            }
        }

        let text = self
            .decode_generated_text(&generated_tokens)?
            .trim()
            .to_string();
        let avg_logprob = if sampled_token_count > 0 {
            (sum_logprobs / sampled_token_count as f64) as f32
        } else {
            f32::NEG_INFINITY
        };
        let compression_ratio = token_compression_ratio(&generated_tokens, self.config.vocab_size);

        Ok(WhisperDecodeAttempt {
            text,
            avg_logprob,
            no_speech_prob,
            ended_with_eot,
            repetition_loop,
            compression_ratio,
            generated_token_count: generated_tokens.len(),
            sampled_token_count,
            decode_steps,
        })
    }

    fn apply_decode_constraints(&self, logits: &mut [f32], at_begin: bool) {
        apply_whisper_decode_constraints(
            logits,
            at_begin,
            &self.suppress_tokens,
            &self.generation.begin_suppress_tokens,
            &self.language_token_ids,
            &self.special,
            self.runtime_tuning.suppress_blank,
            &self.numeral_symbol_tokens,
        );
    }
}

fn read_generation_config(model_dir: &Path) -> Result<WhisperGenerationConfig> {
    let generation_path = model_dir.join("generation_config.json");
    if !generation_path.exists() {
        return Ok(WhisperGenerationConfig::default());
    }
    let generation_data = fs::read_to_string(generation_path)?;
    Ok(serde_json::from_str::<WhisperGenerationConfig>(
        &generation_data,
    )?)
}

fn resolve_special_tokens(
    tokenizer: &Tokenizer,
    generation: &WhisperGenerationConfig,
) -> Result<WhisperSpecialTokens> {
    let sot = tokenizer.token_to_id(whisper::SOT_TOKEN).ok_or_else(|| {
        Error::TokenizationError("Missing <|startoftranscript|> token".to_string())
    })?;
    let sot_prev = tokenizer.token_to_id("<|startofprev|>");
    let transcribe = tokenizer
        .token_to_id(whisper::TRANSCRIBE_TOKEN)
        .or_else(|| generation.task_to_id.get("transcribe").copied())
        .ok_or_else(|| Error::TokenizationError("Missing <|transcribe|> token".to_string()))?;
    let eot = tokenizer
        .token_to_id(whisper::EOT_TOKEN)
        .or(generation.eos_token_id)
        .ok_or_else(|| Error::TokenizationError("Missing <|endoftext|> token".to_string()))?;
    let blank = tokenizer
        .encode(" ")
        .ok()
        .and_then(|ids| (ids.len() == 1).then_some(ids[0]))
        .or_else(|| tokenizer.token_to_id(" "));
    let no_timestamps = generation
        .no_timestamps_token_id
        .or_else(|| tokenizer.token_to_id(whisper::NO_TIMESTAMPS_TOKEN));
    let no_speech = whisper::NO_SPEECH_TOKENS
        .iter()
        .find_map(|token| tokenizer.token_to_id(token));

    Ok(WhisperSpecialTokens {
        sot,
        sot_prev,
        transcribe,
        eot,
        blank,
        no_timestamps,
        no_speech,
    })
}

fn build_numeral_symbol_tokens(tokenizer: &Tokenizer, special: &WhisperSpecialTokens) -> Vec<u32> {
    let special_ids: HashSet<u32> = [
        Some(special.sot),
        special.sot_prev,
        Some(special.transcribe),
        Some(special.eot),
        special.blank,
        special.no_timestamps,
        special.no_speech,
    ]
    .into_iter()
    .flatten()
    .collect();

    let mut tokens: Vec<u32> = tokenizer
        .vocab()
        .into_iter()
        .filter_map(|(token, token_id)| {
            if special_ids.contains(&token_id) || is_whisper_control_token(&token) {
                return None;
            }
            token_contains_numeral_or_symbol(&token).then_some(token_id)
        })
        .collect();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn is_whisper_control_token(token: &str) -> bool {
    token.starts_with("<|") && token.ends_with("|>")
}

fn token_contains_numeral_or_symbol(token: &str) -> bool {
    token.chars().any(|ch| {
        ch.is_numeric()
            || matches!(
                ch,
                '$' | '%'
                    | '+'
                    | '='
                    | '#'
                    | '@'
                    | '*'
                    | '/'
                    | '\\'
                    | '<'
                    | '>'
                    | '^'
                    | '_'
                    | '~'
                    | '&'
            )
    })
}

fn build_language_token_maps(
    tokenizer: &Tokenizer,
    generation: &WhisperGenerationConfig,
) -> (Vec<u32>, HashMap<u32, String>) {
    let mut token_to_lang = HashMap::new();
    let mut lang_ids = Vec::new();

    if generation.lang_to_id.is_empty() {
        for (code, _name) in WHISPER_LANGUAGES {
            let token = format!("<|{}|>", code);
            if let Some(token_id) = tokenizer.token_to_id(&token) {
                lang_ids.push(token_id);
                token_to_lang.insert(token_id, (*code).to_string());
            }
        }
    } else {
        for (token, token_id) in &generation.lang_to_id {
            if let Some(code) = token
                .strip_prefix("<|")
                .and_then(|inner| inner.strip_suffix("|>"))
            {
                lang_ids.push(*token_id);
                token_to_lang.insert(*token_id, code.to_string());
            }
        }
    }

    lang_ids.sort_unstable();
    lang_ids.dedup();
    (lang_ids, token_to_lang)
}

fn has_whisper_language_token(
    generation_lang_to_id: &HashMap<String, u32>,
    code: &str,
    tokenizer: &Tokenizer,
) -> bool {
    let token = format!("<|{}|>", code);
    generation_lang_to_id.contains_key(&token) || tokenizer.token_to_id(&token).is_some()
}

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|raw| {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_nonempty_string(key: &str) -> Option<String> {
    std::env::var(key).ok().and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn apply_whisper_decode_constraints(
    logits: &mut [f32],
    at_begin: bool,
    suppress_tokens: &[u32],
    begin_suppress_tokens: &[u32],
    language_token_ids: &[u32],
    special: &WhisperSpecialTokens,
    suppress_blank: bool,
    numeral_symbol_tokens: &[u32],
) {
    for token_id in suppress_tokens {
        mask_token(logits, *token_id);
    }
    if at_begin {
        for token_id in begin_suppress_tokens {
            mask_token(logits, *token_id);
        }
        if suppress_blank {
            mask_token(logits, special.eot);
            if let Some(blank_token_id) = special.blank {
                mask_token(logits, blank_token_id);
            }
        }
    }

    mask_token(logits, special.sot);
    mask_token(logits, special.transcribe);
    for token_id in language_token_ids {
        mask_token(logits, *token_id);
    }
    for token_id in numeral_symbol_tokens {
        mask_token(logits, *token_id);
    }

    if let Some(no_timestamps_token_id) = special.no_timestamps {
        // whisper.cpp / transformers text-only decode behavior.
        mask_token(logits, no_timestamps_token_id);
        let timestamp_begin = no_timestamps_token_id.saturating_add(1) as usize;
        if timestamp_begin < logits.len() {
            logits[timestamp_begin..].fill(f32::NEG_INFINITY);
        }
    }
}

fn mask_token(logits: &mut [f32], token_id: u32) {
    let idx = token_id as usize;
    if idx < logits.len() {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn logits_to_log_probs(logits: &[f32], temperature: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(logits.len());
    logits_to_log_probs_in_place(logits, temperature, &mut out);
    out
}

fn logits_to_log_probs_in_place(logits: &[f32], temperature: f32, out: &mut Vec<f32>) {
    let inv_temperature = if temperature > 0.0 {
        1.0 / temperature
    } else {
        1.0
    };

    out.clear();
    out.resize(logits.len(), f32::NEG_INFINITY);

    let Some(logsumexp) = scaled_logsumexp(logits, inv_temperature) else {
        return;
    };

    for (idx, logit) in logits.iter().enumerate() {
        if logit.is_finite() {
            out[idx] = (*logit * inv_temperature) - logsumexp;
        }
    }
}

fn scaled_logsumexp(logits: &[f32], inv_temperature: f32) -> Option<f32> {
    let mut max_scaled = f32::NEG_INFINITY;
    for logit in logits {
        if !logit.is_finite() {
            continue;
        }
        let scaled = *logit * inv_temperature;
        if scaled > max_scaled {
            max_scaled = scaled;
        }
    }

    if !max_scaled.is_finite() {
        return None;
    }

    let mut sum_exp = 0.0f64;
    for logit in logits {
        if !logit.is_finite() {
            continue;
        }
        let scaled = *logit * inv_temperature;
        sum_exp += (scaled - max_scaled).exp() as f64;
    }

    if sum_exp <= 0.0 {
        return None;
    }

    Some(max_scaled + (sum_exp as f32).ln())
}

fn best_finite_logit(logits: &[f32], fallback_token: u32) -> (u32, f32) {
    let mut best_idx = fallback_token as usize;
    let mut best_logit = f32::NEG_INFINITY;
    for (idx, logit) in logits.iter().enumerate() {
        if *logit > best_logit {
            best_idx = idx;
            best_logit = *logit;
        }
    }
    if !best_logit.is_finite() {
        return (fallback_token, f32::NEG_INFINITY);
    }
    (best_idx as u32, best_logit)
}

fn probability_for_token_from_logits(
    logits: &[f32],
    token_id: u32,
    logsumexp: f32,
    inv_temperature: f32,
) -> Option<f32> {
    let idx = token_id as usize;
    if idx >= logits.len() {
        return None;
    }
    let logit = logits[idx];
    if !logit.is_finite() {
        return None;
    }
    Some((logit * inv_temperature - logsumexp).exp())
}

fn probability_for_token(log_probs: &[f32], token_id: u32) -> Option<f32> {
    let idx = token_id as usize;
    if idx >= log_probs.len() {
        return None;
    }
    let log_prob = log_probs[idx];
    if !log_prob.is_finite() {
        return None;
    }
    Some(log_prob.exp())
}

fn sample_token_from_log_probs<R: Rng + ?Sized>(
    log_probs: &[f32],
    temperature: f32,
    fallback_token: u32,
    rng: &mut R,
) -> (u32, f32) {
    if temperature <= 0.0 {
        let mut best_idx = fallback_token as usize;
        let mut best_logprob = f32::NEG_INFINITY;
        for (idx, logprob) in log_probs.iter().enumerate() {
            if *logprob > best_logprob {
                best_idx = idx;
                best_logprob = *logprob;
            }
        }
        if !best_logprob.is_finite() {
            return (fallback_token, f32::NEG_INFINITY);
        }
        return (best_idx as u32, best_logprob);
    }

    let mut sum = 0.0f64;
    for logprob in log_probs {
        if logprob.is_finite() {
            sum += logprob.exp() as f64;
        }
    }

    if sum <= 0.0 {
        return (fallback_token, f32::NEG_INFINITY);
    }

    let mut threshold = rng.gen_range(0.0..sum);
    for (idx, logprob) in log_probs.iter().enumerate() {
        if !logprob.is_finite() {
            continue;
        }
        threshold -= logprob.exp() as f64;
        if threshold <= 0.0 {
            return (idx as u32, *logprob);
        }
    }

    let mut best_idx = fallback_token as usize;
    let mut best_logprob = f32::NEG_INFINITY;
    for (idx, logprob) in log_probs.iter().enumerate() {
        if *logprob > best_logprob {
            best_idx = idx;
            best_logprob = *logprob;
        }
    }
    (best_idx as u32, best_logprob)
}

fn token_compression_ratio(tokens: &[u32], vocab_size: usize) -> Option<f32> {
    if tokens.is_empty() || vocab_size == 0 {
        return None;
    }

    let width = ((vocab_size as f64).log2().floor() as usize / 8).saturating_add(1);
    let mut raw = Vec::with_capacity(tokens.len() * width);
    for token in tokens {
        let value = *token as u64;
        for byte in 0..width {
            raw.push(((value >> (8 * byte)) & 0xFF) as u8);
        }
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw).ok()?;
    let compressed = encoder.finish().ok()?;
    if compressed.is_empty() {
        return None;
    }
    Some(raw.len() as f32 / compressed.len() as f32)
}

fn should_skip_as_no_speech(
    attempt: &WhisperDecodeAttempt,
    logprob_threshold: f32,
    no_speech_threshold: f32,
) -> bool {
    attempt.avg_logprob < logprob_threshold
        && attempt
            .no_speech_prob
            .map(|prob| prob > no_speech_threshold)
            .unwrap_or(false)
}

fn decode_retry_reasons(
    attempt: &WhisperDecodeAttempt,
    logprob_threshold: f32,
    compression_ratio_threshold: Option<f32>,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if attempt.repetition_loop {
        reasons.push("repetition_loop");
    }
    if !attempt.ended_with_eot {
        reasons.push("missing_eot");
    }
    if attempt.avg_logprob < logprob_threshold {
        reasons.push("low_logprob");
    }
    if let (Some(ratio), Some(threshold)) = (attempt.compression_ratio, compression_ratio_threshold)
    {
        if ratio > threshold {
            reasons.push("compression_ratio");
        }
    }
    if has_low_word_diversity(&attempt.text) {
        reasons.push("low_word_diversity");
    }
    reasons
}

fn should_retry_decode(
    attempt: &WhisperDecodeAttempt,
    logprob_threshold: f32,
    compression_ratio_threshold: Option<f32>,
) -> bool {
    // Fallback criteria aligned with upstream Whisper implementations:
    // repetition/unfinished decode, low avg logprob, and optional compression ratio.
    !decode_retry_reasons(attempt, logprob_threshold, compression_ratio_threshold).is_empty()
}

fn record_unique_reason(reasons: &mut Vec<&'static str>, reason: &'static str) {
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

fn record_unique_reasons(reasons: &mut Vec<&'static str>, new_reasons: &[&'static str]) {
    for reason in new_reasons {
        record_unique_reason(reasons, reason);
    }
}

fn whisper_attempt_diagnostics(
    temperature: f32,
    attempt: &WhisperDecodeAttempt,
    retry_reasons: &[&'static str],
    no_speech_skip: bool,
) -> serde_json::Value {
    json!({
        "temperature": temperature,
        "avg_logprob": attempt.avg_logprob,
        "no_speech_prob": attempt.no_speech_prob,
        "ended_with_eot": attempt.ended_with_eot,
        "repetition_loop": attempt.repetition_loop,
        "compression_ratio": attempt.compression_ratio,
        "decode_steps": attempt.decode_steps,
        "generated_token_count": attempt.generated_token_count,
        "sampled_token_count": attempt.sampled_token_count,
        "retry_reasons": retry_reasons,
        "no_speech": no_speech_skip,
    })
}

fn has_low_word_diversity(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect();
    if words.len() < 8 {
        return false;
    }

    let unique = words.iter().collect::<HashSet<_>>().len();
    (unique as f32 / words.len() as f32) < 0.6
}

fn is_better_attempt(
    candidate: &WhisperDecodeAttempt,
    current_best: &WhisperDecodeAttempt,
) -> bool {
    if candidate.ended_with_eot != current_best.ended_with_eot {
        return candidate.ended_with_eot;
    }
    if candidate.repetition_loop != current_best.repetition_loop {
        return !candidate.repetition_loop;
    }
    if candidate.avg_logprob != current_best.avg_logprob {
        return candidate.avg_logprob > current_best.avg_logprob;
    }
    candidate.text.len() > current_best.text.len()
}

fn text_delta<'a>(previous: &str, current: &'a str) -> &'a str {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta;
    }

    let mut shared_prefix_bytes = 0usize;
    for (left, right) in previous.chars().zip(current.chars()) {
        if left != right {
            break;
        }
        shared_prefix_bytes += right.len_utf8();
    }

    &current[shared_prefix_bytes..]
}

fn find_suffix_token_repetition(ids: &[u32]) -> Option<(usize, usize)> {
    if ids.len() < REPETITION_GUARD_MIN_TOTAL_TOKENS {
        return None;
    }

    let max_span = (ids.len() / 2).min(REPETITION_GUARD_MAX_SPAN_TOKENS);
    if max_span < REPETITION_GUARD_MIN_SPAN_TOKENS {
        return None;
    }

    for span in (REPETITION_GUARD_MIN_SPAN_TOKENS..=max_span).rev() {
        let tail_start = ids.len() - span;
        let tail = &ids[tail_start..];
        let mut repeats = 1usize;

        while ids.len() >= span.saturating_mul(repeats + 1) {
            let start = ids.len() - span * (repeats + 1);
            let end = start + span;
            if &ids[start..end] == tail {
                repeats += 1;
            } else {
                break;
            }
        }

        if repeats >= 2 {
            return Some((span, repeats));
        }
    }

    None
}

fn decode_step_budget(
    prompt_len: usize,
    max_target_positions: usize,
    generation_max_length: usize,
) -> Result<usize> {
    if max_target_positions == 0 || prompt_len >= max_target_positions {
        return Err(Error::InvalidInput(format!(
            "Whisper decode prompt length {} exceeds decoder context {}",
            prompt_len, max_target_positions
        )));
    }

    let prompt_budget = max_target_positions - prompt_len;

    // Whisper decoder positional embeddings are bounded by max_target_positions.
    // Keep generated tokens within remaining context budget to avoid narrow() overflow.
    Ok(generation_max_length.max(1).min(prompt_budget))
}

#[cfg(test)]
mod tests {
    use super::{
        adaptive_decode_budget, apply_whisper_decode_constraints, build_whisper_prompt_prefix,
        capped_decode_temperatures, decode_retry_reasons, decode_step_budget,
        find_suffix_token_repetition, has_low_word_diversity, logits_to_log_probs,
        logits_to_log_probs_in_place, text_delta, token_contains_numeral_or_symbol,
        trimmed_audio_bounds, use_cuda_whisper_dtype_shim, whisper_device_diagnostics,
        whisper_impl_name, WhisperDecodeAttempt, WhisperSpecialTokens,
    };

    #[test]
    fn whisper_dtype_shim_is_cuda_only() {
        assert!(!use_cuda_whisper_dtype_shim(&candle_core::Device::Cpu));
    }

    #[test]
    fn whisper_device_diagnostics_keep_cpu_and_metal_on_upstream_f32() {
        for kind in [
            crate::backends::DeviceKind::Cpu,
            crate::backends::DeviceKind::Metal,
        ] {
            let diagnostics = whisper_device_diagnostics(kind, candle_core::DType::F32, false);
            let expected_kind = format!("{kind:?}");

            assert_eq!(
                diagnostics.get("kind").and_then(|value| value.as_str()),
                Some(expected_kind.as_str())
            );
            assert_eq!(
                diagnostics
                    .get("model_dtype")
                    .and_then(|value| value.as_str()),
                Some("F32")
            );
            assert_eq!(
                diagnostics
                    .get("cuda_dtype_shim")
                    .and_then(|value| value.as_bool()),
                Some(false)
            );
            assert_eq!(
                diagnostics
                    .get("whisper_impl")
                    .and_then(|value| value.as_str()),
                Some("upstream_candle")
            );
        }
    }

    #[test]
    fn whisper_impl_name_marks_cuda_dtype_shim_explicitly() {
        assert_eq!(whisper_impl_name(false), "upstream_candle");
        assert_eq!(whisper_impl_name(true), "cuda_dtype_shim");
    }

    #[test]
    fn decode_step_budget_clamps_generation_to_remaining_context() {
        let budget = decode_step_budget(4, 448, 448).expect("budget");
        assert_eq!(budget, 444);
    }

    #[test]
    fn decode_step_budget_rejects_prompt_overflow() {
        assert!(decode_step_budget(448, 448, 448).is_err());
        assert!(decode_step_budget(449, 448, 448).is_err());
    }

    #[test]
    fn detects_suffix_token_repetition() {
        let mut ids = Vec::new();
        ids.extend(1u32..=12);
        ids.extend(1u32..=12);
        let repetition = find_suffix_token_repetition(&ids);
        assert_eq!(repetition, Some((12, 2)));
    }

    #[test]
    fn ignores_short_or_non_repeating_suffixes() {
        let ids: Vec<u32> = (1..=16).collect();
        assert_eq!(find_suffix_token_repetition(&ids), None);
    }

    #[test]
    fn in_place_log_probs_match_allocating_variant() {
        let logits = vec![0.25f32, 0.75, -1.0, f32::NEG_INFINITY, 2.0];
        let expected = logits_to_log_probs(&logits, 0.7);
        let mut out = Vec::new();
        logits_to_log_probs_in_place(&logits, 0.7, &mut out);
        assert_eq!(expected.len(), out.len());
        for (left, right) in expected.iter().zip(out.iter()) {
            if left.is_finite() || right.is_finite() {
                assert!((left - right).abs() < 1e-5, "{left} != {right}");
            } else {
                assert!(!left.is_finite() && !right.is_finite());
            }
        }
    }

    #[test]
    fn decode_constraints_suppress_blank_only_at_begin() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: None,
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };
        let mut begin_logits = vec![0.0f32; 10];
        apply_whisper_decode_constraints(
            &mut begin_logits,
            true,
            &[],
            &[],
            &[],
            &special,
            true,
            &[],
        );
        assert_eq!(begin_logits[3], f32::NEG_INFINITY);
        assert_eq!(begin_logits[4], f32::NEG_INFINITY);

        let mut next_logits = vec![0.0f32; 10];
        apply_whisper_decode_constraints(
            &mut next_logits,
            false,
            &[],
            &[],
            &[],
            &special,
            true,
            &[],
        );
        assert!(next_logits[3].is_finite());
        assert!(next_logits[4].is_finite());
    }

    #[test]
    fn decode_constraints_mask_numeral_symbol_tokens() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: None,
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };
        let mut logits = vec![0.0f32; 10];
        apply_whisper_decode_constraints(&mut logits, false, &[], &[], &[], &special, false, &[5]);
        assert_eq!(logits[5], f32::NEG_INFINITY);
        assert_eq!(logits[7], f32::NEG_INFINITY);
        assert_eq!(logits[8], f32::NEG_INFINITY);
        assert_eq!(logits[9], f32::NEG_INFINITY);
    }

    #[test]
    fn numeral_symbol_filter_detects_digits_and_symbols() {
        assert!(token_contains_numeral_or_symbol("12"));
        assert!(token_contains_numeral_or_symbol("$"));
        assert!(!token_contains_numeral_or_symbol("word"));
    }

    #[test]
    fn text_delta_uses_prefix_fast_path() {
        assert_eq!(text_delta("the quick", "the quick brown"), " brown");
    }

    #[test]
    fn text_delta_handles_midstring_rewrites() {
        assert_eq!(text_delta("hello wrld", "hello world"), "orld");
    }

    #[test]
    fn capped_decode_temperatures_limits_retry_count() {
        let temps = capped_decode_temperatures(0.0, 0.2, false, 1);
        assert_eq!(temps, vec![0.0, 0.2]);
    }

    #[test]
    fn capped_decode_temperatures_respects_no_fallback() {
        let temps = capped_decode_temperatures(0.4, 0.2, true, 8);
        assert_eq!(temps, vec![0.4]);
    }

    #[test]
    fn adaptive_decode_budget_scales_with_audio_duration() {
        let budget = adaptive_decode_budget(3.6, 448, 12.0, 32, 8);
        assert_eq!(budget, 52);
    }

    #[test]
    fn adaptive_decode_budget_respects_cap_and_minimum() {
        let budget = adaptive_decode_budget(0.4, 40, 2.0, 32, 8);
        assert_eq!(budget, 32);

        let capped = adaptive_decode_budget(30.0, 120, 12.0, 32, 8);
        assert_eq!(capped, 120);
    }

    #[test]
    fn whisper_prompt_prefix_keeps_default_controls_unchanged() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: Some(9),
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };

        let prefix = build_whisper_prompt_prefix(&special, Some(5), &[], 12, 4).expect("prefix");

        assert_eq!(prefix.ids, vec![1, 5, 2, 7]);
        assert!(!prefix.diagnostics.initial_prompt_requested);
        assert_eq!(prefix.diagnostics.initial_prompt_tokens_used, 0);
    }

    #[test]
    fn whisper_prompt_prefix_truncates_initial_prompt_tail() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: Some(9),
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };

        let prefix = build_whisper_prompt_prefix(&special, Some(5), &[10, 11, 12, 13, 14], 12, 3)
            .expect("prefix");

        assert_eq!(prefix.ids, vec![9, 12, 13, 14, 1, 5, 2, 7]);
        assert!(prefix.diagnostics.initial_prompt_requested);
        assert_eq!(prefix.diagnostics.initial_prompt_token_count, 5);
        assert_eq!(prefix.diagnostics.initial_prompt_tokens_used, 3);
        assert_eq!(prefix.diagnostics.initial_prompt_tokens_truncated, 2);
        assert_eq!(prefix.diagnostics.previous_context_token_id, Some(9));
    }

    #[test]
    fn trimmed_audio_bounds_removes_leading_and_trailing_silence() {
        let sr = 16_000u32;
        let mut audio = vec![0.0f32; 8_000];
        audio.extend(vec![0.2f32; 16_000]);
        audio.extend(vec![0.0f32; 8_000]);

        let (start, end) = trimmed_audio_bounds(&audio, sr, 0.02, 0.0015, 120, 300, 120, 0.8);
        assert!(start > 0);
        assert!(end < audio.len());
        assert!(end > start);
    }

    #[test]
    fn trimmed_audio_bounds_keeps_short_clips_untouched() {
        let sr = 16_000u32;
        let audio = vec![0.0f32; 4_000];
        let (start, end) = trimmed_audio_bounds(&audio, sr, 0.02, 0.0015, 120, 300, 120, 0.8);
        assert_eq!(start, 0);
        assert_eq!(end, audio.len());
    }

    #[test]
    fn trimmed_audio_bounds_preserves_short_leading_silence() {
        let sr = 16_000u32;
        let mut audio = vec![0.0f32; 6_000];
        audio.extend(vec![0.2f32; 16_000]);
        audio.extend(vec![0.0f32; 6_000]);

        let (start, end) = trimmed_audio_bounds(&audio, sr, 0.02, 0.0015, 120, 500, 120, 0.8);
        assert_eq!(start, 0);
        assert!(end < audio.len());
    }

    #[test]
    fn low_word_diversity_flags_repetitive_output() {
        assert!(has_low_word_diversity(
            "The quick quick brown fox fox jumps jumps over the little the little"
        ));
    }

    #[test]
    fn low_word_diversity_allows_normal_transcript() {
        assert!(!has_low_word_diversity(
            "The quick brown fox jumps over the lazy dog"
        ));
    }

    #[test]
    fn decode_retry_reasons_are_structured() {
        let attempt = WhisperDecodeAttempt {
            text: "same same same same same same same same".to_string(),
            avg_logprob: -2.0,
            no_speech_prob: Some(0.1),
            ended_with_eot: false,
            repetition_loop: true,
            compression_ratio: Some(3.0),
            generated_token_count: 8,
            sampled_token_count: 9,
            decode_steps: 9,
        };

        let reasons = decode_retry_reasons(&attempt, -1.0, Some(2.4));

        assert!(reasons.contains(&"repetition_loop"));
        assert!(reasons.contains(&"missing_eot"));
        assert!(reasons.contains(&"low_logprob"));
        assert!(reasons.contains(&"compression_ratio"));
        assert!(reasons.contains(&"low_word_diversity"));
    }
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;

    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = left.saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(audio[left] * (1.0 - frac) + audio[right] * frac);
    }
    out
}

fn language_name_to_code(language: &str) -> Option<&'static str> {
    WHISPER_LANGUAGES
        .iter()
        .find(|(_code, name)| *name == language)
        .map(|(code, _name)| *code)
}

fn language_alias_to_code(language: &str) -> Option<&'static str> {
    match language {
        "burmese" => Some("my"),
        "valencian" => Some("ca"),
        "flemish" => Some("nl"),
        "haitian" => Some("ht"),
        "letzeburgesch" => Some("lb"),
        "pushto" => Some("ps"),
        "panjabi" => Some("pa"),
        "moldavian" | "moldovan" => Some("ro"),
        "sinhalese" => Some("si"),
        "castilian" => Some("es"),
        "mandarin" => Some("zh"),
        _ => None,
    }
}

// Mirrors Whisper multilingual language table from upstream implementations.
const WHISPER_LANGUAGES: [(&str, &str); 100] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];
