//! Native VibeVoice-ASR model path.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor, D};
use serde::Serialize;
use serde_json::json;
use tracing::info;

use crate::backends::{DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{Qwen3Cache, Qwen3Model, Qwen3WeightLayout};
use crate::models::architectures::vibevoice::config::{
    VibeVoiceConfig, VibeVoicePreprocessorConfig,
};
use crate::models::architectures::vibevoice::connector::SpeechConnector;
use crate::models::architectures::vibevoice::prompt::VibeVoicePromptTokenizer;
use crate::models::architectures::vibevoice::tokenizer::{
    VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer, VibeVoiceTokenizerEncoderOutput,
    VibeVoiceTokenizerStreamingCache,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::weights::gguf::load_model_weights;

const DEFAULT_MAX_NEW_TOKENS: usize = 768;
const DEFAULT_MAX_AUDIO_SECONDS: f32 = 60.0 * 60.0;
const DEFAULT_CUDA_MAX_AUDIO_SECONDS: f32 = 2.0 * 60.0;
const CUDA_MAX_AUDIO_SECONDS_ENV: &str = "IZWI_VIBEVOICE_ASR_CUDA_MAX_AUDIO_SECS";
const TOKENIZER_STREAMING_CHUNK_SECONDS: usize = 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceAsrGenerationOptions {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for VibeVoiceAsrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct VibeVoiceAsrSegment {
    start_time: Option<f32>,
    end_time: Option<f32>,
    speaker_id: Option<String>,
    content: String,
}

#[derive(Debug, Clone, PartialEq)]
struct VibeVoiceAsrParsedOutput {
    text: String,
    raw_text: String,
    format: &'static str,
    segments: Vec<VibeVoiceAsrSegment>,
}

#[derive(Debug, Clone)]
struct VibeVoiceAsrPreprocessStats {
    normalized: bool,
    target_db_fs: f32,
    rms_before: f32,
    gain: f32,
    clipping_divisor: f32,
}

#[derive(Debug, Clone)]
struct VibeVoiceAsrEncodeStats {
    streaming: bool,
    chunks: usize,
    chunk_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceAsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

pub struct VibeVoiceAsrModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: VibeVoiceConfig,
    preprocessor: VibeVoicePreprocessorConfig,
    tokenizer: VibeVoicePromptTokenizer,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    semantic_tokenizer: VibeVoiceSemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    language_model: Qwen3Model,
}

impl VibeVoiceAsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::VibeVoiceAsr {
            return Err(Error::InvalidInput(format!(
                "VibeVoiceAsrModel cannot load non-ASR variant {variant}"
            )));
        }
        let config = VibeVoiceConfig::load(model_dir)?;
        if config.is_tts() {
            return Err(Error::ModelLoadError(
                "VibeVoice-ASR loader received a TTS config".to_string(),
            ));
        }
        let preprocessor = VibeVoicePreprocessorConfig::load(model_dir)?;
        let dtype = std::env::var("IZWI_VIBEVOICE_ASR_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
            .map(|raw| {
                device.select_model_dtype_checked(
                    ModelFamily::VibeVoiceAsr,
                    Some(raw),
                    "VibeVoice ASR",
                )
            })
            .transpose()?
            .unwrap_or_else(|| device.select_model_dtype(ModelFamily::VibeVoiceAsr, None));
        let vb = load_model_weights(model_dir, dtype, &device.device)?;
        let tokenizer =
            VibeVoicePromptTokenizer::load(model_dir, config.decoder_config.vocab_size)?;
        let acoustic_tokenizer = VibeVoiceAcousticTokenizer::load(
            &config.acoustic_tokenizer_config,
            vb.pp("model.acoustic_tokenizer"),
        )?;
        let semantic_tokenizer = VibeVoiceSemanticTokenizer::load(
            &config.semantic_tokenizer_config,
            vb.pp("model.semantic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("model.acoustic_connector"),
        )?;
        let semantic_connector = SpeechConnector::load(
            config.semantic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("model.semantic_connector"),
        )?;
        let language_model = Qwen3Model::load_with_layout(
            config.decoder_config.clone(),
            vb,
            Qwen3WeightLayout::VIBEVOICE,
        )?;
        info!(
            "Loaded VibeVoice-ASR from {:?} on {:?} with dtype {:?}",
            model_dir, device.kind, dtype
        );
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            preprocessor,
            tokenizer,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            language_model,
        })
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        self.transcribe_with_details_and_prompt_and_options(
            audio,
            sample_rate,
            language,
            prompt,
            VibeVoiceAsrGenerationOptions::default(),
        )
    }

    pub fn transcribe_with_details_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(audio, sample_rate, language, prompt, options, &mut no_op)
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
            VibeVoiceAsrGenerationOptions::default(),
            on_delta,
        )
    }

    pub fn transcribe_with_callback_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_internal(audio, sample_rate, language, prompt, options, on_delta)?
            .text)
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(vibevoice_asr_max_audio_seconds_hint(self.device.kind))
    }

    fn transcribe_internal(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input cannot be empty".to_string(),
            ));
        }
        let total_started = Instant::now();
        let preprocess_started = Instant::now();
        let (processed_audio, preprocess_stats) =
            preprocess_asr_audio(audio, sample_rate, &self.preprocessor)?;
        let preprocess_ms = elapsed_ms(preprocess_started);
        if processed_audio.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input produced no samples after preprocessing".to_string(),
            ));
        }
        let target_sample_rate = self.preprocessor.target_sample_rate();
        let audio_seconds = processed_audio.len() as f32 / target_sample_rate as f32;
        let compress_ratio = self.preprocessor.speech_tok_compress_ratio.max(1);
        let expected_acoustic_frames = asr_placeholder_count(processed_audio.len(), compress_ratio);
        let mut encoder_audio = processed_audio.clone();
        let encoder_samples = expected_acoustic_frames.saturating_mul(compress_ratio);
        if encoder_audio.len() < encoder_samples {
            encoder_audio.resize(encoder_samples, 0.0);
        }
        let speech = Tensor::from_vec(encoder_audio, (1, 1, encoder_samples), &self.device.device)?
            .to_dtype(self.dtype)?;
        let audio_encode_started = Instant::now();
        let (speech_features, encode_stats) = self.encode_speech(&speech)?;
        let audio_encode_ms = elapsed_ms(audio_encode_started);
        let acoustic_frames = speech_features.dim(1)?;
        if acoustic_frames != expected_acoustic_frames {
            return Err(Error::InferenceError(format!(
                "VibeVoice-ASR tokenizer produced {acoustic_frames} frames but processor reserved {expected_acoustic_frames}"
            )));
        }
        let extra = prompt_instruction(language, prompt);
        let prompt = self.tokenizer.build_asr_prompt(
            audio_seconds,
            expected_acoustic_frames,
            extra.as_deref(),
        )?;
        let input_ids = Tensor::from_vec(
            prompt.input_ids.clone(),
            (1, prompt.input_ids.len()),
            &self.device.device,
        )?;
        let input_embeds = self.language_model.embeddings(&input_ids)?;
        let input_embeds = replace_range_with_features(
            &input_embeds,
            prompt.acoustic_input_range.clone(),
            &speech_features.to_dtype(input_embeds.dtype())?,
        )?;

        let mut cache = self.build_decode_cache();
        let decode_cache_dense_max_tokens = cache.dense_decode_max_tokens();
        let cuda_device_argmax = self.device.kind.is_cuda();
        let prefill_started = Instant::now();
        let logits =
            self.language_model
                .forward_with_embeds(&input_embeds, 0, Some(&mut cache), None)?;
        let prefill_ms = elapsed_ms(prefill_started);
        let mut pos = prompt.input_ids.len();
        let mut next = argmax_last_logits(&logits, cuda_device_argmax)?;
        let mut generated = Vec::new();
        let mut assembled = String::new();
        let built_in_stop_tokens = [
            self.tokenizer.specials().im_end,
            self.tokenizer.specials().endoftext,
        ];
        let stop_tokens = collect_stop_token_ids(&built_in_stop_tokens, &options.stop_token_ids);
        let stop_sequences = sanitize_stop_sequences(&options.stop_sequences);
        let max_new_tokens = options.max_new_tokens.max(1);
        let mut stop_reason = None::<&'static str>;
        let mut stop_token_id = None::<u32>;
        let mut stop_sequence = None::<String>;
        let decode_started = Instant::now();

        for _ in 0..max_new_tokens {
            if stop_tokens.contains(&next) {
                stop_reason = Some(if built_in_stop_tokens.contains(&next) {
                    "model_stop_token"
                } else {
                    "request_stop_token"
                });
                stop_token_id = Some(next);
                break;
            }
            generated.push(next);
            let decoded = self.tokenizer.decode(&generated)?;
            let (visible_decoded, matched_stop_sequence) =
                truncate_at_stop_sequence(&decoded, &stop_sequences);
            if visible_decoded.len() > assembled.len() {
                on_delta(&visible_decoded[assembled.len()..]);
            }
            assembled = visible_decoded;
            if let Some(sequence) = matched_stop_sequence {
                stop_reason = Some("stop_sequence");
                stop_sequence = Some(sequence);
                break;
            }

            let token = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            let logits = self.language_model.forward(&token, pos, Some(&mut cache))?;
            pos += 1;
            next = argmax_last_logits(&logits, cuda_device_argmax)?;
        }
        let decode_ms = elapsed_ms(decode_started);
        let reached_max_tokens = stop_reason.is_none() && generated.len() >= max_new_tokens;
        if reached_max_tokens {
            stop_reason = Some("max_tokens");
        }

        let parsed = parse_vibevoice_asr_output(&assembled);
        Ok(VibeVoiceAsrTranscriptionOutput {
            text: parsed.text.clone(),
            language: language.map(ToOwned::to_owned),
            diagnostics: Some(json!({
                "model_family": "vibevoice_asr",
                "model_dir": self.model_dir.display().to_string(),
                "audio": {
                    "input_sample_rate": sample_rate,
                    "input_samples": audio.len(),
                    "resampled_sample_rate": target_sample_rate,
                    "resampled_samples": processed_audio.len(),
                    "encoder_samples": encoder_samples,
                    "duration_seconds": audio_seconds,
                    "acoustic_frames": acoustic_frames,
                    "expected_acoustic_frames": expected_acoustic_frames,
                    "speech_tok_compress_ratio": compress_ratio,
                    "normalized": preprocess_stats.normalized,
                    "target_db_fs": preprocess_stats.target_db_fs,
                    "rms_before_normalization": preprocess_stats.rms_before,
                    "normalization_gain": preprocess_stats.gain,
                    "clipping_divisor": preprocess_stats.clipping_divisor,
                    "tokenizer_streaming": encode_stats.streaming,
                    "tokenizer_chunks": encode_stats.chunks,
                    "tokenizer_chunk_samples": encode_stats.chunk_samples,
                },
                "prompt": {
                    "tokens": prompt.prompt_token_count,
                    "acoustic_input_tokens": prompt.acoustic_input_range.end.saturating_sub(prompt.acoustic_input_range.start),
                    "language": language,
                    "extra_prompt": extra,
                },
                "decode": {
                    "generated_tokens": generated.len(),
                    "max_new_tokens": max_new_tokens,
                    "stop_reason": stop_reason,
                    "stop_token_id": stop_token_id,
                    "stop_sequence": stop_sequence,
                    "reached_max_tokens": reached_max_tokens,
                    "configured_stop_token_ids": options.stop_token_ids,
                    "configured_stop_sequences": stop_sequences,
                },
                "output": {
                    "format": parsed.format,
                    "raw_text": parsed.raw_text,
                    "segment_count": parsed.segments.len(),
                    "segments": parsed.segments,
                },
                "execution": {
                    "dtype": format!("{:?}", self.dtype),
                    "device_kind": format!("{:?}", self.device.kind),
                    "decoder_layers": self.config.decoder_config.num_hidden_layers,
                    "cuda_dense_decode_cache": decode_cache_dense_max_tokens > 0,
                    "dense_decode_max_tokens": decode_cache_dense_max_tokens,
                    "cuda_device_argmax": cuda_device_argmax,
                },
                "timings_ms": {
                    "preprocess": preprocess_ms,
                    "audio_encode": audio_encode_ms,
                    "prefill": prefill_ms,
                    "decode": decode_ms,
                    "model_total": elapsed_ms(total_started),
                }
            })),
        })
    }

    fn build_decode_cache(&self) -> Qwen3Cache {
        if self.device.kind.is_cuda() {
            return Qwen3Cache::with_page_size_and_dense_decode(
                self.language_model.num_layers(),
                default_kv_page_size(),
                &self.device.device,
            );
        }
        Qwen3Cache::new(self.language_model.num_layers())
    }

    fn encode_speech(&self, speech: &Tensor) -> Result<(Tensor, VibeVoiceAsrEncodeStats)> {
        let total_samples = speech.dim(2)?;
        let chunk_samples = tokenizer_streaming_chunk_samples(
            self.preprocessor.target_sample_rate(),
            self.preprocessor.speech_tok_compress_ratio,
        );
        let can_stream = total_samples > chunk_samples
            && self.config.acoustic_tokenizer_config.causal
            && self.config.semantic_tokenizer_config.causal;
        if can_stream {
            return self.encode_speech_streaming(speech, chunk_samples);
        }

        Ok((
            self.encode_speech_full(speech)?,
            VibeVoiceAsrEncodeStats {
                streaming: false,
                chunks: 1,
                chunk_samples: total_samples,
            },
        ))
    }

    fn encode_speech_full(&self, speech: &Tensor) -> Result<Tensor> {
        let acoustic = self.acoustic_tokenizer.encode(speech)?;
        let acoustic = self.acoustic_tokenizer.sample(&acoustic)?;
        let acoustic = self.acoustic_connector.forward(&acoustic)?;

        let semantic = self.semantic_tokenizer.encode(speech)?.mode();
        let semantic = self.semantic_connector.forward(&semantic)?;

        self.combine_speech_features(acoustic, semantic)
    }

    fn encode_speech_streaming(
        &self,
        speech: &Tensor,
        chunk_samples: usize,
    ) -> Result<(Tensor, VibeVoiceAsrEncodeStats)> {
        let total_samples = speech.dim(2)?;
        let ranges = tokenizer_chunk_ranges(total_samples, chunk_samples);
        let mut acoustic_cache = VibeVoiceTokenizerStreamingCache::new();
        let mut semantic_cache = VibeVoiceTokenizerStreamingCache::new();
        let mut acoustic_means = Vec::with_capacity(ranges.len());
        let mut semantic_means = Vec::with_capacity(ranges.len());
        let mut acoustic_std = None;

        for (start, len) in &ranges {
            let chunk = speech.narrow(2, *start, *len)?;
            let acoustic = self
                .acoustic_tokenizer
                .encode_streaming(&chunk, &mut acoustic_cache)?;
            acoustic_std = acoustic_std.or(acoustic.std);
            acoustic_means.push(acoustic.mean);

            let semantic = self
                .semantic_tokenizer
                .encode_streaming(&chunk, &mut semantic_cache)?;
            semantic_means.push(semantic.mean);
        }

        let acoustic = VibeVoiceTokenizerEncoderOutput {
            mean: Tensor::cat(&acoustic_means, 1)?,
            std: acoustic_std,
        };
        let acoustic = self.acoustic_tokenizer.sample(&acoustic)?;
        let acoustic = self.acoustic_connector.forward(&acoustic)?;

        let semantic = Tensor::cat(&semantic_means, 1)?;
        let semantic = self.semantic_connector.forward(&semantic)?;
        let features = self.combine_speech_features(acoustic, semantic)?;
        Ok((
            features,
            VibeVoiceAsrEncodeStats {
                streaming: true,
                chunks: ranges.len(),
                chunk_samples,
            },
        ))
    }

    fn combine_speech_features(&self, acoustic: Tensor, semantic: Tensor) -> Result<Tensor> {
        if acoustic.dims() != semantic.dims() {
            return Err(Error::InferenceError(format!(
                "VibeVoice-ASR acoustic/semantic feature shape mismatch: {:?} vs {:?}",
                acoustic.dims(),
                semantic.dims()
            )));
        }
        acoustic.broadcast_add(&semantic).map_err(Error::from)
    }
}

fn prompt_instruction(language: Option<&str>, prompt: Option<&str>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(language) = language.filter(|value| {
        let value = value.trim();
        !value.is_empty() && !value.eq_ignore_ascii_case("auto")
    }) {
        parts.push(format!("The spoken language is {}.", language.trim()));
    }
    if let Some(prompt) = prompt.filter(|value| !value.trim().is_empty()) {
        parts.push(prompt.trim().to_string());
    }
    (!parts.is_empty()).then(|| parts.join(" "))
}

fn replace_range_with_features(
    embeds: &Tensor,
    range: std::ops::Range<usize>,
    features: &Tensor,
) -> Result<Tensor> {
    let seq_len = embeds.dim(1)?;
    let feature_len = features.dim(1)?;
    if feature_len != range.end.saturating_sub(range.start) {
        return Err(Error::InferenceError(format!(
            "VibeVoice prompt reserved {} acoustic tokens but encoder produced {feature_len}",
            range.end.saturating_sub(range.start)
        )));
    }
    let mut parts = Vec::new();
    if range.start > 0 {
        parts.push(embeds.narrow(1, 0, range.start)?);
    }
    parts.push(features.clone());
    if range.end < seq_len {
        parts.push(embeds.narrow(1, range.end, seq_len - range.end)?);
    }
    Tensor::cat(&parts, 1).map_err(Error::from)
}

fn argmax_last_logits(logits: &Tensor, use_device_argmax: bool) -> Result<u32> {
    let seq_len = logits.dim(1)?;
    let row = logits.i((0, seq_len - 1))?;
    if use_device_argmax {
        return argmax_logits_row_device(&row);
    }
    argmax_logits_row_host(&row)
}

fn argmax_logits_row_host(row: &Tensor) -> Result<u32> {
    let row = row.to_dtype(DType::F32)?;
    let values = row.to_vec1::<f32>()?;
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| Error::InferenceError("VibeVoice-ASR logits row was empty".to_string()))
}

fn argmax_logits_row_device(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched VibeVoice-ASR logits row: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected VibeVoice-ASR logits row rank for argmax: {rank}"
            )));
        }
    };
    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn collect_stop_token_ids(built_in: &[u32], requested: &[u32]) -> Vec<u32> {
    let mut stop_tokens = Vec::with_capacity(built_in.len() + requested.len());
    for token in built_in.iter().chain(requested.iter()).copied() {
        if !stop_tokens.contains(&token) {
            stop_tokens.push(token);
        }
    }
    stop_tokens
}

fn sanitize_stop_sequences(sequences: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for sequence in sequences {
        let trimmed = sequence.trim();
        if !trimmed.is_empty() && !out.iter().any(|existing: &String| existing == trimmed) {
            out.push(trimmed.to_string());
        }
    }
    out
}

fn truncate_at_stop_sequence(text: &str, stop_sequences: &[String]) -> (String, Option<String>) {
    let mut earliest: Option<(usize, &str)> = None;
    for sequence in stop_sequences {
        let Some(idx) = text.find(sequence) else {
            continue;
        };
        if earliest
            .map(|(existing_idx, _)| idx < existing_idx)
            .unwrap_or(true)
        {
            earliest = Some((idx, sequence.as_str()));
        }
    }

    if let Some((idx, sequence)) = earliest {
        (text[..idx].to_string(), Some(sequence.to_string()))
    } else {
        (text.to_string(), None)
    }
}

fn parse_vibevoice_asr_output(raw: &str) -> VibeVoiceAsrParsedOutput {
    let raw_text = cleanup_transcript_text(raw);
    for candidate in json_output_candidates(&raw_text) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&candidate) else {
            continue;
        };
        let segments = vibevoice_segments_from_value(&value);
        if !segments.is_empty() {
            return VibeVoiceAsrParsedOutput {
                text: join_segment_contents(&segments),
                raw_text,
                format: "segments",
                segments,
            };
        }
    }

    VibeVoiceAsrParsedOutput {
        text: raw_text.clone(),
        raw_text,
        format: "text",
        segments: Vec::new(),
    }
}

fn json_output_candidates(text: &str) -> Vec<String> {
    let stripped = strip_json_code_fence(text.trim());
    let mut candidates = Vec::new();
    push_unique_candidate(&mut candidates, stripped);

    if let Some(candidate) = balanced_json_slice(stripped, '[', ']') {
        push_unique_candidate(&mut candidates, candidate);
    }
    if let Some(candidate) = balanced_json_slice(stripped, '{', '}') {
        push_unique_candidate(&mut candidates, candidate);
    }

    candidates
}

fn strip_json_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let rest = rest
        .strip_prefix("json")
        .or_else(|| rest.strip_prefix("JSON"))
        .unwrap_or(rest)
        .trim_start_matches(|ch: char| ch.is_whitespace());
    rest.rsplit_once("```")
        .map(|(body, _)| body.trim())
        .unwrap_or(trimmed)
}

fn balanced_json_slice(text: &str, open: char, close: char) -> Option<&str> {
    let start = text.find(open)?;
    let end = text.rfind(close)?;
    (end >= start).then(|| text[start..=end].trim())
}

fn push_unique_candidate(candidates: &mut Vec<String>, candidate: &str) {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return;
    }
    if !candidates.iter().any(|existing| existing == candidate) {
        candidates.push(candidate.to_string());
    }
}

fn vibevoice_segments_from_value(value: &serde_json::Value) -> Vec<VibeVoiceAsrSegment> {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(vibevoice_segment_from_value)
            .collect(),
        serde_json::Value::Object(map) => {
            if let Some(segment) = vibevoice_segment_from_map(map) {
                return vec![segment];
            }
            for key in [
                "segments",
                "transcription",
                "transcript",
                "results",
                "utterances",
            ] {
                if let Some(segments) = get_value_case_insensitive(map, key)
                    .map(vibevoice_segments_from_value)
                    .filter(|segments| !segments.is_empty())
                {
                    return segments;
                }
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn vibevoice_segment_from_value(value: &serde_json::Value) -> Option<VibeVoiceAsrSegment> {
    let serde_json::Value::Object(map) = value else {
        return None;
    };
    vibevoice_segment_from_map(map)
}

fn vibevoice_segment_from_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Option<VibeVoiceAsrSegment> {
    let content = ["Content", "content", "Text", "text", "transcript"]
        .iter()
        .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_string))
        .unwrap_or_default();
    let content = content.trim().to_string();
    if content.is_empty() {
        return None;
    }

    Some(VibeVoiceAsrSegment {
        start_time: ["Start time", "start_time", "start", "begin"]
            .iter()
            .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_seconds)),
        end_time: ["End time", "end_time", "end", "stop"]
            .iter()
            .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_seconds)),
        speaker_id: ["Speaker ID", "speaker_id", "speaker", "speaker_id"]
            .iter()
            .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_string))
            .map(|speaker| speaker.trim().to_string())
            .filter(|speaker| !speaker.is_empty()),
        content,
    })
}

fn get_value_case_insensitive<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    map.get(key).or_else(|| {
        map.iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
    })
}

fn value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn value_to_seconds(value: &serde_json::Value) -> Option<f32> {
    match value {
        serde_json::Value::Number(number) => number.as_f64().map(|value| value as f32),
        serde_json::Value::String(text) => parse_timestamp_seconds(text),
        _ => None,
    }
    .filter(|value| value.is_finite() && *value >= 0.0)
}

fn parse_timestamp_seconds(text: &str) -> Option<f32> {
    let trimmed = text
        .trim()
        .trim_end_matches(|ch: char| ch.eq_ignore_ascii_case(&'s'))
        .trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.contains(':') {
        let mut seconds = 0.0f32;
        for part in trimmed.split(':') {
            seconds = seconds * 60.0 + part.trim().parse::<f32>().ok()?;
        }
        return Some(seconds);
    }
    trimmed.parse::<f32>().ok()
}

fn join_segment_contents(segments: &[VibeVoiceAsrSegment]) -> String {
    segments
        .iter()
        .map(|segment| segment.content.trim())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn cleanup_transcript_text(raw: &str) -> String {
    raw.replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .trim()
        .to_string()
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn preprocess_asr_audio(
    audio: &[f32],
    sample_rate: u32,
    config: &VibeVoicePreprocessorConfig,
) -> Result<(Vec<f32>, VibeVoiceAsrPreprocessStats)> {
    let mut resampled = resample_linear(audio, sample_rate, config.target_sample_rate())?;
    for sample in &mut resampled {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
    let stats = if config.normalize_audio {
        normalize_asr_loudness(&mut resampled, config.target_db_fs, config.eps)
    } else {
        VibeVoiceAsrPreprocessStats {
            normalized: false,
            target_db_fs: config.target_db_fs,
            rms_before: audio_rms(&resampled),
            gain: 1.0,
            clipping_divisor: 1.0,
        }
    };
    Ok((resampled, stats))
}

fn normalize_asr_loudness(
    samples: &mut [f32],
    target_db_fs: f32,
    eps: f32,
) -> VibeVoiceAsrPreprocessStats {
    if samples.is_empty() {
        return VibeVoiceAsrPreprocessStats {
            normalized: true,
            target_db_fs,
            rms_before: 0.0,
            gain: 1.0,
            clipping_divisor: 1.0,
        };
    }
    let rms = audio_rms(samples);
    let gain = 10f32.powf(target_db_fs / 20.0) / (rms + eps.max(0.0));
    for sample in samples.iter_mut() {
        *sample *= gain;
    }

    let peak = samples.iter().fold(0.0f32, |peak, &s| peak.max(s.abs()));
    let clipping_divisor = if peak > 1.0 { peak + eps.max(0.0) } else { 1.0 };
    if clipping_divisor != 1.0 {
        for sample in samples.iter_mut() {
            *sample /= clipping_divisor;
        }
    }

    VibeVoiceAsrPreprocessStats {
        normalized: true,
        target_db_fs,
        rms_before: rms,
        gain,
        clipping_divisor,
    }
}

fn audio_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples
        .iter()
        .map(|&sample| (sample as f64) * (sample as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32
}

fn asr_placeholder_count(samples: usize, speech_tok_compress_ratio: usize) -> usize {
    let ratio = speech_tok_compress_ratio.max(1);
    samples.saturating_add(ratio - 1) / ratio
}

fn vibevoice_asr_max_audio_seconds_hint(device_kind: DeviceKind) -> f32 {
    let cuda_override = std::env::var(CUDA_MAX_AUDIO_SECONDS_ENV).ok();
    vibevoice_asr_max_audio_seconds_hint_for(device_kind, cuda_override.as_deref())
}

fn vibevoice_asr_max_audio_seconds_hint_for(
    device_kind: DeviceKind,
    cuda_override: Option<&str>,
) -> f32 {
    if !device_kind.is_cuda() {
        return DEFAULT_MAX_AUDIO_SECONDS;
    }

    cuda_override
        .and_then(parse_positive_finite_f32)
        .unwrap_or(DEFAULT_CUDA_MAX_AUDIO_SECONDS)
}

fn parse_positive_finite_f32(raw: &str) -> Option<f32> {
    raw.trim()
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn tokenizer_streaming_chunk_samples(sample_rate: u32, speech_tok_compress_ratio: usize) -> usize {
    let ratio = speech_tok_compress_ratio.max(1);
    let raw = sample_rate as usize * TOKENIZER_STREAMING_CHUNK_SECONDS;
    let aligned = raw / ratio * ratio;
    aligned.max(ratio)
}

fn tokenizer_chunk_ranges(total_samples: usize, chunk_samples: usize) -> Vec<(usize, usize)> {
    if total_samples == 0 {
        return Vec::new();
    }
    let chunk_samples = chunk_samples.max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < total_samples {
        let len = chunk_samples.min(total_samples - start);
        ranges.push((start, len));
        start = start.saturating_add(len);
    }
    ranges
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "Sample rates must be positive for VibeVoice-ASR resampling".to_string(),
        ));
    }
    if src_rate == dst_rate {
        return Ok(audio.to_vec());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(audio[left] * (1.0 - frac) + audio[right] * frac);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_linear_preserves_identity_rate() {
        let audio = vec![0.0, 0.5, -0.25];
        assert_eq!(resample_linear(&audio, 24_000, 24_000).unwrap(), audio);
    }

    #[test]
    fn asr_placeholder_count_ceil_divides_samples_by_compress_ratio() {
        assert_eq!(asr_placeholder_count(0, 3_200), 0);
        assert_eq!(asr_placeholder_count(1, 3_200), 1);
        assert_eq!(asr_placeholder_count(3_200, 3_200), 1);
        assert_eq!(asr_placeholder_count(3_201, 3_200), 2);
        assert_eq!(asr_placeholder_count(9_599, 3_200), 3);
        assert_eq!(asr_placeholder_count(9_600, 3_200), 3);
    }

    #[test]
    fn max_audio_seconds_hint_preserves_cpu_and_metal_window() {
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cpu, Some("30")),
            DEFAULT_MAX_AUDIO_SECONDS
        );
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Metal, Some("30")),
            DEFAULT_MAX_AUDIO_SECONDS
        );
    }

    #[test]
    fn max_audio_seconds_hint_limits_cuda_window_by_default() {
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cuda, None),
            DEFAULT_CUDA_MAX_AUDIO_SECONDS
        );
    }

    #[test]
    fn max_audio_seconds_hint_accepts_positive_cuda_override() {
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cuda, Some("45.5")),
            45.5
        );
    }

    #[test]
    fn max_audio_seconds_hint_rejects_invalid_cuda_override() {
        for raw in ["", "0", "-1", "nan", "inf", "not-a-number"] {
            assert_eq!(
                vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cuda, Some(raw)),
                DEFAULT_CUDA_MAX_AUDIO_SECONDS
            );
        }
    }

    #[test]
    fn tokenizer_streaming_chunk_size_uses_aligned_sixty_second_windows() {
        assert_eq!(tokenizer_streaming_chunk_samples(24_000, 3_200), 1_440_000);
        assert_eq!(tokenizer_streaming_chunk_samples(1_000, 3_200), 57_600);
        assert_eq!(tokenizer_streaming_chunk_samples(10, 3_200), 3_200);
    }

    #[test]
    fn tokenizer_chunk_ranges_cover_audio_without_overlap() {
        let ranges = tokenizer_chunk_ranges(10_000, 3_200);

        assert_eq!(
            ranges,
            vec![(0, 3_200), (3_200, 3_200), (6_400, 3_200), (9_600, 400)]
        );
    }

    #[test]
    fn normalize_asr_loudness_matches_reference_dbfs_formula() {
        let mut audio = vec![0.5, -0.5, 0.5, -0.5];
        let stats = normalize_asr_loudness(&mut audio, -20.0, 1e-6);

        let rms = audio_rms(&audio);
        assert!((rms - 0.1).abs() < 1e-5);
        assert!(stats.normalized);
        assert!((stats.rms_before - 0.5).abs() < 1e-6);
        assert!((stats.gain - 0.2).abs() < 1e-5);
        assert_eq!(stats.clipping_divisor, 1.0);
    }

    #[test]
    fn normalize_asr_loudness_avoids_clipping() {
        let mut audio = vec![1.0, -1.0];
        let stats = normalize_asr_loudness(&mut audio, 6.0, 1e-6);

        assert!(stats.clipping_divisor > 1.0);
        assert!(audio.iter().all(|sample| sample.abs() <= 1.0));
    }

    #[test]
    fn preprocess_asr_audio_sanitizes_and_resamples() {
        let config = VibeVoicePreprocessorConfig {
            sampling_rate: 4,
            speech_tok_compress_ratio: 2,
            normalize_audio: false,
            target_db_fs: -25.0,
            eps: 1e-6,
        };
        let (audio, stats) =
            preprocess_asr_audio(&[0.0, f32::NAN, 1.0], 2, &config).expect("preprocess");

        assert_eq!(audio.len(), 6);
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!(!stats.normalized);
    }

    #[test]
    fn stop_sequences_are_trimmed_and_deduplicated() {
        let sequences = vec![" END ".to_string(), "".to_string(), "END".to_string()];
        assert_eq!(sanitize_stop_sequences(&sequences), vec!["END".to_string()]);
        assert_eq!(
            truncate_at_stop_sequence("hello END ignored", &sanitize_stop_sequences(&sequences)),
            ("hello ".to_string(), Some("END".to_string()))
        );
    }

    #[test]
    fn stop_token_ids_merge_built_ins_and_request_tokens() {
        assert_eq!(collect_stop_token_ids(&[1, 2], &[2, 3, 1]), vec![1, 2, 3]);
    }

    #[test]
    fn argmax_last_logits_preserves_host_fallback_selection() {
        let device = candle_core::Device::Cpu;
        let logits =
            Tensor::from_vec(vec![0.0f32, 3.0, 1.0, 2.0, -1.0, 5.0], (1, 2, 3), &device).unwrap();

        assert_eq!(argmax_last_logits(&logits, false).unwrap(), 2);
    }

    #[test]
    fn argmax_last_logits_can_select_on_device() {
        let device = candle_core::Device::Cpu;
        let logits =
            Tensor::from_vec(vec![0.0f32, 3.0, 1.0, 2.0, -1.0, 5.0], (1, 2, 3), &device).unwrap();

        assert_eq!(argmax_last_logits(&logits, true).unwrap(), 2);
    }

    #[test]
    fn parses_vibevoice_json_segments_into_plain_text() {
        let parsed = parse_vibevoice_asr_output(
            r#"[{"Start time": 0.0, "End time": "1.25", "Speaker ID": "Speaker 0", "Content": "Hello"}, {"Start time": "00:00:01.25", "End time": 2.0, "Speaker ID": 1, "Content": "world."}]"#,
        );

        assert_eq!(parsed.format, "segments");
        assert_eq!(parsed.text, "Hello world.");
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].speaker_id.as_deref(), Some("Speaker 0"));
        assert_eq!(parsed.segments[1].speaker_id.as_deref(), Some("1"));
        assert!((parsed.segments[1].start_time.unwrap() - 1.25).abs() < 1e-6);
    }

    #[test]
    fn parses_vibevoice_segments_inside_code_fence_and_wrapper() {
        let parsed = parse_vibevoice_asr_output(
            "```json\n{\"segments\":[{\"start\":0,\"end\":1,\"speaker\":\"A\",\"text\":\"Hi there\"}]}\n```",
        );

        assert_eq!(parsed.format, "segments");
        assert_eq!(parsed.text, "Hi there");
        assert_eq!(parsed.segments[0].end_time, Some(1.0));
    }

    #[test]
    fn parse_vibevoice_output_falls_back_to_cleaned_text() {
        let parsed = parse_vibevoice_asr_output("  plain text <|im_end|> ");

        assert_eq!(parsed.format, "text");
        assert_eq!(parsed.text, "plain text");
        assert!(parsed.segments.is_empty());
    }

    #[test]
    fn replace_range_preserves_prompt_length() {
        let device = candle_core::Device::Cpu;
        let embeds = Tensor::zeros((1, 5, 3), DType::F32, &device).unwrap();
        let features = Tensor::ones((1, 2, 3), DType::F32, &device).unwrap();
        let replaced = replace_range_with_features(&embeds, 2..4, &features).unwrap();
        assert_eq!(replaced.dims(), &[1, 5, 3]);
        assert_eq!(
            replaced.i((0, 2, ..)).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 1.0, 1.0]
        );
    }
}
