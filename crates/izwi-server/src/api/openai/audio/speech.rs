//! OpenAI-compatible speech synthesis endpoints.

use axum::{
    body::Body,
    extract::{Extension, State},
    http::{header, StatusCode},
    response::Response,
    Json,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::info;

use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::audio::AudioFormat;
use izwi_core::{
    parse_tts_model_variant, AudioChunk, GenerationConfig, GenerationRequest, ModelVariant,
};

const DEFAULT_STREAM_EVENT_QUEUE_CAPACITY: usize = 32;

/// OpenAI-compatible speech synthesis request.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// OpenAI field name for TTS model.
    pub model: String,
    /// OpenAI field name for text input.
    pub input: String,
    /// OpenAI-style voice selection.
    #[serde(default)]
    pub voice: Option<String>,
    /// OpenAI response format (`wav` and `pcm` currently supported by local runtime).
    #[serde(default)]
    pub response_format: Option<String>,
    /// OpenAI speed.
    #[serde(default)]
    pub speed: Option<f32>,
    /// Optional language hint (e.g. "Auto", "English", "Chinese").
    #[serde(default)]
    pub language: Option<String>,
    /// Optional sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional max token budget.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Alias for max output tokens in newer APIs.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    /// Optional top-k sampling for model-specific runtimes.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// If true, stream chunked audio from same endpoint.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Optional voice design prompt.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Optional reference audio (base64) for voice cloning.
    #[serde(default)]
    pub reference_audio: Option<String>,
    /// Optional reference transcript for cloning.
    #[serde(default)]
    pub reference_text: Option<String>,
}

#[derive(Debug, Serialize)]
struct SpeechStreamEvent {
    event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sequence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_final: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_generated: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_time_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rtf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn stream_event_queue_capacity() -> usize {
    std::env::var("IZWI_AUDIO_STREAM_EVENT_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_STREAM_EVENT_QUEUE_CAPACITY)
}

async fn send_stream_event(
    event_tx: &mpsc::Sender<String>,
    event: SpeechStreamEvent,
) -> Result<(), ()> {
    let payload = serde_json::to_string(&event).unwrap_or_default();
    event_tx.send(payload).await.map_err(|_| ())
}

pub async fn speech(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<SpeechRequest>,
) -> Result<Response<Body>, ApiError> {
    info!("OpenAI speech request: {} chars", req.input.len());

    let variant = parse_tts_model_variant(&req.model)
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {}", err)))?;
    state.runtime.load_model(variant).await?;

    if req.stream.unwrap_or(false) {
        return stream_speech(state, req, ctx.correlation_id).await;
    }

    let _permit = state.acquire_permit().await;

    let timeout = Duration::from_secs(resolve_speech_timeout_secs(
        state.request_timeout_secs,
        variant,
        &req,
    ));
    let format = parse_response_format(req.response_format.as_deref().unwrap_or("wav"))?;

    let result = tokio::time::timeout(timeout, async {
        let gen_request = build_generation_request(&req, ctx.correlation_id, false);
        state.runtime.generate(gen_request).await
    })
    .await
    .map_err(|_| ApiError::internal("Request timeout"))??;

    let encoder = state.runtime.audio_encoder().await;
    let samples = result.samples.clone();
    let audio_bytes = tokio::task::spawn_blocking(move || encoder.encode(&samples, format))
        .await
        .map_err(|e| ApiError::internal(format!("Audio encoding failed: {}", e)))??;

    let content_type = izwi_core::audio::AudioEncoder::content_type(format);
    let duration_secs = result.duration_secs();
    let generation_time_ms = result.total_time_ms;
    let rtf = result.rtf();
    let tokens_generated = result.total_tokens;

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header("X-Generation-Time-Ms", format!("{:.1}", generation_time_ms))
        .header("X-Audio-Duration-Secs", format!("{:.2}", duration_secs))
        .header("X-RTF", format!("{:.3}", rtf))
        .header("X-Tokens-Generated", tokens_generated.to_string())
        .header(
            "Access-Control-Expose-Headers",
            "X-Generation-Time-Ms, X-Audio-Duration-Secs, X-RTF, X-Tokens-Generated",
        )
        .body(Body::from(audio_bytes))
        .unwrap())
}

fn resolve_speech_timeout_secs(
    default_timeout_secs: u64,
    variant: ModelVariant,
    req: &SpeechRequest,
) -> u64 {
    // Keep global timeout behavior for non-Qwen TTS families (e.g. LFM2).
    let Some(model_max_frames) = variant.tts_max_output_frames_hint() else {
        return default_timeout_secs.max(1);
    };
    let Some(frame_rate_hz) = variant.tts_output_frame_rate_hz_hint() else {
        return default_timeout_secs.max(1);
    };

    // `0` means auto for TTS; treat that as native model max.
    let requested_frames = req.max_output_tokens.or(req.max_tokens);
    let effective_frames = match requested_frames {
        Some(0) | None => model_max_frames,
        Some(value) => value.clamp(1, model_max_frames),
    };

    // Estimate output duration from codec frame budget and requested speed.
    let speed = req.speed.unwrap_or(1.0).clamp(0.25, 4.0) as f64;
    let estimated_audio_secs = ((effective_frames as f64) / (frame_rate_hz as f64)) / speed;

    // Conservative real-time-factor multiplier for non-stream long-form synthesis.
    let timeout_rtf = std::env::var("IZWI_TTS_TIMEOUT_RTF")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 1.0)
        .unwrap_or(8.0);
    let timeout_padding_secs = std::env::var("IZWI_TTS_TIMEOUT_PADDING_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30);
    let timeout_max_secs = std::env::var("IZWI_TTS_TIMEOUT_MAX_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(6 * 60 * 60);

    let adaptive_secs = (estimated_audio_secs * timeout_rtf).ceil() as u64;
    let suggested_secs = adaptive_secs
        .saturating_add(timeout_padding_secs)
        .min(timeout_max_secs.max(1));

    default_timeout_secs.max(suggested_secs).max(1)
}

async fn stream_speech(
    state: AppState,
    req: SpeechRequest,
    correlation_id: String,
) -> Result<Response<Body>, ApiError> {
    let format = parse_response_format(req.response_format.as_deref().unwrap_or("pcm"))?;
    let gen_request = build_generation_request(&req, correlation_id, true);
    let stream_request_id = gen_request.id.clone();
    let stream_audio_format = stream_audio_format_label(format);
    let (event_tx, mut event_rx) = mpsc::channel::<String>(stream_event_queue_capacity());

    let engine = state.runtime.clone();
    let semaphore = state.request_semaphore.clone();
    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let error_event = SpeechStreamEvent {
                    event: "error",
                    request_id: Some(stream_request_id.clone()),
                    sequence: None,
                    audio_base64: None,
                    sample_count: None,
                    is_final: None,
                    sample_rate: None,
                    audio_format: None,
                    tokens_generated: None,
                    generation_time_ms: None,
                    audio_duration_secs: None,
                    rtf: None,
                    error: Some("Server is shutting down".to_string()),
                };
                let _ = send_stream_event(&event_tx, error_event).await;

                let done_event = SpeechStreamEvent {
                    event: "done",
                    request_id: Some(stream_request_id),
                    sequence: None,
                    audio_base64: None,
                    sample_count: None,
                    is_final: None,
                    sample_rate: None,
                    audio_format: None,
                    tokens_generated: None,
                    generation_time_ms: None,
                    audio_duration_secs: None,
                    rtf: None,
                    error: None,
                };
                let _ = send_stream_event(&event_tx, done_event).await;
                return;
            }
        };

        let sample_rate = engine.sample_rate().await;
        let start_event = SpeechStreamEvent {
            event: "start",
            request_id: Some(stream_request_id.clone()),
            sequence: None,
            audio_base64: None,
            sample_count: None,
            is_final: None,
            sample_rate: Some(sample_rate),
            audio_format: Some(stream_audio_format),
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            error: None,
        };
        if send_stream_event(&event_tx, start_event).await.is_err() {
            return;
        }

        let (chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(32);
        let generation_engine = engine.clone();
        let generation_task = tokio::spawn(async move {
            generation_engine
                .generate_streaming(gen_request, chunk_tx)
                .await
        });

        let mut total_samples = 0usize;
        let stream_started = Instant::now();
        let encoder = izwi_core::audio::AudioEncoder::new(sample_rate, 1);
        let mut client_closed = false;
        let mut stream_failed = false;
        while let Some(chunk) = chunk_rx.recv().await {
            if chunk.samples.is_empty() {
                continue;
            }

            total_samples += chunk.samples.len();
            let bytes = match encoder.encode(&chunk.samples, format) {
                Ok(bytes) => bytes,
                Err(err) => {
                    let error_event = SpeechStreamEvent {
                        event: "error",
                        request_id: Some(stream_request_id.clone()),
                        sequence: None,
                        audio_base64: None,
                        sample_count: None,
                        is_final: None,
                        sample_rate: None,
                        audio_format: None,
                        tokens_generated: None,
                        generation_time_ms: None,
                        audio_duration_secs: None,
                        rtf: None,
                        error: Some(format!("Failed to encode audio chunk: {}", err)),
                    };
                    let _ = send_stream_event(&event_tx, error_event).await;
                    stream_failed = true;
                    break;
                }
            };

            let chunk_event = SpeechStreamEvent {
                event: "chunk",
                request_id: Some(chunk.request_id.clone()),
                sequence: Some(chunk.sequence),
                audio_base64: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
                sample_count: Some(chunk.samples.len()),
                is_final: Some(chunk.is_final),
                sample_rate: None,
                audio_format: None,
                tokens_generated: None,
                generation_time_ms: None,
                audio_duration_secs: None,
                rtf: None,
                error: None,
            };
            if send_stream_event(&event_tx, chunk_event).await.is_err() {
                client_closed = true;
                break;
            }
        }

        drop(chunk_rx);
        if client_closed {
            let _ = generation_task.await;
            return;
        }

        if stream_failed {
            let _ = generation_task.await;
            let done_event = SpeechStreamEvent {
                event: "done",
                request_id: Some(stream_request_id),
                sequence: None,
                audio_base64: None,
                sample_count: None,
                is_final: None,
                sample_rate: None,
                audio_format: None,
                tokens_generated: None,
                generation_time_ms: None,
                audio_duration_secs: None,
                rtf: None,
                error: None,
            };
            let _ = send_stream_event(&event_tx, done_event).await;
            return;
        }

        let generation_outcome = generation_task.await;
        match generation_outcome {
            Ok(Ok(())) => {
                let generation_time_ms = stream_started.elapsed().as_secs_f32() * 1000.0;
                let audio_duration_secs = total_samples as f32 / sample_rate as f32;
                let tokens_generated = total_samples / 256;
                let rtf = if audio_duration_secs > 0.0 {
                    (generation_time_ms / 1000.0) / audio_duration_secs
                } else {
                    0.0
                };

                let final_event = SpeechStreamEvent {
                    event: "final",
                    request_id: Some(stream_request_id.clone()),
                    sequence: None,
                    audio_base64: None,
                    sample_count: None,
                    is_final: None,
                    sample_rate: None,
                    audio_format: None,
                    tokens_generated: Some(tokens_generated),
                    generation_time_ms: Some(generation_time_ms),
                    audio_duration_secs: Some(audio_duration_secs),
                    rtf: Some(rtf),
                    error: None,
                };
                let _ = send_stream_event(&event_tx, final_event).await;
            }
            Ok(Err(err)) => {
                let error_event = SpeechStreamEvent {
                    event: "error",
                    request_id: Some(stream_request_id.clone()),
                    sequence: None,
                    audio_base64: None,
                    sample_count: None,
                    is_final: None,
                    sample_rate: None,
                    audio_format: None,
                    tokens_generated: None,
                    generation_time_ms: None,
                    audio_duration_secs: None,
                    rtf: None,
                    error: Some(err.to_string()),
                };
                let _ = send_stream_event(&event_tx, error_event).await;
            }
            Err(err) => {
                let error_event = SpeechStreamEvent {
                    event: "error",
                    request_id: Some(stream_request_id.clone()),
                    sequence: None,
                    audio_base64: None,
                    sample_count: None,
                    is_final: None,
                    sample_rate: None,
                    audio_format: None,
                    tokens_generated: None,
                    generation_time_ms: None,
                    audio_duration_secs: None,
                    rtf: None,
                    error: Some(format!("Streaming task failed: {}", err)),
                };
                let _ = send_stream_event(&event_tx, error_event).await;
            }
        }

        let done_event = SpeechStreamEvent {
            event: "done",
            request_id: Some(stream_request_id),
            sequence: None,
            audio_base64: None,
            sample_count: None,
            is_final: None,
            sample_rate: None,
            audio_format: None,
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            error: None,
        };
        let _ = send_stream_event(&event_tx, done_event).await;
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

fn build_generation_request(
    req: &SpeechRequest,
    correlation_id: String,
    streaming: bool,
) -> GenerationRequest {
    let mut gen_config = GenerationConfig {
        streaming,
        ..GenerationConfig::default()
    };
    if let Some(temp) = req.temperature {
        gen_config.options.temperature = temp;
    }
    if let Some(speed) = req.speed {
        gen_config.options.speed = speed;
    }
    if let Some(max_tokens) = req.max_output_tokens.or(req.max_tokens) {
        gen_config.options.max_tokens = max_tokens;
    }
    if let Some(top_k) = req.top_k {
        gen_config.options.top_k = top_k;
    }
    gen_config.options.speaker = req.voice.clone();

    GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        correlation_id: Some(correlation_id),
        text: req.input.clone(),
        config: gen_config,
        language: req.language.clone(),
        reference_audio: req.reference_audio.clone(),
        reference_text: req.reference_text.clone(),
        voice_description: req.instructions.clone(),
    }
}

fn parse_response_format(format: &str) -> Result<AudioFormat, ApiError> {
    match format.to_ascii_lowercase().as_str() {
        "wav" => Ok(AudioFormat::Wav),
        "pcm" | "pcm16" | "pcm_i16" | "raw_i16" => Ok(AudioFormat::RawI16),
        "raw_f32" | "pcm_f32" => Ok(AudioFormat::RawF32),
        // Accepted OpenAI names that are not yet supported by local encoder.
        "mp3" | "opus" | "aac" | "flac" => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. This runtime currently supports wav and pcm",
            format
        ))),
        unsupported => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Supported formats: wav, pcm",
            unsupported
        ))),
    }
}

fn stream_audio_format_label(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::RawF32 => "pcm_f32",
        AudioFormat::RawI16 => "pcm_i16",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_auto_timeout_expands_for_long_form() {
        let req = SpeechRequest {
            model: "Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
            input: "hello".to_string(),
            voice: Some("Aiden".to_string()),
            response_format: Some("wav".to_string()),
            speed: None,
            language: None,
            temperature: None,
            max_tokens: Some(0),
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            instructions: None,
            reference_audio: None,
            reference_text: None,
        };

        let timeout =
            resolve_speech_timeout_secs(300, ModelVariant::Qwen3Tts12Hz06BCustomVoice, &req);
        assert!(timeout > 300, "expected adaptive timeout > default");
    }

    #[test]
    fn explicit_small_frame_budget_keeps_timeout_near_default() {
        let req = SpeechRequest {
            model: "Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
            input: "hello".to_string(),
            voice: Some("Aiden".to_string()),
            response_format: Some("wav".to_string()),
            speed: Some(1.0),
            language: None,
            temperature: None,
            max_tokens: Some(256),
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            instructions: None,
            reference_audio: None,
            reference_text: None,
        };

        let timeout =
            resolve_speech_timeout_secs(300, ModelVariant::Qwen3Tts12Hz06BCustomVoice, &req);
        assert_eq!(timeout, 300);
    }

    #[test]
    fn non_qwen_tts_uses_default_timeout() {
        let req = SpeechRequest {
            model: "LFM2.5-Audio-1.5B".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            speed: None,
            language: None,
            temperature: None,
            max_tokens: Some(0),
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            instructions: None,
            reference_audio: None,
            reference_text: None,
        };

        let timeout = resolve_speech_timeout_secs(300, ModelVariant::Lfm25Audio15B, &req);
        assert_eq!(timeout, 300);
    }
}
