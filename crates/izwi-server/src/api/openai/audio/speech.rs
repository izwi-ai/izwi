//! OpenAI-compatible speech synthesis endpoints.

use axum::{
    Json,
    body::Body,
    extract::{Extension, State},
    http::{StatusCode, header},
    response::Response,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::info;

use crate::api::request_context::RequestContext;
use crate::api::saved_voices::resolve_saved_voice_reference;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::runtime_models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
use izwi_core::runtime_models::architectures::voxtral::tts::voxtral_tts_auto_max_frames_for_text;
use izwi_core::{
    AudioChunk, GenerationConfig, GenerationRequest, ModelVariant, parse_tts_model_variant,
};

const DEFAULT_STREAM_EVENT_QUEUE_CAPACITY: usize = 32;
const SPEECH_RESPONSE_EXPOSED_HEADERS: &str = "X-Generation-Time-Ms, X-Audio-Duration-Secs, X-RTF, X-Tokens-Generated, X-Audio-Sample-Rate, X-Requested-Response-Format, X-Actual-Response-Format, X-Response-Format-Fallback";

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
    /// OpenAI response format. WAV and raw PCM are native in the OSS runtime.
    #[serde(default)]
    pub response_format: Option<String>,
    /// Explicitly allow recognized compressed formats to return WAV bytes when
    /// no native compressed encoder is available.
    #[serde(default)]
    pub allow_format_fallback: Option<bool>,
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
    /// OpenAI-style stream transport hint. `sse` enables server-sent events.
    #[serde(default)]
    pub stream_format: Option<String>,
    /// Optional voice design prompt.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Optional reference audio (base64) for voice cloning.
    #[serde(default)]
    pub reference_audio: Option<String>,
    /// Optional reference transcript for cloning.
    #[serde(default)]
    pub reference_text: Option<String>,
    /// Optional saved voice identifier resolved server-side.
    #[serde(default)]
    pub saved_voice_id: Option<String>,
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
    let req = resolve_saved_voice_request(&state, normalize_speech_request(req)).await?;
    info!("OpenAI speech request: {} chars", req.input.len());

    let variant = parse_tts_model_variant(&req.model)
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {}", err)))?;
    validate_speech_voice_contract(&req, variant)?;
    let streaming = resolve_streaming_mode(&req)?;
    let resolved_format = parse_response_format(
        req.response_format.as_deref().unwrap_or("wav"),
        req.allow_format_fallback.unwrap_or(false),
    )?;

    state.runtime.load_model(variant).await?;

    if streaming {
        return stream_speech(state, req, ctx.correlation_id, variant, resolved_format).await;
    }

    let _permit = state.acquire_permit().await;

    let timeout = Duration::from_secs(resolve_speech_timeout_secs(
        state.request_timeout_secs,
        variant,
        &req,
    ));
    let format = resolved_format.format;
    let actual_format = resolved_format.label;
    let format_fallback = resolved_format.fallback;

    let result = tokio::time::timeout(timeout, async {
        let gen_request = build_generation_request(&req, ctx.correlation_id, false, variant);
        state.runtime.generate(gen_request).await
    })
    .await
    .map_err(|_| ApiError::internal("Request timeout"))??;

    let sample_rate = result.sample_rate;
    let samples = result.samples.clone();
    let audio_bytes =
        tokio::task::spawn_blocking(move || encode_speech_samples(&samples, sample_rate, format))
            .await
            .map_err(|e| ApiError::internal(format!("Audio encoding failed: {}", e)))??;

    let content_type = AudioEncoder::content_type(format);
    let duration_secs = result.duration_secs();
    let generation_time_ms = result.total_time_ms;
    let rtf = result.rtf();
    let tokens_generated = result.total_tokens;

    let mut builder = Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header("X-Generation-Time-Ms", format!("{:.1}", generation_time_ms))
        .header("X-Audio-Duration-Secs", format!("{:.2}", duration_secs))
        .header("X-RTF", format!("{:.3}", rtf))
        .header("X-Tokens-Generated", tokens_generated.to_string())
        .header("X-Audio-Sample-Rate", sample_rate.to_string())
        .header(
            "Access-Control-Expose-Headers",
            SPEECH_RESPONSE_EXPOSED_HEADERS,
        )
        .header(
            "X-Requested-Response-Format",
            req.response_format
                .as_deref()
                .unwrap_or("wav")
                .to_ascii_lowercase(),
        )
        .header("X-Actual-Response-Format", actual_format);
    if let Some(fallback) = format_fallback {
        builder = builder
            .header("X-Response-Format-Fallback", fallback.as_str())
            .header(
                "Warning",
                format!(
                    "299 Izwi \"Requested response_format returned {actual_format}; fallback was explicitly enabled\""
                ),
            );
    }
    Ok(builder.body(Body::from(audio_bytes)).unwrap())
}

fn resolve_speech_timeout_secs(
    default_timeout_secs: u64,
    variant: ModelVariant,
    req: &SpeechRequest,
) -> u64 {
    // Keep global timeout behavior for non-Qwen TTS families.
    let Some(model_max_frames) = variant.tts_max_output_frames_hint() else {
        return default_timeout_secs.max(1);
    };
    let Some(frame_rate_hz) = variant.tts_output_frame_rate_hz_hint() else {
        return default_timeout_secs.max(1);
    };

    // `0`/omitted means auto for TTS. Voxtral uses a text-sized default
    // budget because every native frame is expensive and audio is emitted only
    // after the final codec pass; other TTS models keep the historical max.
    let requested_frames = req.max_output_tokens.or(req.max_tokens);
    let effective_frames = match requested_frames {
        Some(0) | None if variant == ModelVariant::Voxtral4BTts2603 => {
            voxtral_tts_auto_max_frames_for_text(&req.input)
        }
        Some(0) | None if variant == ModelVariant::VibeVoice15BTts => {
            vibevoice_tts_auto_max_frames_for_text(&req.input)
        }
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

fn normalize_speech_request(mut req: SpeechRequest) -> SpeechRequest {
    req.voice = normalize_optional_trimmed(req.voice);
    req.response_format = normalize_optional_trimmed(req.response_format);
    req.language = normalize_optional_trimmed(req.language);
    req.instructions = normalize_optional_trimmed(req.instructions);
    req.reference_audio = normalize_optional_trimmed(req.reference_audio);
    req.reference_text = normalize_optional_trimmed(req.reference_text);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    req.stream_format = normalize_optional_trimmed(req.stream_format);
    req
}

async fn resolve_saved_voice_request(
    state: &AppState,
    mut req: SpeechRequest,
) -> Result<SpeechRequest, ApiError> {
    let Some(saved_voice_id) = req.saved_voice_id.as_deref() else {
        return Ok(req);
    };

    let has_direct_reference_audio = req
        .reference_audio
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    let has_direct_reference_text = req
        .reference_text
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    if has_direct_reference_audio || has_direct_reference_text {
        return Err(ApiError::bad_request(
            "Use either `saved_voice_id` or direct `reference_audio`/`reference_text`, not both.",
        ));
    }

    let saved_voice = resolve_saved_voice_reference(state, saved_voice_id).await?;
    req.saved_voice_id = Some(saved_voice.voice_id);
    req.reference_audio = Some(saved_voice.reference_audio_base64);
    req.reference_text = Some(saved_voice.reference_text);
    Ok(req)
}

fn validate_speech_voice_contract(
    req: &SpeechRequest,
    variant: ModelVariant,
) -> Result<(), ApiError> {
    let has_direct_reference_audio = req
        .reference_audio
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    let has_direct_reference_text = req
        .reference_text
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    if has_direct_reference_audio != has_direct_reference_text {
        return Err(ApiError::bad_request(
            "Provide both `reference_audio` and `reference_text` together.",
        ));
    }

    if has_direct_reference_audio {
        let supports_reference_voice = variant
            .speech_capabilities()
            .map(|capabilities| capabilities.supports_reference_voice)
            .unwrap_or(false);
        if !supports_reference_voice {
            return Err(ApiError::bad_request(format!(
                "{variant} does not support reference or saved voices.",
            )));
        }
    }

    let has_voice_description = req
        .instructions
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    if has_voice_description {
        let supports_voice_description = variant
            .speech_capabilities()
            .map(|capabilities| capabilities.supports_voice_description)
            .unwrap_or(false);
        if !supports_voice_description {
            return Err(ApiError::bad_request(format!(
                "{variant} does not support voice direction prompts.",
            )));
        }
    }

    if variant == ModelVariant::VibeVoice15BTts && !has_direct_reference_audio {
        return Err(ApiError::bad_request(
            "VibeVoice-1.5B requires `saved_voice_id` or both `reference_audio` and `reference_text`.",
        ));
    }

    Ok(())
}

async fn stream_speech(
    state: AppState,
    req: SpeechRequest,
    correlation_id: String,
    variant: ModelVariant,
    resolved_format: ResolvedSpeechFormat,
) -> Result<Response<Body>, ApiError> {
    let format = resolved_format.format;
    let format_fallback = resolved_format.fallback;
    let gen_request = build_generation_request(&req, correlation_id, true, variant);
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
                    event: "audio.failed",
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
                return;
            }
        };

        let fallback_sample_rate = engine.sample_rate().await;
        let start_event = SpeechStreamEvent {
            event: "audio.started",
            request_id: Some(stream_request_id.clone()),
            sequence: None,
            audio_base64: None,
            sample_count: None,
            is_final: None,
            sample_rate: Some(fallback_sample_rate),
            audio_format: Some(stream_audio_format),
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            error: format_fallback
                .as_ref()
                .map(|fallback| format!("Requested format fallback: {fallback}")),
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
        let mut audio_duration_secs = 0.0f32;
        let mut last_sample_rate = fallback_sample_rate;
        let stream_started = Instant::now();
        let mut client_closed = false;
        let mut stream_failed = false;
        while let Some(chunk) = chunk_rx.recv().await {
            if chunk.samples.is_empty() {
                continue;
            }

            let chunk_sample_rate = chunk.sample_rate_or(fallback_sample_rate).max(1);
            total_samples += chunk.samples.len();
            audio_duration_secs += chunk.samples.len() as f32 / chunk_sample_rate as f32;
            last_sample_rate = chunk_sample_rate;
            let bytes = match encode_speech_samples(&chunk.samples, chunk_sample_rate, format) {
                Ok(bytes) => bytes,
                Err(err) => {
                    let error_event = SpeechStreamEvent {
                        event: "audio.failed",
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
                event: "audio.chunk",
                request_id: Some(chunk.request_id.clone()),
                sequence: Some(chunk.sequence),
                audio_base64: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
                sample_count: Some(chunk.samples.len()),
                is_final: Some(chunk.is_final),
                sample_rate: Some(chunk_sample_rate),
                audio_format: Some(stream_audio_format),
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
                let tokens_generated = total_samples / 256;
                let rtf = if audio_duration_secs > 0.0 {
                    (generation_time_ms / 1000.0) / audio_duration_secs
                } else {
                    0.0
                };

                let final_event = SpeechStreamEvent {
                    event: "audio.done",
                    request_id: Some(stream_request_id.clone()),
                    sequence: None,
                    audio_base64: None,
                    sample_count: None,
                    is_final: None,
                    sample_rate: Some(last_sample_rate),
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
                    event: "audio.failed",
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
                    event: "audio.failed",
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

        let _ = stream_request_id;
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
    variant: ModelVariant,
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
    } else if matches!(
        variant,
        ModelVariant::Voxtral4BTts2603 | ModelVariant::VibeVoice15BTts
    ) {
        gen_config.options.max_tokens = 0;
    }
    if let Some(top_k) = req.top_k {
        gen_config.options.top_k = top_k;
    }
    gen_config.options.speaker = req.voice.clone();

    GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        model_variant: Some(variant),
        correlation_id: Some(correlation_id),
        text: req.input.clone(),
        config: gen_config,
        language: req.language.clone(),
        reference_audio: req.reference_audio.clone(),
        reference_text: req.reference_text.clone(),
        voice_description: req.instructions.clone(),
    }
}

fn normalize_optional_trimmed(raw: Option<String>) -> Option<String> {
    let trimmed = raw.unwrap_or_default().trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn has_non_empty_text(raw: &str) -> bool {
    !raw.trim().is_empty()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedSpeechFormat {
    format: AudioFormat,
    label: &'static str,
    fallback: Option<String>,
}

fn parse_response_format(
    format: &str,
    allow_fallback: bool,
) -> Result<ResolvedSpeechFormat, ApiError> {
    let normalized = format.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "wav" => Ok(ResolvedSpeechFormat {
            format: AudioFormat::Wav,
            label: "wav",
            fallback: None,
        }),
        "pcm" | "pcm16" | "pcm_i16" | "raw_i16" => Ok(ResolvedSpeechFormat {
            format: AudioFormat::RawI16,
            label: "pcm_i16",
            fallback: None,
        }),
        "raw_f32" | "pcm_f32" => Ok(ResolvedSpeechFormat {
            format: AudioFormat::RawF32,
            label: "pcm_f32",
            fallback: None,
        }),
        "mp3" | "opus" | "ogg" | "aac" | "flac" if allow_fallback => Ok(ResolvedSpeechFormat {
            format: AudioFormat::Wav,
            label: "wav",
            fallback: Some(format!("{normalized}->wav")),
        }),
        "mp3" | "opus" | "ogg" | "aac" | "flac" => Err(ApiError::bad_request(format!(
            "response_format `{normalized}` is recognized, but compressed audio encoding is not available in this OSS build. Use `wav`, `pcm_i16`, or `pcm_f32`, or set `allow_format_fallback: true` to receive WAV bytes."
        ))),
        unsupported => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Native formats: wav, pcm_i16, pcm_f32. Recognized compressed formats require `allow_format_fallback: true`: mp3, opus, ogg, aac, flac",
            unsupported
        ))),
    }
}

fn resolve_streaming_mode(req: &SpeechRequest) -> Result<bool, ApiError> {
    let stream_bool = req.stream.unwrap_or(false);
    let stream_format = req
        .stream_format
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase());
    match stream_format.as_deref() {
        None | Some("") => Ok(stream_bool),
        Some("sse") => Ok(true),
        Some(other) => Err(ApiError::bad_request(format!(
            "Unsupported stream_format: {}. Supported value: sse",
            other
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

fn encode_speech_samples(
    samples: &[f32],
    sample_rate: u32,
    format: AudioFormat,
) -> izwi_core::Result<Vec<u8>> {
    AudioEncoder::new(sample_rate, 1).encode(samples, format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn qwen_auto_timeout_expands_for_long_form() {
        let req = SpeechRequest {
            model: "Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
            input: "hello".to_string(),
            voice: Some("Aiden".to_string()),
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: Some(0),
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            saved_voice_id: None,
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
            allow_format_fallback: None,
            speed: Some(1.0),
            language: None,
            temperature: None,
            max_tokens: Some(256),
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            saved_voice_id: None,
        };

        let timeout =
            resolve_speech_timeout_secs(300, ModelVariant::Qwen3Tts12Hz06BCustomVoice, &req);
        assert_eq!(timeout, 300);
    }

    #[test]
    fn non_qwen_tts_uses_default_timeout() {
        let req = SpeechRequest {
            model: "Kokoro-82M".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: Some(0),
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            saved_voice_id: None,
        };

        let timeout = resolve_speech_timeout_secs(300, ModelVariant::Kokoro82M, &req);
        assert_eq!(timeout, 300);
    }

    #[test]
    fn voxtral_tts_omitted_max_tokens_uses_text_sized_auto_budget() {
        let req = SpeechRequest {
            model: "Voxtral-4B-TTS-2603".to_string(),
            input: "The costs split cleanly into three buckets".to_string(),
            voice: Some("casual_male".to_string()),
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            saved_voice_id: None,
        };

        let timeout = resolve_speech_timeout_secs(1, ModelVariant::Voxtral4BTts2603, &req);
        assert_eq!(timeout, 72);

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::Voxtral4BTts2603,
        );
        assert_eq!(generation.config.options.max_tokens, 0);
    }

    #[test]
    fn vibevoice_tts_omitted_max_tokens_uses_text_sized_auto_budget() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "The costs split cleanly into three buckets".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: Some("UklGRg==".to_string()),
            reference_text: Some("hello".to_string()),
            saved_voice_id: None,
        };

        let timeout = resolve_speech_timeout_secs(1, ModelVariant::VibeVoice15BTts, &req);
        assert_eq!(timeout, 59);

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::VibeVoice15BTts,
        );
        assert_eq!(generation.config.options.max_tokens, 0);
    }

    #[test]
    fn vibevoice_speech_requests_require_reference_voice() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            saved_voice_id: None,
        };

        let err = validate_speech_voice_contract(&req, ModelVariant::VibeVoice15BTts)
            .expect_err("missing reference should fail");
        assert!(err.message.contains("requires `saved_voice_id`"));
    }

    #[test]
    fn direct_reference_audio_and_text_must_be_paired() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: Some("UklGRg==".to_string()),
            reference_text: None,
            saved_voice_id: None,
        };

        let err = validate_speech_voice_contract(&req, ModelVariant::VibeVoice15BTts)
            .expect_err("partial reference should fail");
        assert!(err.message.contains("Provide both `reference_audio`"));
    }

    #[test]
    fn generation_request_preserves_vibevoice_reference_fields_and_voice_label() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "hello".to_string(),
            voice: Some("Speaker 1".to_string()),
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: Some("UklGRg==".to_string()),
            reference_text: Some("reference words".to_string()),
            saved_voice_id: None,
        };

        validate_speech_voice_contract(&req, ModelVariant::VibeVoice15BTts)
            .expect("valid reference");
        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::VibeVoice15BTts,
        );

        assert_eq!(
            generation.config.options.speaker.as_deref(),
            Some("Speaker 1")
        );
        assert_eq!(generation.reference_audio.as_deref(), Some("UklGRg=="));
        assert_eq!(
            generation.reference_text.as_deref(),
            Some("reference words")
        );
    }

    #[test]
    fn parse_response_format_rejects_mp3_without_explicit_fallback() {
        let error =
            parse_response_format("mp3", false).expect_err("mp3 should require fallback opt-in");

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.message.contains("allow_format_fallback"));
    }

    #[test]
    fn parse_response_format_maps_mp3_to_wav_with_explicit_fallback() {
        let resolved = parse_response_format("mp3", true).expect("format should parse");

        assert_eq!(resolved.format, AudioFormat::Wav);
        assert_eq!(resolved.label, "wav");
        assert_eq!(resolved.fallback.as_deref(), Some("mp3->wav"));
    }

    #[test]
    fn resolve_streaming_mode_honors_stream_format_sse() {
        let req = SpeechRequest {
            model: "Kokoro-82M".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: Some("sse".to_string()),
            instructions: None,
            reference_audio: None,
            reference_text: None,
            saved_voice_id: None,
        };
        assert!(resolve_streaming_mode(&req).expect("streaming mode"));
    }

    #[test]
    fn speech_wav_encoding_uses_generated_sample_rate() {
        let bytes = encode_speech_samples(&[0.0, 0.25, -0.25], 16_000, AudioFormat::Wav)
            .expect("encode wav");
        let reader = hound::WavReader::new(Cursor::new(bytes)).expect("decode wav");

        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().channels, 1);
    }

    #[test]
    fn speech_stream_chunk_event_includes_audio_rate_and_format() {
        let event = SpeechStreamEvent {
            event: "audio.chunk",
            request_id: Some("req".to_string()),
            sequence: Some(3),
            audio_base64: Some("AA==".to_string()),
            sample_count: Some(1),
            is_final: Some(false),
            sample_rate: Some(48_000),
            audio_format: Some("pcm_i16"),
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            error: None,
        };

        let value = serde_json::to_value(event).expect("serialize event");

        assert_eq!(value["sample_rate"], 48_000);
        assert_eq!(value["audio_format"], "pcm_i16");
    }
}
