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
use crate::api::saved_voices::resolve_saved_voice_reference;
use crate::api::tts_policy::resolve_tts_output_frames;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::{
    parse_tts_model_variant, AudioChunk, GenerationConfig, GenerationRequest, ModelVariant,
    WorkloadClass,
};

const DEFAULT_STREAM_EVENT_QUEUE_CAPACITY: usize = 32;
const SPEECH_RESPONSE_EXPOSED_HEADERS: &str = "X-Generation-Time-Ms, X-Audio-Duration-Secs, X-RTF, X-Tokens-Generated, X-Audio-Sample-Rate, X-Izwi-Tts-Diagnostics, X-Requested-Response-Format, X-Actual-Response-Format, X-Response-Format-Fallback";

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
    timing: Option<serde_json::Value>,
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

    let tenant_key = ctx.tenant_key();
    if streaming {
        return stream_speech(
            state,
            req,
            ctx.correlation_id,
            tenant_key,
            variant,
            resolved_format,
        )
        .await;
    }

    if variant == ModelVariant::FishAudioS2Pro {
        return fish_file_speech(state, req, ctx.correlation_id, tenant_key, resolved_format).await;
    }

    let permit = state
        .acquire_workload_permit(WorkloadClass::Interactive)
        .await;
    state.runtime.load_model(variant).await?;

    let timeout = Duration::from_secs(resolve_speech_timeout_secs(
        state.request_timeout_secs,
        variant,
        &req,
    ));
    let format = resolved_format.format;
    let actual_format = resolved_format.label;
    let format_fallback = resolved_format.fallback;

    let result = tokio::time::timeout(timeout, async {
        let mut runtime_context = permit.runtime_context();
        runtime_context.tenant_key = tenant_key;
        let gen_request = build_generation_request(&req, ctx.correlation_id, false, variant)
            .with_runtime_context(runtime_context);
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
    if let Some(diagnostics) = result.diagnostics.as_ref() {
        builder = builder.header("X-Izwi-Tts-Diagnostics", diagnostics.to_string());
    }
    Ok(builder.body(Body::from(audio_bytes)).unwrap())
}

/// Synchronous requests retain cancellation with their HTTP future. PCM is
/// spooled incrementally; the response owns the finalized file until EOF/drop.
async fn fish_file_speech(
    state: AppState,
    req: SpeechRequest,
    correlation_id: String,
    tenant_key: Option<[u8; 32]>,
    resolved_format: ResolvedSpeechFormat,
) -> Result<Response<Body>, ApiError> {
    use crate::speech_history_store::SpeechWavSpool;
    use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
    let variant = ModelVariant::FishAudioS2Pro;
    let mut request = build_generation_request(&req, correlation_id, true, variant);
    request.runtime_context.tenant_key = tenant_key;
    // Validate admission before model load or output allocation.
    crate::api::tts_long_form::SpeechTextPlan::fish(&req.input, request.config.options.max_tokens)?;
    state.runtime.load_model(variant).await?;
    let started = Instant::now();
    let (sender, mut receiver) = mpsc::channel::<AudioChunk>(2);
    let format = resolved_format.format;
    let collect = async {
        let raw_owner = crate::speech_spool::new_speech_tempfile()
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let mut raw = tokio::fs::File::from_std(
            raw_owner
                .reopen()
                .map_err(|e| ApiError::internal(e.to_string()))?,
        );
        let mut reservation = crate::speech_resource_budget::spool_budget()
            .reserve(0)
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let mut wav = None;
        let mut sample_rate = None;
        let mut samples = 0u64;
        let mut statistics = None;
        while let Some(chunk) = receiver.recv().await {
            accumulate_stream_statistics(&mut statistics, &chunk);
            if chunk.samples.is_empty() {
                continue;
            }
            let rate = chunk.sample_rate_or(44_100);
            if sample_rate.is_some_and(|previous| previous != rate) {
                return Err(ApiError::internal(
                    "Speech sample rate changed between segments",
                ));
            }
            sample_rate = Some(rate);
            samples = samples
                .checked_add(chunk.samples.len() as u64)
                .ok_or_else(|| ApiError::internal("Speech sample count overflow"))?;
            if format == AudioFormat::Wav {
                if wav.is_none() {
                    wav = Some(
                        SpeechWavSpool::new(rate, u32::MAX as usize - 44)
                            .map_err(|e| ApiError::internal(e.to_string()))?,
                    );
                }
                let pcm = encode_speech_samples(&chunk.samples, rate, AudioFormat::RawI16)?;
                wav.as_mut()
                    .unwrap()
                    .append_pcm(&pcm)
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?;
            } else {
                let pcm = encode_speech_samples(&chunk.samples, rate, format)?;
                reservation
                    .grow(pcm.len())
                    .map_err(|e| ApiError::internal(e.to_string()))?;
                raw.write_all(&pcm)
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?;
            }
        }
        let rate =
            sample_rate.ok_or_else(|| ApiError::internal("Speech generation produced no audio"))?;
        let artifact = match wav {
            Some(wav) => Some(
                wav.finish_file()
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?,
            ),
            None => None,
        };
        let (mut file, length) = if let Some(wav) = artifact.as_ref() {
            (
                wav.open()
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?,
                wav.len(),
            )
        } else {
            raw.flush()
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
            raw.seek(std::io::SeekFrom::Start(0))
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
            let length = raw
                .metadata()
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?
                .len();
            (raw, length)
        };
        let stream = async_stream::try_stream! {
            // Capture owners before polling: dropping even an unpolled body releases disk.
            let _owners = (artifact, raw_owner, reservation);
            let mut buffer = vec![0u8; 64 * 1024];
            loop {
                let count = file.read(&mut buffer).await?;
                if count == 0 { break; }
                yield bytes::Bytes::copy_from_slice(&buffer[..count]);
            }
        };
        let body = Body::from_stream(Box::pin(stream)
            as std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send>,
            >);
        Ok::<_, ApiError>((body, length, rate, samples, statistics))
    };
    let generation = async {
        crate::api::tts_long_form::generate_speech_plan_stream(
            &state,
            variant,
            request,
            sender,
            WorkloadClass::Interactive,
        )
        .await
        .map_err(ApiError::from)
    };
    let timeout = Duration::from_secs(state.request_timeout_secs.max(1));
    let (_, (body, length, rate, samples, statistics)) =
        tokio::time::timeout(timeout, async { tokio::try_join!(generation, collect) })
            .await
            .map_err(|_| {
                ApiError::internal(
            "Synchronous speech deadline exceeded; use durable speech history for long jobs",
        )
            })??;
    let duration = samples as f64 / f64::from(rate);
    let elapsed = started.elapsed().as_secs_f64();
    let mut builder = Response::builder()
        .header(header::CONTENT_TYPE, AudioEncoder::content_type(format))
        .header(header::CONTENT_LENGTH, length)
        .header("X-Audio-Sample-Rate", rate)
        .header("X-Generation-Time-Ms", format!("{:.1}", elapsed * 1000.0))
        .header("X-Audio-Duration-Secs", format!("{duration:.2}"))
        .header("X-RTF", format!("{:.3}", elapsed / duration))
        .header(
            "Access-Control-Expose-Headers",
            SPEECH_RESPONSE_EXPOSED_HEADERS,
        )
        .header(
            "X-Requested-Response-Format",
            req.response_format.as_deref().unwrap_or("wav"),
        )
        .header("X-Actual-Response-Format", resolved_format.label);
    if let Some(stats) = statistics {
        builder = builder.header("X-Tokens-Generated", stats.tokens_generated);
    }
    if let Some(fallback) = resolved_format.fallback {
        builder = builder.header("X-Response-Format-Fallback", fallback);
    }
    builder
        .body(body)
        .map_err(|e| ApiError::internal(e.to_string()))
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

    // `0`/omitted means the same model-aware automatic frame budget used by
    // the first-party product route.
    let requested_frames = req.max_output_tokens.or(req.max_tokens);
    let effective_frames = resolve_tts_output_frames(variant, &req.input, requested_frames)
        .unwrap_or(model_max_frames);

    // Estimate output duration from codec frame budget and requested speed.
    let speed = req.speed.unwrap_or(1.0).clamp(0.25, 4.0) as f64;
    let estimated_audio_secs = ((effective_frames as f64) / (frame_rate_hz as f64)) / speed;

    // Conservative real-time-factor multiplier for non-stream long-form synthesis.
    let timeout_rtf = std::env::var("IZWI_TTS_TIMEOUT_RTF")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 1.0)
        .unwrap_or_else(|| {
            if variant == ModelVariant::VibeVoice15BTts {
                izwi_core::runtime_models::architectures::vibevoice::tts::VIBEVOICE_TTS_DEFAULT_TIMEOUT_RTF
            } else {
                8.0
            }
        });
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

    match variant {
        ModelVariant::VibeVoice15BTts if !has_direct_reference_audio => {
            return Err(ApiError::bad_request(
                "VibeVoice-1.5B requires `saved_voice_id` or both `reference_audio` and `reference_text`.",
            ));
        }
        ModelVariant::FishAudioS2Pro if !has_direct_reference_audio => {
            return Err(ApiError::bad_request(
                "FishAudio-S2-Pro requires `saved_voice_id` or both `reference_audio` and `reference_text`.",
            ));
        }
        _ => {}
    }

    Ok(())
}

async fn stream_speech(
    state: AppState,
    req: SpeechRequest,
    correlation_id: String,
    tenant_key: Option<[u8; 32]>,
    variant: ModelVariant,
    resolved_format: ResolvedSpeechFormat,
) -> Result<Response<Body>, ApiError> {
    let stream_entry_started = Instant::now();
    let format = resolved_format.format;
    let format_fallback = resolved_format.fallback;
    let mut gen_request = build_generation_request(&req, correlation_id, true, variant);
    let planned_segment_count = if variant == ModelVariant::FishAudioS2Pro {
        crate::api::tts_long_form::SpeechTextPlan::fish(
            &req.input,
            gen_request.config.options.max_tokens,
        )?
        .segments
        .len()
    } else {
        1
    };
    let stream_request_id = gen_request.id.clone();
    let stream_audio_format = stream_audio_format_label(format);
    let (event_tx, mut event_rx) = mpsc::channel::<String>(stream_event_queue_capacity());

    let engine = state.runtime.clone();
    let admission_state = state.clone();
    tokio::spawn(async move {
        let permit = match tokio::select! {
            _ = event_tx.closed() => return,
            result = admission_state.acquire_owned_workload_permit(WorkloadClass::Streaming) => result,
        } {
            Ok(permit) => permit,
            Err(_) => {
                let error_event = SpeechStreamEvent {
                    event: "audio.failed",
                    timing: None,
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
        let mut runtime_context = permit.runtime_context();
        runtime_context.tenant_key = tenant_key;
        gen_request = gen_request.with_runtime_context(runtime_context);
        let load_result = tokio::select! {
            _ = event_tx.closed() => return,
            result = engine.load_model(variant) => result,
        };
        if let Err(err) = load_result {
            let _ = send_stream_event(
                &event_tx,
                SpeechStreamEvent {
                    event: "audio.failed",
                    timing: None,
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
                },
            )
            .await;
            return;
        }

        let fallback_sample_rate = engine.sample_rate().await;
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(32);
        let generation_engine = engine.clone();
        let retained_permit = if variant == ModelVariant::FishAudioS2Pro {
            drop(permit);
            None
        } else {
            Some(permit)
        };
        let generation_task = tokio::spawn(async move {
            let _permit = retained_permit;
            if variant == ModelVariant::FishAudioS2Pro {
                crate::api::tts_long_form::generate_speech_plan_stream(
                    &admission_state,
                    variant,
                    gen_request,
                    chunk_tx,
                    WorkloadClass::Streaming,
                )
                .await
            } else {
                generation_engine
                    .generate_streaming(gen_request, chunk_tx)
                    .await
            }
        });

        let mut total_samples = 0usize;
        let mut terminal_statistics = None;
        let mut first_pcm_ms = None;
        let mut last_pcm_ms = None;
        let mut audio_duration_secs = 0.0f32;
        let mut last_sample_rate = fallback_sample_rate;
        let stream_started = Instant::now();
        let mut client_closed = false;
        let mut stream_failed = false;
        let mut audio_started = false;
        while let Some(chunk) = next_stream_chunk(&mut chunk_rx, &event_tx).await {
            // A successful runtime terminal marker may contain statistics only.
            accumulate_stream_statistics(&mut terminal_statistics, &chunk);
            if !audio_started {
                if let Some(event) = speech_started_event(
                    &chunk,
                    &stream_request_id,
                    fallback_sample_rate,
                    stream_audio_format,
                    format_fallback.as_deref(),
                ) {
                    if send_stream_event(&event_tx, event).await.is_err() {
                        client_closed = true;
                        break;
                    }
                    audio_started = true;
                }
            }
            if chunk.samples.is_empty() {
                continue;
            }

            let chunk_sample_rate = chunk.sample_rate_or(fallback_sample_rate).max(1);
            let pcm_ms = stream_entry_started.elapsed().as_secs_f64() * 1000.0;
            first_pcm_ms.get_or_insert(pcm_ms);
            last_pcm_ms = Some(pcm_ms);
            total_samples += chunk.samples.len();
            audio_duration_secs += chunk.samples.len() as f32 / chunk_sample_rate as f32;
            last_sample_rate = chunk_sample_rate;
            let bytes = match encode_speech_samples(&chunk.samples, chunk_sample_rate, format) {
                Ok(bytes) => bytes,
                Err(err) => {
                    let error_event = SpeechStreamEvent {
                        event: "audio.failed",
                        timing: None,
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
                timing: None,
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
        if client_closed || event_tx.is_closed() {
            generation_task.abort();
            let _ = generation_task.await;
            return;
        }

        if stream_failed {
            generation_task.abort();
            let _ = generation_task.await;
            let done_event = SpeechStreamEvent {
                event: "done",
                timing: None,
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
                let tokens_generated = terminal_statistics
                    .as_ref()
                    .map(|stats| stats.tokens_generated);
                let execution_time_ms = terminal_statistics
                    .as_ref()
                    .map(|stats| stats.generation_time_ms);
                let rtf = if audio_duration_secs > 0.0 {
                    (generation_time_ms / 1000.0) / audio_duration_secs
                } else {
                    0.0
                };

                let final_event = SpeechStreamEvent {
                    event: "audio.done",
                    timing: Some(serde_json::json!({
                        "generation_time_basis": "post_admission_stream_wall",
                        "request_timing_basis": "stream_handler_entry_after_request_validation",
                        "execution_time_ms": execution_time_ms,
                        "execution_rtf": execution_time_ms.filter(|_| audio_duration_secs > 0.0)
                            .map(|ms| ms / 1000.0 / audio_duration_secs),
                        "first_pcm_ms": first_pcm_ms,
                        "request_to_last_pcm_ms": last_pcm_ms,
                        "request_to_last_pcm_rtf": last_pcm_ms.filter(|_| audio_duration_secs > 0.0)
                            .map(|ms| ms / 1000.0 / f64::from(audio_duration_secs)),
                        "pcm_sample_count": total_samples,
                        "token_unit": if variant == ModelVariant::FishAudioS2Pro { "semantic_frames" } else { "model_tokens" },
                        "planned_segment_count": planned_segment_count,
                    })),
                    request_id: Some(stream_request_id.clone()),
                    sequence: None,
                    audio_base64: None,
                    sample_count: None,
                    is_final: None,
                    sample_rate: Some(last_sample_rate),
                    audio_format: None,
                    tokens_generated,
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
                    timing: None,
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
                    timing: None,
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
        loop {
            match tokio::time::timeout(Duration::from_secs(10), event_rx.recv()).await {
                Ok(Some(payload)) => yield Ok::<_, Infallible>(format!("data: {payload}\n\n")),
                Ok(None) => break,
                Err(_) => yield Ok::<_, Infallible>(": keepalive\n\n".to_string()),
            }
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap())
}

/// HTTP listeners own synchronous OpenAI generation, including while inference
/// has not produced its first chunk. Closing the body must wake this wait.
async fn next_stream_chunk(
    chunks: &mut mpsc::Receiver<AudioChunk>,
    events: &mpsc::Sender<String>,
) -> Option<AudioChunk> {
    tokio::select! {
        biased;
        _ = events.closed() => None,
        chunk = chunks.recv() => chunk,
    }
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
    if let Some(max_tokens) = resolve_tts_output_frames(
        variant,
        &req.input,
        req.max_output_tokens.or(req.max_tokens),
    ) {
        gen_config.options.max_tokens = if variant == ModelVariant::FishAudioS2Pro {
            req.max_output_tokens.or(req.max_tokens).unwrap_or(0)
        } else {
            max_tokens
        };
    }
    if let Some(top_k) = req.top_k {
        gen_config.options.top_k = top_k;
    }
    gen_config.options.speaker = req.voice.clone();

    GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        model_variant: Some(variant),
        correlation_id: Some(correlation_id),
        runtime_context: Default::default(),
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

// The runtime codec default may differ from the selected model's output rate.
// Wait for real PCM before publishing playback metadata; empty terminal chunks
// carry statistics, not a usable audio format.
fn speech_started_event(
    chunk: &AudioChunk,
    request_id: &str,
    fallback_sample_rate: u32,
    audio_format: &'static str,
    format_fallback: Option<&str>,
) -> Option<SpeechStreamEvent> {
    if chunk.samples.is_empty() {
        return None;
    }
    Some(SpeechStreamEvent {
        event: "audio.started",
        timing: None,
        request_id: Some(request_id.to_string()),
        sequence: None,
        audio_base64: None,
        sample_count: None,
        is_final: None,
        sample_rate: Some(chunk.sample_rate_or(fallback_sample_rate).max(1)),
        audio_format: Some(audio_format),
        tokens_generated: None,
        generation_time_ms: None,
        audio_duration_secs: None,
        rtf: None,
        error: format_fallback.map(|fallback| format!("Requested format fallback: {fallback}")),
    })
}

fn stream_audio_format_label(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::RawF32 => "pcm_f32",
        AudioFormat::RawI16 => "pcm_i16",
    }
}

/// Non-final statistics are deltas; terminal statistics are cumulative. Audio
/// samples never imply a token count, because codec geometry varies by model.
fn accumulate_stream_statistics(total: &mut Option<izwi_core::ChunkStats>, chunk: &AudioChunk) {
    let Some(stats) = chunk.stats.as_ref() else {
        return;
    };
    match total.as_mut() {
        Some(total) if !chunk.is_final => {
            total.tokens_generated = total
                .tokens_generated
                .saturating_add(stats.tokens_generated);
            total.generation_time_ms += stats.generation_time_ms;
        }
        _ => *total = Some(stats.clone()),
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

    #[tokio::test]
    async fn disconnected_openai_listener_wakes_before_first_pcm() {
        let (_producer, mut chunks) = mpsc::channel(2);
        let (events, listener) = mpsc::channel(2);
        drop(listener);
        assert!(tokio::time::timeout(
            Duration::from_millis(100),
            next_stream_chunk(&mut chunks, &events)
        )
        .await
        .expect("disconnect must not await first PCM")
        .is_none());
    }

    #[test]
    fn qwen_auto_timeout_expands_for_long_form() {
        let req = SpeechRequest {
            model: "Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
            input: "hello ".repeat(100),
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
        assert_eq!(
            generation.config.options.max_tokens,
            resolve_tts_output_frames(ModelVariant::Voxtral4BTts2603, &req.input, None)
                .expect("Voxtral frame hint")
        );
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
        assert_eq!(timeout, 1_110);

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::VibeVoice15BTts,
        );
        assert_eq!(
            generation.config.options.max_tokens,
            resolve_tts_output_frames(ModelVariant::VibeVoice15BTts, &req.input, None)
                .expect("VibeVoice frame hint")
        );
    }

    #[test]
    fn fish_s2_tts_preserves_automatic_intent_until_segment_planning() {
        let req = SpeechRequest {
            model: "FishAudio-S2-Pro".to_string(),
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

        let timeout = resolve_speech_timeout_secs(1, ModelVariant::FishAudioS2Pro, &req);
        // The conservative Fish segment estimate includes room for pauses and EOS.
        assert_eq!(timeout, 98);

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::FishAudioS2Pro,
        );
        assert_eq!(
            generation.config.options.max_tokens, 0,
            "automatic Fish requests must be planned before per-segment budgeting"
        );
    }

    #[test]
    fn qwen_tts_omitted_max_tokens_uses_shared_text_sized_budget() {
        let req = SpeechRequest {
            model: "Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
            input: "The quick brown fox jumps over the lazy dog.".to_string(),
            voice: Some("Aiden".to_string()),
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

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        );
        assert_eq!(
            generation.config.options.max_tokens,
            resolve_tts_output_frames(ModelVariant::Qwen3Tts12Hz06BCustomVoice, &req.input, None,)
                .expect("Qwen frame hint")
        );
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
    fn fish_s2_speech_requests_require_reference_voice() {
        let req = SpeechRequest {
            model: "FishAudio-S2-Pro".to_string(),
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

        let err = validate_speech_voice_contract(&req, ModelVariant::FishAudioS2Pro)
            .expect_err("missing reference should fail");
        assert!(err.message.contains("FishAudio-S2-Pro requires"));
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
    fn speech_stream_start_waits_for_pcm_and_uses_model_rate_and_requested_format() {
        let empty = AudioChunk::final_chunk("fish".into(), 0, Vec::new());
        assert!(speech_started_event(&empty, "fish", 24_000, "pcm_i16", None).is_none());

        let pcm = AudioChunk::new("fish".into(), 0, vec![0.25]).with_sample_rate(44_100);
        for format in [AudioFormat::Wav, AudioFormat::RawI16, AudioFormat::RawF32] {
            let label = stream_audio_format_label(format);
            let event = speech_started_event(&pcm, "fish", 24_000, label, None).unwrap();
            assert_eq!(event.event, "audio.started");
            assert_eq!(event.sample_rate, Some(44_100));
            assert_eq!(event.audio_format, Some(label));
        }

        let legacy_pcm = AudioChunk::new("legacy".into(), 0, vec![0.25]);
        let event =
            speech_started_event(&legacy_pcm, "legacy", 24_000, "wav", Some("mp3 to wav")).unwrap();
        assert_eq!(event.sample_rate, Some(24_000));
        assert_eq!(
            event.error.as_deref(),
            Some("Requested format fallback: mp3 to wav")
        );
    }

    #[test]
    fn streaming_terminal_only_statistics_replace_deltas_and_unknown_stays_unknown() {
        let mut statistics = None;
        let mut first = AudioChunk::new("fish".into(), 0, vec![0.0; 2048]);
        accumulate_stream_statistics(&mut statistics, &first);
        assert!(statistics.is_none());
        first.stats = Some(izwi_core::ChunkStats {
            generation_time_ms: 10.0,
            tokens_generated: 1,
            rtf: 0.2,
        });
        accumulate_stream_statistics(&mut statistics, &first);
        let mut terminal = AudioChunk::final_chunk("fish".into(), 1, Vec::new());
        terminal.stats = Some(izwi_core::ChunkStats {
            generation_time_ms: 25.0,
            tokens_generated: 2,
            rtf: 0.3,
        });
        accumulate_stream_statistics(&mut statistics, &terminal);
        let statistics = statistics.unwrap();
        assert_eq!(statistics.tokens_generated, 2);
        assert_eq!(statistics.generation_time_ms, 25.0);
    }

    #[test]
    fn streaming_wav_chunks_are_independent_containers_with_matching_raw_pcm() {
        for samples in [&[0.0f32, 0.25][..], &[-0.25f32][..]] {
            let wav = encode_speech_samples(samples, 44_100, AudioFormat::Wav).unwrap();
            let raw = encode_speech_samples(samples, 44_100, AudioFormat::RawI16).unwrap();
            assert_eq!(&wav[..4], b"RIFF");
            let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
            assert_eq!(reader.duration() as usize, samples.len());
            let decoded = reader
                .into_samples::<i16>()
                .map(|sample| sample.unwrap())
                .flat_map(i16::to_le_bytes)
                .collect::<Vec<_>>();
            assert_eq!(decoded, raw);
            assert_eq!(raw.len(), samples.len() * 2);
        }
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
            timing: None,
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
