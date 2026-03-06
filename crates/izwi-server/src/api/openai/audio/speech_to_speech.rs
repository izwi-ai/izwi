//! Speech-to-speech endpoint backed by LFM2.5-Audio interleaved generation.

use axum::{
    body::Body,
    extract::{Extension, Multipart, Request, State},
    http::{header, StatusCode},
    response::Response,
    Json, RequestExt,
};
use base64::Engine;
use std::convert::Infallible;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::{parse_model_variant, Error as CoreError, ModelVariant};

const DEFAULT_STREAM_EVENT_QUEUE_CAPACITY: usize = 32;

#[derive(Debug, Default)]
struct SpeechToSpeechRequest {
    audio_base64: Option<String>,
    model: Option<String>,
    language: Option<String>,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    stream: bool,
}

#[derive(Debug, serde::Serialize)]
struct SpeechToSpeechResponse {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    transcription: Option<String>,
    audio_base64: String,
    sample_rate: u32,
    generation_time_ms: f64,
}

#[derive(Debug, serde::Serialize)]
struct SpeechToSpeechStreamEvent {
    event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    sequence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_final: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transcription: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct SpeechToSpeechJsonPayload {
    audio_base64: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
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
    event: SpeechToSpeechStreamEvent,
) -> Result<(), ()> {
    let payload = serde_json::to_string(&event).unwrap_or_default();
    event_tx.send(payload).await.map_err(|_| ())
}

fn try_send_stream_event(
    event_tx: &mpsc::Sender<String>,
    event: SpeechToSpeechStreamEvent,
) -> std::result::Result<(), String> {
    let payload = serde_json::to_string(&event).unwrap_or_default();
    event_tx.try_send(payload).map_err(|err| match err {
        mpsc::error::TrySendError::Full(_) => {
            "Streaming response backpressure exceeded queue capacity".to_string()
        }
        mpsc::error::TrySendError::Closed(_) => "Streaming response channel closed".to_string(),
    })
}

pub async fn speech_to_speech(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    req: Request,
) -> Result<Response<Body>, ApiError> {
    let mut req = parse_request(req).await?;
    let audio_base64 = req
        .audio_base64
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;

    let variant = resolve_lfm2_variant(req.model.as_deref())?;
    state.runtime.load_model(variant).await?;

    if req.stream {
        return stream_response(state, req, audio_base64, ctx.correlation_id).await;
    }

    let _permit = state.acquire_permit().await;
    let timeout = Duration::from_secs(state.request_timeout_secs);

    let result = tokio::time::timeout(timeout, async {
        state
            .runtime
            .lfm2_speech_to_speech_with_correlation(
                &audio_base64,
                req.language.as_deref(),
                req.system_prompt.as_deref(),
                req.temperature,
                req.top_k,
                Some(&ctx.correlation_id),
            )
            .await
    })
    .await
    .map_err(|_| ApiError::internal("Request timeout"))??;

    let wav_base64 = samples_to_wav_base64(&result.samples, result.sample_rate)?;

    let payload = SpeechToSpeechResponse {
        text: result.text,
        transcription: result.input_transcription,
        audio_base64: wav_base64,
        sample_rate: result.sample_rate,
        generation_time_ms: result.generation_time_ms,
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap_or_default()))
        .unwrap())
}

async fn stream_response(
    state: AppState,
    req: SpeechToSpeechRequest,
    audio_base64: String,
    correlation_id: String,
) -> Result<Response<Body>, ApiError> {
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let language = req.language;
    let system_prompt = req.system_prompt;
    let temperature = req.temperature;
    let top_k = req.top_k;

    let (event_tx, mut event_rx) = mpsc::channel::<String>(stream_event_queue_capacity());
    let engine = state.runtime.clone();
    let semaphore = state.request_semaphore.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = send_stream_event(
                    &event_tx,
                    SpeechToSpeechStreamEvent {
                        event: "error",
                        sequence: None,
                        is_final: None,
                        delta: None,
                        text: None,
                        transcription: None,
                        audio_base64: None,
                        sample_rate: None,
                        generation_time_ms: None,
                        error: Some("Server is shutting down".to_string()),
                    },
                )
                .await;
                let _ = send_stream_event(
                    &event_tx,
                    SpeechToSpeechStreamEvent {
                        event: "done",
                        sequence: None,
                        is_final: None,
                        delta: None,
                        text: None,
                        transcription: None,
                        audio_base64: None,
                        sample_rate: None,
                        generation_time_ms: None,
                        error: None,
                    },
                )
                .await;
                return;
            }
        };

        if send_stream_event(
            &event_tx,
            SpeechToSpeechStreamEvent {
                event: "start",
                sequence: None,
                is_final: None,
                delta: None,
                text: None,
                transcription: None,
                audio_base64: None,
                sample_rate: None,
                generation_time_ms: None,
                error: None,
            },
        )
        .await
        .is_err()
        {
            return;
        }

        let delta_tx = event_tx.clone();
        let audio_tx = event_tx.clone();
        let stream_sample_rate = 24_000u32;
        let stream_encoder = AudioEncoder::new(stream_sample_rate, 1);
        let result = tokio::time::timeout(timeout, async {
            engine
                .lfm2_speech_to_speech_streaming_with_correlation(
                    &audio_base64,
                    language.as_deref(),
                    system_prompt.as_deref(),
                    temperature,
                    top_k,
                    Some(correlation_id.as_str()),
                    move |delta| {
                        try_send_stream_event(
                            &delta_tx,
                            SpeechToSpeechStreamEvent {
                                event: "delta",
                                sequence: None,
                                is_final: None,
                                delta: Some(delta),
                                text: None,
                                transcription: None,
                                audio_base64: None,
                                sample_rate: None,
                                generation_time_ms: None,
                                error: None,
                            },
                        )
                        .map_err(CoreError::InferenceError)
                    },
                    move |audio_chunk| {
                        if audio_chunk.samples.is_empty() && !audio_chunk.is_final {
                            return Ok(());
                        }

                        let bytes = stream_encoder
                            .encode(&audio_chunk.samples, AudioFormat::RawI16)
                            .map_err(|err| {
                                CoreError::InferenceError(format!(
                                    "Failed to encode audio chunk: {}",
                                    err
                                ))
                            })?;
                        try_send_stream_event(
                            &audio_tx,
                            SpeechToSpeechStreamEvent {
                                event: "audio_chunk",
                                sequence: Some(audio_chunk.sequence),
                                is_final: Some(audio_chunk.is_final),
                                delta: None,
                                text: None,
                                transcription: None,
                                audio_base64: Some(
                                    base64::engine::general_purpose::STANDARD.encode(bytes),
                                ),
                                sample_rate: Some(stream_sample_rate),
                                generation_time_ms: None,
                                error: None,
                            },
                        )
                        .map_err(CoreError::InferenceError)
                    },
                )
                .await
        })
        .await;

        match result {
            Ok(Ok(output)) => match samples_to_wav_base64(&output.samples, output.sample_rate) {
                Ok(audio_base64) => {
                    let _ = send_stream_event(
                        &event_tx,
                        SpeechToSpeechStreamEvent {
                            event: "final",
                            sequence: None,
                            is_final: None,
                            delta: None,
                            text: Some(output.text),
                            transcription: output.input_transcription,
                            audio_base64: Some(audio_base64),
                            sample_rate: Some(output.sample_rate),
                            generation_time_ms: Some(output.generation_time_ms),
                            error: None,
                        },
                    )
                    .await;
                }
                Err(err) => {
                    let _ = send_stream_event(
                        &event_tx,
                        SpeechToSpeechStreamEvent {
                            event: "error",
                            sequence: None,
                            is_final: None,
                            delta: None,
                            text: None,
                            transcription: None,
                            audio_base64: None,
                            sample_rate: None,
                            generation_time_ms: None,
                            error: Some(err.message),
                        },
                    )
                    .await;
                }
            },
            Ok(Err(err)) => {
                let _ = send_stream_event(
                    &event_tx,
                    SpeechToSpeechStreamEvent {
                        event: "error",
                        sequence: None,
                        is_final: None,
                        delta: None,
                        text: None,
                        transcription: None,
                        audio_base64: None,
                        sample_rate: None,
                        generation_time_ms: None,
                        error: Some(err.to_string()),
                    },
                )
                .await;
            }
            Err(_) => {
                let _ = send_stream_event(
                    &event_tx,
                    SpeechToSpeechStreamEvent {
                        event: "error",
                        sequence: None,
                        is_final: None,
                        delta: None,
                        text: None,
                        transcription: None,
                        audio_base64: None,
                        sample_rate: None,
                        generation_time_ms: None,
                        error: Some("Request timed out".to_string()),
                    },
                )
                .await;
            }
        }

        let _ = send_stream_event(
            &event_tx,
            SpeechToSpeechStreamEvent {
                event: "done",
                sequence: None,
                is_final: None,
                delta: None,
                text: None,
                transcription: None,
                audio_base64: None,
                sample_rate: None,
                generation_time_ms: None,
                error: None,
            },
        )
        .await;
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

async fn parse_request(req: Request) -> Result<SpeechToSpeechRequest, ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let Json(payload) = req
            .extract::<Json<SpeechToSpeechJsonPayload>, _>()
            .await
            .map_err(|err| ApiError::bad_request(format!("Invalid JSON payload: {err}")))?;

        return Ok(SpeechToSpeechRequest {
            audio_base64: Some(payload.audio_base64),
            model: payload.model,
            language: payload.language,
            system_prompt: payload.system_prompt,
            temperature: payload.temperature,
            top_k: payload.top_k,
            stream: payload.stream.unwrap_or(false),
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|err| ApiError::bad_request(format!("Invalid multipart payload: {err}")))?;

        let mut out = SpeechToSpeechRequest::default();

        while let Some(field) = multipart.next_field().await.map_err(|err| {
            ApiError::bad_request(format!("Failed reading multipart field: {err}"))
        })? {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let bytes = field.bytes().await.map_err(|err| {
                        ApiError::bad_request(format!("Invalid audio field: {err}"))
                    })?;
                    if !bytes.is_empty() {
                        out.audio_base64 =
                            Some(base64::engine::general_purpose::STANDARD.encode(&bytes));
                    }
                }
                "model" => out.model = Some(field_text(field).await?),
                "language" => out.language = Some(field_text(field).await?),
                "system_prompt" | "instructions" => {
                    out.system_prompt = Some(field_text(field).await?)
                }
                "temperature" => {
                    let value = field_text(field).await?;
                    out.temperature = value.trim().parse::<f32>().ok();
                }
                "top_k" => {
                    let value = field_text(field).await?;
                    out.top_k = value.trim().parse::<usize>().ok();
                }
                "stream" => {
                    let value = field_text(field).await?;
                    out.stream = matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes"
                    );
                }
                _ => {
                    let _ = field.bytes().await;
                }
            }
        }

        return Ok(out);
    }

    Err(ApiError::bad_request(
        "Unsupported content type. Use application/json or multipart/form-data.",
    ))
}

async fn field_text(field: axum::extract::multipart::Field<'_>) -> Result<String, ApiError> {
    field
        .text()
        .await
        .map_err(|err| ApiError::bad_request(format!("Invalid text field: {err}")))
}

fn resolve_lfm2_variant(model_id: Option<&str>) -> Result<ModelVariant, ApiError> {
    let model_id = model_id.unwrap_or("LFM2.5-Audio-1.5B");
    let variant =
        parse_model_variant(model_id).map_err(|err| ApiError::bad_request(err.to_string()))?;
    if !variant.is_lfm2() {
        return Err(ApiError::bad_request(format!(
            "Unsupported speech-to-speech model '{}'. Supported: LFM2.5-Audio-1.5B, LFM2.5-Audio-1.5B-4bit",
            model_id
        )));
    }
    Ok(variant)
}

fn samples_to_wav_base64(samples: &[f32], sample_rate: u32) -> Result<String, ApiError> {
    let encoder = AudioEncoder::new(sample_rate, 1);
    let wav_bytes = encoder
        .encode(samples, AudioFormat::Wav)
        .map_err(|err| ApiError::internal(format!("Failed to encode WAV response: {}", err)))?;

    Ok(base64::engine::general_purpose::STANDARD.encode(wav_bytes))
}
