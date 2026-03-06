use std::convert::Infallible;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::{Extension, Multipart, Path, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json, RequestExt,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;
use crate::transcription_store::{
    NewTranscriptionRecord, StoredTranscriptionAudio, TranscriptionRecord,
    TranscriptionRecordSummary,
};

const HISTORY_LIST_LIMIT: usize = 200;

#[derive(Debug, Serialize)]
pub struct TranscriptionRecordListResponse {
    pub records: Vec<TranscriptionRecordSummary>,
}

#[derive(Debug, Serialize)]
pub struct DeleteTranscriptionRecordResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Default)]
struct ParsedTranscriptionCreateRequest {
    audio_bytes: Vec<u8>,
    audio_mime_type: Option<String>,
    audio_filename: Option<String>,
    model_id: Option<String>,
    language: Option<String>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct JsonCreateRequest {
    audio_base64: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct StreamStartEvent {
    event: &'static str,
}

#[derive(Debug, Serialize)]
struct StreamDeltaEvent {
    event: &'static str,
    delta: String,
}

#[derive(Debug, Serialize)]
struct StreamFinalEvent {
    event: &'static str,
    record: TranscriptionRecord,
}

#[derive(Debug, Serialize)]
struct StreamErrorEvent {
    event: &'static str,
    error: String,
}

#[derive(Debug, Serialize)]
struct StreamDoneEvent {
    event: &'static str,
}

pub async fn list_records(
    State(state): State<AppState>,
) -> Result<Json<TranscriptionRecordListResponse>, ApiError> {
    let records = state
        .transcription_store
        .list_records(HISTORY_LIST_LIMIT)
        .await
        .map_err(map_store_error)?;

    Ok(Json(TranscriptionRecordListResponse { records }))
}

pub async fn get_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<TranscriptionRecord>, ApiError> {
    let record = state
        .transcription_store
        .get_record(record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Transcription record not found"))?;

    Ok(Json(record))
}

pub async fn get_record_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Response, ApiError> {
    let audio = state
        .transcription_store
        .get_audio(record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Transcription audio not found"))?;

    Ok(audio_response(audio))
}

pub async fn delete_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<DeleteTranscriptionRecordResponse>, ApiError> {
    let deleted = state
        .transcription_store
        .delete_record(record_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("Transcription record not found"));
    }

    Ok(Json(DeleteTranscriptionRecordResponse {
        id: record_id,
        deleted: true,
    }))
}

pub async fn create_record(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    req: Request,
) -> Result<Response, ApiError> {
    let parsed = parse_create_request(req).await?;

    if parsed.stream {
        return create_record_stream(state, parsed, ctx.correlation_id).await;
    }

    let _permit = state.acquire_permit().await;
    let started = Instant::now();

    let output = state
        .runtime
        .asr_transcribe_streaming_bytes_with_correlation(
            parsed.audio_bytes.as_slice(),
            parsed.model_id.as_deref(),
            parsed.language.as_deref(),
            Some(&ctx.correlation_id),
            |_delta| {},
        )
        .await?;

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let rtf = if output.duration_secs > 0.0 {
        Some((elapsed_ms / 1000.0) / output.duration_secs as f64)
    } else {
        None
    };

    let record = state
        .transcription_store
        .create_record(NewTranscriptionRecord {
            model_id: parsed.model_id,
            language: output.language.or(parsed.language),
            duration_secs: Some(output.duration_secs as f64),
            processing_time_ms: elapsed_ms,
            rtf,
            audio_mime_type: parsed
                .audio_mime_type
                .unwrap_or_else(|| "audio/wav".to_string()),
            audio_filename: parsed.audio_filename,
            audio_bytes: parsed.audio_bytes,
            transcription: output.text,
        })
        .await
        .map_err(map_store_error)?;

    Ok(Json(record).into_response())
}

async fn create_record_stream(
    state: AppState,
    parsed: ParsedTranscriptionCreateRequest,
    correlation_id: String,
) -> Result<Response, ApiError> {
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let model_id = parsed.model_id;
    let requested_language = parsed.language;
    let audio_mime_type = parsed
        .audio_mime_type
        .unwrap_or_else(|| "audio/wav".to_string());
    let audio_filename = parsed.audio_filename;
    let audio_bytes = parsed.audio_bytes;

    let runtime = state.runtime.clone();
    let transcription_store = state.transcription_store.clone();
    let semaphore = state.request_semaphore.clone();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::to_string(&StreamErrorEvent {
                        event: "error",
                        error: "Server is shutting down".to_string(),
                    })
                    .unwrap_or_default(),
                );
                let _ = event_tx.send(
                    serde_json::to_string(&StreamDoneEvent { event: "done" }).unwrap_or_default(),
                );
                return;
            }
        };

        let _ = event_tx
            .send(serde_json::to_string(&StreamStartEvent { event: "start" }).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let started = Instant::now();

        let generation_result = tokio::time::timeout(timeout, async {
            runtime
                .asr_transcribe_streaming_bytes_with_correlation(
                    audio_bytes.as_slice(),
                    model_id.as_deref(),
                    requested_language.as_deref(),
                    Some(correlation_id.as_str()),
                    move |delta| {
                        let _ = delta_tx.send(
                            serde_json::to_string(&StreamDeltaEvent {
                                event: "delta",
                                delta,
                            })
                            .unwrap_or_default(),
                        );
                    },
                )
                .await
        })
        .await;

        match generation_result {
            Ok(Ok(output)) => {
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let rtf = if output.duration_secs > 0.0 {
                    Some((elapsed_ms / 1000.0) / output.duration_secs as f64)
                } else {
                    None
                };

                match transcription_store
                    .create_record(NewTranscriptionRecord {
                        model_id: model_id.clone(),
                        language: output.language.or(requested_language.clone()),
                        duration_secs: Some(output.duration_secs as f64),
                        processing_time_ms: elapsed_ms,
                        rtf,
                        audio_mime_type,
                        audio_filename,
                        audio_bytes,
                        transcription: output.text,
                    })
                    .await
                {
                    Ok(record) => {
                        let _ = event_tx.send(
                            serde_json::to_string(&StreamFinalEvent {
                                event: "final",
                                record,
                            })
                            .unwrap_or_default(),
                        );
                    }
                    Err(err) => {
                        let _ = event_tx.send(
                            serde_json::to_string(&StreamErrorEvent {
                                event: "error",
                                error: format!("Failed to save transcription record: {err}"),
                            })
                            .unwrap_or_default(),
                        );
                    }
                }
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(
                    serde_json::to_string(&StreamErrorEvent {
                        event: "error",
                        error: err.to_string(),
                    })
                    .unwrap_or_default(),
                );
            }
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::to_string(&StreamErrorEvent {
                        event: "error",
                        error: "Transcription request timed out".to_string(),
                    })
                    .unwrap_or_default(),
                );
            }
        }

        let _ = event_tx
            .send(serde_json::to_string(&StreamDoneEvent { event: "done" }).unwrap_or_default());
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(Event::default().data(payload));
        }
    };

    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(response)
}

async fn parse_create_request(req: Request) -> Result<ParsedTranscriptionCreateRequest, ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let Json(payload) = req
            .extract::<Json<JsonCreateRequest>, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid JSON payload: {e}")))?;

        let model_id = payload.model_id.or(payload.model);
        let audio_bytes = decode_audio_base64(payload.audio_base64.as_str())?;

        return Ok(ParsedTranscriptionCreateRequest {
            audio_bytes,
            audio_mime_type: Some("audio/wav".to_string()),
            audio_filename: None,
            model_id,
            language: payload.language,
            stream: payload.stream.unwrap_or(false),
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart payload: {e}")))?;

        let mut out = ParsedTranscriptionCreateRequest::default();

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::bad_request(format!("Failed reading multipart field: {e}")))?
        {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let filename = field.file_name().map(|value| value.to_string());
                    let mime_type = field.content_type().map(|value| value.to_string());
                    let bytes = field.bytes().await.map_err(|e| {
                        ApiError::bad_request(format!("Failed reading '{name}' bytes: {e}"))
                    })?;
                    if !bytes.is_empty() {
                        out.audio_bytes = bytes.to_vec();
                        out.audio_filename = filename;
                        out.audio_mime_type = mime_type;
                    }
                }
                "audio_base64" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'audio_base64' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.audio_bytes = decode_audio_base64(text.as_str())?;
                    }
                }
                "model" | "model_id" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{name}' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.model_id = Some(text.trim().to_string());
                    }
                }
                "language" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'language' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.language = Some(text.trim().to_string());
                    }
                }
                "stream" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'stream' field: {e}"
                        ))
                    })?;
                    out.stream = parse_bool(text.as_str());
                }
                _ => {}
            }
        }

        if out.audio_bytes.is_empty() {
            return Err(ApiError::bad_request(
                "Missing audio input (`file` or `audio_base64`)",
            ));
        }

        return Ok(out);
    }

    Err(ApiError {
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        message: "Expected `Content-Type: application/json` or `multipart/form-data`".to_string(),
    })
}

fn decode_audio_base64(input: &str) -> Result<Vec<u8>, ApiError> {
    let payload = input
        .split_once(',')
        .map(|(_, value)| value)
        .unwrap_or(input)
        .trim();

    if payload.is_empty() {
        return Err(ApiError::bad_request("Audio payload is empty"));
    }

    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|err| ApiError::bad_request(format!("Invalid base64 audio payload: {err}")))
}

fn parse_bool(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn audio_response(audio: StoredTranscriptionAudio) -> Response {
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(audio.audio_bytes));

    if let Ok(content_type) = HeaderValue::from_str(audio.audio_mime_type.as_str()) {
        response = response.map(|mut body| {
            let headers = body.headers_mut();
            headers.insert(header::CONTENT_TYPE, content_type);
            body
        });
    }

    if let Some(filename) = audio.audio_filename {
        let disposition = format!("inline; filename=\"{}\"", filename.replace('"', ""));
        if let Ok(value) = HeaderValue::from_str(disposition.as_str()) {
            response = response.map(|mut body| {
                let headers = body.headers_mut();
                headers.insert(header::CONTENT_DISPOSITION, value);
                body
            });
        }
    }

    response.unwrap_or_else(|_| {
        Response::builder()
            .status(StatusCode::OK)
            .body(Body::empty())
            .unwrap_or_else(|_| Response::new(Body::empty()))
    })
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Transcription storage error: {err}"))
}
