use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Extension, Multipart, Path, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json, RequestExt,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::audio_payload::{
    decode_base64_audio_payload, inspect_audio_payload_with_diagnostics,
    read_multipart_audio_base64_payload, read_multipart_audio_file_payload,
};
use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;
use crate::transcription_store::{
    CompleteTranscriptionRecord, NewTranscriptionRecord, TranscriptionProcessingStatus,
    TranscriptionRecord, TranscriptionSegmentRecord, TranscriptionStore,
    TranscriptionSummaryStatus, TranscriptionWordRecord, UpdateTranscriptionSummary,
};
use izwi_core::{
    parse_chat_model_variant, AsrProgress, AsrProgressPhase, ChatMessage, ChatRequestConfig,
    ChatRole, GenerationParams, ModelVariant, RuntimeService,
};

use super::AUDIO_UPLOAD_LIMIT_BYTES;
use crate::api::speech_text_upload::{multipart_upload_error, resolve_source_audio_mime_type};

const DEFAULT_TRANSCRIPTION_ALIGNER_MODEL: &str = "Qwen3-ForcedAligner-0.6B";
const DEFAULT_TRANSCRIPTION_SUMMARY_MODEL: &str = "Qwen3.5-4B";
const DEFAULT_TRANSCRIPTION_SUMMARY_MAX_TOKENS: usize = 384;
const TRANSCRIPTION_SUMMARY_SYSTEM_PROMPT: &str = "You summarize transcripts for fast review. Return only the final summary text. Do not output markdown, bullet markers, XML tags, code fences, or <think> content. Keep the summary concise while covering key topics, decisions, action items, and unresolved questions when present.";
const MAX_SEGMENT_WORDS: usize = 18;
const MAX_SEGMENT_DURATION_SECS: f32 = 9.0;
const MIN_SENTENCE_BREAK_WORDS: usize = 5;
const SEGMENT_GAP_BREAK_SECS: f32 = 0.85;

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
    aligner_model_id: Option<String>,
    language: Option<String>,
    include_timestamps: bool,
    stream: bool,
    generate_summary: bool,
}

#[derive(Debug, Deserialize)]
struct JsonCreateRequest {
    audio_base64: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default, alias = "aligner_model")]
    aligner_model_id: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    include_timestamps: Option<bool>,
    #[serde(default)]
    word_timestamps: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    generate_summary: Option<bool>,
}

#[derive(Debug, Serialize)]
struct StreamStartEvent {
    event: &'static str,
}

#[derive(Debug, Serialize)]
struct StreamCreatedEvent {
    event: &'static str,
    record: TranscriptionRecord,
}

#[derive(Debug, Serialize)]
struct StreamDeltaEvent {
    event: &'static str,
    delta: String,
}

#[derive(Debug, Serialize)]
struct StreamProgressEvent {
    event: &'static str,
    progress: AsrProgress,
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

#[derive(Debug)]
struct GeneratedTranscriptionArtifacts {
    text: String,
    language: Option<String>,
    duration_secs: f64,
    aligner_model_id: Option<String>,
    segments: Vec<TranscriptionSegmentRecord>,
    words: Vec<TranscriptionWordRecord>,
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

pub async fn regenerate_summary(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(record_id): Path<String>,
) -> Result<Json<TranscriptionRecord>, ApiError> {
    let record = state
        .transcription_store
        .update_summary(
            record_id,
            UpdateTranscriptionSummary {
                status: TranscriptionSummaryStatus::Pending,
                model_id: Some(DEFAULT_TRANSCRIPTION_SUMMARY_MODEL.to_string()),
                text: None,
                error: None,
                updated_at: None,
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Transcription record not found"))?;

    maybe_spawn_summary_generation(
        state.runtime.clone(),
        state.transcription_store.clone(),
        state.request_semaphore.clone(),
        &record,
        Some(ctx.correlation_id),
    );

    Ok(Json(record))
}

pub async fn create_record(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    req: Request,
) -> Result<Response, ApiError> {
    let mut parsed = parse_create_request(req).await?;
    let placeholder = create_pending_record(&state, &mut parsed).await?;

    if parsed.stream {
        return create_record_stream(state, parsed, placeholder, ctx.correlation_id).await;
    }

    spawn_transcription_processing_task(
        state.runtime.clone(),
        state.transcription_store.clone(),
        state.request_semaphore.clone(),
        placeholder.id.clone(),
        parsed,
        Some(ctx.correlation_id),
        None,
    );

    Ok((StatusCode::ACCEPTED, Json(placeholder)).into_response())
}

async fn create_record_stream(
    state: AppState,
    parsed: ParsedTranscriptionCreateRequest,
    placeholder: TranscriptionRecord,
    correlation_id: String,
) -> Result<Response, ApiError> {
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let _ = event_tx.send(
        serde_json::to_string(&StreamCreatedEvent {
            event: "created",
            record: placeholder.clone(),
        })
        .unwrap_or_default(),
    );

    spawn_transcription_processing_task(
        state.runtime.clone(),
        state.transcription_store.clone(),
        state.request_semaphore.clone(),
        placeholder.id.clone(),
        parsed,
        Some(correlation_id),
        Some(event_tx),
    );

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

async fn create_pending_record(
    state: &AppState,
    parsed: &mut ParsedTranscriptionCreateRequest,
) -> Result<TranscriptionRecord, ApiError> {
    state
        .transcription_store
        .create_record(NewTranscriptionRecord {
            model_id: parsed.model_id.clone(),
            aligner_model_id: parsed.aligner_model_id.clone(),
            language: parsed.language.clone(),
            processing_status: TranscriptionProcessingStatus::Pending,
            processing_error: None,
            processing_progress: None,
            duration_secs: None,
            processing_time_ms: 0.0,
            rtf: None,
            audio_mime_type: resolve_source_audio_mime_type(
                parsed.audio_mime_type.as_deref(),
                parsed.audio_filename.as_deref(),
            ),
            audio_filename: parsed.audio_filename.clone(),
            audio_bytes: std::mem::take(&mut parsed.audio_bytes),
            transcription: String::new(),
            segments: Vec::new(),
            words: Vec::new(),
            summary_status: TranscriptionSummaryStatus::NotRequested,
            summary_model_id: None,
            summary_text: None,
            summary_error: None,
            summary_updated_at: None,
        })
        .await
        .map_err(map_store_error)
}

fn spawn_transcription_processing_task(
    runtime: Arc<RuntimeService>,
    transcription_store: Arc<TranscriptionStore>,
    semaphore: Arc<tokio::sync::Semaphore>,
    record_id: String,
    parsed: ParsedTranscriptionCreateRequest,
    correlation_id: Option<String>,
    event_tx: Option<mpsc::UnboundedSender<String>>,
) {
    tokio::spawn(async move {
        let send_event = |payload: String| {
            if let Some(tx) = &event_tx {
                let _ = tx.send(payload);
            }
        };

        let _permit = match semaphore.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let error_message = "Server is shutting down".to_string();
                let _ = transcription_store
                    .update_processing_status(
                        record_id.clone(),
                        TranscriptionProcessingStatus::Failed,
                        Some(error_message.clone()),
                    )
                    .await;
                send_event(
                    serde_json::to_string(&StreamErrorEvent {
                        event: "error",
                        error: error_message,
                    })
                    .unwrap_or_default(),
                );
                send_event(
                    serde_json::to_string(&StreamDoneEvent { event: "done" }).unwrap_or_default(),
                );
                return;
            }
        };

        let _ = transcription_store
            .update_processing_status(
                record_id.clone(),
                TranscriptionProcessingStatus::Processing,
                None,
            )
            .await;
        let initial_progress = transcription_processing_progress();
        let _ = transcription_store
            .update_processing_progress(record_id.clone(), Some(initial_progress.clone()))
            .await;
        send_event(progress_event_payload(initial_progress));
        send_event(serde_json::to_string(&StreamStartEvent { event: "start" }).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let progress_tx = event_tx.clone();
        let progress_store = transcription_store.clone();
        let progress_record_id = record_id.clone();
        let started = Instant::now();
        let model_id = parsed.model_id.clone();
        let aligner_model_id = parsed.aligner_model_id.clone();
        let requested_language = parsed.language.clone();
        let include_timestamps = parsed.include_timestamps;
        let generate_summary = parsed.generate_summary;
        let correlation_id_ref = correlation_id.clone();
        let audio_bytes = match transcription_store.get_audio(record_id.clone()).await {
            Ok(Some(audio)) => audio.audio_bytes,
            Ok(None) => {
                let message = "Transcription audio payload not found".to_string();
                let _ = transcription_store
                    .update_processing_status(
                        record_id.clone(),
                        TranscriptionProcessingStatus::Failed,
                        Some(message.clone()),
                    )
                    .await;
                send_event(
                    serde_json::to_string(&StreamErrorEvent {
                        event: "error",
                        error: message,
                    })
                    .unwrap_or_default(),
                );
                send_event(
                    serde_json::to_string(&StreamDoneEvent { event: "done" }).unwrap_or_default(),
                );
                return;
            }
            Err(err) => {
                let message = format!("Failed to read transcription audio payload: {err}");
                let _ = transcription_store
                    .update_processing_status(
                        record_id.clone(),
                        TranscriptionProcessingStatus::Failed,
                        Some(message.clone()),
                    )
                    .await;
                send_event(
                    serde_json::to_string(&StreamErrorEvent {
                        event: "error",
                        error: message,
                    })
                    .unwrap_or_default(),
                );
                send_event(
                    serde_json::to_string(&StreamDoneEvent { event: "done" }).unwrap_or_default(),
                );
                return;
            }
        };

        // Keep transcription processing unbounded by wall-clock timeout so
        // valid long jobs can finish and persist successfully.
        let generation_result = generate_transcription_artifacts(
            runtime.clone(),
            audio_bytes.as_slice(),
            model_id.as_deref(),
            aligner_model_id.as_deref(),
            requested_language.as_deref(),
            include_timestamps,
            correlation_id_ref.as_deref(),
            move |delta| {
                if let Some(tx) = &delta_tx {
                    let _ = tx.send(
                        serde_json::to_string(&StreamDeltaEvent {
                            event: "delta",
                            delta,
                        })
                        .unwrap_or_default(),
                    );
                }
            },
            move |progress| {
                if let Some(tx) = &progress_tx {
                    let _ = tx.send(progress_event_payload(progress.clone()));
                }
                let store = progress_store.clone();
                let id = progress_record_id.clone();
                tokio::spawn(async move {
                    let _ = store.update_processing_progress(id, Some(progress)).await;
                });
            },
        )
        .await;

        match generation_result {
            Ok(artifacts) => {
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let rtf = if artifacts.duration_secs > 0.0 {
                    Some((elapsed_ms / 1000.0) / artifacts.duration_secs)
                } else {
                    None
                };
                let (summary_status, summary_model_id) =
                    initial_summary_state(artifacts.text.as_str(), generate_summary);

                match transcription_store
                    .complete_record(
                        record_id.clone(),
                        CompleteTranscriptionRecord {
                            model_id,
                            aligner_model_id: artifacts.aligner_model_id,
                            language: artifacts.language,
                            duration_secs: Some(artifacts.duration_secs),
                            processing_time_ms: elapsed_ms,
                            rtf,
                            transcription: artifacts.text,
                            segments: artifacts.segments,
                            words: artifacts.words,
                            summary_status,
                            summary_model_id,
                            summary_text: None,
                            summary_error: None,
                            summary_updated_at: None,
                        },
                    )
                    .await
                {
                    Ok(Some(record)) => {
                        maybe_spawn_summary_generation(
                            runtime.clone(),
                            transcription_store.clone(),
                            semaphore.clone(),
                            &record,
                            correlation_id.clone(),
                        );
                        send_event(
                            serde_json::to_string(&StreamFinalEvent {
                                event: "final",
                                record,
                            })
                            .unwrap_or_default(),
                        );
                    }
                    Ok(None) => {
                        send_event(
                            serde_json::to_string(&StreamErrorEvent {
                                event: "error",
                                error: "Transcription record not found".to_string(),
                            })
                            .unwrap_or_default(),
                        );
                    }
                    Err(err) => {
                        let message = format!("Failed to save transcription record: {err}");
                        let _ = transcription_store
                            .update_processing_status(
                                record_id.clone(),
                                TranscriptionProcessingStatus::Failed,
                                Some(message.clone()),
                            )
                            .await;
                        send_event(
                            serde_json::to_string(&StreamErrorEvent {
                                event: "error",
                                error: message,
                            })
                            .unwrap_or_default(),
                        );
                    }
                }
            }
            Err(err) => {
                let _ = transcription_store
                    .update_processing_status(
                        record_id.clone(),
                        TranscriptionProcessingStatus::Failed,
                        Some(err.message.clone()),
                    )
                    .await;
                send_event(
                    serde_json::to_string(&StreamErrorEvent {
                        event: "error",
                        error: err.message,
                    })
                    .unwrap_or_default(),
                );
            }
        }

        send_event(serde_json::to_string(&StreamDoneEvent { event: "done" }).unwrap_or_default());
    });
}

async fn generate_transcription_artifacts<F, P>(
    runtime: std::sync::Arc<izwi_core::RuntimeService>,
    audio_bytes: &[u8],
    model_id: Option<&str>,
    aligner_model_id: Option<&str>,
    requested_language: Option<&str>,
    include_timestamps: bool,
    correlation_id: Option<&str>,
    on_delta: F,
    on_progress: P,
) -> Result<GeneratedTranscriptionArtifacts, ApiError>
where
    F: FnMut(String) + Send + 'static,
    P: FnMut(AsrProgress) + Send + 'static,
{
    let progress_callback = std::sync::Arc::new(std::sync::Mutex::new(on_progress));
    let runtime_progress_callback = progress_callback.clone();
    let output = runtime
        .asr_transcribe_streaming_bytes_with_progress_and_correlation(
            audio_bytes,
            model_id,
            requested_language,
            correlation_id,
            on_delta,
            move |progress| {
                if let Ok(mut callback) = runtime_progress_callback.lock() {
                    callback(progress);
                }
            },
        )
        .await?;

    let resolved_language = output
        .language
        .clone()
        .or_else(|| requested_language.map(|value| value.to_string()));

    let (aligner_model_id, words, segments) =
        if include_timestamps && !output.text.trim().is_empty() {
            if let Ok(mut callback) = progress_callback.lock() {
                callback(transcription_aligning_progress(output.duration_secs));
            }
            let resolved_aligner_model_id =
                aligner_model_id.unwrap_or(DEFAULT_TRANSCRIPTION_ALIGNER_MODEL);
            let alignments = runtime
                .force_align_bytes_with_model_and_language(
                    audio_bytes,
                    output.text.as_str(),
                    resolved_language.as_deref(),
                    Some(resolved_aligner_model_id),
                )
                .await?;
            let words = alignments_to_word_records(&alignments);
            let segments = build_segment_records(output.text.as_str(), &words);
            (Some(resolved_aligner_model_id.to_string()), words, segments)
        } else {
            (None, Vec::new(), Vec::new())
        };

    Ok(GeneratedTranscriptionArtifacts {
        text: output.text,
        language: resolved_language,
        duration_secs: output.duration_secs as f64,
        aligner_model_id,
        segments,
        words,
    })
}

fn progress_event_payload(progress: AsrProgress) -> String {
    serde_json::to_string(&StreamProgressEvent {
        event: "progress",
        progress,
    })
    .unwrap_or_default()
}

fn transcription_processing_progress() -> AsrProgress {
    AsrProgress {
        phase: AsrProgressPhase::Processing,
        current_chunk: None,
        total_chunks: None,
        processed_audio_secs: None,
        total_audio_secs: None,
        percent: None,
    }
}

fn transcription_aligning_progress(duration_secs: f32) -> AsrProgress {
    let duration_secs = if duration_secs.is_finite() && duration_secs >= 0.0 {
        Some(f64::from(duration_secs))
    } else {
        None
    };
    AsrProgress {
        phase: AsrProgressPhase::Aligning,
        current_chunk: None,
        total_chunks: None,
        processed_audio_secs: duration_secs,
        total_audio_secs: duration_secs,
        percent: Some(100.0),
    }
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
        let audio_payload = decode_base64_audio_payload(payload.audio_base64.as_str())?;
        inspect_audio_payload_with_diagnostics("transcription.create", &audio_payload)?;
        let audio_mime_type = audio_payload
            .content_type_hint()
            .map(str::to_string)
            .unwrap_or_else(|| "audio/wav".to_string());

        return Ok(ParsedTranscriptionCreateRequest {
            audio_bytes: audio_payload.bytes,
            audio_mime_type: Some(audio_mime_type),
            audio_filename: None,
            model_id,
            aligner_model_id: sanitize_optional(payload.aligner_model_id),
            language: sanitize_optional(payload.language),
            include_timestamps: payload
                .include_timestamps
                .or(payload.word_timestamps)
                .unwrap_or(false),
            stream: payload.stream.unwrap_or(false),
            generate_summary: payload.generate_summary.unwrap_or(false),
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart payload: {e}")))?;

        let mut out = ParsedTranscriptionCreateRequest::default();

        while let Some(field) = multipart.next_field().await.map_err(|err| {
            multipart_upload_error("Transcription", "field", AUDIO_UPLOAD_LIMIT_BYTES, err)
        })? {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    if let Some(payload) = read_multipart_audio_file_payload(
                        field,
                        "Transcription",
                        &name,
                        AUDIO_UPLOAD_LIMIT_BYTES,
                    )
                    .await?
                    {
                        inspect_audio_payload_with_diagnostics("transcription.create", &payload)?;
                        out.audio_filename = payload.filename;
                        out.audio_mime_type = payload.source_mime_type;
                        out.audio_bytes = payload.bytes;
                    }
                }
                "audio_base64" => {
                    if let Some(payload) = read_multipart_audio_base64_payload(
                        field,
                        "Transcription",
                        "audio_base64",
                        AUDIO_UPLOAD_LIMIT_BYTES,
                    )
                    .await?
                    {
                        inspect_audio_payload_with_diagnostics("transcription.create", &payload)?;
                        if out.audio_mime_type.is_none() {
                            out.audio_mime_type = payload.content_type_hint().map(str::to_string);
                        }
                        out.audio_bytes = payload.bytes;
                    }
                }
                "model" | "model_id" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{name}' field: {e}"
                        ))
                    })?;
                    out.model_id = sanitize_optional(Some(text));
                }
                "aligner_model" | "aligner_model_id" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{name}' field: {e}"
                        ))
                    })?;
                    out.aligner_model_id = sanitize_optional(Some(text));
                }
                "language" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'language' field: {e}"
                        ))
                    })?;
                    out.language = sanitize_optional(Some(text));
                }
                "include_timestamps" | "word_timestamps" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{name}' field: {e}"
                        ))
                    })?;
                    out.include_timestamps = parse_bool(text.as_str());
                }
                "stream" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'stream' field: {e}"
                        ))
                    })?;
                    out.stream = parse_bool(text.as_str());
                }
                "generate_summary" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'generate_summary' field: {e}"
                        ))
                    })?;
                    out.generate_summary = parse_bool(text.as_str());
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

fn parse_bool(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sanitize_optional(raw: Option<String>) -> Option<String> {
    raw.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
fn multipart_field_api_error(
    field_name: &str,
    status: StatusCode,
    raw: impl AsRef<str>,
) -> ApiError {
    crate::api::speech_text_upload::multipart_upload_api_error(
        "Transcription",
        field_name,
        AUDIO_UPLOAD_LIMIT_BYTES,
        status,
        raw,
    )
}

fn alignments_to_word_records(alignments: &[(String, u32, u32)]) -> Vec<TranscriptionWordRecord> {
    alignments
        .iter()
        .filter_map(|(word, start_ms, end_ms)| {
            if end_ms <= start_ms {
                return None;
            }

            let token = word.trim();
            if token.is_empty() {
                return None;
            }

            Some(TranscriptionWordRecord {
                word: token.to_string(),
                start: *start_ms as f32 / 1000.0,
                end: *end_ms as f32 / 1000.0,
            })
        })
        .collect()
}

fn build_segment_records(
    transcription: &str,
    words: &[TranscriptionWordRecord],
) -> Vec<TranscriptionSegmentRecord> {
    if words.is_empty() {
        return Vec::new();
    }

    let original_tokens = transcription.split_whitespace().collect::<Vec<_>>();
    let use_original_tokens = original_tokens.len() == words.len();
    let mut segments = Vec::new();
    let mut segment_start = 0usize;

    for idx in 1..=words.len() {
        let should_break = if idx >= words.len() {
            true
        } else {
            should_break_segment(
                words,
                segment_start,
                idx,
                &original_tokens,
                use_original_tokens,
            )
        };

        if !should_break {
            continue;
        }

        let segment_end = idx.saturating_sub(1);
        let start = words[segment_start].start;
        let end = words[segment_end].end;
        let text = if use_original_tokens {
            original_tokens[segment_start..=segment_end].join(" ")
        } else {
            words[segment_start..=segment_end]
                .iter()
                .map(|word| word.word.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };

        if !text.trim().is_empty() && end > start {
            segments.push(TranscriptionSegmentRecord {
                start,
                end,
                text,
                word_start: segment_start,
                word_end: segment_end,
            });
        }

        segment_start = idx;
    }

    segments
}

fn should_break_segment(
    words: &[TranscriptionWordRecord],
    segment_start: usize,
    next_index: usize,
    original_tokens: &[&str],
    use_original_tokens: bool,
) -> bool {
    let previous = &words[next_index - 1];
    let current = &words[next_index];
    let segment_duration = previous.end - words[segment_start].start;
    let segment_word_count = next_index - segment_start;
    let gap = current.start - previous.end;
    let previous_token = if use_original_tokens {
        original_tokens
            .get(next_index - 1)
            .copied()
            .unwrap_or(previous.word.as_str())
    } else {
        previous.word.as_str()
    };

    gap >= SEGMENT_GAP_BREAK_SECS
        || segment_word_count >= MAX_SEGMENT_WORDS
        || segment_duration >= MAX_SEGMENT_DURATION_SECS
        || (segment_word_count >= MIN_SENTENCE_BREAK_WORDS
            && previous_token.ends_with(['.', '!', '?']))
}

fn initial_summary_state(
    transcription: &str,
    generate_summary: bool,
) -> (TranscriptionSummaryStatus, Option<String>) {
    if !generate_summary || transcription.trim().is_empty() {
        (TranscriptionSummaryStatus::NotRequested, None)
    } else {
        (
            TranscriptionSummaryStatus::Pending,
            Some(DEFAULT_TRANSCRIPTION_SUMMARY_MODEL.to_string()),
        )
    }
}

fn maybe_spawn_summary_generation(
    runtime: Arc<RuntimeService>,
    transcription_store: Arc<TranscriptionStore>,
    semaphore: Arc<tokio::sync::Semaphore>,
    record: &TranscriptionRecord,
    correlation_id: Option<String>,
) {
    if record.summary_status != TranscriptionSummaryStatus::Pending {
        return;
    }

    spawn_summary_generation_task(
        runtime,
        transcription_store,
        semaphore,
        record.id.clone(),
        record.transcription.clone(),
        correlation_id,
    );
}

fn spawn_summary_generation_task(
    runtime: Arc<RuntimeService>,
    transcription_store: Arc<TranscriptionStore>,
    semaphore: Arc<tokio::sync::Semaphore>,
    record_id: String,
    transcription: String,
    correlation_id: Option<String>,
) {
    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => return,
        };

        let summary_result = if transcription.trim().is_empty() {
            Err("Summary generation failed: transcription is empty".to_string())
        } else {
            generate_transcription_summary(
                runtime,
                transcription.as_str(),
                correlation_id.as_deref(),
            )
            .await
        };

        let summary_update = match summary_result {
            Ok(summary_text) => UpdateTranscriptionSummary {
                status: TranscriptionSummaryStatus::Ready,
                model_id: Some(DEFAULT_TRANSCRIPTION_SUMMARY_MODEL.to_string()),
                text: Some(summary_text),
                error: None,
                updated_at: None,
            },
            Err(err) => UpdateTranscriptionSummary {
                status: TranscriptionSummaryStatus::Failed,
                model_id: Some(DEFAULT_TRANSCRIPTION_SUMMARY_MODEL.to_string()),
                text: None,
                error: Some(truncate_summary_error(err.as_str())),
                updated_at: None,
            },
        };

        if let Err(err) = transcription_store
            .update_summary(record_id.clone(), summary_update)
            .await
        {
            tracing::warn!(
                "transcription summary persist failed: record_id={} error={}",
                record_id,
                err
            );
        }
    });
}

async fn generate_transcription_summary(
    runtime: Arc<RuntimeService>,
    transcription: &str,
    correlation_id: Option<&str>,
) -> Result<String, String> {
    let variant =
        parse_chat_model_variant(Some(DEFAULT_TRANSCRIPTION_SUMMARY_MODEL)).map_err(|err| {
            format!("Invalid summary model '{DEFAULT_TRANSCRIPTION_SUMMARY_MODEL}': {err}")
        })?;

    let first = generate_transcription_summary_attempt(
        runtime.clone(),
        variant,
        transcription,
        correlation_id,
        None,
    )
    .await;
    match first {
        Ok(summary) => Ok(summary),
        Err(err) if should_retry_transcription_summary_generation(&err) => {
            tracing::warn!(
                summary_model = DEFAULT_TRANSCRIPTION_SUMMARY_MODEL,
                error = %err,
                "transcription summary sampled decode failed; retrying with thinking disabled"
            );
            generate_transcription_summary_attempt(
                runtime,
                variant,
                transcription,
                correlation_id,
                Some(false),
            )
            .await
            .map_err(|retry_err| format!("{err}; retry with thinking disabled failed: {retry_err}"))
        }
        Err(err) => Err(err),
    }
}

async fn generate_transcription_summary_attempt(
    runtime: Arc<RuntimeService>,
    variant: ModelVariant,
    transcription: &str,
    correlation_id: Option<&str>,
    enable_thinking: Option<bool>,
) -> Result<String, String> {
    let generation = runtime
        .chat_generate_with_generation_params_and_chat_config_and_correlation(
            variant,
            transcription_summary_messages(transcription),
            transcription_summary_params(),
            ChatRequestConfig {
                enable_thinking,
                tools: Vec::new(),
                media_inputs: Vec::new(),
            },
            correlation_id,
        )
        .await
        .map_err(|err| format!("Summary generation failed: {err}"))?;

    sanitize_summary_output(generation.text.as_str())
        .ok_or_else(|| "Summary generation returned empty text".to_string())
}

fn transcription_summary_params() -> GenerationParams {
    let mut params = GenerationParams::default();
    params.max_tokens = DEFAULT_TRANSCRIPTION_SUMMARY_MAX_TOKENS;
    params.temperature = 0.2;
    params.top_p = 0.9;
    params
}

fn transcription_summary_messages(transcription: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: TRANSCRIPTION_SUMMARY_SYSTEM_PROMPT.to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: format!(
                "Summarize the following transcript.\n\nTranscript:\n{}",
                transcription
            ),
        },
    ]
}

fn should_retry_transcription_summary_generation(error: &str) -> bool {
    let normalized = error.to_ascii_lowercase();
    normalized.contains("no valid qwen3.5 logits to sample")
        || normalized.contains("non-finite")
        || (normalized.contains("logit") && normalized.contains("nan"))
}

fn sanitize_summary_output(raw: &str) -> Option<String> {
    let without_think = strip_think_sections(raw);
    let without_fence_markers = strip_code_fence_markers(without_think.as_str());
    let normalized = without_fence_markers
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn strip_think_sections(input: &str) -> String {
    let mut out = if let Some((_, tail)) = input.rsplit_once("</think>") {
        tail.to_string()
    } else {
        input.to_string()
    };
    let open_tag = "<think>";
    let close_tag = "</think>";

    loop {
        let Some(start) = out.find(open_tag) else {
            break;
        };

        if let Some(end_rel) = out[start + open_tag.len()..].find(close_tag) {
            let end = start + open_tag.len() + end_rel;
            let mut next = String::with_capacity(out.len());
            next.push_str(&out[..start]);
            next.push_str(&out[end + close_tag.len()..]);
            out = next;
        } else {
            out.truncate(start);
            break;
        }
    }

    out.replace(open_tag, " ").replace(close_tag, " ")
}

fn strip_code_fence_markers(input: &str) -> String {
    input
        .lines()
        .filter(|line| !line.trim_start().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate_summary_error(raw: &str) -> String {
    const MAX_ERROR_CHARS: usize = 320;
    raw.chars().take(MAX_ERROR_CHARS).collect::<String>()
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Transcription storage error: {err}"))
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{header, StatusCode},
    };
    use base64::Engine as _;
    use izwi_core::audio::{AudioEncoder, AudioFormat};
    use izwi_core::{AsrProgress, AsrProgressPhase};

    use super::{
        alignments_to_word_records, build_segment_records, initial_summary_state,
        multipart_field_api_error, parse_bool, parse_create_request, progress_event_payload,
        sanitize_summary_output, should_retry_transcription_summary_generation,
        transcription_summary_messages, transcription_summary_params, TranscriptionSummaryStatus,
        TranscriptionWordRecord,
    };

    fn wav_bytes() -> Vec<u8> {
        AudioEncoder::new(16_000, 1)
            .encode(&[0.0, 0.1, -0.1, 0.0], AudioFormat::Wav)
            .expect("wav should encode")
    }

    fn wav_data_url(content_type: &str) -> String {
        let b64 = base64::engine::general_purpose::STANDARD.encode(wav_bytes());
        format!("data:{content_type};base64, {b64}\n")
    }

    #[tokio::test]
    async fn json_create_request_accepts_data_url_audio_and_uses_mime_hint() {
        let request = axum::http::Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "audio_base64": wav_data_url("audio/wav"),
                    "model": "Parakeet-TDT-0.6B-v3"
                })
                .to_string(),
            ))
            .expect("request should build");

        let parsed = parse_create_request(request)
            .await
            .expect("request should parse");

        assert!(!parsed.audio_bytes.is_empty());
        assert_eq!(parsed.audio_mime_type.as_deref(), Some("audio/wav"));
        assert_eq!(parsed.model_id.as_deref(), Some("Parakeet-TDT-0.6B-v3"));
        assert!(!parsed.generate_summary);
    }

    #[tokio::test]
    async fn json_create_request_accepts_generate_summary_true() {
        let request = axum::http::Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "audio_base64": wav_data_url("audio/wav"),
                    "model": "Parakeet-TDT-0.6B-v3",
                    "generate_summary": true
                })
                .to_string(),
            ))
            .expect("request should build");

        let parsed = parse_create_request(request)
            .await
            .expect("request should parse");

        assert!(parsed.generate_summary);
    }

    #[tokio::test]
    async fn multipart_create_request_accepts_generate_summary_true() {
        let boundary = "izwi-transcription-boundary";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(&wav_bytes());
        body.extend_from_slice(
            format!(
                "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"generate_summary\"\r\n\r\ntrue\r\n--{boundary}--\r\n"
            )
            .as_bytes(),
        );
        let request = axum::http::Request::builder()
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .expect("request should build");

        let parsed = parse_create_request(request)
            .await
            .expect("request should parse");

        assert!(parsed.generate_summary);
    }

    #[tokio::test]
    async fn json_create_request_rejects_undecodable_audio_before_queueing() {
        let request = axum::http::Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "audio_base64": "YXVkaW8=",
                    "model": "Parakeet-TDT-0.6B-v3"
                })
                .to_string(),
            ))
            .expect("request should build");

        let err = parse_create_request(request)
            .await
            .expect_err("undecodable audio should fail before queueing");

        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("failed to decode audio metadata"));
    }

    #[test]
    fn converts_alignment_millis_into_word_records() {
        let words = alignments_to_word_records(&[
            ("Hello".to_string(), 0, 450),
            ("there".to_string(), 500, 900),
            ("".to_string(), 950, 1200),
        ]);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[1].end, 0.9);
    }

    #[test]
    fn builds_segments_from_word_timings() {
        let words = vec![
            TranscriptionWordRecord {
                word: "Hello".to_string(),
                start: 0.0,
                end: 0.4,
            },
            TranscriptionWordRecord {
                word: "there.".to_string(),
                start: 0.45,
                end: 0.9,
            },
            TranscriptionWordRecord {
                word: "General".to_string(),
                start: 1.9,
                end: 2.4,
            },
            TranscriptionWordRecord {
                word: "Kenobi.".to_string(),
                start: 2.45,
                end: 3.0,
            },
        ];

        let segments = build_segment_records("Hello there. General Kenobi.", &words);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "Hello there.");
        assert_eq!(segments[0].word_start, 0);
        assert_eq!(segments[1].text, "General Kenobi.");
    }

    #[test]
    fn parses_truthy_boolean_values() {
        assert!(parse_bool("true"));
        assert!(parse_bool("YES"));
        assert!(parse_bool("1"));
        assert!(!parse_bool("false"));
    }

    #[test]
    fn progress_event_payload_uses_sse_contract_shape() {
        let payload = progress_event_payload(AsrProgress {
            phase: AsrProgressPhase::ChunkFinished,
            current_chunk: Some(1),
            total_chunks: Some(2),
            processed_audio_secs: Some(3.0),
            total_audio_secs: Some(6.0),
            percent: Some(50.0),
        });
        let value: serde_json::Value =
            serde_json::from_str(payload.as_str()).expect("valid progress payload");

        assert_eq!(value["event"], "progress");
        assert_eq!(value["progress"]["phase"], "chunk_finished");
        assert_eq!(value["progress"]["current_chunk"], 1);
        assert_eq!(value["progress"]["total_chunks"], 2);
        assert_eq!(value["progress"]["percent"], 50.0);
    }

    #[test]
    fn multipart_error_mentions_transcription_upload_limit() {
        let err = multipart_field_api_error(
            "file",
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body too large",
        );
        assert_eq!(err.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(err.message.contains("64 MiB"));
        assert!(err.message.contains("original compressed file"));
    }

    #[test]
    fn defaults_summary_state_to_not_requested_for_non_empty_transcriptions() {
        let (status, model_id) = initial_summary_state("hello world", false);
        assert_eq!(status, TranscriptionSummaryStatus::NotRequested);
        assert_eq!(model_id, None);

        let (status, model_id) = initial_summary_state("hello world", true);
        assert_eq!(status, TranscriptionSummaryStatus::Pending);
        assert_eq!(model_id.as_deref(), Some("Qwen3.5-4B"));

        let (status, model_id) = initial_summary_state("   ", true);
        assert_eq!(status, TranscriptionSummaryStatus::NotRequested);
        assert_eq!(model_id, None);
    }

    #[test]
    fn sanitizes_summary_output_for_display_and_storage() {
        let raw = "<think>reasoning</think>\n```text\nThis is the summary.\n```\n";
        assert_eq!(
            sanitize_summary_output(raw).as_deref(),
            Some("This is the summary.")
        );

        let close_only = "planning first</think>\nFinal summary";
        assert_eq!(
            sanitize_summary_output(close_only).as_deref(),
            Some("Final summary")
        );
    }

    #[test]
    fn transcription_summary_uses_sampled_qwen35_settings() {
        let params = transcription_summary_params();
        assert_eq!(params.max_tokens, 384);
        assert_eq!(params.temperature, 0.2);
        assert_eq!(params.top_p, 0.9);

        let messages = transcription_summary_messages("hello world");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, izwi_core::ChatRole::System);
        assert!(messages[1].content.contains("hello world"));
    }

    #[test]
    fn transcription_summary_retries_invalid_qwen35_logits() {
        assert!(should_retry_transcription_summary_generation(
            "Summary generation failed: Inference error: No valid Qwen3.5 logits to sample"
        ));
        assert!(should_retry_transcription_summary_generation(
            "Summary generation failed: non-finite logits"
        ));
        assert!(should_retry_transcription_summary_generation(
            "Summary generation failed: logits contained NaN values"
        ));
        assert!(!should_retry_transcription_summary_generation(
            "Summary generation failed: Model not found"
        ));
        assert!(!should_retry_transcription_summary_generation(
            "Summary generation failed: missing banana model"
        ));
    }
}
