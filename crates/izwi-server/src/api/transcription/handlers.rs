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
    TranscriptionRecordSummary, TranscriptionSegmentRecord, TranscriptionWordRecord,
};

const HISTORY_LIST_LIMIT: usize = 200;
const DEFAULT_TRANSCRIPTION_ALIGNER_MODEL: &str = "Qwen3-ForcedAligner-0.6B";
const MAX_SEGMENT_WORDS: usize = 18;
const MAX_SEGMENT_DURATION_SECS: f32 = 9.0;
const MIN_SENTENCE_BREAK_WORDS: usize = 5;
const SEGMENT_GAP_BREAK_SECS: f32 = 0.85;

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
    aligner_model_id: Option<String>,
    language: Option<String>,
    include_timestamps: bool,
    stream: bool,
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

#[derive(Debug)]
struct GeneratedTranscriptionArtifacts {
    text: String,
    language: Option<String>,
    duration_secs: f64,
    aligner_model_id: Option<String>,
    segments: Vec<TranscriptionSegmentRecord>,
    words: Vec<TranscriptionWordRecord>,
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

    let artifacts = generate_transcription_artifacts(
        state.runtime.clone(),
        parsed.audio_bytes.as_slice(),
        parsed.model_id.as_deref(),
        parsed.aligner_model_id.as_deref(),
        parsed.language.as_deref(),
        parsed.include_timestamps,
        Some(&ctx.correlation_id),
        |_delta| {},
    )
    .await?;

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let rtf = if artifacts.duration_secs > 0.0 {
        Some((elapsed_ms / 1000.0) / artifacts.duration_secs)
    } else {
        None
    };

    let record = state
        .transcription_store
        .create_record(NewTranscriptionRecord {
            model_id: parsed.model_id,
            aligner_model_id: artifacts.aligner_model_id,
            language: artifacts.language,
            duration_secs: Some(artifacts.duration_secs),
            processing_time_ms: elapsed_ms,
            rtf,
            audio_mime_type: parsed
                .audio_mime_type
                .unwrap_or_else(|| "audio/wav".to_string()),
            audio_filename: parsed.audio_filename,
            audio_bytes: parsed.audio_bytes,
            transcription: artifacts.text,
            segments: artifacts.segments,
            words: artifacts.words,
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
    let aligner_model_id = parsed.aligner_model_id;
    let requested_language = parsed.language;
    let include_timestamps = parsed.include_timestamps;
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
            generate_transcription_artifacts(
                runtime,
                audio_bytes.as_slice(),
                model_id.as_deref(),
                aligner_model_id.as_deref(),
                requested_language.as_deref(),
                include_timestamps,
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
            Ok(Ok(artifacts)) => {
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let rtf = if artifacts.duration_secs > 0.0 {
                    Some((elapsed_ms / 1000.0) / artifacts.duration_secs)
                } else {
                    None
                };

                match transcription_store
                    .create_record(NewTranscriptionRecord {
                        model_id: model_id.clone(),
                        aligner_model_id: artifacts.aligner_model_id,
                        language: artifacts.language,
                        duration_secs: Some(artifacts.duration_secs),
                        processing_time_ms: elapsed_ms,
                        rtf,
                        audio_mime_type,
                        audio_filename,
                        audio_bytes,
                        transcription: artifacts.text,
                        segments: artifacts.segments,
                        words: artifacts.words,
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
                        error: err.message,
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

async fn generate_transcription_artifacts<F>(
    runtime: std::sync::Arc<izwi_core::RuntimeService>,
    audio_bytes: &[u8],
    model_id: Option<&str>,
    aligner_model_id: Option<&str>,
    requested_language: Option<&str>,
    include_timestamps: bool,
    correlation_id: Option<&str>,
    on_delta: F,
) -> Result<GeneratedTranscriptionArtifacts, ApiError>
where
    F: FnMut(String) + Send + 'static,
{
    let output = runtime
        .asr_transcribe_streaming_bytes_with_correlation(
            audio_bytes,
            model_id,
            requested_language,
            correlation_id,
            on_delta,
        )
        .await?;

    let resolved_language = output
        .language
        .clone()
        .or_else(|| requested_language.map(|value| value.to_string()));

    let (aligner_model_id, words, segments) =
        if include_timestamps && !output.text.trim().is_empty() {
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
            aligner_model_id: sanitize_optional(payload.aligner_model_id),
            language: sanitize_optional(payload.language),
            include_timestamps: payload
                .include_timestamps
                .or(payload.word_timestamps)
                .unwrap_or(false),
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

fn sanitize_optional(raw: Option<String>) -> Option<String> {
    raw.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
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

#[cfg(test)]
mod tests {
    use super::{
        alignments_to_word_records, build_segment_records, parse_bool, TranscriptionWordRecord,
    };

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
}
