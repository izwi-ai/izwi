//! OpenAI-compatible transcription endpoints.

use axum::{
    Json, RequestExt,
    body::Body,
    extract::{Extension, Multipart, Request, State},
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use base64::Engine;
use std::convert::Infallible;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, info};

use super::resolve_audio_upload_limit_bytes;
use crate::api::request_context::RequestContext;
use crate::api::speech_text_upload::multipart_upload_api_error;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::parse_model_variant;

#[derive(Debug, Default)]
struct TranscriptionRequest {
    audio_base64: Option<String>,
    model: Option<String>,
    aligner_model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    stream: bool,
    prompt: Option<String>,
    _temperature: Option<f32>,
    timestamp_granularities: Option<Vec<String>>,
}

#[derive(Debug, serde::Serialize)]
struct JsonTranscriptionResponse {
    text: String,
}

#[derive(Debug, serde::Serialize)]
struct VerboseJsonTranscriptionResponse {
    text: String,
    language: Option<String>,
    duration: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<TranscriptionTimestampWord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<Vec<TranscriptionTimestampSegment>>,
    processing_time_ms: f64,
    rtf: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_asr_diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TranscriptionTimestampWord {
    word: String,
    start: f32,
    end: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TranscriptionTimestampSegment {
    id: usize,
    start: f32,
    end: f32,
    text: String,
}

const TIMESTAMP_SEGMENT_MAX_DURATION_SECS: f32 = 8.0;
const TIMESTAMP_SEGMENT_MAX_CHARS: usize = 84;
const TIMESTAMP_SEGMENT_LONG_PAUSE_SECS: f32 = 0.8;
const TIMESTAMP_SEGMENT_MIN_PUNCTUATION_SECS: f32 = 1.0;
const TIMESTAMP_SEGMENT_BOUNDARY_EPS: f32 = 0.001;

#[derive(Debug, Default, Clone, Copy)]
struct TimestampGranularityRequest {
    words: bool,
    segments: bool,
}

impl TimestampGranularityRequest {
    fn parse(values: Option<&[String]>) -> Result<Self, ApiError> {
        let mut request = Self::default();
        for value in values.unwrap_or_default() {
            match value.trim().to_ascii_lowercase().as_str() {
                "" => {}
                "word" | "words" => request.words = true,
                "segment" | "segments" => request.segments = true,
                other => {
                    return Err(ApiError::bad_request(format!(
                        "Unsupported timestamp granularity: {}. Supported: word, segment",
                        other
                    )));
                }
            }
        }
        Ok(request)
    }

    fn any(self) -> bool {
        self.words || self.segments
    }
}

pub async fn transcriptions(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    req: Request,
) -> Result<Response<Body>, ApiError> {
    let mut req = parse_transcription_request(req).await?;
    validate_transcription_model(req.model.as_deref())?;
    let audio_base64 = req
        .audio_base64
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;

    info!("OpenAI transcription request: {} bytes", audio_base64.len());

    if req.stream {
        return transcriptions_stream(state, req, audio_base64, ctx.correlation_id).await;
    }

    let response_format = req
        .response_format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();
    let timestamp_request =
        TimestampGranularityRequest::parse(req.timestamp_granularities.as_deref())?;
    if timestamp_request.any() && response_format != "verbose_json" {
        return Err(ApiError::bad_request(
            "`timestamp_granularities` requires `response_format: \"verbose_json\"`",
        ));
    }

    let _permit = state.acquire_permit().await;
    let started = Instant::now();
    let output = state
        .runtime
        .asr_transcribe_with_prompt_and_correlation(
            &audio_base64,
            req.model.as_deref(),
            req.language.as_deref(),
            req.prompt.as_deref(),
            Some(&ctx.correlation_id),
        )
        .await?;

    let mut words = None;
    let mut segments = None;
    let mut subtitle_segments = None;
    let chunk_boundaries = extract_chunk_segment_boundaries(output.asr_diagnostics.as_ref());
    if timestamp_request.any() && !output.text.trim().is_empty() {
        let alignments = state
            .runtime
            .force_align_with_model_and_language(
                &audio_base64,
                output.text.as_str(),
                output.language.as_deref().or(req.language.as_deref()),
                req.aligner_model.as_deref(),
            )
            .await?;
        let timestamp_words = alignments_to_timestamp_words(alignments);
        if timestamp_request.words {
            words = Some(timestamp_words.clone());
        }
        if timestamp_request.segments {
            segments = Some(build_timestamp_segments(
                &timestamp_words,
                &chunk_boundaries,
            ));
        }
    } else if matches!(response_format.as_str(), "srt" | "vtt") && !output.text.trim().is_empty() {
        match state
            .runtime
            .force_align_with_model_and_language(
                &audio_base64,
                output.text.as_str(),
                output.language.as_deref().or(req.language.as_deref()),
                req.aligner_model.as_deref(),
            )
            .await
        {
            Ok(alignments) => {
                let timestamp_words = alignments_to_timestamp_words(alignments);
                let built_segments = build_timestamp_segments(&timestamp_words, &chunk_boundaries);
                if !built_segments.is_empty() {
                    subtitle_segments = Some(built_segments);
                }
            }
            Err(err) => {
                debug!(
                    "Falling back to coarse transcription subtitle timestamps: {}",
                    err
                );
            }
        }
    }

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let rtf = if output.duration_secs > 0.0 {
        Some((elapsed_ms / 1000.0) / output.duration_secs as f64)
    } else {
        None
    };

    match response_format.as_str() {
        "json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&JsonTranscriptionResponse { text: output.text }).unwrap(),
            ))
            .unwrap()),
        "verbose_json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&VerboseJsonTranscriptionResponse {
                    text: output.text,
                    language: output.language,
                    duration: output.duration_secs,
                    words,
                    segments,
                    processing_time_ms: elapsed_ms,
                    rtf,
                    izwi_asr_diagnostics: output.asr_diagnostics,
                })
                .unwrap(),
            ))
            .unwrap()),
        "text" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(output.text))
            .unwrap()),
        "srt" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(format_srt(
                &output.text,
                output.duration_secs,
                subtitle_segments.as_deref(),
            )))
            .unwrap()),
        "vtt" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/vtt; charset=utf-8")
            .body(Body::from(format_vtt(
                &output.text,
                output.duration_secs,
                subtitle_segments.as_deref(),
            )))
            .unwrap()),
        other => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Supported: json, verbose_json, text, srt, vtt",
            other
        ))),
    }
}

async fn transcriptions_stream(
    state: AppState,
    req: TranscriptionRequest,
    audio_base64: String,
    correlation_id: String,
) -> Result<Response<Body>, ApiError> {
    let model = req.model;
    let language = req.language;
    let prompt = req.prompt;

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let engine = state.runtime.clone();
    let semaphore = state.request_semaphore.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(transcript_error_event_payload("Server is shutting down"));
                return;
            }
        };

        let delta_tx = event_tx.clone();
        // Keep transcription streaming unbounded by wall-clock timeout so valid
        // long jobs are not cut off mid-flight.
        let result = engine
            .asr_transcribe_streaming_with_prompt_and_correlation(
                &audio_base64,
                model.as_deref(),
                language.as_deref(),
                prompt.as_deref(),
                Some(correlation_id.as_str()),
                move |delta| {
                    let _ = delta_tx.send(transcript_delta_event_payload(delta));
                },
            )
            .await;

        match result {
            Ok(output) => {
                let _ = event_tx.send(transcript_done_event_payload(
                    output.text,
                    output.language,
                    output.duration_secs,
                ));
            }
            Err(err) => {
                let _ = event_tx.send(transcript_error_event_payload(&err.to_string()));
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(Event::default().data(payload));
        }
    };

    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );
    response
        .headers_mut()
        .insert("x-accel-buffering", header::HeaderValue::from_static("no"));
    Ok(response)
}

#[derive(Debug, serde::Deserialize)]
struct JsonRequestBody {
    audio_base64: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    #[serde(alias = "aligner_model_id")]
    aligner_model: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    timestamp_granularities: Option<Vec<String>>,
}

async fn parse_transcription_request(req: Request) -> Result<TranscriptionRequest, ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let Json(payload) = req
            .extract::<Json<JsonRequestBody>, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid JSON payload: {e}")))?;

        return Ok(TranscriptionRequest {
            audio_base64: Some(payload.audio_base64),
            model: payload.model,
            aligner_model: payload.aligner_model,
            language: payload.language,
            response_format: payload.response_format,
            stream: payload.stream.unwrap_or(false),
            prompt: payload.prompt,
            _temperature: payload.temperature,
            timestamp_granularities: payload.timestamp_granularities,
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart payload: {e}")))?;

        let mut out = TranscriptionRequest::default();

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| multipart_field_error("field", e))?
        {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let bytes = field
                        .bytes()
                        .await
                        .map_err(|e| multipart_field_error(&name, e))?;
                    if !bytes.is_empty() {
                        out.audio_base64 =
                            Some(base64::engine::general_purpose::STANDARD.encode(&bytes));
                    }
                }
                "audio_base64" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'audio_base64' field: {}",
                            e
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.audio_base64 = Some(text);
                    }
                }
                "model" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'model' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.model = Some(text.trim().to_string());
                    }
                }
                "aligner_model" | "aligner_model_id" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {e}",
                            name
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.aligner_model = Some(text.trim().to_string());
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
                "response_format" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'response_format' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.response_format = Some(text.trim().to_string());
                    }
                }
                "prompt" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'prompt' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.prompt = Some(text.trim().to_string());
                    }
                }
                "temperature" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'temperature' field: {e}"
                        ))
                    })?;
                    out._temperature = text.trim().parse::<f32>().ok();
                }
                "timestamp_granularities[]" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'timestamp_granularities[]' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.timestamp_granularities
                            .get_or_insert_with(Vec::new)
                            .push(text.trim().to_string());
                    }
                }
                "stream" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'stream' field: {e}"
                        ))
                    })?;
                    out.stream = matches!(
                        text.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    );
                }
                _ => {}
            }
        }

        return Ok(out);
    }

    Err(ApiError {
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        message: "Expected `Content-Type: application/json` or `multipart/form-data`".to_string(),
    })
}

fn multipart_field_error(
    field_name: &str,
    err: axum::extract::multipart::MultipartError,
) -> ApiError {
    multipart_field_api_error(field_name, err.status(), err.body_text())
}

fn multipart_field_api_error(
    field_name: &str,
    status: StatusCode,
    raw: impl AsRef<str>,
) -> ApiError {
    multipart_upload_api_error(
        "OpenAI-compatible audio",
        field_name,
        resolve_audio_upload_limit_bytes(),
        status,
        raw,
    )
}

fn validate_transcription_model(model: Option<&str>) -> Result<(), ApiError> {
    let Some(model_id) = model else {
        return Ok(());
    };
    let trimmed = model_id.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let variant = parse_model_variant(trimmed).map_err(|_| {
        ApiError::bad_request(format!("Unsupported transcription model: {}", trimmed))
    })?;

    if variant.is_asr() || variant.is_voxtral() || variant.is_audio_chat() {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "Unsupported transcription model: {}",
            trimmed
        )))
    }
}

fn alignments_to_timestamp_words(
    alignments: Vec<(String, u32, u32)>,
) -> Vec<TranscriptionTimestampWord> {
    alignments
        .into_iter()
        .map(|(word, start_ms, end_ms)| TranscriptionTimestampWord {
            word,
            start: millis_to_secs(start_ms),
            end: millis_to_secs(end_ms),
        })
        .collect()
}

fn build_timestamp_segments(
    words: &[TranscriptionTimestampWord],
    chunk_boundaries: &[f32],
) -> Vec<TranscriptionTimestampSegment> {
    let mut segments = Vec::new();
    let mut current_words = Vec::<String>::new();
    let mut current_start: Option<f32> = None;

    for (idx, word) in words.iter().enumerate() {
        let trimmed = word.word.trim();
        if trimmed.is_empty() {
            continue;
        }
        let start = word.start.max(0.0);
        let end = word.end.max(start);
        let segment_start = *current_start.get_or_insert(start);
        current_words.push(trimmed.to_string());

        let next = words.get(idx + 1);
        let pause_after = next
            .map(|next_word| (next_word.start - end).max(0.0))
            .unwrap_or(0.0);
        let current_text_len = current_words
            .iter()
            .map(|part| part.len() + 1)
            .sum::<usize>();
        let duration = end - segment_start;
        let is_last = next.is_none();
        let crosses_chunk_boundary = next
            .map(|next_word| {
                timestamp_segment_crosses_chunk_boundary(
                    segment_start,
                    end,
                    next_word.start,
                    chunk_boundaries,
                )
            })
            .unwrap_or(false);
        let should_close = is_last
            || crosses_chunk_boundary
            || pause_after >= TIMESTAMP_SEGMENT_LONG_PAUSE_SECS
            || current_text_len >= TIMESTAMP_SEGMENT_MAX_CHARS
            || duration >= TIMESTAMP_SEGMENT_MAX_DURATION_SECS
            || (duration >= TIMESTAMP_SEGMENT_MIN_PUNCTUATION_SECS && ends_sentence_like(trimmed));

        if should_close {
            push_timestamp_segment(&mut segments, segment_start, end, &current_words);
            current_words.clear();
            current_start = None;
        }
    }

    segments
}

fn timestamp_segment_crosses_chunk_boundary(
    segment_start: f32,
    word_end: f32,
    next_word_start: f32,
    chunk_boundaries: &[f32],
) -> bool {
    chunk_boundaries.iter().any(|boundary| {
        boundary.is_finite()
            && *boundary > segment_start + TIMESTAMP_SEGMENT_BOUNDARY_EPS
            && (*boundary <= word_end + TIMESTAMP_SEGMENT_BOUNDARY_EPS
                || *boundary <= next_word_start + TIMESTAMP_SEGMENT_BOUNDARY_EPS)
    })
}

fn extract_chunk_segment_boundaries(diagnostics: Option<&serde_json::Value>) -> Vec<f32> {
    let Some(diagnostics) = diagnostics else {
        return Vec::new();
    };
    let Some(chunking) = diagnostics
        .get("chunking")
        .or_else(|| diagnostics.get("model_diagnostics")?.get("chunking"))
    else {
        return Vec::new();
    };
    let Some(chunks) = chunking.get("chunks").and_then(|chunks| chunks.as_array()) else {
        return Vec::new();
    };
    let sample_rate = chunking
        .get("sample_rate")
        .and_then(|value| value.as_f64())
        .filter(|sample_rate| sample_rate.is_finite() && *sample_rate > 0.0);
    let mut starts = chunks
        .iter()
        .filter_map(|chunk| {
            chunk
                .get("start_seconds")
                .and_then(|value| value.as_f64())
                .or_else(|| {
                    let sample_rate = sample_rate?;
                    let start_sample = chunk.get("start_sample")?.as_u64()?;
                    Some(start_sample as f64 / sample_rate)
                })
        })
        .filter(|start| start.is_finite() && *start > 0.0)
        .map(|start| start as f32)
        .collect::<Vec<_>>();

    starts.sort_by(|left, right| left.total_cmp(right));
    starts.dedup_by(|left, right| (*left - *right).abs() <= TIMESTAMP_SEGMENT_BOUNDARY_EPS);
    starts
}

fn push_timestamp_segment(
    segments: &mut Vec<TranscriptionTimestampSegment>,
    start: f32,
    end: f32,
    words: &[String],
) {
    let text = words.join(" ").trim().to_string();
    if text.is_empty() {
        return;
    }
    segments.push(TranscriptionTimestampSegment {
        id: segments.len(),
        start,
        end: end.max(start + 0.01),
        text,
    });
}

fn ends_sentence_like(text: &str) -> bool {
    text.chars()
        .rev()
        .find(|ch| !matches!(ch, '"' | '\'' | ')' | ']' | '}' | '”' | '’'))
        .map(|ch| matches!(ch, '.' | '!' | '?' | '。' | '！' | '？'))
        .unwrap_or(false)
}

fn millis_to_secs(ms: u32) -> f32 {
    (ms as f32) / 1000.0
}

fn transcript_delta_event_payload(delta: String) -> String {
    serde_json::json!({
        "type": "transcript.text.delta",
        "delta": delta,
    })
    .to_string()
}

fn transcript_done_event_payload(
    text: String,
    language: Option<String>,
    duration_secs: f32,
) -> String {
    serde_json::json!({
        "type": "transcript.text.done",
        "text": text,
        "language": language,
        "audio_duration_secs": duration_secs,
    })
    .to_string()
}

fn transcript_error_event_payload(message: &str) -> String {
    serde_json::json!({
        "type": "error",
        "error": {
            "message": message
        }
    })
    .to_string()
}

fn format_srt(
    text: &str,
    duration_secs: f32,
    segments: Option<&[TranscriptionTimestampSegment]>,
) -> String {
    if let Some(segments) = segments.filter(|segments| !segments.is_empty()) {
        return segments
            .iter()
            .enumerate()
            .map(|(idx, segment)| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    idx + 1,
                    secs_to_srt(segment.start),
                    secs_to_srt(segment.end),
                    segment.text.trim()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
    }

    format!(
        "1\n{} --> {}\n{}\n",
        secs_to_srt(0.0),
        secs_to_srt(duration_secs.max(0.1)),
        text.trim()
    )
}

fn format_vtt(
    text: &str,
    duration_secs: f32,
    segments: Option<&[TranscriptionTimestampSegment]>,
) -> String {
    if let Some(segments) = segments.filter(|segments| !segments.is_empty()) {
        let cues = segments
            .iter()
            .map(|segment| {
                format!(
                    "{} --> {}\n{}\n",
                    secs_to_vtt(segment.start),
                    secs_to_vtt(segment.end),
                    segment.text.trim()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        return format!("WEBVTT\n\n{cues}");
    }

    format!(
        "WEBVTT\n\n{} --> {}\n{}\n",
        secs_to_vtt(0.0),
        secs_to_vtt(duration_secs.max(0.1)),
        text.trim()
    )
}

fn secs_to_srt(secs: f32) -> String {
    let total_ms = (secs.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_sec = total_ms / 1000;
    let s = total_sec % 60;
    let total_min = total_sec / 60;
    let m = total_min % 60;
    let h = total_min / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn secs_to_vtt(secs: f32) -> String {
    secs_to_srt(secs).replace(',', ".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_srt_and_vtt() {
        let srt = format_srt("hello", 1.23, None);
        let vtt = format_vtt("hello", 1.23, None);
        assert!(srt.contains("-->"));
        assert!(vtt.starts_with("WEBVTT"));
    }

    #[test]
    fn stream_delta_event_uses_openai_type_field() {
        let payload = transcript_delta_event_payload("hello".to_string());
        let json: serde_json::Value = serde_json::from_str(&payload).expect("json payload");
        assert_eq!(
            json.get("type").and_then(|v| v.as_str()),
            Some("transcript.text.delta")
        );
        assert_eq!(json.get("delta").and_then(|v| v.as_str()), Some("hello"));
    }

    #[test]
    fn multipart_error_mentions_configured_limit() {
        let expected_limit = resolve_audio_upload_limit_bytes() / (1024 * 1024);
        let err = multipart_field_api_error(
            "file",
            StatusCode::BAD_REQUEST,
            "multipart/form-data field parse failed",
        );
        assert!(err.message.contains(&format!("{expected_limit} MiB")));
    }

    #[test]
    fn multipart_error_preserves_payload_too_large_status() {
        let err = multipart_field_api_error(
            "file",
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body too large",
        );
        assert_eq!(err.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(err.message.contains("original compressed file"));
    }

    #[test]
    fn transcription_model_validation_accepts_known_asr_model() {
        validate_transcription_model(Some("Parakeet-TDT-0.6B-v3"))
            .expect("known ASR model should be accepted");
    }

    #[test]
    fn transcription_model_validation_accepts_voxtral() {
        validate_transcription_model(Some("mistralai/Voxtral-Mini-4B-Realtime-2602"))
            .expect("Voxtral should be accepted for offline transcription");
    }

    #[test]
    fn transcription_model_validation_rejects_tts_or_chat_model() {
        let tts = validate_transcription_model(Some("Kokoro-82M"))
            .expect_err("TTS model should be rejected for transcription");
        assert!(tts.message.contains("Unsupported transcription model"));

        let chat = validate_transcription_model(Some("Qwen3-8B-GGUF"))
            .expect_err("chat model should be rejected for transcription");
        assert!(chat.message.contains("Unsupported transcription model"));
    }

    #[test]
    fn transcription_model_validation_rejects_unknown_model() {
        let err = validate_transcription_model(Some("definitely-not-a-real-model"))
            .expect_err("unknown model should be rejected");
        assert!(err.message.contains("Unsupported transcription model"));
    }

    #[test]
    fn parses_timestamp_granularities() {
        let requested = vec!["word".to_string(), "segments".to_string()];
        let parsed =
            TimestampGranularityRequest::parse(Some(&requested)).expect("supported granularities");
        assert!(parsed.words);
        assert!(parsed.segments);
    }

    #[test]
    fn rejects_unknown_timestamp_granularity() {
        let requested = vec!["phoneme".to_string()];
        let err = TimestampGranularityRequest::parse(Some(&requested))
            .expect_err("unknown granularity should be rejected");
        assert!(err.message.contains("Unsupported timestamp granularity"));
    }

    #[test]
    fn converts_alignment_millis_into_verbose_timestamp_words() {
        let words = alignments_to_timestamp_words(vec![("hello".to_string(), 250, 900)]);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "hello");
        assert!((words[0].start - 0.25).abs() < f32::EPSILON);
        assert!((words[0].end - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn groups_verbose_timestamp_segments_by_pause() {
        let words = vec![
            TranscriptionTimestampWord {
                word: "hello".to_string(),
                start: 0.1,
                end: 0.4,
            },
            TranscriptionTimestampWord {
                word: "world".to_string(),
                start: 0.5,
                end: 0.9,
            },
            TranscriptionTimestampWord {
                word: "again".to_string(),
                start: 2.0,
                end: 2.4,
            },
        ];
        let segments = build_timestamp_segments(&words, &[]);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].id, 0);
        assert_eq!(segments[0].text, "hello world");
        assert!((segments[0].start - 0.1).abs() < f32::EPSILON);
        assert!((segments[0].end - 0.9).abs() < f32::EPSILON);
        assert_eq!(segments[1].id, 1);
        assert_eq!(segments[1].text, "again");
    }

    #[test]
    fn groups_verbose_timestamp_segments_by_punctuation() {
        let words = vec![
            TranscriptionTimestampWord {
                word: "Hello.".to_string(),
                start: 0.0,
                end: 1.1,
            },
            TranscriptionTimestampWord {
                word: "World".to_string(),
                start: 1.2,
                end: 1.6,
            },
        ];
        let segments = build_timestamp_segments(&words, &[]);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "Hello.");
        assert_eq!(segments[1].text, "World");
    }

    #[test]
    fn groups_verbose_timestamp_segments_by_chunk_boundary() {
        let words = vec![
            TranscriptionTimestampWord {
                word: "hello".to_string(),
                start: 0.1,
                end: 0.4,
            },
            TranscriptionTimestampWord {
                word: "world".to_string(),
                start: 0.5,
                end: 0.8,
            },
            TranscriptionTimestampWord {
                word: "again".to_string(),
                start: 1.05,
                end: 1.4,
            },
        ];

        let diagnostics = serde_json::json!({
            "chunking": {
                "sample_rate": 16000,
                "chunks": [
                    { "start_seconds": 0.0, "end_seconds": 1.0 },
                    { "start_seconds": 1.0, "end_seconds": 2.0 }
                ]
            }
        });
        let chunk_boundaries = extract_chunk_segment_boundaries(Some(&diagnostics));
        let segments = build_timestamp_segments(&words, &chunk_boundaries);

        assert_eq!(chunk_boundaries, vec![1.0]);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "hello world");
        assert_eq!(segments[1].text, "again");
    }

    #[test]
    fn extracts_chunk_boundaries_from_sample_offsets() {
        let diagnostics = serde_json::json!({
            "chunking": {
                "sample_rate": 16000,
                "chunks": [
                    { "start_sample": 0, "end_sample": 16000 },
                    { "start_sample": 16000, "end_sample": 32000 },
                    { "start_sample": 32000, "end_sample": 48000 }
                ]
            }
        });

        let chunk_boundaries = extract_chunk_segment_boundaries(Some(&diagnostics));

        assert_eq!(chunk_boundaries, vec![1.0, 2.0]);
    }

    #[test]
    fn renders_multi_cue_srt_and_vtt_from_segments() {
        let segments = vec![
            TranscriptionTimestampSegment {
                id: 0,
                start: 0.1,
                end: 0.9,
                text: "hello world".to_string(),
            },
            TranscriptionTimestampSegment {
                id: 1,
                start: 2.0,
                end: 2.4,
                text: "again".to_string(),
            },
        ];

        let srt = format_srt("fallback", 3.0, Some(&segments));
        let vtt = format_vtt("fallback", 3.0, Some(&segments));

        assert!(srt.contains("1\n00:00:00,100 --> 00:00:00,900"));
        assert!(srt.contains("2\n00:00:02,000 --> 00:00:02,400"));
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("00:00:02.000 --> 00:00:02.400"));
    }
}
