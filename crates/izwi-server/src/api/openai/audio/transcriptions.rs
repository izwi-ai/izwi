//! OpenAI-compatible transcription endpoints.

use axum::{
    body::Body,
    extract::{Extension, Multipart, Request, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json, RequestExt,
};
use std::convert::Infallible;
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, Notify};
use tracing::{debug, info, warn};

use super::resolve_audio_upload_limit_bytes;
use crate::api::audio_payload::{
    decode_base64_audio_payload, inspect_audio_payload_canonical_with_diagnostics,
    read_multipart_audio_base64_payload, read_multipart_audio_file_payload, AudioPayload,
};
use crate::api::request_context::RequestContext;
use crate::api::speech_text_upload::multipart_upload_api_error;
use crate::api::transcript_alignment::align_transcript;
use crate::error::{core_api_error, ApiError};
use crate::state::AppState;
use izwi_core::{
    parse_model_variant, resolve_asr_model_variant, validate_asr_language, AlignmentSource,
    AsrTranscription, WindowedAlignment, WorkloadClass,
};

const TRANSCRIPTION_STREAM_CAPACITY: usize = 64;
#[cfg(not(test))]
const TRANSCRIPTION_TERMINAL_SEND_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const TRANSCRIPTION_TERMINAL_SEND_TIMEOUT: Duration = Duration::from_millis(100);

/// Transcript text the SSE queue could not take yet.
///
/// The runtime calls the delta callback synchronously and can hand over a
/// whole committed outbox (one delta per decoded token) within one poll, far
/// faster than the SSE body task is scheduled, so a burst overruns the bounded
/// queue even while the client reads continuously. Text that does not fit
/// waits here and goes out as one coalesced delta once the queue has room:
/// the queue stays bounded, the backlog is bounded by the transcript itself,
/// generation never waits on the client, and the deltas still concatenate to
/// the final text.
#[derive(Debug, Default)]
struct PendingTranscriptText {
    text: Mutex<String>,
    notify: Notify,
}

impl PendingTranscriptText {
    fn lock(&self) -> MutexGuard<'_, String> {
        self.text.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Send `delta` now if nothing is waiting ahead of it and the queue has
    /// room; otherwise append it to the pending text.
    fn push(&self, event_tx: &mpsc::Sender<String>, delta: String) {
        let mut pending = self.lock();
        if pending.is_empty() {
            match event_tx.try_reserve() {
                Ok(permit) => {
                    permit.send(transcript_delta_event_payload(delta));
                    return;
                }
                // A closed receiver has nobody left to receive deltas. The
                // driver observes `event_tx.closed()` and cancels generation.
                Err(TrySendError::Closed(())) => return,
                Err(TrySendError::Full(())) => {}
            }
        }
        pending.push_str(&delta);
        drop(pending);
        self.notify.notify_one();
    }

    fn has_text(&self) -> bool {
        !self.lock().is_empty()
    }

    /// Send the pending text, if any, through a reserved queue slot.
    fn flush(&self, permit: mpsc::Permit<'_, String>) {
        let mut pending = self.lock();
        if !pending.is_empty() {
            let text = std::mem::take(&mut *pending);
            permit.send(transcript_delta_event_payload(text));
        }
    }

    fn take(&self) -> Option<String> {
        let mut pending = self.lock();
        (!pending.is_empty()).then(|| std::mem::take(&mut *pending))
    }
}

/// Run `generation` while forwarding its pending text whenever the queue has
/// room. Returns the events that close the stream (the text still pending,
/// then `done` or `error`), or `None` once the client has disconnected.
async fn drive_transcription_stream<F>(
    event_tx: &mpsc::Sender<String>,
    pending: &PendingTranscriptText,
    generation: F,
) -> Option<Vec<String>>
where
    F: Future<Output = izwi_core::Result<AsrTranscription>>,
{
    tokio::pin!(generation);
    let result = loop {
        tokio::select! {
            result = &mut generation => break result,
            _ = event_tx.closed() => return None,
            permit = event_tx.reserve(), if pending.has_text() => match permit {
                Ok(permit) => pending.flush(permit),
                Err(_) => return None,
            },
            _ = pending.notify.notified() => {}
        }
    };

    let mut terminal = pending
        .take()
        .map(transcript_delta_event_payload)
        .into_iter()
        .collect::<Vec<_>>();
    terminal.push(match result {
        Ok(output) => {
            transcript_done_event_payload(output.text, output.language, output.duration_secs)
        }
        Err(err) => transcript_error_event_payload(&err.to_string()),
    });
    Some(terminal)
}

async fn send_transcription_terminal(event_tx: mpsc::Sender<String>, payloads: Vec<String>) {
    let _ = tokio::time::timeout(TRANSCRIPTION_TERMINAL_SEND_TIMEOUT, async {
        for payload in payloads {
            if event_tx.send(payload).await.is_err() {
                return;
            }
        }
    })
    .await;
}

#[derive(Debug, Default)]
struct TranscriptionRequest {
    audio: Option<AudioPayload>,
    model: Option<String>,
    aligner_model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    stream: bool,
    prompt: Option<String>,
    max_tokens: Option<usize>,
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
    validate_transcription_language(req.model.as_deref(), req.language.as_deref())?;
    let audio = req
        .audio
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;

    info!(
        "OpenAI transcription request: {} source bytes",
        audio.bytes.len()
    );
    // Reject a format the response cannot take before spending ASR on it.
    let response_format = req
        .response_format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();
    validate_transcription_response_format(&response_format)?;

    if req.stream {
        return transcriptions_stream(state, req, audio, ctx.correlation_id).await;
    }
    let audio_bytes = audio.bytes;

    let timestamp_request =
        TimestampGranularityRequest::parse(req.timestamp_granularities.as_deref())?;
    if timestamp_request.any() && response_format != "verbose_json" {
        return Err(ApiError::bad_request(
            "`timestamp_granularities` requires `response_format: \"verbose_json\"`",
        ));
    }

    let permit = state
        .acquire_workload_permit(WorkloadClass::Interactive)
        .await;
    let started = Instant::now();
    // A model that emits word timestamps itself (Granite Speech Plus) runs
    // its timestamp task and returns `timestamp_words`; any other model's
    // text is timed by the forced aligner below.
    let output = state
        .runtime
        .asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
            audio_bytes.as_slice(),
            req.model.as_deref(),
            req.language.as_deref(),
            req.prompt.as_deref(),
            req.max_tokens,
            Some(&ctx.correlation_id),
            permit.runtime_context(),
            timestamp_request.any(),
        )
        .await
        .map_err(core_api_error)?;

    let mut words = None;
    let mut segments = None;
    let mut subtitle_segments = None;
    let mut alignment_error = None;
    let mut alignment_report = None;
    let chunk_boundaries = extract_chunk_segment_boundaries(output.asr_diagnostics.as_ref());
    let model_words = extract_model_timestamp_words(output.asr_diagnostics.as_ref());
    let model_segments = extract_model_timestamp_segments(output.asr_diagnostics.as_ref());
    if timestamp_request.words && !model_words.is_empty() {
        words = Some(model_words.clone());
    }
    if timestamp_request.segments && !model_segments.is_empty() {
        segments = Some(model_segments.clone());
    }
    if timestamp_request.segments && segments.is_none() && !model_words.is_empty() {
        segments = Some(build_timestamp_segments(&model_words, &chunk_boundaries));
    }
    if timestamp_request.any()
        && !output.text.trim().is_empty()
        && ((timestamp_request.words && words.is_none())
            || (timestamp_request.segments && segments.is_none()))
    {
        // The transcription already succeeded; an aligner failure costs the
        // word timings, not the transcript.
        match align_transcription_words(
            &state,
            audio_bytes.as_slice(),
            &output,
            req.language.as_deref(),
            req.aligner_model.as_deref(),
        )
        .await
        {
            Ok(alignment) => {
                let timestamp_words = alignments_to_timestamp_words(alignment.words());
                alignment_report = Some(AlignmentReport::from(&alignment));
                if timestamp_request.words && words.is_none() {
                    words = Some(timestamp_words.clone());
                }
                if timestamp_request.segments && segments.is_none() {
                    segments = Some(build_timestamp_segments(
                        &timestamp_words,
                        &chunk_boundaries,
                    ));
                }
            }
            Err(err) => {
                warn!("Forced alignment failed; returning untimed words: {}", err);
                alignment_error = Some(err.to_string());
            }
        }
    } else if matches!(response_format.as_str(), "srt" | "vtt") && !output.text.trim().is_empty() {
        if !model_segments.is_empty() {
            subtitle_segments = Some(model_segments.clone());
        }
        if subtitle_segments.is_none() && !model_words.is_empty() {
            subtitle_segments = Some(build_timestamp_segments(&model_words, &chunk_boundaries));
        }
        if subtitle_segments.is_none() {
            match align_transcription_words(
                &state,
                audio_bytes.as_slice(),
                &output,
                req.language.as_deref(),
                req.aligner_model.as_deref(),
            )
            .await
            {
                Ok(alignment) => {
                    let timestamp_words = alignments_to_timestamp_words(alignment.words());
                    alignment_report = Some(AlignmentReport::from(&alignment));
                    let built_segments =
                        build_timestamp_segments(&timestamp_words, &chunk_boundaries);
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
    }

    // OpenAI's verbose_json defaults to segment granularity, and a failed
    // alignment still answers with segments.
    if response_format == "verbose_json"
        && segments.is_none()
        && (timestamp_request.segments || !timestamp_request.any() || alignment_error.is_some())
    {
        segments = Some(fallback_timestamp_segments(
            &output.text,
            output.duration_secs,
            output.asr_diagnostics.as_ref(),
            model_segments,
            &model_words,
            &chunk_boundaries,
        ));
    }
    let asr_diagnostics = with_alignment_report(
        with_alignment_error(output.asr_diagnostics, alignment_error),
        alignment_report,
    );

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
                    // The model's detected language, else the one asked for.
                    language: output.language.or_else(|| req.language.clone()),
                    duration: output.duration_secs,
                    words,
                    segments,
                    processing_time_ms: elapsed_ms,
                    rtf,
                    izwi_asr_diagnostics: asr_diagnostics,
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
        _ => unreachable!("response format validated before transcription"),
    }
}

const SUPPORTED_TRANSCRIPTION_FORMATS: &str = "json, verbose_json, text, srt, vtt";

fn validate_transcription_response_format(response_format: &str) -> Result<(), ApiError> {
    match response_format {
        "json" | "verbose_json" | "text" | "srt" | "vtt" => Ok(()),
        "diarized_json" => Err(ApiError::bad_request(format!(
            "response_format \"diarized_json\" is not supported by /v1/audio/transcriptions; create a speaker-attributed transcript with POST /v1/diarizations. Supported: {SUPPORTED_TRANSCRIPTION_FORMATS}"
        ))),
        other => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {other}. Supported: {SUPPORTED_TRANSCRIPTION_FORMATS}"
        ))),
    }
}

/// Word timings for a finished transcription: each long-form chunk's text
/// is aligned against its own range, so long audio is never one forward.
async fn align_transcription_words(
    state: &AppState,
    audio_bytes: &[u8],
    output: &AsrTranscription,
    requested_language: Option<&str>,
    aligner_model: Option<&str>,
) -> izwi_core::Result<WindowedAlignment> {
    let language = output.language.as_deref().or(requested_language);
    align_transcript(
        &output.text,
        output.duration_secs,
        output.asr_diagnostics.as_ref(),
        |windows| {
            state
                .runtime
                .force_align_windows_bytes_with_model_and_language(
                    audio_bytes,
                    windows,
                    language,
                    aligner_model,
                )
        },
    )
    .await
}

/// Where a response's word timings came from, and the windows that fell
/// back to interval timings.
#[derive(Debug, Clone, PartialEq)]
struct AlignmentReport {
    source: AlignmentSource,
    warnings: Vec<String>,
}

impl From<&WindowedAlignment> for AlignmentReport {
    fn from(alignment: &WindowedAlignment) -> Self {
        Self {
            source: alignment.source(),
            warnings: alignment.warnings(),
        }
    }
}

async fn transcriptions_stream(
    state: AppState,
    req: TranscriptionRequest,
    audio: AudioPayload,
    correlation_id: String,
) -> Result<Response<Body>, ApiError> {
    let model = req.model;
    let language = req.language;
    let prompt = req.prompt;
    let max_tokens = req.max_tokens;
    let audio_bytes = audio.bytes;

    let (event_tx, mut event_rx) = mpsc::channel::<String>(TRANSCRIPTION_STREAM_CAPACITY);
    let engine = state.runtime.clone();
    let admission_state = state.clone();

    tokio::spawn(async move {
        let permit = match admission_state
            .acquire_owned_workload_permit(WorkloadClass::Streaming)
            .await
        {
            Ok(permit) => permit,
            Err(_) => {
                send_transcription_terminal(
                    event_tx,
                    vec![transcript_error_event_payload("Server is shutting down")],
                )
                .await;
                return;
            }
        };

        let delta_tx = event_tx.clone();
        let pending = Arc::new(PendingTranscriptText::default());
        let delta_pending = pending.clone();
        let generation = engine.asr_transcribe_streaming_bytes_with_runtime_context(
            audio_bytes.as_slice(),
            model.as_deref(),
            language.as_deref(),
            prompt.as_deref(),
            max_tokens,
            Some(correlation_id.as_str()),
            permit.runtime_context(),
            move |delta| delta_pending.push(&delta_tx, delta),
        );
        let terminal = drive_transcription_stream(&event_tx, &pending, generation).await;
        drop(permit);
        if let Some(terminal) = terminal {
            send_transcription_terminal(event_tx, terminal).await;
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
    max_tokens: Option<usize>,
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
        let audio_payload = decode_base64_audio_payload(payload.audio_base64.as_str())?;
        inspect_audio_payload_canonical_with_diagnostics("openai.audio.transcriptions", &audio_payload)?;

        return Ok(TranscriptionRequest {
            audio: Some(audio_payload),
            model: payload.model,
            aligner_model: payload.aligner_model,
            language: payload.language,
            response_format: payload.response_format,
            stream: payload.stream.unwrap_or(false),
            prompt: payload.prompt,
            max_tokens: payload.max_tokens,
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
                    if let Some(payload) = read_multipart_audio_file_payload(
                        field,
                        "OpenAI-compatible audio",
                        &name,
                        resolve_audio_upload_limit_bytes(),
                    )
                    .await?
                    {
                        inspect_audio_payload_canonical_with_diagnostics(
                            "openai.audio.transcriptions",
                            &payload,
                        )?;
                        out.audio = Some(payload);
                    }
                }
                "audio_base64" => {
                    if let Some(payload) = read_multipart_audio_base64_payload(
                        field,
                        "OpenAI-compatible audio",
                        "audio_base64",
                        resolve_audio_upload_limit_bytes(),
                    )
                    .await?
                    {
                        inspect_audio_payload_canonical_with_diagnostics(
                            "openai.audio.transcriptions",
                            &payload,
                        )?;
                        out.audio = Some(payload);
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
                "max_tokens" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'max_tokens' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        let value = text.trim().parse::<usize>().map_err(|_| {
                            ApiError::bad_request(
                                "Invalid multipart 'max_tokens' field: expected a non-negative integer",
                            )
                        })?;
                        out.max_tokens = Some(value);
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
        detail: None,
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

/// OpenAI rejects language codes a model does not support; so do the ASR
/// families that publish a language list (Granite Speech).
fn validate_transcription_language(
    model: Option<&str>,
    language: Option<&str>,
) -> Result<(), ApiError> {
    validate_asr_language(resolve_asr_model_variant(model), language)
        .map_err(|err| ApiError::bad_request(err.to_string()))
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

/// Segments when no timestamp source was asked for or it failed: the model's
/// own segments or words, else one segment per long-form chunk (its committed
/// text over its time range), else one segment spanning the audio.
fn fallback_timestamp_segments(
    text: &str,
    duration_secs: f32,
    diagnostics: Option<&serde_json::Value>,
    model_segments: Vec<TranscriptionTimestampSegment>,
    model_words: &[TranscriptionTimestampWord],
    chunk_boundaries: &[f32],
) -> Vec<TranscriptionTimestampSegment> {
    if !model_segments.is_empty() {
        return model_segments;
    }
    if !model_words.is_empty() {
        return build_timestamp_segments(model_words, chunk_boundaries);
    }
    let chunk_segments = extract_chunk_transcript_segments(diagnostics);
    if !chunk_segments.is_empty() {
        return chunk_segments;
    }
    let mut segments = Vec::new();
    push_timestamp_segment(
        &mut segments,
        0.0,
        duration_secs.max(0.0),
        &[text.to_string()],
    );
    segments
}

/// One segment per long-form chunk from the chunk executor's diagnostics.
/// Overlapping chunks start where the previous segment ended, so segments
/// never overlap and their texts concatenate to the transcript.
fn extract_chunk_transcript_segments(
    diagnostics: Option<&serde_json::Value>,
) -> Vec<TranscriptionTimestampSegment> {
    let Some(chunks) = diagnostics
        .and_then(|diagnostics| {
            diagnostics
                .get("chunking")
                .or_else(|| diagnostics.get("model_diagnostics")?.get("chunking"))
        })
        .and_then(|chunking| chunking.get("chunk_transcriptions"))
        .and_then(|chunks| chunks.as_array())
    else {
        return Vec::new();
    };
    let mut segments = Vec::new();
    let mut previous_end = 0.0f32;
    for chunk in chunks {
        let text = chunk
            .get("committed_text")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .trim();
        let start = chunk.get("start_seconds").and_then(|value| value.as_f64());
        let end = chunk.get("end_seconds").and_then(|value| value.as_f64());
        let (Some(start), Some(end)) = (start, end) else {
            continue;
        };
        if text.is_empty() || !start.is_finite() || !end.is_finite() {
            continue;
        }
        let start = (start as f32).max(previous_end);
        let end = (end as f32).max(start);
        push_timestamp_segment(&mut segments, start, end, &[text.to_string()]);
        previous_end = end;
    }
    segments
}

/// Report a forced-alignment failure in the diagnostics of a response that
/// otherwise succeeded.
fn with_alignment_error(
    diagnostics: Option<serde_json::Value>,
    alignment_error: Option<String>,
) -> Option<serde_json::Value> {
    let Some(alignment_error) = alignment_error else {
        return diagnostics;
    };
    let mut diagnostics = match diagnostics {
        Some(serde_json::Value::Object(root)) => root,
        Some(other) => serde_json::Map::from_iter([("model_diagnostics".to_string(), other)]),
        None => serde_json::Map::new(),
    };
    diagnostics.insert(
        "alignment_error".to_string(),
        serde_json::Value::String(alignment_error),
    );
    Some(serde_json::Value::Object(diagnostics))
}

/// Add the alignment source and its warnings to the diagnostics.
fn with_alignment_report(
    diagnostics: Option<serde_json::Value>,
    report: Option<AlignmentReport>,
) -> Option<serde_json::Value> {
    let Some(report) = report else {
        return diagnostics;
    };
    let mut diagnostics = match diagnostics {
        Some(serde_json::Value::Object(root)) => root,
        Some(other) => serde_json::Map::from_iter([("model_diagnostics".to_string(), other)]),
        None => serde_json::Map::new(),
    };
    diagnostics.insert(
        "alignment_source".to_string(),
        serde_json::json!(report.source),
    );
    if !report.warnings.is_empty() {
        diagnostics.insert(
            "alignment_warnings".to_string(),
            serde_json::json!(report.warnings),
        );
    }
    Some(serde_json::Value::Object(diagnostics))
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

fn extract_model_timestamp_words(
    diagnostics: Option<&serde_json::Value>,
) -> Vec<TranscriptionTimestampWord> {
    let Some(diagnostics) = diagnostics else {
        return Vec::new();
    };
    let Some(raw_words) = diagnostic_array(diagnostics, "timestamp_words") else {
        return Vec::new();
    };

    let mut previous_end = 0.0f32;
    raw_words
        .iter()
        .filter_map(|word| {
            let text = word
                .get("word")
                .or_else(|| word.get("text"))
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if text.is_empty() {
                return None;
            }
            let start = word
                .get("start_time_seconds")
                .or_else(|| word.get("start_time"))
                .or_else(|| word.get("start"))
                .and_then(|value| value.as_f64())
                .map(|value| value as f32)
                .unwrap_or(previous_end)
                .max(0.0);
            let end = word
                .get("end_time_seconds")
                .or_else(|| word.get("end_time"))
                .or_else(|| word.get("end"))
                .and_then(|value| value.as_f64())
                .map(|value| value as f32)
                .unwrap_or(start);
            if !start.is_finite() || !end.is_finite() {
                return None;
            }
            let end = end.max(start + 0.01);
            previous_end = end;
            Some(TranscriptionTimestampWord {
                word: text,
                start,
                end,
            })
        })
        .collect()
}

fn extract_model_timestamp_segments(
    diagnostics: Option<&serde_json::Value>,
) -> Vec<TranscriptionTimestampSegment> {
    let Some(diagnostics) = diagnostics else {
        return Vec::new();
    };
    let Some(raw_segments) = diagnostic_array(diagnostics, "segments") else {
        return Vec::new();
    };

    raw_segments
        .iter()
        .filter_map(|segment| {
            let start = segment
                .get("start_time")
                .or_else(|| segment.get("start"))
                .and_then(|value| value.as_f64())?;
            let end = segment
                .get("end_time")
                .or_else(|| segment.get("end"))
                .and_then(|value| value.as_f64())
                .unwrap_or(start);
            let text = segment
                .get("content")
                .or_else(|| segment.get("text"))
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if !start.is_finite() || !end.is_finite() || text.is_empty() {
                return None;
            }
            Some((start as f32, end as f32, text))
        })
        .enumerate()
        .map(|(id, (start, end, text))| TranscriptionTimestampSegment {
            id,
            start: start.max(0.0),
            end: end.max(start + 0.01),
            text,
        })
        .collect()
}

fn diagnostic_array<'a>(
    diagnostics: &'a serde_json::Value,
    key: &str,
) -> Option<&'a Vec<serde_json::Value>> {
    diagnostics
        .get(key)
        .and_then(|value| value.as_array())
        .or_else(|| {
            diagnostics
                .get("output")?
                .get(key)
                .and_then(|value| value.as_array())
        })
        .or_else(|| {
            diagnostics
                .get("model_diagnostics")?
                .get(key)
                .and_then(|value| value.as_array())
        })
        .or_else(|| {
            diagnostics
                .get("model_diagnostics")?
                .get("output")?
                .get(key)
                .and_then(|value| value.as_array())
        })
}

fn push_timestamp_segment(
    segments: &mut Vec<TranscriptionTimestampSegment>,
    start: f32,
    end: f32,
    words: &[String],
) {
    // Aligner words are surface spans: Chinese and Japanese words join
    // without spaces, as the transcript writes them.
    let text = words
        .iter()
        .fold(String::new(), |mut text, word| {
            text.push_str(izwi_core::spacing_between(&text, word));
            text.push_str(word);
            text
        })
        .trim()
        .to_string();
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
    use base64::Engine as _;
    use izwi_core::audio::{AudioEncoder, AudioFormat};

    fn wav_data_url() -> String {
        let wav = AudioEncoder::new(16_000, 1)
            .encode(&[0.0, 0.1, -0.1, 0.0], AudioFormat::Wav)
            .expect("wav should encode");
        let b64 = base64::engine::general_purpose::STANDARD.encode(wav);
        format!("data:audio/wav;base64, {b64}\n")
    }

    #[tokio::test]
    async fn parse_json_request_canonicalizes_data_url_audio() {
        let request = Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "audio_base64": wav_data_url(),
                    "model": "Parakeet-TDT-0.6B-v3",
                    "response_format": "json"
                })
                .to_string(),
            ))
            .expect("request should build");

        let parsed = parse_transcription_request(request)
            .await
            .expect("request should parse");

        let audio = parsed.audio.expect("audio payload should parse");
        assert!(audio.bytes.starts_with(b"RIFF"));
        assert_eq!(audio.data_url_mime_type.as_deref(), Some("audio/wav"));
        assert_eq!(parsed.model.as_deref(), Some("Parakeet-TDT-0.6B-v3"));
    }

    #[tokio::test]
    async fn parse_json_request_rejects_undecodable_audio() {
        let request = Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "audio_base64": "data:audio/wav;base64, YXVk\naW8=",
                    "model": "Parakeet-TDT-0.6B-v3"
                })
                .to_string(),
            ))
            .expect("request should build");

        let err = parse_transcription_request(request)
            .await
            .expect_err("invalid audio should fail");

        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("Invalid audio payload"));
    }

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

    fn transcription(text: String) -> AsrTranscription {
        AsrTranscription {
            text,
            language: Some("en".to_string()),
            duration_secs: 60.0,
            asr_diagnostics: None,
        }
    }

    /// Run a generation that emits `deltas` back to back within one poll, the
    /// way the runtime hands over a committed outbox, and return every event
    /// the client received.
    async fn stream_burst(
        capacity: usize,
        deltas: Vec<String>,
        read_while_generating: bool,
    ) -> Vec<serde_json::Value> {
        let (event_tx, mut event_rx) = mpsc::channel::<String>(capacity);
        let gate = Arc::new(Notify::new());
        let consumer_gate = gate.clone();
        let consumer = tokio::spawn(async move {
            if !read_while_generating {
                consumer_gate.notified().await;
            }
            let mut events = Vec::new();
            while let Some(event) = event_rx.recv().await {
                events.push(serde_json::from_str::<serde_json::Value>(&event).expect("event JSON"));
            }
            events
        });
        let pending = PendingTranscriptText::default();
        let delta_tx = event_tx.clone();
        let generation = async {
            let mut text = String::new();
            for delta in deltas {
                text.push_str(&delta);
                pending.push(&delta_tx, delta);
            }
            Ok(transcription(text))
        };
        let terminal = drive_transcription_stream(&event_tx, &pending, generation)
            .await
            .expect("the client is still connected");
        drop(delta_tx);
        gate.notify_one();
        send_transcription_terminal(event_tx, terminal).await;
        consumer.await.expect("consumer task")
    }

    fn assert_deltas_concatenate_to_done(events: &[serde_json::Value]) -> usize {
        let (done, deltas) = events.split_last().expect("events");
        assert_eq!(done["type"], "transcript.text.done", "{done}");
        let mut streamed = String::new();
        for delta in deltas {
            assert_eq!(delta["type"], "transcript.text.delta", "{delta}");
            streamed.push_str(delta["delta"].as_str().expect("delta text"));
        }
        assert_eq!(streamed, done["text"].as_str().expect("done text"));
        deltas.len()
    }

    #[tokio::test]
    async fn token_rate_burst_streams_completely_to_a_reading_client() {
        // A 60 s LFM2.5 chunk-driver stream: one delta per token, delivered
        // from a committed outbox far faster than the SSE body is polled.
        let deltas = (0..2_000).map(|idx| format!("w{idx} ")).collect();
        let events = stream_burst(TRANSCRIPTION_STREAM_CAPACITY, deltas, true).await;
        assert_deltas_concatenate_to_done(&events);
    }

    #[tokio::test]
    async fn saturated_stream_coalesces_deltas_instead_of_failing() {
        let deltas = ["first ", "second ", "third ", "fourth"]
            .map(str::to_string)
            .to_vec();
        let events = stream_burst(1, deltas, false).await;
        let delta_events = assert_deltas_concatenate_to_done(&events);
        assert_eq!(events[0]["delta"], "first ");
        assert_eq!(events[1]["delta"], "second third fourth");
        assert_eq!(delta_events, 2);
    }

    #[tokio::test]
    async fn generation_error_still_sends_the_text_streamed_before_it() {
        let (event_tx, mut event_rx) = mpsc::channel::<String>(1);
        let pending = PendingTranscriptText::default();
        let generation = async {
            pending.push(&event_tx, "a ".to_string());
            pending.push(&event_tx, "b".to_string());
            Err(izwi_core::Error::InferenceError("boom".to_string()))
        };
        let terminal = drive_transcription_stream(&event_tx, &pending, generation)
            .await
            .expect("the client is still connected");
        let reader = tokio::spawn(async move {
            let mut events = Vec::new();
            while let Some(event) = event_rx.recv().await {
                events.push(serde_json::from_str::<serde_json::Value>(&event).unwrap());
            }
            events
        });
        send_transcription_terminal(event_tx, terminal).await;
        let events = reader.await.unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0]["delta"], "a ");
        assert_eq!(events[1]["delta"], "b");
        assert_eq!(events[2]["type"], "error");
    }

    #[tokio::test]
    async fn disconnected_client_cancels_generation() {
        let (event_tx, event_rx) = mpsc::channel::<String>(1);
        drop(event_rx);
        let pending = PendingTranscriptText::default();
        let terminal =
            drive_transcription_stream(&event_tx, &pending, std::future::pending()).await;
        assert!(terminal.is_none());
    }

    #[tokio::test]
    async fn non_draining_transcription_consumer_cannot_retain_workload_capacity() {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let (event_tx, mut event_rx) = mpsc::channel(1);
        event_tx.send("queued-delta".to_string()).await.unwrap();

        let driver = tokio::spawn(async move {
            let pending = PendingTranscriptText::default();
            let generation = async {
                pending.push(&event_tx, "unsent".to_string());
                Ok(transcription("done".to_string()))
            };
            let terminal = drive_transcription_stream(&event_tx, &pending, generation).await;
            drop(permit);
            if let Some(terminal) = terminal {
                send_transcription_terminal(event_tx, terminal).await;
            }
        });

        tokio::time::timeout(Duration::from_millis(50), async {
            while semaphore.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal transport wait must not retain the workload permit");
        tokio::time::timeout(
            TRANSCRIPTION_TERMINAL_SEND_TIMEOUT + Duration::from_millis(50),
            driver,
        )
        .await
        .expect("terminal transport wait must be bounded")
        .expect("stream driver should not panic");
        assert_eq!(event_rx.recv().await.as_deref(), Some("queued-delta"));
        assert!(event_rx.recv().await.is_none());
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
    fn transcription_model_validation_accepts_granite_speech() {
        validate_transcription_model(Some("ibm-granite/granite-speech-4.1-2b-plus"))
            .expect("Granite Speech should be accepted for offline transcription");
    }

    #[test]
    fn transcription_model_validation_rejects_tts_or_chat_model() {
        let tts = validate_transcription_model(Some("Kokoro-82M"))
            .expect_err("TTS model should be rejected for transcription");
        assert!(tts.message.contains("Unsupported transcription model"));

        let chat = validate_transcription_model(Some("Qwen3.5-4B"))
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
    fn transcription_language_is_checked_against_the_routed_granite_checkpoint() {
        validate_transcription_language(Some("Granite-Speech-4.1-2B"), Some("ja")).unwrap();
        validate_transcription_language(Some("granite speech"), Some("German")).unwrap();
        let err = validate_transcription_language(Some("Granite-Speech-4.1-2B-Plus"), Some("ja"))
            .unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err
            .message
            .contains("supported source languages: de, en, es, fr, pt"));
        assert_eq!(
            validate_transcription_language(Some("Granite-Speech-4.1-2B"), Some("klingon"))
                .unwrap_err()
                .status,
            StatusCode::BAD_REQUEST
        );
        validate_transcription_language(Some("Whisper-Large-v3-Turbo"), Some("klingon")).unwrap();
        validate_transcription_language(None, Some("klingon")).unwrap();
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

    /// Chinese and Japanese aligner words make segment text without spaces;
    /// spaced scripts keep theirs.
    #[test]
    fn timestamp_segments_join_cjk_words_without_spaces() {
        let word = |word: &str, start: f32| TranscriptionTimestampWord {
            word: word.to_string(),
            start,
            end: start + 0.2,
        };
        let chinese = build_timestamp_segments(
            &[word("我们", 0.0), word("明天", 0.2), word("见面。", 0.4)],
            &[],
        );
        assert_eq!(chinese[0].text, "我们明天见面。");
        let japanese = build_timestamp_segments(&[word("今日は", 0.0), word("晴れ", 0.2)], &[]);
        assert_eq!(japanese[0].text, "今日は晴れ");
        let mixed = build_timestamp_segments(
            &[word("Hello,", 0.0), word("world", 0.2), word("你好", 0.4)],
            &[],
        );
        assert_eq!(mixed[0].text, "Hello, world你好");
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
    fn extracts_model_timestamp_segments_from_vibevoice_diagnostics() {
        let diagnostics = serde_json::json!({
            "output": {
                "segments": [
                    {
                        "start_time": 0.0,
                        "end_time": 1.5,
                        "speaker_id": "Speaker 0",
                        "content": "hello"
                    },
                    {
                        "start_time": 1.5,
                        "end_time": 2.0,
                        "speaker_id": "Speaker 1",
                        "content": "world"
                    }
                ]
            }
        });

        let segments = extract_model_timestamp_segments(Some(&diagnostics));

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].id, 0);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[1].start, 1.5);
    }

    #[test]
    fn extracts_granite_timestamp_words_from_root_diagnostics() {
        let diagnostics = serde_json::json!({
            "family": "granite_speech_asr",
            "timestamp_words": [
                {
                    "word": "hello",
                    "end_time_seconds": 0.45
                },
                {
                    "word": "world",
                    "end_time_seconds": 0.9
                }
            ]
        });

        let words = extract_model_timestamp_words(Some(&diagnostics));

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert!((words[0].start - 0.0).abs() < f32::EPSILON);
        assert!((words[0].end - 0.45).abs() < f32::EPSILON);
        assert_eq!(words[1].word, "world");
        assert!((words[1].start - 0.45).abs() < f32::EPSILON);
        assert!((words[1].end - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn builds_verbose_segments_from_granite_timestamp_words() {
        let diagnostics = serde_json::json!({
            "family": "granite_speech_asr",
            "timestamp_words": [
                { "word": "hello", "end_time_seconds": 0.4 },
                { "word": "world", "end_time_seconds": 0.9 }
            ]
        });

        let words = extract_model_timestamp_words(Some(&diagnostics));
        let segments = build_timestamp_segments(&words, &[]);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello world");
        assert!((segments[0].start - 0.0).abs() < f32::EPSILON);
        assert!((segments[0].end - 0.9).abs() < f32::EPSILON);
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

    #[test]
    fn default_segments_follow_the_long_form_chunks() {
        let diagnostics = serde_json::json!({
            "chunking": {
                "chunk_transcriptions": [
                    { "start_seconds": 0.0, "end_seconds": 30.0, "committed_text": "one two three" },
                    { "start_seconds": 28.5, "end_seconds": 55.0, "committed_text": "four" },
                    { "start_seconds": 55.0, "end_seconds": 60.0, "committed_text": "" },
                    { "skipped": true }
                ]
            }
        });

        let segments = fallback_timestamp_segments(
            "one two three four",
            60.0,
            Some(&diagnostics),
            Vec::new(),
            &[],
            &[],
        );

        let spans: Vec<(f32, f32, &str)> = segments
            .iter()
            .map(|segment| (segment.start, segment.end, segment.text.as_str()))
            .collect();
        assert_eq!(spans, [(0.0, 30.0, "one two three"), (30.0, 55.0, "four")]);
        assert_eq!(segments[1].id, 1);
    }

    #[test]
    fn default_segments_prefer_model_timings_then_span_a_single_window() {
        let model_words = vec![
            TranscriptionTimestampWord {
                word: "Hello".to_string(),
                start: 0.2,
                end: 0.6,
            },
            TranscriptionTimestampWord {
                word: "there.".to_string(),
                start: 0.7,
                end: 1.1,
            },
        ];
        let from_words =
            fallback_timestamp_segments("Hello there.", 3.0, None, Vec::new(), &model_words, &[]);
        assert_eq!(from_words.len(), 1);
        assert_eq!((from_words[0].start, from_words[0].end), (0.2, 1.1));

        let single = fallback_timestamp_segments("Hello there.", 3.0, None, Vec::new(), &[], &[]);
        assert_eq!(single.len(), 1);
        assert_eq!(
            (single[0].start, single[0].end, single[0].text.as_str()),
            (0.0, 3.0, "Hello there.")
        );

        assert!(fallback_timestamp_segments("  ", 3.0, None, Vec::new(), &[], &[]).is_empty());
    }

    #[test]
    fn alignment_source_and_warnings_are_reported_in_the_diagnostics() {
        assert_eq!(with_alignment_report(None, None), None);
        let diagnostics = with_alignment_report(
            Some(serde_json::json!({"chunking": {}})),
            Some(AlignmentReport {
                source: AlignmentSource::Heuristic,
                warnings: vec!["Words in 30.00-60.00 s use evenly spaced timings".into()],
            }),
        )
        .expect("diagnostics");
        assert_eq!(diagnostics["alignment_source"], "heuristic");
        assert_eq!(
            diagnostics["alignment_warnings"].as_array().unwrap().len(),
            1
        );
        assert!(diagnostics.get("chunking").is_some());
        let model = with_alignment_report(
            None,
            Some(AlignmentReport {
                source: AlignmentSource::Model,
                warnings: Vec::new(),
            }),
        )
        .expect("diagnostics");
        assert_eq!(model["alignment_source"], "model");
        assert!(model.get("alignment_warnings").is_none());
    }

    #[test]
    fn alignment_errors_are_reported_in_the_diagnostics() {
        assert_eq!(with_alignment_error(None, None), None);
        let diagnostics = with_alignment_error(
            Some(serde_json::json!({ "model_family": "parakeet" })),
            Some("aligner window exceeded".to_string()),
        )
        .expect("diagnostics");
        assert_eq!(diagnostics["model_family"], "parakeet");
        assert_eq!(diagnostics["alignment_error"], "aligner window exceeded");

        let created = with_alignment_error(None, Some("boom".to_string())).expect("diagnostics");
        assert_eq!(created["alignment_error"], "boom");
    }
}
