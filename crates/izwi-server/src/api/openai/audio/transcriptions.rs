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
use base64::Engine;
use std::convert::Infallible;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::info;

use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::parse_model_variant;
use super::resolve_audio_upload_limit_bytes;

#[derive(Debug, Default)]
struct TranscriptionRequest {
    audio_base64: Option<String>,
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    stream: bool,
    // Accepted for compatibility; currently not used by runtime.
    _prompt: Option<String>,
    _temperature: Option<f32>,
    _timestamp_granularities: Option<Vec<String>>,
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
    processing_time_ms: f64,
    rtf: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_asr_diagnostics: Option<serde_json::Value>,
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

    let _permit = state.acquire_permit().await;

    let started = Instant::now();
    let output = state
        .runtime
        .asr_transcribe_with_correlation(
            &audio_base64,
            req.model.as_deref(),
            req.language.as_deref(),
            Some(&ctx.correlation_id),
        )
        .await?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let response_format = req
        .response_format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();

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
            .body(Body::from(format_srt(&output.text, output.duration_secs)))
            .unwrap()),
        "vtt" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/vtt; charset=utf-8")
            .body(Body::from(format_vtt(&output.text, output.duration_secs)))
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
            .asr_transcribe_streaming_with_correlation(
                &audio_base64,
                model.as_deref(),
                language.as_deref(),
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
            language: payload.language,
            response_format: payload.response_format,
            stream: payload.stream.unwrap_or(false),
            _prompt: payload.prompt,
            _temperature: payload.temperature,
            _timestamp_granularities: payload.timestamp_granularities,
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
            .map_err(|e| ApiError::bad_request(format!("Failed reading multipart field: {e}")))?
        {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let bytes = field
                        .bytes()
                        .await
                        .map_err(|e| multipart_field_error(&name, &e.to_string()))?;
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
                        out._prompt = Some(text.trim().to_string());
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
                        out._timestamp_granularities
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

fn multipart_field_error(field_name: &str, raw: &str) -> ApiError {
    let lowered = raw.to_ascii_lowercase();
    if lowered.contains("multipart/form-data") {
        let limit_mb = resolve_audio_upload_limit_bytes() / (1024 * 1024);
        return ApiError::bad_request(format!(
            "Failed reading multipart '{}' field: {}. \
This is commonly caused by oversized uploads or malformed multipart boundaries. \
Ensure `Content-Type` includes a valid boundary (let your HTTP client set it automatically for FormData) and keep payload under {} MiB.",
            field_name, raw, limit_mb
        ));
    }

    ApiError::bad_request(format!(
        "Failed reading multipart '{}' field: {}",
        field_name, raw
    ))
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

fn format_srt(text: &str, duration_secs: f32) -> String {
    format!(
        "1\n{} --> {}\n{}\n",
        secs_to_srt(0.0),
        secs_to_srt(duration_secs.max(0.1)),
        text.trim()
    )
}

fn format_vtt(text: &str, duration_secs: f32) -> String {
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
        let srt = format_srt("hello", 1.23);
        let vtt = format_vtt("hello", 1.23);
        assert!(srt.contains("-->"));
        assert!(vtt.starts_with("WEBVTT"));
    }

    #[test]
    fn stream_delta_event_uses_openai_type_field() {
        let payload = transcript_delta_event_payload("hello".to_string());
        let json: serde_json::Value = serde_json::from_str(&payload).expect("json payload");
        assert_eq!(json.get("type").and_then(|v| v.as_str()), Some("transcript.text.delta"));
        assert_eq!(json.get("delta").and_then(|v| v.as_str()), Some("hello"));
    }

    #[test]
    fn multipart_error_mentions_configured_limit() {
        let expected_limit = resolve_audio_upload_limit_bytes() / (1024 * 1024);
        let err = multipart_field_error("file", "multipart/form-data field parse failed");
        assert!(err.message.contains(&format!("{expected_limit} MiB")));
    }

    #[test]
    fn transcription_model_validation_accepts_known_asr_model() {
        validate_transcription_model(Some("Parakeet-TDT-0.6B-v3"))
            .expect("known ASR model should be accepted");
    }

    #[test]
    fn transcription_model_validation_rejects_unknown_model() {
        let err = validate_transcription_model(Some("definitely-not-a-real-model"))
            .expect_err("unknown model should be rejected");
        assert!(err.message.contains("Unsupported transcription model"));
    }
}
