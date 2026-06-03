//! Forced alignment audio endpoint.

use axum::{
    Json, RequestExt,
    body::Body,
    extract::{Multipart, Request, State},
    http::{StatusCode, header},
    response::Response,
};
use serde::Serialize;
use std::time::Instant;
use utoipa::ToSchema;

use super::resolve_audio_upload_limit_bytes;
use crate::api::audio_payload::{
    AudioPayload, decode_base64_audio_payload, read_multipart_audio_base64_payload,
    read_multipart_audio_file_payload,
};
use crate::api::speech_text_upload::multipart_upload_api_error;
use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Default)]
struct AlignmentRequest {
    audio: Option<AudioPayload>,
    text: Option<String>,
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
}

#[derive(Debug, serde::Deserialize, ToSchema)]
pub struct AlignmentJsonRequest {
    pub audio_base64: String,
    pub text: String,
    pub model: Option<String>,
    pub language: Option<String>,
    pub response_format: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AlignmentMultipartRequest {
    #[schema(value_type = String, format = Binary)]
    pub file: Option<String>,
    pub audio_base64: Option<String>,
    pub text: String,
    pub model: Option<String>,
    pub language: Option<String>,
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AlignmentWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AlignmentResponse {
    pub alignments: Vec<AlignmentWord>,
    pub duration: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct VerboseAlignmentResponse {
    pub alignments: Vec<AlignmentWord>,
    pub duration: f32,
    pub model: Option<String>,
    pub language: Option<String>,
    pub word_count: usize,
    pub processing_time_ms: f64,
}

pub async fn align(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response<Body>, ApiError> {
    let mut req = parse_alignment_request(req).await?;
    let audio = req
        .audio
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;
    let text = req
        .text
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing reference text (`text`)"))?;
    validate_alignment_text(text.as_str())?;

    let response_format = req
        .response_format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();
    if !matches!(response_format.as_str(), "json" | "verbose_json" | "text") {
        return Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Supported: json, verbose_json, text",
            response_format
        )));
    }

    let _permit = state.acquire_permit().await;
    let started = Instant::now();
    let raw_alignments = state
        .runtime
        .force_align_bytes_with_model_and_language(
            audio.bytes.as_slice(),
            text.as_str(),
            req.language.as_deref(),
            req.model.as_deref(),
        )
        .await?;
    let processing_time_ms = started.elapsed().as_secs_f64() * 1000.0;

    let alignments = alignment_words(raw_alignments);
    let duration = alignment_duration(&alignments);

    match response_format.as_str() {
        "json" => json_response(&AlignmentResponse {
            alignments,
            duration,
        }),
        "verbose_json" => json_response(&VerboseAlignmentResponse {
            word_count: alignments.len(),
            alignments,
            duration,
            model: req.model,
            language: req.language,
            processing_time_ms,
        }),
        "text" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(format_alignment_text(&alignments)))
            .unwrap()),
        _ => unreachable!("response format validated above"),
    }
}

fn json_response<T: Serialize>(body: &T) -> Result<Response<Body>, ApiError> {
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(body).unwrap()))
        .unwrap())
}

async fn parse_alignment_request(req: Request) -> Result<AlignmentRequest, ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let Json(payload) = req
            .extract::<Json<AlignmentJsonRequest>, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid JSON payload: {e}")))?;
        let audio_payload = decode_base64_audio_payload(payload.audio_base64.as_str())?;

        return Ok(AlignmentRequest {
            audio: Some(audio_payload),
            text: Some(payload.text),
            model: payload.model,
            language: payload.language,
            response_format: payload.response_format,
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart payload: {e}")))?;
        let mut out = AlignmentRequest::default();

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
                        out.audio = Some(payload);
                    }
                }
                "text" | "reference_text" => {
                    let text = read_multipart_text(field, &name).await?;
                    if !text.trim().is_empty() {
                        out.text = Some(text.trim().to_string());
                    }
                }
                "model" => {
                    let text = read_multipart_text(field, "model").await?;
                    if !text.trim().is_empty() {
                        out.model = Some(text.trim().to_string());
                    }
                }
                "language" => {
                    let text = read_multipart_text(field, "language").await?;
                    if !text.trim().is_empty() {
                        out.language = Some(text.trim().to_string());
                    }
                }
                "response_format" => {
                    let text = read_multipart_text(field, "response_format").await?;
                    if !text.trim().is_empty() {
                        out.response_format = Some(text.trim().to_string());
                    }
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

async fn read_multipart_text(
    field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> Result<String, ApiError> {
    field
        .text()
        .await
        .map_err(|e| ApiError::bad_request(format!("Failed reading multipart '{name}' field: {e}")))
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

fn validate_alignment_text(text: &str) -> Result<(), ApiError> {
    if text.trim().is_empty() {
        Err(ApiError::bad_request("Reference text cannot be empty"))
    } else {
        Ok(())
    }
}

fn alignment_words(raw: Vec<(String, u32, u32)>) -> Vec<AlignmentWord> {
    raw.into_iter()
        .map(|(word, start_ms, end_ms)| AlignmentWord {
            word,
            start: millis_to_secs(start_ms),
            end: millis_to_secs(end_ms),
        })
        .collect()
}

fn millis_to_secs(ms: u32) -> f32 {
    (ms as f32) / 1000.0
}

fn alignment_duration(words: &[AlignmentWord]) -> f32 {
    words.iter().map(|word| word.end).fold(0.0, f32::max)
}

fn format_alignment_text(words: &[AlignmentWord]) -> String {
    let mut out = String::new();
    for word in words {
        out.push_str(&format!(
            "{:<24} {:.2} - {:.2}\n",
            word.word, word.start, word.end
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn parse_json_request_canonicalizes_data_url_audio() {
        let request = Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "audio_base64": "data:audio/wav;base64, YXVk\naW8=",
                    "text": "audio"
                })
                .to_string(),
            ))
            .expect("request should build");

        let parsed = parse_alignment_request(request)
            .await
            .expect("request should parse");

        let audio = parsed.audio.expect("audio payload should parse");
        assert_eq!(audio.bytes, b"audio");
        assert_eq!(audio.to_base64(), "YXVkaW8=");
        assert_eq!(audio.data_url_mime_type.as_deref(), Some("audio/wav"));
        assert_eq!(parsed.text.as_deref(), Some("audio"));
    }

    #[test]
    fn alignment_words_convert_millis_to_seconds() {
        let words = alignment_words(vec![("hello".to_string(), 120, 980)]);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "hello");
        assert!((words[0].start - 0.12).abs() < f32::EPSILON);
        assert!((words[0].end - 0.98).abs() < f32::EPSILON);
    }

    #[test]
    fn formats_text_alignment_for_cli() {
        let text = format_alignment_text(&[AlignmentWord {
            word: "hello".to_string(),
            start: 0.0,
            end: 0.45,
        }]);
        assert!(text.contains("hello"));
        assert!(text.contains("0.00 - 0.45"));
    }

    #[test]
    fn rejects_blank_reference_text() {
        let err = validate_alignment_text("  ").expect_err("blank text should be rejected");
        assert!(err.message.contains("Reference text cannot be empty"));
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
}
