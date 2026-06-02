use axum::extract::multipart::{Field, MultipartError};
use base64::Engine;
use izwi_core::audio::{AudioInspection, inspect_audio_bytes};

use crate::api::speech_text_upload::multipart_upload_api_error;
use crate::error::ApiError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AudioPayload {
    pub bytes: Vec<u8>,
    pub source_mime_type: Option<String>,
    pub filename: Option<String>,
    pub data_url_mime_type: Option<String>,
}

impl AudioPayload {
    pub(crate) fn from_bytes(
        bytes: impl Into<Vec<u8>>,
        source_mime_type: Option<String>,
        filename: Option<String>,
    ) -> Self {
        Self {
            bytes: bytes.into(),
            source_mime_type: sanitize_metadata(source_mime_type),
            filename: sanitize_metadata(filename),
            data_url_mime_type: None,
        }
    }

    pub(crate) fn content_type_hint(&self) -> Option<&str> {
        self.source_mime_type
            .as_deref()
            .or(self.data_url_mime_type.as_deref())
    }

    pub(crate) fn to_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.bytes)
    }
}

pub(crate) fn decode_base64_audio_payload(raw: &str) -> Result<AudioPayload, ApiError> {
    decode_base64_payload(raw, "audio")
}

pub(crate) fn decode_base64_media_payload(raw: &str) -> Result<AudioPayload, ApiError> {
    decode_base64_payload(raw, "media")
}

pub(crate) fn decode_optional_base64_audio_payload(
    raw: &str,
) -> Result<Option<AudioPayload>, ApiError> {
    if raw.trim().is_empty() {
        return Ok(None);
    }
    decode_base64_audio_payload(raw).map(Some)
}

pub(crate) async fn read_multipart_audio_file_payload(
    field: Field<'_>,
    route_label: &str,
    field_name: &str,
    limit_bytes: usize,
) -> Result<Option<AudioPayload>, ApiError> {
    let source_mime_type = field.content_type().map(str::to_string);
    let filename = field.file_name().map(str::to_string);
    let bytes = field
        .bytes()
        .await
        .map_err(|err| multipart_audio_error(route_label, field_name, limit_bytes, err))?;
    if bytes.is_empty() {
        return Ok(None);
    }
    Ok(Some(AudioPayload::from_bytes(
        bytes.to_vec(),
        source_mime_type,
        filename,
    )))
}

pub(crate) async fn read_multipart_audio_base64_payload(
    field: Field<'_>,
    field_name: &str,
) -> Result<Option<AudioPayload>, ApiError> {
    let text = field.text().await.map_err(|err| {
        ApiError::bad_request(format!(
            "Failed reading multipart '{field_name}' field: {err}"
        ))
    })?;
    decode_optional_base64_audio_payload(text.as_str())
}

pub(crate) fn inspect_audio_payload(payload: &AudioPayload) -> Result<AudioInspection, ApiError> {
    inspect_audio_payload_bytes(&payload.bytes)
}

pub(crate) fn inspect_audio_payload_bytes(bytes: &[u8]) -> Result<AudioInspection, ApiError> {
    inspect_audio_bytes(bytes).map_err(|err| {
        ApiError::bad_request(format!(
            "Invalid audio payload: failed to decode audio metadata: {err}"
        ))
    })
}

pub(crate) fn is_audio_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .map(str::trim)
        .is_some_and(|value| value.to_ascii_lowercase().starts_with("audio/"))
}

pub(crate) fn split_data_url_base64(raw: &str) -> (Option<String>, &str) {
    let trimmed = raw.trim();
    let Some(data_url) = strip_data_url_prefix(trimmed) else {
        return (None, raw);
    };
    let Some((metadata, payload)) = data_url.split_once(',') else {
        return (None, raw);
    };
    if !metadata.to_ascii_lowercase().contains(";base64") {
        return (None, raw);
    }
    let content_type = metadata
        .split(';')
        .next()
        .and_then(|value| sanitize_metadata(Some(value.to_string())));
    (content_type, payload)
}

fn decode_base64_payload(raw: &str, payload_kind: &str) -> Result<AudioPayload, ApiError> {
    let (data_url_mime_type, payload) = split_data_url_base64(raw);
    let normalized = payload
        .chars()
        .filter(|value| !value.is_ascii_whitespace())
        .collect::<String>();

    if normalized.is_empty() {
        return Err(ApiError::bad_request(format!(
            "{} payload is empty",
            title_case(payload_kind)
        )));
    }

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(normalized.as_bytes())
        .map_err(|err| {
            ApiError::bad_request(format!("Invalid base64 {payload_kind} payload: {err}"))
        })?;
    if bytes.is_empty() {
        return Err(ApiError::bad_request(format!(
            "{} payload cannot be empty",
            title_case(payload_kind)
        )));
    }

    Ok(AudioPayload {
        bytes,
        source_mime_type: None,
        filename: None,
        data_url_mime_type,
    })
}

fn multipart_audio_error(
    route_label: &str,
    field_name: &str,
    limit_bytes: usize,
    err: MultipartError,
) -> ApiError {
    multipart_upload_api_error(
        route_label,
        field_name,
        limit_bytes,
        err.status(),
        err.body_text(),
    )
}

fn strip_data_url_prefix(raw: &str) -> Option<&str> {
    raw.get(..5)
        .filter(|prefix| prefix.eq_ignore_ascii_case("data:"))
        .map(|_| &raw[5..])
}

fn sanitize_metadata(raw: Option<String>) -> Option<String> {
    raw.filter(|value| !value.contains(['\r', '\n']))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && !value.contains(['\r', '\n']))
}

fn title_case(raw: &str) -> String {
    let mut chars = raw.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn decodes_raw_data_url_and_whitespace_base64_audio() {
        let payload = decode_base64_audio_payload("data:audio/wav;base64, YXVk\naW8= ")
            .expect("payload should decode");

        assert_eq!(payload.bytes, b"audio");
        assert_eq!(payload.data_url_mime_type.as_deref(), Some("audio/wav"));
        assert_eq!(payload.content_type_hint(), Some("audio/wav"));
        assert_eq!(payload.to_base64(), "YXVkaW8=");
    }

    #[test]
    fn rejects_empty_and_invalid_audio_payloads() {
        assert!(decode_base64_audio_payload("   ").is_err());
        assert!(decode_base64_audio_payload("not base64").is_err());
    }

    #[test]
    fn keeps_file_metadata_hints_clean() {
        let payload = AudioPayload::from_bytes(
            b"audio".to_vec(),
            Some(" audio/webm ".to_string()),
            Some(" capture.webm ".to_string()),
        );

        assert_eq!(payload.source_mime_type.as_deref(), Some("audio/webm"));
        assert_eq!(payload.filename.as_deref(), Some("capture.webm"));
    }

    #[test]
    fn drops_unsafe_metadata_hints() {
        let payload = AudioPayload::from_bytes(
            b"audio".to_vec(),
            Some("audio/wav\ntext/plain".to_string()),
            Some("\rfile.wav".to_string()),
        );

        assert_eq!(payload.source_mime_type, None);
        assert_eq!(payload.filename, None);
    }

    #[test]
    fn split_data_url_reports_content_type() {
        assert_eq!(
            split_data_url_base64("data:audio/wav;base64,YXVkaW8="),
            (Some("audio/wav".to_string()), "YXVkaW8=")
        );
    }

    #[test]
    fn optional_base64_ignores_empty_fields() {
        assert_eq!(
            decode_optional_base64_audio_payload("   ")
                .expect("empty optional field should be accepted"),
            None
        );
    }

    #[test]
    fn media_payload_uses_media_error_wording() {
        let err = decode_base64_media_payload("not base64")
            .expect_err("invalid media payload should fail");
        assert!(err.message.contains("Invalid base64 media payload"));
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn detects_audio_content_types() {
        assert!(is_audio_content_type("audio/webm;codecs=opus"));
        assert!(is_audio_content_type(" AUDIO/WAV "));
        assert!(!is_audio_content_type("video/mp4"));
        assert!(!is_audio_content_type("application/octet-stream"));
    }
}
