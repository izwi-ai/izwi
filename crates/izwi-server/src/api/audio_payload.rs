use axum::{
    extract::multipart::{Field, MultipartError},
    http::StatusCode,
};
use base64::Engine;
use izwi_core::audio::{AudioInspection, inspect_audio_bytes};
use serde::Serialize;
use tracing::info;

use crate::api::speech_text_upload::multipart_upload_api_error;
use crate::error::ApiError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AudioPayload {
    pub bytes: Vec<u8>,
    pub source_mime_type: Option<String>,
    pub filename: Option<String>,
    pub data_url_mime_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct AudioIngestDiagnostics {
    pub route: String,
    pub source_bytes: usize,
    pub source_mime_type: Option<String>,
    pub data_url_mime_type: Option<String>,
    pub filename_extension: Option<String>,
    pub decoded_sample_rate: u32,
    pub decoded_channels: u16,
    pub decoded_sample_count: usize,
    pub decoded_duration_secs: f32,
    pub peak: f32,
    pub rms: f32,
    pub clipped_samples: usize,
    pub clipped_ratio: f32,
    pub resampler: &'static str,
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

    #[cfg(test)]
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
    mut field: Field<'_>,
    route_label: &str,
    field_name: &str,
    limit_bytes: usize,
) -> Result<Option<AudioPayload>, ApiError> {
    let source_mime_type = field.content_type().map(str::to_string);
    let filename = field.file_name().map(str::to_string);
    let mut bytes = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|err| multipart_audio_error(route_label, field_name, limit_bytes, err))?
    {
        if bytes.len().saturating_add(chunk.len()) > limit_bytes {
            return Err(audio_field_too_large(route_label, field_name, limit_bytes));
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.is_empty() {
        return Ok(None);
    }
    Ok(Some(AudioPayload::from_bytes(
        bytes,
        source_mime_type,
        filename,
    )))
}

pub(crate) async fn read_multipart_audio_base64_payload(
    mut field: Field<'_>,
    route_label: &str,
    field_name: &str,
    decoded_limit_bytes: usize,
) -> Result<Option<AudioPayload>, ApiError> {
    let encoded_limit_bytes = encoded_base64_limit_bytes(decoded_limit_bytes);
    let mut text = String::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|err| multipart_audio_error(route_label, field_name, encoded_limit_bytes, err))?
    {
        if text.len().saturating_add(chunk.len()) > encoded_limit_bytes {
            return Err(audio_field_too_large(
                route_label,
                field_name,
                encoded_limit_bytes,
            ));
        }
        let chunk_text = std::str::from_utf8(&chunk).map_err(|_| {
            ApiError::bad_request(format!(
                "Invalid multipart '{field_name}' field: expected UTF-8 base64 text"
            ))
        })?;
        text.push_str(chunk_text);
    }
    decode_optional_base64_audio_payload(text.as_str())
}

pub(crate) fn inspect_audio_payload(payload: &AudioPayload) -> Result<AudioInspection, ApiError> {
    inspect_audio_payload_bytes(&payload.bytes)
}

pub(crate) fn inspect_audio_payload_with_diagnostics(
    route: &str,
    payload: &AudioPayload,
) -> Result<AudioInspection, ApiError> {
    let inspection = inspect_audio_payload(payload)?;
    AudioIngestDiagnostics::from_payload(route, payload, &inspection).emit();
    Ok(inspection)
}

pub(crate) fn inspect_audio_payload_bytes(bytes: &[u8]) -> Result<AudioInspection, ApiError> {
    inspect_audio_bytes(bytes).map_err(|err| {
        ApiError::bad_request(format!(
            "Invalid audio payload: failed to decode audio metadata: {err}"
        ))
    })
}

pub(crate) fn inspect_audio_payload_bytes_with_diagnostics(
    route: &str,
    bytes: &[u8],
    source_mime_type: Option<&str>,
    filename: Option<&str>,
) -> Result<AudioInspection, ApiError> {
    let inspection = inspect_audio_payload_bytes(bytes)?;
    AudioIngestDiagnostics::from_parts(
        route,
        bytes.len(),
        source_mime_type,
        None,
        filename,
        &inspection,
    )
    .emit();
    Ok(inspection)
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

impl AudioIngestDiagnostics {
    fn from_payload(route: &str, payload: &AudioPayload, inspection: &AudioInspection) -> Self {
        Self::from_parts(
            route,
            payload.bytes.len(),
            payload.source_mime_type.as_deref(),
            payload.data_url_mime_type.as_deref(),
            payload.filename.as_deref(),
            inspection,
        )
    }

    fn from_parts(
        route: &str,
        source_bytes: usize,
        source_mime_type: Option<&str>,
        data_url_mime_type: Option<&str>,
        filename: Option<&str>,
        inspection: &AudioInspection,
    ) -> Self {
        let clipped_ratio = if inspection.sample_count == 0 {
            0.0
        } else {
            inspection.clipped_samples as f32 / inspection.sample_count as f32
        };

        Self {
            route: route.to_string(),
            source_bytes,
            source_mime_type: source_mime_type.map(ToOwned::to_owned),
            data_url_mime_type: data_url_mime_type.map(ToOwned::to_owned),
            filename_extension: filename.and_then(filename_extension),
            decoded_sample_rate: inspection.sample_rate,
            decoded_channels: 1,
            decoded_sample_count: inspection.sample_count,
            decoded_duration_secs: inspection.duration_secs,
            peak: inspection.peak,
            rms: inspection.rms,
            clipped_samples: inspection.clipped_samples,
            clipped_ratio,
            resampler: "none",
        }
    }

    fn emit(&self) {
        info!(
            target: "izwi.audio",
            route = self.route.as_str(),
            source_bytes = self.source_bytes,
            source_mime_type = self.source_mime_type.as_deref().unwrap_or(""),
            data_url_mime_type = self.data_url_mime_type.as_deref().unwrap_or(""),
            filename_extension = self.filename_extension.as_deref().unwrap_or(""),
            decoded_sample_rate = self.decoded_sample_rate,
            decoded_channels = self.decoded_channels,
            decoded_sample_count = self.decoded_sample_count,
            decoded_duration_secs = self.decoded_duration_secs,
            peak = self.peak,
            rms = self.rms,
            clipped_samples = self.clipped_samples,
            clipped_ratio = self.clipped_ratio,
            resampler = self.resampler,
            "audio ingest diagnostics"
        );
    }
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

fn audio_field_too_large(route_label: &str, field_name: &str, limit_bytes: usize) -> ApiError {
    multipart_upload_api_error(
        route_label,
        field_name,
        limit_bytes,
        StatusCode::PAYLOAD_TOO_LARGE,
        "field exceeded configured audio upload limit",
    )
}

fn encoded_base64_limit_bytes(decoded_limit_bytes: usize) -> usize {
    decoded_limit_bytes
        .saturating_mul(4)
        .saturating_div(3)
        .saturating_add(8 * 1024)
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

fn filename_extension(filename: &str) -> Option<String> {
    let basename = filename
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(filename)
        .trim();
    let extension = basename.rsplit_once('.')?.1.trim().to_ascii_lowercase();
    if extension.is_empty()
        || extension.len() > 16
        || !extension.chars().all(|ch| ch.is_ascii_alphanumeric())
    {
        return None;
    }
    Some(extension)
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

    #[test]
    fn ingest_diagnostics_report_safe_metadata_without_filename() {
        let payload = AudioPayload::from_bytes(
            b"RIFF".to_vec(),
            Some(" audio/wav ".to_string()),
            Some("private session.Name.WAV".to_string()),
        );
        let inspection = AudioInspection::from_mono_samples(&[0.0, 0.5, -1.0, 1.0], 4);

        let diagnostics =
            AudioIngestDiagnostics::from_payload("transcription.create", &payload, &inspection);

        assert_eq!(diagnostics.route, "transcription.create");
        assert_eq!(diagnostics.source_bytes, 4);
        assert_eq!(diagnostics.source_mime_type.as_deref(), Some("audio/wav"));
        assert_eq!(diagnostics.filename_extension.as_deref(), Some("wav"));
        assert_eq!(diagnostics.decoded_sample_rate, 4);
        assert_eq!(diagnostics.decoded_channels, 1);
        assert_eq!(diagnostics.clipped_samples, 2);
        assert_eq!(diagnostics.clipped_ratio, 0.5);

        let serialized = serde_json::to_value(&diagnostics).expect("serialize diagnostics");
        assert!(serialized.get("filename").is_none());
        assert_eq!(serialized["filename_extension"], "wav");
    }

    #[test]
    fn filename_extension_rejects_path_like_or_unsafe_extensions() {
        assert_eq!(
            filename_extension("/Users/me/audio.webm"),
            Some("webm".to_string())
        );
        assert_eq!(
            filename_extension("capture.wav.bak"),
            Some("bak".to_string())
        );
        assert_eq!(filename_extension("capture.bad-ext"), None);
        assert_eq!(filename_extension("capture."), None);
    }
}
