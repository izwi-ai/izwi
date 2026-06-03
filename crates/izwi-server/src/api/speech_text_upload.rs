use axum::{extract::multipart::MultipartError, http::StatusCode};

use crate::error::ApiError;

/// First-party transcription and diarization upload limit while multipart audio
/// fields are still collected in bounded memory before being written to media
/// storage.
///
/// Do not raise this limit until the ingestion path streams file fields directly
/// to temp/media storage. Background processing consumes the stored object.
pub(crate) const FIRST_PARTY_AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn resolve_source_audio_mime_type(
    raw_mime_type: Option<&str>,
    filename: Option<&str>,
) -> String {
    let raw = raw_mime_type.map(str::trim).unwrap_or_default();
    if !raw.is_empty() && !raw.eq_ignore_ascii_case("application/octet-stream") {
        return raw.to_string();
    }

    let extension = filename
        .and_then(|value| value.rsplit_once('.').map(|(_, extension)| extension))
        .map(|extension| extension.trim().to_ascii_lowercase());

    match extension.as_deref() {
        Some("wav") | Some("wave") => "audio/wav",
        Some("mp3") | Some("mpeg") | Some("mpga") => "audio/mpeg",
        Some("m4a") | Some("mp4") => "audio/mp4",
        Some("webm") => "audio/webm",
        Some("ogg") | Some("oga") | Some("opus") => "audio/ogg",
        Some("flac") => "audio/flac",
        Some("aac") => "audio/aac",
        _ if !raw.is_empty() => raw,
        _ => "application/octet-stream",
    }
    .to_string()
}

pub(crate) fn multipart_upload_error(
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

pub(crate) fn multipart_upload_api_error(
    route_label: &str,
    field_name: &str,
    limit_bytes: usize,
    status: StatusCode,
    raw: impl AsRef<str>,
) -> ApiError {
    let raw = raw.as_ref();
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        let limit_mb = limit_bytes / (1024 * 1024);
        return ApiError {
            status,
            message: format!(
                "Failed reading multipart '{}' field: {}. {} uploads are limited to {} MiB; upload the original compressed file when possible.",
                field_name, raw, route_label, limit_mb
            ),
        };
    }

    let lowered = raw.to_ascii_lowercase();
    if lowered.contains("multipart/form-data") {
        let limit_mb = limit_bytes / (1024 * 1024);
        return ApiError {
            status,
            message: format!(
                "Failed reading multipart '{}' field: {}. \
This is commonly caused by oversized uploads or malformed multipart boundaries. \
Ensure `Content-Type` includes a valid boundary (let your HTTP client set it automatically for FormData) and keep payload under {} MiB.",
                field_name, raw, limit_mb
            ),
        };
    }

    ApiError {
        status,
        message: format!("Failed reading multipart '{}' field: {}", field_name, raw),
    }
}

#[cfg(test)]
mod tests {
    use super::{FIRST_PARTY_AUDIO_UPLOAD_LIMIT_BYTES, resolve_source_audio_mime_type};

    #[test]
    fn first_party_upload_limit_stays_bounded_while_ingestion_is_buffered() {
        assert_eq!(FIRST_PARTY_AUDIO_UPLOAD_LIMIT_BYTES, 64 * 1024 * 1024);
    }

    #[test]
    fn keeps_specific_source_audio_mime_type() {
        assert_eq!(
            resolve_source_audio_mime_type(Some("audio/webm;codecs=opus"), Some("audio.webm")),
            "audio/webm;codecs=opus"
        );
    }

    #[test]
    fn infers_source_audio_mime_type_from_filename_for_generic_uploads() {
        assert_eq!(
            resolve_source_audio_mime_type(Some("application/octet-stream"), Some("meeting.mp3")),
            "audio/mpeg"
        );
        assert_eq!(
            resolve_source_audio_mime_type(None, Some("interview.m4a")),
            "audio/mp4"
        );
    }

    #[test]
    fn uses_octet_stream_when_source_media_type_is_unknown() {
        assert_eq!(
            resolve_source_audio_mime_type(None, Some("recording.bin")),
            "application/octet-stream"
        );
    }
}
