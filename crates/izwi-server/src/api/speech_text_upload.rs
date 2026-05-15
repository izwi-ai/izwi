use axum::{extract::multipart::MultipartError, http::StatusCode};

use crate::error::ApiError;

pub(crate) const FIRST_PARTY_AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;

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
