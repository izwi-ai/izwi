use axum::{
    body::Body,
    extract::Path,
    http::{header, StatusCode},
    response::Response,
};

use crate::error::ApiError;
use crate::storage_layout;

pub async fn get_media(Path(path): Path<String>) -> Result<Response, ApiError> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request("Media path cannot be empty"));
    }

    let media_root = storage_layout::resolve_media_root();
    let bytes = storage_layout::read_media_file(&media_root, trimmed).map_err(map_media_error)?;
    let content_type = content_type_from_path(trimmed);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CACHE_CONTROL, "public, max-age=31536000, immutable")
        .body(Body::from(bytes))
        .map_err(|err| ApiError::internal(format!("Failed building media response: {err}")))
}

fn map_media_error(err: anyhow::Error) -> ApiError {
    let message = err.to_string();
    if message.contains("Unsafe media path component")
        || message.contains("Absolute media paths are not allowed")
    {
        return ApiError::bad_request("Invalid media path");
    }

    let lowered = message.to_ascii_lowercase();
    if lowered.contains("no such file")
        || lowered.contains("cannot find the file")
        || lowered.contains("os error 2")
    {
        return ApiError::not_found("Media file not found");
    }

    ApiError::internal(format!("Failed reading media file: {err}"))
}

fn content_type_from_path(path: &str) -> &'static str {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    match extension.as_deref() {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        Some("svg") => "image/svg+xml",
        Some("avif") => "image/avif",
        Some("heic") => "image/heic",
        Some("heif") => "image/heif",
        Some("mp4") => "video/mp4",
        Some("webm") => "video/webm",
        Some("mov") => "video/quicktime",
        Some("avi") => "video/x-msvideo",
        Some("mkv") => "video/x-matroska",
        Some("mpeg") | Some("mpg") => "video/mpeg",
        Some("3gp") => "video/3gpp",
        _ => "application/octet-stream",
    }
}
