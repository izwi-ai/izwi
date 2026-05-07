use axum::{
    body::Body,
    extract::{Path, State},
    http::{StatusCode, header},
    response::Response,
};

use crate::error::ApiError;
use crate::persistence::{MediaStorageError, read_media_object};
use crate::state::AppState;
use crate::storage_layout;

pub async fn get_media(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Result<Response, ApiError> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request("Media path cannot be empty"));
    }

    let (bytes, content_type) = if let Some(persistence) = state.persistence.as_ref() {
        let stored = read_media_object(&persistence.media_storage(), trimmed)
            .await
            .map_err(map_media_error)?;
        let content_type = normalize_content_type(&stored.metadata.content_type)
            .unwrap_or_else(|| content_type_from_path(trimmed).to_string());
        (stored.bytes, content_type)
    } else {
        let media_root = storage_layout::resolve_media_root();
        (
            storage_layout::read_media_file(&media_root, trimmed).map_err(map_media_error)?,
            content_type_from_path(trimmed).to_string(),
        )
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type.as_str())
        .header(header::CACHE_CONTROL, "public, max-age=31536000, immutable")
        .body(Body::from(bytes))
        .map_err(|err| ApiError::internal(format!("Failed building media response: {err}")))
}

fn map_media_error(err: impl Into<anyhow::Error>) -> ApiError {
    let err = err.into();
    if err
        .downcast_ref::<MediaStorageError>()
        .is_some_and(MediaStorageError::is_not_found)
    {
        return ApiError::not_found("Media file not found");
    }

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

fn normalize_content_type(content_type: &str) -> Option<String> {
    if content_type.contains(['\r', '\n']) {
        return None;
    }

    let trimmed = content_type.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn content_type_from_path(path: &str) -> &'static str {
    storage_layout::content_type_from_media_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_content_type_is_normalized_for_http_header() {
        assert_eq!(
            normalize_content_type(" audio/custom-codec "),
            Some("audio/custom-codec".to_string())
        );
        assert_eq!(normalize_content_type("\ntext/plain"), None);
        assert_eq!(normalize_content_type(""), None);
    }

    #[test]
    fn media_storage_not_found_maps_to_404() {
        let error = MediaStorageError::NotFound {
            key: "opaque-key".to_string(),
            message: "missing".to_string(),
        };

        let api_error = map_media_error(error);

        assert_eq!(api_error.status, StatusCode::NOT_FOUND);
    }
}
