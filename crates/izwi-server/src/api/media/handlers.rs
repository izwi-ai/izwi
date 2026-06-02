use axum::{
    Json,
    body::Body,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::Response,
};
use izwi_hooks::{HookMetadata, MediaNamespace};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path as FsPath, PathBuf},
    time::UNIX_EPOCH,
};

use crate::api::audio_payload::{
    decode_base64_media_payload, inspect_audio_payload_bytes_with_diagnostics,
    is_audio_content_type,
};
use crate::error::ApiError;
use crate::persistence::{
    MediaStorageError, delete_media_object, persist_audio_object, read_media_object,
};
use crate::state::AppState;
use crate::storage_layout;

#[derive(Debug, Deserialize)]
pub struct ListMediaQuery {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct UploadMediaRequest {
    #[serde(default, alias = "audio_base64")]
    data_base64: Option<String>,
    #[serde(default, alias = "mime_type")]
    content_type: Option<String>,
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MediaObjectSummary {
    pub path: String,
    pub url: String,
    pub content_type: String,
    pub size_bytes: u64,
    pub modified_at: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ListMediaResponse {
    pub media: Vec<MediaObjectSummary>,
}

#[derive(Debug, Serialize)]
pub struct UploadMediaResponse {
    pub path: String,
    pub url: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct DeleteMediaResponse {
    pub path: String,
    pub deleted: bool,
}

pub async fn list_media(
    State(state): State<AppState>,
    Query(query): Query<ListMediaQuery>,
) -> Result<Json<ListMediaResponse>, ApiError> {
    let media_root = listable_media_root(&state)?;
    let media = list_local_media_files(&media_root, query.limit.unwrap_or(100).clamp(1, 500))
        .map_err(|err| ApiError::internal(format!("Failed listing media files: {err}")))?;
    Ok(Json(ListMediaResponse { media }))
}

fn listable_media_root(state: &AppState) -> Result<PathBuf, ApiError> {
    match state.persistence.as_ref() {
        Some(persistence) => persistence
            .local_media_root()
            .cloned()
            .ok_or_else(|| ApiError {
                status: StatusCode::NOT_IMPLEMENTED,
                message:
                    "GET /v1/media listing is only available for local media storage providers"
                        .to_string(),
            }),
        None => Ok(storage_layout::resolve_media_root()),
    }
}

pub async fn create_media(
    State(state): State<AppState>,
    Json(req): Json<UploadMediaRequest>,
) -> Result<Json<UploadMediaResponse>, ApiError> {
    let raw_data = req
        .data_base64
        .as_deref()
        .ok_or_else(|| ApiError::bad_request("Missing data_base64 media payload"))?;
    let payload = decode_base64_media_payload(raw_data)?;
    let data_url_content_type = payload.data_url_mime_type.clone();
    let bytes = payload.bytes;

    let content_type = normalize_content_type(
        req.content_type
            .or(data_url_content_type)
            .as_deref()
            .unwrap_or_else(|| {
                req.filename
                    .as_deref()
                    .map(content_type_from_path)
                    .unwrap_or("application/octet-stream")
            }),
    )
    .ok_or_else(|| ApiError::bad_request("Invalid content_type"))?;
    if is_audio_content_type(&content_type) {
        inspect_audio_payload_bytes_with_diagnostics(
            "media.upload",
            &bytes,
            Some(content_type.as_str()),
            req.filename.as_deref(),
        )?;
    }
    let namespace = sanitize_media_namespace(req.namespace.as_deref());
    let record_id = uuid::Uuid::new_v4().to_string();
    let filename = req.filename.as_deref().and_then(normalize_filename);
    let path = if let Some(persistence) = state.persistence.as_ref() {
        let provider = persistence.media_storage();
        let mut metadata = HookMetadata::new();
        metadata.insert("route".to_string(), "/v1/media".to_string());
        persist_audio_object(
            &provider,
            MediaNamespace::Other(format!("media/{namespace}")),
            record_id,
            filename.as_deref(),
            &content_type,
            &bytes,
            metadata,
        )
        .await
        .map_err(map_media_error)?
    } else {
        let media_root = storage_layout::resolve_media_root();
        storage_layout::persist_audio_file(
            &media_root,
            storage_layout::MediaGroup::Uploads,
            &namespace,
            &record_id,
            filename.as_deref(),
            &content_type,
            &bytes,
        )
        .map_err(map_media_error)?
    };

    Ok(Json(UploadMediaResponse {
        url: media_url(&path),
        path,
        content_type,
        filename,
        size_bytes: bytes.len() as u64,
    }))
}

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

pub async fn delete_media(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Result<Json<DeleteMediaResponse>, ApiError> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request("Media path cannot be empty"));
    }

    if let Some(persistence) = state.persistence.as_ref() {
        let provider = persistence.media_storage();
        read_media_object(&provider, trimmed)
            .await
            .map_err(map_media_error)?;
        delete_media_object(&provider, Some(trimmed))
            .await
            .map_err(map_media_error)?;
    } else {
        let media_root = storage_layout::resolve_media_root();
        storage_layout::read_media_file(&media_root, trimmed).map_err(map_media_error)?;
        storage_layout::delete_media_file(&media_root, Some(trimmed)).map_err(map_media_error)?;
    }

    Ok(Json(DeleteMediaResponse {
        path: trimmed.to_string(),
        deleted: true,
    }))
}

fn map_media_error(err: impl Into<anyhow::Error>) -> ApiError {
    let err = err.into();
    if err.chain().any(|cause| {
        cause
            .downcast_ref::<MediaStorageError>()
            .is_some_and(MediaStorageError::is_not_found)
    }) {
        return ApiError::not_found("Media file not found");
    }

    if err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    }) {
        return ApiError::not_found("Media file not found");
    }

    if err.to_string().contains("Audio payload cannot be empty") {
        return ApiError::bad_request("Media payload cannot be empty");
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

fn list_local_media_files(
    media_root: &FsPath,
    limit: usize,
) -> anyhow::Result<Vec<MediaObjectSummary>> {
    if !media_root.exists() {
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();
    collect_media_paths(media_root, media_root, &mut paths)?;
    paths.sort_by(|left, right| right.modified_at.cmp(&left.modified_at));
    paths.truncate(limit);
    Ok(paths)
}

fn collect_media_paths(
    media_root: &FsPath,
    dir: &FsPath,
    out: &mut Vec<MediaObjectSummary>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_media_paths(media_root, &path, out)?;
            continue;
        }
        if !metadata.is_file() {
            continue;
        }

        let relative = path
            .strip_prefix(media_root)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        let modified_at = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis() as u64);
        out.push(MediaObjectSummary {
            url: media_url(&relative),
            content_type: content_type_from_path(&relative).to_string(),
            path: relative,
            size_bytes: metadata.len(),
            modified_at,
        });
    }
    Ok(())
}

fn media_url(path: &str) -> String {
    format!("/v1/media/{path}")
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

fn normalize_filename(filename: &str) -> Option<String> {
    let trimmed = filename.trim();
    if trimmed.is_empty()
        || trimmed.contains(['\r', '\n'])
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed == "."
        || trimmed == ".."
    {
        return None;
    }
    Some(trimmed.chars().take(180).collect())
}

fn sanitize_media_namespace(namespace: Option<&str>) -> String {
    let cleaned = namespace
        .unwrap_or("api")
        .split('/')
        .filter(|segment| {
            !segment.is_empty()
                && segment.len() <= 80
                && segment
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
        })
        .collect::<Vec<_>>()
        .join("/");

    if cleaned.is_empty() {
        "api".to_string()
    } else {
        cleaned
    }
}

fn content_type_from_path(path: &str) -> &'static str {
    storage_layout::content_type_from_media_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;

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
    fn media_upload_helpers_sanitize_untrusted_fields() {
        assert_eq!(
            crate::api::audio_payload::split_data_url_base64("data:audio/wav;base64,YXVkaW8="),
            (Some("audio/wav".to_string()), "YXVkaW8=")
        );
        assert_eq!(normalize_filename("../secret.wav"), None);
        assert_eq!(
            sanitize_media_namespace(Some("tenant-1/session_a")),
            "tenant-1/session_a"
        );
        assert_eq!(sanitize_media_namespace(Some("../bad")), "bad");
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

    #[test]
    fn nested_io_not_found_maps_to_404() {
        let error = Err::<(), _>(std::io::Error::from(std::io::ErrorKind::NotFound))
            .context("Failed to read media file")
            .unwrap_err();

        let api_error = map_media_error(error);

        assert_eq!(api_error.status, StatusCode::NOT_FOUND);
    }
}
