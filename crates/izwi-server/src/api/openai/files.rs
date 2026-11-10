//! `/v1/files`: OpenAI's Files API for the Batch API.
//!
//! A client uploads a batch's JSONL input (`purpose: "batch"`) and any audio
//! its lines name by `file_id` (`purpose: "user_data"`), and downloads what
//! the batch wrote (`batch_output`, server-written, kept 30 days). Other
//! OpenAI purposes are refused. A file belongs to the principal that
//! created it; every other principal gets `404`.

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Extension, Multipart, Path, Query, State},
    http::{header, StatusCode},
    response::Response,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

use crate::api::request_context::RequestContext;
use crate::artifact_store::ArtifactTenant;
use crate::error::{store_api_error, ApiError};
use crate::file_store::{FileListQuery, FilePurpose, FileSpool, StoredFile};
use crate::state::AppState;

/// Multipart framing around the file itself.
const MULTIPART_SLACK: usize = 1024 * 1024;
const CONTENT_CHUNK_BYTES: usize = 64 * 1024;
/// The purposes a client may upload, for error messages.
const UPLOAD_PURPOSES: &str = "batch, user_data";

pub fn router() -> Router<AppState> {
    let upload_limit = usize::try_from(crate::env_config::tuning().files_max_bytes)
        .unwrap_or(usize::MAX)
        .saturating_add(MULTIPART_SLACK);
    Router::new()
        .route(
            "/files",
            get(list_files)
                .post(create_file)
                .layer(DefaultBodyLimit::max(upload_limit)),
        )
        .route("/files/{file_id}", get(get_file).delete(delete_file))
        .route("/files/{file_id}/content", get(get_file_content))
}

#[derive(Debug, Serialize)]
pub struct FileObject {
    id: String,
    object: &'static str,
    bytes: u64,
    /// Epoch seconds, as OpenAI reports times.
    created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
    filename: String,
    purpose: &'static str,
    /// Deprecated by OpenAI, still required by its SDK types.
    status: &'static str,
}

impl From<StoredFile> for FileObject {
    fn from(file: StoredFile) -> Self {
        Self {
            id: file.id,
            object: "file",
            bytes: file.size_bytes,
            created_at: file.created_at / 1000,
            expires_at: file.expires_at.map(|at| at / 1000),
            filename: file.filename,
            purpose: file.purpose.as_str(),
            status: "processed",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FileList {
    object: &'static str,
    data: Vec<FileObject>,
    has_more: bool,
    first_id: Option<String>,
    last_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeletedFile {
    id: String,
    object: &'static str,
    deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    #[serde(default)]
    purpose: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    after: Option<String>,
    #[serde(default)]
    order: Option<String>,
}

pub(crate) fn tenant(ctx: &RequestContext) -> ArtifactTenant {
    ArtifactTenant::from_scheduling_key(ctx.tenant_key())
}

fn store_error(err: anyhow::Error) -> ApiError {
    store_api_error(err, "File storage error")
}

pub(crate) fn no_such_file(id: &str) -> ApiError {
    ApiError::not_found(format!("No such File object: {id}"))
}

fn multipart_error(field: &str, err: axum::extract::multipart::MultipartError) -> ApiError {
    let message = format!("Failed reading multipart '{field}' field: {err}");
    if err.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::payload_too_large(message)
    } else {
        ApiError::bad_request(message)
    }
}

/// The name a file is stored under: the upload's base name, without control
/// characters, at most 255 bytes.
pub(crate) fn sanitize_filename(raw: Option<&str>) -> String {
    let base = raw
        .unwrap_or_default()
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default();
    let mut name: String = base.chars().filter(|ch| !ch.is_control()).collect();
    while name.len() > 255 {
        name.pop();
    }
    if name.is_empty() || name == "." || name == ".." {
        return "upload".to_string();
    }
    name
}

/// Stream the `file` field to a temporary file, and refuse it with `413`
/// once it passes `max_bytes`. Returns the upload and its name.
async fn spool_upload(
    mut field: axum::extract::multipart::Field<'_>,
    max_bytes: u64,
) -> Result<(String, FileSpool), ApiError> {
    let filename = sanitize_filename(field.file_name());
    let spool_error =
        |err: std::io::Error| ApiError::internal(format!("Failed to spool the upload: {err}"));
    let mut spool = FileSpool::new().map_err(spool_error)?;
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|err| multipart_error("file", err))?
    {
        if spool.size_bytes().saturating_add(chunk.len() as u64) > max_bytes {
            return Err(ApiError::payload_too_large(format!(
                "The file is larger than {max_bytes} bytes (IZWI_FILES_MAX_BYTES)"
            )));
        }
        spool.write(&chunk).await.map_err(spool_error)?;
    }
    Ok((filename, spool))
}

pub async fn create_file(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    mut multipart: Multipart,
) -> Result<Json<FileObject>, ApiError> {
    let mut upload = None;
    let mut purpose = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| multipart_error("field", err))?
    {
        match field.name().unwrap_or_default() {
            "file" => upload = Some(spool_upload(field, state.file_store.max_bytes()).await?),
            "purpose" => {
                purpose = Some(
                    field
                        .text()
                        .await
                        .map_err(|err| multipart_error("purpose", err))?,
                );
            }
            // `expires_after` and anything newer: uploads are kept until
            // deleted.
            _ => {}
        }
    }
    let purpose = purpose.ok_or_else(|| {
        ApiError::invalid_param(
            "purpose",
            format!("`purpose` is required: one of {UPLOAD_PURPOSES}"),
        )
    })?;
    let purpose = match FilePurpose::parse(purpose.trim()) {
        Some(purpose @ (FilePurpose::Batch | FilePurpose::UserData)) => purpose,
        Some(FilePurpose::BatchOutput) => {
            return Err(ApiError::invalid_param(
                "purpose",
                format!(
                    "`batch_output` files are written by the server. Upload purposes: {UPLOAD_PURPOSES}"
                ),
            ))
        }
        None => {
            return Err(ApiError::invalid_param(
                "purpose",
                format!(
                    "Unsupported purpose `{}`. Supported: {UPLOAD_PURPOSES}",
                    purpose.trim()
                ),
            ))
        }
    };
    let (filename, spool) =
        upload.ok_or_else(|| ApiError::invalid_param("file", "`file` is required: the upload"))?;
    if spool.size_bytes() == 0 {
        return Err(ApiError::invalid_param("file", "The file is empty"));
    }
    let file = state
        .file_store
        .create(&tenant(&ctx), purpose, filename, spool)
        .await
        .map_err(store_error)?;
    Ok(Json(file.into()))
}

pub async fn list_files(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Query(query): Query<ListQuery>,
) -> Result<Json<FileList>, ApiError> {
    let purpose = match query.purpose.as_deref() {
        None => None,
        Some(purpose) => Some(FilePurpose::parse(purpose).ok_or_else(|| {
            ApiError::invalid_param(
                "purpose",
                format!(
                    "Unsupported purpose `{purpose}`. Supported: {UPLOAD_PURPOSES}, batch_output"
                ),
            )
        })?),
    };
    let ascending = match query.order.as_deref() {
        None | Some("desc") => false,
        Some("asc") => true,
        Some(other) => {
            return Err(ApiError::invalid_param(
                "order",
                format!("`order` is `asc` or `desc`, not `{other}`"),
            ))
        }
    };
    let (files, has_more) = state
        .file_store
        .list(
            &tenant(&ctx),
            FileListQuery {
                purpose,
                after: query.after,
                limit: query.limit.unwrap_or(10_000),
                ascending,
            },
        )
        .await
        .map_err(store_error)?;
    let data: Vec<FileObject> = files.into_iter().map(Into::into).collect();
    Ok(Json(FileList {
        object: "list",
        first_id: data.first().map(|file| file.id.clone()),
        last_id: data.last().map(|file| file.id.clone()),
        data,
        has_more,
    }))
}

pub async fn get_file(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(file_id): Path<String>,
) -> Result<Json<FileObject>, ApiError> {
    let file = state
        .file_store
        .get(&tenant(&ctx), &file_id)
        .await
        .map_err(store_error)?
        .ok_or_else(|| no_such_file(&file_id))?;
    Ok(Json(file.into()))
}

pub async fn get_file_content(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(file_id): Path<String>,
) -> Result<Response, ApiError> {
    let (_, content) = state
        .file_store
        .open(&tenant(&ctx), &file_id)
        .await
        .map_err(store_error)?
        .ok_or_else(|| no_such_file(&file_id))?;
    let mut reader = content.reader;
    // No Content-Length: the bytes are verified as they are read, and a
    // failed check must end the body short of a complete response.
    let stream = async_stream::stream! {
        let mut chunk = vec![0u8; CONTENT_CHUNK_BYTES];
        loop {
            match reader.read(&mut chunk).await {
                Ok(0) => break,
                Ok(read) => yield Ok(bytes::Bytes::copy_from_slice(&chunk[..read])),
                Err(err) => {
                    yield Err(err);
                    break;
                }
            }
        }
    };
    Response::builder()
        .header(header::CONTENT_TYPE, crate::file_store::FILE_CONTENT_TYPE)
        .body(Body::from_stream(stream))
        .map_err(|err| ApiError::internal(format!("Failed to build the file response: {err}")))
}

pub async fn delete_file(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(file_id): Path<String>,
) -> Result<Json<DeletedFile>, ApiError> {
    if !state
        .file_store
        .delete(&tenant(&ctx), &file_id)
        .await
        .map_err(store_error)?
    {
        return Err(no_such_file(&file_id));
    }
    Ok(Json(DeletedFile {
        id: file_id,
        object: "file",
        deleted: true,
    }))
}

/// How often expired `batch_output` files are removed.
const EXPIRY_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3600);
const EXPIRY_SWEEP_BATCH: usize = 256;

/// Remove files past their `expires_at`, now and then hourly, until shutdown.
pub fn spawn_expiry_sweep(state: AppState) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut shutdown = state.stream_shutdown.subscribe();
        while !*shutdown.borrow() {
            match state.file_store.delete_expired(EXPIRY_SWEEP_BATCH).await {
                // A full page: more may be due.
                Ok(EXPIRY_SWEEP_BATCH) => continue,
                Ok(0) => {}
                Ok(deleted) => tracing::info!(deleted, "Deleted expired batch output files"),
                Err(err) => tracing::warn!(error = %err, "Failed to delete expired files"),
            }
            tokio::select! {
                _ = shutdown.changed() => {}
                _ = tokio::time::sleep(EXPIRY_SWEEP_INTERVAL) => {}
            }
        }
    })
}

#[cfg(test)]
#[path = "files_tests.rs"]
pub(crate) mod tests;
