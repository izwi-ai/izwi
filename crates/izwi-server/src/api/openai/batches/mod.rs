//! `/v1/batches`: OpenAI's Batch API over the audio routes.
//!
//! A batch runs the lines of an uploaded JSONL file (`/v1/files`, purpose
//! `batch`) against one endpoint as batch work, within 24 hours, and writes
//! its results to an output file. OpenAI batches chat, embeddings and
//! images; Izwi batches `/v1/audio/transcriptions`,
//! `/v1/audio/translations` and `/v1/audio/speech` in the same shape, so
//! the stock SDK's `client.batches` calls work unchanged:
//!
//! ```json
//! {"custom_id": "call-1", "method": "POST", "url": "/v1/audio/transcriptions",
//!  "body": {"model": "…", "file_id": "file-…", "response_format": "json"}}
//! → {"id": "batch_req_…", "custom_id": "call-1",
//!    "response": {"status_code": 200, "request_id": "batch_req_…",
//!                 "body": {"text": "…"}},
//!    "error": null}
//! ```
//!
//! A line's `body` is the JSON body its synchronous route takes. A
//! transcription or translation line names its audio by `file_id` (a
//! `user_data` file) or carries it as `audio_base64`. A speech line's audio
//! is written to its own `batch_output` file, and its response body is
//! `{"audio_file_id", "format", "bytes"}`. A batch belongs to the principal
//! that created it. One created with `metadata.webhook_url` has an event
//! posted there when it ends (`webhook.rs`).

pub(crate) mod executor;
mod runner;
pub(crate) mod webhook;

pub use executor::spawn as spawn_executor;

use axum::{
    extract::{Extension, Path, Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::openai::files::tenant;
use crate::api::request_context::RequestContext;
use crate::batch_api_store::{BatchRecord, BatchStatus};
use crate::error::{store_api_error, ApiError, ApiJson, ErrorClass};
use crate::file_store::FilePurpose;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/batches", get(list_batches).post(create_batch))
        .route("/batches/{batch_id}", get(get_batch))
        .route("/batches/{batch_id}/cancel", post(cancel_batch))
}

/// The routes a batch can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchEndpoint {
    Transcriptions,
    Translations,
    Speech,
}

impl BatchEndpoint {
    const ALL: [Self; 3] = [Self::Transcriptions, Self::Translations, Self::Speech];

    pub(crate) const fn path(self) -> &'static str {
        match self {
            Self::Transcriptions => "/v1/audio/transcriptions",
            Self::Translations => "/v1/audio/translations",
            Self::Speech => "/v1/audio/speech",
        }
    }

    pub(crate) fn parse(path: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|endpoint| endpoint.path() == path)
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateBatchRequest {
    input_file_id: String,
    endpoint: String,
    completion_window: String,
    #[serde(default)]
    metadata: Option<Value>,
}

/// OpenAI's batch object. Times are epoch seconds; one not reached yet is
/// null.
#[derive(Debug, Serialize)]
pub struct BatchObject {
    id: String,
    object: &'static str,
    endpoint: String,
    errors: Option<Value>,
    input_file_id: String,
    completion_window: &'static str,
    status: BatchStatus,
    output_file_id: Option<String>,
    error_file_id: Option<String>,
    created_at: i64,
    in_progress_at: Option<i64>,
    expires_at: i64,
    finalizing_at: Option<i64>,
    completed_at: Option<i64>,
    failed_at: Option<i64>,
    expired_at: Option<i64>,
    cancelling_at: Option<i64>,
    cancelled_at: Option<i64>,
    request_counts: RequestCounts,
    metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
struct RequestCounts {
    total: i64,
    completed: i64,
    failed: i64,
}

impl From<BatchRecord> for BatchObject {
    fn from(batch: BatchRecord) -> Self {
        let secs = |millis: Option<i64>| millis.map(|millis| millis / 1000);
        Self {
            id: batch.id,
            object: "batch",
            endpoint: batch.endpoint,
            errors: batch.errors,
            input_file_id: batch.input_file_id,
            completion_window: COMPLETION_WINDOW,
            status: batch.status,
            output_file_id: batch.output_file_id,
            error_file_id: batch.error_file_id,
            created_at: batch.created_at / 1000,
            in_progress_at: secs(batch.in_progress_at),
            expires_at: batch.expires_at / 1000,
            finalizing_at: secs(batch.finalizing_at),
            completed_at: secs(batch.completed_at),
            failed_at: secs(batch.failed_at),
            expired_at: secs(batch.expired_at),
            cancelling_at: secs(batch.cancelling_at),
            cancelled_at: secs(batch.cancelled_at),
            request_counts: RequestCounts {
                total: batch.request_total,
                completed: batch.request_completed,
                failed: batch.request_failed,
            },
            metadata: batch.metadata,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct BatchList {
    object: &'static str,
    data: Vec<BatchObject>,
    first_id: Option<String>,
    last_id: Option<String>,
    has_more: bool,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    #[serde(default)]
    after: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

const COMPLETION_WINDOW: &str = "24h";
/// OpenAI's limits on `metadata`.
const METADATA_MAX_PAIRS: usize = 16;
const METADATA_MAX_KEY_CHARS: usize = 64;
const METADATA_MAX_VALUE_CHARS: usize = 512;

fn store_error(err: anyhow::Error) -> ApiError {
    store_api_error(err, "Batch storage error")
}

fn no_such_batch(id: &str) -> ApiError {
    ApiError::not_found(format!("No such Batch object: {id}"))
}

fn supported_endpoints() -> String {
    BatchEndpoint::ALL.map(BatchEndpoint::path).join(", ")
}

/// `metadata` as OpenAI takes it: up to 16 string pairs.
fn validate_metadata(metadata: Option<Value>) -> Result<Option<Value>, ApiError> {
    let invalid = || {
        ApiError::invalid_param(
            "metadata",
            format!(
                "`metadata` is an object of at most {METADATA_MAX_PAIRS} string pairs, with keys \
                 up to {METADATA_MAX_KEY_CHARS} and values up to {METADATA_MAX_VALUE_CHARS} \
                 characters"
            ),
        )
    };
    match metadata {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(pairs)) => {
            let valid = pairs.len() <= METADATA_MAX_PAIRS
                && pairs.iter().all(|(key, value)| {
                    key.chars().count() <= METADATA_MAX_KEY_CHARS
                        && value
                            .as_str()
                            .is_some_and(|value| value.chars().count() <= METADATA_MAX_VALUE_CHARS)
                });
            if valid {
                Ok(Some(Value::Object(pairs)))
            } else {
                Err(invalid())
            }
        }
        Some(_) => Err(invalid()),
    }
}

pub async fn create_batch(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(request): ApiJson<CreateBatchRequest>,
) -> Result<Json<BatchObject>, ApiError> {
    let Some(endpoint) = BatchEndpoint::parse(&request.endpoint) else {
        return Err(ApiError::invalid_param(
            "endpoint",
            format!(
                "Unsupported endpoint `{}`. Supported: {}",
                request.endpoint,
                supported_endpoints()
            ),
        ));
    };
    if request.completion_window != COMPLETION_WINDOW {
        return Err(ApiError::invalid_param(
            "completion_window",
            format!(
                "Unsupported completion_window `{}`. Supported: {COMPLETION_WINDOW}",
                request.completion_window
            ),
        ));
    }
    let metadata = validate_metadata(request.metadata)?;
    webhook::validate_metadata(
        crate::env_config::tuning().batch_webhook.as_ref(),
        metadata.as_ref(),
    )?;
    let tenant = tenant(&ctx);
    let input = state
        .file_store
        .get(&tenant, &request.input_file_id)
        .await
        .map_err(store_error)?
        .ok_or_else(|| {
            ApiError::invalid_param(
                "input_file_id",
                format!("No such File object: {}", request.input_file_id),
            )
        })?;
    if input.purpose != FilePurpose::Batch {
        return Err(ApiError::invalid_param(
            "input_file_id",
            format!(
                "File {} has purpose `{}`; a batch's input is uploaded with purpose `batch`",
                input.id,
                input.purpose.as_str()
            ),
        ));
    }
    let batch = state
        .batch_api_store
        .create(&tenant, endpoint.path(), &input.id, metadata.as_ref())
        .await
        .map_err(store_error)?;
    state.batch_api_wake.notify_waiters();
    Ok(Json(batch.into()))
}

pub async fn list_batches(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Query(query): Query<ListQuery>,
) -> Result<Json<BatchList>, ApiError> {
    let (batches, has_more) = state
        .batch_api_store
        .list(
            &tenant(&ctx),
            query.after.as_deref(),
            query.limit.unwrap_or(20),
        )
        .await
        .map_err(store_error)?;
    let data: Vec<BatchObject> = batches.into_iter().map(Into::into).collect();
    Ok(Json(BatchList {
        object: "list",
        first_id: data.first().map(|batch| batch.id.clone()),
        last_id: data.last().map(|batch| batch.id.clone()),
        data,
        has_more,
    }))
}

pub async fn get_batch(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(batch_id): Path<String>,
) -> Result<Json<BatchObject>, ApiError> {
    let batch = state
        .batch_api_store
        .get(&tenant(&ctx), &batch_id)
        .await
        .map_err(store_error)?
        .ok_or_else(|| no_such_batch(&batch_id))?;
    Ok(Json(batch.into()))
}

/// Cancel a batch that is validating or in progress: it turns `cancelling`,
/// the lines in flight stop, and it ends `cancelled` with the results of
/// the lines that finished. Cancelling a batch that is already cancelling
/// returns it as it is; one that has ended, or is writing its output, is a
/// `409`.
pub async fn cancel_batch(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(batch_id): Path<String>,
) -> Result<Json<BatchObject>, ApiError> {
    let batch = state
        .batch_api_store
        .request_cancel(&tenant(&ctx), &batch_id)
        .await
        .map_err(store_error)?
        .ok_or_else(|| no_such_batch(&batch_id))?;
    if batch.status != BatchStatus::Cancelling {
        return Err(ApiError::of(
            ErrorClass::Conflict,
            format!(
                "Batch {batch_id} is `{}` and can no longer be cancelled",
                batch.status.as_str()
            ),
        ));
    }
    state.batch_api_wake.notify_waiters();
    Ok(Json(batch.into()))
}

#[cfg(test)]
mod tests;
