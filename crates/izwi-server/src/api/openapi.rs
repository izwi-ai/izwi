//! OpenAPI document served by the local API server.

use axum::Json;
use serde::Serialize;
use utoipa::{OpenApi, ToSchema};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Izwi API",
        version = env!("CARGO_PKG_VERSION"),
        description = "Local HTTP API for the Izwi runtime, OpenAI-compatible endpoints, and first-party workflow surfaces."
    ),
    servers(
        (url = "/", description = "Local Izwi server")
    ),
    paths(livez, readyz),
    components(schemas(
        ApiErrorBody,
        ApiErrorEnvelope,
        CursorPagination,
        CursorPaginationQuery,
        LiveResponse,
        OpenAiModel,
        OpenAiModelsResponse,
        ProbeCheck,
        ReadyResponse,
    )),
    tags(
        (name = "runtime", description = "Runtime health, readiness, and operational probes"),
        (name = "openai-compatible", description = "OpenAI-compatible API surface"),
        (name = "admin", description = "Local administrative API surface"),
        (name = "workflows", description = "First-party persisted workflow API surface")
    )
)]
pub struct IzwiOpenApi;

pub async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(document())
}

pub fn document() -> utoipa::openapi::OpenApi {
    IzwiOpenApi::openapi()
}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/livez",
    tag = "runtime",
    responses(
        (status = 200, description = "Server process is alive", body = LiveResponse)
    )
)]
fn livez() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/readyz",
    tag = "runtime",
    responses(
        (status = 200, description = "Server is ready to serve requests", body = ReadyResponse),
        (status = 503, description = "Server is alive but not ready", body = ReadyResponse)
    )
)]
fn readyz() {}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ApiErrorEnvelope {
    pub error: ApiErrorBody,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ProbeCheck {
    pub name: String,
    pub ok: bool,
    pub message: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct LiveResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ReadyResponse {
    pub status: String,
    pub version: String,
    pub ready: bool,
    pub phase: String,
    pub draining: bool,
    pub uptime_secs: u64,
    pub checks: Vec<ProbeCheck>,
    pub startup_warnings: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAiModelsResponse {
    pub object: String,
    pub data: Vec<OpenAiModel>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAiModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub root: Option<String>,
    pub parent: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct CursorPaginationQuery {
    pub limit: Option<usize>,
    pub cursor: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct CursorPagination {
    pub next_cursor: Option<String>,
    pub has_more: bool,
    pub limit: usize,
}
