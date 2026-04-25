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
    paths(
        livez,
        readyz,
        list_models,
        get_model,
        create_chat_completion,
        create_speech,
        create_transcription,
        create_response,
        get_response,
        delete_response,
        cancel_response,
        list_response_input_items,
    ),
    components(schemas(
        ApiErrorBody,
        ApiErrorEnvelope,
        ChatCompletionChoice,
        ChatCompletionChunk,
        ChatCompletionDelta,
        ChatCompletionMessage,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionStreamOptions,
        CursorPagination,
        CursorPaginationQuery,
        LiveResponse,
        OpenAiModel,
        OpenAiModelsResponse,
        ProbeCheck,
        ReadyResponse,
        ResponseDeletedObject,
        ResponseInputItemsList,
        ResponseObject,
        ResponsesCreateRequest,
        ServerSentEvent,
        SpeechRequest,
        SpeechStreamEvent,
        TranscriptionJsonRequest,
        TranscriptionMultipartRequest,
        TranscriptionResponse,
        Usage,
        VerboseTranscriptionResponse,
    )),
    tags(
        (name = "runtime", description = "Runtime health, readiness, and operational probes"),
        (name = "openai-compatible", description = "OpenAI-compatible API surface"),
        (name = "openai-preview", description = "Preview OpenAI-compatible API surface"),
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
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "openai-compatible",
    responses(
        (status = 200, description = "List locally available OpenAI-compatible models", body = OpenAiModelsResponse),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn list_models() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/v1/models/{model}",
    tag = "openai-compatible",
    params(
        ("model" = String, Path, description = "Model identifier")
    ),
    responses(
        (status = 200, description = "Retrieve a locally available OpenAI-compatible model", body = OpenAiModel),
        (status = 400, description = "Invalid model identifier", body = ApiErrorEnvelope),
        (status = 404, description = "Model not found", body = ApiErrorEnvelope)
    )
)]
fn get_model() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "openai-compatible",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Chat completion JSON when stream is false; server-sent events when stream is true", body = ChatCompletionResponse),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_chat_completion() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    tag = "openai-compatible",
    request_body = SpeechRequest,
    responses(
        (status = 200, description = "Generated audio bytes, or server-sent audio events when stream_format is sse"),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_speech() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    tag = "openai-compatible",
    request_body(content = TranscriptionMultipartRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Transcription result as JSON, verbose JSON, text, SRT, VTT, or server-sent events", body = TranscriptionResponse),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_transcription() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "openai-preview",
    request_body = ResponsesCreateRequest,
    responses(
        (status = 200, description = "Preview process-local response object, or server-sent events when stream is true", body = ResponseObject),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_response() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/v1/responses/{response_id}",
    tag = "openai-preview",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview process-local stored response record", body = ResponseObject),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn get_response() {}

#[allow(dead_code)]
#[utoipa::path(
    delete,
    path = "/v1/responses/{response_id}",
    tag = "openai-preview",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview deletion result for a process-local response record", body = ResponseDeletedObject),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn delete_response() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/responses/{response_id}/cancel",
    tag = "openai-preview",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview cancellation result for a process-local response record", body = ResponseObject),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn cancel_response() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/v1/responses/{response_id}/input_items",
    tag = "openai-preview",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview input items captured for a process-local response record", body = ResponseInputItemsList),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn list_response_input_items() {}

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

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub stream_options: Option<ChatCompletionStreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<serde_json::Value>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: Option<serde_json::Value>,
    pub enable_thinking: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionStreamOptions {
    pub include_usage: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
    pub izwi_generation_time_ms: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionDelta>,
    pub usage: Option<Usage>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionDelta {
    pub index: usize,
    pub delta: ChatCompletionMessage,
    pub finish_reason: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub speed: Option<f32>,
    pub language: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub top_k: Option<usize>,
    pub stream: Option<bool>,
    pub stream_format: Option<String>,
    pub instructions: Option<String>,
    pub reference_audio: Option<String>,
    pub reference_text: Option<String>,
    pub saved_voice_id: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct SpeechStreamEvent {
    pub event: String,
    pub request_id: Option<String>,
    pub sequence: Option<usize>,
    pub audio_base64: Option<String>,
    pub sample_count: Option<usize>,
    pub is_final: Option<bool>,
    pub sample_rate: Option<u32>,
    pub audio_format: Option<String>,
    pub tokens_generated: Option<usize>,
    pub generation_time_ms: Option<f32>,
    pub audio_duration_secs: Option<f32>,
    pub rtf: Option<f32>,
    pub error: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionMultipartRequest {
    #[schema(value_type = String, format = Binary)]
    pub file: Option<String>,
    pub audio_base64: Option<String>,
    pub model: Option<String>,
    pub language: Option<String>,
    pub response_format: Option<String>,
    pub stream: Option<bool>,
    pub prompt: Option<String>,
    pub temperature: Option<f32>,
    pub timestamp_granularities: Option<Vec<String>>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionJsonRequest {
    pub audio_base64: String,
    pub model: Option<String>,
    pub language: Option<String>,
    pub response_format: Option<String>,
    pub stream: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionResponse {
    pub text: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct VerboseTranscriptionResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration: f32,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub izwi_asr_diagnostics: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponsesCreateRequest {
    pub model: String,
    pub input: Option<serde_json::Value>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub metadata: Option<serde_json::Value>,
    pub user: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub store: Option<bool>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: Option<serde_json::Value>,
    pub enable_thinking: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub output: Vec<serde_json::Value>,
    pub usage: ResponseUsage,
    pub error: Option<ApiErrorBody>,
    pub metadata: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseDeletedObject {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseInputItemsList {
    pub object: String,
    pub data: Vec<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ServerSentEvent {
    pub event: Option<String>,
    pub data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::document;
    use std::collections::HashSet;

    #[test]
    fn openapi_documents_compatibility_contract_endpoints() {
        let contract: serde_json::Value =
            serde_json::from_str(include_str!("../../../../docs/openai-compatibility-contract.json"))
                .expect("contract should parse");
        let supported = contract["scope"]["supported_endpoints"]
            .as_array()
            .expect("supported endpoints should be an array");
        let openapi = serde_json::to_value(document()).expect("openapi should serialize");
        let paths = openapi["paths"].as_object().expect("paths should exist");

        for endpoint in supported {
            let endpoint = endpoint
                .as_str()
                .expect("supported endpoint should be a string");
            let documented = endpoint.replace(":response_id", "{response_id}");
            assert!(
                paths.contains_key(&documented),
                "{documented} should be documented"
            );
        }
    }

    #[test]
    fn openapi_marks_stable_and_preview_methods() {
        let openapi = serde_json::to_value(document()).expect("openapi should serialize");
        let paths = openapi["paths"].as_object().expect("paths should exist");

        let expected = [
            ("/v1/models", "get"),
            ("/v1/models/{model}", "get"),
            ("/v1/chat/completions", "post"),
            ("/v1/audio/speech", "post"),
            ("/v1/audio/transcriptions", "post"),
            ("/v1/responses", "post"),
            ("/v1/responses/{response_id}", "get"),
            ("/v1/responses/{response_id}", "delete"),
            ("/v1/responses/{response_id}/cancel", "post"),
            ("/v1/responses/{response_id}/input_items", "get"),
        ];

        for (path, method) in expected {
            assert!(
                paths
                    .get(path)
                    .and_then(|operations| operations.get(method))
                    .is_some(),
                "{method} {path} should be documented"
            );
        }

        let preview_paths: HashSet<&str> = [
            "/v1/responses",
            "/v1/responses/{response_id}",
            "/v1/responses/{response_id}/cancel",
            "/v1/responses/{response_id}/input_items",
        ]
        .into_iter()
        .collect();

        for path in preview_paths {
            let operations = paths.get(path).expect("preview path should exist");
            let has_preview_tag = operations
                .as_object()
                .expect("operations should be an object")
                .values()
                .any(|operation| {
                    operation["tags"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|tag| tag.as_str() == Some("openai-preview"))
                });
            assert!(has_preview_tag, "{path} should be tagged preview");
        }
    }
}
