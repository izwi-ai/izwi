use std::convert::Infallible;

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::api::request_context::RequestContext;
use crate::app::chat::{
    generate_chat, parse_chat_model, spawn_chat_stream, ChatExecutionRequest, ChatStreamEvent,
};
use crate::chat_store::{ChatThreadMessage, ChatThreadSummary};
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{ChatMessage, ChatRole};

#[derive(Debug, Serialize)]
pub struct ChatThreadListResponse {
    pub threads: Vec<ChatThreadSummary>,
}

#[derive(Debug, Deserialize)]
pub struct CreateChatThreadRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatThreadDetailResponse {
    pub thread: ChatThreadSummary,
    pub messages: Vec<ChatThreadMessage>,
}

#[derive(Debug, Serialize)]
pub struct DeleteChatThreadResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateChatThreadRequest {
    pub title: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct CreateThreadMessageRequest {
    pub model: String,
    pub content: String,
    #[serde(default)]
    pub content_parts: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatGenerationStats {
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateThreadMessageResponse {
    pub thread_id: String,
    pub model_id: String,
    pub user_message: ChatThreadMessage,
    pub assistant_message: ChatThreadMessage,
    pub stats: ChatGenerationStats,
}

#[derive(Debug, Serialize)]
struct ThreadStreamStartEvent {
    event: &'static str,
    thread_id: String,
    model_id: String,
    user_message: ChatThreadMessage,
}

#[derive(Debug, Serialize)]
struct ThreadStreamDeltaEvent {
    event: &'static str,
    delta: String,
}

#[derive(Debug, Serialize)]
struct ThreadStreamDoneEvent {
    event: &'static str,
    thread_id: String,
    model_id: String,
    assistant_message: ChatThreadMessage,
    stats: ChatGenerationStats,
}

#[derive(Debug, Serialize)]
struct ThreadStreamErrorEvent {
    event: &'static str,
    error: String,
}

pub async fn list_threads(
    State(state): State<AppState>,
) -> Result<Json<ChatThreadListResponse>, ApiError> {
    let threads = state
        .chat_store
        .list_threads()
        .await
        .map_err(map_store_error)?;

    Ok(Json(ChatThreadListResponse { threads }))
}

pub async fn create_thread(
    State(state): State<AppState>,
    Json(req): Json<CreateChatThreadRequest>,
) -> Result<Json<ChatThreadSummary>, ApiError> {
    let thread = state
        .chat_store
        .create_thread(req.title, req.model_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(thread))
}

pub async fn get_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<ChatThreadDetailResponse>, ApiError> {
    let thread = get_thread_or_not_found(&state, &thread_id).await?;
    let messages = state
        .chat_store
        .list_messages(thread_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(ChatThreadDetailResponse { thread, messages }))
}

pub async fn list_thread_messages(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<Vec<ChatThreadMessage>>, ApiError> {
    get_thread_or_not_found(&state, &thread_id).await?;
    let messages = state
        .chat_store
        .list_messages(thread_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(messages))
}

pub async fn delete_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<DeleteChatThreadResponse>, ApiError> {
    let deleted = state
        .chat_store
        .delete_thread(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("Thread not found"));
    }

    Ok(Json(DeleteChatThreadResponse {
        id: thread_id,
        deleted: true,
    }))
}

pub async fn update_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    Json(req): Json<UpdateChatThreadRequest>,
) -> Result<Json<ChatThreadSummary>, ApiError> {
    if req.title.trim().is_empty() {
        return Err(ApiError::bad_request("Thread title cannot be empty"));
    }

    let updated = state
        .chat_store
        .update_thread_title(thread_id, req.title)
        .await
        .map_err(map_store_error)?;

    let thread = updated.ok_or_else(|| ApiError::not_found("Thread not found"))?;
    Ok(Json(thread))
}

pub async fn create_thread_message(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<CreateThreadMessageRequest>,
) -> Result<Response, ApiError> {
    let model_variant = parse_chat_model(&req.model)?;
    let model_id = model_variant.dir_name().to_string();
    if content_parts_contain_media(req.content_parts.as_deref()) {
        return Err(ApiError::bad_request(
            "Image/video inputs in threaded chat are not currently supported",
        ));
    }

    let prepared_content_parts = req.content_parts.clone();
    let flattened_content = flatten_thread_content(&req.content, prepared_content_parts.as_deref())
        .map_err(|err| {
            ApiError::bad_request(format!("Invalid chat message content payload: {err}"))
        })?;
    if flattened_content.runtime.trim().is_empty() {
        return Err(ApiError::bad_request("Message content cannot be empty"));
    }

    get_thread_or_not_found(&state, &thread_id).await?;
    let existing_messages = state
        .chat_store
        .list_messages(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    let user_message = state
        .chat_store
        .append_message(
            thread_id.clone(),
            "user".to_string(),
            flattened_content.display.clone(),
            prepared_content_parts.clone(),
            Some(model_id.clone()),
            None,
            None,
        )
        .await
        .map_err(map_store_or_not_found)?;

    let runtime_messages = build_runtime_messages(
        &existing_messages,
        &flattened_content.runtime,
        req.system_prompt.as_deref(),
    )?;
    let execution_request = ChatExecutionRequest {
        variant: model_variant,
        messages: runtime_messages,
        max_completion_tokens: req.max_completion_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        presence_penalty: None,
        correlation_id: Some(ctx.correlation_id),
    };

    if req.stream.unwrap_or(false) {
        return create_streaming_thread_message(
            state,
            model_id,
            thread_id,
            user_message,
            execution_request,
        )
        .await;
    }

    let generation = generate_chat(&state, execution_request).await?;

    let assistant_message = state
        .chat_store
        .append_message(
            thread_id.clone(),
            "assistant".to_string(),
            generation.text.clone(),
            None,
            Some(model_id.clone()),
            Some(generation.tokens_generated),
            Some(generation.generation_time_ms),
        )
        .await
        .map_err(map_store_or_not_found)?;

    let response = CreateThreadMessageResponse {
        thread_id,
        model_id,
        user_message,
        assistant_message,
        stats: ChatGenerationStats {
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        },
    };

    Ok(Json(response).into_response())
}

async fn create_streaming_thread_message(
    state: AppState,
    model_id: String,
    thread_id: String,
    user_message: ChatThreadMessage,
    execution_request: ChatExecutionRequest,
) -> Result<Response, ApiError> {
    let chat_store = state.chat_store.clone();
    let thread_id_for_task = thread_id.clone();
    let model_id_for_task = model_id.clone();
    let user_message_for_start = user_message.clone();
    let mut event_rx = spawn_chat_stream(state, execution_request);

    let stream = async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            let (payload, terminal) = match event {
                ChatStreamEvent::Started => (
                    serde_json::to_string(&ThreadStreamStartEvent {
                        event: "start",
                        thread_id: thread_id_for_task.clone(),
                        model_id: model_id_for_task.clone(),
                        user_message: user_message_for_start.clone(),
                    })
                    .unwrap_or_default(),
                    false,
                ),
                ChatStreamEvent::Delta(delta) => (
                    serde_json::to_string(&ThreadStreamDeltaEvent {
                        event: "delta",
                        delta,
                    })
                    .unwrap_or_default(),
                    false,
                ),
                ChatStreamEvent::Completed(generation) => {
                    let payload = match chat_store
                        .append_message(
                            thread_id_for_task.clone(),
                            "assistant".to_string(),
                            generation.text.clone(),
                            None,
                            Some(model_id_for_task.clone()),
                            Some(generation.tokens_generated),
                            Some(generation.generation_time_ms),
                        )
                        .await
                    {
                        Ok(assistant_message) => serde_json::to_string(&ThreadStreamDoneEvent {
                            event: "done",
                            thread_id: thread_id_for_task.clone(),
                            model_id: model_id_for_task.clone(),
                            assistant_message,
                            stats: ChatGenerationStats {
                                tokens_generated: generation.tokens_generated,
                                generation_time_ms: generation.generation_time_ms,
                            },
                        })
                        .unwrap_or_default(),
                        Err(err) => serde_json::to_string(&ThreadStreamErrorEvent {
                            event: "error",
                            error: format!("Failed to persist assistant message: {err}"),
                        })
                        .unwrap_or_default(),
                    };
                    (payload, true)
                }
                ChatStreamEvent::Failed(error) => (
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error,
                    })
                    .unwrap_or_default(),
                    true,
                ),
                ChatStreamEvent::TimedOut => (
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: "Chat request timed out".to_string(),
                    })
                    .unwrap_or_default(),
                    true,
                ),
                ChatStreamEvent::ShuttingDown => (
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: "Server is shutting down".to_string(),
                    })
                    .unwrap_or_default(),
                    true,
                ),
            };
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
            if terminal {
                break;
            }
        }
        yield Ok::<_, Infallible>("data: [DONE]\n\n".to_string());
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

fn build_runtime_messages(
    existing: &[ChatThreadMessage],
    new_user_content: &str,
    system_prompt: Option<&str>,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut messages = Vec::new();

    if let Some(prompt) = system_prompt
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
    {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: prompt.to_string(),
        });
    }

    for message in existing {
        let role = parse_stored_role(&message.role)?;
        messages.push(ChatMessage {
            role,
            content: message.content.clone(),
        });
    }

    messages.push(ChatMessage {
        role: ChatRole::User,
        content: new_user_content.to_string(),
    });

    Ok(messages)
}

#[derive(Debug, Default)]
struct FlattenedThreadContent {
    display: String,
    runtime: String,
}

fn flatten_thread_content(
    raw_content: &str,
    content_parts: Option<&[serde_json::Value]>,
) -> Result<FlattenedThreadContent, String> {
    let raw_trimmed = raw_content.trim().to_string();
    let Some(parts) = content_parts else {
        return Ok(FlattenedThreadContent {
            display: raw_trimmed.clone(),
            runtime: raw_trimmed,
        });
    };

    if parts.is_empty() {
        return Ok(FlattenedThreadContent {
            display: raw_trimmed.clone(),
            runtime: raw_trimmed,
        });
    }

    let mut out = FlattenedThreadContent::default();
    for part in parts {
        if content_part_is_image(part) {
            return Err(
                "Image/video inputs are not currently supported for chat messages".to_string(),
            );
        }
        if content_part_is_video(part) {
            return Err(
                "Image/video inputs are not currently supported for chat messages".to_string(),
            );
        }
        if let Some(text) = resolve_text_part(part) {
            out.runtime.push_str(&text);
            out.display.push_str(&text);
        }
    }

    if out.runtime.trim().is_empty() && !raw_trimmed.is_empty() {
        out.runtime = raw_trimmed.clone();
        out.display = raw_trimmed;
    }

    Ok(out)
}

fn resolve_text_part(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        serde_json::Value::Object(map) => map
            .get("text")
            .or_else(|| map.get("input_text"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        _ => None,
    }
}

fn content_part_is_image(part: &serde_json::Value) -> bool {
    let Some(map) = part.as_object() else {
        return false;
    };
    if matches!(
        map.get("type")
            .or_else(|| map.get("kind"))
            .and_then(|v| v.as_str()),
        Some("image") | Some("image_url") | Some("input_image")
    ) {
        return true;
    }
    map.contains_key("image") || map.contains_key("image_url") || map.contains_key("input_image")
}

fn content_part_is_video(part: &serde_json::Value) -> bool {
    let Some(map) = part.as_object() else {
        return false;
    };
    if matches!(
        map.get("type")
            .or_else(|| map.get("kind"))
            .and_then(|v| v.as_str()),
        Some("video") | Some("video_url") | Some("input_video")
    ) {
        return true;
    }
    map.contains_key("video") || map.contains_key("video_url") || map.contains_key("input_video")
}

fn content_parts_contain_media(content_parts: Option<&[serde_json::Value]>) -> bool {
    let Some(parts) = content_parts else {
        return false;
    };
    parts
        .iter()
        .any(|part| content_part_is_image(part) || content_part_is_video(part))
}

fn parse_stored_role(role: &str) -> Result<ChatRole, ApiError> {
    match role {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        other => Err(ApiError::internal(format!(
            "Invalid stored chat role: {other}"
        ))),
    }
}

async fn get_thread_or_not_found(
    state: &AppState,
    thread_id: &str,
) -> Result<ChatThreadSummary, ApiError> {
    let thread = state
        .chat_store
        .get_thread(thread_id.to_string())
        .await
        .map_err(map_store_error)?;

    thread.ok_or_else(|| ApiError::not_found("Thread not found"))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Chat storage error: {err}"))
}

fn map_store_or_not_found(err: anyhow::Error) -> ApiError {
    let error_text = err.to_string();
    if error_text.contains("Thread not found") {
        ApiError::not_found("Thread not found")
    } else {
        map_store_error(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flattens_thread_text_parts() {
        let flattened = flatten_thread_content(
            "",
            Some(&[
                json!({"type":"text","text":"Look "}),
                json!({"type":"text","text":" now"}),
            ]),
        )
        .expect("flatten thread content");

        assert_eq!(flattened.runtime, "Look  now");
        assert_eq!(flattened.display, "Look  now");
    }

    #[test]
    fn flatten_thread_content_rejects_media_parts() {
        let err = flatten_thread_content("", Some(&[json!({"type":"image_url","image_url":{}})]))
            .expect_err("media parts should fail");
        assert!(err.contains("not currently supported"));
    }

    #[test]
    fn content_parts_contain_media_detects_multimodal_parts() {
        assert!(!content_parts_contain_media(None));
        assert!(!content_parts_contain_media(Some(&[
            json!({"type":"text","text":"hello"})
        ])));
        assert!(content_parts_contain_media(Some(&[
            json!({"type":"image_url","image_url":{"url":"https://example.com/cat.png"}})
        ])));
        assert!(content_parts_contain_media(Some(&[
            json!({"type":"input_video","input_video":{"url":"https://example.com/clip.mp4"}})
        ])));
    }

    #[test]
    fn build_runtime_messages_appends_existing_messages_and_prompt() {
        let messages = build_runtime_messages(
            &[ChatThreadMessage {
                id: "message-1".to_string(),
                thread_id: "thread-1".to_string(),
                role: "assistant".to_string(),
                content: "Hello".to_string(),
                content_parts: None,
                created_at: 1,
                tokens_generated: None,
                generation_time_ms: None,
                model_id: None,
            }],
            "How are you?",
            Some("Be concise."),
        )
        .expect("runtime messages");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert_eq!(messages[2].role, ChatRole::User);
    }
}
