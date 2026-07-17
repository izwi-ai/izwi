use std::convert::Infallible;

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::OwnedMutexGuard;

use crate::api::request_context::RequestContext;
use crate::app::chat::{
    generate_chat, parse_chat_model, spawn_chat_stream_with_keepalive, ChatExecutionRequest,
    ChatStreamEvent,
};
use crate::app::chat_content::{
    flatten_thread_content, validate_media_inputs_for_variant, FlattenedMultimodalContent,
};
use crate::chat_store::{sanitize_system_prompt, ChatThreadMessage, ChatThreadSummary};
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{ChatMediaInput, ChatMessage, ChatRequestConfig, ChatRole};

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
    #[serde(default)]
    pub system_prompt: Option<String>,
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
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
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
        .create_thread_with_system_prompt(req.title, req.model_id, req.system_prompt)
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
    if req.title.is_none() && req.system_prompt.is_none() {
        return Err(ApiError::bad_request("No thread settings were provided"));
    }
    if req
        .title
        .as_deref()
        .is_some_and(|title| title.trim().is_empty())
    {
        return Err(ApiError::bad_request("Thread title cannot be empty"));
    }

    let updated = state
        .chat_store
        .update_thread_settings(thread_id, req.title, req.system_prompt)
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
    let prepared_content_parts = req.content_parts.clone();
    let flattened_content = flatten_thread_content(&req.content, prepared_content_parts.as_deref())
        .map_err(|err| {
            ApiError::bad_request(format!("Invalid chat message content payload: {err}"))
        })?;
    if flattened_content.runtime_text.trim().is_empty() && !flattened_content.has_media() {
        return Err(ApiError::bad_request("Message content cannot be empty"));
    }

    // Hold one turn lock across history read, generation, and terminal
    // persistence. Concurrent sends to the same thread therefore observe the
    // preceding assistant response instead of branching from stale history.
    let turn_guard = state.chat_store.lock_turn(&thread_id).await;
    let thread = get_thread_or_not_found(&state, &thread_id).await?;
    let requested_system_prompt = req.system_prompt.clone();
    let effective_system_prompt = match requested_system_prompt.as_deref() {
        Some(prompt) => sanitize_system_prompt(Some(prompt)),
        None => thread.system_prompt.clone(),
    };
    let existing_messages = state
        .chat_store
        .list_messages(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    let (runtime_messages, media_inputs) = build_runtime_messages(
        &existing_messages,
        &flattened_content,
        effective_system_prompt.as_deref(),
    )?;
    validate_media_inputs_for_variant(model_variant, &media_inputs)
        .map_err(ApiError::bad_request)?;

    let user_message = state
        .chat_store
        .prepare_user_message(
            thread_id.clone(),
            flattened_content.display_text.clone(),
            prepared_content_parts.clone(),
        )
        .await
        .map_err(map_store_or_not_found)?;

    let execution_request = ChatExecutionRequest {
        variant: model_variant,
        messages: runtime_messages,
        max_completion_tokens: req.max_completion_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        presence_penalty: None,
        chat_config: ChatRequestConfig {
            enable_thinking: req.enable_thinking,
            tools: Vec::new(),
            media_inputs,
        },
        correlation_id: Some(ctx.correlation_id),
    };

    if req.stream.unwrap_or(false) {
        return create_streaming_thread_message(
            state,
            model_id,
            thread_id,
            user_message,
            execution_request,
            turn_guard,
            requested_system_prompt,
        )
        .await;
    }

    let generation = generate_chat(&state, execution_request).await?;

    let (user_message, assistant_message) = state
        .chat_store
        .append_turn_with_system_prompt(
            user_message,
            generation.text.clone(),
            model_id.clone(),
            generation.tokens_generated,
            generation.generation_time_ms,
            requested_system_prompt,
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
    turn_guard: OwnedMutexGuard<()>,
    system_prompt_update: Option<String>,
) -> Result<Response, ApiError> {
    let chat_store = state.chat_store.clone();
    let thread_id_for_task = thread_id.clone();
    let model_id_for_task = model_id.clone();
    let user_message_for_start = user_message.clone();
    let (mut event_rx, stream_completion) =
        spawn_chat_stream_with_keepalive(state, execution_request, turn_guard);

    let stream = async_stream::stream! {
        let mut stream_completion = Some(stream_completion);
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
                        .append_turn_with_system_prompt(
                            user_message_for_start.clone(),
                            generation.text.clone(),
                            model_id_for_task.clone(),
                            generation.tokens_generated,
                            generation.generation_time_ms,
                            system_prompt_update.clone(),
                        )
                        .await
                    {
                        Ok((_user_message, assistant_message)) => serde_json::to_string(&ThreadStreamDoneEvent {
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
                ChatStreamEvent::ShuttingDown => (
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: "Server is shutting down".to_string(),
                    })
                    .unwrap_or_default(),
                    true,
                ),
            };
            if terminal {
                if let Some(completion) = stream_completion.take() {
                    completion.acknowledge();
                }
            }
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
    new_user_content: &FlattenedMultimodalContent,
    system_prompt: Option<&str>,
) -> Result<(Vec<ChatMessage>, Vec<ChatMediaInput>), ApiError> {
    let mut messages = Vec::new();
    let mut media_inputs = Vec::new();

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
        let flattened = flatten_thread_content(&message.content, message.content_parts.as_deref())
            .map_err(|err| {
                ApiError::internal(format!("Invalid stored chat message payload: {err}"))
            })?;
        media_inputs.extend(flattened.media_inputs);
        messages.push(ChatMessage {
            role,
            content: flattened.runtime_text,
        });
    }

    media_inputs.extend(new_user_content.media_inputs.clone());
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: new_user_content.runtime_text.clone(),
    });

    Ok((messages, media_inputs))
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

        assert_eq!(flattened.runtime_text, "Look  now");
        assert_eq!(flattened.display_text, "Look  now");
    }

    #[test]
    fn flatten_thread_content_collects_media_parts() {
        let flattened = flatten_thread_content(
            "",
            Some(&[json!({"type":"image_url","image_url":{"url":"https://example.com/cat.png"}})]),
        )
        .expect("media parts should flatten");

        assert!(flattened.display_text.is_empty());
        assert!(flattened.runtime_text.contains("<|image_pad|>"));
        assert_eq!(flattened.media_inputs.len(), 1);
    }

    #[test]
    fn build_runtime_messages_appends_existing_messages_and_prompt() {
        let (messages, media_inputs) = build_runtime_messages(
            &[ChatThreadMessage {
                id: "message-1".to_string(),
                thread_id: "thread-1".to_string(),
                role: "assistant".to_string(),
                content: "Hello".to_string(),
                content_parts: None,
                created_at: 1,
                tokens_generated: None,
                generation_time_ms: None,
            }],
            &FlattenedMultimodalContent {
                display_text: "How are you?".to_string(),
                runtime_text: "How are you?".to_string(),
                media_inputs: Vec::new(),
            },
            Some("Be concise."),
        )
        .expect("runtime messages");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert_eq!(messages[2].role, ChatRole::User);
        assert!(media_inputs.is_empty());
    }

    #[test]
    fn build_runtime_messages_collects_media_from_history_and_new_message() {
        let (messages, media_inputs) = build_runtime_messages(
            &[ChatThreadMessage {
                id: "message-1".to_string(),
                thread_id: "thread-1".to_string(),
                role: "user".to_string(),
                content: "".to_string(),
                content_parts: Some(vec![json!({
                    "type":"image_url",
                    "image_url":{"url":"https://example.com/history.png"}
                })]),
                created_at: 1,
                tokens_generated: None,
                generation_time_ms: None,
            }],
            &FlattenedMultimodalContent {
                display_text: "Describe this".to_string(),
                runtime_text: "<|vision_start|><|image_pad|><|vision_end|>Describe this"
                    .to_string(),
                media_inputs: vec![ChatMediaInput {
                    kind: izwi_core::ChatMediaKind::Image,
                    source: "https://example.com/new.png".to_string(),
                }],
            },
            None,
        )
        .expect("runtime messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(media_inputs.len(), 2);
    }
}
