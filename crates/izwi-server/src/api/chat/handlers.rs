use std::convert::Infallible;

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::api::request_context::RequestContext;
use crate::app::chat::{
    generate_chat, parse_chat_model, spawn_chat_stream, ChatExecutionRequest, ChatStreamEvent,
};
use crate::app::chat_content::{
    flatten_thread_content, validate_media_inputs_for_variant, FlattenedMultimodalContent,
};
use crate::chat_store::{ChatStore, ChatThreadMessage, ChatThreadSummary};
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{ChatGeneration, ChatMediaInput, ChatMessage, ChatRequestConfig, ChatRole};

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
    let prepared_content_parts = req.content_parts.clone();
    let flattened_content = flatten_thread_content(&req.content, prepared_content_parts.as_deref())
        .map_err(|err| {
            ApiError::bad_request(format!("Invalid chat message content payload: {err}"))
        })?;
    if flattened_content.runtime_text.trim().is_empty() && !flattened_content.has_media() {
        return Err(ApiError::bad_request("Message content cannot be empty"));
    }

    get_thread_or_not_found(&state, &thread_id).await?;
    let existing_messages = state
        .chat_store
        .list_messages(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    let (runtime_messages, media_inputs) = build_runtime_messages(
        &existing_messages,
        &flattened_content,
        req.system_prompt.as_deref(),
    )?;
    validate_media_inputs_for_variant(model_variant, &media_inputs)
        .map_err(ApiError::bad_request)?;

    let user_message = state.chat_store.prepare_user_message(
        thread_id.clone(),
        flattened_content.display_text.clone(),
        prepared_content_parts.clone(),
    );

    let execution_request = ChatExecutionRequest {
        variant: model_variant,
        messages: runtime_messages,
        max_completion_tokens: req.max_completion_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        presence_penalty: None,
        stop_sequences: Vec::new(),
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
        )
        .await;
    }

    let generation = generate_chat(&state, execution_request).await;
    let (generation, user_message, assistant_message) =
        persist_generated_thread_turn(&state.chat_store, user_message, &model_id, generation)
            .await?;

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

async fn persist_generated_thread_turn(
    chat_store: &ChatStore,
    user_message: ChatThreadMessage,
    model_id: &str,
    generation: Result<ChatGeneration, ApiError>,
) -> Result<(ChatGeneration, ChatThreadMessage, ChatThreadMessage), ApiError> {
    let generation = generation?;
    let (user_message, assistant_message) = chat_store
        .append_turn(
            user_message,
            generation.text.clone(),
            Some(model_id.to_string()),
            generation.tokens_generated,
            generation.generation_time_ms,
        )
        .await
        .map_err(map_store_or_not_found)?;
    Ok((generation, user_message, assistant_message))
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
    let event_rx = spawn_chat_stream(state, execution_request);
    let stream = thread_message_stream(
        chat_store,
        model_id_for_task,
        thread_id_for_task,
        user_message_for_start,
        event_rx,
    );

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

fn thread_message_stream(
    chat_store: std::sync::Arc<ChatStore>,
    model_id: String,
    thread_id: String,
    user_message: ChatThreadMessage,
    mut event_rx: tokio::sync::mpsc::Receiver<ChatStreamEvent>,
) -> impl Stream<Item = Result<String, Infallible>> {
    async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            let (payload, terminal) = match event {
                ChatStreamEvent::Started => (
                    serde_json::to_string(&ThreadStreamStartEvent {
                        event: "start",
                        thread_id: thread_id.clone(),
                        model_id: model_id.clone(),
                        user_message: user_message.clone(),
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
                        .append_turn(
                            user_message.clone(),
                            generation.text.clone(),
                            Some(model_id.clone()),
                            generation.tokens_generated,
                            generation.generation_time_ms,
                        )
                        .await
                    {
                        Ok((persisted_user_message, assistant_message)) => {
                            debug_assert_eq!(persisted_user_message.id, user_message.id);
                            debug_assert_eq!(persisted_user_message.created_at, user_message.created_at);
                            serde_json::to_string(&ThreadStreamDoneEvent {
                                event: "done",
                                thread_id: thread_id.clone(),
                                model_id: model_id.clone(),
                                assistant_message,
                                stats: ChatGenerationStats {
                                    tokens_generated: generation.tokens_generated,
                                    generation_time_ms: generation.generation_time_ms,
                                },
                            })
                            .unwrap_or_default()
                        }
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
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
            if terminal {
                break;
            }
        }
        yield Ok::<_, Infallible>("data: [DONE]\n\n".to_string());
    }
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
    use crate::db::StoreDatabase;
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::mpsc;

    fn setup_stream_store() -> (TempDir, Arc<ChatStore>) {
        let temp_dir = tempfile::tempdir().expect("temp dir should create");
        let store = ChatStore::initialize_with_database(StoreDatabase::new(
            temp_dir.path().join("chat.sqlite3"),
        ));
        (temp_dir, Arc::new(store))
    }

    fn test_generation() -> ChatGeneration {
        ChatGeneration {
            text: "assistant reply".to_string(),
            prompt_tokens: 11,
            tokens_generated: 3,
            generation_time_ms: 4.5,
        }
    }

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

    #[tokio::test]
    async fn completed_stream_atomically_persists_the_wire_visible_user_and_assistant() {
        let (_temp, store) = setup_stream_store();
        let thread = store
            .create_thread(None, Some("old-model".to_string()))
            .await
            .expect("thread should create");
        let pending = store.prepare_user_message(
            thread.id.clone(),
            "hello".to_string(),
            Some(vec![json!({"type":"text","text":"hello"})]),
        );
        let pending_id = pending.id.clone();
        let pending_created_at = pending.created_at;
        let (event_tx, event_rx) = mpsc::channel(4);
        event_tx
            .send(ChatStreamEvent::Started)
            .await
            .expect("start should queue");
        event_tx
            .send(ChatStreamEvent::Completed(test_generation()))
            .await
            .expect("completion should queue");
        drop(event_tx);

        let chunks = thread_message_stream(
            store.clone(),
            "Qwen3.5-4B".to_string(),
            thread.id.clone(),
            pending,
            event_rx,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks should succeed");
        assert_eq!(chunks.len(), 3);
        let start: serde_json::Value = serde_json::from_str(
            chunks[0]
                .strip_prefix("data: ")
                .expect("start data prefix")
                .trim(),
        )
        .expect("start event should be json");
        assert_eq!(start["event"], "start");
        assert_eq!(start["user_message"]["id"], pending_id);
        assert_eq!(
            start["user_message"]["created_at"],
            serde_json::Value::from(pending_created_at)
        );
        let done: serde_json::Value = serde_json::from_str(
            chunks[1]
                .strip_prefix("data: ")
                .expect("done data prefix")
                .trim(),
        )
        .expect("done event should be json");
        assert_eq!(done["event"], "done");

        let messages = store
            .list_messages(thread.id)
            .await
            .expect("messages should list");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].id, pending_id);
        assert_eq!(messages[0].created_at, pending_created_at);
        assert_eq!(done["assistant_message"]["id"], messages[1].id);
    }

    #[tokio::test]
    async fn non_streaming_inference_failure_does_not_persist_the_pending_user() {
        let (_temp, store) = setup_stream_store();
        let thread = store
            .create_thread(None, Some("Qwen3.5-4B".to_string()))
            .await
            .expect("thread should create");
        let pending = store.prepare_user_message(thread.id.clone(), "hello".to_string(), None);

        let err = persist_generated_thread_turn(
            &store,
            pending,
            "Qwen3.5-4B",
            Err(ApiError::internal("inference failed")),
        )
        .await
        .expect_err("inference failure should propagate");
        assert_eq!(err.message, "inference failed");
        assert!(store
            .list_messages(thread.id)
            .await
            .expect("messages should list")
            .is_empty());
    }

    #[tokio::test]
    async fn failed_stream_does_not_persist_an_unmatched_user_message() {
        let (_temp, store) = setup_stream_store();
        let thread = store
            .create_thread(None, Some("Qwen3.5-4B".to_string()))
            .await
            .expect("thread should create");
        let pending = store.prepare_user_message(thread.id.clone(), "hello".to_string(), None);
        let (event_tx, event_rx) = mpsc::channel(4);
        event_tx
            .send(ChatStreamEvent::Started)
            .await
            .expect("start should queue");
        event_tx
            .send(ChatStreamEvent::Failed("inference failed".to_string()))
            .await
            .expect("failure should queue");
        drop(event_tx);

        let chunks = thread_message_stream(
            store.clone(),
            "Qwen3.5-4B".to_string(),
            thread.id.clone(),
            pending,
            event_rx,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks should succeed");
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("inference failed")));
        assert!(store
            .list_messages(thread.id)
            .await
            .expect("messages should list")
            .is_empty());
    }

    #[tokio::test]
    async fn dropped_stream_does_not_persist_the_pending_user_message() {
        let (_temp, store) = setup_stream_store();
        let thread = store
            .create_thread(None, Some("Qwen3.5-4B".to_string()))
            .await
            .expect("thread should create");
        let pending = store.prepare_user_message(thread.id.clone(), "hello".to_string(), None);
        let (event_tx, event_rx) = mpsc::channel(4);
        event_tx
            .send(ChatStreamEvent::Started)
            .await
            .expect("start should queue");

        {
            let stream = thread_message_stream(
                store.clone(),
                "Qwen3.5-4B".to_string(),
                thread.id.clone(),
                pending,
                event_rx,
            );
            futures::pin_mut!(stream);
            let start = stream
                .next()
                .await
                .expect("start chunk")
                .expect("start chunk should succeed");
            assert!(start.contains("\"event\":\"start\""));
        }

        assert!(event_tx
            .send(ChatStreamEvent::Completed(test_generation()))
            .await
            .is_err());
        assert!(store
            .list_messages(thread.id)
            .await
            .expect("messages should list")
            .is_empty());
    }
}
