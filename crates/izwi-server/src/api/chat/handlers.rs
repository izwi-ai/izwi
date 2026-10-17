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
    generate_chat, parse_chat_model, resolve_chat_request_config, start_chat_stream_with_keepalive,
    validate_chat_request_ranges, ChatExecutionRequest, ChatStreamEvent,
};
use crate::app::chat_content::{
    flatten_thread_content, validate_media_inputs_for_variant, FlattenedMultimodalContent,
};
use crate::chat_store::{sanitize_system_prompt, ChatThreadMessage, ChatThreadSummary};
use crate::error::{ApiError, ApiJson};
use crate::state::AppState;
use izwi_core::{
    ChatMediaInput, ChatMessage, ChatReasoningEffort, ChatRole, ChatTemplateKwargs, ChatTruncation,
};

const CHAT_STREAM_INTERRUPTED_ERROR: &str = "Chat stream ended before a terminal event";
const CHAT_EMPTY_RESPONSE_ERROR: &str =
    "The model generated no visible response. Please retry the message.";
const CHAT_EMPTY_TERMINAL_ERROR: &str =
    "Chat stream returned an empty final response after generating text. Please retry the message.";

fn validate_thread_response(text: &str, saw_nonempty_delta: bool) -> Result<(), &'static str> {
    // Reasoning-only output is still valid raw model content. Rendering decides
    // whether it has a visible final answer; do not strip reasoning in the API.
    if text.trim().is_empty() {
        Err(if saw_nonempty_delta {
            CHAT_EMPTY_TERMINAL_ERROR
        } else {
            CHAT_EMPTY_RESPONSE_ERROR
        })
    } else {
        Ok(())
    }
}

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
    #[serde(default)]
    pub reasoning_effort: Option<ChatReasoningEffort>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// `"auto"` drops the thread's oldest turns when it outgrows the model's
    /// context; `"disabled"` (the default) rejects the message with
    /// `context_length_exceeded`. The chat UI opts in.
    #[serde(default)]
    pub truncation: Option<ChatTruncation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatGenerationStats {
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
    /// The engine's per-phase timing (queue, prefill, decode wall time and
    /// tokens, TTFT). The chat UI derives decode speed from it, because
    /// `generation_time_ms` includes queue wait and prompt processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<izwi_core::engine::LatencyBreakdown>,
}

impl ChatGenerationStats {
    fn from_generation(generation: &izwi_core::ChatGeneration) -> Self {
        Self {
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
            timing: generation.latency_breakdown.clone(),
        }
    }
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

fn thread_stream_error_payload(error: impl Into<String>) -> String {
    serde_json::to_string(&ThreadStreamErrorEvent {
        event: "error",
        error: error.into(),
    })
    .unwrap_or_default()
}

fn thread_stream_interruption_payload(saw_terminal: bool) -> Option<String> {
    (!saw_terminal).then(|| thread_stream_error_payload(CHAT_STREAM_INTERRUPTED_ERROR))
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
    ApiJson(req): ApiJson<CreateChatThreadRequest>,
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
    ApiJson(req): ApiJson<UpdateChatThreadRequest>,
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
    ApiJson(req): ApiJson<CreateThreadMessageRequest>,
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

    let chat_config = resolve_chat_request_config(
        req.enable_thinking,
        req.reasoning_effort,
        req.preserve_thinking,
        req.chat_template_kwargs.as_ref(),
        Vec::new(),
        media_inputs,
        req.truncation,
    )?;
    let execution_request = ChatExecutionRequest {
        variant: model_variant,
        messages: runtime_messages,
        max_completion_tokens: req.max_completion_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        presence_penalty: req.presence_penalty,
        seed: None,
        stop: Vec::new(),
        logprobs: None,
        top_logprobs: None,
        response_format_json_object: false,
        skip_special_tokens: None,
        chat_config,
        correlation_id: Some(ctx.correlation_id),
    };
    validate_chat_request_ranges(&execution_request)?;

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
    validate_thread_response(&generation.text, false).map_err(ApiError::internal)?;

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
        stats: ChatGenerationStats::from_generation(&generation),
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
    let correlation_id = execution_request.correlation_id.clone();
    let (mut event_rx, stream_completion) =
        start_chat_stream_with_keepalive(state, execution_request, turn_guard).await?;

    let stream = async_stream::stream! {
        let mut stream_completion = Some(stream_completion);
        let mut saw_terminal = false;
        let mut saw_nonempty_delta = false;
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
                ChatStreamEvent::Delta { text, .. } => {
                    saw_nonempty_delta |= !text.trim().is_empty();
                    (
                        serde_json::to_string(&ThreadStreamDeltaEvent {
                            event: "delta",
                            delta: text,
                        })
                        .unwrap_or_default(),
                        false,
                    )
                }
                ChatStreamEvent::Completed(generation) => {
                    let payload = if let Err(error) =
                        validate_thread_response(&generation.text, saw_nonempty_delta)
                    {
                        thread_stream_error_payload(error)
                    } else {
                        match chat_store
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
                                stats: ChatGenerationStats::from_generation(&generation),
                            })
                            .unwrap_or_default(),
                            Err(err) => {
                                tracing::error!(
                                    thread_id = %thread_id_for_task,
                                    correlation_id = correlation_id.as_deref().unwrap_or(""),
                                    error = %format!("{err:#}"),
                                    "failed to persist streamed chat turn"
                                );
                                thread_stream_error_payload(format!(
                                    "Failed to persist assistant message: {err:#}"
                                ))
                            }
                        }
                    };
                    (payload, true)
                }
                ChatStreamEvent::Failed(error) => (
                    thread_stream_error_payload(error.message),
                    true,
                ),
            };
            if terminal {
                saw_terminal = true;
                if let Some(completion) = stream_completion.take() {
                    completion.acknowledge();
                }
            }
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
            if terminal {
                break;
            }
        }
        if let Some(payload) = thread_stream_interruption_payload(saw_terminal) {
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
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
        if role == ChatRole::Assistant
            && flattened.runtime_text.trim().is_empty()
            && !flattened.has_media()
        {
            // Older versions persisted empty successful responses. They are
            // not model answers and must not become assistant prompt turns.
            continue;
        }
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
    tracing::error!(error = %format!("{err:#}"), "chat storage error");
    ApiError::internal(format!("Chat storage error: {err:#}"))
}

fn map_store_or_not_found(err: anyhow::Error) -> ApiError {
    if crate::error::StoreError::find(&err).is_some() {
        crate::error::store_api_error(err, "Chat storage error")
    } else {
        map_store_error(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn generation_with_timing(
        timing: Option<izwi_core::engine::LatencyBreakdown>,
    ) -> izwi_core::ChatGeneration {
        izwi_core::ChatGeneration {
            latency_breakdown: timing,
            finish_reason: None,
            text: "done".to_string(),
            prompt_tokens: 12,
            tokens_generated: 40,
            generation_time_ms: 2_000.0,
            cached_prompt_tokens: None,
            logprobs: Vec::new(),
        }
    }

    #[test]
    fn thread_stats_carry_the_engine_timing_for_decode_speed() {
        let timing: izwi_core::engine::LatencyBreakdown = serde_json::from_value(json!({
            "queue_wait_ms": 300.0,
            "prefill_ms": 1_200.0,
            "decode_ms": 380.0,
            "decode_wall_ms": 400.0,
            "decode_tokens": 39,
            "ttft_ms": 1_550.0,
            "total_ms": 2_000.0,
            "prefill_steps": 2,
            "decode_steps": 39,
        }))
        .expect("timing fixture");

        let stats = ChatGenerationStats::from_generation(&generation_with_timing(Some(timing)));
        let value = serde_json::to_value(&stats).expect("stats serialize");
        assert_eq!(value["tokens_generated"], 40);
        assert_eq!(value["generation_time_ms"], 2_000.0);
        assert_eq!(value["timing"]["decode_wall_ms"], 400.0);
        assert_eq!(value["timing"]["decode_tokens"], 39);
        assert_eq!(value["timing"]["ttft_ms"], 1_550.0);

        let untimed = ChatGenerationStats::from_generation(&generation_with_timing(None));
        let value = serde_json::to_value(&untimed).expect("stats serialize");
        assert!(value.get("timing").is_none(), "{value}");
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
    fn closed_thread_stream_emits_explicit_error_without_terminal_event() {
        let payload = thread_stream_interruption_payload(false)
            .expect("an interrupted stream should emit an error");
        let json: serde_json::Value = serde_json::from_str(&payload).expect("valid error payload");

        assert_eq!(
            json.get("event").and_then(|value| value.as_str()),
            Some("error")
        );
        assert_eq!(
            json.get("error").and_then(|value| value.as_str()),
            Some(CHAT_STREAM_INTERRUPTED_ERROR)
        );
        assert!(thread_stream_interruption_payload(true).is_none());
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
    fn empty_thread_responses_fail_in_both_modes_without_stripping_reasoning() {
        assert_eq!(
            validate_thread_response(" \n", false),
            Err(CHAT_EMPTY_RESPONSE_ERROR)
        );
        assert_eq!(
            validate_thread_response("", true),
            Err(CHAT_EMPTY_TERMINAL_ERROR)
        );
        for streamed in [false, true] {
            assert!(validate_thread_response("<think>reasoning only</think>", streamed).is_ok());
            assert!(validate_thread_response("Visible answer", streamed).is_ok());
        }
    }

    #[test]
    fn runtime_history_skips_legacy_empty_assistant_entries() {
        let existing = ["", " \n", "<think>reasoning</think>", "Answer"]
            .iter()
            .enumerate()
            .map(|(index, content)| ChatThreadMessage {
                id: format!("message-{index}"),
                thread_id: "thread-1".into(),
                role: "assistant".into(),
                content: (*content).into(),
                content_parts: None,
                created_at: index as u64,
                tokens_generated: Some(48),
                generation_time_ms: Some(224.0),
            })
            .collect::<Vec<_>>();
        let (messages, _) = build_runtime_messages(
            &existing,
            &FlattenedMultimodalContent {
                display_text: "Next question".into(),
                runtime_text: "Next question".into(),
                media_inputs: Vec::new(),
            },
            None,
        )
        .unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "<think>reasoning</think>");
        assert_eq!(messages[1].content, "Answer");
        assert_eq!(messages[2].role, ChatRole::User);
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

    #[test]
    fn storage_errors_keep_the_underlying_database_cause() {
        let err = anyhow::anyhow!("(code: 517) database is locked")
            .context("Failed to append chat turn message");
        let api_error = map_store_or_not_found(err);
        assert_eq!(api_error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            api_error.message,
            "Chat storage error: Failed to append chat turn message: \
             (code: 517) database is locked"
        );

        let missing =
            map_store_or_not_found(crate::error::StoreError::not_found("Thread not found").into());
        assert_eq!(missing.status, StatusCode::NOT_FOUND);
        assert_eq!(missing.message, "Thread not found");
        // The text alone no longer decides the status.
        let text = map_store_or_not_found(anyhow::anyhow!("Thread not found"));
        assert_eq!(text.status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
