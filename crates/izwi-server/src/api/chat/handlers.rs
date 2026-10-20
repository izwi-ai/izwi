use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Body,
    extract::{Extension, Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::OwnedMutexGuard;

use crate::api::pagination::{encode_cursor, CursorPagination, CursorPaginationQuery};
use crate::api::request_context::RequestContext;
use crate::app::chat::{
    generate_chat, parse_chat_model, resolve_chat_request_config, start_chat_stream_with_keepalive,
    validate_chat_request_ranges, ChatExecutionRequest, ChatStreamCompletion, ChatStreamEvent,
};
use crate::app::chat_content::{
    flatten_thread_content, validate_media_inputs_for_variant, variant_accepts_image_input,
    FlattenedMultimodalContent,
};
use crate::chat_store::{
    sanitize_system_prompt, ChatStore, ChatThreadListCursor, ChatThreadListQuery,
    ChatThreadMessage, ChatThreadSummary, ChatTurn, TurnPlacement, CANCELLED_FINISH_REASON,
};
use crate::error::{ApiError, ApiJson};
use crate::state::AppState;
use izwi_core::chat_reasoning::reasoning_opened_in_prompt;
use izwi_core::{
    ChatFinishReason, ChatGeneration, ChatMediaInput, ChatMessage, ChatReasoningEffort, ChatRole,
    ChatTemplateKwargs, ChatTruncation,
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
    /// Present when the request asked for a page (`limit`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pagination: Option<CursorPagination>,
    /// Threads matching `q`, ignoring paging.
    pub total_count: usize,
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatThreadListParams {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub cursor: Option<String>,
    /// Case-insensitive search over titles and message text.
    #[serde(default)]
    pub q: Option<String>,
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

/// Generation settings shared by sending, editing and regenerating.
#[derive(Debug, Clone, Deserialize)]
pub struct ThreadGenerationOptions {
    pub model: String,
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

/// A new message, or the new text of an edited one.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateThreadMessageRequest {
    pub content: String,
    #[serde(default)]
    pub content_parts: Option<Vec<serde_json::Value>>,
    #[serde(flatten)]
    pub options: ThreadGenerationOptions,
}

#[derive(Debug, Serialize)]
pub struct DeleteChatMessageResponse {
    pub id: String,
    pub deleted: bool,
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

/// Lists threads, most recently updated first. Without `limit` every
/// matching thread is returned, as before; with it, results are paged by
/// `cursor`.
pub async fn list_threads(
    State(state): State<AppState>,
    Query(params): Query<ChatThreadListParams>,
) -> Result<Json<ChatThreadListResponse>, ApiError> {
    let paging = CursorPaginationQuery {
        limit: params.limit,
        cursor: params.cursor,
    };
    let limit = params.limit.map(|_| paging.resolved_limit(50, 500));
    let page = state
        .chat_store
        .list_threads_page(ChatThreadListQuery {
            search: params.q,
            limit,
            after: paging.decode_cursor::<ChatThreadListCursor>()?,
        })
        .await
        .map_err(map_store_error)?;

    let pagination = limit.map(|limit| CursorPagination {
        has_more: page.next_cursor.is_some(),
        next_cursor: page.next_cursor.as_ref().map(encode_cursor),
        limit,
    });
    Ok(Json(ChatThreadListResponse {
        threads: page.threads,
        pagination,
        total_count: page.total_count,
    }))
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
    let target = TurnTarget::Append {
        content: req.content,
        content_parts: req.content_parts,
    };
    run_thread_turn(state, ctx, thread_id, req.options, target).await
}

/// Rewrites one of the user's messages, drops everything after it and
/// generates a new reply. Nothing changes if generation fails.
pub async fn edit_thread_message(
    State(state): State<AppState>,
    Path((thread_id, message_id)): Path<(String, String)>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<CreateThreadMessageRequest>,
) -> Result<Response, ApiError> {
    let target = TurnTarget::Edit {
        message_id,
        content: req.content,
        content_parts: req.content_parts,
    };
    run_thread_turn(state, ctx, thread_id, req.options, target).await
}

/// Replaces a reply (and anything after it) with a new one generated from
/// the history before it, with the request's model.
pub async fn regenerate_thread_message(
    State(state): State<AppState>,
    Path((thread_id, message_id)): Path<(String, String)>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(options): ApiJson<ThreadGenerationOptions>,
) -> Result<Response, ApiError> {
    run_thread_turn(
        state,
        ctx,
        thread_id,
        options,
        TurnTarget::Regenerate { message_id },
    )
    .await
}

pub async fn delete_thread_message(
    State(state): State<AppState>,
    Path((thread_id, message_id)): Path<(String, String)>,
) -> Result<Json<DeleteChatMessageResponse>, ApiError> {
    // Wait for a running turn so it cannot commit around the deletion.
    let _turn_guard = state.chat_store.lock_turn(&thread_id).await;
    get_thread_or_not_found(&state, &thread_id).await?;
    let deleted = state
        .chat_store
        .delete_message(thread_id, message_id.clone())
        .await
        .map_err(map_store_error)?;
    if !deleted {
        return Err(ApiError::not_found("Message not found"));
    }
    Ok(Json(DeleteChatMessageResponse {
        id: message_id,
        deleted: true,
    }))
}

/// Which turn a request generates.
enum TurnTarget {
    /// A new message after the thread's latest one.
    Append {
        content: String,
        content_parts: Option<Vec<serde_json::Value>>,
    },
    /// A new reply to the user message before this assistant message.
    Regenerate { message_id: String },
    /// This user message with new text, and a new reply.
    Edit {
        message_id: String,
        content: String,
        content_parts: Option<Vec<serde_json::Value>>,
    },
}

/// The history a turn generates from and the user message it commits.
struct ResolvedTurn {
    history: Vec<ChatThreadMessage>,
    user_message: ChatThreadMessage,
    content: FlattenedMultimodalContent,
    placement: TurnPlacement,
}

fn flatten_request_content(
    content: &str,
    content_parts: Option<&[serde_json::Value]>,
) -> Result<FlattenedMultimodalContent, ApiError> {
    let flattened = flatten_thread_content(content, content_parts).map_err(|err| {
        ApiError::bad_request(format!("Invalid chat message content payload: {err}"))
    })?;
    if flattened.runtime_text.trim().is_empty() && !flattened.has_media() {
        return Err(ApiError::bad_request("Message content cannot be empty"));
    }
    Ok(flattened)
}

fn message_index(messages: &[ChatThreadMessage], message_id: &str) -> Result<usize, ApiError> {
    messages
        .iter()
        .position(|message| message.id == message_id)
        .ok_or_else(|| ApiError::not_found("Message not found"))
}

async fn resolve_turn(
    state: &AppState,
    thread_id: &str,
    mut existing: Vec<ChatThreadMessage>,
    target: TurnTarget,
) -> Result<ResolvedTurn, ApiError> {
    match target {
        TurnTarget::Append {
            content,
            content_parts,
        } => {
            let flattened = flatten_request_content(&content, content_parts.as_deref())?;
            let user_message = state
                .chat_store
                .prepare_user_message(
                    thread_id.to_string(),
                    flattened.display_text.clone(),
                    content_parts,
                )
                .await
                .map_err(map_store_or_not_found)?;
            Ok(ResolvedTurn {
                history: existing,
                user_message,
                content: flattened,
                placement: TurnPlacement::Append,
            })
        }
        TurnTarget::Regenerate { message_id } => {
            let index = message_index(&existing, &message_id)?;
            if existing[index].role != "assistant" {
                return Err(ApiError::bad_request("Only a reply can be regenerated"));
            }
            let user_index = existing[..index]
                .iter()
                .rposition(|message| message.role == "user")
                .ok_or_else(|| ApiError::bad_request("This reply has no message to answer"))?;
            existing.truncate(user_index + 1);
            let user_message = existing.pop().expect("the user message was kept");
            let content = flatten_thread_content(
                &user_message.content,
                user_message.content_parts.as_deref(),
            )
            .map_err(|err| {
                ApiError::internal(format!("Invalid stored chat message payload: {err}"))
            })?;
            Ok(ResolvedTurn {
                history: existing,
                user_message,
                content,
                placement: TurnPlacement::ReplaceFromUser,
            })
        }
        TurnTarget::Edit {
            message_id,
            content,
            content_parts,
        } => {
            let index = message_index(&existing, &message_id)?;
            if existing[index].role != "user" {
                return Err(ApiError::bad_request(
                    "Only your own messages can be edited",
                ));
            }
            let flattened = flatten_request_content(&content, content_parts.as_deref())?;
            existing.truncate(index + 1);
            let mut user_message = existing.pop().expect("the edited message was kept");
            user_message.content = flattened.display_text.clone();
            user_message.content_parts = content_parts;
            Ok(ResolvedTurn {
                history: existing,
                user_message,
                content: flattened,
                placement: TurnPlacement::ReplaceFromUser,
            })
        }
    }
}

async fn run_thread_turn(
    state: AppState,
    ctx: RequestContext,
    thread_id: String,
    options: ThreadGenerationOptions,
    target: TurnTarget,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    let model_variant = parse_chat_model(&options.model)?;
    let model_id = model_variant.dir_name().to_string();

    // Hold one turn lock across history read, generation, and terminal
    // persistence. Concurrent sends to the same thread therefore observe the
    // preceding assistant response instead of branching from stale history.
    let turn_guard = state.chat_store.lock_turn(&thread_id).await;
    let thread = get_thread_or_not_found(&state, &thread_id).await?;
    let requested_system_prompt = options.system_prompt.clone();
    let effective_system_prompt = match requested_system_prompt.as_deref() {
        Some(prompt) => sanitize_system_prompt(Some(prompt)),
        None => thread.system_prompt.clone(),
    };
    let existing_messages = state
        .chat_store
        .list_messages(thread_id.clone())
        .await
        .map_err(map_store_error)?;
    let turn = resolve_turn(&state, &thread_id, existing_messages, target).await?;

    // A thread that once used an image model can continue on a text-only
    // one: earlier turns are sent as their text, while a new attachment on
    // this message is still rejected below.
    let (runtime_messages, media_inputs) = build_runtime_messages(
        &turn.history,
        &turn.content,
        effective_system_prompt.as_deref(),
        variant_accepts_image_input(model_variant),
    )?;
    validate_media_inputs_for_variant(model_variant, &media_inputs)
        .map_err(ApiError::bad_request)?;

    let chat_config = resolve_chat_request_config(
        options.enable_thinking,
        options.reasoning_effort,
        options.preserve_thinking,
        options.chat_template_kwargs.as_ref(),
        Vec::new(),
        media_inputs,
        options.truncation,
    )?;
    let execution_request = ChatExecutionRequest {
        variant: model_variant,
        messages: runtime_messages,
        max_completion_tokens: options.max_completion_tokens,
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        repetition_penalty: options.repetition_penalty,
        presence_penalty: options.presence_penalty,
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

    let commit = TurnCommit {
        chat_store: state.chat_store.clone(),
        user_message: turn.user_message,
        model_id,
        placement: turn.placement,
        system_prompt_update: requested_system_prompt,
        reasoning_opened_in_prompt: reasoning_opened_in_prompt(
            model_variant,
            execution_request.chat_config.enable_thinking,
        ),
    };

    if options.stream.unwrap_or(false) {
        return create_streaming_thread_message(
            state,
            thread_id,
            commit,
            execution_request,
            turn_guard,
            started,
        )
        .await;
    }

    let generation = generate_chat(&state, execution_request).await?;
    validate_thread_response(&generation.text, false).map_err(ApiError::internal)?;

    let (user_message, assistant_message) = commit
        .completed(&generation, None)
        .await
        .map_err(map_store_or_not_found)?;

    let response = CreateThreadMessageResponse {
        thread_id,
        model_id: commit.model_id,
        user_message,
        assistant_message,
        stats: ChatGenerationStats::from_generation(&generation),
    };

    Ok(Json(response).into_response())
}

/// What a turn commits once its generation ends.
struct TurnCommit {
    chat_store: Arc<ChatStore>,
    user_message: ChatThreadMessage,
    model_id: String,
    placement: TurnPlacement,
    system_prompt_update: Option<String>,
    /// The prompt opened `<think>`, so the reply streams its reasoning
    /// without the opener.
    reasoning_opened_in_prompt: bool,
}

impl TurnCommit {
    fn turn(
        &self,
        assistant_content: String,
        tokens_generated: Option<usize>,
        generation_time_ms: Option<f64>,
        reasoning_ms: Option<u64>,
        finish_reason: Option<String>,
    ) -> ChatTurn {
        ChatTurn {
            user_message: self.user_message.clone(),
            assistant_content,
            model_id: self.model_id.clone(),
            tokens_generated,
            generation_time_ms,
            reasoning_ms,
            finish_reason,
            system_prompt_update: self.system_prompt_update.clone(),
            placement: self.placement,
        }
    }

    async fn completed(
        &self,
        generation: &ChatGeneration,
        reasoning_ms: Option<u64>,
    ) -> anyhow::Result<(ChatThreadMessage, ChatThreadMessage)> {
        self.chat_store
            .commit_turn(self.turn(
                generation.text.clone(),
                Some(generation.tokens_generated),
                Some(generation.generation_time_ms),
                reasoning_ms,
                chat_finish_reason(generation),
            ))
            .await
    }
}

/// When a streamed reply's reasoning ended: the moment `</think>` first
/// appears, measured from the start of the turn.
struct ReasoningClock {
    started: Instant,
    reasoning_ms: Option<u64>,
    scanned: usize,
}

impl ReasoningClock {
    const CLOSE_TAG: &'static [u8] = b"</think>";

    fn new(started: Instant) -> Self {
        Self {
            started,
            reasoning_ms: None,
            scanned: 0,
        }
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// `text` is the reply so far; only its new tail is scanned.
    fn observe(&mut self, text: &str) {
        if self.reasoning_ms.is_some() {
            return;
        }
        let from = self.scanned.saturating_sub(Self::CLOSE_TAG.len() - 1);
        if text.as_bytes()[from..]
            .windows(Self::CLOSE_TAG.len())
            .any(|window| window.eq_ignore_ascii_case(Self::CLOSE_TAG))
        {
            self.reasoning_ms = Some(self.elapsed_ms());
        }
        self.scanned = text.len();
    }

    /// The reasoning time of a reply that ended with `text`: a reply that
    /// never closed its reasoning reasoned until it ended.
    fn finish(&self, text: &str) -> Option<u64> {
        self.reasoning_ms.or_else(|| {
            text.to_ascii_lowercase()
                .contains("<think>")
                .then(|| self.elapsed_ms())
        })
    }
}

/// Commits the partial reply as a stopped turn when the client goes away
/// before the stream ends (the response body, and so this, is dropped
/// mid-stream). It holds the stream's completion, which keeps the thread's
/// turn lock until the stopped turn is stored.
struct StopOnDisconnect {
    commit: TurnCommit,
    clock: ReasoningClock,
    text: String,
    completion: Option<ChatStreamCompletion>,
}

impl StopOnDisconnect {
    /// The stream ended on its own; nothing is stored on drop.
    fn finish(&mut self) {
        if let Some(completion) = self.completion.take() {
            completion.acknowledge();
        }
    }
}

impl Drop for StopOnDisconnect {
    fn drop(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let mut text = std::mem::take(&mut self.text);
        // A reply stopped before `</think>` is all reasoning. Its opener was
        // in the prompt, so store it: otherwise the reasoning reads as the
        // answer.
        if self.commit.reasoning_opened_in_prompt && !text.is_empty() && !text.contains("</think>")
        {
            text.insert_str(0, "<think>");
        }
        let reasoning_ms = self.clock.finish(&text);
        let generation_time_ms = self.clock.started.elapsed().as_secs_f64() * 1000.0;
        let turn = self.commit.turn(
            text,
            None,
            Some(generation_time_ms),
            reasoning_ms,
            Some(CANCELLED_FINISH_REASON.to_string()),
        );
        let chat_store = self.commit.chat_store.clone();
        runtime.spawn(async move {
            if let Err(err) = chat_store.commit_turn(turn).await {
                tracing::warn!(error = %format!("{err:#}"), "failed to store a stopped chat turn");
            }
            drop(completion);
        });
    }
}

async fn create_streaming_thread_message(
    state: AppState,
    thread_id: String,
    commit: TurnCommit,
    execution_request: ChatExecutionRequest,
    turn_guard: OwnedMutexGuard<()>,
    started: Instant,
) -> Result<Response, ApiError> {
    let thread_id_for_task = thread_id.clone();
    let model_id_for_task = commit.model_id.clone();
    let user_message_for_start = commit.user_message.clone();
    let correlation_id = execution_request.correlation_id.clone();
    let (mut event_rx, stream_completion) =
        start_chat_stream_with_keepalive(state, execution_request, turn_guard).await?;

    let stream = async_stream::stream! {
        let mut turn = StopOnDisconnect {
            commit,
            clock: ReasoningClock::new(started),
            text: String::new(),
            completion: Some(stream_completion),
        };
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
                    turn.text.push_str(&text);
                    turn.clock.observe(&turn.text);
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
                        let reasoning_ms = turn.clock.finish(&generation.text);
                        match turn.commit.completed(&generation, reasoning_ms).await {
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
                turn.finish();
            }
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
            if terminal {
                break;
            }
        }
        // A stream that ends without a terminal event failed; it is not a stop.
        turn.finish();
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

/// Why an assistant reply ended, in the OpenAI vocabulary the UI already
/// knows: `length` at the token limit, `stop` otherwise for a completed
/// reply.
fn chat_finish_reason(generation: &ChatGeneration) -> Option<String> {
    let reason = match generation.finish_reason {
        Some(ChatFinishReason::MaxTokens) => "length",
        Some(ChatFinishReason::Aborted) => "cancelled",
        Some(ChatFinishReason::Error) => "error",
        Some(ChatFinishReason::StopToken | ChatFinishReason::StopSequence) | None => "stop",
    };
    Some(reason.to_string())
}

/// Placeholder for a past image turn when the target model reads text only
/// and the turn has no text of its own.
const OMITTED_IMAGE_TEXT: &str = "[Image attachment]";

fn build_runtime_messages(
    existing: &[ChatThreadMessage],
    new_user_content: &FlattenedMultimodalContent,
    system_prompt: Option<&str>,
    include_history_media: bool,
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
            && (flattened.runtime_text.trim().is_empty()
                || is_unfinished_reasoning(&flattened.runtime_text))
            && !flattened.has_media()
        {
            // Older versions persisted empty successful responses, and a
            // reply stopped while it reasoned has no answer. Neither is a
            // model answer, so neither becomes an assistant prompt turn.
            continue;
        }
        let content = if include_history_media || !flattened.has_media() {
            media_inputs.extend(flattened.media_inputs);
            flattened.runtime_text
        } else if flattened.display_text.trim().is_empty() {
            // runtime_text carries one vision placeholder per image, which
            // must not reach the prompt without its media input.
            OMITTED_IMAGE_TEXT.to_string()
        } else {
            flattened.display_text
        };
        push_runtime_message(&mut messages, role, content);
    }

    media_inputs.extend(new_user_content.media_inputs.clone());
    push_runtime_message(
        &mut messages,
        ChatRole::User,
        new_user_content.runtime_text.clone(),
    );

    Ok((messages, media_inputs))
}

/// A reply stopped before its reasoning closed: `<think>` with no `</think>`.
fn is_unfinished_reasoning(text: &str) -> bool {
    text.trim_start().starts_with("<think>") && !text.contains("</think>")
}

/// Chat templates expect user and assistant turns to alternate. A deleted
/// reply or a stopped turn with no text leaves two user messages in a row,
/// so consecutive user messages are sent as one.
fn push_runtime_message(messages: &mut Vec<ChatMessage>, role: ChatRole, content: String) {
    match messages.last_mut() {
        Some(last) if last.role == role && role == ChatRole::User => {
            last.content.push_str("\n\n");
            last.content.push_str(&content);
        }
        _ => messages.push(ChatMessage { role, content }),
    }
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

pub(super) async fn get_thread_or_not_found(
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

pub(super) fn map_store_error(err: anyhow::Error) -> ApiError {
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
        let existing = [
            "",
            " \n",
            "<think>stopped while reasoning",
            "<think>reasoning</think>",
            "Answer",
        ]
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
            model_id: None,
            finish_reason: None,
            reasoning_ms: None,
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
            true,
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
                model_id: None,
                finish_reason: None,
                reasoning_ms: None,
            }],
            &FlattenedMultimodalContent {
                display_text: "How are you?".to_string(),
                runtime_text: "How are you?".to_string(),
                media_inputs: Vec::new(),
            },
            Some("Be concise."),
            true,
        )
        .expect("runtime messages");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert_eq!(messages[2].role, ChatRole::User);
        assert!(media_inputs.is_empty());
    }

    fn reply(content: &str) -> ChatThreadMessage {
        ChatThreadMessage {
            id: "reply-1".to_string(),
            thread_id: "thread-1".to_string(),
            role: "assistant".to_string(),
            content: content.to_string(),
            content_parts: None,
            created_at: 2,
            tokens_generated: None,
            generation_time_ms: None,
            model_id: None,
            finish_reason: None,
            reasoning_ms: None,
        }
    }

    #[test]
    fn build_runtime_messages_collects_media_from_history_and_new_message() {
        let (messages, media_inputs) = build_runtime_messages(
            &[
                ChatThreadMessage {
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
                    model_id: None,
                    finish_reason: None,
                    reasoning_ms: None,
                },
                reply("A bar chart."),
            ],
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
            true,
        )
        .expect("runtime messages");

        assert_eq!(messages.len(), 3);
        assert_eq!(media_inputs.len(), 2);
    }

    fn image_turn(content: &str) -> ChatThreadMessage {
        ChatThreadMessage {
            id: "message-1".to_string(),
            thread_id: "thread-1".to_string(),
            role: "user".to_string(),
            content: content.to_string(),
            content_parts: Some(vec![
                json!({"type":"text","text": content}),
                json!({"type":"image_url","image_url":{"url":"https://example.com/history.png"}}),
            ]),
            created_at: 1,
            tokens_generated: None,
            generation_time_ms: None,
            model_id: None,
            finish_reason: None,
            reasoning_ms: None,
        }
    }

    #[test]
    fn text_only_models_get_history_image_turns_as_text() {
        let next = FlattenedMultimodalContent {
            display_text: "Now summarise it".to_string(),
            runtime_text: "Now summarise it".to_string(),
            media_inputs: Vec::new(),
        };
        let (messages, media_inputs) = build_runtime_messages(
            &[
                image_turn("What is in this chart?"),
                reply("Sales by month."),
            ],
            &next,
            None,
            false,
        )
        .expect("runtime messages");

        assert!(media_inputs.is_empty());
        assert_eq!(messages[0].content, "What is in this chart?");
        assert!(messages
            .iter()
            .all(|message| !message.content.contains("<|image_pad|>")));

        // The same history keeps its image for an image-capable model.
        let (messages, media_inputs) = build_runtime_messages(
            &[
                image_turn("What is in this chart?"),
                reply("Sales by month."),
            ],
            &next,
            None,
            true,
        )
        .expect("runtime messages");
        assert_eq!(media_inputs.len(), 1);
        assert!(messages[0].content.contains("<|image_pad|>"));
    }

    #[test]
    fn text_only_models_get_a_placeholder_for_image_only_turns() {
        let image_only = ChatThreadMessage {
            content: String::new(),
            content_parts: Some(vec![json!({
                "type":"image_url",
                "image_url":{"url":"https://example.com/history.png"}
            })]),
            ..image_turn("")
        };
        let next = FlattenedMultimodalContent {
            display_text: "Thanks".to_string(),
            runtime_text: "Thanks".to_string(),
            media_inputs: Vec::new(),
        };
        let (messages, media_inputs) =
            build_runtime_messages(&[image_only, reply("A photo.")], &next, None, false)
                .expect("runtime messages");
        assert!(media_inputs.is_empty());
        assert_eq!(messages[0].content, OMITTED_IMAGE_TEXT);
    }

    #[test]
    fn reasoning_ends_when_the_close_tag_arrives_even_split_across_deltas() {
        let started = Instant::now() - std::time::Duration::from_secs(5);
        let mut clock = ReasoningClock::new(started);
        clock.observe("<think>Planning the</th");
        assert_eq!(clock.reasoning_ms, None);
        clock.observe("<think>Planning the</think>Answer");
        let reasoning_ms = clock.reasoning_ms.expect("the close tag ends reasoning");
        assert!(reasoning_ms >= 5_000, "{reasoning_ms}");
        assert_eq!(
            clock.finish("<think>Planning the</think>Answer"),
            Some(reasoning_ms)
        );

        let clock = ReasoningClock::new(started);
        assert!(clock.finish("<think>still reasoning").is_some());
        assert_eq!(clock.finish("A plain answer"), None);
    }

    #[test]
    fn consecutive_user_messages_reach_the_model_as_one() {
        let user = |id: &str, content: &str| ChatThreadMessage {
            id: id.into(),
            thread_id: "thread-1".into(),
            role: "user".into(),
            content: content.into(),
            content_parts: None,
            created_at: 1,
            tokens_generated: None,
            generation_time_ms: None,
            model_id: None,
            finish_reason: None,
            reasoning_ms: None,
        };
        // A stopped turn with no reply text, then a new question.
        let stopped_reply = ChatThreadMessage {
            role: "assistant".into(),
            finish_reason: Some(CANCELLED_FINISH_REASON.into()),
            ..user("m2", "")
        };
        let (messages, _) = build_runtime_messages(
            &[user("m1", "Tell me a story"), stopped_reply],
            &FlattenedMultimodalContent {
                display_text: "A short one".into(),
                runtime_text: "A short one".into(),
                media_inputs: Vec::new(),
            },
            Some("Be brief."),
            true,
        )
        .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].role, ChatRole::User);
        assert_eq!(messages[1].content, "Tell me a story\n\nA short one");
    }

    #[tokio::test]
    async fn a_dropped_stream_stores_its_partial_reply_as_stopped() {
        let _guard = crate::test_support::env_lock();
        let temp_dir = tempfile::tempdir().unwrap();
        std::env::set_var("IZWI_DB_PATH", temp_dir.path().join("izwi.sqlite3"));
        std::env::set_var("IZWI_MEDIA_DIR", temp_dir.path().join("media"));
        let store = Arc::new(ChatStore::initialize().unwrap());
        let thread = store.create_thread(None, None).await.unwrap();
        let pending = store
            .prepare_user_message(thread.id.clone(), "Tell me a story".into(), None)
            .await
            .unwrap();
        let commit = || TurnCommit {
            chat_store: store.clone(),
            user_message: pending.clone(),
            model_id: "Qwen3.5-0.8B".into(),
            placement: TurnPlacement::Append,
            system_prompt_update: None,
            reasoning_opened_in_prompt: false,
        };

        // A stream that ended on its own stores nothing on drop.
        let (completion, completion_rx) = ChatStreamCompletion::channel();
        let mut finished = StopOnDisconnect {
            commit: commit(),
            clock: ReasoningClock::new(Instant::now()),
            text: "Once upon".into(),
            completion: Some(completion),
        };
        finished.finish();
        drop(finished);
        assert!(completion_rx.await.is_ok(), "finish acknowledges");
        assert!(store
            .list_messages(thread.id.clone())
            .await
            .unwrap()
            .is_empty());

        // A dropped one stores the prompt and partial reply before it lets
        // the turn lock go.
        let (completion, completion_rx) = ChatStreamCompletion::channel();
        drop(StopOnDisconnect {
            commit: commit(),
            clock: ReasoningClock::new(Instant::now()),
            text: "Once upon".into(),
            completion: Some(completion),
        });
        let _ = completion_rx.await;
        let messages = store.list_messages(thread.id.clone()).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].id, pending.id);
        assert_eq!(messages[1].content, "Once upon");
        assert_eq!(
            messages[1].finish_reason.as_deref(),
            Some(CANCELLED_FINISH_REASON)
        );

        // Stopped while reasoning after a prompt that opened `<think>`: the
        // opener is stored, so the reply reads as reasoning, not an answer.
        let next = store
            .prepare_user_message(thread.id.clone(), "Another".into(), None)
            .await
            .unwrap();
        let (completion, completion_rx) = ChatStreamCompletion::channel();
        drop(StopOnDisconnect {
            commit: TurnCommit {
                user_message: next,
                reasoning_opened_in_prompt: true,
                ..commit()
            },
            clock: ReasoningClock::new(Instant::now()),
            text: "The user wants a story".into(),
            completion: Some(completion),
        });
        let _ = completion_rx.await;
        let messages = store.list_messages(thread.id.clone()).await.unwrap();
        let stopped = messages.last().unwrap();
        assert_eq!(stopped.content, "<think>The user wants a story");
        assert!(stopped.reasoning_ms.is_some());
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
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
