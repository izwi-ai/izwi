//! OpenAI-compatible chat completions endpoints.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{rejection::JsonRejection, Extension, State},
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};

use crate::api::openai::compat::{
    compatibility_profile, strict_guardrail_message, OpenAiCompatibilityProfile,
};
use crate::api::openai::convert::{
    append_assistant_tool_calls, flatten_content, tool_result_message, InboundContent,
};
use crate::api::openai::tool_calls::{
    parse_answer, resolve_tool_policy, ParsedOutput, ToolCall, ToolCallDelta, ToolCallFormat,
    ToolCallParser,
};
use crate::api::request_context::{RequestContext, StreamOutcome};
use crate::app::chat::{
    chat_top_k, parse_chat_model, resolve_chat_request_config, validate_chat_request_ranges,
    ChatBackend, ChatExecutionRequest, ChatStreamEvent, ChatStreamObserver, LocalChat,
};
use crate::app::chat_content::validate_media_inputs_for_variant;
use crate::error::ApiError;
use crate::gateway::{GatewayAdmissionGuard, GatewayChat, GatewayState};
use crate::ids::new_prefixed_id;
use crate::state::AppState;
use izwi_core::{
    chat_supports_json_object, ChatGeneration, ChatMediaInput, ChatMessage, ChatReasoningEffort,
    ChatRole, ChatTemplateKwargs, ChatTruncation, ModelFamily, ModelVariant,
};
use izwi_serving_protocol::MAX_STOP_SEQUENCE_BYTES;

const CHAT_STREAM_INTERRUPTED_ERROR: &str = "Chat stream ended before a terminal event";

/// An in-band error event: the OpenAI error object of the HTTP error the
/// request would have had before the stream started.
fn openai_chat_stream_error_payload(error: &ApiError) -> String {
    error.openai_body().to_string()
}

fn openai_chat_stream_interruption_payload(saw_terminal: bool) -> Option<String> {
    (!saw_terminal).then(|| {
        openai_chat_stream_error_payload(&ApiError::internal(CHAT_STREAM_INTERRUPTED_ERROR))
    })
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAiInboundMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// `-1` (vLLM) or `0` disables top-k sampling.
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Sampling seed (a signed 64-bit integer, as in OpenAI and vLLM).
    #[serde(default)]
    pub seed: Option<i64>,
    /// DS9.3: include per-token logprobs in the response.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// DS9.3: number of top alternatives per token (0-20).
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// DS9.2: structured output request. `json_object` constrains generation
    /// to one valid JSON value on models whose sampler enforces it.
    #[serde(default)]
    pub response_format: Option<OpenAiResponseFormat>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ChatReasoningEffort>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// `"parsed"` moves `<think>` reasoning into `reasoning_content`;
    /// `"none"` (the default) leaves it inside `content`.
    #[serde(default)]
    pub reasoning_format: Option<String>,
    /// Izwi extension mirroring OpenAI Responses: `"disabled"` (the default)
    /// rejects a request that does not fit the context with
    /// `context_length_exceeded`; `"auto"` drops the oldest turns instead.
    #[serde(default)]
    pub truncation: Option<ChatTruncation>,
}

/// DS9.2: OpenAI `response_format`. Only `json_object` is supported;
/// `json_schema` is rejected by `kind` with a documented error, and any
/// `json_schema` payload key is ignored by deserialization.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionStreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiInboundMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<InboundContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// On a `tool` message: the id of the call it answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_generation_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_timing: Option<izwi_core::engine::LatencyBreakdown>,
}

#[derive(Debug, Serialize)]
struct OpenAiChoice {
    index: usize,
    message: OpenAiAssistantMessage,
    finish_reason: &'static str,
    /// DS9.3: per-token logprobs; absent unless the request asked for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<OpenAiChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
struct OpenAiAssistantMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    /// DS9.1: OpenAI-shape prompt token details. Always present with usage;
    /// `cached_tokens` is 0 when the serving runtime did not measure prefix
    /// reuse.
    prompt_tokens_details: OpenAiPromptTokensDetails,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiPromptTokensDetails {
    cached_tokens: u64,
}

/// DS9.3: OpenAI-shape per-choice logprobs wrapper.
#[derive(Debug, Clone, Serialize)]
struct OpenAiChoiceLogprobs {
    content: Vec<OpenAiTokenLogprob>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiTokenLogprob {
    token: String,
    logprob: f32,
    bytes: Vec<u8>,
    top_logprobs: Vec<OpenAiTopTokenLogprob>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiTopTokenLogprob {
    token: String,
    logprob: f32,
    bytes: Vec<u8>,
}

/// DS9.3: map core logprob entries into the OpenAI response shape. `None`
/// when the request did not collect logprobs.
fn openai_logprobs(entries: &[izwi_core::engine::TokenLogprob]) -> Option<OpenAiChoiceLogprobs> {
    if entries.is_empty() {
        return None;
    }
    Some(OpenAiChoiceLogprobs {
        content: entries
            .iter()
            .map(|entry| OpenAiTokenLogprob {
                token: entry.token.clone(),
                logprob: entry.logprob,
                bytes: entry.bytes.clone(),
                top_logprobs: entry
                    .top_logprobs
                    .iter()
                    .map(|top| OpenAiTopTokenLogprob {
                        token: top.token.clone(),
                        logprob: top.logprob,
                        bytes: top.bytes.clone(),
                    })
                    .collect(),
            })
            .collect(),
    })
}

#[derive(Debug, Serialize)]
struct OpenAiChatChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChunkChoice>,
    /// With `stream_options.include_usage`: `null` on every chunk but the
    /// last, which carries the usage and no choices.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Option<OpenAiUsage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_generation_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_timing: Option<izwi_core::engine::LatencyBreakdown>,
}

#[derive(Debug, Serialize)]
struct OpenAiChunkChoice {
    index: usize,
    delta: OpenAiDelta,
    /// DS9.3: the logprobs of this chunk's tokens; absent unless requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<OpenAiChoiceLogprobs>,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InboundMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn to_core_messages_with_media(
    variant: ModelVariant,
    messages: Vec<OpenAiInboundMessage>,
) -> Result<(Vec<ChatMessage>, Vec<ChatMediaInput>), ApiError> {
    let mut core = Vec::with_capacity(messages.len());
    let mut media_inputs = Vec::new();
    // The ids of the latest assistant turn's tool calls, which the tool
    // results after it answer.
    let mut call_ids: Vec<String> = Vec::new();

    for message in messages {
        let role = parse_role(&message.role)?;
        let flattened = flatten_content(message.content)?;
        let mut content = flattened.runtime_text.clone();

        if role == InboundMessageRole::Assistant {
            let tool_calls = message.tool_calls.as_deref().unwrap_or(&[]);
            append_assistant_tool_calls(variant, &mut content, tool_calls)?;
            call_ids = tool_calls
                .iter()
                .filter_map(|call| call.get("id").and_then(serde_json::Value::as_str))
                .map(str::to_owned)
                .collect();
        }
        if role == InboundMessageRole::Tool {
            // The templates read results in call order and carry no ids, so
            // an id is checked against the calls it claims to answer.
            if let Some(id) = &message.tool_call_id {
                if !call_ids.is_empty() && !call_ids.contains(id) {
                    return Err(ApiError::invalid_param(
                        "messages",
                        format!(
                            "Tool message `tool_call_id` {id:?} does not match a tool call of the preceding assistant message"
                        ),
                    ));
                }
            }
        }

        match role {
            InboundMessageRole::System => {
                if flattened.has_media() {
                    return Err(ApiError::bad_request(
                        "System messages cannot contain images or videos",
                    ));
                }
                if content.trim().is_empty() {
                    return Err(ApiError::bad_request(
                        "System message content cannot be empty",
                    ));
                }
                core.push(ChatMessage {
                    role: ChatRole::System,
                    content,
                });
            }
            InboundMessageRole::User => {
                if content.trim().is_empty() && !flattened.has_media() {
                    return Err(ApiError::bad_request(
                        "User message content cannot be empty",
                    ));
                }
                media_inputs.extend(flattened.media_inputs);
                core.push(ChatMessage {
                    role: ChatRole::User,
                    content,
                });
            }
            InboundMessageRole::Assistant => {
                if content.trim().is_empty()
                    && !flattened.has_media()
                    && message
                        .tool_calls
                        .as_ref()
                        .is_none_or(|tool_calls| tool_calls.is_empty())
                {
                    return Err(ApiError::bad_request(
                        "Assistant message must include content or tool calls",
                    ));
                }
                media_inputs.extend(flattened.media_inputs);
                core.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content,
                });
            }
            InboundMessageRole::Tool => {
                if content.trim().is_empty() && !flattened.has_media() {
                    return Err(ApiError::bad_request(
                        "Tool message content cannot be empty",
                    ));
                }
                media_inputs.extend(flattened.media_inputs);
                core.push(tool_result_message(variant, content));
            }
        }
    }

    Ok((core, media_inputs))
}

fn parse_role(raw: &str) -> Result<InboundMessageRole, ApiError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "system" | "developer" => Ok(InboundMessageRole::System),
        "user" => Ok(InboundMessageRole::User),
        "assistant" => Ok(InboundMessageRole::Assistant),
        "tool" => Ok(InboundMessageRole::Tool),
        other => Err(ApiError::bad_request(format!(
            "Unsupported chat message role: {}",
            other
        ))),
    }
}

fn validate_chat_request_compatibility(
    req: &ChatCompletionRequest,
    profile: OpenAiCompatibilityProfile,
) -> Result<(), ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid_param(
            "n",
            "This server currently supports only `n=1` for chat completions",
        ));
    }

    if let Some(top_logprobs) = req.top_logprobs {
        if top_logprobs > 20 {
            return Err(ApiError::invalid_param(
                "top_logprobs",
                "`top_logprobs` must be between 0 and 20",
            ));
        }
        if !req.logprobs.unwrap_or(false) {
            return Err(ApiError::invalid_param(
                "top_logprobs",
                "`top_logprobs` requires `logprobs` to be true",
            ));
        }
    }

    if req
        .frequency_penalty
        .is_some_and(|value| !(value.is_finite() && (-2.0..=2.0).contains(&value)))
    {
        return Err(ApiError::invalid_param(
            "frequency_penalty",
            "`frequency_penalty` must be between -2 and 2",
        ));
    }

    if let Some(response_format) = &req.response_format {
        match response_format.kind.as_str() {
            "json_object" => {}
            "json_schema" => {
                return Err(ApiError::bad_request(
                    "`response_format` type `json_schema` is not supported; use `json_object`",
                ));
            }
            other => {
                return Err(ApiError::bad_request(format!(
                    "unknown `response_format` type `{other}`; supported: `json_object`"
                )));
            }
        }
    }

    if profile.is_strict()
        && req
            .frequency_penalty
            .is_some_and(|value| value.abs() > f32::EPSILON)
    {
        return Err(ApiError::bad_request(strict_guardrail_message(
            "frequency_penalty",
        )));
    }

    Ok(())
}

pub async fn completions(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    validate_chat_request_compatibility(&req, compatibility_profile())?;
    let (execution_request, format) = prepare_execution_request(&req, &ctx)?;
    complete_chat(
        LocalChat(state),
        req,
        execution_request,
        format,
        ctx.stream_outcome.clone(),
    )
    .await
}

pub async fn gateway_completions(
    State(state): State<GatewayState>,
    Extension(ctx): Extension<RequestContext>,
    Extension(admission): Extension<GatewayAdmissionGuard>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = payload.map_err(|rejection| {
        if rejection.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::payload_too_large("Gateway chat request body exceeds the configured limit")
        } else {
            ApiError::bad_request("Gateway chat request body is not valid JSON")
        }
    })?;
    validate_chat_request_compatibility(&req, compatibility_profile())?;
    let (execution_request, format) = prepare_execution_request(&req, &ctx)?;
    state
        .enforce_chat_rate_quota(&ctx, &execution_request)
        .await?;
    let tenant_work = state.begin_tenant_work(&ctx)?;
    let backend = GatewayChat {
        state: &state,
        context: &ctx,
        admission,
        tenant_work,
    };
    complete_chat(
        backend,
        req,
        execution_request,
        format,
        ctx.stream_outcome.clone(),
    )
    .await
}

/// How a request's answer is rendered: extensions, reasoning split and tool
/// calls.
#[derive(Debug, Clone, Copy)]
struct AnswerFormat {
    compat_profile: OpenAiCompatibilityProfile,
    /// See [`reasoning_split_mode`].
    reasoning: Option<bool>,
    /// The tool-call format to parse; `None` leaves the answer text.
    tool_calls: Option<ToolCallFormat>,
}

/// Runs a validated chat request on `backend` and renders the OpenAI answer:
/// an SSE stream, or one completion object.
async fn complete_chat<B: ChatBackend>(
    backend: B,
    req: ChatCompletionRequest,
    execution_request: ChatExecutionRequest,
    format: AnswerFormat,
    stream_outcome: StreamOutcome,
) -> Result<Response, ApiError> {
    let variant = execution_request.variant;
    if req.stream.unwrap_or(false) {
        let (event_rx, observer) = backend.stream(execution_request).await?;
        return Ok(render_chat_stream(
            req,
            variant.dir_name().to_string(),
            event_rx,
            format,
            AuditedStream {
                observer,
                outcome: stream_outcome,
            },
        )
        .into_response());
    }
    let generation = backend.generate(execution_request).await?;
    Ok(render_completion_response(variant, generation, format))
}

/// The most OpenAI `stop` strings a chat request may carry.
const MAX_CHAT_STOP_SEQUENCES: usize = 4;

/// OpenAI `stop`: one string or an array of at most four non-empty strings.
fn chat_stop_sequences(stop: Option<&serde_json::Value>) -> Result<Vec<String>, ApiError> {
    let invalid = || {
        ApiError::bad_request(format!(
            "`stop` must be a string or an array of at most {MAX_CHAT_STOP_SEQUENCES} non-empty strings of at most {MAX_STOP_SEQUENCE_BYTES} bytes"
        ))
    };
    let sequences = match stop {
        None | Some(serde_json::Value::Null) => return Ok(Vec::new()),
        Some(serde_json::Value::String(stop)) => vec![stop.clone()],
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|item| item.as_str().map(str::to_owned).ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(invalid()),
    };
    if sequences.len() > MAX_CHAT_STOP_SEQUENCES
        || sequences
            .iter()
            .any(|stop| stop.is_empty() || stop.len() > MAX_STOP_SEQUENCE_BYTES)
    {
        return Err(invalid());
    }
    Ok(sequences)
}

/// DS9.2: models whose sampler enforces the JSON grammar. Others reject
/// `json_object` rather than silently ignoring the request.
fn ensure_response_format_supported(
    variant: ModelVariant,
    response_format: &OpenAiResponseFormat,
) -> Result<(), ApiError> {
    if response_format.kind == "json_object" && !chat_supports_json_object(variant) {
        return Err(ApiError::bad_request(format!(
            "constrained decoding (`response_format: json_object`) is not supported for model {variant}; it requires a grammar-aware sampler"
        )));
    }
    Ok(())
}

/// The engine request for `req` and how its answer is rendered.
fn prepare_execution_request(
    req: &ChatCompletionRequest,
    ctx: &RequestContext,
) -> Result<(ChatExecutionRequest, AnswerFormat), ApiError> {
    let variant = parse_chat_model(&req.model)?;
    let stop_sequences = chat_stop_sequences(req.stop.as_ref())?;
    if let Some(response_format) = &req.response_format {
        ensure_response_format_supported(variant, response_format)?;
    }
    let tool_policy = resolve_tool_policy(variant, req.tools.as_deref(), req.tool_choice.as_ref())?;
    let (messages, media_inputs) = to_core_messages_with_media(variant, req.messages.clone())?;
    validate_media_inputs_for_variant(variant, &media_inputs).map_err(ApiError::bad_request)?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Chat request must include at least one message",
        ));
    }
    let chat_config = resolve_chat_request_config(
        req.enable_thinking,
        req.reasoning_effort,
        req.preserve_thinking,
        req.chat_template_kwargs.as_ref(),
        tool_policy.tools,
        media_inputs,
        req.truncation,
    )?;
    let execution = ChatExecutionRequest {
        variant,
        messages,
        max_completion_tokens: req.max_completion_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: chat_top_k(req.top_k)?,
        repetition_penalty: req.repetition_penalty,
        presence_penalty: req.presence_penalty,
        // A bit-preserving cast keeps distinct seeds distinct.
        seed: req.seed.map(|seed| seed as u64),
        stop: stop_sequences,
        logprobs: req.logprobs,
        top_logprobs: req.top_logprobs,
        response_format_json_object: req
            .response_format
            .as_ref()
            .is_some_and(|format| format.kind == "json_object"),
        chat_config,
        correlation_id: Some(ctx.correlation_id.clone()),
    };
    validate_chat_request_ranges(&execution)?;
    let format = AnswerFormat {
        compat_profile: compatibility_profile(),
        reasoning: reasoning_split_mode(req, &execution)?,
        tool_calls: tool_policy.format,
    };
    Ok((execution, format))
}

/// `Some(opened_in_prompt)` when the request asked for `reasoning_format:
/// "parsed"`. `opened_in_prompt` is true for chat families whose prompt
/// opens the `<think>` block when thinking is on (Qwen3.5/3.6/3.8); the
/// LFM2.5 Thinking model writes its own opener.
fn reasoning_split_mode(
    req: &ChatCompletionRequest,
    execution: &ChatExecutionRequest,
) -> Result<Option<bool>, ApiError> {
    match req.reasoning_format.as_deref().map(str::trim) {
        None | Some("none") => Ok(None),
        Some("parsed") => {
            let variant = execution.variant;
            let opened_in_prompt = variant.family() != ModelFamily::Lfm2Chat
                && variant.chat_capabilities().is_some_and(|caps| {
                    caps.supports_thinking
                        && execution
                            .chat_config
                            .enable_thinking
                            .unwrap_or(caps.default_thinking_enabled)
                });
            Ok(Some(opened_in_prompt))
        }
        Some(other) => Err(ApiError::bad_request(format!(
            "Unsupported `reasoning_format` {other:?}; use \"none\" or \"parsed\""
        ))),
    }
}

fn render_completion_response(
    variant: ModelVariant,
    generation: ChatGeneration,
    format: AnswerFormat,
) -> Response {
    let completion_tokens = generation.tokens_generated;
    let prompt_tokens = generation.prompt_tokens;
    let (reasoning_content, text) = match format.reasoning {
        Some(opened_in_prompt) => {
            super::reasoning::split_reasoning(&generation.text, opened_in_prompt)
        }
        None => (None, generation.text),
    };
    let (content, tool_calls) = match format.tool_calls {
        Some(tool_format) => parse_answer(tool_format, &text),
        None => (Some(text), Vec::new()),
    };
    let compat_profile = format.compat_profile;

    let response = OpenAiChatCompletionResponse {
        id: new_prefixed_id("chatcmpl-"),
        object: "chat.completion",
        created: now_unix_secs(),
        model: variant.dir_name().to_string(),
        choices: vec![OpenAiChoice {
            index: 0,
            finish_reason: finish_reason(!tool_calls.is_empty(), generation.finish_reason),
            message: OpenAiAssistantMessage {
                role: "assistant",
                reasoning_content,
                content,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            },
            logprobs: openai_logprobs(&generation.logprobs),
        }],
        usage: OpenAiUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: OpenAiPromptTokensDetails {
                cached_tokens: generation.cached_prompt_tokens.unwrap_or(0),
            },
        },
        izwi_generation_time_ms: compat_profile
            .is_relaxed()
            .then_some(generation.generation_time_ms),
        izwi_timing: compat_profile
            .is_relaxed()
            .then_some(generation.latency_breakdown)
            .flatten(),
    };

    Json(response).into_response()
}

/// Tells the request audit how a stream ended in-band.
struct AuditedStream<O> {
    observer: O,
    outcome: StreamOutcome,
}

impl<O: ChatStreamObserver> ChatStreamObserver for AuditedStream<O> {
    fn record_delta(&mut self) {
        self.observer.record_delta();
    }

    fn record_completion(&mut self, generation: &ChatGeneration) {
        self.outcome.record_success();
        self.observer.record_completion(generation);
    }

    fn record_failure(&mut self) {
        self.outcome.record_failure();
        self.observer.record_failure();
    }
}

/// Streamed answer text on its way to the client: reasoning is split off
/// first, then tool-call markup becomes `tool_calls`.
struct AnswerStream {
    splitter: Option<super::reasoning::ReasoningSplitter>,
    tool_calls: Option<ToolCallParser>,
}

impl AnswerStream {
    fn new(format: AnswerFormat) -> Self {
        Self {
            splitter: format
                .reasoning
                .map(super::reasoning::ReasoningSplitter::new),
            tool_calls: format.tool_calls.map(ToolCallParser::new),
        }
    }

    /// The delta for one piece of model text.
    fn push(&mut self, text: &str) -> OpenAiDelta {
        let (reasoning, content) = match self.splitter.as_mut() {
            Some(splitter) => splitter.push(text),
            None => (String::new(), text.to_string()),
        };
        let parsed = match self.tool_calls.as_mut() {
            Some(parser) => parser.push(&content),
            None => ParsedOutput {
                content,
                tool_calls: Vec::new(),
            },
        };
        answer_delta(reasoning, parsed)
    }

    /// The delta for the text held back at the end of the answer.
    fn finish(&mut self) -> OpenAiDelta {
        let (reasoning, content) = self
            .splitter
            .as_mut()
            .map(|splitter| splitter.finish())
            .unwrap_or_default();
        let parsed = match self.tool_calls.as_mut() {
            Some(parser) => {
                let mut parsed = parser.push(&content);
                parsed.extend(parser.finish());
                parsed
            }
            None => ParsedOutput {
                content,
                tool_calls: Vec::new(),
            },
        };
        answer_delta(reasoning, parsed)
    }

    fn saw_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(ToolCallParser::saw_tool_calls)
    }
}

fn answer_delta(reasoning: String, parsed: ParsedOutput) -> OpenAiDelta {
    OpenAiDelta {
        role: None,
        reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
        content: (!parsed.content.is_empty()).then_some(parsed.content),
        tool_calls: (!parsed.tool_calls.is_empty()).then_some(parsed.tool_calls),
    }
}

impl OpenAiDelta {
    fn is_empty(&self) -> bool {
        self.role.is_none()
            && self.reasoning_content.is_none()
            && self.content.is_none()
            && self.tool_calls.is_none()
    }
}

fn render_chat_stream(
    req: ChatCompletionRequest,
    model_id: String,
    mut event_rx: tokio::sync::mpsc::Receiver<ChatStreamEvent>,
    format: AnswerFormat,
    observer: impl ChatStreamObserver,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let completion_id = new_prefixed_id("chatcmpl-");
    let created = now_unix_secs();
    let chunk = move |choices: Vec<OpenAiChunkChoice>| OpenAiChatChunk {
        id: completion_id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model_id.clone(),
        choices,
        usage: include_usage.then_some(None),
        izwi_generation_time_ms: None,
        izwi_timing: None,
    };
    let choice = |delta, logprobs, finish_reason| {
        vec![OpenAiChunkChoice {
            index: 0,
            delta,
            logprobs,
            finish_reason,
        }]
    };

    let stream = async_stream::stream! {
        let mut observer = observer;
        let mut answer = AnswerStream::new(format);
        let mut saw_terminal = false;
        while let Some(event) = event_rx.recv().await {
            let (payloads, terminal) = match event {
                ChatStreamEvent::Started => {
                    let delta = OpenAiDelta {
                        role: Some("assistant"),
                        reasoning_content: None,
                        content: None,
                        tool_calls: None,
                    };
                    (vec![chunk_json(&chunk(choice(delta, None, None)))], false)
                }
                ChatStreamEvent::Delta { text, logprobs } => {
                    if !text.is_empty() {
                        observer.record_delta();
                    }
                    let delta = answer.push(&text);
                    let logprobs = openai_logprobs(&logprobs);
                    if delta.is_empty() && logprobs.is_none() {
                        (Vec::new(), false)
                    } else {
                        (vec![chunk_json(&chunk(choice(delta, logprobs, None)))], false)
                    }
                }
                ChatStreamEvent::Completed(generation) => {
                    observer.record_completion(&generation);
                    // Text held back for a tag is final now.
                    let delta = answer.finish();
                    let finish = finish_reason(answer.saw_tool_calls(), generation.finish_reason);
                    let mut last = chunk(choice(delta, None, Some(finish)));
                    if format.compat_profile.is_relaxed() {
                        last.izwi_generation_time_ms = Some(generation.generation_time_ms);
                        last.izwi_timing = generation.latency_breakdown;
                    }
                    let mut payloads = vec![chunk_json(&last)];
                    if include_usage {
                        let mut usage = chunk(Vec::new());
                        usage.usage = Some(Some(OpenAiUsage {
                            prompt_tokens: generation.prompt_tokens,
                            completion_tokens: generation.tokens_generated,
                            total_tokens: generation.prompt_tokens + generation.tokens_generated,
                            prompt_tokens_details: OpenAiPromptTokensDetails {
                                cached_tokens: generation.cached_prompt_tokens.unwrap_or(0),
                            },
                        }));
                        payloads.push(chunk_json(&usage));
                    }
                    (payloads, true)
                }
                ChatStreamEvent::Failed(error) => {
                    observer.record_failure();
                    (vec![openai_chat_stream_error_payload(&error)], true)
                }
            };
            for payload in payloads {
                yield Ok(Event::default().data(payload));
            }
            if terminal {
                saw_terminal = true;
                break;
            }
        }
        if let Some(payload) = openai_chat_stream_interruption_payload(saw_terminal) {
            observer.record_failure();
            yield Ok(Event::default().data(payload));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}

fn chunk_json(chunk: &OpenAiChatChunk) -> String {
    serde_json::to_string(chunk).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::openai::convert::InboundContentPart;
    use axum::http::StatusCode;
    use serde_json::json;

    #[test]
    fn closed_chat_stream_emits_explicit_error_without_terminal_event() {
        let payload = openai_chat_stream_interruption_payload(false)
            .expect("an interrupted stream should emit an error");
        let json: serde_json::Value = serde_json::from_str(&payload).expect("valid error payload");

        assert_eq!(
            json.pointer("/error/message")
                .and_then(|value| value.as_str()),
            Some(CHAT_STREAM_INTERRUPTED_ERROR)
        );
        assert_eq!(
            json.pointer("/error/type").and_then(|value| value.as_str()),
            Some("server_error")
        );
        assert!(openai_chat_stream_interruption_payload(true).is_none());
    }

    pub(super) fn generation(text: &str) -> ChatGeneration {
        ChatGeneration {
            latency_breakdown: None,
            finish_reason: None,
            text: text.to_string(),
            prompt_tokens: 3,
            tokens_generated: 5,
            generation_time_ms: 0.0,
            cached_prompt_tokens: None,
            logprobs: Vec::new(),
        }
    }

    /// The `data:` payloads of the SSE stream that `events` render to for
    /// the request `body`.
    pub(super) async fn sse_payloads(
        body: serde_json::Value,
        format: AnswerFormat,
        events: Vec<ChatStreamEvent>,
    ) -> Vec<String> {
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let (tx, rx) = tokio::sync::mpsc::channel(events.len().max(1));
        for event in events {
            tx.try_send(event).unwrap();
        }
        drop(tx);
        let response = render_chat_stream(req, "fixture".into(), rx, format, ()).into_response();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(str::to_owned)
            .collect()
    }

    /// The text, assembled tool calls and finish reason a client reads from
    /// a chat stream.
    fn read_stream(payloads: &[String]) -> (String, Vec<serde_json::Value>, serde_json::Value) {
        let mut content = String::new();
        let mut calls: Vec<serde_json::Value> = Vec::new();
        let mut finish = serde_json::Value::Null;
        for payload in payloads.iter().filter(|payload| *payload != "[DONE]") {
            let chunk: serde_json::Value = serde_json::from_str(payload).unwrap();
            let Some(choice) = chunk["choices"].get(0) else {
                continue;
            };
            if let Some(text) = choice["delta"]["content"].as_str() {
                content.push_str(text);
            }
            for delta in choice["delta"]["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
            {
                let index = delta["index"].as_u64().unwrap() as usize;
                let arguments = delta["function"]["arguments"].as_str().unwrap_or_default();
                if index == calls.len() {
                    calls.push(json!({
                        "id": delta["id"],
                        "type": delta["type"],
                        "name": delta["function"]["name"],
                        "arguments": arguments,
                    }));
                } else {
                    let call = &mut calls[index]["arguments"];
                    *call = json!(format!("{}{arguments}", call.as_str().unwrap()));
                }
            }
            if !choice["finish_reason"].is_null() {
                finish = choice["finish_reason"].clone();
            }
        }
        (content, calls, finish)
    }

    pub(super) fn deltas(text: &str, size: usize) -> Vec<ChatStreamEvent> {
        let chars: Vec<char> = text.chars().collect();
        std::iter::once(ChatStreamEvent::Started)
            .chain(chars.chunks(size).map(|piece| ChatStreamEvent::Delta {
                text: piece.iter().collect(),
                logprobs: Vec::new(),
            }))
            .chain(std::iter::once(ChatStreamEvent::Completed(Box::new(
                generation(text),
            ))))
            .collect()
    }

    pub(super) fn tool_format(tool_calls: Option<ToolCallFormat>) -> AnswerFormat {
        AnswerFormat {
            compat_profile: OpenAiCompatibilityProfile::Strict,
            reasoning: None,
            tool_calls,
        }
    }

    /// EDGE-F5: streamed tool calls arrive as `delta.tool_calls`, with no
    /// markup in `delta.content`, however the deltas split the tags.
    #[tokio::test]
    async fn streamed_tool_calls_never_leak_markup_into_content() {
        let cases = [
            (
                ToolCallFormat::QwenXml,
                "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                ToolCallFormat::LfmPythonic,
                "Let me check.<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>",
            ),
        ];
        let body = json!({"model": "m", "messages": [], "stream": true});
        for (format, text) in cases {
            for size in [1, 2, 3, 5, 8, 13, text.len()] {
                let payloads =
                    sse_payloads(body.clone(), tool_format(Some(format)), deltas(text, size)).await;
                let (content, calls, finish) = read_stream(&payloads);
                assert_eq!(content, "Let me check.", "{format:?} split {size}");
                assert_eq!(calls.len(), 1, "{format:?} split {size}");
                assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
                assert_eq!(calls[0]["type"], "function");
                assert_eq!(calls[0]["name"], "get_weather");
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(
                        calls[0]["arguments"].as_str().unwrap()
                    )
                    .unwrap(),
                    json!({"city": "Paris"})
                );
                assert_eq!(finish, "tool_calls");
            }
        }
    }

    /// EDGE-F18: stream logprobs sit on the choice, usage comes in its own
    /// final chunk with no choices (`null` before it), and ids are
    /// `chatcmpl-` ids shared by every chunk.
    #[tokio::test]
    async fn stream_chunks_have_the_openai_wire_shape() {
        let entry = izwi_core::engine::TokenLogprob {
            token: "Hi".to_string(),
            logprob: -0.5,
            bytes: b"Hi".to_vec(),
            top_logprobs: Vec::new(),
        };
        let events = vec![
            ChatStreamEvent::Started,
            ChatStreamEvent::Delta {
                text: "Hi".into(),
                logprobs: vec![entry],
            },
            ChatStreamEvent::Completed(Box::new(generation("Hi"))),
        ];
        let body = json!({
            "model": "m", "messages": [], "stream": true, "logprobs": true,
            "stream_options": {"include_usage": true}
        });
        let payloads = sse_payloads(body, tool_format(None), events).await;
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<serde_json::Value> = payloads[..payloads.len() - 1]
            .iter()
            .map(|payload| serde_json::from_str(payload).unwrap())
            .collect();
        assert_eq!(chunks.len(), 4, "role, text, finish, usage: {chunks:?}");
        let id = chunks[0]["id"].as_str().unwrap();
        assert!(id.starts_with("chatcmpl-"), "{id}");
        assert!(chunks.iter().all(|chunk| chunk["id"] == id));

        let text = &chunks[1]["choices"][0];
        assert_eq!(text["delta"]["content"], "Hi");
        assert!(text["delta"].get("logprobs").is_none());
        assert_eq!(text["logprobs"]["content"][0]["token"], "Hi");

        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
        for chunk in &chunks[..3] {
            assert!(chunk["usage"].is_null() && chunk.get("usage").is_some());
        }
        assert_eq!(chunks[3]["choices"], json!([]));
        assert_eq!(chunks[3]["usage"]["prompt_tokens"], 3);
        assert_eq!(chunks[3]["usage"]["completion_tokens"], 5);
        assert_eq!(chunks[3]["usage"]["total_tokens"], 8);

        // Without `include_usage` there is no usage field at all.
        let body = json!({"model": "m", "messages": [], "stream": true});
        let payloads = sse_payloads(body, tool_format(None), deltas("Hi", 1)).await;
        assert!(payloads
            .iter()
            .filter(|payload| *payload != "[DONE]")
            .all(|payload| !payload.contains("\"usage\"")));
    }

    /// Without offered tools the answer is text, markup and all.
    #[tokio::test]
    async fn tool_markup_is_text_when_no_tools_were_offered() {
        let text = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        let body = json!({"model": "m", "messages": [], "stream": true});
        let payloads = sse_payloads(body, tool_format(None), deltas(text, 4)).await;
        let (content, calls, finish) = read_stream(&payloads);
        assert_eq!((content.as_str(), calls.len()), (text, 0));
        assert_eq!(finish, "stop");

        let response = render_completion_response(
            ModelVariant::Qwen359BGguf,
            generation(text),
            tool_format(None),
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], text);
        assert!(body["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn tool_results_must_answer_a_call_of_the_preceding_turn() {
        let message = |role: &str, tool_calls, tool_call_id: Option<&str>| OpenAiInboundMessage {
            role: role.to_string(),
            content: Some(InboundContent::Text("x".to_string())),
            tool_calls,
            tool_call_id: tool_call_id.map(str::to_owned),
        };
        let call = Some(vec![json!({
            "id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}
        })]);
        let conversation = |id| {
            vec![
                message("user", None, None),
                message("assistant", call.clone(), None),
                message("tool", None, Some(id)),
            ]
        };
        to_core_messages_with_media(ModelVariant::Qwen359BGguf, conversation("call_a"))
            .expect("the result answers the call");
        let error = to_core_messages_with_media(ModelVariant::Qwen359BGguf, conversation("call_b"))
            .expect_err("an unknown call id is refused");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.message.contains("call_b"), "{}", error.message);
    }

    #[test]
    fn lfm25_tool_turns_use_the_lfm_wire_format() {
        let text = |role: &str, content: &str| OpenAiInboundMessage {
            role: role.to_string(),
            content: Some(InboundContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
        };
        let messages = vec![
            text("user", "Weather in Paris?"),
            OpenAiInboundMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                })]),
                tool_call_id: None,
            },
            text("tool", "{\"temperature_c\": 21}"),
        ];
        let (lfm, _) =
            to_core_messages_with_media(ModelVariant::Lfm2512BInstructGguf, messages.clone())
                .unwrap();
        assert_eq!(
            lfm[1].content,
            "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"
        );
        assert_eq!(lfm[2].role, ChatRole::Tool);
        assert_eq!(lfm[2].content, "{\"temperature_c\": 21}");

        // Other families keep the Qwen tool format.
        let (qwen, _) = to_core_messages_with_media(ModelVariant::Qwen314BGguf, messages).unwrap();
        assert!(qwen[1].content.contains("<function=get_weather>"));
        assert_eq!(qwen[2].role, ChatRole::User);
        assert!(qwen[2].content.starts_with("<tool_response>"));
    }

    #[test]
    fn to_core_messages_collects_multimodal_parts() {
        let (messages, media_inputs) = to_core_messages_with_media(
            ModelVariant::Qwen314BGguf,
            vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(InboundContent::Parts(vec![InboundContentPart {
                    kind: Some("image_url".to_string()),
                    text: None,
                    input_text: None,
                    image_url: Some(json!({"url":"https://example.com/cat.png"})),
                    input_image: None,
                    image: None,
                    video: None,
                    video_url: None,
                    input_video: None,
                }])),
                tool_calls: None,
                tool_call_id: None,
            }],
        )
        .expect("multimodal parts should normalize");

        assert_eq!(messages.len(), 1);
        assert!(messages[0].content.contains("<|image_pad|>"));
        assert_eq!(media_inputs.len(), 1);
        assert!(
            validate_media_inputs_for_variant(ModelVariant::Qwen314BGguf, &media_inputs)
                .expect_err("non-qwen35 multimodal should fail")
                .contains("currently supported only for Qwen3.5")
        );
    }

    #[test]
    fn validates_rejects_non_zero_frequency_penalty() {
        let req = ChatCompletionRequest {
            model: "Qwen3.5-4B".to_string(),
            messages: vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(InboundContent::Text("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: Some(1),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            frequency_penalty: Some(0.5),
            presence_penalty: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            reasoning_effort: None,
            preserve_thinking: None,
            chat_template_kwargs: None,
            reasoning_format: None,
            truncation: None,
        };

        let err = validate_chat_request_compatibility(&req, OpenAiCompatibilityProfile::Strict)
            .expect_err("should reject");
        assert!(err.message.contains("frequency_penalty"));
    }

    /// EXE-F9 / EDGE-F4: `stop` is implemented, so strict mode accepts it.
    #[test]
    fn strict_profile_accepts_stop_sequences() {
        let req = ChatCompletionRequest {
            model: "Qwen3.5-4B".to_string(),
            messages: vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(InboundContent::Text("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: Some(1),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            stop: Some(json!(["END"])),
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            reasoning_effort: None,
            preserve_thinking: None,
            chat_template_kwargs: None,
            reasoning_format: None,
            truncation: None,
        };

        validate_chat_request_compatibility(&req, OpenAiCompatibilityProfile::Strict)
            .expect("strict mode accepts stop");
    }

    #[test]
    fn stop_accepts_a_string_or_up_to_four_strings() {
        let parse = |value: serde_json::Value| chat_stop_sequences(Some(&value));
        assert_eq!(parse(json!(null)).unwrap(), Vec::<String>::new());
        assert_eq!(chat_stop_sequences(None).unwrap(), Vec::<String>::new());
        assert_eq!(parse(json!("END")).unwrap(), ["END"]);
        assert_eq!(parse(json!([])).unwrap(), Vec::<String>::new());
        assert_eq!(
            parse(json!(["a", "b", "c", "d"])).unwrap(),
            ["a", "b", "c", "d"]
        );
        for invalid in [
            json!(["a", "b", "c", "d", "e"]),
            json!(""),
            json!(["ok", ""]),
            json!([1]),
            json!(7),
            json!({"stop": "x"}),
            json!("x".repeat(MAX_STOP_SEQUENCE_BYTES + 1)),
        ] {
            let error = parse(invalid.clone()).expect_err("invalid stop");
            assert_eq!(error.status, StatusCode::BAD_REQUEST, "{invalid}");
            assert!(error.message.contains("`stop`"), "{}", error.message);
        }
    }

    #[test]
    fn relaxed_profile_allows_stop_and_frequency_penalty_passthrough() {
        let req = ChatCompletionRequest {
            model: "Qwen3.5-4B".to_string(),
            messages: vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(InboundContent::Text("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: Some(1),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            frequency_penalty: Some(1.0),
            presence_penalty: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            stop: Some(json!(["END"])),
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            reasoning_effort: None,
            preserve_thinking: None,
            chat_template_kwargs: None,
            reasoning_format: None,
            truncation: None,
        };

        validate_chat_request_compatibility(&req, OpenAiCompatibilityProfile::Relaxed)
            .expect("relaxed mode should allow passthrough");
    }
}

/// Tool calls keep their OpenAI termination contract; other answers report
/// the engine's budget termination rather than every run as `stop`.
fn finish_reason(
    tool_calls: bool,
    reason: Option<izwi_core::engine::OutputFinishReason>,
) -> &'static str {
    if tool_calls {
        "tool_calls"
    } else if reason == Some(izwi_core::engine::OutputFinishReason::MaxTokens) {
        "length"
    } else {
        "stop"
    }
}

#[cfg(test)]
mod timing_contract_tests {
    use super::*;

    fn reasoning_mode_for(body: serde_json::Value) -> Result<Option<bool>, ApiError> {
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let ctx = RequestContext::new(
            "reasoning".into(),
            izwi_hooks::Principal::local_anonymous(),
        );
        prepare_execution_request(&req, &ctx).map(|(_, format)| format.reasoning)
    }

    #[test]
    fn reasoning_format_is_opt_in_and_knows_who_opens_the_block() {
        let user = serde_json::json!([{"role": "user", "content": "hi"}]);
        // Default: reasoning stays in `content`.
        assert_eq!(
            reasoning_mode_for(serde_json::json!({"model": "LFM2.5-1.2B-Thinking-GGUF", "messages": user}))
                .unwrap(),
            None
        );
        // LFM2.5 Thinking writes its own `<think>`.
        assert_eq!(
            reasoning_mode_for(serde_json::json!({
                "model": "LFM2.5-1.2B-Thinking-GGUF", "messages": user, "reasoning_format": "parsed"
            }))
            .unwrap(),
            Some(false)
        );
        // Qwen3.8 opens the block in its prompt unless thinking is off.
        assert_eq!(
            reasoning_mode_for(serde_json::json!({
                "model": "Qwen3.8-27B-FP8", "messages": user, "reasoning_format": "parsed"
            }))
            .unwrap(),
            Some(true)
        );
        assert_eq!(
            reasoning_mode_for(serde_json::json!({
                "model": "Qwen3.8-27B-FP8", "messages": user, "reasoning_format": "parsed",
                "enable_thinking": false
            }))
            .unwrap(),
            Some(false)
        );
        assert!(reasoning_mode_for(serde_json::json!({
            "model": "LFM2.5-1.2B-Thinking-GGUF", "messages": user, "reasoning_format": "deepseek"
        }))
        .is_err());
    }

    #[tokio::test]
    async fn parsed_reasoning_fills_reasoning_content_and_keeps_tool_calls() {
        let generation = |text: &str| ChatGeneration {
            latency_breakdown: None,
            finish_reason: None,
            text: text.to_string(),
            prompt_tokens: 1,
            tokens_generated: 1,
            generation_time_ms: 0.0,
            cached_prompt_tokens: None,
            logprobs: Vec::new(),
        };
        let body = |response: Response| async move {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        };
        let format = |reasoning, tool_calls| AnswerFormat {
            compat_profile: OpenAiCompatibilityProfile::Strict,
            reasoning,
            tool_calls,
        };
        let text = "<think>check the tool</think>\n\n<|tool_call_start|>[now()]<|tool_call_end|>";
        let parsed = body(render_completion_response(
            ModelVariant::Lfm2512BThinkingGguf,
            generation(text),
            format(Some(false), Some(ToolCallFormat::LfmPythonic)),
        ))
        .await;
        let message = &parsed["choices"][0]["message"];
        assert_eq!(message["reasoning_content"], "check the tool");
        assert!(message.get("content").is_none());
        assert_eq!(message["tool_calls"][0]["function"]["name"], "now");
        assert_eq!(parsed["choices"][0]["finish_reason"], "tool_calls");

        let raw = body(render_completion_response(
            ModelVariant::Lfm2512BThinkingGguf,
            generation("<think>r</think>A"),
            format(None, None),
        ))
        .await;
        assert!(raw["choices"][0]["message"].get("reasoning_content").is_none());
        assert_eq!(raw["choices"][0]["message"]["content"], "<think>r</think>A");
    }

    #[test]
    fn budget_termination_is_length_without_overriding_tool_calls() {
        use izwi_core::engine::OutputFinishReason::{MaxTokens, StopToken};
        assert_eq!(finish_reason(false, Some(MaxTokens)), "length");
        assert_eq!(finish_reason(false, Some(StopToken)), "stop");
        assert_eq!(finish_reason(true, Some(MaxTokens)), "tool_calls");
    }

    #[test]
    fn json_and_sse_preserve_independent_timing_and_omit_absent_extensions() {
        let timing: izwi_core::engine::LatencyBreakdown =
            serde_json::from_value(serde_json::json!({
                "queue_wait_ms": 10.0, "prefill_ms": 20.0, "decode_ms": 30.0,
                "decode_wall_ms": 50.0, "decode_tokens": 3, "post_first_token_ms": 50.0,
                "ttft_ms": 30.0, "total_ms": 90.0, "prefill_steps": 1, "decode_steps": 2
            }))
            .unwrap();
        let json = OpenAiChatCompletionResponse {
            id: "timing".into(),
            object: "chat.completion",
            created: 0,
            model: "fixture".into(),
            choices: vec![],
            usage: OpenAiUsage {
                prompt_tokens: 5,
                completion_tokens: 4,
                total_tokens: 9,
                prompt_tokens_details: OpenAiPromptTokensDetails { cached_tokens: 0 },
            },
            izwi_generation_time_ms: Some(90.0),
            izwi_timing: Some(timing.clone()),
        };
        let mut sse = OpenAiChatChunk {
            id: "timing".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "fixture".into(),
            choices: vec![],
            usage: None,
            izwi_generation_time_ms: Some(90.0),
            izwi_timing: Some(timing),
        };
        let json = serde_json::to_value(json).unwrap();
        let chunk = serde_json::to_value(&sse).unwrap();
        assert_eq!(json["izwi_timing"], chunk["izwi_timing"]);
        assert_eq!(json["izwi_generation_time_ms"], 90.0);
        assert_eq!(chunk["izwi_timing"]["decode_wall_ms"], 50.0);
        assert_eq!(chunk["izwi_timing"]["decode_tokens"], 3);
        sse.izwi_generation_time_ms = None;
        sse.izwi_timing = None;
        let strict = serde_json::to_value(sse).unwrap();
        assert!(strict.get("izwi_timing").is_none());
        assert!(strict.get("izwi_generation_time_ms").is_none());
    }

    #[test]
    fn logprob_requests_validate_openai_bounds() {
        let req = |logprobs: Option<bool>, top_logprobs: Option<u8>| ChatCompletionRequest {
            model: "m".into(),
            messages: vec![],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            logprobs,
            top_logprobs,
            response_format: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            reasoning_effort: None,
            preserve_thinking: None,
            chat_template_kwargs: None,
            reasoning_format: None,
            truncation: None,
        };
        assert!(validate_chat_request_compatibility(&req(Some(true), Some(20)), profile()).is_ok());
        assert!(validate_chat_request_compatibility(&req(Some(true), None), profile()).is_ok());
        let too_many = validate_chat_request_compatibility(&req(Some(true), Some(21)), profile());
        assert!(too_many.is_err(), "top_logprobs above 20 must be rejected");
        let missing_flag = validate_chat_request_compatibility(&req(None, Some(3)), profile());
        assert!(
            missing_flag.is_err(),
            "top_logprobs without logprobs must be rejected"
        );
    }

    fn profile() -> OpenAiCompatibilityProfile {
        OpenAiCompatibilityProfile::Relaxed
    }

    #[test]
    fn response_format_types_are_validated() {
        let req = |kind: &str| ChatCompletionRequest {
            model: "m".into(),
            messages: vec![],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            response_format: Some(OpenAiResponseFormat { kind: kind.into() }),
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            reasoning_effort: None,
            preserve_thinking: None,
            chat_template_kwargs: None,
            reasoning_format: None,
            truncation: None,
        };
        assert!(validate_chat_request_compatibility(&req("json_object"), profile()).is_ok());
        assert!(
            validate_chat_request_compatibility(&req("json_schema"), profile()).is_err(),
            "json_schema must be rejected with a documented error"
        );
        assert!(
            validate_chat_request_compatibility(&req("yaml"), profile()).is_err(),
            "unknown types must be rejected"
        );
    }

    #[test]
    fn response_format_deserialization_ignores_json_schema_payload_keys() {
        let format: OpenAiResponseFormat = serde_json::from_value(serde_json::json!({
            "type": "json_object",
            "json_schema": { "name": "strict-output", "strict": true }
        }))
        .expect("the json_schema payload key is ignored, not a parse error");
        assert_eq!(format.kind, "json_object");
    }

    #[test]
    fn json_object_is_rejected_on_models_without_a_grammar_aware_sampler() {
        let format = OpenAiResponseFormat {
            kind: "json_object".into(),
        };
        assert!(ensure_response_format_supported(ModelVariant::Qwen314BGguf, &format).is_ok());
        assert!(
            ensure_response_format_supported(ModelVariant::Qwen3827BFp8, &format).is_err(),
            "qwen3.8 has its own sampler without the grammar seam"
        );
        assert!(
            ensure_response_format_supported(ModelVariant::Qwen36Moe35BA3BFp8, &format).is_ok(),
            "qwen3.6-moe wires the DS9.2 grammar into its decode states"
        );
    }

    #[test]
    fn logprob_entries_render_openai_shape() {
        let entries = vec![izwi_core::engine::TokenLogprob {
            token: " hello".to_string(),
            logprob: -0.25,
            bytes: b" hello".to_vec(),
            top_logprobs: vec![
                izwi_core::engine::TopTokenLogprob {
                    token: " hello".to_string(),
                    logprob: -0.25,
                    bytes: b" hello".to_vec(),
                },
                izwi_core::engine::TopTokenLogprob {
                    token: " hi".to_string(),
                    logprob: -1.5,
                    bytes: b" hi".to_vec(),
                },
            ],
        }];
        let rendered = serde_json::to_value(openai_logprobs(&entries).unwrap()).unwrap();
        let first = &rendered["content"][0];
        assert_eq!(first["token"], " hello");
        assert_eq!(first["logprob"], -0.25);
        assert_eq!(
            first["bytes"],
            serde_json::json!([32, 104, 101, 108, 108, 111])
        );
        assert_eq!(first["top_logprobs"].as_array().unwrap().len(), 2);
        assert_eq!(first["top_logprobs"][1]["token"], " hi");
        assert!(openai_logprobs(&[]).is_none());
    }

    #[test]
    fn usage_reports_openai_shaped_cached_tokens() {
        let measured = serde_json::to_value(OpenAiUsage {
            prompt_tokens: 10,
            completion_tokens: 4,
            total_tokens: 14,
            prompt_tokens_details: OpenAiPromptTokensDetails { cached_tokens: 7 },
        })
        .unwrap();
        assert_eq!(measured["prompt_tokens"], 10);
        assert_eq!(measured["prompt_tokens_details"]["cached_tokens"], 7);

        let unmeasured = serde_json::to_value(OpenAiUsage {
            prompt_tokens: 10,
            completion_tokens: 4,
            total_tokens: 14,
            prompt_tokens_details: OpenAiPromptTokensDetails { cached_tokens: 0 },
        })
        .unwrap();
        assert_eq!(
            unmeasured["prompt_tokens_details"]["cached_tokens"], 0,
            "unmeasured cache must render as an explicit zero, OpenAI parity"
        );
    }
}

#[cfg(test)]
#[path = "sse_contract.rs"]
mod sse_contract;
