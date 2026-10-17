//! OpenAI-compatible chat completions endpoints.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Extension, State},
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
use crate::error::{ApiError, ApiJson, ErrorScope};
use crate::gateway::{GatewayAdmissionGuard, GatewayChat, GatewayState, PreAdmittedTenantWork};
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
fn openai_chat_stream_error_payload(scope: &ErrorScope, error: &ApiError) -> String {
    scope.openai_body(error).to_string()
}

fn openai_chat_stream_interruption_payload(
    scope: &ErrorScope,
    saw_terminal: bool,
) -> Option<String> {
    (!saw_terminal).then(|| {
        openai_chat_stream_error_payload(scope, &ApiError::internal(CHAT_STREAM_INTERRUPTED_ERROR))
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
    ApiJson(req): ApiJson<ChatCompletionRequest>,
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
    Extension(tenant_work): Extension<PreAdmittedTenantWork>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<Response, ApiError> {
    let req = state.read_chat_body(&headers, body).await?;
    let dispatcher = state.chat_dispatcher_for(&req.model)?;
    validate_chat_request_compatibility(&req, compatibility_profile())?;
    let (execution_request, format) = prepare_execution_request(&req, &ctx)?;
    let usage = state
        .enforce_chat_rate_quota(&ctx, &execution_request)
        .await?;
    let tenant_work = tenant_work.take()?;
    let backend = GatewayChat {
        state: &state,
        dispatcher,
        context: &ctx,
        admission,
        tenant_work,
        usage,
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

    // The body is polled after the handler returns, outside the request's
    // error scope; capture it here.
    let errors = ErrorScope::current();
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
                    (vec![openai_chat_stream_error_payload(&errors, &error)], true)
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
        if let Some(payload) = openai_chat_stream_interruption_payload(&errors, saw_terminal) {
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
#[path = "completions_tests.rs"]
mod tests;

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
#[path = "completions_timing_contract_tests.rs"]
mod timing_contract_tests;

#[cfg(test)]
#[path = "sse_contract.rs"]
mod sse_contract;
