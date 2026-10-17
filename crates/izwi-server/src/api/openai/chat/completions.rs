//! OpenAI-compatible chat completions endpoints.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Extension, State},
    response::{IntoResponse, Response},
    Json,
};

use super::dto::{
    openai_logprobs, ChatCompletionRequest, InboundMessageRole, OpenAiAssistantMessage,
    OpenAiChatCompletionResponse, OpenAiChoice, OpenAiInboundMessage, OpenAiPromptTokensDetails,
    OpenAiResponseFormat, OpenAiUsage,
};
use super::stream::{render_chat_stream, AuditedStream};
use crate::api::openai::compat::{
    compatibility_profile, strict_guardrail_message, OpenAiCompatibilityProfile,
};
use crate::api::openai::convert::{
    append_assistant_tool_calls, flatten_content, tool_result_message,
};
use crate::api::openai::tool_calls::{parse_answer, resolve_tool_policy, ToolCallFormat};
use crate::api::request_context::{RequestContext, StreamOutcome};
use crate::app::chat::{
    chat_top_k, parse_chat_model, resolve_chat_request_config, validate_chat_request_ranges,
    ChatBackend, ChatExecutionRequest, LocalChat,
};
use crate::app::chat_content::validate_media_inputs_for_variant;
use crate::error::{ApiError, ApiJson};
use crate::gateway::{GatewayAdmissionGuard, GatewayChat, GatewayState, PreAdmittedTenantWork};
use crate::ids::new_prefixed_id;
use crate::state::AppState;
use izwi_core::{
    chat_supports_json_object, ChatGeneration, ChatMediaInput, ChatMessage, ChatRole, ModelFamily,
    ModelVariant,
};
use izwi_serving_protocol::MAX_STOP_SEQUENCE_BYTES;

pub(crate) fn now_unix_secs() -> u64 {
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
pub(crate) struct AnswerFormat {
    pub(crate) compat_profile: OpenAiCompatibilityProfile,
    /// See [`reasoning_split_mode`].
    pub(crate) reasoning: Option<bool>,
    /// The tool-call format to parse; `None` leaves the answer text.
    pub(crate) tool_calls: Option<ToolCallFormat>,
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
        skip_special_tokens: req.skip_special_tokens,
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

#[cfg(test)]
#[path = "completions_tests.rs"]
mod tests;

/// Tool calls keep their OpenAI termination contract; other answers report
/// the engine's budget termination rather than every run as `stop`.
pub(crate) fn finish_reason(
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
