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
use crate::api::request_context::RequestContext;
use crate::app::chat::{
    generate_chat, parse_chat_model, spawn_chat_stream, ChatExecutionRequest, ChatStreamEvent,
};
use crate::app::chat_content::{
    flatten_content_parts, validate_media_inputs_for_variant, FlattenedMultimodalContent,
};
use crate::error::ApiError;
use crate::ids::new_uuid;
use crate::state::AppState;
use izwi_core::{ChatMediaInput, ChatMessage, ChatRequestConfig, ChatRole, ModelVariant};

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
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
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
    pub content: Option<OpenAiInboundContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OpenAiInboundContent {
    Text(String),
    Parts(Vec<OpenAiInboundContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiInboundContentPart {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub input_text: Option<String>,
    #[serde(default)]
    pub image_url: Option<serde_json::Value>,
    #[serde(default)]
    pub input_image: Option<serde_json::Value>,
    #[serde(default)]
    pub image: Option<serde_json::Value>,
    #[serde(default)]
    pub video: Option<serde_json::Value>,
    #[serde(default)]
    pub video_url: Option<serde_json::Value>,
    #[serde(default)]
    pub input_video: Option<serde_json::Value>,
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
}

#[derive(Debug, Serialize)]
struct OpenAiChoice {
    index: usize,
    message: OpenAiAssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiAssistantMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiOutputToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct OpenAiChatChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_generation_time_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct OpenAiChunkChoice {
    index: usize,
    delta: OpenAiDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiDeltaToolCall>>,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function: Option<OpenAiDeltaToolFunction>,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaToolFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiOutputToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiOutputToolFunction,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiOutputToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct ParsedAssistantToolOutput {
    content: String,
    tool_calls: Vec<OpenAiOutputToolCall>,
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
    _variant: ModelVariant,
    messages: Vec<OpenAiInboundMessage>,
    _enable_thinking: Option<bool>,
    _tools: Option<Vec<serde_json::Value>>,
    _tool_choice: Option<serde_json::Value>,
) -> Result<(Vec<ChatMessage>, Vec<ChatMediaInput>), ApiError> {
    let mut core = Vec::with_capacity(messages.len());
    let mut media_inputs = Vec::new();

    for message in messages {
        let role = parse_role(&message.role)?;
        let flattened = flatten_content(message.content)?;
        let mut content = flattened.runtime_text.clone();

        if role == InboundMessageRole::Assistant
            && message
                .tool_calls
                .as_ref()
                .is_some_and(|tool_calls| !tool_calls.is_empty())
        {
            let tool_call_xml =
                render_tool_calls_xml(message.tool_calls.as_deref().unwrap_or(&[]))?;
            if !tool_call_xml.is_empty() {
                if !content.trim().is_empty() {
                    content.push_str("\n\n");
                }
                content.push_str(&tool_call_xml);
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

                let wrapped = format!("<tool_response>\n{}\n</tool_response>", content.trim());
                core.push(ChatMessage {
                    role: ChatRole::User,
                    content: wrapped,
                });
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

fn flatten_content(
    content: Option<OpenAiInboundContent>,
) -> Result<FlattenedMultimodalContent, ApiError> {
    match content {
        None => Ok(FlattenedMultimodalContent::default()),
        Some(OpenAiInboundContent::Text(text)) => Ok(FlattenedMultimodalContent {
            display_text: text.clone(),
            runtime_text: text,
            media_inputs: Vec::new(),
        }),
        Some(OpenAiInboundContent::Parts(parts)) => {
            flatten_content_parts(&content_parts_to_values(&parts)?).map_err(ApiError::bad_request)
        }
    }
}

fn content_parts_to_values(
    parts: &[OpenAiInboundContentPart],
) -> Result<Vec<serde_json::Value>, ApiError> {
    parts
        .iter()
        .map(|part| serde_json::to_value(part).map_err(|err| ApiError::internal(err.to_string())))
        .collect()
}

fn render_tool_calls_xml(tool_calls: &[serde_json::Value]) -> Result<String, ApiError> {
    let mut rendered = Vec::with_capacity(tool_calls.len());

    for tool_call in tool_calls {
        let function_obj = tool_call.get("function").unwrap_or(tool_call);
        let name = function_obj
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                ApiError::bad_request("Each assistant tool call must include `function.name`")
            })?;

        let args = normalize_tool_call_arguments(function_obj.get("arguments"));
        let mut block = String::new();
        block.push_str("<tool_call>\n");
        block.push_str(&format!("<function={name}>\n"));

        let mut keys: Vec<String> = args.keys().cloned().collect();
        keys.sort();
        for key in keys {
            let value = args.get(&key).cloned().unwrap_or(serde_json::Value::Null);
            block.push_str(&format!("<parameter={key}>\n"));
            block.push_str(&tool_parameter_text(&value));
            block.push_str("\n</parameter>\n");
        }

        block.push_str("</function>\n</tool_call>");
        rendered.push(block);
    }

    Ok(rendered.join("\n"))
}

fn normalize_tool_call_arguments(
    raw: Option<&serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    match raw {
        None | Some(serde_json::Value::Null) => serde_json::Map::new(),
        Some(serde_json::Value::Object(map)) => map.clone(),
        Some(serde_json::Value::String(raw_string)) => {
            match serde_json::from_str::<serde_json::Value>(raw_string) {
                Ok(serde_json::Value::Object(map)) => map,
                Ok(value) => {
                    let mut map = serde_json::Map::new();
                    map.insert("arguments".to_string(), value);
                    map
                }
                Err(_) => {
                    let mut map = serde_json::Map::new();
                    map.insert(
                        "arguments".to_string(),
                        serde_json::Value::String(raw_string.clone()),
                    );
                    map
                }
            }
        }
        Some(value) => {
            let mut map = serde_json::Map::new();
            map.insert("arguments".to_string(), value.clone());
            map
        }
    }
}

fn tool_parameter_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
        }
        _ => value.to_string(),
    }
}

fn parse_assistant_tool_output(raw_output: &str) -> ParsedAssistantToolOutput {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut cursor = 0usize;

    while let Some(start_rel) = raw_output[cursor..].find("<tool_call>") {
        let start = cursor + start_rel;
        text.push_str(&raw_output[cursor..start]);

        let block_start = start + "<tool_call>".len();
        let Some(end_rel) = raw_output[block_start..].find("</tool_call>") else {
            text.push_str(&raw_output[start..]);
            cursor = raw_output.len();
            break;
        };
        let end = block_start + end_rel;
        let block = &raw_output[block_start..end];
        let block_with_tags = &raw_output[start..(end + "</tool_call>".len())];

        if let Some(call) = parse_tool_call_block(block) {
            tool_calls.push(call);
        } else {
            text.push_str(block_with_tags);
        }

        cursor = end + "</tool_call>".len();
    }

    if cursor < raw_output.len() {
        text.push_str(&raw_output[cursor..]);
    }

    ParsedAssistantToolOutput {
        content: text.trim().to_string(),
        tool_calls,
    }
}

fn parse_tool_call_block(block: &str) -> Option<OpenAiOutputToolCall> {
    let fn_start = block.find("<function=")?;
    let name_start = fn_start + "<function=".len();
    let name_end_rel = block[name_start..].find('>')?;
    let name_end = name_start + name_end_rel;
    let function_name = block[name_start..name_end].trim();
    if function_name.is_empty() {
        return None;
    }

    let fn_body_start = name_end + 1;
    let fn_close_rel = block[fn_body_start..].find("</function>")?;
    let fn_body_end = fn_body_start + fn_close_rel;
    let fn_body = &block[fn_body_start..fn_body_end];

    let mut args = serde_json::Map::new();
    let mut cursor = 0usize;
    while let Some(param_rel) = fn_body[cursor..].find("<parameter=") {
        let param_start = cursor + param_rel + "<parameter=".len();
        let key_end_rel = fn_body[param_start..].find('>')?;
        let key_end = param_start + key_end_rel;
        let key = fn_body[param_start..key_end].trim();
        if key.is_empty() {
            return None;
        }

        let value_start = key_end + 1;
        let value_end_rel = fn_body[value_start..].find("</parameter>")?;
        let value_end = value_start + value_end_rel;
        let raw_value = fn_body[value_start..value_end].trim();
        let value = serde_json::from_str::<serde_json::Value>(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
        args.insert(key.to_string(), value);
        cursor = value_end + "</parameter>".len();
    }

    Some(OpenAiOutputToolCall {
        id: new_uuid(),
        kind: "function",
        function: OpenAiOutputToolFunction {
            name: function_name.to_string(),
            arguments: serde_json::to_string(&serde_json::Value::Object(args)).ok()?,
        },
    })
}

fn build_assistant_response_parts(
    text: String,
) -> (
    Option<String>,
    Option<Vec<OpenAiOutputToolCall>>,
    &'static str,
) {
    let parsed = parse_assistant_tool_output(&text);
    if parsed.tool_calls.is_empty() {
        (Some(text), None, "stop")
    } else {
        let content = if parsed.content.is_empty() {
            None
        } else {
            Some(parsed.content)
        };
        (content, Some(parsed.tool_calls), "tool_calls")
    }
}

fn stream_tool_call_deltas(tool_calls: &[OpenAiOutputToolCall]) -> Vec<OpenAiDeltaToolCall> {
    tool_calls
        .iter()
        .enumerate()
        .map(|(idx, call)| OpenAiDeltaToolCall {
            index: idx,
            id: Some(call.id.clone()),
            kind: Some(call.kind),
            function: Some(OpenAiDeltaToolFunction {
                name: Some(call.function.name.clone()),
                arguments: Some(call.function.arguments.clone()),
            }),
        })
        .collect()
}

fn validate_chat_request_compatibility(
    req: &ChatCompletionRequest,
    profile: OpenAiCompatibilityProfile,
) -> Result<(), ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request(
            "This server currently supports only `n=1` for chat completions",
        ));
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

    if profile.is_strict() && req.stop.is_some() {
        return Err(ApiError::bad_request(strict_guardrail_message("stop")));
    }

    if profile.is_strict() {
        if let Some(tool_choice) = &req.tool_choice {
            if !tool_choice.is_null()
                && tool_choice.as_str() != Some("auto")
                && tool_choice.as_str() != Some("none")
            {
                return Err(ApiError::bad_request(strict_guardrail_message(
                    "tool_choice (supported values: \"auto\", \"none\", or null)",
                )));
            }
        }
    }

    Ok(())
}

pub async fn completions(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let compat_profile = compatibility_profile();
    validate_chat_request_compatibility(&req, compat_profile)?;

    let variant = parse_chat_model(&req.model)?;
    let (messages, media_inputs) = to_core_messages_with_media(
        variant,
        req.messages.clone(),
        req.enable_thinking,
        req.tools.clone(),
        req.tool_choice.clone(),
    )?;
    validate_media_inputs_for_variant(variant, &media_inputs).map_err(ApiError::bad_request)?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Chat request must include at least one message",
        ));
    }
    let execution_request = ChatExecutionRequest {
        variant,
        messages,
        max_completion_tokens: req.max_completion_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        presence_penalty: req.presence_penalty,
        chat_config: ChatRequestConfig {
            enable_thinking: req.enable_thinking,
            tools: req.tools.clone().unwrap_or_default(),
            media_inputs,
        },
        correlation_id: Some(ctx.correlation_id),
    };

    if req.stream.unwrap_or(false) {
        let stream_response = complete_stream(state, req, execution_request, compat_profile).await?;
        return Ok(stream_response.into_response());
    }

    let generation = generate_chat(&state, execution_request).await?;

    let completion_id = new_uuid();
    let created = now_unix_secs();
    let completion_tokens = generation.tokens_generated;
    let prompt_tokens = generation.prompt_tokens;
    let (assistant_content, assistant_tool_calls, finish_reason) =
        build_assistant_response_parts(generation.text);

    let response = OpenAiChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created,
        model: variant.dir_name().to_string(),
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiAssistantMessage {
                role: "assistant",
                content: assistant_content,
                tool_calls: assistant_tool_calls,
            },
            finish_reason,
        }],
        usage: OpenAiUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        izwi_generation_time_ms: compat_profile
            .is_relaxed()
            .then_some(generation.generation_time_ms),
    };

    Ok(Json(response).into_response())
}

async fn complete_stream(
    state: AppState,
    req: ChatCompletionRequest,
    execution_request: ChatExecutionRequest,
    compat_profile: OpenAiCompatibilityProfile,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let model_id = execution_request.variant.dir_name().to_string();

    let completion_id = new_uuid();
    let created = now_unix_secs();
    let mut event_rx = spawn_chat_stream(state, execution_request);

    let stream = async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            let (payload, terminal) = match event {
                ChatStreamEvent::Started => (
                    serde_json::to_string(&OpenAiChatChunk {
                        id: completion_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiDelta {
                                role: Some("assistant"),
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                        izwi_generation_time_ms: None,
                    })
                    .unwrap_or_default(),
                    false,
                ),
                ChatStreamEvent::Delta(delta) => (
                    serde_json::to_string(&OpenAiChatChunk {
                        id: completion_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiDelta {
                                role: None,
                                content: Some(delta),
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                        izwi_generation_time_ms: None,
                    })
                    .unwrap_or_default(),
                    false,
                ),
                ChatStreamEvent::Completed(generation) => {
                    let (_, tool_calls, finish_reason) =
                        build_assistant_response_parts(generation.text);
                    let delta_tool_calls =
                        tool_calls.as_ref().map(|calls| stream_tool_call_deltas(calls));
                    (
                        serde_json::to_string(&OpenAiChatChunk {
                            id: completion_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id.clone(),
                            choices: vec![OpenAiChunkChoice {
                                index: 0,
                                delta: OpenAiDelta {
                                    role: None,
                                    content: None,
                                    tool_calls: delta_tool_calls,
                                },
                                finish_reason: Some(if tool_calls.is_some() {
                                    "tool_calls"
                                } else {
                                    finish_reason
                                }),
                            }],
                            usage: include_usage.then_some(OpenAiUsage {
                                prompt_tokens: generation.prompt_tokens,
                                completion_tokens: generation.tokens_generated,
                                total_tokens: generation.prompt_tokens
                                    + generation.tokens_generated,
                            }),
                            izwi_generation_time_ms: compat_profile
                                .is_relaxed()
                                .then_some(generation.generation_time_ms),
                        })
                        .unwrap_or_default(),
                        true,
                    )
                }
                ChatStreamEvent::Failed(error) => (
                    serde_json::json!({
                        "error": {
                            "message": error,
                            "type": "server_error"
                        }
                    })
                    .to_string(),
                    true,
                ),
                ChatStreamEvent::ShuttingDown => (
                    serde_json::json!({
                        "error": {
                            "message": "Server is shutting down",
                            "type": "server_error"
                        }
                    })
                    .to_string(),
                    true,
                ),
            };
            yield Ok(Event::default().data(payload));
            if terminal {
                break;
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flattens_text_parts_content() {
        let flattened = flatten_content(Some(OpenAiInboundContent::Parts(vec![
            OpenAiInboundContentPart {
                kind: Some("text".to_string()),
                text: Some("hello".to_string()),
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                video_url: None,
                input_video: None,
            },
            OpenAiInboundContentPart {
                kind: Some("input_text".to_string()),
                text: None,
                input_text: Some("world".to_string()),
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                video_url: None,
                input_video: None,
            },
        ])))
        .expect("flatten content");

        assert_eq!(flattened.runtime_text, "helloworld");
        assert!(!flattened.has_media());
    }

    #[test]
    fn flattens_multimodal_parts_without_inlining_tokens() {
        let flattened = flatten_content(Some(OpenAiInboundContent::Parts(vec![
            OpenAiInboundContentPart {
                kind: Some("text".to_string()),
                text: Some("Before ".to_string()),
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                video_url: None,
                input_video: None,
            },
            OpenAiInboundContentPart {
                kind: Some("image_url".to_string()),
                text: None,
                input_text: None,
                image_url: Some(json!({"url":"https://example.com/cat.png"})),
                input_image: None,
                image: None,
                video: None,
                video_url: None,
                input_video: None,
            },
            OpenAiInboundContentPart {
                kind: Some("video".to_string()),
                text: None,
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: Some(json!({"url":"https://example.com/clip.mp4"})),
                video_url: None,
                input_video: None,
            },
            OpenAiInboundContentPart {
                kind: Some("text".to_string()),
                text: Some(" after".to_string()),
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                video_url: None,
                input_video: None,
            },
        ])))
        .expect("flatten content");

        assert_eq!(flattened.display_text, "Before  after");
        assert!(flattened.runtime_text.contains("<|image_pad|>"));
        assert!(flattened.runtime_text.contains("<|video_pad|>"));
        assert_eq!(flattened.media_inputs.len(), 2);
    }

    #[test]
    fn flatten_content_rejects_multimodal_part_without_source() {
        let err = flatten_content(Some(OpenAiInboundContent::Parts(vec![
            OpenAiInboundContentPart {
                kind: Some("image_url".to_string()),
                text: None,
                input_text: None,
                image_url: Some(json!({})),
                input_image: None,
                image: None,
                video: None,
                video_url: None,
                input_video: None,
            },
        ])))
        .expect_err("missing source should fail");

        assert!(err.message.contains("missing a usable"));
    }

    #[test]
    fn renders_assistant_tool_calls_to_qwen_xml() {
        let xml = render_tool_calls_xml(&[json!({
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"Harare\",\"unit\":\"celsius\"}"
            }
        })])
        .expect("tool call xml should render");

        assert!(xml.contains("<tool_call>"));
        assert!(xml.contains("<function=get_weather>"));
        assert!(xml.contains("<parameter=city>\nHarare\n</parameter>"));
        assert!(xml.contains("<parameter=unit>\ncelsius\n</parameter>"));
    }

    #[test]
    fn parses_qwen_tool_call_output_into_openai_shape() {
        let output = "Checking that now.\n<tool_call>\n<function=get_time>\n<parameter=timezone>\n\"Africa/Harare\"\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_assistant_tool_output(output);

        assert_eq!(parsed.content, "Checking that now.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get_time");
        assert!(parsed.tool_calls[0]
            .function
            .arguments
            .contains("Africa/Harare"));
    }

    #[test]
    fn builds_tool_call_finish_reason_when_tool_output_detected() {
        let (content, tool_calls, finish_reason) = build_assistant_response_parts(
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>".to_string(),
        );

        assert!(content.is_none());
        assert_eq!(finish_reason, "tool_calls");
        assert_eq!(tool_calls.map(|calls| calls.len()), Some(1));
    }

    #[test]
    fn to_core_messages_collects_multimodal_parts() {
        let (messages, media_inputs) = to_core_messages_with_media(
            ModelVariant::Qwen38BGguf,
            vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(OpenAiInboundContent::Parts(vec![
                    OpenAiInboundContentPart {
                        kind: Some("image_url".to_string()),
                        text: None,
                        input_text: None,
                        image_url: Some(json!({"url":"https://example.com/cat.png"})),
                        input_image: None,
                        image: None,
                        video: None,
                        video_url: None,
                        input_video: None,
                    },
                ])),
                tool_calls: None,
            }],
            None,
            None,
            None,
        )
        .expect("multimodal parts should normalize");

        assert_eq!(messages.len(), 1);
        assert!(messages[0].content.contains("<|image_pad|>"));
        assert_eq!(media_inputs.len(), 1);
        assert!(
            validate_media_inputs_for_variant(ModelVariant::Qwen38BGguf, &media_inputs)
                .expect_err("non-qwen35 multimodal should fail")
                .contains("currently supported only for Qwen3.5")
        );
    }

    #[test]
    fn validates_rejects_non_zero_frequency_penalty() {
        let req = ChatCompletionRequest {
            model: "Qwen3-8B-GGUF".to_string(),
            messages: vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(OpenAiInboundContent::Text("hello".to_string())),
                tool_calls: None,
            }],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: Some(1),
            temperature: None,
            top_p: None,
            frequency_penalty: Some(0.5),
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
        };

        let err = validate_chat_request_compatibility(&req, OpenAiCompatibilityProfile::Strict)
            .expect_err("should reject");
        assert!(err.message.contains("frequency_penalty"));
    }

    #[test]
    fn validates_rejects_stop_sequences() {
        let req = ChatCompletionRequest {
            model: "Qwen3-8B-GGUF".to_string(),
            messages: vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(OpenAiInboundContent::Text("hello".to_string())),
                tool_calls: None,
            }],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: Some(1),
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: Some(json!(["END"])),
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
        };

        let err = validate_chat_request_compatibility(&req, OpenAiCompatibilityProfile::Strict)
            .expect_err("should reject");
        assert!(err.message.contains("stop"));
    }

    #[test]
    fn relaxed_profile_allows_stop_and_frequency_penalty_passthrough() {
        let req = ChatCompletionRequest {
            model: "Qwen3-8B-GGUF".to_string(),
            messages: vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(OpenAiInboundContent::Text("hello".to_string())),
                tool_calls: None,
            }],
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            n: Some(1),
            temperature: None,
            top_p: None,
            frequency_penalty: Some(1.0),
            presence_penalty: None,
            stop: Some(json!(["END"])),
            user: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
        };

        validate_chat_request_compatibility(&req, OpenAiCompatibilityProfile::Relaxed)
            .expect("relaxed mode should allow passthrough");
    }

    #[test]
    fn stream_tool_call_deltas_include_index_and_arguments() {
        let tool_calls = vec![OpenAiOutputToolCall {
            id: "call_123".to_string(),
            kind: "function",
            function: OpenAiOutputToolFunction {
                name: "get_weather".to_string(),
                arguments: "{\"city\":\"Harare\"}".to_string(),
            },
        }];

        let deltas = stream_tool_call_deltas(&tool_calls);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].id.as_deref(), Some("call_123"));
        assert_eq!(deltas[0].kind, Some("function"));
        assert_eq!(
            deltas[0]
                .function
                .as_ref()
                .and_then(|function| function.name.as_deref()),
            Some("get_weather")
        );
        assert!(
            deltas[0]
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_deref())
                .is_some_and(|args| args.contains("Harare"))
        );
    }
}
