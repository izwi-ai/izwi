//! OpenAI-compatible chat completions endpoints.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Extension, State},
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::openai::common::{parse_tool_choice, should_inject_tools, tool_choice_instruction};
use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{parse_chat_model_variant, ModelVariant};
use izwi_core::{ChatMessage, ChatRole};

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

#[derive(Debug, Clone, Deserialize)]
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
    izwi_generation_time_ms: f64,
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

const QWEN_VISION_IMAGE_TOKEN: &str = "<|vision_start|><|image_pad|><|vision_end|>";
const QWEN_VISION_VIDEO_TOKEN: &str = "<|vision_start|><|video_pad|><|vision_end|>";

fn max_new_tokens(
    variant: ModelVariant,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    let requested = max_completion_tokens.or(max_tokens);

    let default = match variant {
        ModelVariant::Gemma34BIt => 4096,
        ModelVariant::Gemma31BIt => 4096,
        _ => 1536,
    };

    requested.unwrap_or(default).clamp(1, 4096)
}

fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn to_core_messages(
    variant: ModelVariant,
    messages: Vec<OpenAiInboundMessage>,
    enable_thinking: Option<bool>,
    tools: Option<Vec<serde_json::Value>>,
    tool_choice: Option<serde_json::Value>,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut core = Vec::with_capacity(messages.len() + 2);

    if is_qwen35_chat_variant(variant) {
        let tool_choice = parse_tool_choice(tool_choice.as_ref())?;
        let tools_for_choice = tools.as_deref().unwrap_or(&[]);

        if let Some(enable_thinking) = enable_thinking {
            core.push(ChatMessage {
                role: ChatRole::System,
                content: izwi_core::qwen35_thinking_control_content(enable_thinking),
            });
        }

        if should_inject_tools(&tool_choice) {
            if let Some(tools) = tools.as_ref().filter(|entries| !entries.is_empty()) {
                if let Some(control) = izwi_core::qwen35_tools_control_content(tools) {
                    core.push(ChatMessage {
                        role: ChatRole::System,
                        content: control,
                    });
                }
            }
        }

        if let Some(instruction) = tool_choice_instruction(&tool_choice, tools_for_choice)? {
            core.push(ChatMessage {
                role: ChatRole::System,
                content: instruction,
            });
        }
    }

    for message in messages {
        let role = parse_role(&message.role)?;
        let mut content = flatten_content(message.content);

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
                if content.trim().is_empty() {
                    return Err(ApiError::bad_request(
                        "User message content cannot be empty",
                    ));
                }
                core.push(ChatMessage {
                    role: ChatRole::User,
                    content,
                });
            }
            InboundMessageRole::Assistant => {
                if content.trim().is_empty() {
                    return Err(ApiError::bad_request(
                        "Assistant message must include content or tool calls",
                    ));
                }
                core.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content,
                });
            }
            InboundMessageRole::Tool => {
                if content.trim().is_empty() {
                    return Err(ApiError::bad_request(
                        "Tool message content cannot be empty",
                    ));
                }

                let wrapped = format!("<tool_response>\n{}\n</tool_response>", content.trim());
                core.push(ChatMessage {
                    role: ChatRole::User,
                    content: wrapped,
                });
            }
        }
    }

    Ok(core)
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

fn flatten_content(content: Option<OpenAiInboundContent>) -> String {
    match content {
        None => String::new(),
        Some(OpenAiInboundContent::Text(text)) => text,
        Some(OpenAiInboundContent::Parts(parts)) => {
            let mut out = String::new();
            for part in parts {
                if content_part_is_image(&part) {
                    out.push_str(QWEN_VISION_IMAGE_TOKEN);
                    continue;
                }
                if content_part_is_video(&part) {
                    out.push_str(QWEN_VISION_VIDEO_TOKEN);
                    continue;
                }
                if let Some(text) = part.text.or(part.input_text) {
                    out.push_str(&text);
                }
            }
            out
        }
    }
}

fn content_part_is_image(part: &OpenAiInboundContentPart) -> bool {
    if matches!(
        part.kind.as_deref(),
        Some("image") | Some("image_url") | Some("input_image")
    ) {
        return true;
    }
    part.image.is_some() || part.image_url.is_some() || part.input_image.is_some()
}

fn content_part_is_video(part: &OpenAiInboundContentPart) -> bool {
    if matches!(
        part.kind.as_deref(),
        Some("video") | Some("input_video") | Some("video_url")
    ) {
        return true;
    }
    part.video.is_some() || part.input_video.is_some()
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
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        kind: "function",
        function: OpenAiOutputToolFunction {
            name: function_name.to_string(),
            arguments: serde_json::to_string(&serde_json::Value::Object(args)).ok()?,
        },
    })
}

fn is_qwen35_chat_variant(variant: ModelVariant) -> bool {
    matches!(
        variant,
        ModelVariant::Qwen3508B
            | ModelVariant::Qwen352B
            | ModelVariant::Qwen354B
            | ModelVariant::Qwen359B
    )
}

fn request_contains_multimodal_content(messages: &[OpenAiInboundMessage]) -> bool {
    messages.iter().any(|message| match &message.content {
        Some(OpenAiInboundContent::Parts(parts)) => parts
            .iter()
            .any(|part| content_part_is_image(part) || content_part_is_video(part)),
        _ => false,
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

pub async fn completions(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request(
            "This server currently supports only `n=1` for chat completions",
        ));
    }

    let variant = parse_chat_model(&req.model)?;
    if request_contains_multimodal_content(&req.messages) && !is_qwen35_chat_variant(variant) {
        return Err(ApiError::bad_request(
            "Multimodal chat input is currently supported only for Qwen3.5 chat variants",
        ));
    }
    let messages = to_core_messages(
        variant,
        req.messages.clone(),
        req.enable_thinking,
        req.tools.clone(),
        req.tool_choice.clone(),
    )?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Chat request must include at least one message",
        ));
    }

    if req.stream.unwrap_or(false) {
        let stream_response =
            complete_stream(state, req, variant, messages, ctx.correlation_id).await?;
        return Ok(stream_response.into_response());
    }

    let _permit = state.acquire_permit().await;

    let generation = state
        .runtime
        .chat_generate_with_correlation(
            variant,
            messages,
            max_new_tokens(variant, req.max_completion_tokens, req.max_tokens),
            Some(&ctx.correlation_id),
        )
        .await?;

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix_secs();
    let completion_tokens = generation.tokens_generated;
    let prompt_tokens = 0usize;
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
        izwi_generation_time_ms: generation.generation_time_ms,
    };

    Ok(Json(response).into_response())
}

async fn complete_stream(
    state: AppState,
    req: ChatCompletionRequest,
    variant: ModelVariant,
    messages: Vec<ChatMessage>,
    correlation_id: String,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let max_tokens = max_new_tokens(variant, req.max_completion_tokens, req.max_tokens);
    let model_id = variant.dir_name().to_string();
    let timeout = Duration::from_secs(state.request_timeout_secs);

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix_secs();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let engine = state.runtime.clone();
    let semaphore = state.request_semaphore.clone();
    let completion_id_for_task = completion_id.clone();
    let model_id_for_task = model_id.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "error": {
                            "message": "Server is shutting down",
                            "type": "server_error"
                        }
                    })
                    .to_string(),
                );
                let _ = event_tx.send("[DONE]".to_string());
                return;
            }
        };

        let start_chunk = OpenAiChatChunk {
            id: completion_id_for_task.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id_for_task.clone(),
            choices: vec![OpenAiChunkChoice {
                index: 0,
                delta: OpenAiDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
            izwi_generation_time_ms: None,
        };
        let _ = event_tx.send(serde_json::to_string(&start_chunk).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let result = tokio::time::timeout(timeout, async {
            engine
                .chat_generate_streaming_with_correlation(
                    variant,
                    messages,
                    max_tokens,
                    Some(correlation_id.as_str()),
                    move |delta| {
                        let chunk = OpenAiChatChunk {
                            id: completion_id_for_task.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id_for_task.clone(),
                            choices: vec![OpenAiChunkChoice {
                                index: 0,
                                delta: OpenAiDelta {
                                    role: None,
                                    content: Some(delta),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                            izwi_generation_time_ms: None,
                        };
                        let _ = delta_tx.send(serde_json::to_string(&chunk).unwrap_or_default());
                    },
                )
                .await
        })
        .await;

        match result {
            Ok(Ok(generation)) => {
                let (_, tool_calls, finish_reason) =
                    build_assistant_response_parts(generation.text);
                let final_chunk = OpenAiChatChunk {
                    id: completion_id,
                    object: "chat.completion.chunk",
                    created,
                    model: model_id,
                    choices: vec![OpenAiChunkChoice {
                        index: 0,
                        delta: OpenAiDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some(if tool_calls.is_some() {
                            "tool_calls"
                        } else {
                            finish_reason
                        }),
                    }],
                    usage: include_usage.then_some(OpenAiUsage {
                        prompt_tokens: 0,
                        completion_tokens: generation.tokens_generated,
                        total_tokens: generation.tokens_generated,
                    }),
                    izwi_generation_time_ms: Some(generation.generation_time_ms),
                };
                let _ = event_tx.send(serde_json::to_string(&final_chunk).unwrap_or_default());
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "error": {
                            "message": err.to_string(),
                            "type": "server_error"
                        }
                    })
                    .to_string(),
                );
            }
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "error": {
                            "message": "Chat request timed out",
                            "type": "timeout_error"
                        }
                    })
                    .to_string(),
                );
            }
        }

        let _ = event_tx.send("[DONE]".to_string());
    });

    let stream = async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            yield Ok(Event::default().data(event.clone()));
            if event == "[DONE]" {
                break;
            }
        }
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
                input_video: None,
            },
        ])));

        assert_eq!(flattened, "helloworld");
    }

    #[test]
    fn flattens_multimodal_parts_to_qwen_vision_tokens() {
        let flattened = flatten_content(Some(OpenAiInboundContent::Parts(vec![
            OpenAiInboundContentPart {
                kind: Some("text".to_string()),
                text: Some("Before ".to_string()),
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: None,
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
                input_video: None,
            },
        ])));

        assert_eq!(
            flattened,
            format!("Before {QWEN_VISION_IMAGE_TOKEN}{QWEN_VISION_VIDEO_TOKEN} after")
        );
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
    fn detects_multimodal_parts_in_request_messages() {
        let messages = vec![
            OpenAiInboundMessage {
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
                        input_video: None,
                    },
                ])),
                tool_calls: None,
            },
            OpenAiInboundMessage {
                role: "assistant".to_string(),
                content: Some(OpenAiInboundContent::Text("ok".to_string())),
                tool_calls: None,
            },
        ];

        assert!(request_contains_multimodal_content(&messages));
    }

    #[test]
    fn tool_choice_none_skips_tools_control_marker() {
        let core_messages = to_core_messages(
            ModelVariant::Qwen352B,
            vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(OpenAiInboundContent::Text("hello".to_string())),
                tool_calls: None,
            }],
            None,
            Some(vec![json!({
                "type": "function",
                "function": {"name": "ping", "parameters": {"type": "object"}}
            })]),
            Some(json!("none")),
        )
        .expect("to_core_messages");

        assert!(!core_messages
            .iter()
            .any(|msg| izwi_core::parse_qwen35_tools_control_content(&msg.content).is_some()));
        assert!(core_messages
            .iter()
            .any(|msg| msg.content.contains("Tool use is disabled")));
    }

    #[test]
    fn tool_choice_required_rejects_missing_tools() {
        let err = to_core_messages(
            ModelVariant::Qwen352B,
            vec![OpenAiInboundMessage {
                role: "user".to_string(),
                content: Some(OpenAiInboundContent::Text("hello".to_string())),
                tool_calls: None,
            }],
            None,
            None,
            Some(json!("required")),
        )
        .expect_err("required without tools should fail");

        assert!(err.message.contains("required"));
    }
}
