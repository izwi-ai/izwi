use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::Response,
    Json,
};
use tokio::sync::mpsc;

use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::{AppState, StoredResponseInputItem, StoredResponseRecord};
use izwi_core::{
    parse_chat_model_variant, qwen35_thinking_control_content, qwen35_tools_control_content,
    ChatMessage, ChatRole, ModelVariant,
};

use super::dto::{
    ResponseDeletedObject, ResponseError, ResponseInput, ResponseInputContent,
    ResponseInputItemContent, ResponseInputItemObject, ResponseInputItemsList, ResponseObject,
    ResponseOutputContent, ResponseOutputItem, ResponseStreamCompletedPayload,
    ResponseStreamCreatedPayload, ResponseStreamDeltaPayload, ResponseStreamEnvelope,
    ResponseUsage, ResponsesCreateRequest,
};

pub async fn create_response(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<ResponsesCreateRequest>,
) -> Result<Response<Body>, ApiError> {
    let model_variant = parse_chat_model_variant(Some(&req.model))
        .map_err(|err| ApiError::bad_request(err.to_string()))?;

    let (messages, input_items) = build_input_messages(
        model_variant,
        req.instructions.as_deref(),
        req.input.clone(),
        req.enable_thinking,
        req.tools.clone(),
    )?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Responses request requires non-empty `input` or `instructions`",
        ));
    }

    if req.stream.unwrap_or(false) {
        return create_streaming_response(
            state,
            req,
            model_variant,
            messages,
            input_items,
            ctx.correlation_id,
        )
        .await;
    }

    let _permit = state.acquire_permit().await;

    let output = state
        .runtime
        .chat_generate_with_correlation(
            model_variant,
            messages,
            req.max_output_tokens.unwrap_or(1536).clamp(1, 4096),
            Some(&ctx.correlation_id),
        )
        .await?;

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = now_unix_secs();

    let usage = ResponseUsage {
        input_tokens: 0,
        output_tokens: output.tokens_generated,
        total_tokens: output.tokens_generated,
    };

    let response = ResponseObject {
        id: response_id.clone(),
        object: "response",
        created_at,
        status: "completed".to_string(),
        model: model_variant.dir_name().to_string(),
        output: vec![assistant_output_item(output.text.clone())],
        usage: usage.clone(),
        error: None,
        metadata: req.metadata.clone(),
    };

    persist_response(
        &state,
        StoredResponseRecord {
            id: response_id,
            created_at,
            status: "completed".to_string(),
            model: model_variant.dir_name().to_string(),
            input_items,
            output_text: Some(output.text),
            output_tokens: usage.output_tokens,
            error: None,
            metadata: req.metadata,
        },
        req.store,
    )
    .await;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&response).unwrap_or_default(),
        ))
        .unwrap())
}

pub async fn get_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseObject>, ApiError> {
    let store = state.response_store.read().await;
    let record = store
        .get(&response_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Response not found"))?;

    Ok(Json(record_to_response(record)))
}

pub async fn delete_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseDeletedObject>, ApiError> {
    let mut store = state.response_store.write().await;
    if store.remove(&response_id).is_none() {
        return Err(ApiError::not_found("Response not found"));
    }

    Ok(Json(ResponseDeletedObject {
        id: response_id,
        object: "response.deleted",
        deleted: true,
    }))
}

pub async fn cancel_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseObject>, ApiError> {
    let mut store = state.response_store.write().await;
    let record = store
        .get_mut(&response_id)
        .ok_or_else(|| ApiError::not_found("Response not found"))?;

    if record.status != "completed" {
        record.status = "cancelled".to_string();
        record.error = Some("Response was cancelled".to_string());
    }

    Ok(Json(record_to_response(record.clone())))
}

pub async fn list_response_input_items(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseInputItemsList>, ApiError> {
    let store = state.response_store.read().await;
    let record = store
        .get(&response_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Response not found"))?;

    let data = record
        .input_items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| ResponseInputItemObject {
            id: format!("initem_{}_{idx}", response_id),
            item_type: "message",
            role: item.role,
            content: vec![ResponseInputItemContent {
                content_type: "input_text",
                text: item.content,
            }],
        })
        .collect();

    Ok(Json(ResponseInputItemsList {
        object: "list",
        data,
    }))
}

async fn create_streaming_response(
    state: AppState,
    req: ResponsesCreateRequest,
    model_variant: izwi_core::ModelVariant,
    messages: Vec<ChatMessage>,
    input_items: Vec<StoredResponseInputItem>,
    correlation_id: String,
) -> Result<Response<Body>, ApiError> {
    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = now_unix_secs();
    let metadata = req.metadata.clone();
    let response_id_for_task = response_id.clone();
    let model_name = model_variant.dir_name().to_string();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let semaphore = state.request_semaphore.clone();
    let engine = state.runtime.clone();
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let store_state = state.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task,
                        "error": {"message": "Server is shutting down"}
                    })
                    .to_string(),
                );
                let _ = event_tx.send("[DONE]".to_string());
                return;
            }
        };

        let created_response = ResponseObject {
            id: response_id_for_task.clone(),
            object: "response",
            created_at,
            status: "in_progress".to_string(),
            model: model_name.clone(),
            output: Vec::new(),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
            },
            error: None,
            metadata: metadata.clone(),
        };

        let created_event = ResponseStreamEnvelope {
            event_type: "response.created",
            payload: ResponseStreamCreatedPayload {
                response: created_response,
            },
        };
        let _ = event_tx.send(serde_json::to_string(&created_event).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let full_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let full_text_for_cb = full_text.clone();
        let response_id_for_delta = response_id_for_task.clone();

        let result = tokio::time::timeout(timeout, async {
            engine
                .chat_generate_streaming_with_correlation(
                    model_variant,
                    messages,
                    req.max_output_tokens.unwrap_or(1536).clamp(1, 4096),
                    Some(correlation_id.as_str()),
                    move |delta| {
                        if let Ok(mut text) = full_text_for_cb.lock() {
                            text.push_str(&delta);
                        }

                        let delta_event = ResponseStreamEnvelope {
                            event_type: "response.output_text.delta",
                            payload: ResponseStreamDeltaPayload {
                                response_id: response_id_for_delta.clone(),
                                delta,
                            },
                        };

                        let _ =
                            delta_tx.send(serde_json::to_string(&delta_event).unwrap_or_default());
                    },
                )
                .await
        })
        .await;

        match result {
            Ok(Ok(generation)) => {
                let output_text = full_text.lock().map(|s| s.clone()).unwrap_or_default();
                let completed = ResponseObject {
                    id: response_id_for_task.clone(),
                    object: "response",
                    created_at,
                    status: "completed".to_string(),
                    model: model_name.clone(),
                    output: vec![assistant_output_item(output_text.clone())],
                    usage: ResponseUsage {
                        input_tokens: 0,
                        output_tokens: generation.tokens_generated,
                        total_tokens: generation.tokens_generated,
                    },
                    error: None,
                    metadata: metadata.clone(),
                };

                persist_response(
                    &store_state,
                    StoredResponseRecord {
                        id: response_id_for_task.clone(),
                        created_at,
                        status: "completed".to_string(),
                        model: model_name,
                        input_items,
                        output_text: Some(output_text),
                        output_tokens: generation.tokens_generated,
                        error: None,
                        metadata,
                    },
                    req.store,
                )
                .await;

                let completed_event = ResponseStreamEnvelope {
                    event_type: "response.completed",
                    payload: ResponseStreamCompletedPayload {
                        response: completed,
                    },
                };
                let _ = event_tx.send(serde_json::to_string(&completed_event).unwrap_or_default());
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task,
                        "error": {"message": err.to_string()}
                    })
                    .to_string(),
                );
            }
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task,
                        "error": {"message": "Response request timed out"}
                    })
                    .to_string(),
                );
            }
        }

        let _ = event_tx.send("[DONE]".to_string());
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
            if payload == "[DONE]" {
                break;
            }
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseInboundRole {
    System,
    User,
    Assistant,
    Tool,
}

const QWEN_VISION_IMAGE_TOKEN: &str = "<|vision_start|><|image_pad|><|vision_end|>";
const QWEN_VISION_VIDEO_TOKEN: &str = "<|vision_start|><|video_pad|><|vision_end|>";

fn build_input_messages(
    model_variant: ModelVariant,
    instructions: Option<&str>,
    input: Option<ResponseInput>,
    enable_thinking: Option<bool>,
    tools: Option<Vec<serde_json::Value>>,
) -> Result<(Vec<ChatMessage>, Vec<StoredResponseInputItem>), ApiError> {
    let mut messages = Vec::new();
    let mut stored = Vec::new();

    if is_qwen35_chat_variant(model_variant) {
        if let Some(enable_thinking) = enable_thinking {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: qwen35_thinking_control_content(enable_thinking),
            });
        }

        if let Some(tools) = tools.filter(|entries| !entries.is_empty()) {
            if let Some(control) = qwen35_tools_control_content(&tools) {
                messages.push(ChatMessage {
                    role: ChatRole::System,
                    content: control,
                });
            }
        }
    }

    if let Some(instructions) = instructions {
        if !instructions.trim().is_empty() {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: instructions.to_string(),
            });
            stored.push(StoredResponseInputItem {
                role: "system".to_string(),
                content: instructions.to_string(),
            });
        }
    }

    let input_items = match input {
        None => Vec::new(),
        Some(ResponseInput::Text(text)) => vec![("user".to_string(), text, false)],
        Some(ResponseInput::One(item)) => vec![normalize_input_item(item)?],
        Some(ResponseInput::Many(items)) => items
            .into_iter()
            .map(normalize_input_item)
            .collect::<Result<Vec<_>, _>>()?,
    };

    for (role, content, is_tool) in input_items {
        let parsed_role = parse_input_role(&role)?;
        if content.trim().is_empty() {
            return Err(ApiError::bad_request(
                "Response input item content cannot be empty",
            ));
        }

        match parsed_role {
            ResponseInboundRole::System => {
                messages.push(ChatMessage {
                    role: ChatRole::System,
                    content: content.clone(),
                });
                stored.push(StoredResponseInputItem { role, content });
            }
            ResponseInboundRole::User => {
                messages.push(ChatMessage {
                    role: ChatRole::User,
                    content: content.clone(),
                });
                stored.push(StoredResponseInputItem { role, content });
            }
            ResponseInboundRole::Assistant => {
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: content.clone(),
                });
                stored.push(StoredResponseInputItem { role, content });
            }
            ResponseInboundRole::Tool => {
                let wrapped = if is_tool {
                    format!("<tool_response>\n{}\n</tool_response>", content.trim())
                } else {
                    content.clone()
                };
                messages.push(ChatMessage {
                    role: ChatRole::User,
                    content: wrapped,
                });
                stored.push(StoredResponseInputItem { role, content });
            }
        }
    }

    Ok((messages, stored))
}

fn normalize_input_item(
    item: super::dto::ResponseInputItem,
) -> Result<(String, String, bool), ApiError> {
    let role = item
        .role
        .unwrap_or_else(|| "user".to_string())
        .trim()
        .to_ascii_lowercase();

    let mut content = flatten_content(item.content);
    if role == "assistant"
        && item
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    {
        let tool_xml = render_tool_calls_xml(item.tool_calls.as_deref().unwrap_or(&[]))?;
        if !tool_xml.is_empty() {
            if !content.trim().is_empty() {
                content.push_str("\n\n");
            }
            content.push_str(&tool_xml);
        }
    }

    if content.trim().is_empty() {
        return Err(ApiError::bad_request(
            "Response input item content cannot be empty",
        ));
    }

    Ok((role.clone(), content, role == "tool"))
}

fn flatten_content(content: Option<ResponseInputContent>) -> String {
    match content {
        None => String::new(),
        Some(ResponseInputContent::Text(text)) => text,
        Some(ResponseInputContent::Parts(parts)) => {
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
                if let Some(text) = part.input_text.or(part.text) {
                    out.push_str(&text);
                }
            }
            out
        }
    }
}

fn parse_input_role(raw: &str) -> Result<ResponseInboundRole, ApiError> {
    match raw {
        "system" | "developer" => Ok(ResponseInboundRole::System),
        "assistant" => Ok(ResponseInboundRole::Assistant),
        "user" => Ok(ResponseInboundRole::User),
        "tool" => Ok(ResponseInboundRole::Tool),
        other => Err(ApiError::bad_request(format!(
            "Unsupported response input role: {}",
            other
        ))),
    }
}

fn content_part_is_image(part: &super::dto::ResponseInputContentPart) -> bool {
    if matches!(
        part.kind.as_deref(),
        Some("image") | Some("image_url") | Some("input_image")
    ) {
        return true;
    }
    part.image.is_some() || part.image_url.is_some() || part.input_image.is_some()
}

fn content_part_is_video(part: &super::dto::ResponseInputContentPart) -> bool {
    if matches!(
        part.kind.as_deref(),
        Some("video") | Some("video_url") | Some("input_video")
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

fn is_qwen35_chat_variant(variant: ModelVariant) -> bool {
    matches!(
        variant,
        ModelVariant::Qwen3508B
            | ModelVariant::Qwen352B
            | ModelVariant::Qwen354B
            | ModelVariant::Qwen359B
    )
}

fn assistant_output_item(text: String) -> ResponseOutputItem {
    ResponseOutputItem {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        item_type: "message",
        role: "assistant",
        content: vec![ResponseOutputContent {
            content_type: "output_text",
            text,
        }],
    }
}

fn record_to_response(record: StoredResponseRecord) -> ResponseObject {
    ResponseObject {
        id: record.id,
        object: "response",
        created_at: record.created_at,
        status: record.status.clone(),
        model: record.model,
        output: record
            .output_text
            .map(assistant_output_item)
            .into_iter()
            .collect(),
        usage: ResponseUsage {
            input_tokens: 0,
            output_tokens: record.output_tokens,
            total_tokens: record.output_tokens,
        },
        error: record.error.map(|message| ResponseError {
            message,
            code: "response_error",
        }),
        metadata: record.metadata,
    }
}

async fn persist_response(state: &AppState, record: StoredResponseRecord, store: Option<bool>) {
    if store == Some(false) {
        return;
    }

    let mut response_store = state.response_store.write().await;
    response_store.insert(record.id.clone(), record);
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::super::dto::{ResponseInputContentPart, ResponseInputItem};
    use super::*;
    use serde_json::json;

    #[test]
    fn builds_messages_from_text_and_instructions() {
        let (messages, stored) = build_input_messages(
            ModelVariant::Qwen359B,
            Some("Be concise."),
            Some(ResponseInput::Text("Hello".to_string())),
            Some(true),
            Some(vec![json!({
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object"}
                }
            })]),
        )
        .expect("build input messages");

        assert_eq!(messages.len(), 4);
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, "system");
        assert_eq!(stored[1].role, "user");
    }

    #[test]
    fn flattens_part_content() {
        let flattened = flatten_content(Some(ResponseInputContent::Parts(vec![
            ResponseInputContentPart {
                kind: Some("input_text".to_string()),
                text: None,
                input_text: Some("part1".to_string()),
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                input_video: None,
            },
            ResponseInputContentPart {
                kind: Some("text".to_string()),
                text: Some("part2".to_string()),
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                input_video: None,
            },
        ])));
        assert_eq!(flattened, "part1part2");
    }

    #[test]
    fn flattens_multimodal_parts_to_qwen_tokens() {
        let flattened = flatten_content(Some(ResponseInputContent::Parts(vec![
            ResponseInputContentPart {
                kind: Some("text".to_string()),
                text: Some("Look ".to_string()),
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                input_video: None,
            },
            ResponseInputContentPart {
                kind: Some("image_url".to_string()),
                text: None,
                input_text: None,
                image_url: Some(json!({"url":"https://example.com/img.png"})),
                input_image: None,
                image: None,
                video: None,
                input_video: None,
            },
            ResponseInputContentPart {
                kind: Some("video".to_string()),
                text: None,
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: Some(json!({"url":"https://example.com/clip.mp4"})),
                input_video: None,
            },
        ])));
        assert_eq!(
            flattened,
            format!("Look {QWEN_VISION_IMAGE_TOKEN}{QWEN_VISION_VIDEO_TOKEN}")
        );
    }

    #[test]
    fn normalizes_input_item_default_role() {
        let item = ResponseInputItem {
            role: None,
            content: Some(ResponseInputContent::Text("hello".to_string())),
            tool_calls: None,
        };
        let (role, text, is_tool) = normalize_input_item(item).expect("normalize");
        assert_eq!(role, "user");
        assert_eq!(text, "hello");
        assert!(!is_tool);
    }

    #[test]
    fn normalizes_assistant_tool_calls_into_qwen_xml() {
        let item = ResponseInputItem {
            role: Some("assistant".to_string()),
            content: None,
            tool_calls: Some(vec![json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Harare\"}"
                }
            })]),
        };

        let (role, content, is_tool) = normalize_input_item(item).expect("normalize tool call");
        assert_eq!(role, "assistant");
        assert!(!is_tool);
        assert!(content.contains("<tool_call>"));
        assert!(content.contains("<function=get_weather>"));
        assert!(content.contains("<parameter=city>"));
    }
}
