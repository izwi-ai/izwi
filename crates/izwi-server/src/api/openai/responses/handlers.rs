use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::Response,
    Json,
};

use crate::api::request_context::RequestContext;
use crate::app::chat::{generate_chat, spawn_chat_stream, ChatExecutionRequest, ChatStreamEvent};
use crate::app::chat_content::{
    flatten_content_parts, validate_media_inputs_for_variant, FlattenedMultimodalContent,
};
use crate::error::ApiError;
use crate::ids::new_uuid;
use crate::state::{AppState, StoredResponseInputItem, StoredResponseRecord};
use izwi_core::{
    parse_chat_model_variant, ChatMediaInput, ChatMessage, ChatRequestConfig, ChatRole,
    ModelVariant,
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

    let (messages, input_items, media_inputs) = build_input_messages(
        model_variant,
        req.instructions.as_deref(),
        req.input.clone(),
        req.enable_thinking,
        req.tools.clone(),
        req.tool_choice.clone(),
    )?;
    validate_media_inputs_for_variant(model_variant, &media_inputs)
        .map_err(ApiError::bad_request)?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Responses request requires non-empty `input` or `instructions`",
        ));
    }

    let execution_request = ChatExecutionRequest {
        variant: model_variant,
        messages,
        max_completion_tokens: req.max_output_tokens,
        max_tokens: None,
        temperature: req.temperature,
        top_p: req.top_p,
        presence_penalty: None,
        chat_config: ChatRequestConfig {
            enable_thinking: req.enable_thinking,
            tools: req.tools.clone().unwrap_or_default(),
            media_inputs,
        },
        correlation_id: Some(ctx.correlation_id.clone()),
    };

    if req.stream.unwrap_or(false) {
        return create_streaming_response(state, req, execution_request, input_items).await;
    }

    let output = generate_chat(&state, execution_request).await?;

    let response_id = new_uuid();
    let created_at = now_unix_secs();

    let usage = ResponseUsage {
        input_tokens: output.prompt_tokens,
        output_tokens: output.tokens_generated,
        total_tokens: output.prompt_tokens + output.tokens_generated,
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
            input_tokens: usage.input_tokens,
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
    execution_request: ChatExecutionRequest,
    input_items: Vec<StoredResponseInputItem>,
) -> Result<Response<Body>, ApiError> {
    let response_id = new_uuid();
    let created_at = now_unix_secs();
    let metadata = req.metadata.clone();
    let response_id_for_task = response_id.clone();
    let message_id = new_uuid();
    let model_name = execution_request.variant.dir_name().to_string();
    let store_state = state.clone();
    let mut event_rx = spawn_chat_stream(state.clone(), execution_request);

    let stream = async_stream::stream! {
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
        let created_event = ResponseStreamEnvelope::<ResponseStreamCreatedPayload> {
            event_type: "response.created",
            payload: ResponseStreamCreatedPayload {
                response: created_response.clone(),
            },
        };
        yield Ok::<_, Infallible>(sse_event(
            "response.created",
            &serde_json::to_string(&created_event).unwrap_or_default(),
        ));
        let in_progress_event = serde_json::json!({
            "type": "response.in_progress",
            "response": created_response,
        });
        yield Ok::<_, Infallible>(sse_event(
            "response.in_progress",
            &in_progress_event.to_string(),
        ));
        let output_item_added_event = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": message_id.clone(),
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
        });
        yield Ok::<_, Infallible>(sse_event(
            "response.output_item.added",
            &output_item_added_event.to_string(),
        ));
        let content_part_added_event = serde_json::json!({
            "type": "response.content_part.added",
            "item_id": message_id.clone(),
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": []
            }
        });
        yield Ok::<_, Infallible>(sse_event(
            "response.content_part.added",
            &content_part_added_event.to_string(),
        ));

        let mut full_text = String::new();
        while let Some(event) = event_rx.recv().await {
            let payload = match event {
                ChatStreamEvent::Started => continue,
                ChatStreamEvent::Delta(delta) => {
                    full_text.push_str(&delta);
                    serde_json::to_string(&ResponseStreamEnvelope {
                        event_type: "response.output_text.delta",
                        payload: ResponseStreamDeltaPayload {
                            response_id: response_id_for_task.clone(),
                            delta,
                        },
                    })
                    .unwrap_or_default()
                }
                ChatStreamEvent::Completed(generation) => {
                    let output_text = if generation.text.is_empty() {
                        full_text.clone()
                    } else {
                        generation.text.clone()
                    };
                    let completed = ResponseObject {
                        id: response_id_for_task.clone(),
                        object: "response",
                        created_at,
                        status: "completed".to_string(),
                        model: model_name.clone(),
                        output: vec![assistant_output_item_with_id(
                            message_id.clone(),
                            output_text.clone(),
                        )],
                        usage: ResponseUsage {
                            input_tokens: generation.prompt_tokens,
                            output_tokens: generation.tokens_generated,
                            total_tokens: generation.prompt_tokens + generation.tokens_generated,
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
                            model: model_name.clone(),
                            input_items: input_items.clone(),
                            output_text: Some(output_text.clone()),
                            input_tokens: generation.prompt_tokens,
                            output_tokens: generation.tokens_generated,
                            error: None,
                            metadata: metadata.clone(),
                        },
                        req.store,
                    )
                    .await;

                    let output_text_done_event = serde_json::json!({
                        "type": "response.output_text.done",
                        "item_id": message_id.clone(),
                        "output_index": 0,
                        "content_index": 0,
                        "text": output_text,
                    });
                    yield Ok::<_, Infallible>(sse_event(
                        "response.output_text.done",
                        &output_text_done_event.to_string(),
                    ));
                    let content_part_done_event = serde_json::json!({
                        "type": "response.content_part.done",
                        "item_id": message_id.clone(),
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": completed.output[0].content[0].text,
                            "annotations": []
                        }
                    });
                    yield Ok::<_, Infallible>(sse_event(
                        "response.content_part.done",
                        &content_part_done_event.to_string(),
                    ));
                    let output_item_done_event = serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": message_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": completed.output[0].content[0].text,
                                "annotations": []
                            }]
                        }
                    });
                    yield Ok::<_, Infallible>(sse_event(
                        "response.output_item.done",
                        &output_item_done_event.to_string(),
                    ));

                    let payload = serde_json::to_string(&ResponseStreamEnvelope {
                        event_type: "response.completed",
                        payload: ResponseStreamCompletedPayload {
                            response: completed,
                        },
                    })
                    .unwrap_or_default();
                    yield Ok::<_, Infallible>(sse_event("response.completed", &payload));
                    break;
                }
                ChatStreamEvent::Failed(error) => {
                    let failed = serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task.clone(),
                        "error": {"message": error}
                    })
                    .to_string();
                    persist_response(
                        &store_state,
                        StoredResponseRecord {
                            id: response_id_for_task.clone(),
                            created_at,
                            status: "failed".to_string(),
                            model: model_name.clone(),
                            input_items: input_items.clone(),
                            output_text: None,
                            input_tokens: 0,
                            output_tokens: 0,
                            error: Some("Response generation failed".to_string()),
                            metadata: metadata.clone(),
                        },
                        req.store,
                    )
                    .await;
                    failed
                }
                ChatStreamEvent::ShuttingDown => {
                    let failed = serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task.clone(),
                        "error": {"message": "Server is shutting down"}
                    })
                    .to_string();
                    persist_response(
                        &store_state,
                        StoredResponseRecord {
                            id: response_id_for_task.clone(),
                            created_at,
                            status: "failed".to_string(),
                            model: model_name.clone(),
                            input_items: input_items.clone(),
                            output_text: None,
                            input_tokens: 0,
                            output_tokens: 0,
                            error: Some("Server is shutting down".to_string()),
                            metadata: metadata.clone(),
                        },
                        req.store,
                    )
                    .await;
                    failed
                }
            };
            let event_type = match serde_json::from_str::<serde_json::Value>(&payload)
                .ok()
                .and_then(|value| value.get("type").and_then(|value| value.as_str()).map(str::to_string))
            {
                Some(value) => value,
                None => "response.event".to_string(),
            };
            yield Ok::<_, Infallible>(sse_event(&event_type, &payload));
            if payload.contains("\"type\":\"response.failed\"") {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseInboundRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone)]
struct NormalizedInputItem {
    role: String,
    display_content: String,
    runtime_content: String,
    is_tool: bool,
    media_inputs: Vec<ChatMediaInput>,
}

fn build_input_messages(
    _model_variant: ModelVariant,
    instructions: Option<&str>,
    input: Option<ResponseInput>,
    _enable_thinking: Option<bool>,
    _tools: Option<Vec<serde_json::Value>>,
    _tool_choice: Option<serde_json::Value>,
) -> Result<
    (
        Vec<ChatMessage>,
        Vec<StoredResponseInputItem>,
        Vec<ChatMediaInput>,
    ),
    ApiError,
> {
    let mut messages = Vec::new();
    let mut stored = Vec::new();
    let mut media_inputs = Vec::new();

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
        Some(ResponseInput::Text(text)) => vec![NormalizedInputItem {
            role: "user".to_string(),
            display_content: text.clone(),
            runtime_content: text,
            is_tool: false,
            media_inputs: Vec::new(),
        }],
        Some(ResponseInput::One(item)) => vec![normalize_input_item(item)?],
        Some(ResponseInput::Many(items)) => items
            .into_iter()
            .map(normalize_input_item)
            .collect::<Result<Vec<_>, _>>()?,
    };

    for item in input_items {
        let role = item.role.clone();
        let parsed_role = parse_input_role(&role)?;
        if item.runtime_content.trim().is_empty() && item.media_inputs.is_empty() {
            return Err(ApiError::bad_request(
                "Response input item content cannot be empty",
            ));
        }
        media_inputs.extend(item.media_inputs.clone());

        match parsed_role {
            ResponseInboundRole::System => {
                if !item.media_inputs.is_empty() {
                    return Err(ApiError::bad_request(
                        "System response input items cannot contain images or videos",
                    ));
                }
                messages.push(ChatMessage {
                    role: ChatRole::System,
                    content: item.runtime_content.clone(),
                });
                stored.push(StoredResponseInputItem {
                    role,
                    content: item.display_content,
                });
            }
            ResponseInboundRole::User => {
                messages.push(ChatMessage {
                    role: ChatRole::User,
                    content: item.runtime_content.clone(),
                });
                stored.push(StoredResponseInputItem {
                    role,
                    content: item.display_content,
                });
            }
            ResponseInboundRole::Assistant => {
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: item.runtime_content.clone(),
                });
                stored.push(StoredResponseInputItem {
                    role,
                    content: item.display_content,
                });
            }
            ResponseInboundRole::Tool => {
                let wrapped = if item.is_tool {
                    format!(
                        "<tool_response>\n{}\n</tool_response>",
                        item.runtime_content.trim()
                    )
                } else {
                    item.runtime_content.clone()
                };
                messages.push(ChatMessage {
                    role: ChatRole::User,
                    content: wrapped,
                });
                stored.push(StoredResponseInputItem {
                    role,
                    content: item.display_content,
                });
            }
        }
    }

    Ok((messages, stored, media_inputs))
}

fn normalize_input_item(
    item: super::dto::ResponseInputItem,
) -> Result<NormalizedInputItem, ApiError> {
    let role = item
        .role
        .unwrap_or_else(|| "user".to_string())
        .trim()
        .to_ascii_lowercase();

    let flattened = flatten_content(item.content)?;
    let mut runtime_content = flattened.runtime_text.clone();
    if role == "assistant"
        && item
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    {
        let tool_xml = render_tool_calls_xml(item.tool_calls.as_deref().unwrap_or(&[]))?;
        if !tool_xml.is_empty() {
            if !runtime_content.trim().is_empty() {
                runtime_content.push_str("\n\n");
            }
            runtime_content.push_str(&tool_xml);
        }
    }

    if runtime_content.trim().is_empty() && !flattened.has_media() {
        return Err(ApiError::bad_request(
            "Response input item content cannot be empty",
        ));
    }

    Ok(NormalizedInputItem {
        role: role.clone(),
        display_content: flattened.display_text,
        runtime_content,
        is_tool: role == "tool",
        media_inputs: flattened.media_inputs,
    })
}

fn flatten_content(
    content: Option<ResponseInputContent>,
) -> Result<FlattenedMultimodalContent, ApiError> {
    match content {
        None => Ok(FlattenedMultimodalContent::default()),
        Some(ResponseInputContent::Text(text)) => Ok(FlattenedMultimodalContent {
            display_text: text.clone(),
            runtime_text: text,
            media_inputs: Vec::new(),
        }),
        Some(ResponseInputContent::Parts(parts)) => {
            flatten_content_parts(&content_parts_to_values(&parts)?).map_err(ApiError::bad_request)
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

fn content_parts_to_values(
    parts: &[super::dto::ResponseInputContentPart],
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

fn assistant_output_item(text: String) -> ResponseOutputItem {
    assistant_output_item_with_id(new_uuid(), text)
}

fn assistant_output_item_with_id(id: String, text: String) -> ResponseOutputItem {
    ResponseOutputItem {
        id,
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
            input_tokens: record.input_tokens,
            output_tokens: record.output_tokens,
            total_tokens: record.input_tokens + record.output_tokens,
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

    state.store_response_record(record).await;
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn sse_event(event_name: &str, payload: &str) -> String {
    format!("event: {event_name}\ndata: {payload}\n\n")
}

#[cfg(test)]
mod tests {
    use super::super::dto::{ResponseInputContentPart, ResponseInputItem};
    use super::*;
    use serde_json::json;

    #[test]
    fn builds_messages_from_text_and_instructions() {
        let (messages, stored, media_inputs) = build_input_messages(
            ModelVariant::Qwen38BGguf,
            Some("Be concise."),
            Some(ResponseInput::Text("Hello".to_string())),
            None,
            None,
            None,
        )
        .expect("build input messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, "system");
        assert_eq!(stored[1].role, "user");
        assert!(media_inputs.is_empty());
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
                video_url: None,
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
                video_url: None,
                input_video: None,
            },
        ])))
        .expect("flatten");
        assert_eq!(flattened.runtime_text, "part1part2");
        assert!(!flattened.has_media());
    }

    #[test]
    fn flattens_multimodal_parts_without_inlining_tokens() {
        let flattened = flatten_content(Some(ResponseInputContent::Parts(vec![
            ResponseInputContentPart {
                kind: Some("text".to_string()),
                text: Some("Look ".to_string()),
                input_text: None,
                image_url: None,
                input_image: None,
                image: None,
                video: None,
                video_url: None,
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
                video_url: None,
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
                video_url: None,
                input_video: None,
            },
        ])))
        .expect("flatten");
        assert_eq!(flattened.display_text, "Look ");
        assert!(flattened.runtime_text.contains("<|image_pad|>"));
        assert!(flattened.runtime_text.contains("<|video_pad|>"));
        assert_eq!(flattened.media_inputs.len(), 2);
    }

    #[test]
    fn flatten_content_rejects_multimodal_part_without_source() {
        let err = flatten_content(Some(ResponseInputContent::Parts(vec![
            ResponseInputContentPart {
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
    fn normalizes_input_item_default_role() {
        let item = ResponseInputItem {
            role: None,
            content: Some(ResponseInputContent::Text("hello".to_string())),
            tool_calls: None,
        };
        let normalized = normalize_input_item(item).expect("normalize");
        assert_eq!(normalized.role, "user");
        assert_eq!(normalized.runtime_content, "hello");
        assert!(!normalized.is_tool);
        assert!(normalized.media_inputs.is_empty());
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

        let normalized = normalize_input_item(item).expect("normalize tool call");
        assert_eq!(normalized.role, "assistant");
        assert!(!normalized.is_tool);
        assert!(normalized.media_inputs.is_empty());
        assert!(normalized.runtime_content.contains("<tool_call>"));
        assert!(normalized
            .runtime_content
            .contains("<function=get_weather>"));
        assert!(normalized.runtime_content.contains("<parameter=city>"));
    }

    #[test]
    fn build_input_messages_collects_multimodal_input() {
        let (messages, _stored, media_inputs) = build_input_messages(
            ModelVariant::Qwen38BGguf,
            None,
            Some(ResponseInput::Many(vec![ResponseInputItem {
                role: Some("user".to_string()),
                content: Some(ResponseInputContent::Parts(vec![
                    ResponseInputContentPart {
                        kind: Some("image_url".to_string()),
                        text: None,
                        input_text: None,
                        image_url: Some(json!({"url":"https://example.com/img.png"})),
                        input_image: None,
                        image: None,
                        video: None,
                        video_url: None,
                        input_video: None,
                    },
                ])),
                tool_calls: None,
            }])),
            None,
            None,
            None,
        )
        .expect("multimodal input should normalize");

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
    fn sse_event_renders_named_event_and_data_lines() {
        let payload = "{\"type\":\"response.created\"}";
        let encoded = sse_event("response.created", payload);
        assert!(encoded.starts_with("event: response.created\n"));
        assert!(encoded.contains("\ndata: {\"type\":\"response.created\"}\n"));
        assert!(encoded.ends_with("\n\n"));
    }

    #[test]
    fn assistant_output_item_with_id_preserves_message_id() {
        let item = assistant_output_item_with_id("msg_custom".to_string(), "hello".to_string());
        assert_eq!(item.id, "msg_custom");
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role, "assistant");
        assert_eq!(item.content.len(), 1);
        assert_eq!(item.content[0].content_type, "output_text");
        assert_eq!(item.content[0].text, "hello");
    }
}
