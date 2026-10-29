use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::Response,
    Json,
};
use futures::Stream;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::api::openai::convert::{
    append_assistant_tool_calls, flatten_content, inbound_content_value, tool_result_message,
};
use crate::api::openai::tool_calls::{
    parse_answer, resolve_tool_policy, ParsedOutput, ToolCall, ToolCallFormat, ToolCallParser,
};
use crate::api::openai::tool_response::{
    reject_deferred_tool_results, ToolResponseDialect, ToolResponses, ToolResult,
};
use crate::api::request_context::RequestContext;
use crate::app::chat::{
    chat_top_k, generate_chat, resolve_chat_request_config, start_chat_stream,
    validate_chat_request_ranges, ChatExecutionRequest, ChatStreamEvent,
};
use crate::app::chat_content::validate_media_inputs_for_variant;
use crate::error::{ApiError, ApiJson, ErrorScope};
use crate::ids::new_prefixed_id;
use crate::state::{AppState, StoredFunctionCall, StoredResponseInputItem, StoredResponseRecord};
use izwi_core::chat_reasoning::{reasoning_opened_in_prompt, PromptOpener};
use izwi_core::{
    parse_chat_model_variant, ChatGeneration, ChatMediaInput, ChatMessage, ChatRole, ModelVariant,
};

use super::dto::{
    ResponseDeletedObject, ResponseError, ResponseInput, ResponseInputItemContent,
    ResponseInputItemObject, ResponseInputItemsList, ResponseInputTokensDetails, ResponseObject,
    ResponseOutputContent, ResponseOutputItem, ResponseStreamCompletedPayload,
    ResponseStreamCreatedPayload, ResponseStreamDeltaPayload, ResponseStreamEnvelope,
    ResponseUsage, ResponsesCreateRequest,
};

const RESPONSE_STREAM_INTERRUPTED_ERROR: &str = "Response stream ended before a terminal event";

fn response_stream_failure_payload(response_id: &str, message: impl Into<String>) -> String {
    serde_json::json!({
        "type": "response.failed",
        "response_id": response_id,
        "error": {"message": message.into()}
    })
    .to_string()
}

fn response_stream_interruption_payload(response_id: &str, saw_terminal: bool) -> Option<String> {
    (!saw_terminal)
        .then(|| response_stream_failure_payload(response_id, RESPONSE_STREAM_INTERRUPTED_ERROR))
}

/// Responses declares a function tool flat (`{"type": "function", "name",
/// "parameters"}`) and names a forced one as `{"type": "function", "name"}`;
/// chat completions nests both under `function`, which the chat templates
/// and the tool policy read. The nested form is accepted as is.
fn chat_completions_function(value: &Value) -> Value {
    match value.as_object() {
        Some(object)
            if object.get("type").and_then(Value::as_str) == Some("function")
                && !object.contains_key("function")
                && object.contains_key("name") =>
        {
            let function = object
                .iter()
                .filter(|(key, _)| key.as_str() != "type")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<serde_json::Map<_, _>>();
            json!({"type": "function", "function": function})
        }
        _ => value.clone(),
    }
}

pub async fn create_response(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<ResponsesCreateRequest>,
) -> Result<Response<Body>, ApiError> {
    let model_variant = parse_chat_model_variant(Some(&req.model))
        .map_err(|err| ApiError::bad_request(err.to_string()))?;

    let tools = req.tools.as_deref().map(|tools| {
        tools
            .iter()
            .map(chat_completions_function)
            .collect::<Vec<_>>()
    });
    let (messages, input_items, media_inputs) = build_input_messages(
        model_variant,
        req.instructions.as_deref(),
        req.input.clone(),
        tools.as_deref().unwrap_or_default(),
    )?;
    validate_media_inputs_for_variant(model_variant, &media_inputs)
        .map_err(ApiError::bad_request)?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Responses request requires non-empty `input` or `instructions`",
        ));
    }

    let tool_choice = req.tool_choice.as_ref().map(chat_completions_function);
    let tool_policy = resolve_tool_policy(model_variant, tools.as_deref(), tool_choice.as_ref())?;
    let tool_format = tool_policy.format;
    let chat_config = resolve_chat_request_config(
        req.enable_thinking,
        req.reasoning_effort,
        req.preserve_thinking,
        req.chat_template_kwargs.as_ref(),
        tool_policy.tools,
        media_inputs,
        req.truncation,
    )?;
    // Unsplit output gets back the `<think>` the prompt opened.
    let opened_in_prompt = reasoning_opened_in_prompt(model_variant, chat_config.enable_thinking);
    if req.max_output_tokens == Some(0) {
        return Err(ApiError::invalid_param(
            "max_output_tokens",
            "`max_output_tokens` must be at least 1",
        ));
    }
    let execution_request = ChatExecutionRequest {
        variant: model_variant,
        messages,
        max_completion_tokens: req.max_output_tokens,
        max_tokens: None,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: chat_top_k(req.top_k)?,
        repetition_penalty: req.repetition_penalty,
        presence_penalty: req.presence_penalty,
        seed: None,
        stop: Vec::new(),
        logprobs: None,
        top_logprobs: None,
        response_format_json_object: false,
        skip_special_tokens: req.skip_special_tokens,
        chat_config,
        correlation_id: Some(ctx.correlation_id.clone()),
    };
    validate_chat_request_ranges(&execution_request)?;

    if req.stream.unwrap_or(false) {
        let event_rx = start_chat_stream(state.clone(), execution_request).await?;
        let stream = response_event_stream(
            state,
            ResponseStreamRequest {
                response_id: new_prefixed_id("resp_"),
                created_at: now_unix_secs(),
                model: model_variant.dir_name().to_string(),
                metadata: req.metadata,
                store: req.store,
                tenant_scope: ctx.tenant_scope(),
                input_items,
                tool_format,
                opened_in_prompt,
            },
            event_rx,
        );
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(stream))
            .unwrap());
    }

    let output = generate_chat(&state, execution_request).await?;

    let response_id = new_prefixed_id("resp_");
    let created_at = now_unix_secs();

    let usage = ResponseUsage {
        input_tokens: output.prompt_tokens,
        output_tokens: output.tokens_generated,
        total_tokens: output.prompt_tokens + output.tokens_generated,
        input_tokens_details: ResponseInputTokensDetails {
            cached_tokens: output.cached_prompt_tokens.unwrap_or(0),
        },
    };
    let (output_text, function_calls) = answer_output(
        tool_format,
        PromptOpener::new(opened_in_prompt).push(&output.text),
    );

    let response = ResponseObject {
        id: response_id.clone(),
        object: "response",
        created_at,
        status: "completed".to_string(),
        model: model_variant.dir_name().to_string(),
        output: output_items(
            new_prefixed_id("msg_"),
            output_text.clone(),
            &function_calls,
        ),
        usage: usage.clone(),
        error: None,
        metadata: req.metadata.clone(),
    };

    persist_response(
        &state,
        StoredResponseRecord {
            id: response_id,
            tenant_scope: ctx.tenant_scope(),
            created_at,
            status: "completed".to_string(),
            model: model_variant.dir_name().to_string(),
            input_items,
            output_text,
            output_function_calls: function_calls,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            input_cached_tokens: usage.input_tokens_details.cached_tokens,
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

/// A whole answer's message text and tool calls: parsed in `tool_format`,
/// or all text without one.
fn answer_output(
    tool_format: Option<ToolCallFormat>,
    text: String,
) -> (Option<String>, Vec<StoredFunctionCall>) {
    match tool_format {
        Some(format) => {
            let (text, calls) = parse_answer(format, &text);
            (text, calls.into_iter().map(stored_function_call).collect())
        }
        None => (Some(text), Vec::new()),
    }
}

fn stored_function_call(call: ToolCall) -> StoredFunctionCall {
    StoredFunctionCall {
        id: new_prefixed_id("fc_"),
        call_id: call.id,
        name: call.function.name,
        arguments: call.function.arguments,
    }
}

/// A response's output: its message (when the answer has text, or made no
/// call), then one `function_call` item per tool call.
fn output_items(
    message_id: String,
    text: Option<String>,
    calls: &[StoredFunctionCall],
) -> Vec<ResponseOutputItem> {
    let message = match text {
        Some(text) => Some(assistant_output_item_with_id(message_id, text)),
        None if calls.is_empty() => Some(assistant_output_item_with_id(message_id, String::new())),
        None => None,
    };
    message
        .into_iter()
        .chain(calls.iter().map(function_call_item))
        .collect()
}

fn function_call_item(call: &StoredFunctionCall) -> ResponseOutputItem {
    ResponseOutputItem::FunctionCall {
        id: call.id.clone(),
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
        status: "completed",
    }
}

/// Whether the caller's tenant owns a stored response. Another tenant's
/// record is reported as missing, so ids do not leak across tenants.
fn owned_by(record: &StoredResponseRecord, ctx: &RequestContext) -> bool {
    record.tenant_scope == ctx.tenant_scope()
}

fn response_not_found() -> ApiError {
    ApiError::not_found("Response not found")
}

pub async fn get_response(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseObject>, ApiError> {
    let store = state.response_store.read().await;
    let record = store
        .get(&response_id)
        .filter(|record| owned_by(record, &ctx))
        .cloned()
        .ok_or_else(response_not_found)?;

    Ok(Json(record_to_response(record)))
}

pub async fn delete_response(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseDeletedObject>, ApiError> {
    let mut store = state.response_store.write().await;
    if !store
        .get(&response_id)
        .is_some_and(|record| owned_by(record, &ctx))
    {
        return Err(response_not_found());
    }
    store.remove(&response_id);

    Ok(Json(ResponseDeletedObject {
        id: response_id,
        object: "response.deleted",
        deleted: true,
    }))
}

pub async fn cancel_response(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseObject>, ApiError> {
    let mut store = state.response_store.write().await;
    let record = store
        .get_mut(&response_id)
        .filter(|record| owned_by(record, &ctx))
        .ok_or_else(response_not_found)?;

    if record.status != "completed" {
        record.status = "cancelled".to_string();
        record.error = Some("Response was cancelled".to_string());
    }

    Ok(Json(record_to_response(record.clone())))
}

pub async fn list_response_input_items(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseInputItemsList>, ApiError> {
    let store = state.response_store.read().await;
    let record = store
        .get(&response_id)
        .filter(|record| owned_by(record, &ctx))
        .cloned()
        .ok_or_else(response_not_found)?;

    let data = record
        .input_items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| {
            let id = format!("initem_{}_{idx}", response_id);
            match item.item {
                Some(Value::Object(mut raw)) => {
                    raw.insert("id".to_string(), Value::String(id));
                    Value::Object(raw)
                }
                _ => serde_json::to_value(ResponseInputItemObject {
                    id,
                    item_type: "message",
                    role: item.role,
                    content: vec![ResponseInputItemContent {
                        content_type: "input_text",
                        text: item.content,
                    }],
                })
                .unwrap_or_default(),
            }
        })
        .collect();

    Ok(Json(ResponseInputItemsList {
        object: "list",
        data,
    }))
}

/// What a streamed response needs besides its chat events.
struct ResponseStreamRequest {
    response_id: String,
    created_at: u64,
    model: String,
    metadata: Option<Value>,
    store: Option<bool>,
    tenant_scope: String,
    input_items: Vec<StoredResponseInputItem>,
    /// The tool-call format to parse the answer in; `None` keeps it text.
    tool_format: Option<ToolCallFormat>,
    /// The prompt opened the `<think>` block (Qwen3.5/3.6/3.8 with thinking on).
    opened_in_prompt: bool,
}

/// One streamed `function_call` output item.
struct StreamedCall {
    call: StoredFunctionCall,
    output_index: usize,
}

/// The output items of a streamed response as they open: the assistant
/// message on its first text, a `function_call` on each call's first
/// fragment. Every method returns the SSE events it made.
struct ResponseOutputStream {
    response_id: String,
    message_id: String,
    parser: Option<ToolCallParser>,
    opened_in_prompt: bool,
    opener: PromptOpener,
    next_output_index: usize,
    /// The message's output index once it opened.
    message_index: Option<usize>,
    text: String,
    calls: Vec<StreamedCall>,
}

impl ResponseOutputStream {
    fn new(
        response_id: String,
        tool_format: Option<ToolCallFormat>,
        opened_in_prompt: bool,
    ) -> Self {
        Self {
            response_id,
            message_id: new_prefixed_id("msg_"),
            parser: tool_format.map(ToolCallParser::new),
            opened_in_prompt,
            opener: PromptOpener::new(opened_in_prompt),
            next_output_index: 0,
            message_index: None,
            text: String::new(),
            calls: Vec::new(),
        }
    }

    fn push(&mut self, delta: String) -> Vec<String> {
        let delta = self.opener.push(&delta);
        let parsed = match self.parser.as_mut() {
            Some(parser) => parser.push(&delta),
            None => ParsedOutput {
                content: delta,
                tool_calls: Vec::new(),
            },
        };
        self.apply(parsed)
    }

    fn apply(&mut self, parsed: ParsedOutput) -> Vec<String> {
        let mut events = Vec::new();
        if !parsed.content.is_empty() {
            let output_index = self.open_message(&mut events);
            self.text.push_str(&parsed.content);
            events.push(sse_json(
                "response.output_text.delta",
                &ResponseStreamEnvelope {
                    event_type: "response.output_text.delta",
                    payload: ResponseStreamDeltaPayload {
                        response_id: self.response_id.clone(),
                        item_id: self.message_id.clone(),
                        output_index,
                        content_index: 0,
                        delta: parsed.content,
                    },
                },
            ));
        }
        for fragment in parsed.tool_calls {
            if let Some(call_id) = fragment.id {
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                let call = StoredFunctionCall {
                    id: new_prefixed_id("fc_"),
                    call_id,
                    name: fragment.function.name.unwrap_or_default(),
                    arguments: String::new(),
                };
                events.push(sse_json(
                    "response.output_item.added",
                    &json!({
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": call.id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": "",
                        },
                    }),
                ));
                self.calls.push(StreamedCall { call, output_index });
            }
            let Some(streamed) = self.calls.get_mut(fragment.index) else {
                continue;
            };
            if fragment.function.arguments.is_empty() {
                continue;
            }
            streamed
                .call
                .arguments
                .push_str(&fragment.function.arguments);
            events.push(sse_json(
                "response.function_call_arguments.delta",
                &json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": streamed.call.id,
                    "output_index": streamed.output_index,
                    "delta": fragment.function.arguments,
                }),
            ));
        }
        events
    }

    fn open_message(&mut self, events: &mut Vec<String>) -> usize {
        if let Some(index) = self.message_index {
            return index;
        }
        let index = self.next_output_index;
        self.next_output_index += 1;
        self.message_index = Some(index);
        events.push(sse_json(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": index,
                "item": {
                    "id": self.message_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }),
        ));
        events.push(sse_json(
            "response.content_part.added",
            &json!({
                "type": "response.content_part.added",
                "item_id": self.message_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }),
        ));
        index
    }

    /// Flush the parser and close every open item. `final_text` replaces
    /// the streamed text of an answer without tools (the engine's whole
    /// answer). Returns the events, the output items in output order, and
    /// the message text and calls to store.
    fn finish(mut self, final_text: &str) -> FinishedOutput {
        let mut events = match self.parser.as_mut().map(ToolCallParser::finish) {
            Some(tail) => self.apply(tail),
            None => Vec::new(),
        };
        if self.parser.is_none() && !final_text.is_empty() {
            self.text = PromptOpener::new(self.opened_in_prompt).push(final_text);
        }
        if self.message_index.is_none() && self.calls.is_empty() {
            self.open_message(&mut events);
        }
        let mut closing: Vec<(usize, ResponseOutputItem, Vec<String>)> = Vec::new();
        if let Some(index) = self.message_index {
            let message = assistant_output_item_with_id(self.message_id.clone(), self.text.clone());
            closing.push((
                index,
                message.clone(),
                vec![
                    sse_json(
                        "response.output_text.done",
                        &json!({
                            "type": "response.output_text.done",
                            "item_id": self.message_id,
                            "output_index": index,
                            "content_index": 0,
                            "text": self.text,
                        }),
                    ),
                    sse_json(
                        "response.content_part.done",
                        &json!({
                            "type": "response.content_part.done",
                            "item_id": self.message_id,
                            "output_index": index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": self.text, "annotations": []},
                        }),
                    ),
                    sse_json(
                        "response.output_item.done",
                        &json!({
                            "type": "response.output_item.done",
                            "output_index": index,
                            "item": {
                                "id": self.message_id,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": self.text,
                                    "annotations": [],
                                }],
                            },
                        }),
                    ),
                ],
            ));
        }
        for streamed in &self.calls {
            let call = &streamed.call;
            closing.push((
                streamed.output_index,
                function_call_item(call),
                vec![
                    sse_json(
                        "response.function_call_arguments.done",
                        &json!({
                            "type": "response.function_call_arguments.done",
                            "item_id": call.id,
                            "output_index": streamed.output_index,
                            "arguments": call.arguments,
                        }),
                    ),
                    sse_json(
                        "response.output_item.done",
                        &json!({
                            "type": "response.output_item.done",
                            "output_index": streamed.output_index,
                            "item": function_call_item(call),
                        }),
                    ),
                ],
            ));
        }
        closing.sort_by_key(|(index, ..)| *index);
        let mut output = Vec::with_capacity(closing.len());
        for (_, item, item_events) in closing {
            output.push(item);
            events.extend(item_events);
        }
        FinishedOutput {
            events,
            output,
            text: self.message_index.map(|_| self.text),
            calls: self
                .calls
                .into_iter()
                .map(|streamed| streamed.call)
                .collect(),
        }
    }
}

struct FinishedOutput {
    events: Vec<String>,
    output: Vec<ResponseOutputItem>,
    text: Option<String>,
    calls: Vec<StoredFunctionCall>,
}

/// The SSE events of a streamed response: created and in progress, the
/// output items as they open and grow, their done events, and the terminal
/// `response.completed` or `response.failed`, then `[DONE]`.
fn response_event_stream(
    state: AppState,
    request: ResponseStreamRequest,
    mut event_rx: mpsc::Receiver<ChatStreamEvent>,
) -> impl Stream<Item = Result<String, Infallible>> {
    // The body is polled after the handler returns, outside the request's
    // error scope; capture it here.
    let errors = ErrorScope::current();
    async_stream::stream! {
        let ResponseStreamRequest {
            response_id,
            created_at,
            model,
            metadata,
            store,
            tenant_scope,
            input_items,
            tool_format,
            opened_in_prompt,
        } = request;
        let record = |status: &str,
                      output_text: Option<String>,
                      output_function_calls: Vec<StoredFunctionCall>,
                      generation: Option<&ChatGeneration>,
                      error: Option<String>| StoredResponseRecord {
            id: response_id.clone(),
            tenant_scope: tenant_scope.clone(),
            created_at,
            status: status.to_string(),
            model: model.clone(),
            input_items: input_items.clone(),
            output_text,
            output_function_calls,
            input_tokens: generation.map_or(0, |generation| generation.prompt_tokens),
            output_tokens: generation.map_or(0, |generation| generation.tokens_generated),
            input_cached_tokens: generation
                .and_then(|generation| generation.cached_prompt_tokens)
                .unwrap_or(0),
            error,
            metadata: metadata.clone(),
        };
        let created_response = ResponseObject {
            id: response_id.clone(),
            object: "response",
            created_at,
            status: "in_progress".to_string(),
            model: model.clone(),
            output: Vec::new(),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
                input_tokens_details: ResponseInputTokensDetails { cached_tokens: 0 },
            },
            error: None,
            metadata: metadata.clone(),
        };
        yield Ok::<_, Infallible>(sse_json(
            "response.created",
            &ResponseStreamEnvelope {
                event_type: "response.created",
                payload: ResponseStreamCreatedPayload {
                    response: created_response.clone(),
                },
            },
        ));
        yield Ok(sse_json(
            "response.in_progress",
            &json!({"type": "response.in_progress", "response": created_response}),
        ));

        let mut output =
            ResponseOutputStream::new(response_id.clone(), tool_format, opened_in_prompt);
        let mut saw_terminal = false;
        while let Some(event) = event_rx.recv().await {
            match event {
                ChatStreamEvent::Started => {}
                ChatStreamEvent::Delta { text, .. } => {
                    for event in output.push(text) {
                        yield Ok(event);
                    }
                }
                ChatStreamEvent::Completed(generation) => {
                    let finished = output.finish(&generation.text);
                    for event in finished.events {
                        yield Ok(event);
                    }
                    let completed = ResponseObject {
                        id: response_id.clone(),
                        object: "response",
                        created_at,
                        status: "completed".to_string(),
                        model: model.clone(),
                        output: finished.output,
                        usage: ResponseUsage {
                            input_tokens: generation.prompt_tokens,
                            output_tokens: generation.tokens_generated,
                            total_tokens: generation.prompt_tokens + generation.tokens_generated,
                            input_tokens_details: ResponseInputTokensDetails {
                                cached_tokens: generation.cached_prompt_tokens.unwrap_or(0),
                            },
                        },
                        error: None,
                        metadata: metadata.clone(),
                    };
                    persist_response(
                        &state,
                        record(
                            "completed",
                            finished.text,
                            finished.calls,
                            Some(generation.as_ref()),
                            None,
                        ),
                        store,
                    )
                    .await;
                    yield Ok(sse_json(
                        "response.completed",
                        &ResponseStreamEnvelope {
                            event_type: "response.completed",
                            payload: ResponseStreamCompletedPayload { response: completed },
                        },
                    ));
                    saw_terminal = true;
                    break;
                }
                ChatStreamEvent::Failed(error) => {
                    persist_response(
                        &state,
                        record(
                            "failed",
                            None,
                            Vec::new(),
                            None,
                            Some("Response generation failed".to_string()),
                        ),
                        store,
                    )
                    .await;
                    yield Ok(sse_event(
                        "response.failed",
                        &response_stream_failure_payload(&response_id, errors.client_message(&error)),
                    ));
                    saw_terminal = true;
                    break;
                }
            }
        }
        if let Some(payload) = response_stream_interruption_payload(&response_id, saw_terminal) {
            persist_response(
                &state,
                record(
                    "failed",
                    None,
                    Vec::new(),
                    None,
                    Some(RESPONSE_STREAM_INTERRUPTED_ERROR.to_string()),
                ),
                store,
            )
            .await;
            yield Ok(sse_event("response.failed", &payload));
        }
        yield Ok("data: [DONE]\n\n".to_string());
    }
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
    media_inputs: Vec<ChatMediaInput>,
}

/// The input's messages as the runtime's. `tools` are the request's tool
/// definitions, deferred ones included, which an LFM3 tool result's
/// `tool_reference` items name.
fn build_input_messages(
    model_variant: ModelVariant,
    instructions: Option<&str>,
    input: Option<ResponseInput>,
    tools: &[Value],
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
                item: None,
            });
        }
    }

    let items = match input {
        None => Vec::new(),
        Some(ResponseInput::Text(text)) => vec![super::dto::ResponseInputItem {
            role: Some("user".to_string()),
            content: Some(crate::api::openai::convert::InboundContent::Text(text)),
            ..Default::default()
        }],
        Some(ResponseInput::One(item)) => vec![*item],
        Some(ResponseInput::Many(items)) => items,
    };

    // Consecutive `function_call` items are one assistant turn, written
    // after the text of an assistant message right before them.
    let mut pending_calls = Vec::new();
    let mut previous_was_assistant_message = false;
    // LFM3's templates render tool results themselves (deferred tools,
    // merged results), so their turns are rendered here as the template does.
    let mut tool_responses =
        ToolResponseDialect::of(model_variant).map(|dialect| ToolResponses::new(dialect, tools));
    let mut items = items.into_iter().peekable();
    while let Some(item) = items.next() {
        // Only the next result's id and `additional_tools` decide a merge.
        let next_result = items
            .peek()
            .filter(|next| is_tool_result_item(next))
            .map(|next| ToolResult {
                content: None,
                additional_tools: next.additional_tools.as_ref(),
                tool_call_id: next.call_id.as_deref(),
            });
        match item.item_type.as_deref() {
            Some("function_call") => {
                let (call, raw) = function_call_input(&item)?;
                pending_calls.push(call);
                stored.push(StoredResponseInputItem {
                    role: "assistant".to_string(),
                    content: String::new(),
                    item: Some(raw),
                });
                continue;
            }
            Some("function_call_output") => {
                flush_function_calls(
                    model_variant,
                    &mut messages,
                    &mut pending_calls,
                    previous_was_assistant_message,
                )?;
                previous_was_assistant_message = false;
                let (output, raw) = function_call_output_input(&item)?;
                let result = ToolResult {
                    content: item.output.as_ref(),
                    additional_tools: item.additional_tools.as_ref(),
                    tool_call_id: item.call_id.as_deref(),
                };
                reject_deferred_tool_results(model_variant, result)?;
                match tool_responses.as_mut() {
                    Some(responses) => {
                        if let Some(rendered) = responses.render(result, next_result)? {
                            messages.push(tool_result_message(model_variant, rendered));
                        }
                    }
                    None => messages.push(tool_result_message(model_variant, output.clone())),
                }
                stored.push(StoredResponseInputItem {
                    role: "tool".to_string(),
                    content: output,
                    item: Some(raw),
                });
                continue;
            }
            None | Some("message") => {}
            Some(other) => {
                return Err(ApiError::invalid_param(
                    "input",
                    format!("Unsupported response input item type `{other}`"),
                ));
            }
        }
        flush_function_calls(
            model_variant,
            &mut messages,
            &mut pending_calls,
            previous_was_assistant_message,
        )?;
        if is_tool_result_item(&item) {
            let content = inbound_content_value(item.content.as_ref())?;
            let result = ToolResult {
                content: content.as_ref(),
                additional_tools: item.additional_tools.as_ref(),
                tool_call_id: item.call_id.as_deref(),
            };
            reject_deferred_tool_results(model_variant, result)?;
            if let Some(responses) = tool_responses.as_mut() {
                previous_was_assistant_message = false;
                let display = flatten_content(item.content.clone())?.display_text;
                if let Some(rendered) = responses.render(result, next_result)? {
                    messages.push(tool_result_message(model_variant, rendered));
                }
                stored.push(StoredResponseInputItem {
                    role: "tool".to_string(),
                    content: display,
                    item: None,
                });
                continue;
            }
        }
        let item = normalize_input_item(model_variant, item)?;
        let role = item.role.clone();
        let parsed_role = parse_input_role(&role)?;
        if item.runtime_content.trim().is_empty() && item.media_inputs.is_empty() {
            return Err(ApiError::bad_request(
                "Response input item content cannot be empty",
            ));
        }
        if parsed_role != ResponseInboundRole::User
            && item
                .media_inputs
                .iter()
                .any(|input| input.kind == izwi_core::ChatMediaKind::Audio)
        {
            return Err(ApiError::bad_request("Only user messages can carry audio"));
        }
        media_inputs.extend(item.media_inputs.clone());
        previous_was_assistant_message = parsed_role == ResponseInboundRole::Assistant;

        let message = match parsed_role {
            ResponseInboundRole::System => {
                if !item.media_inputs.is_empty() {
                    return Err(ApiError::bad_request(
                        "System response input items cannot contain images, videos or audio",
                    ));
                }
                ChatMessage {
                    role: ChatRole::System,
                    content: item.runtime_content.clone(),
                }
            }
            ResponseInboundRole::User => ChatMessage {
                role: ChatRole::User,
                content: item.runtime_content.clone(),
            },
            ResponseInboundRole::Assistant => ChatMessage {
                role: ChatRole::Assistant,
                content: item.runtime_content.clone(),
            },
            ResponseInboundRole::Tool => {
                tool_result_message(model_variant, item.runtime_content.clone())
            }
        };
        messages.push(message);
        stored.push(StoredResponseInputItem {
            role,
            content: item.display_content,
            item: None,
        });
    }
    flush_function_calls(
        model_variant,
        &mut messages,
        &mut pending_calls,
        previous_was_assistant_message,
    )?;

    Ok((messages, stored, media_inputs))
}

/// Write `pending` calls as an assistant turn: onto the assistant message
/// they follow, or as a turn of their own.
fn flush_function_calls(
    variant: ModelVariant,
    messages: &mut Vec<ChatMessage>,
    pending: &mut Vec<Value>,
    after_assistant_message: bool,
) -> Result<(), ApiError> {
    if pending.is_empty() {
        return Ok(());
    }
    let calls = std::mem::take(pending);
    match messages.last_mut() {
        Some(last) if after_assistant_message => {
            append_assistant_tool_calls(variant, &mut last.content, &calls)
        }
        _ => {
            let mut content = String::new();
            append_assistant_tool_calls(variant, &mut content, &calls)?;
            messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content,
            });
            Ok(())
        }
    }
}

/// A `function_call` input item as a chat completions tool call, and as
/// the item to list back.
fn function_call_input(item: &super::dto::ResponseInputItem) -> Result<(Value, Value), ApiError> {
    let name = item
        .name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            ApiError::invalid_param("input", "A `function_call` input item needs a `name`")
        })?;
    let arguments = match &item.arguments {
        None | Some(Value::Null) => "{}".to_string(),
        Some(Value::String(arguments)) => arguments.clone(),
        Some(other) => other.to_string(),
    };
    let call_id = item.call_id.clone().unwrap_or_default();
    Ok((
        json!({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }),
        json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }),
    ))
}

/// Whether an input item is a tool result: a `function_call_output`, or a
/// message in the `tool` role.
fn is_tool_result_item(item: &super::dto::ResponseInputItem) -> bool {
    match item.item_type.as_deref() {
        Some("function_call_output") => true,
        None | Some("message") => item
            .role
            .as_deref()
            .is_some_and(|role| role.trim().eq_ignore_ascii_case("tool")),
        _ => false,
    }
}

/// A `function_call_output` input item's result text, and the item to
/// list back. Content parts contribute their text.
fn function_call_output_input(
    item: &super::dto::ResponseInputItem,
) -> Result<(String, Value), ApiError> {
    let output = match &item.output {
        Some(Value::String(output)) => output.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => {
            return Err(ApiError::invalid_param(
                "input",
                "A `function_call_output` input item needs an `output` string",
            ))
        }
    };
    Ok((
        output.clone(),
        json!({
            "type": "function_call_output",
            "call_id": item.call_id.clone().unwrap_or_default(),
            "output": output,
        }),
    ))
}

fn normalize_input_item(
    variant: ModelVariant,
    item: super::dto::ResponseInputItem,
) -> Result<NormalizedInputItem, ApiError> {
    let role = item
        .role
        .unwrap_or_else(|| "user".to_string())
        .trim()
        .to_ascii_lowercase();

    let flattened = flatten_content(item.content)?;
    let mut runtime_content = flattened.runtime_text.clone();
    if role == "assistant" {
        let tool_calls = item.tool_calls.as_deref().unwrap_or(&[]);
        append_assistant_tool_calls(variant, &mut runtime_content, tool_calls)?;
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
        media_inputs: flattened.media_inputs,
    })
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

fn assistant_output_item_with_id(id: String, text: String) -> ResponseOutputItem {
    ResponseOutputItem::Message {
        id,
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
        output: if record.output_text.is_none() && record.output_function_calls.is_empty() {
            Vec::new()
        } else {
            output_items(
                new_prefixed_id("msg_"),
                record.output_text,
                &record.output_function_calls,
            )
        },
        usage: ResponseUsage {
            input_tokens: record.input_tokens,
            output_tokens: record.output_tokens,
            total_tokens: record.input_tokens + record.output_tokens,
            input_tokens_details: ResponseInputTokensDetails {
                cached_tokens: record.input_cached_tokens,
            },
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

fn sse_json(event_name: &str, payload: &impl serde::Serialize) -> String {
    sse_event(
        event_name,
        &serde_json::to_string(payload).unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    use super::super::dto::ResponseInputItem;
    use super::*;
    use crate::api::openai::convert::{InboundContent, InboundContentPart};
    use crate::test_support::env_lock;
    use izwi_core::{backends::BackendPreference, RuntimeService, ServeRuntimeConfig};
    use serde_json::json;

    #[test]
    fn closed_response_stream_emits_explicit_terminal_failure() {
        let payload = response_stream_interruption_payload("resp_test", false)
            .expect("an interrupted response stream should fail");
        let json: serde_json::Value =
            serde_json::from_str(&payload).expect("valid failure payload");

        assert_eq!(
            json.get("type").and_then(|value| value.as_str()),
            Some("response.failed")
        );
        assert_eq!(
            json.get("response_id").and_then(|value| value.as_str()),
            Some("resp_test")
        );
        assert_eq!(
            json.pointer("/error/message")
                .and_then(|value| value.as_str()),
            Some(RESPONSE_STREAM_INTERRUPTED_ERROR)
        );
        assert!(response_stream_interruption_payload("resp_test", true).is_none());
    }

    #[test]
    fn stored_record_usage_reports_openai_shaped_cached_tokens() {
        let record = StoredResponseRecord {
            id: "resp_cached".to_string(),
            tenant_scope: String::new(),
            created_at: 0,
            status: "completed".to_string(),
            model: "test".to_string(),
            input_items: Vec::new(),
            output_text: Some("done".to_string()),
            output_function_calls: Vec::new(),
            input_tokens: 12,
            output_tokens: 3,
            input_cached_tokens: 7,
            error: None,
            metadata: None,
        };
        let response = record_to_response(record);
        assert_eq!(response.usage.input_tokens, 12);
        assert_eq!(response.usage.input_tokens_details.cached_tokens, 7);
    }

    #[test]
    fn builds_messages_from_text_and_instructions() {
        let (messages, stored, media_inputs) = build_input_messages(
            ModelVariant::Qwen359BGguf,
            Some("Be concise."),
            Some(ResponseInput::Text("Hello".to_string())),
            &[],
        )
        .expect("build input messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, "system");
        assert_eq!(stored[1].role, "user");
        assert!(media_inputs.is_empty());
    }

    /// No chat model hears audio: a user item's `input_audio` part is
    /// refused rather than dropped, and other roles cannot carry audio.
    #[test]
    fn audio_input_items_are_refused() {
        let input: ResponseInput = serde_json::from_value(json!([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is said? "},
                {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
            ]
        }]))
        .unwrap();
        let (_, _, media_inputs) =
            build_input_messages(ModelVariant::Qwen354BGguf, None, Some(input), &[]).unwrap();
        let error = validate_media_inputs_for_variant(ModelVariant::Qwen354BGguf, &media_inputs)
            .unwrap_err();
        assert!(error.contains("do not hear audio clips"), "{error}");

        let assistant: ResponseInput = serde_json::from_value(json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [
                {"type": "input_audio", "input_audio": {"data": "UklGRg=="}}
            ]}
        ]))
        .unwrap();
        let error = build_input_messages(ModelVariant::Qwen354BGguf, None, Some(assistant), &[])
            .unwrap_err();
        assert!(
            error.message.contains("Only user messages"),
            "{}",
            error.message
        );
    }

    #[test]
    fn normalizes_input_item_default_role() {
        let item = ResponseInputItem {
            role: None,
            content: Some(InboundContent::Text("hello".to_string())),
            ..Default::default()
        };
        let normalized = normalize_input_item(ModelVariant::Qwen359BGguf, item).expect("normalize");
        assert_eq!(normalized.role, "user");
        assert_eq!(normalized.runtime_content, "hello");
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
            ..Default::default()
        };

        let normalized =
            normalize_input_item(ModelVariant::Qwen359BGguf, item).expect("normalize tool call");
        assert_eq!(normalized.role, "assistant");
        assert!(normalized.media_inputs.is_empty());
        assert!(normalized.runtime_content.contains("<tool_call>"));
        assert!(normalized
            .runtime_content
            .contains("<function=get_weather>"));
        assert!(normalized.runtime_content.contains("<parameter=city>"));
    }

    /// Responses renders tool turns in the model family's format, as chat
    /// completions does.
    #[test]
    fn lfm25_tool_turns_use_the_lfm_wire_format() {
        let item = |role: &str, content: Option<&str>, tool_calls| ResponseInputItem {
            role: Some(role.to_string()),
            content: content.map(|text| InboundContent::Text(text.to_string())),
            tool_calls,
            ..Default::default()
        };
        let (messages, _, _) = build_input_messages(
            ModelVariant::Lfm2512BInstructGguf,
            None,
            Some(ResponseInput::Many(vec![
                item("user", Some("Weather in Paris?"), None),
                item(
                    "assistant",
                    None,
                    Some(vec![json!({
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    })]),
                ),
                item("tool", Some("{\"temperature_c\": 21}"), None),
            ])),
            &[],
        )
        .expect("tool turns normalize");
        assert_eq!(
            messages[1].content,
            "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"
        );
        assert_eq!(messages[2].role, ChatRole::Tool);
        assert_eq!(messages[2].content, "{\"temperature_c\": 21}");
    }

    #[test]
    fn build_input_messages_collects_multimodal_input() {
        let (messages, _stored, media_inputs) = build_input_messages(
            ModelVariant::Qwen314BGguf,
            None,
            Some(ResponseInput::Many(vec![ResponseInputItem {
                role: Some("user".to_string()),
                content: Some(InboundContent::Parts(vec![InboundContentPart {
                    kind: Some("image_url".to_string()),
                    text: None,
                    input_text: None,
                    image_url: Some(json!({"url":"https://example.com/img.png"})),
                    input_image: None,
                    image: None,
                    video: None,
                    video_url: None,
                    input_video: None,
                    ..InboundContentPart::default()
                }])),
                ..Default::default()
            }])),
            &[],
        )
        .expect("multimodal input should normalize");

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
        assert_eq!(
            serde_json::to_value(item).unwrap(),
            json!({
                "type": "message",
                "id": "msg_custom",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            })
        );
    }

    #[tokio::test]
    async fn store_false_does_not_retain_response_record() {
        let _guard = env_lock();
        let (state, _temp_dir) = test_app_state("responses_store_false");

        persist_response(&state, response_record("resp-skip"), Some(false)).await;
        assert!(
            state.response_store.read().await.is_empty(),
            "store:false should skip in-memory retention"
        );

        persist_response(&state, response_record("resp-store"), None).await;
        assert!(
            state.response_store.read().await.contains_key("resp-store"),
            "default store behavior should retain completed responses"
        );
    }

    #[tokio::test]
    async fn stored_responses_are_visible_only_to_the_owning_tenant() {
        let _guard = env_lock();
        let (state, _temp_dir) = test_app_state("responses_tenant_scope");
        let context = |tenant: &str| {
            let mut principal = izwi_hooks::Principal::local_anonymous();
            principal.id = format!("user-{tenant}");
            principal.tenant_id = Some(tenant.to_string());
            Extension(RequestContext::new("test".to_string(), principal))
        };
        let (owner, other) = (context("tenant-a"), context("tenant-b"));
        let mut record = response_record("resp-a");
        record.status = "in_progress".to_string();
        record.tenant_scope = owner.0.tenant_scope();
        state.store_response_record(record).await;
        let id = || Path("resp-a".to_string());
        let status = |err: ApiError| err.status;

        let get = get_response(State(state.clone()), other.clone(), id()).await;
        assert_eq!(get.err().map(status), Some(StatusCode::NOT_FOUND));
        let items = list_response_input_items(State(state.clone()), other.clone(), id()).await;
        assert_eq!(items.err().map(status), Some(StatusCode::NOT_FOUND));
        let cancel = cancel_response(State(state.clone()), other.clone(), id()).await;
        assert_eq!(cancel.err().map(status), Some(StatusCode::NOT_FOUND));
        let delete = delete_response(State(state.clone()), other, id()).await;
        assert_eq!(delete.err().map(status), Some(StatusCode::NOT_FOUND));
        assert_eq!(
            state.response_store.read().await["resp-a"].status,
            "in_progress",
            "another tenant must not cancel or delete the record"
        );

        assert!(get_response(State(state.clone()), owner.clone(), id())
            .await
            .is_ok());
        assert!(delete_response(State(state.clone()), owner.clone(), id())
            .await
            .is_ok());
        assert!(state.response_store.read().await.is_empty());
    }

    const QWEN_CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    fn generation(text: &str) -> ChatGeneration {
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

    /// The `data:` payloads of the stream `events` render to.
    async fn stream_payloads(
        state: &AppState,
        tool_format: Option<ToolCallFormat>,
        events: Vec<ChatStreamEvent>,
    ) -> Vec<serde_json::Value> {
        use futures::StreamExt;
        let (tx, rx) = mpsc::channel(events.len().max(1));
        for event in events {
            tx.send(event).await.unwrap();
        }
        drop(tx);
        let request = ResponseStreamRequest {
            response_id: "resp_test".to_string(),
            created_at: 0,
            model: "test".to_string(),
            metadata: None,
            store: None,
            tenant_scope: String::new(),
            input_items: Vec::new(),
            tool_format,
            opened_in_prompt: false,
        };
        let chunks: Vec<String> = response_event_stream(state.clone(), request, rx)
            .map(|chunk| chunk.unwrap())
            .collect()
            .await;
        chunks
            .iter()
            .filter_map(|chunk| chunk.lines().find_map(|line| line.strip_prefix("data: ")))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    fn event_types(payloads: &[serde_json::Value]) -> Vec<&str> {
        payloads
            .iter()
            .map(|payload| payload["type"].as_str().unwrap())
            .collect()
    }

    /// A tool call in the answer is a `function_call` output item, streamed
    /// with its own events, never text.
    #[tokio::test]
    async fn streamed_tool_calls_become_function_call_items() {
        let _guard = env_lock();
        let (state, _temp_dir) = test_app_state("responses_function_call_stream");
        let answer = format!("Let me check. {QWEN_CALL}");
        let payloads = stream_payloads(
            &state,
            Some(ToolCallFormat::QwenXml),
            vec![
                ChatStreamEvent::Started,
                ChatStreamEvent::Delta {
                    text: "Let me check. ".to_string(),
                    logprobs: Vec::new(),
                },
                ChatStreamEvent::Delta {
                    text: QWEN_CALL.to_string(),
                    logprobs: Vec::new(),
                },
                ChatStreamEvent::Completed(Box::new(generation(&answer))),
            ],
        )
        .await;
        assert_eq!(
            event_types(&payloads),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for payload in &payloads {
            assert!(
                !payload.to_string().contains("<tool_call>"),
                "markup leaked: {payload}"
            );
        }
        let call_added = &payloads[5];
        assert_eq!(call_added["output_index"], 1);
        assert_eq!(call_added["item"]["type"], "function_call");
        assert_eq!(call_added["item"]["name"], "get_weather");
        let arguments = payloads[10]["arguments"].as_str().unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(arguments).unwrap(),
            json!({"city": "Paris"})
        );

        let output = &payloads.last().unwrap()["response"]["output"];
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "Let me check.");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["status"], "completed");
        assert_eq!(output[1]["arguments"], arguments);
        let call_id = output[1]["call_id"].as_str().unwrap();
        assert!(call_id.starts_with("call_"), "{call_id}");

        // The stored response lists the same call.
        let stored = record_to_response(state.response_store.read().await["resp_test"].clone());
        assert_eq!(
            serde_json::to_value(&stored.output).unwrap()[1]["call_id"],
            call_id
        );
    }

    /// Without tools the stream is the plain message sequence.
    #[tokio::test]
    async fn streamed_text_without_tools_is_one_message() {
        let _guard = env_lock();
        let (state, _temp_dir) = test_app_state("responses_text_stream");
        let payloads = stream_payloads(
            &state,
            None,
            vec![
                ChatStreamEvent::Started,
                ChatStreamEvent::Delta {
                    text: "Hello".to_string(),
                    logprobs: Vec::new(),
                },
                ChatStreamEvent::Completed(Box::new(generation("Hello"))),
            ],
        )
        .await;
        assert_eq!(
            event_types(&payloads),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(payloads[4]["delta"], "Hello");
        assert_eq!(payloads[4]["output_index"], 0);
        let output = &payloads.last().unwrap()["response"]["output"];
        assert_eq!(output.as_array().unwrap().len(), 1);
        assert_eq!(output[0]["content"][0]["text"], "Hello");
    }

    /// A `<think>` block the prompt opened reads `<think>…</think>` in the
    /// deltas and the stored text alike.
    #[test]
    fn streamed_text_gets_back_the_opener_the_prompt_held() {
        let mut output = ResponseOutputStream::new("resp_x".to_string(), None, true);
        let streamed: String = ["r</think>", "A"]
            .into_iter()
            .flat_map(|delta| output.push(delta.to_string()))
            .filter_map(|event| {
                let data = event.lines().find_map(|line| line.strip_prefix("data: "))?;
                let payload: serde_json::Value = serde_json::from_str(data).ok()?;
                (payload["type"] == "response.output_text.delta")
                    .then(|| payload["delta"].as_str().unwrap().to_string())
            })
            .collect();
        assert_eq!(streamed, "<think>r</think>A");
        let finished = output.finish("r</think>A");
        assert_eq!(finished.text.as_deref(), Some("<think>r</think>A"));
    }

    #[test]
    fn whole_answers_split_into_text_and_function_calls() {
        let (text, calls) = answer_output(Some(ToolCallFormat::QwenXml), QWEN_CALL.to_string());
        assert_eq!(text, None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let output = output_items("msg_x".to_string(), text, &calls);
        assert_eq!(output.len(), 1, "a call-only answer has no message item");
        let (text, calls) = answer_output(None, QWEN_CALL.to_string());
        assert_eq!(text.as_deref(), Some(QWEN_CALL));
        assert!(calls.is_empty());
    }

    /// `function_call` input items become the assistant turn that made them
    /// (after an assistant message right before them), and
    /// `function_call_output` items become tool results.
    #[test]
    fn function_call_input_items_render_the_tool_turns() {
        let input = serde_json::from_value::<ResponseInput>(json!([
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"role": "assistant", "content": "Checking both."},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather",
             "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "get_weather",
             "arguments": "{\"city\":\"Rome\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "21C"},
            {"type": "function_call_output", "call_id": "call_2", "output": "25C"},
        ]))
        .unwrap();
        let (messages, stored, _) =
            build_input_messages(ModelVariant::Qwen359BGguf, None, Some(input), &[]).unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert!(messages[1]
            .content
            .starts_with("Checking both.\n\n<tool_call>"));
        assert_eq!(
            messages[1]
                .content
                .matches("<function=get_weather>")
                .count(),
            2
        );
        assert!(messages[1].content.contains("Rome"));
        assert_eq!(messages[2].role, ChatRole::User);
        assert!(messages[2].content.contains("<tool_response>\n21C"));
        assert!(messages[3].content.contains("25C"));
        assert_eq!(stored.len(), 6);
        assert_eq!(stored[2].item.as_ref().unwrap()["type"], "function_call");
        assert_eq!(stored[4].item.as_ref().unwrap()["call_id"], "call_1");

        // Calls with no message before them are a turn of their own.
        let input = serde_json::from_value::<ResponseInput>(json!([
            {"role": "user", "content": "Weather?"},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "21C"},
        ]))
        .unwrap();
        let (messages, _, _) =
            build_input_messages(ModelVariant::Qwen359BGguf, None, Some(input), &[]).unwrap();
        assert_eq!(messages.len(), 3);
        assert!(messages[1].content.starts_with("<tool_call>"));

        let unknown = serde_json::from_value::<ResponseInput>(json!([
            {"type": "web_search_call", "id": "ws_1"}
        ]))
        .unwrap();
        assert!(
            build_input_messages(ModelVariant::Qwen359BGguf, None, Some(unknown), &[]).is_err()
        );
    }

    #[test]
    fn lfm3_function_call_outputs_name_deferred_tools() {
        let tools = [
            json!({"type": "function", "function": {"name": "search_tools"}}),
            json!({"type": "function", "defer_loading": true,
                   "function": {"name": "get_weather", "parameters": {}}}),
        ];
        let input = serde_json::from_value::<ResponseInput>(json!([
            {"role": "user", "content": "Weather?"},
            {"type": "function_call", "call_id": "call_1", "name": "search_tools", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "Found 1 tool."},
            {"type": "function_call_output", "call_id": "call_1",
             "output": [{"type": "tool_reference", "name": "get_weather"}]},
        ]))
        .unwrap();
        let (messages, stored, _) =
            build_input_messages(ModelVariant::Lfm317BGguf, None, Some(input.clone()), &tools)
                .unwrap();
        // The two outputs answering one call are one tool turn.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].role, ChatRole::Tool);
        assert_eq!(
            messages[2].content,
            "<tool_response>\nFound 1 tool.\n<additional_tools>\n\
             {\"name\": \"get_weather\", \"parameters\": {}}\n</additional_tools>\n</tool_response>"
        );
        assert_eq!(stored.len(), 4);

        // A model whose template has no deferred tools refuses the reference.
        assert!(
            build_input_messages(ModelVariant::Qwen359BGguf, None, Some(input), &tools).is_err()
        );
    }

    /// Responses' flat function tools and named tool choice read like
    /// chat completions' nested ones.
    #[test]
    fn flat_function_tools_are_accepted() {
        let flat = json!({"type": "function", "name": "get_weather",
                          "parameters": {"type": "object"}});
        assert_eq!(
            chat_completions_function(&flat),
            json!({"type": "function", "function": {
                "name": "get_weather", "parameters": {"type": "object"}}})
        );
        let nested = json!({"type": "function", "function": {"name": "get_weather"}});
        assert_eq!(chat_completions_function(&nested), nested);
        let policy = resolve_tool_policy(
            ModelVariant::Qwen359BGguf,
            Some(&[chat_completions_function(&flat)]),
            None,
        )
        .unwrap();
        assert_eq!(policy.format, Some(ToolCallFormat::QwenXml));
        let named = chat_completions_function(&json!({"type": "function", "name": "get_weather"}));
        assert_eq!(
            resolve_tool_policy(ModelVariant::Qwen359BGguf, Some(&[nested]), Some(&named))
                .unwrap_err()
                .status,
            StatusCode::BAD_REQUEST
        );
    }

    fn response_record(id: &str) -> StoredResponseRecord {
        StoredResponseRecord {
            id: id.to_string(),
            tenant_scope: String::new(),
            created_at: now_unix_secs(),
            status: "completed".to_string(),
            model: "test".to_string(),
            input_items: Vec::new(),
            output_text: Some("ok".to_string()),
            output_function_calls: Vec::new(),
            input_tokens: 1,
            output_tokens: 1,
            input_cached_tokens: 0,
            error: None,
            metadata: None,
        }
    }

    fn test_app_state(name: &str) -> (AppState, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().expect("temp dir should create");
        let models_dir = temp_dir.path().join("models");
        let media_dir = temp_dir.path().join("media");
        std::fs::create_dir_all(&models_dir).expect("models dir should be created");

        std::env::set_var(
            "IZWI_DB_PATH",
            temp_dir.path().join(format!("{name}.sqlite3")),
        );
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let serve_config = ServeRuntimeConfig {
            backend: BackendPreference::Cpu,
            models_dir,
            ..ServeRuntimeConfig::default()
        };
        let runtime =
            with_suppressed_panic_hook(|| RuntimeService::new(serve_config.engine_config()))
                .expect("runtime should initialize");
        let state = AppState::new(runtime, &serve_config).expect("state should initialize");

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");

        (state, temp_dir)
    }

    fn with_suppressed_panic_hook<T>(f: impl FnOnce() -> T) -> T {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = f();
        std::panic::set_hook(default_hook);
        result
    }
}
