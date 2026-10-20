use super::*;
use crate::api::openai::chat::stream::openai_chat_stream_interruption_payload;
use crate::api::openai::convert::{InboundContent, InboundContentPart};
use crate::app::chat::ChatStreamEvent;
use crate::error::ErrorScope;
use axum::http::StatusCode;
use serde_json::json;

#[test]
fn closed_chat_stream_emits_explicit_error_without_terminal_event() {
    let payload = openai_chat_stream_interruption_payload(&ErrorScope::current(), false)
        .expect("an interrupted stream should emit an error");
    let json: serde_json::Value = serde_json::from_str(&payload).expect("valid error payload");

    // A server fault: outside a loopback server's scope the client sees
    // the generic message, not the internal one.
    assert!(json
        .pointer("/error/message")
        .and_then(|value| value.as_str())
        .is_some_and(|message| message.starts_with(crate::error::SERVER_FAULT_MESSAGE)));
    assert_eq!(
        json.pointer("/error/type").and_then(|value| value.as_str()),
        Some("server_error")
    );
    assert!(openai_chat_stream_interruption_payload(&ErrorScope::current(), true).is_none());
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
        parse_reasoning: false,
        reasoning_opened_in_prompt: false,
        tool_calls,
    }
}

/// Unparsed, a stream whose prompt opened `<think>` gets the opener back in
/// its first content delta, so `content` reads `<think>…</think>…`.
#[tokio::test]
async fn unparsed_stream_writes_back_the_opener_the_prompt_held() {
    let text = "r</think>\n\nA";
    let format = AnswerFormat {
        reasoning_opened_in_prompt: true,
        ..tool_format(None)
    };
    let body = json!({"model": "m", "messages": [], "stream": true});
    for size in [1, 3, text.len()] {
        let payloads = sse_payloads(body.clone(), format, deltas(text, size)).await;
        let (content, _, _) = read_stream(&payloads);
        assert_eq!(content, "<think>r</think>\n\nA", "split {size}");
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
                serde_json::from_str::<serde_json::Value>(calls[0]["arguments"].as_str().unwrap())
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
    let entry = izwi_model_catalog::TokenLogprob {
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
        to_core_messages_with_media(ModelVariant::Lfm2512BInstructGguf, messages.clone()).unwrap();
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
        tools: None,
        tool_choice: None,
        enable_thinking: None,
        reasoning_effort: None,
        preserve_thinking: None,
        chat_template_kwargs: None,
        reasoning_format: None,
        truncation: None,
        skip_special_tokens: None,
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
        tools: None,
        tool_choice: None,
        enable_thinking: None,
        reasoning_effort: None,
        preserve_thinking: None,
        chat_template_kwargs: None,
        reasoning_format: None,
        truncation: None,
        skip_special_tokens: None,
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
        tools: None,
        tool_choice: None,
        enable_thinking: None,
        reasoning_effort: None,
        preserve_thinking: None,
        chat_template_kwargs: None,
        reasoning_format: None,
        truncation: None,
        skip_special_tokens: None,
    };

    validate_chat_request_compatibility(&req, OpenAiCompatibilityProfile::Relaxed)
        .expect("relaxed mode should allow passthrough");
}
