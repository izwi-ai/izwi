//! Golden SSE transcripts of the chat completions stream, checked against a
//! strict model of the OpenAI `ChatCompletionChunk` schema (the fields and
//! types of openai-python's `ChatCompletionChunk`, plus izwi's documented
//! extensions) and against the stream rules OpenAI clients rely on.
//!
//! Each case renders a scripted event sequence through the same renderer
//! both the local and the gateway paths use. The normalized transcript
//! (ids and timestamps replaced) must match `testdata/sse/<case>.sse`;
//! `IZWI_UPDATE_SSE_GOLDENS=1` rewrites the files.

use serde_json::{json, Map, Value};

use super::tests::{deltas, generation, sse_payloads, tool_format};
use super::*;
use crate::app::chat::ChatStreamEvent;

/// Keys izwi adds to the OpenAI chunk object, and to `delta`.
const CHUNK_EXTENSIONS: &[&str] = &["izwi_timing", "izwi_generation_time_ms"];
const DELTA_EXTENSIONS: &[&str] = &["reasoning_content"];

type Check = Result<(), String>;

fn object<'a>(value: &'a Value, at: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{at}: expected an object, got {value}"))
}

/// Every key of `map` is one of `allowed`.
fn known_keys(map: &Map<String, Value>, allowed: &[&str], at: &str) -> Check {
    match map.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => Err(format!("{at}: unknown field `{key}`")),
        None => Ok(()),
    }
}

fn string(value: &Value, at: &str) -> Check {
    value
        .is_string()
        .then_some(())
        .ok_or_else(|| format!("{at}: expected a string, got {value}"))
}

fn optional_string(value: Option<&Value>, at: &str) -> Check {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(value) => string(value, at),
    }
}

fn integer(value: &Value, at: &str) -> Check {
    value
        .as_u64()
        .map(|_| ())
        .ok_or_else(|| format!("{at}: expected a non-negative integer, got {value}"))
}

fn byte_list(value: Option<&Value>, at: &str) -> Check {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(bytes))
            if bytes
                .iter()
                .all(|byte| byte.as_u64().is_some_and(|b| b <= 255)) =>
        {
            Ok(())
        }
        Some(value) => Err(format!("{at}: expected a byte list, got {value}")),
    }
}

/// `ChatCompletionTokenLogprob` and its `TopLogprob`s.
fn token_logprob(value: &Value, at: &str, top: bool) -> Check {
    let entry = object(value, at)?;
    let allowed: &[&str] = if top {
        &["token", "logprob", "bytes"]
    } else {
        &["token", "logprob", "bytes", "top_logprobs"]
    };
    known_keys(entry, allowed, at)?;
    string(&entry["token"], &format!("{at}.token"))?;
    if !entry["logprob"].is_number() {
        return Err(format!("{at}.logprob: expected a number"));
    }
    byte_list(entry.get("bytes"), &format!("{at}.bytes"))?;
    if !top {
        let tops = entry["top_logprobs"]
            .as_array()
            .ok_or_else(|| format!("{at}.top_logprobs: expected a list"))?;
        for (index, item) in tops.iter().enumerate() {
            token_logprob(item, &format!("{at}.top_logprobs[{index}]"), true)?;
        }
    }
    Ok(())
}

/// `CompletionUsage`.
fn usage(value: &Value, at: &str) -> Check {
    let usage = object(value, at)?;
    known_keys(
        usage,
        &[
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        ],
        at,
    )?;
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        integer(&usage[key], &format!("{at}.{key}"))?;
    }
    let total =
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap();
    if usage["total_tokens"].as_u64() != Some(total) {
        return Err(format!("{at}.total_tokens: not prompt + completion"));
    }
    if let Some(details) = usage.get("prompt_tokens_details") {
        let details = object(details, &format!("{at}.prompt_tokens_details"))?;
        known_keys(details, &["cached_tokens", "audio_tokens"], at)?;
        for value in details.values() {
            integer(value, &format!("{at}.prompt_tokens_details"))?;
        }
    }
    Ok(())
}

/// `ChoiceDelta`, including `ChoiceDeltaToolCall`s.
fn delta(value: &Value, at: &str) -> Check {
    let delta = object(value, at)?;
    let mut allowed = vec!["role", "content", "refusal", "tool_calls", "function_call"];
    allowed.extend(DELTA_EXTENSIONS);
    known_keys(delta, &allowed, at)?;
    match delta.get("role") {
        None | Some(Value::Null) => {}
        Some(role) if role == "assistant" => {}
        Some(role) => return Err(format!("{at}.role: unexpected {role}")),
    }
    optional_string(delta.get("content"), &format!("{at}.content"))?;
    optional_string(
        delta.get("reasoning_content"),
        &format!("{at}.reasoning_content"),
    )?;
    let Some(calls) = delta.get("tool_calls").filter(|calls| !calls.is_null()) else {
        return Ok(());
    };
    let calls = calls
        .as_array()
        .ok_or_else(|| format!("{at}.tool_calls: expected a list"))?;
    for (index, call) in calls.iter().enumerate() {
        let at = format!("{at}.tool_calls[{index}]");
        let call = object(call, &at)?;
        known_keys(call, &["index", "id", "type", "function"], &at)?;
        integer(&call["index"], &format!("{at}.index"))?;
        optional_string(call.get("id"), &format!("{at}.id"))?;
        match call.get("type") {
            None | Some(Value::Null) => {}
            Some(kind) if kind == "function" => {}
            Some(kind) => return Err(format!("{at}.type: unexpected {kind}")),
        }
        if let Some(function) = call.get("function").filter(|function| !function.is_null()) {
            let function = object(function, &format!("{at}.function"))?;
            known_keys(function, &["name", "arguments"], &at)?;
            optional_string(function.get("name"), &format!("{at}.function.name"))?;
            optional_string(
                function.get("arguments"),
                &format!("{at}.function.arguments"),
            )?;
        }
    }
    Ok(())
}

/// `ChatCompletionChunk`.
fn chunk(value: &Value) -> Check {
    let chunk = object(value, "chunk")?;
    let mut allowed = vec![
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
        "service_tier",
    ];
    allowed.extend(CHUNK_EXTENSIONS);
    known_keys(chunk, &allowed, "chunk")?;
    string(&chunk["id"], "chunk.id")?;
    if chunk["object"] != "chat.completion.chunk" {
        return Err(format!("chunk.object: {}", chunk["object"]));
    }
    integer(&chunk["created"], "chunk.created")?;
    string(&chunk["model"], "chunk.model")?;
    optional_string(chunk.get("system_fingerprint"), "chunk.system_fingerprint")?;
    if let Some(value) = chunk.get("usage").filter(|usage| !usage.is_null()) {
        usage(value, "chunk.usage")?;
    }
    let choices = chunk["choices"]
        .as_array()
        .ok_or("chunk.choices: expected a list")?;
    for (index, choice) in choices.iter().enumerate() {
        let at = format!("chunk.choices[{index}]");
        let choice = object(choice, &at)?;
        known_keys(
            choice,
            &["index", "delta", "finish_reason", "logprobs"],
            &at,
        )?;
        integer(&choice["index"], &format!("{at}.index"))?;
        delta(&choice["delta"], &format!("{at}.delta"))?;
        match choice.get("finish_reason") {
            Some(Value::Null) => {}
            Some(Value::String(reason))
                if [
                    "stop",
                    "length",
                    "tool_calls",
                    "content_filter",
                    "function_call",
                ]
                .contains(&reason.as_str()) => {}
            other => return Err(format!("{at}.finish_reason: unexpected {other:?}")),
        }
        if let Some(logprobs) = choice
            .get("logprobs")
            .filter(|logprobs| !logprobs.is_null())
        {
            let logprobs = object(logprobs, &format!("{at}.logprobs"))?;
            known_keys(logprobs, &["content", "refusal"], &at)?;
            for (entry_index, entry) in logprobs["content"]
                .as_array()
                .ok_or_else(|| format!("{at}.logprobs.content: expected a list"))?
                .iter()
                .enumerate()
            {
                token_logprob(
                    entry,
                    &format!("{at}.logprobs.content[{entry_index}]"),
                    false,
                )?;
            }
        }
    }
    Ok(())
}

/// The in-band error event openai-python raises as `APIError`.
fn error_event(value: &Value) -> Check {
    let event = object(value, "error event")?;
    known_keys(event, &["error"], "error event")?;
    let error = object(&event["error"], "error")?;
    known_keys(error, &["message", "type", "param", "code"], "error")?;
    string(&error["message"], "error.message")?;
    string(&error["type"], "error.type")?;
    optional_string(error.get("param"), "error.param")?;
    optional_string(error.get("code"), "error.code")
}

/// The stream rules on top of the per-chunk schema.
fn validate_stream(payloads: &[String], include_usage: bool) -> Check {
    let (last, events) = payloads.split_last().ok_or("empty stream")?;
    if last != "[DONE]" {
        return Err(format!("the stream must end with [DONE], got {last}"));
    }
    let events: Vec<Value> = events
        .iter()
        .map(|payload| serde_json::from_str(payload).map_err(|err| format!("{payload}: {err}")))
        .collect::<Result<_, _>>()?;
    let (errors, chunks): (Vec<&Value>, Vec<&Value>) = events
        .iter()
        .partition(|event| event.get("error").is_some());
    for event in &errors {
        error_event(event)?;
    }
    if errors.len() > 1 || (errors.len() == 1 && events.last() != errors.first().copied()) {
        return Err("an error event must be the last event".into());
    }
    let first = chunks.first().ok_or("no chunks")?;
    for value in &chunks {
        chunk(value)?;
        for key in ["id", "created", "model"] {
            if value[key] != first[key] {
                return Err(format!("chunk `{key}` changed within the stream"));
            }
        }
    }
    if !first["id"].as_str().unwrap().starts_with("chatcmpl-") {
        return Err(format!("id {} is not a chatcmpl- id", first["id"]));
    }
    if first["choices"][0]["delta"]["role"] != "assistant" {
        return Err("the first chunk must carry the assistant role".into());
    }

    let finishes: Vec<&Value> = chunks
        .iter()
        .flat_map(|chunk| chunk["choices"].as_array().unwrap())
        .map(|choice| &choice["finish_reason"])
        .filter(|reason| !reason.is_null())
        .collect();
    if errors.is_empty() && finishes.len() != 1 {
        return Err(format!("expected one finish_reason, got {finishes:?}"));
    }
    if !errors.is_empty() && !finishes.is_empty() {
        return Err("a failed stream has no finish_reason".into());
    }

    for (index, value) in chunks.iter().enumerate() {
        let is_last = index == chunks.len() - 1;
        match (include_usage, value.get("usage")) {
            (false, None) => {}
            (false, Some(_)) => return Err("usage without stream_options.include_usage".into()),
            (true, Some(Value::Null)) if !(is_last && errors.is_empty()) => {}
            (true, Some(Value::Object(_))) if is_last && errors.is_empty() => {
                if value["choices"] != json!([]) {
                    return Err("the usage chunk must have no choices".into());
                }
            }
            (true, other) => return Err(format!("chunk {index}: unexpected usage {other:?}")),
        }
    }

    // Tool calls: dense indexes, the first fragment names the call, and
    // the arguments concatenate to a JSON object.
    let mut calls: Vec<(String, String)> = Vec::new();
    for value in &chunks {
        for choice in value["choices"].as_array().unwrap() {
            for call in choice["delta"]["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
            {
                let index = call["index"].as_u64().unwrap() as usize;
                let arguments = call["function"]["arguments"].as_str().unwrap_or_default();
                if index == calls.len() {
                    let starts = call["id"].is_string()
                        && call["type"] == "function"
                        && call["function"]["name"].is_string();
                    if !starts {
                        return Err(format!(
                            "tool call {index} starts without id, type and name"
                        ));
                    }
                    calls.push((
                        call["function"]["name"].as_str().unwrap().into(),
                        arguments.into(),
                    ));
                } else if index + 1 == calls.len() {
                    if call.get("id").is_some() || call["function"].get("name").is_some() {
                        return Err(format!("tool call {index} renamed mid-stream"));
                    }
                    calls[index].1.push_str(arguments);
                } else {
                    return Err(format!("tool call index {index} out of order"));
                }
            }
        }
    }
    for (name, arguments) in &calls {
        if !serde_json::from_str::<Value>(arguments).is_ok_and(|value| value.is_object()) {
            return Err(format!(
                "tool call {name}: arguments {arguments:?} are not a JSON object"
            ));
        }
    }
    let tool_finish = finishes
        .first()
        .is_some_and(|reason| *reason == "tool_calls");
    if tool_finish == calls.is_empty() {
        return Err("finish_reason is tool_calls exactly when the stream made calls".into());
    }
    Ok(())
}

/// The transcript with its random ids and timestamps replaced.
fn normalized(payloads: &[String]) -> String {
    let mut out = String::new();
    for payload in payloads {
        let line = match serde_json::from_str::<Value>(payload) {
            Ok(mut value) if value.get("id").is_some() => {
                value["id"] = json!("chatcmpl-ID");
                value["created"] = json!(0);
                let calls = value["choices"]
                    .as_array_mut()
                    .into_iter()
                    .flatten()
                    .filter_map(|choice| choice.pointer_mut("/delta/tool_calls"))
                    .filter_map(Value::as_array_mut)
                    .flatten();
                for call in calls {
                    if let Some(id) = call.get_mut("id") {
                        *id = json!("call_ID");
                    }
                }
                value.to_string()
            }
            _ => payload.clone(),
        };
        out.push_str("data: ");
        out.push_str(&line);
        out.push_str("\n\n");
    }
    out
}

/// Render `events` for `body`, check the stream, and compare it with its
/// golden transcript.
async fn golden(case: &str, body: Value, format: AnswerFormat, events: Vec<ChatStreamEvent>) {
    let include_usage = body["stream_options"]["include_usage"] == json!(true);
    let payloads = sse_payloads(body, format, events).await;
    if let Err(problem) = validate_stream(&payloads, include_usage) {
        panic!("{case}: {problem}\n{}", payloads.join("\n"));
    }
    let transcript = normalized(&payloads);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/api/openai/chat/testdata/sse")
        .join(format!("{case}.sse"));
    if std::env::var_os("IZWI_UPDATE_SSE_GOLDENS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &transcript).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{}: {err}; set IZWI_UPDATE_SSE_GOLDENS=1", path.display()));
    assert_eq!(
        transcript, expected,
        "{case}: transcript differs from its golden"
    );
}

fn stream_body(extra: Value) -> Value {
    let mut body = json!({"model": "fixture", "messages": [], "stream": true});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

fn logprob(token: &str, alternatives: &[(&str, f32)]) -> izwi_model_catalog::TokenLogprob {
    izwi_model_catalog::TokenLogprob {
        token: token.to_string(),
        logprob: alternatives.first().map_or(-0.5, |(_, logprob)| *logprob),
        bytes: token.as_bytes().to_vec(),
        top_logprobs: alternatives
            .iter()
            .map(|(token, logprob)| izwi_model_catalog::TopTokenLogprob {
                token: token.to_string(),
                logprob: *logprob,
                bytes: token.as_bytes().to_vec(),
            })
            .collect(),
    }
}

#[tokio::test]
async fn plain_text_stream() {
    golden(
        "plain_text",
        stream_body(json!({})),
        tool_format(None),
        deltas("Hello there, world.", 6),
    )
    .await;
}

#[tokio::test]
async fn tool_call_streams() {
    golden(
        "tool_call_qwen",
        stream_body(json!({"tools": [{"type": "function", "function": {"name": "get_weather"}}]})),
        tool_format(Some(ToolCallFormat::QwenXml)),
        deltas(
            "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
            7,
        ),
    )
    .await;
    golden(
        "tool_call_lfm",
        stream_body(json!({"tools": [{"type": "function", "function": {"name": "now"}}]})),
        tool_format(Some(ToolCallFormat::LfmPythonic)),
        deltas(
            "<|tool_call_start|>[get_weather(city='Paris'), now()]<|tool_call_end|>",
            5,
        ),
    )
    .await;
}

#[tokio::test]
async fn logprobs_stream() {
    let events = vec![
        ChatStreamEvent::Started,
        ChatStreamEvent::Delta {
            text: "Hi".into(),
            logprobs: vec![logprob("Hi", &[("Hi", -0.25), ("Hey", -1.5)])],
        },
        ChatStreamEvent::Delta {
            text: "!".into(),
            logprobs: vec![logprob("!", &[("!", -0.5), (".", -2.0)])],
        },
        // A stop string's token: its logprob, no text.
        ChatStreamEvent::Delta {
            text: String::new(),
            logprobs: vec![logprob("END", &[("END", -0.75), (" ", -3.0)])],
        },
        ChatStreamEvent::Completed(Box::new(generation("Hi!"))),
    ];
    golden(
        "logprobs",
        stream_body(json!({"logprobs": true, "top_logprobs": 2})),
        tool_format(None),
        events,
    )
    .await;
}

#[tokio::test]
async fn usage_stream() {
    golden(
        "usage",
        stream_body(json!({"stream_options": {"include_usage": true}})),
        tool_format(None),
        deltas("Counted.", 4),
    )
    .await;
}

#[tokio::test]
async fn error_stream() {
    let events = vec![
        ChatStreamEvent::Started,
        ChatStreamEvent::Delta {
            text: "Partial".into(),
            logprobs: Vec::new(),
        },
        ChatStreamEvent::Failed(ApiError::service_unavailable(
            "Chat stream consumer is too slow; generation was cancelled",
        )),
    ];
    golden(
        "error",
        stream_body(json!({"stream_options": {"include_usage": true}})),
        tool_format(None),
        events,
    )
    .await;
}

/// The validator itself rejects the shapes this stream used to have.
#[test]
fn the_validator_rejects_the_old_stream_shapes() {
    let role = r#"{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;
    let stream = |middle: &[&str]| -> Vec<String> {
        std::iter::once(role)
            .chain(middle.iter().copied())
            .chain(std::iter::once("[DONE]"))
            .map(str::to_owned)
            .collect()
    };
    let finish = r#"{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
    assert!(validate_stream(&stream(&[finish]), false).is_ok());

    let nested_logprobs = r#"{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"x","logprobs":{"content":[]}},"finish_reason":null}]}"#;
    assert!(validate_stream(&stream(&[nested_logprobs, finish]), false).is_err());

    let merged_usage = r#"{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
    assert!(validate_stream(&stream(&[merged_usage]), true).is_err());

    let bare_uuid = stream(&[finish])
        .into_iter()
        .map(|payload| payload.replace("chatcmpl-a", "0b6c7bc6"))
        .collect::<Vec<_>>();
    assert!(validate_stream(&bare_uuid, false).is_err());

    let leaked_markup_call = r#"{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
    assert!(validate_stream(&stream(&[leaked_markup_call]), false).is_err());
}
