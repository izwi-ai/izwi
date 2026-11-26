//! The Inkling 2 chat prompt, written by hand to match the published
//! `chat_template.jinja`.
//!
//! A prompt is a run of messages, each `role token, optional name, content
//! type token, text, <|end_message|>`. A turn with several parts (thinking,
//! text, tool calls) is several such messages with the same role; an
//! assistant turn closes with `<|content_model_end_sampling|>`. Nothing opens
//! the reply: the model writes its own `<|message_model|>` header.
//!
//! The first non-system message is preceded by a system message stating the
//! thinking effort, a number in `[0, 0.99]`.
//!
//! The server hands every family messages as text. Past assistant turns
//! carry their reasoning as a leading `<think>` block and their tool calls
//! as `<tool_call>` JSON blocks (the Hermes form the reply parser also
//! writes, see `reply`); this module turns both back into the model's own
//! message types.

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::models::shared::chat::{ChatMessage, ChatReasoningEffort, ChatRequestConfig, ChatRole};
use crate::tokenizer::Tokenizer;

pub const MESSAGE_USER: &str = "<|message_user|>";
pub const MESSAGE_MODEL: &str = "<|message_model|>";
pub const MESSAGE_SYSTEM: &str = "<|message_system|>";
pub const MESSAGE_TOOL: &str = "<|message_tool|>";
pub const CONTENT_TEXT: &str = "<|content_text|>";
pub const CONTENT_IMAGE: &str = "<|content_image|>";
pub const CONTENT_THINKING: &str = "<|content_thinking|>";
pub const CONTENT_XML: &str = "<|content_xml|>";
pub const CONTENT_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub const CONTENT_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
pub const CONTENT_TOOL_ERROR: &str = "<|content_tool_error|>";
pub const END_MESSAGE: &str = "<|end_message|>";
pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";
/// The slot an image's features replace, one per image token.
pub const IMAGE_SLOT: &str = "<|unused_200054|>";

/// Markers of the text form past assistant turns arrive in.
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// The template's default when a request names no effort.
pub const DEFAULT_THINKING_EFFORT: f64 = 0.9;

/// One piece of a prompt: a special token by name, or text between two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptPiece {
    Special(&'static str),
    Text(String),
}

/// A rendered prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inkling2Prompt {
    pub pieces: Vec<PromptPiece>,
}

impl Inkling2Prompt {
    fn special(&mut self, token: &'static str) {
        self.pieces.push(PromptPiece::Special(token));
    }

    fn text(&mut self, text: impl Into<String>) {
        let text = text.into();
        if !text.is_empty() {
            self.pieces.push(PromptPiece::Text(text));
        }
    }

    /// `role [name] content_type text <|end_message|>`.
    fn message(&mut self, role: &'static str, name: &str, kind: &'static str, text: &str) {
        self.special(role);
        self.text(name);
        self.special(kind);
        self.text(text);
        self.special(END_MESSAGE);
    }

    /// The prompt as the template would print it.
    pub fn rendered(&self) -> String {
        self.pieces
            .iter()
            .map(|piece| match piece {
                PromptPiece::Special(token) => *token,
                PromptPiece::Text(text) => text.as_str(),
            })
            .collect()
    }

    /// Token ids: each special token is one id; the text between two is
    /// encoded on its own, as a tokenizer that splits on special tokens does.
    pub fn encode(&self, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for piece in &self.pieces {
            match piece {
                PromptPiece::Special(token) => ids.push(special_id(tokenizer, token)?),
                PromptPiece::Text(text) => ids.extend(tokenizer.encode(text)?),
            }
        }
        Ok(ids)
    }
}

pub fn special_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        Error::TokenizationError(format!("the Inkling 2 tokenizer has no {token} token"))
    })
}

/// The effort number a request asks for. Thinking switched off is effort 0.
pub fn thinking_effort(request: &ChatRequestConfig) -> f64 {
    // A turn that must be a tool call does not think first.
    if request.enable_thinking == Some(false) || !request.response_prefix.is_empty() {
        return 0.0;
    }
    match request.reasoning_effort {
        None => DEFAULT_THINKING_EFFORT,
        Some(ChatReasoningEffort::Xhigh) => 0.99,
        Some(ChatReasoningEffort::Medium) => 0.7,
        Some(ChatReasoningEffort::Low) => 0.2,
    }
}

/// The effort as the template prints it (`0` for none, else the number).
fn effort_text(effort: f64) -> String {
    if effort == 0.0 {
        "0".to_string()
    } else {
        format!("{effort}")
    }
}

/// `value` with every object's keys sorted, as `tojson(sort_keys=true)`
/// prints it.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|key| (key.clone(), sorted(&map[key])))
                    .collect::<Map<_, _>>(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

fn compact(value: &Value) -> Result<String> {
    serde_json::to_string(value)
        .map_err(|err| Error::InvalidInput(format!("a tool definition is not JSON: {err}")))
}

/// The tool declaration message body: each tool reduced to description,
/// name, parameters and type, printed as compact JSON with sorted keys.
fn tool_declarations(tools: &[Value]) -> Result<String> {
    let specs = tools
        .iter()
        .map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::InvalidInput("a tool definition has no name".into()))?;
            let mut spec = Map::new();
            spec.insert(
                "description".into(),
                Value::String(
                    function
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                ),
            );
            spec.insert("name".into(), Value::String(name.to_string()));
            spec.insert(
                "parameters".into(),
                match function.get("parameters") {
                    Some(parameters) if !is_empty(parameters) => parameters.clone(),
                    _ => Value::Object(Map::new()),
                },
            );
            spec.insert(
                "type".into(),
                Value::String(
                    tool.get("type")
                        .and_then(Value::as_str)
                        .filter(|kind| !kind.is_empty())
                        .unwrap_or("function")
                        .to_string(),
                ),
            );
            Ok(sorted(&Value::Object(spec)))
        })
        .collect::<Result<Vec<_>>>()?;
    compact(&Value::Array(specs))
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Object(map) => map.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::String(text) => text.is_empty(),
        _ => false,
    }
}

/// A past tool call: its name and its arguments object.
#[derive(Debug, Clone, PartialEq)]
pub struct PastToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

impl PastToolCall {
    /// `{"name":"f","args":{...}}`: top-level argument keys sorted, values
    /// printed compactly in their own order (as the template does).
    pub fn native_json(&self) -> Result<String> {
        let mut out = String::from("{\"name\":");
        out.push_str(&compact(&Value::String(self.name.clone()))?);
        out.push_str(",\"args\":{");
        let mut keys = self.arguments.keys().collect::<Vec<_>>();
        keys.sort();
        for (index, key) in keys.into_iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(&compact(&Value::String(key.clone()))?);
            out.push(':');
            out.push_str(&compact(&self.arguments[key])?);
        }
        out.push_str("}}");
        Ok(out)
    }
}

/// A past assistant turn split into the model's message types.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PastAssistantTurn {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<PastToolCall>,
}

/// Splits assistant text as the server stores it: an optional leading
/// `<think>` block, text, then `<tool_call>` JSON blocks. A block that is
/// not a JSON object with a name stays in the text.
pub fn parse_assistant_text(text: &str) -> PastAssistantTurn {
    let mut turn = PastAssistantTurn::default();
    let mut rest = text;
    let trimmed = rest.trim_start();
    if let Some(after) = trimmed.strip_prefix(THINK_OPEN) {
        if let Some(end) = after.find(THINK_CLOSE) {
            turn.reasoning = after[..end].trim().to_string();
            rest = &after[end + THINK_CLOSE.len()..];
        }
    }
    let mut content = String::new();
    while let Some(start) = rest.find(TOOL_CALL_OPEN) {
        let after = &rest[start + TOOL_CALL_OPEN.len()..];
        let Some(end) = after.find(TOOL_CALL_CLOSE) else {
            break;
        };
        let call = serde_json::from_str::<Value>(after[..end].trim())
            .ok()
            .and_then(|value| {
                let name = value.get("name")?.as_str()?.to_string();
                let arguments = match value.get("arguments").or_else(|| value.get("args")) {
                    Some(Value::Object(map)) => map.clone(),
                    None | Some(Value::Null) => Map::new(),
                    // Arguments sent as a JSON string.
                    Some(Value::String(text)) => serde_json::from_str::<Value>(text)
                        .ok()?
                        .as_object()?
                        .clone(),
                    Some(_) => return None,
                };
                Some(PastToolCall { name, arguments })
            });
        match call {
            Some(call) => {
                content.push_str(&rest[..start]);
                turn.tool_calls.push(call);
            }
            None => content
                .push_str(&rest[..start + TOOL_CALL_OPEN.len() + end + TOOL_CALL_CLOSE.len()]),
        }
        rest = &after[end + TOOL_CALL_CLOSE.len()..];
    }
    content.push_str(rest);
    turn.content = content.trim().to_string();
    turn
}

/// A tool call the answer must make (`tool_choice`). The server states it
/// as the opening of a Hermes call it has already written for the client:
/// `<tool_call>\n{"name": "f", "arguments": ` for a named function, or
/// `<tool_call>\n{"name": "` when any function will do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForcedCall {
    /// The prompt opens the call message; `json` is the call JSON it writes
    /// for the model to continue.
    Named { name: String, json: String },
    /// The model must call something; the prompt cannot open a call without
    /// its name, so only the reply is held to it.
    Any,
}

pub fn forced_call(request: &ChatRequestConfig) -> Result<Option<ForcedCall>> {
    let prefix = request.response_prefix.as_str();
    if prefix.is_empty() {
        return Ok(None);
    }
    let name = (|| {
        let after = prefix.split_once("\"name\"")?.1;
        let after = after.trim_start().strip_prefix(':')?.trim_start();
        let after = after.strip_prefix('"')?;
        let end = after.find('"')?;
        Some(after[..end].to_string())
    })();
    Ok(Some(match name {
        Some(name) => ForcedCall::Named {
            json: format!(
                "{{\"name\":{},\"args\":",
                compact(&Value::String(name.clone()))?
            ),
            name,
        },
        None => ForcedCall::Any,
    }))
}

/// Renders a conversation.
pub fn render_prompt(
    messages: &[ChatMessage],
    request: &ChatRequestConfig,
) -> Result<Inkling2Prompt> {
    if messages.is_empty() {
        return Err(Error::InvalidInput(
            "Chat request must include at least one message".to_string(),
        ));
    }
    let mut prompt = Inkling2Prompt::default();
    if !request.tools.is_empty() {
        prompt.special(MESSAGE_SYSTEM);
        prompt.text("tool_declare");
        prompt.special(CONTENT_XML);
        prompt.text(tool_declarations(&request.tools)?);
        prompt.special(END_MESSAGE);
    }
    let effort = format!(
        "Thinking effort level: {}",
        effort_text(thinking_effort(request))
    );
    let mut effort_written = false;
    // Names of the calls the latest assistant turn made, for the tool
    // results that answer them in order.
    let mut pending_calls: Vec<String> = Vec::new();
    for message in messages {
        if !effort_written && message.role != ChatRole::System {
            prompt.message(MESSAGE_SYSTEM, "", CONTENT_TEXT, &effort);
            effort_written = true;
        }
        match message.role {
            ChatRole::System => prompt.message(MESSAGE_SYSTEM, "", CONTENT_TEXT, &message.content),
            ChatRole::User => prompt.message(MESSAGE_USER, "", CONTENT_TEXT, &message.content),
            ChatRole::Tool => {
                let name = if pending_calls.is_empty() {
                    String::new()
                } else {
                    pending_calls.remove(0)
                };
                prompt.message(MESSAGE_TOOL, &name, CONTENT_TEXT, &message.content);
            }
            ChatRole::Assistant => {
                let turn = parse_assistant_text(&message.content);
                if !turn.reasoning.is_empty() {
                    prompt.message(MESSAGE_MODEL, "", CONTENT_THINKING, &turn.reasoning);
                }
                // An empty text message is written only when the turn has
                // nothing else.
                if !turn.content.is_empty()
                    || (turn.tool_calls.is_empty() && turn.reasoning.is_empty())
                {
                    prompt.message(MESSAGE_MODEL, "", CONTENT_TEXT, &turn.content);
                }
                pending_calls.clear();
                for call in &turn.tool_calls {
                    prompt.message(
                        MESSAGE_MODEL,
                        &call.name,
                        CONTENT_TOOL_JSON,
                        &call.native_json()?,
                    );
                    pending_calls.push(call.name.clone());
                }
                prompt.special(END_SAMPLING);
            }
        }
    }
    if !effort_written {
        prompt.message(MESSAGE_SYSTEM, "", CONTENT_TEXT, &effort);
    }
    // A forced call to a named function: open the call for the model, which
    // continues with the arguments.
    if let Some(ForcedCall::Named { name, json }) = forced_call(request)? {
        prompt.special(MESSAGE_MODEL);
        prompt.text(name);
        prompt.special(CONTENT_TOOL_JSON);
        prompt.text(json);
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use izwi_model_catalog::chat::tools::{
        append_assistant_tool_calls, assistant_content_with_reasoning, ToolDialect,
    };

    use super::super::config::tests::fixture;
    use super::*;

    /// A fixture conversation (OpenAI form) as the server hands it to a
    /// family: text only, reasoning and tool calls folded into the assistant
    /// text.
    fn as_server_messages(case: &Value) -> Option<Vec<ChatMessage>> {
        let mut out = Vec::new();
        for message in case["messages"].as_array().unwrap() {
            let role = match message["role"].as_str().unwrap() {
                "system" => ChatRole::System,
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                "tool" => ChatRole::Tool,
                other => panic!("role {other}"),
            };
            // Content given as parts is joined by the server before it
            // reaches a family; those cases are not comparable.
            let text = message["content"].as_str()?;
            let mut content =
                assistant_content_with_reasoning(message["reasoning_content"].as_str(), text);
            if let Some(calls) = message["tool_calls"].as_array() {
                append_assistant_tool_calls(ToolDialect::HermesJson, &mut content, calls).unwrap();
            }
            out.push(ChatMessage { role, content });
        }
        Some(out)
    }

    fn request_of(case: &Value) -> ChatRequestConfig {
        let mut request = ChatRequestConfig {
            tools: case["tools"].as_array().cloned().unwrap_or_default(),
            ..ChatRequestConfig::default()
        };
        match &case["reasoning_effort"] {
            Value::Null => {}
            Value::String(effort) => match effort.as_str() {
                "none" => request.enable_thinking = Some(false),
                "low" => request.reasoning_effort = Some(ChatReasoningEffort::Low),
                "medium" => request.reasoning_effort = Some(ChatReasoningEffort::Medium),
                "xhigh" => request.reasoning_effort = Some(ChatReasoningEffort::Xhigh),
                // Levels the server's effort type has no value for.
                _ => request.tools.push(Value::Null),
            },
            _ => request.tools.push(Value::Null),
        }
        request
    }

    #[test]
    fn renderer_matches_the_published_template() {
        let cases: Vec<Value> = serde_json::from_slice(
            &std::fs::read(fixture("inkling2").join("template_cases.json")).unwrap(),
        )
        .unwrap();
        let mut compared = Vec::new();
        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let request = request_of(case);
            if request.tools.iter().any(Value::is_null) {
                continue;
            }
            let Some(messages) = as_server_messages(case) else {
                continue;
            };
            let rendered = render_prompt(&messages, &request).unwrap().rendered();
            let want = case["rendered"].as_str().unwrap();
            assert_eq!(rendered, want, "case {name}");
            compared.push(name.to_string());
        }
        for needed in [
            "one_user_turn",
            "system_then_user",
            "two_system_messages",
            "multi_turn",
            "assistant_with_reasoning",
            "effort_none",
            "effort_low",
            "effort_medium",
            "effort_xhigh",
            "only_system",
            "tools_declared",
            "tool_round_trip",
            "two_tool_calls_with_text",
        ] {
            assert!(
                compared.iter().any(|name| name == needed),
                "{needed} was not compared"
            );
        }
    }

    #[test]
    fn assistant_text_splits_into_reasoning_text_and_calls() {
        let turn = parse_assistant_text(
            "<think>\nplan\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"b\": 1, \"a\": \"x\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"g\", \"arguments\": \"{\\\"k\\\": true}\"}\n</tool_call>",
        );
        assert_eq!(turn.reasoning, "plan");
        assert_eq!(turn.content, "Sure.");
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(
            turn.tool_calls[0].native_json().unwrap(),
            "{\"name\":\"f\",\"args\":{\"a\":\"x\",\"b\":1}}"
        );
        assert_eq!(
            turn.tool_calls[1].native_json().unwrap(),
            "{\"name\":\"g\",\"args\":{\"k\":true}}"
        );
        // Markup that is not a call stays text.
        let turn = parse_assistant_text("See <tool_call>not json</tool_call> here");
        assert!(turn.tool_calls.is_empty());
        assert_eq!(turn.content, "See <tool_call>not json</tool_call> here");
        // An unclosed think block is text.
        let turn = parse_assistant_text("<think>never closed");
        assert_eq!(
            (turn.reasoning.as_str(), turn.content.as_str()),
            ("", "<think>never closed")
        );
    }

    #[test]
    fn a_forced_call_opens_the_tool_message() {
        let request = ChatRequestConfig {
            response_prefix: "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ".to_string(),
            ..ChatRequestConfig::default()
        };
        let messages = [ChatMessage {
            role: ChatRole::User,
            content: "Weather?".into(),
        }];
        let rendered = render_prompt(&messages, &request).unwrap().rendered();
        assert!(rendered.ends_with(
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
        ));
        assert!(rendered.contains("Thinking effort level: 0<|end_message|>"));
        // Any function: the prompt cannot open the call, so it ends at the
        // conversation.
        let request = ChatRequestConfig {
            response_prefix: "<tool_call>\n{\"name\": \"".to_string(),
            ..ChatRequestConfig::default()
        };
        assert_eq!(forced_call(&request).unwrap(), Some(ForcedCall::Any));
        let rendered = render_prompt(&messages, &request).unwrap().rendered();
        assert!(rendered.ends_with("<|content_text|>Weather?<|end_message|>"));
    }

    #[test]
    fn tool_results_take_the_names_of_the_calls_they_answer() {
        let messages = [
            ChatMessage { role: ChatRole::User, content: "Go".into() },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>".into(),
            },
            ChatMessage { role: ChatRole::Tool, content: "1".into() },
            ChatMessage { role: ChatRole::Tool, content: "2".into() },
            ChatMessage { role: ChatRole::Tool, content: "3".into() },
        ];
        let rendered = render_prompt(&messages, &ChatRequestConfig::default())
            .unwrap()
            .rendered();
        assert!(rendered.ends_with(
            "<|message_tool|>a<|content_text|>1<|end_message|><|message_tool|>b<|content_text|>2<|end_message|><|message_tool|><|content_text|>3<|end_message|>"
        ));
    }
}
