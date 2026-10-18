//! OpenAI tool calling: the tool-call format a model family writes, how a
//! request's `tools` and `tool_choice` apply, and an incremental parser that
//! turns the model's tool-call markup into OpenAI `tool_calls`.
//!
//! The parser runs where the local and gateway chat paths meet, on the
//! answer text either path streams. Markup never reaches `content`: text that
//! could still open a call is held back until the next delta decides it, a
//! Qwen call is sent as soon as its name is known and its arguments one
//! parameter at a time, and a block that does not parse is released as text.

use izwi_model_catalog::{chat_supports_tools, ModelFamily, ModelVariant};
use serde::Serialize;
use serde_json::Value;

use super::lfm_tools;
use crate::error::ApiError;
use crate::ids::new_prefixed_id;

const QWEN_OPEN: &str = "<tool_call>";
const QWEN_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// How a model family writes tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// The Qwen3.5 family: `<tool_call><function=f><parameter=k>v</parameter>
    /// </function></tool_call>`. A JSON `{"name", "arguments"}` body is
    /// accepted too.
    QwenXml,
    /// LFM2.5: `<|tool_call_start|>[f(k='v')]<|tool_call_end|>`.
    LfmPythonic,
}

impl ToolCallFormat {
    /// The format of a model whose chat template renders tools.
    pub fn of(variant: ModelVariant) -> Option<Self> {
        if !chat_supports_tools(variant) {
            return None;
        }
        Some(if variant.family() == ModelFamily::Lfm2Chat {
            Self::LfmPythonic
        } else {
            Self::QwenXml
        })
    }

    fn open(self) -> &'static str {
        match self {
            Self::QwenXml => QWEN_OPEN,
            Self::LfmPythonic => lfm_tools::TOOL_CALL_START,
        }
    }
}

/// How a request's `tools` and `tool_choice` apply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolPolicy {
    /// The tool definitions rendered into the prompt.
    pub tools: Vec<Value>,
    /// The format tool calls are parsed in; `None` leaves the answer text.
    pub format: Option<ToolCallFormat>,
}

/// Resolve OpenAI `tools` and `tool_choice` for `variant`. `"auto"` (the
/// default) offers the tools and parses calls; `"none"` does neither. Forcing
/// a call (`"required"` or a named function) needs constrained decoding,
/// which no chat template here provides, so it is refused.
pub fn resolve_tool_policy(
    variant: ModelVariant,
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
) -> Result<ToolPolicy, ApiError> {
    let tools = tools.unwrap_or_default();
    let offered = match tool_choice {
        None | Some(Value::Null) => true,
        Some(Value::String(choice)) if choice == "auto" => true,
        Some(Value::String(choice)) if choice == "none" => false,
        Some(choice) if choice == "required" || is_named_function(choice) => {
            return Err(ApiError::unsupported_param(
                "tool_choice",
                "`tool_choice` \"required\" and named functions are not supported; use \"auto\" or \"none\"",
            ));
        }
        Some(_) => {
            return Err(ApiError::invalid_param(
                "tool_choice",
                "`tool_choice` must be \"none\", \"auto\", \"required\" or a named function",
            ));
        }
    };
    if let Some(index) = tools.iter().position(|tool| !is_function_tool(tool)) {
        return Err(ApiError::invalid_param(
            "tools",
            format!("`tools[{index}]` must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"),
        ));
    }
    if tools.is_empty() || !offered {
        return Ok(ToolPolicy::default());
    }
    let format = ToolCallFormat::of(variant).ok_or_else(|| {
        ApiError::unsupported_param(
            "tools",
            format!("Model {variant} has no tool-calling chat template"),
        )
    })?;
    Ok(ToolPolicy {
        tools: tools.to_vec(),
        format: Some(format),
    })
}

fn is_named_function(choice: &Value) -> bool {
    choice.get("type").and_then(Value::as_str) == Some("function")
        && choice
            .pointer("/function/name")
            .and_then(Value::as_str)
            .is_some()
}

fn is_function_tool(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str) == Some("function")
        && tool
            .pointer("/function/name")
            .and_then(Value::as_str)
            .is_some_and(|name| !name.trim().is_empty())
}

/// A complete tool call (OpenAI `message.tool_calls[]`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// A fragment of a streamed tool call (OpenAI `delta.tool_calls[]`): the
/// first carries the id, type and name; the arguments concatenate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    pub function: ToolCallDeltaFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCallDeltaFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}

/// Answer text and tool-call fragments the parser has made final.
#[derive(Debug, Default, PartialEq)]
pub struct ParsedOutput {
    pub content: String,
    pub tool_calls: Vec<ToolCallDelta>,
}

impl ParsedOutput {
    pub fn extend(&mut self, other: ParsedOutput) {
        self.content.push_str(&other.content);
        for delta in other.tool_calls {
            self.push_delta(delta);
        }
    }

    fn push_delta(&mut self, delta: ToolCallDelta) {
        match self.tool_calls.last_mut() {
            Some(last) if last.index == delta.index && delta.id.is_none() => {
                last.function.arguments.push_str(&delta.function.arguments);
            }
            _ => self.tool_calls.push(delta),
        }
    }

    fn start_call(&mut self, index: usize, name: String, arguments: String) {
        self.tool_calls.push(ToolCallDelta {
            index,
            id: Some(new_prefixed_id("call_")),
            kind: Some("function"),
            function: ToolCallDeltaFunction {
                name: Some(name),
                arguments,
            },
        });
    }

    fn push_arguments(&mut self, index: usize, arguments: String) {
        self.push_delta(ToolCallDelta {
            index,
            id: None,
            kind: None,
            function: ToolCallDeltaFunction {
                name: None,
                arguments,
            },
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Text,
    /// Inside `<tool_call>` until the body shows its form.
    QwenBlockStart,
    /// A `<function=...>` call whose name was sent; `params` arguments
    /// followed it.
    QwenParams {
        params: usize,
    },
    /// After `</function>`, before `</tool_call>`.
    QwenBlockEnd,
    /// Any other `<tool_call>` body, held whole until its close tag.
    QwenHeld,
    /// An LFM2.5 call list, held whole until its end marker.
    LfmBlock,
}

/// Incremental tool-call parser over a model's answer text.
pub struct ToolCallParser {
    format: ToolCallFormat,
    state: State,
    /// Input not yet decided.
    pending: String,
    /// Whitespace after the last text, sent only once more text follows it,
    /// so whitespace around calls is dropped.
    held_space: String,
    /// Calls started so far; the next call's index.
    calls: usize,
}

impl ToolCallParser {
    pub fn new(format: ToolCallFormat) -> Self {
        Self {
            format,
            state: State::Text,
            pending: String::new(),
            held_space: String::new(),
            calls: 0,
        }
    }

    /// Whether the answer has started a tool call.
    pub fn saw_tool_calls(&self) -> bool {
        self.calls > 0
    }

    /// Feed one delta; returns what it made final.
    pub fn push(&mut self, delta: &str) -> ParsedOutput {
        self.pending.push_str(delta);
        let mut out = ParsedOutput::default();
        while self.step(&mut out) {}
        out
    }

    /// Flush the end of the answer. An unterminated block is released as
    /// text; a Qwen call cut inside its parameters closes its arguments.
    pub fn finish(&mut self) -> ParsedOutput {
        let mut out = ParsedOutput::default();
        let pending = std::mem::take(&mut self.pending);
        match self.state {
            State::Text => self.emit_text(&pending, &mut out),
            State::QwenBlockStart | State::QwenHeld => {
                self.emit_text(&format!("{QWEN_OPEN}{pending}"), &mut out)
            }
            State::LfmBlock => {
                self.emit_text(&format!("{}{pending}", self.format.open()), &mut out)
            }
            State::QwenParams { params } => {
                out.push_arguments(self.calls - 1, close_arguments(params).to_string())
            }
            State::QwenBlockEnd => {}
        }
        self.state = State::Text;
        if self.calls == 0 {
            out.content.push_str(&std::mem::take(&mut self.held_space));
        }
        out
    }

    /// Text to send, minus trailing whitespace, which waits for what follows.
    fn emit_text(&mut self, text: &str, out: &mut ParsedOutput) {
        let body = text.trim_end();
        if body.is_empty() {
            self.held_space.push_str(text);
            return;
        }
        out.content.push_str(&std::mem::take(&mut self.held_space));
        out.content.push_str(body);
        self.held_space.push_str(&text[body.len()..]);
    }

    fn start_call(&mut self, name: String, arguments: String, out: &mut ParsedOutput) {
        self.held_space.clear();
        out.start_call(self.calls, name, arguments);
        self.calls += 1;
    }

    /// One parsing step; `true` when it consumed input and should run again.
    fn step(&mut self, out: &mut ParsedOutput) -> bool {
        match self.state {
            State::Text => {
                let open = self.format.open();
                if let Some(at) = self.pending.find(open) {
                    let before: String = self.pending.drain(..at + open.len()).collect();
                    self.emit_text(&before[..at], out);
                    self.state = match self.format {
                        ToolCallFormat::QwenXml => State::QwenBlockStart,
                        ToolCallFormat::LfmPythonic => State::LfmBlock,
                    };
                    return true;
                }
                let emit = self.pending.len() - partial_tag_suffix(&self.pending, open);
                let text: String = self.pending.drain(..emit).collect();
                self.emit_text(&text, out);
                false
            }
            State::QwenBlockStart => {
                let body = self.pending.trim_start();
                if let Some(rest) = body.strip_prefix(FUNCTION_OPEN) {
                    let Some(end) = rest.find('>') else {
                        return false;
                    };
                    let name = rest[..end].trim().to_string();
                    if name.is_empty() {
                        self.state = State::QwenHeld;
                        return true;
                    }
                    self.pending = rest[end + 1..].to_string();
                    self.start_call(name, String::new(), out);
                    self.state = State::QwenParams { params: 0 };
                    return true;
                }
                if body.is_empty() || FUNCTION_OPEN.starts_with(body) {
                    return false;
                }
                self.state = State::QwenHeld;
                true
            }
            State::QwenParams { params } => self.step_qwen_params(params, out),
            State::QwenBlockEnd => {
                let body = self.pending.trim_start();
                if let Some(rest) = body.strip_prefix(QWEN_CLOSE) {
                    self.pending = rest.to_string();
                    self.state = State::Text;
                    return true;
                }
                if body.is_empty() || QWEN_CLOSE.starts_with(body) {
                    return false;
                }
                // The block was never closed; what follows is the answer.
                self.state = State::Text;
                true
            }
            State::QwenHeld => {
                let Some(at) = self.pending.find(QWEN_CLOSE) else {
                    return false;
                };
                let body: String = self.pending.drain(..at + QWEN_CLOSE.len()).collect();
                let body = &body[..at];
                match json_call(body) {
                    Some((name, arguments)) => {
                        self.start_call(name, arguments, out);
                        self.state = State::Text;
                    }
                    None => {
                        self.emit_text(&format!("{QWEN_OPEN}{body}{QWEN_CLOSE}"), out);
                        self.state = State::Text;
                    }
                }
                true
            }
            State::LfmBlock => {
                let end_marker = lfm_tools::TOOL_CALL_END;
                let Some(at) = self.pending.find(end_marker) else {
                    return false;
                };
                let body: String = self.pending.drain(..at + end_marker.len()).collect();
                let body = &body[..at];
                match lfm_tools::parse_call_list(body) {
                    Some(calls) => {
                        for call in calls {
                            let arguments = Value::Object(call.arguments).to_string();
                            self.start_call(call.name, arguments, out);
                        }
                        self.state = State::Text;
                    }
                    None => {
                        let open = self.format.open();
                        self.emit_text(&format!("{open}{body}{end_marker}"), out);
                        self.state = State::Text;
                    }
                }
                true
            }
        }
    }

    /// Inside a Qwen function: each complete parameter becomes one argument
    /// fragment, and `</function>` closes the arguments object.
    fn step_qwen_params(&mut self, params: usize, out: &mut ParsedOutput) -> bool {
        let index = self.calls - 1;
        let body = self.pending.trim_start();
        if let Some(rest) = body.strip_prefix(PARAMETER_OPEN) {
            let Some(key_end) = rest.find('>') else {
                return false;
            };
            let Some(value_end) = rest[key_end + 1..]
                .find(PARAMETER_CLOSE)
                .map(|at| key_end + 1 + at)
            else {
                return false;
            };
            let key = rest[..key_end].trim();
            let raw = rest[key_end + 1..value_end].trim();
            let value = serde_json::from_str::<Value>(raw)
                .unwrap_or_else(|_| Value::String(raw.to_string()));
            let separator = if params == 0 { "{" } else { "," };
            let key = Value::String(key.to_string());
            out.push_arguments(index, format!("{separator}{key}:{value}"));
            self.pending = rest[value_end + PARAMETER_CLOSE.len()..].to_string();
            self.state = State::QwenParams { params: params + 1 };
            return true;
        }
        for (close, next) in [
            (FUNCTION_CLOSE, State::QwenBlockEnd),
            // A block closed without `</function>`.
            (QWEN_CLOSE, State::Text),
        ] {
            if let Some(rest) = body.strip_prefix(close) {
                out.push_arguments(index, close_arguments(params).to_string());
                self.pending = rest.to_string();
                self.state = next;
                return true;
            }
        }
        if body.is_empty()
            || [PARAMETER_OPEN, FUNCTION_CLOSE, QWEN_CLOSE]
                .iter()
                .any(|tag| tag.starts_with(body))
        {
            return false;
        }
        // Text between parameters is not part of the call; skip to the next
        // tag.
        let skip = body[1..].find('<').map_or(body.len(), |at| at + 1);
        self.pending = body[skip..].to_string();
        true
    }
}

/// The end of an arguments object after `params` fragments.
fn close_arguments(params: usize) -> &'static str {
    if params == 0 {
        "{}"
    } else {
        "}"
    }
}

/// A Hermes-style `{"name": ..., "arguments": {...}}` call body.
fn json_call(body: &str) -> Option<(String, String)> {
    let call: Value = serde_json::from_str(body.trim()).ok()?;
    let name = call.get("name")?.as_str()?.to_string();
    let arguments = match call.get("arguments") {
        None | Some(Value::Null) => "{}".to_string(),
        Some(Value::Object(map)) => Value::Object(map.clone()).to_string(),
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text).ok()? {
            Value::Object(map) => Value::Object(map).to_string(),
            _ => return None,
        },
        Some(_) => return None,
    };
    Some((name, arguments))
}

/// The length of the longest suffix of `text` that begins `tag`.
fn partial_tag_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len().min(text.len() + 1))
        .rev()
        .find(|&len| {
            text.is_char_boundary(text.len() - len) && tag.starts_with(&text[text.len() - len..])
        })
        .unwrap_or(0)
}

/// Parse a whole answer: its text (trimmed when it made calls, `None` when
/// that leaves nothing) and its calls.
pub fn parse_answer(format: ToolCallFormat, text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut parser = ToolCallParser::new(format);
    let mut parsed = parser.push(text);
    parsed.extend(parser.finish());
    let mut calls: Vec<ToolCall> = Vec::new();
    for delta in parsed.tool_calls {
        match calls.get_mut(delta.index) {
            Some(call) => call.function.arguments.push_str(&delta.function.arguments),
            None => calls.push(ToolCall {
                id: delta.id.unwrap_or_else(|| new_prefixed_id("call_")),
                kind: "function",
                function: ToolCallFunction {
                    name: delta.function.name.unwrap_or_default(),
                    arguments: delta.function.arguments,
                },
            }),
        }
    }
    if calls.is_empty() {
        return (Some(parsed.content), calls);
    }
    let content = parsed.content.trim();
    ((!content.is_empty()).then(|| content.to_string()), calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Feed `text` in `chunk`-character deltas; returns the streamed text,
    /// the calls assembled from their fragments, and the fragment count.
    fn stream(format: ToolCallFormat, text: &str, chunk: usize) -> (String, Vec<ToolCall>, usize) {
        let mut parser = ToolCallParser::new(format);
        let chars: Vec<char> = text.chars().collect();
        let mut out = ParsedOutput::default();
        let mut fragments = 0;
        for piece in chars.chunks(chunk) {
            let parsed = parser.push(&piece.iter().collect::<String>());
            fragments += parsed.tool_calls.len();
            out.extend(parsed);
        }
        let tail = parser.finish();
        fragments += tail.tool_calls.len();
        out.extend(tail);
        let mut calls: Vec<ToolCall> = Vec::new();
        for delta in out.tool_calls {
            match calls.get_mut(delta.index) {
                Some(call) => {
                    assert!(delta.id.is_none() && delta.function.name.is_none());
                    call.function.arguments.push_str(&delta.function.arguments);
                }
                None => {
                    assert_eq!(delta.index, calls.len(), "indexes are dense");
                    assert!(delta
                        .id
                        .as_deref()
                        .is_some_and(|id| id.starts_with("call_")));
                    assert_eq!(delta.kind, Some("function"));
                    calls.push(ToolCall {
                        id: delta.id.unwrap(),
                        kind: "function",
                        function: ToolCallFunction {
                            name: delta.function.name.expect("first fragment names the call"),
                            arguments: delta.function.arguments,
                        },
                    });
                }
            }
        }
        (out.content, calls, fragments)
    }

    fn arguments(call: &ToolCall) -> Value {
        serde_json::from_str(&call.function.arguments).expect("arguments are JSON")
    }

    const QWEN: &str = "Checking that now.\n<tool_call>\n<function=get_time>\n<parameter=timezone>\n\"Africa/Harare\"\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn qwen_calls_stream_without_markup_at_every_split() {
        for chunk in 1..=QWEN.len() {
            let (content, calls, _) = stream(ToolCallFormat::QwenXml, QWEN, chunk);
            assert_eq!(content, "Checking that now.", "chunk {chunk}");
            assert_eq!(calls.len(), 1, "chunk {chunk}");
            assert_eq!(calls[0].function.name, "get_time");
            assert_eq!(
                arguments(&calls[0]),
                json!({"timezone": "Africa/Harare", "days": 2}),
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn qwen_arguments_stream_one_parameter_at_a_time() {
        // Character by character: the name, each parameter and the close
        // arrive as separate fragments.
        let (_, calls, fragments) = stream(ToolCallFormat::QwenXml, QWEN, 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(fragments, 4);
        // The whole answer at once merges them.
        let (_, _, fragments) = stream(ToolCallFormat::QwenXml, QWEN, QWEN.len());
        assert_eq!(fragments, 1);
    }

    #[test]
    fn qwen_parses_several_calls_json_bodies_and_empty_arguments() {
        let text = "<tool_call>\n<function=now>\n</function>\n</tool_call>\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        for chunk in [1, 3, 7, text.len()] {
            let (content, calls, _) = stream(ToolCallFormat::QwenXml, text, chunk);
            assert_eq!(content, "");
            assert_eq!(calls.len(), 2);
            assert_eq!(
                (calls[0].function.name.as_str(), arguments(&calls[0])),
                ("now", json!({}))
            );
            assert_eq!(
                (calls[1].function.name.as_str(), arguments(&calls[1])),
                ("weather", json!({"city": "Paris"}))
            );
        }
    }

    #[test]
    fn text_that_only_looks_like_a_call_is_released_unchanged() {
        for text in [
            "a < b and <tool_ is not a tag",
            "keep <tool_call>not a call</tool_call> as text",
            "unterminated <tool_call>\n{\"name\": \"f\"",
            "  spaced  \n",
        ] {
            for chunk in [1, 2, 5, text.len()] {
                let (content, calls, _) = stream(ToolCallFormat::QwenXml, text, chunk);
                assert!(calls.is_empty(), "{text:?}");
                assert_eq!(content, text, "chunk {chunk}");
            }
        }
    }

    #[test]
    fn a_call_cut_off_inside_its_parameters_keeps_the_complete_ones() {
        let text = "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2";
        let (content, calls, _) = stream(ToolCallFormat::QwenXml, text, 4);
        assert_eq!(content, "");
        assert_eq!(arguments(&calls[0]), json!({"a": 1}));
    }

    const LFM: &str = "Checking.<|tool_call_start|>[get_weather(city='Paris', days=2), now()]<|tool_call_end|> Done.";

    #[test]
    fn lfm_calls_stream_without_markup_at_every_split() {
        for chunk in 1..=LFM.len() {
            let (content, calls, _) = stream(ToolCallFormat::LfmPythonic, LFM, chunk);
            assert_eq!(content, "Checking. Done.", "chunk {chunk}");
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].function.name, "get_weather");
            assert_eq!(arguments(&calls[0]), json!({"city": "Paris", "days": 2}));
            assert_eq!(calls[1].function.name, "now");
            assert_eq!(arguments(&calls[1]), json!({}));
        }
    }

    #[test]
    fn malformed_or_unterminated_lfm_blocks_stay_in_the_text() {
        for text in [
            "<|tool_call_start|>[f(a=)]<|tool_call_end|>",
            "<|tool_call_start|>not a list<|tool_call_end|>",
            "partial <|tool_call_start|>[f(a=1)]",
        ] {
            let (content, calls, _) = stream(ToolCallFormat::LfmPythonic, text, 3);
            assert!(calls.is_empty(), "{text}");
            assert_eq!(content, text);
        }
    }

    #[test]
    fn whole_answers_parse_like_the_stream() {
        let (content, calls) = parse_answer(ToolCallFormat::QwenXml, QWEN);
        assert_eq!(content.as_deref(), Some("Checking that now."));
        assert_eq!(
            arguments(&calls[0]),
            json!({"timezone": "Africa/Harare", "days": 2})
        );
        let (content, calls) = parse_answer(
            ToolCallFormat::LfmPythonic,
            "<|tool_call_start|>[now()]<|tool_call_end|>",
        );
        assert_eq!((content, calls.len()), (None, 1));
        let (content, calls) = parse_answer(ToolCallFormat::QwenXml, "  plain answer\n");
        assert_eq!(content.as_deref(), Some("  plain answer\n"));
        assert!(calls.is_empty());
    }

    #[test]
    fn tool_choice_offers_parses_or_refuses() {
        let qwen = ModelVariant::Qwen359BGguf;
        let tools = [json!({"type": "function", "function": {"name": "now"}})];
        let policy = |variant, tools: &[Value], choice: Option<Value>| {
            resolve_tool_policy(variant, Some(tools), choice.as_ref())
        };

        let auto = policy(qwen, &tools, None).unwrap();
        assert_eq!(auto.tools, tools);
        assert_eq!(auto.format, Some(ToolCallFormat::QwenXml));
        assert_eq!(policy(qwen, &tools, Some(json!("auto"))).unwrap(), auto);
        assert_eq!(
            policy(ModelVariant::Lfm2512BInstructGguf, &tools, None)
                .unwrap()
                .format,
            Some(ToolCallFormat::LfmPythonic)
        );
        // "none" neither offers the tools nor parses calls.
        assert_eq!(
            policy(qwen, &tools, Some(json!("none"))).unwrap(),
            ToolPolicy::default()
        );
        assert_eq!(policy(qwen, &[], None).unwrap(), ToolPolicy::default());

        let refused = |variant, tools: &[Value], choice: Option<Value>| {
            let error = policy(variant, tools, choice).expect_err("refused");
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
            error.detail.unwrap()
        };
        for forced in [
            json!("required"),
            json!({"type": "function", "function": {"name": "now"}}),
        ] {
            let detail = refused(qwen, &tools, Some(forced));
            assert_eq!(detail.param, Some("tool_choice"));
            assert_eq!(detail.code, Some("unsupported_parameter"));
        }
        assert_eq!(
            refused(qwen, &tools, Some(json!("sometimes"))).code,
            Some("invalid_value")
        );
        assert_eq!(
            refused(qwen, &[json!({"name": "now"})], None).param,
            Some("tools")
        );
        // A model without a tool template refuses tools it would ignore.
        let detail = refused(ModelVariant::Qwen314BGguf, &tools, None);
        assert_eq!(
            (detail.param, detail.code),
            (Some("tools"), Some("unsupported_parameter"))
        );
        assert_eq!(
            policy(ModelVariant::Qwen314BGguf, &tools, Some(json!("none"))).unwrap(),
            ToolPolicy::default()
        );
    }
}
