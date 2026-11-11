//! An incremental parser that turns a model's tool-call markup into OpenAI
//! `tool_calls`.
//!
//! It runs on the answer text every chat path streams. Markup never reaches
//! `content`: text that could still open a call is held back until the next
//! delta decides it, an XML call is sent as soon as its name is known and its
//! arguments one parameter at a time, and a block that does not parse (or
//! names a tool the request did not declare) is released as text. A finished
//! generation is never failed over its markup.

use serde::Serialize;
use serde_json::Value;

use super::dialect::ToolDialect;
use super::lfm;
use super::schema::ToolSchemas;

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

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
            id: Some(new_call_id()),
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

    /// The fragments assembled into whole calls.
    pub fn into_calls(self) -> (String, Vec<ToolCall>) {
        let mut calls: Vec<ToolCall> = Vec::new();
        for delta in self.tool_calls {
            match calls.get_mut(delta.index) {
                Some(call) => call.function.arguments.push_str(&delta.function.arguments),
                None => calls.push(ToolCall {
                    id: delta.id.unwrap_or_else(new_call_id),
                    kind: "function",
                    function: ToolCallFunction {
                        name: delta.function.name.unwrap_or_default(),
                        arguments: delta.function.arguments,
                    },
                }),
            }
        }
        (self.content, calls)
    }
}

fn new_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Text,
    /// Inside `<tool_call>` until the body shows its form.
    BlockStart,
    /// A `<function=...>` call whose name was sent; `params` arguments
    /// followed it.
    Params {
        params: usize,
    },
    /// After `</function>`, before `</tool_call>`.
    BlockEnd,
    /// Any other `<tool_call>` body (a Hermes JSON call), held whole until
    /// its close tag.
    Held,
    /// An LFM2.5 call list, held whole until its end marker.
    LfmBlock,
    /// The request allows one call and it is complete: the rest is dropped.
    Done,
}

/// Incremental tool-call parser over a model's answer text.
pub struct ToolCallParser {
    dialect: ToolDialect,
    schemas: ToolSchemas,
    state: State,
    /// Input not yet decided.
    pending: String,
    /// Whitespace after the last text, sent only once more text follows it,
    /// so whitespace around calls is dropped.
    held_space: String,
    /// Calls started so far; the next call's index.
    calls: usize,
    /// The name of the XML call being read, for its parameters' types.
    current: String,
    /// The most calls the answer may make (`parallel_tool_calls: false`).
    max_calls: Option<usize>,
    /// The answer ended inside a call.
    truncated: bool,
}

impl ToolCallParser {
    pub fn new(dialect: ToolDialect) -> Self {
        Self {
            dialect,
            schemas: ToolSchemas::default(),
            state: State::Text,
            pending: String::new(),
            held_space: String::new(),
            calls: 0,
            current: String::new(),
            max_calls: None,
            truncated: false,
        }
    }

    /// The request's tool definitions: calls to other names stay text, and
    /// bare-text arguments are read by their declared type.
    pub fn with_tools(mut self, tools: &[Value]) -> Self {
        self.schemas = ToolSchemas::new(tools);
        self
    }

    /// Keep only the first `max_calls` calls and drop what follows them.
    pub fn with_max_calls(mut self, max_calls: Option<usize>) -> Self {
        self.max_calls = max_calls;
        self
    }

    /// Text the prompt already wrote for the model at the start of its turn
    /// (a forced call's opening), read as the start of the answer.
    pub fn with_prefill(mut self, prefill: &str) -> Self {
        self.pending.push_str(prefill);
        self
    }

    /// Whether the answer has started a tool call.
    pub fn saw_tool_calls(&self) -> bool {
        self.calls > 0
    }

    /// Whether the answer ended inside a call, which then lacks arguments.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Feed one delta; returns what it made final.
    pub fn push(&mut self, delta: &str) -> ParsedOutput {
        self.pending.push_str(delta);
        let mut out = ParsedOutput::default();
        while self.step(&mut out) {}
        out
    }

    /// Flush the end of the answer. An unterminated block is released as
    /// text unless its body is already a whole call; an XML call cut inside
    /// its parameters keeps the complete ones.
    pub fn finish(&mut self) -> ParsedOutput {
        let mut out = ParsedOutput::default();
        while self.step(&mut out) {}
        let pending = std::mem::take(&mut self.pending);
        match self.state {
            State::Text => self.emit_text(&pending, &mut out),
            State::BlockStart => self.emit_text(&format!("{CALL_OPEN}{pending}"), &mut out),
            State::Held => match self.json_call(&pending) {
                Some((name, arguments)) => self.start_call(name, arguments, &mut out),
                None => {
                    self.truncated = true;
                    self.emit_text(&format!("{CALL_OPEN}{pending}"), &mut out)
                }
            },
            State::LfmBlock => {
                // A stop sequence takes the end marker; a whole list stands.
                self.pending = format!("{pending}{}", lfm::TOOL_CALL_END);
                let before = self.calls;
                while self.step(&mut out) {}
                if self.calls == before {
                    self.truncated = true;
                    let text = out
                        .content
                        .strip_suffix(lfm::TOOL_CALL_END)
                        .map(str::to_owned);
                    if let Some(text) = text {
                        out.content = text;
                    }
                }
            }
            State::Params { params } => {
                self.truncated = true;
                out.push_arguments(self.calls - 1, close_arguments(params).to_string())
            }
            State::BlockEnd | State::Done => {}
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

    /// The state after a call is complete: more text, or nothing more when
    /// the request's call limit is reached.
    fn after_call(&self) -> State {
        if self.max_calls.is_some_and(|max| self.calls >= max) {
            State::Done
        } else {
            State::Text
        }
    }

    /// One parsing step; `true` when it consumed input and should run again.
    fn step(&mut self, out: &mut ParsedOutput) -> bool {
        match self.state {
            State::Text => {
                let open = self.dialect.call_opener();
                if let Some(at) = self.pending.find(open) {
                    let before: String = self.pending.drain(..at + open.len()).collect();
                    self.emit_text(&before[..at], out);
                    self.state = match self.dialect {
                        ToolDialect::LfmPythonic => State::LfmBlock,
                        _ => State::BlockStart,
                    };
                    return true;
                }
                let emit = self.pending.len() - partial_tag_suffix(&self.pending, open);
                let text: String = self.pending.drain(..emit).collect();
                self.emit_text(&text, out);
                false
            }
            State::BlockStart => {
                let body = self.pending.trim_start();
                if let Some(rest) = body.strip_prefix(FUNCTION_OPEN) {
                    let Some(end) = rest.find('>') else {
                        return false;
                    };
                    let name = rest[..end].trim().to_string();
                    if name.is_empty() || !self.schemas.declares(&name) {
                        self.state = State::Held;
                        return true;
                    }
                    self.pending = rest[end + 1..].to_string();
                    self.current = name.clone();
                    self.start_call(name, String::new(), out);
                    self.state = State::Params { params: 0 };
                    return true;
                }
                if body.is_empty() || FUNCTION_OPEN.starts_with(body) {
                    return false;
                }
                self.state = State::Held;
                true
            }
            State::Params { params } => self.step_params(params, out),
            State::BlockEnd => {
                let body = self.pending.trim_start();
                if let Some(rest) = body.strip_prefix(CALL_CLOSE) {
                    self.pending = rest.to_string();
                    self.state = self.after_call();
                    return true;
                }
                if body.is_empty() || CALL_CLOSE.starts_with(body) {
                    return false;
                }
                // The block was never closed; what follows is the answer.
                self.state = self.after_call();
                true
            }
            State::Held => {
                // The block ends at the first close tag that completes its
                // JSON: a close tag inside a JSON string is part of the
                // value.
                let mut from = 0;
                loop {
                    let Some(at) = self.pending[from..].find(CALL_CLOSE).map(|at| from + at) else {
                        return false;
                    };
                    let body = &self.pending[..at];
                    if let Some((name, arguments)) = self.json_call(body) {
                        self.pending.drain(..at + CALL_CLOSE.len());
                        self.start_call(name, arguments, out);
                        self.state = self.after_call();
                        return true;
                    }
                    if !is_json_prefix(body) {
                        // Not a call: the block is text.
                        let block: String = self.pending.drain(..at + CALL_CLOSE.len()).collect();
                        self.emit_text(&format!("{CALL_OPEN}{block}"), out);
                        self.state = State::Text;
                        return true;
                    }
                    from = at + CALL_CLOSE.len();
                }
            }
            State::LfmBlock => {
                let end_marker = lfm::TOOL_CALL_END;
                let Some(at) = self.pending.find(end_marker) else {
                    return false;
                };
                let body: String = self.pending.drain(..at + end_marker.len()).collect();
                let body = &body[..at];
                let calls = lfm::parse_call_list(body)
                    .filter(|calls| calls.iter().all(|call| self.schemas.declares(&call.name)));
                match calls {
                    Some(calls) => {
                        let room = self
                            .max_calls
                            .map_or(usize::MAX, |max| max.saturating_sub(self.calls));
                        for call in calls.into_iter().take(room) {
                            let arguments = Value::Object(call.arguments).to_string();
                            self.start_call(call.name, arguments, out);
                        }
                        self.state = self.after_call();
                    }
                    None => {
                        let open = self.dialect.call_opener();
                        self.emit_text(&format!("{open}{body}{end_marker}"), out);
                        self.state = State::Text;
                    }
                }
                true
            }
            State::Done => {
                self.pending.clear();
                false
            }
        }
    }

    /// Inside an XML function: each complete parameter becomes one argument
    /// fragment, and `</function>` closes the arguments object.
    fn step_params(&mut self, params: usize, out: &mut ParsedOutput) -> bool {
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
            let raw = parameter_text(&rest[key_end + 1..value_end]);
            let value = self.schemas.text_argument(&self.current, key, raw);
            let separator = if params == 0 { "{" } else { "," };
            let key = Value::String(key.to_string());
            out.push_arguments(index, format!("{separator}{key}:{value}"));
            self.pending = rest[value_end + PARAMETER_CLOSE.len()..].to_string();
            self.state = State::Params { params: params + 1 };
            return true;
        }
        for (close, closes_block) in [
            (FUNCTION_CLOSE, false),
            // A block closed without `</function>`.
            (CALL_CLOSE, true),
        ] {
            if let Some(rest) = body.strip_prefix(close) {
                out.push_arguments(index, close_arguments(params).to_string());
                self.pending = rest.to_string();
                self.state = if closes_block {
                    self.after_call()
                } else {
                    State::BlockEnd
                };
                return true;
            }
        }
        if body.is_empty()
            || [PARAMETER_OPEN, FUNCTION_CLOSE, CALL_CLOSE]
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

    /// A Hermes-style `{"name": ..., "arguments": {...}}` call body naming a
    /// declared tool.
    fn json_call(&self, body: &str) -> Option<(String, String)> {
        let call: Value = serde_json::from_str(body.trim()).ok()?;
        let name = call.get("name")?.as_str()?.to_string();
        if !self.schemas.declares(&name) {
            return None;
        }
        let arguments = match call.get("arguments").or_else(|| call.get("parameters")) {
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
}

/// Whether `body` could still become one JSON value with more text: it
/// parses up to its end and stops there unfinished.
fn is_json_prefix(body: &str) -> bool {
    let body = body.trim();
    !body.is_empty() && serde_json::from_str::<Value>(body).is_err_and(|error| error.is_eof())
}

/// A parameter's value between its tags: the templates put one newline after
/// the opening tag and one before the closing tag, and those are not part of
/// the value. A value with no such framing is trimmed.
fn parameter_text(body: &str) -> &str {
    match body
        .strip_prefix('\n')
        .and_then(|rest| rest.strip_suffix('\n'))
    {
        Some(framed) => framed,
        None => body.trim(),
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

/// The length of the longest suffix of `text` that begins `tag`.
fn partial_tag_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len().min(text.len() + 1))
        .rev()
        .find(|&len| {
            text.is_char_boundary(text.len() - len) && tag.starts_with(&text[text.len() - len..])
        })
        .unwrap_or(0)
}

/// Parse a whole answer with `parser`: its text (trimmed when it made calls,
/// `None` when that leaves nothing) and its calls.
pub fn parse_answer(mut parser: ToolCallParser, text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut parsed = parser.push(text);
    parsed.extend(parser.finish());
    let (content, calls) = parsed.into_calls();
    if calls.is_empty() {
        return (Some(content), calls);
    }
    let content = content.trim();
    ((!content.is_empty()).then(|| content.to_string()), calls)
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
