//! LFM2.5 tool-call wire format.
//!
//! LFM2.5 emits tool calls as a Pythonic list between two marker tokens,
//! optionally followed by text:
//!
//! ```text
//! <|tool_call_start|>[get_weather(city='Paris'), get_time()]<|tool_call_end|>
//! ```
//!
//! Past calls are rendered back into the assistant turn the same way, as
//! Liquid's updated chat template does (`format_arg_value`: strings in single
//! quotes, mappings and lists as JSON, other scalars as Python `str`). On
//! request the model can also write the list as JSON objects
//! (`[{"name": ..., "arguments": {...}}]`); the parser accepts both.

use serde_json::{Map, Value};

use crate::error::ApiError;

pub(super) const TOOL_CALL_START: &str = "<|tool_call_start|>";
pub(super) const TOOL_CALL_END: &str = "<|tool_call_end|>";

/// One parsed call: the function name and its arguments object.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LfmToolCall {
    pub(super) name: String,
    pub(super) arguments: Map<String, Value>,
}

/// Render OpenAI `tool_calls` as the LFM2.5 marker block.
pub(super) fn render_tool_calls(tool_calls: &[Value]) -> Result<String, ApiError> {
    let mut calls = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        let function = tool_call.get("function").unwrap_or(tool_call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::bad_request("Each assistant tool call must include `function.name`")
            })?;
        let arguments = call_arguments(function.get("arguments"))?;
        let rendered = arguments
            .iter()
            .map(|(key, value)| format!("{key}={}", python_argument(value)))
            .collect::<Vec<_>>()
            .join(", ");
        calls.push(format!("{name}({rendered})"));
    }
    Ok(format!(
        "{TOOL_CALL_START}[{}]{TOOL_CALL_END}",
        calls.join(", ")
    ))
}

/// OpenAI sends `arguments` as a JSON-encoded string; the template needs the
/// mapping. Empty, `{}` and `null` mean no arguments.
fn call_arguments(raw: Option<&Value>) -> Result<Map<String, Value>, ApiError> {
    let invalid = || ApiError::bad_request("Tool call `arguments` must be a JSON object");
    match raw {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(Value::String(text)) => {
            let text = text.trim();
            if text.is_empty() || text == "null" {
                return Ok(Map::new());
            }
            match serde_json::from_str::<Value>(text).map_err(|_| invalid())? {
                Value::Object(map) => Ok(map),
                Value::Null => Ok(Map::new()),
                _ => Err(invalid()),
            }
        }
        Some(_) => Err(invalid()),
    }
}

/// The template's `format_arg_value`.
fn python_argument(value: &Value) -> String {
    match value {
        Value::String(text) => {
            let escaped = text
                .replace('\\', "\\\\")
                .replace('\'', "\\'")
                .replace('\n', "\\n")
                .replace('\r', "\\r");
            format!("'{escaped}'")
        }
        Value::Array(_) | Value::Object(_) => {
            let mut out = String::new();
            write_python_json(value, &mut out);
            out
        }
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        Value::Number(number) => number.to_string(),
    }
}

/// Python `json.dumps(value, ensure_ascii=False)` spacing.
pub(super) fn write_python_json(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push_str(": ");
                write_python_json(item, out);
            }
            out.push('}');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

/// Parse the call list between the markers: Pythonic calls, or a JSON
/// list of `{"name", "arguments"}` objects. `None` when it is neither.
pub(super) fn parse_call_list(body: &str) -> Option<Vec<LfmToolCall>> {
    if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(body.trim()) {
        return items.iter().map(json_call).collect();
    }
    let mut parser = Parser::new(body);
    let calls = parser.call_list()?;
    parser.skip_ws();
    parser.at_end().then_some(calls)
}

fn json_call(item: &Value) -> Option<LfmToolCall> {
    let name = item.get("name")?.as_str()?.to_string();
    let arguments = match item.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text).ok()? {
            Value::Object(map) => map,
            _ => return None,
        },
        Some(_) => return None,
    };
    Some(LfmToolCall { name, arguments })
}

/// Recursive-descent parser for a Pythonic call list.
struct Parser {
    chars: Vec<char>,
    at: usize,
}

impl Parser {
    fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            at: 0,
        }
    }

    fn at_end(&self) -> bool {
        self.at >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.at += 1;
        }
    }

    fn eat(&mut self, expected: char) -> bool {
        self.skip_ws();
        if self.peek() == Some(expected) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn identifier(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '-'))
        {
            self.at += 1;
        }
        (self.at > start).then(|| self.chars[start..self.at].iter().collect())
    }

    fn call_list(&mut self) -> Option<Vec<LfmToolCall>> {
        if !self.eat('[') {
            return None;
        }
        let mut calls = Vec::new();
        if self.eat(']') {
            return Some(calls);
        }
        loop {
            calls.push(self.call()?);
            if self.eat(']') {
                return Some(calls);
            }
            if !self.eat(',') {
                return None;
            }
        }
    }

    fn call(&mut self) -> Option<LfmToolCall> {
        let name = self.identifier()?;
        if !self.eat('(') {
            return None;
        }
        let mut arguments = Map::new();
        if self.eat(')') {
            return Some(LfmToolCall { name, arguments });
        }
        loop {
            let key = self.identifier()?;
            if !self.eat('=') {
                return None;
            }
            let value = self.value()?;
            arguments.insert(key, value);
            if self.eat(')') {
                return Some(LfmToolCall { name, arguments });
            }
            if !self.eat(',') {
                return None;
            }
        }
    }

    fn value(&mut self) -> Option<Value> {
        self.skip_ws();
        match self.peek()? {
            '\'' | '"' => self.string().map(Value::String),
            '[' => {
                self.at += 1;
                let mut items = Vec::new();
                if self.eat(']') {
                    return Some(Value::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    if self.eat(']') {
                        return Some(Value::Array(items));
                    }
                    if !self.eat(',') {
                        return None;
                    }
                }
            }
            '{' => {
                self.at += 1;
                let mut map = Map::new();
                if self.eat('}') {
                    return Some(Value::Object(map));
                }
                loop {
                    self.skip_ws();
                    let key = match self.value()? {
                        Value::String(key) => key,
                        _ => return None,
                    };
                    if !self.eat(':') {
                        return None;
                    }
                    map.insert(key, self.value()?);
                    if self.eat('}') {
                        return Some(Value::Object(map));
                    }
                    if !self.eat(',') {
                        return None;
                    }
                }
            }
            _ => self.scalar(),
        }
    }

    fn string(&mut self) -> Option<String> {
        let quote = self.peek()?;
        self.at += 1;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.at += 1;
            if c == quote {
                return Some(out);
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let escaped = self.peek()?;
            self.at += 1;
            out.push(match escaped {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                'u' => {
                    let hex: String = self.chars.get(self.at..self.at + 4)?.iter().collect();
                    self.at += 4;
                    char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?
                }
                other => other,
            });
        }
    }

    fn scalar(&mut self) -> Option<Value> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '+' | '_'))
        {
            self.at += 1;
        }
        let token: String = self.chars[start..self.at].iter().collect();
        match token.as_str() {
            "True" | "true" => Some(Value::Bool(true)),
            "False" | "false" => Some(Value::Bool(false)),
            "None" | "null" => Some(Value::Null),
            number => serde_json::from_str::<serde_json::Number>(number)
                .ok()
                .map(Value::Number),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_openai_tool_calls_like_the_upstream_template() {
        let calls = [
            json!({"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}),
            json!({"function": {"name": "flags", "arguments": {"on": true, "tags": ["a", "b"], "x": null, "q": "it's\nok"}}}),
            json!({"function": {"name": "now", "arguments": "{}"}}),
        ];
        assert_eq!(
            render_tool_calls(&calls).unwrap(),
            "<|tool_call_start|>[get_weather(city='Paris', days=2), \
             flags(on=True, q='it\\'s\\nok', tags=[\"a\", \"b\"], x=None), now()]<|tool_call_end|>"
        );
        assert!(render_tool_calls(&[json!({"function": {"arguments": "{}"}})]).is_err());
        assert!(
            render_tool_calls(&[json!({"function": {"name": "f", "arguments": "[1]"}})]).is_err()
        );
    }

    #[test]
    fn parses_pythonic_calls() {
        let calls = parse_call_list(
            "[get_weather(city=\"Paris\", days=2), \
             set(on=True, ratio=-0.5, tags=['a', 'b'], meta={'k': None})]",
        )
        .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            Value::Object(calls[0].arguments.clone()),
            json!({"city": "Paris", "days": 2})
        );
        assert_eq!(
            Value::Object(calls[1].arguments.clone()),
            json!({"on": true, "ratio": -0.5, "tags": ["a", "b"], "meta": {"k": null}})
        );
    }

    #[test]
    fn parses_json_calls_and_empty_argument_lists() {
        let calls = parse_call_list("[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]").unwrap();
        assert_eq!(calls[0].name, "f");
        let calls = parse_call_list("[now()]").unwrap();
        assert_eq!(calls[0].name, "now");
        assert!(calls[0].arguments.is_empty());
    }

    #[test]
    fn malformed_call_lists_do_not_parse() {
        for body in ["[f(a=)]", "not a list", "[f(a=1)] trailing"] {
            assert!(parse_call_list(body).is_none(), "{body}");
        }
    }

    #[test]
    fn rendered_calls_parse_back_to_the_same_arguments() {
        let call = json!({"function": {"name": "f", "arguments": {"s": "a'b\\c\nd", "n": 3, "l": [1, {"k": "v"}]}}});
        let rendered = render_tool_calls(&[call]).unwrap();
        let body = rendered
            .strip_prefix(TOOL_CALL_START)
            .and_then(|rest| rest.strip_suffix(TOOL_CALL_END))
            .unwrap();
        let calls = parse_call_list(body).unwrap();
        assert_eq!(
            Value::Object(calls[0].arguments.clone()),
            json!({"s": "a'b\\c\nd", "n": 3, "l": [1, {"k": "v"}]})
        );
    }
}
