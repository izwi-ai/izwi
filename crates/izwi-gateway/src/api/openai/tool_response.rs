//! Tool results for the LFM3 templates, rendered as their
//! `render_tool_response` does (LiquidAI/LFM3-1.7B @ 40ebe0aa,
//! LFM3-Audio-2B @ 9d49701c). A result becomes the template's
//! `<tool_response>` block, which the prompt renderers pass through.
//!
//! Deferred tools (`defer_loading`) reach the model this way: they stay out
//! of the system turn's `<tools>`, and a tool search's result names them,
//! either as `tool_reference` items (LFM3 chat), `{"result": {"tools":
//! [...]}}` items, or an `additional_tools` key on the tool message. The
//! block lists their definitions in `<additional_tools>`.

use izwi_model_catalog::chat::tools::{python_json, tool_function, tool_to_json};
use izwi_model_catalog::{ModelFamily, ModelVariant};
use serde_json::{json, Value};

use crate::error::ApiError;

const TOOL_RESPONSE_OPEN: &str = "<tool_response>";
const ADDITIONAL_TOOLS_OPEN: &str = "<additional_tools>";

/// Which LFM3 template renders a model's tool results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolResponseDialect {
    /// LFM3-1.7B: text items as text, `tool_reference` runs as one
    /// `<additional_tools>` block, results answering the same call merged.
    Lfm3Chat,
    /// LFM3-Audio-2B: every other item as JSON, no `tool_reference`, no
    /// merging.
    Lfm3Audio,
}

impl ToolResponseDialect {
    pub fn of(variant: ModelVariant) -> Option<Self> {
        match variant.family() {
            ModelFamily::Lfm3Chat => Some(Self::Lfm3Chat),
            ModelFamily::Lfm3Audio => Some(Self::Lfm3Audio),
            _ => None,
        }
    }
}

/// One tool result as the client sent it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolResult<'a> {
    /// A string, or a list of content items.
    pub content: Option<&'a Value>,
    pub additional_tools: Option<&'a Value>,
    pub tool_call_id: Option<&'a str>,
}

/// Renders a request's tool results in order, with the request's tool
/// definitions (deferred ones included) to resolve `tool_reference` items.
pub struct ToolResponses<'a> {
    dialect: ToolResponseDialect,
    tools: &'a [Value],
    pending: Vec<Value>,
}

impl<'a> ToolResponses<'a> {
    pub fn new(dialect: ToolResponseDialect, tools: &'a [Value]) -> Self {
        Self {
            dialect,
            tools,
            pending: Vec::new(),
        }
    }

    /// The tool turn's content for `result`, or `None` when it merges into
    /// `next`, the tool result right after it (LFM3 chat merges results that
    /// answer the same call).
    pub fn render(
        &mut self,
        result: ToolResult<'_>,
        next: Option<ToolResult<'_>>,
    ) -> Result<Option<String>, ApiError> {
        let flag = result.additional_tools.filter(|value| truthy(value));
        if let Some(additional_tools) = flag {
            return self
                .render_additional_tools(additional_tools, result.content)
                .map(Some);
        }
        let merges = self.dialect == ToolResponseDialect::Lfm3Chat
            && next.is_some_and(|next| {
                next.additional_tools
                    .filter(|value| truthy(value))
                    .is_none()
                    && result
                        .tool_call_id
                        .is_some_and(|id| !id.is_empty() && next.tool_call_id == Some(id))
            });
        if merges || !self.pending.is_empty() {
            self.pending.extend(items(result.content));
            if merges {
                return Ok(None);
            }
            let content = Value::Array(std::mem::take(&mut self.pending));
            return self.render_response(&content).map(Some);
        }
        match (self.dialect, result.content) {
            (ToolResponseDialect::Lfm3Chat, None) => self
                .render_response(&Value::String(String::new()))
                .map(Some),
            (ToolResponseDialect::Lfm3Audio, content) => match content.filter(|c| truthy(c)) {
                Some(content) => self.render_response(content).map(Some),
                // The audio template writes an empty tool turn.
                None => Ok(Some(String::new())),
            },
            (_, Some(content)) => self.render_response(content).map(Some),
        }
    }

    /// The template's `additional_tools` branch: definitions in the key
    /// itself, or in the content with the key as a flag.
    fn render_additional_tools(
        &self,
        additional_tools: &Value,
        content: Option<&Value>,
    ) -> Result<String, ApiError> {
        let definitions = match additional_tools {
            Value::Object(_) | Value::Array(_) => {
                if content.is_some_and(truthy) {
                    return Err(ApiError::bad_request(
                        "A tool message lists its tools either in `additional_tools` or in its content, not both",
                    ));
                }
                Some(additional_tools)
            }
            _ => content,
        };
        Ok(match definitions {
            Some(Value::String(text)) if text.contains(TOOL_RESPONSE_OPEN) => text.clone(),
            Some(Value::String(text)) if text.contains(ADDITIONAL_TOOLS_OPEN) => {
                format!("{TOOL_RESPONSE_OPEN}\n{text}\n</tool_response>")
            }
            Some(definitions) if truthy(definitions) => format!(
                "{TOOL_RESPONSE_OPEN}\n{ADDITIONAL_TOOLS_OPEN}\n{}\n</additional_tools>\n</tool_response>",
                tool_lines(definitions)
            ),
            // Nothing to reveal: the template writes an empty turn.
            _ => String::new(),
        })
    }

    fn render_response(&self, content: &Value) -> Result<String, ApiError> {
        let items = match content {
            Value::String(text) if text.contains(TOOL_RESPONSE_OPEN) => return Ok(text.clone()),
            Value::String(text) => {
                return Ok(format!("{TOOL_RESPONSE_OPEN}\n{text}\n</tool_response>"))
            }
            Value::Array(items) => items.as_slice(),
            other => std::slice::from_ref(other),
        };
        let mut out = format!("{TOOL_RESPONSE_OPEN}\n");
        for (index, item) in items.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            if is_tool_reference(item) {
                if self.dialect == ToolResponseDialect::Lfm3Audio {
                    return Err(ApiError::bad_request(
                        "LFM3-Audio does not resolve `tool_reference` items; list the tools' definitions in the tool message's `additional_tools`",
                    ));
                }
                if index == 0 || !is_tool_reference(&items[index - 1]) {
                    out.push_str(ADDITIONAL_TOOLS_OPEN);
                    out.push('\n');
                }
                out.push_str(&self.referenced_tool(item)?);
                if items
                    .get(index + 1)
                    .is_none_or(|next| !is_tool_reference(next))
                {
                    out.push_str("\n</additional_tools>");
                }
            } else if let Some(definitions) = result_tools(item) {
                out.push_str(ADDITIONAL_TOOLS_OPEN);
                out.push('\n');
                out.push_str(&tool_lines(definitions));
                out.push_str("\n</additional_tools>");
            } else {
                match self.dialect {
                    ToolResponseDialect::Lfm3Chat => out.push_str(&parse_content(item)),
                    ToolResponseDialect::Lfm3Audio => out.push_str(&python_json(item)),
                }
            }
        }
        out.push_str("\n</tool_response>");
        Ok(out)
    }

    /// The definition a `tool_reference` names, from the request's tools
    /// (the last match wins, as in the template).
    fn referenced_tool(&self, item: &Value) -> Result<String, ApiError> {
        let name = item.get("name").and_then(Value::as_str).ok_or_else(|| {
            ApiError::bad_request("A `tool_reference` content item needs a `name`")
        })?;
        self.tools
            .iter()
            .rfind(|tool| tool_function(tool).get("name").and_then(Value::as_str) == Some(name))
            .map(tool_to_json)
            .ok_or_else(|| {
                ApiError::bad_request(format!(
                    "Tool reference '{name}' not found in available tools. Every tool referenced by a tool_reference must be in `tools` on every request."
                ))
            })
    }
}

/// Reject the LFM3-only tool-result constructs for a model whose template
/// has none.
pub fn reject_deferred_tool_results(
    variant: ModelVariant,
    result: ToolResult<'_>,
) -> Result<(), ApiError> {
    if ToolResponseDialect::of(variant).is_some() {
        return Ok(());
    }
    let references = result
        .content
        .and_then(Value::as_array)
        .is_some_and(|items| items.iter().any(is_tool_reference));
    if references || result.additional_tools.is_some_and(truthy) {
        return Err(ApiError::bad_request(format!(
            "Deferred tool results (`tool_reference` items, `additional_tools`) need an LFM3 model; {variant} does not read them"
        )));
    }
    Ok(())
}

/// A merged result's items: text as one text item, a list as its items.
fn items(content: Option<&Value>) -> Vec<Value> {
    match content {
        None => Vec::new(),
        Some(Value::String(text)) if text.is_empty() => Vec::new(),
        Some(Value::String(text)) => vec![json!({"type": "text", "text": text})],
        Some(Value::Array(items)) => items.clone(),
        Some(other) => vec![other.clone()],
    }
}

fn is_tool_reference(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("tool_reference")
}

/// `item.result.tools` of a tool search's result item.
fn result_tools(item: &Value) -> Option<&Value> {
    item.get("result")
        .filter(|result| result.is_object())
        .and_then(|result| result.get("tools"))
}

/// Tool definitions one per line: a string as is, a definition or a list
/// of them through `tool_to_json`.
fn tool_lines(definitions: &Value) -> String {
    match definitions {
        Value::String(text) => text.clone(),
        Value::Array(tools) => tools
            .iter()
            .map(tool_to_json)
            .collect::<Vec<_>>()
            .join("\n"),
        definition => tool_to_json(definition),
    }
}

/// LFM3 chat's `parse_content` for one item. Responses' `input_text` reads
/// as text, as everywhere else in izwi.
fn parse_content(item: &Value) -> String {
    match item {
        Value::String(text) => text.clone(),
        Value::Array(items) => items.iter().map(parse_content).collect(),
        Value::Object(map) => match map.get("type").and_then(Value::as_str) {
            Some("image") => "<|vision_start|><|image|><|vision_end|>".to_string(),
            Some("video") => "<|vision_start|><|video|><|vision_end|>".to_string(),
            Some("text" | "input_text") => match map.get("text") {
                Some(Value::String(text)) => text.clone(),
                None | Some(Value::Null) => String::new(),
                Some(other) => python_json(other),
            },
            _ => python_json(item),
        },
        other => python_json(other),
    }
}

/// Jinja truthiness.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

#[cfg(test)]
#[path = "tool_response_tests.rs"]
mod tests;
