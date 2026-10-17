//! Request conversions shared by the OpenAI chat completions and Responses
//! routes: message content, and past tool calls and tool results in the
//! prompt format of the model family that reads them.

use izwi_model_catalog::{ChatMessage, ChatRole, ModelFamily, ModelVariant};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::lfm_tools;
use crate::app::chat_content::{flatten_content_parts, FlattenedMultimodalContent};
use crate::error::ApiError;

/// Message content: a string or a list of typed parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InboundContent {
    Text(String),
    Parts(Vec<InboundContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InboundContentPart {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub input_text: Option<String>,
    #[serde(default)]
    pub image_url: Option<Value>,
    #[serde(default)]
    pub input_image: Option<Value>,
    #[serde(default)]
    pub image: Option<Value>,
    #[serde(default)]
    pub video: Option<Value>,
    #[serde(default)]
    pub video_url: Option<Value>,
    #[serde(default)]
    pub input_video: Option<Value>,
}

/// Flatten message content into prompt text plus the media it references.
pub fn flatten_content(
    content: Option<InboundContent>,
) -> Result<FlattenedMultimodalContent, ApiError> {
    match content {
        None => Ok(FlattenedMultimodalContent::default()),
        Some(InboundContent::Text(text)) => Ok(FlattenedMultimodalContent {
            display_text: text.clone(),
            runtime_text: text,
            media_inputs: Vec::new(),
        }),
        Some(InboundContent::Parts(parts)) => {
            let values = parts
                .iter()
                .map(|part| {
                    serde_json::to_value(part).map_err(|err| ApiError::internal(err.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            flatten_content_parts(&values).map_err(ApiError::bad_request)
        }
    }
}

/// Append an assistant turn's past `tool_calls` to its text, in the format
/// the model family writes them.
pub fn append_assistant_tool_calls(
    variant: ModelVariant,
    content: &mut String,
    tool_calls: &[Value],
) -> Result<(), ApiError> {
    if tool_calls.is_empty() {
        return Ok(());
    }
    if variant.family() == ModelFamily::Lfm2Chat {
        // The upstream template writes the calls right after the text.
        content.push_str(&lfm_tools::render_tool_calls(tool_calls)?);
        return Ok(());
    }
    let rendered = render_qwen_tool_calls(tool_calls)?;
    if !content.trim().is_empty() {
        content.push_str("\n\n");
    }
    content.push_str(&rendered);
    Ok(())
}

/// A tool result turn. LFM2.5 has a native `tool` role; the Qwen templates
/// read results as a user turn wrapped in `<tool_response>`.
pub fn tool_result_message(variant: ModelVariant, content: String) -> ChatMessage {
    if variant.family() == ModelFamily::Lfm2Chat {
        ChatMessage {
            role: ChatRole::Tool,
            content,
        }
    } else {
        ChatMessage {
            role: ChatRole::User,
            content: format!("<tool_response>\n{}\n</tool_response>", content.trim()),
        }
    }
}

/// Render OpenAI `tool_calls` as Qwen `<tool_call>` blocks.
fn render_qwen_tool_calls(tool_calls: &[Value]) -> Result<String, ApiError> {
    let mut rendered = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        let function = tool_call.get("function").unwrap_or(tool_call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::bad_request("Each assistant tool call must include `function.name`")
            })?;

        let args = normalize_tool_call_arguments(function.get("arguments"));
        let mut block = String::new();
        block.push_str("<tool_call>\n");
        block.push_str(&format!("<function={name}>\n"));
        let mut keys: Vec<&String> = args.keys().collect();
        keys.sort();
        for key in keys {
            block.push_str(&format!("<parameter={key}>\n"));
            block.push_str(&tool_parameter_text(&args[key]));
            block.push_str("\n</parameter>\n");
        }
        block.push_str("</function>\n</tool_call>");
        rendered.push(block);
    }
    Ok(rendered.join("\n"))
}

/// OpenAI sends `arguments` as a JSON-encoded object; anything else is kept
/// under an `arguments` key.
fn normalize_tool_call_arguments(raw: Option<&Value>) -> Map<String, Value> {
    let wrap = |value: Value| Map::from_iter([("arguments".to_string(), value)]);
    match raw {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::Object(map)) => map,
            Ok(value) => wrap(value),
            Err(_) => wrap(Value::String(raw.clone())),
        },
        Some(value) => wrap(value.clone()),
    }
}

fn tool_parameter_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn part(kind: &str) -> InboundContentPart {
        InboundContentPart {
            kind: Some(kind.to_string()),
            text: None,
            input_text: None,
            image_url: None,
            input_image: None,
            image: None,
            video: None,
            video_url: None,
            input_video: None,
        }
    }

    #[test]
    fn flattens_text_parts_content() {
        let flattened = flatten_content(Some(InboundContent::Parts(vec![
            InboundContentPart {
                text: Some("hello".to_string()),
                ..part("text")
            },
            InboundContentPart {
                input_text: Some("world".to_string()),
                ..part("input_text")
            },
        ])))
        .expect("flatten content");

        assert_eq!(flattened.runtime_text, "helloworld");
        assert!(!flattened.has_media());
    }

    #[test]
    fn flattens_multimodal_parts_without_inlining_tokens() {
        let flattened = flatten_content(Some(InboundContent::Parts(vec![
            InboundContentPart {
                text: Some("Before ".to_string()),
                ..part("text")
            },
            InboundContentPart {
                image_url: Some(json!({"url":"https://example.com/cat.png"})),
                ..part("image_url")
            },
            InboundContentPart {
                video: Some(json!({"url":"https://example.com/clip.mp4"})),
                ..part("video")
            },
            InboundContentPart {
                text: Some(" after".to_string()),
                ..part("text")
            },
        ])))
        .expect("flatten content");

        assert_eq!(flattened.display_text, "Before  after");
        assert!(flattened.runtime_text.contains("<|image_pad|>"));
        assert!(flattened.runtime_text.contains("<|video_pad|>"));
        assert_eq!(flattened.media_inputs.len(), 2);
    }

    #[test]
    fn flatten_content_rejects_multimodal_part_without_source() {
        let err = flatten_content(Some(InboundContent::Parts(vec![InboundContentPart {
            image_url: Some(json!({})),
            ..part("image_url")
        }])))
        .expect_err("missing source should fail");

        assert!(err.message.contains("missing a usable"));
    }

    #[test]
    fn renders_assistant_tool_calls_to_qwen_xml() {
        let mut content = String::new();
        append_assistant_tool_calls(
            ModelVariant::Qwen359BGguf,
            &mut content,
            &[json!({
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Harare\",\"unit\":\"celsius\"}"
                }
            })],
        )
        .expect("tool call xml should render");

        assert!(content.starts_with("<tool_call>"));
        assert!(content.contains("<function=get_weather>"));
        assert!(content.contains("<parameter=city>\nHarare\n</parameter>"));
        assert!(content.contains("<parameter=unit>\ncelsius\n</parameter>"));
    }

    #[test]
    fn lfm25_tool_turns_use_the_lfm_wire_format() {
        let call =
            json!({"function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}});
        let mut lfm = String::new();
        append_assistant_tool_calls(
            ModelVariant::Lfm2512BInstructGguf,
            &mut lfm,
            std::slice::from_ref(&call),
        )
        .unwrap();
        assert_eq!(
            lfm,
            "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"
        );
        let result = tool_result_message(ModelVariant::Lfm2512BInstructGguf, "21".into());
        assert_eq!(
            (result.role, result.content.as_str()),
            (ChatRole::Tool, "21")
        );

        // Other families keep the Qwen tool format.
        let mut qwen = String::from("Checking.");
        append_assistant_tool_calls(ModelVariant::Qwen314BGguf, &mut qwen, &[call]).unwrap();
        assert!(qwen.starts_with("Checking.\n\n<tool_call>"));
        let result = tool_result_message(ModelVariant::Qwen314BGguf, " 21 ".into());
        assert_eq!(result.role, ChatRole::User);
        assert_eq!(result.content, "<tool_response>\n21\n</tool_response>");
    }
}
