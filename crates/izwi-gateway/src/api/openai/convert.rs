//! Request conversions shared by the OpenAI chat completions and Responses
//! routes: message content, and past tool calls and tool results in the
//! prompt format of the model family that reads them.

use izwi_model_catalog::chat::tools::{self, ToolDialect};
use izwi_model_catalog::{ChatMessage, ChatRole, ModelVariant};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::app::chat_content::{flatten_content_parts, FlattenedMultimodalContent};
use crate::error::ApiError;

/// Message content: a string or a list of typed parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InboundContent {
    Text(String),
    Parts(Vec<InboundContentPart>),
}

/// One content part. Fields this server reads are named; any others (a
/// `tool_reference`'s `name`, a tool search's `result`) are kept in `extra`,
/// so a part serializes back as the client sent it, minus absent fields.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InboundContentPart {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_image: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_url: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_video: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Message content as JSON: a string, or the list of parts.
pub fn inbound_content_value(content: Option<&InboundContent>) -> Result<Option<Value>, ApiError> {
    match content {
        None => Ok(None),
        Some(InboundContent::Text(text)) => Ok(Some(Value::String(text.clone()))),
        Some(InboundContent::Parts(parts)) => serde_json::to_value(parts)
            .map(Some)
            .map_err(|err| ApiError::internal(err.to_string())),
    }
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
/// the model family writes them (the Qwen XML format for a family without a
/// tool template).
pub fn append_assistant_tool_calls(
    variant: ModelVariant,
    content: &mut String,
    tool_calls: &[Value],
) -> Result<(), ApiError> {
    let dialect = ToolDialect::of(variant).unwrap_or(ToolDialect::QwenXml);
    tools::append_assistant_tool_calls(dialect, content, tool_calls).map_err(ApiError::bad_request)
}

/// An assistant turn's text with the `reasoning_content` a client sent back
/// beside it, where the prompt builders read reasoning.
pub fn assistant_content_with_reasoning(reasoning: Option<&str>, content: &str) -> String {
    tools::assistant_content_with_reasoning(reasoning, content)
}

/// A tool result turn. Families with a tool template get a `tool` message,
/// which their prompt builder writes as the template does (a native tool
/// turn, or merged `<tool_response>` blocks in a user turn). Any other
/// family reads it as a user turn wrapped in `<tool_response>`.
pub fn tool_result_message(variant: ModelVariant, content: String) -> ChatMessage {
    if ToolDialect::of(variant).is_some() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn part(kind: &str) -> InboundContentPart {
        InboundContentPart {
            kind: Some(kind.to_string()),
            ..InboundContentPart::default()
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
    fn tool_turns_follow_the_model_family() {
        let call = json!({"function": {"name": "get_weather", "arguments": "{\"unit\":\"c\",\"city\":\"Paris\"}"}});
        let rendered = |variant, text: &str| {
            let mut content = text.to_string();
            append_assistant_tool_calls(variant, &mut content, std::slice::from_ref(&call))
                .unwrap();
            content
        };
        assert_eq!(
            rendered(ModelVariant::Qwen359BGguf, "Checking."),
            "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=unit>\nc\n</parameter>\n\
             <parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(
            rendered(ModelVariant::Qwen314BGguf, ""),
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"unit\": \"c\", \"city\": \"Paris\"}}\n</tool_call>"
        );
        assert_eq!(
            rendered(ModelVariant::Lfm2512BInstructGguf, ""),
            "<|tool_call_start|>[get_weather(unit='c', city='Paris')]<|tool_call_end|>"
        );
        assert!(rendered(ModelVariant::Lfm317BGguf, "Checking.")
            .starts_with("Checking.<tool_call>\n<function=get_weather>"));

        // Families with a tool template get the result as a tool message.
        for variant in [
            ModelVariant::Qwen359BGguf,
            ModelVariant::Qwen314BGguf,
            ModelVariant::Lfm317BGguf,
            ModelVariant::Lfm2512BInstructGguf,
        ] {
            let result = tool_result_message(variant, " 21 ".into());
            assert_eq!(
                (result.role, result.content.as_str()),
                (ChatRole::Tool, " 21 ")
            );
        }
        let result = tool_result_message(ModelVariant::Gemma34BIt, " 21 ".into());
        assert_eq!(result.role, ChatRole::User);
        assert_eq!(result.content, "<tool_response>\n21\n</tool_response>");
    }
}
