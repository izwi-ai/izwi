//! Shared chat message types across text-chat model families.

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_prompt_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Model-agnostic chat generation controls used by native chat backends.
///
/// The default preserves legacy deterministic greedy decoding for direct callers.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatGenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_token_ids: Vec<u32>,
    pub seed: u64,
}

impl Default for ChatGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            stop_token_ids: Vec::new(),
            seed: 0,
        }
    }
}

const QWEN35_THINKING_CONTROL_PREFIX: &str = "__izwi_qwen35_enable_thinking=";
const QWEN35_TOOLS_CONTROL_PREFIX: &str = "__izwi_qwen35_tools_json=";
const QWEN35_MULTIMODAL_CONTROL_PREFIX: &str = "__izwi_qwen35_multimodal_json=";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35MultimodalKind {
    Image,
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen35MultimodalInput {
    pub kind: Qwen35MultimodalKind,
    pub source: String,
}

/// Internal control marker used by server/runtime to steer Qwen3.5 chat-template
/// thinking mode without exposing implementation details to users.
pub fn qwen35_thinking_control_content(enable_thinking: bool) -> String {
    format!("{QWEN35_THINKING_CONTROL_PREFIX}{enable_thinking}")
}

pub fn parse_qwen35_thinking_control_content(content: &str) -> Option<bool> {
    let raw = content.trim();
    let suffix = raw.strip_prefix(QWEN35_THINKING_CONTROL_PREFIX)?;
    match suffix.trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Internal control marker used by server/runtime to pass Qwen3.5 tool schema
/// definitions to prompt construction without exposing implementation details to users.
pub fn qwen35_tools_control_content(tools: &[Value]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    serde_json::to_string(tools)
        .ok()
        .map(|json| format!("{QWEN35_TOOLS_CONTROL_PREFIX}{json}"))
}

pub fn parse_qwen35_tools_control_content(content: &str) -> Option<Vec<Value>> {
    let raw = content.trim();
    let suffix = raw.strip_prefix(QWEN35_TOOLS_CONTROL_PREFIX)?;
    serde_json::from_str::<Vec<Value>>(suffix).ok()
}

/// Internal control marker used by server/runtime to pass Qwen3.5 image/video
/// source descriptors to native multimodal runtime, aligned with placeholder
/// order in the paired chat message.
pub fn qwen35_multimodal_control_content(items: &[Qwen35MultimodalInput]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let json = serde_json::to_vec(items).ok()?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(json);
    Some(format!("{QWEN35_MULTIMODAL_CONTROL_PREFIX}{encoded}"))
}

pub fn parse_qwen35_multimodal_control_content(
    content: &str,
) -> Option<Vec<Qwen35MultimodalInput>> {
    let raw = content.trim();
    let suffix = raw.strip_prefix(QWEN35_MULTIMODAL_CONTROL_PREFIX)?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(suffix)
        .ok()?;
    serde_json::from_slice::<Vec<Qwen35MultimodalInput>>(&decoded).ok()
}

#[cfg(test)]
mod tests {
    use super::{
        parse_qwen35_multimodal_control_content, parse_qwen35_thinking_control_content,
        parse_qwen35_tools_control_content, qwen35_multimodal_control_content,
        qwen35_thinking_control_content, qwen35_tools_control_content, ChatGenerationConfig,
        Qwen35MultimodalInput, Qwen35MultimodalKind,
    };
    use serde_json::json;

    #[test]
    fn qwen35_control_roundtrip_true() {
        let content = qwen35_thinking_control_content(true);
        assert_eq!(parse_qwen35_thinking_control_content(&content), Some(true));
    }

    #[test]
    fn qwen35_control_roundtrip_false() {
        let content = qwen35_thinking_control_content(false);
        assert_eq!(parse_qwen35_thinking_control_content(&content), Some(false));
    }

    #[test]
    fn qwen35_control_ignores_non_control_text() {
        assert_eq!(
            parse_qwen35_thinking_control_content("You are a helpful assistant."),
            None
        );
    }

    #[test]
    fn qwen35_tools_control_roundtrip() {
        let tools = vec![
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        },
                        "required": ["city"]
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {"type": "object"}
                }
            }),
        ];

        let encoded = qwen35_tools_control_content(&tools).expect("tools marker should encode");
        let decoded =
            parse_qwen35_tools_control_content(&encoded).expect("tools marker should decode");
        assert_eq!(decoded, tools);
    }

    #[test]
    fn qwen35_tools_control_ignores_non_control_text() {
        assert_eq!(
            parse_qwen35_tools_control_content("not a control payload"),
            None
        );
    }

    #[test]
    fn qwen35_multimodal_control_roundtrip() {
        let items = vec![
            Qwen35MultimodalInput {
                kind: Qwen35MultimodalKind::Image,
                source: "https://example.com/cat.png".to_string(),
            },
            Qwen35MultimodalInput {
                kind: Qwen35MultimodalKind::Video,
                source: "https://example.com/clip.mp4".to_string(),
            },
        ];

        let encoded =
            qwen35_multimodal_control_content(&items).expect("multimodal marker should encode");
        let decoded = parse_qwen35_multimodal_control_content(&encoded)
            .expect("multimodal marker should decode");
        assert_eq!(decoded, items);
    }

    #[test]
    fn qwen35_multimodal_control_ignores_non_control_text() {
        assert_eq!(
            parse_qwen35_multimodal_control_content("not a control payload"),
            None
        );
    }

    #[test]
    fn chat_generation_config_defaults_to_legacy_greedy_decode() {
        let config = ChatGenerationConfig::default();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert!(config.stop_token_ids.is_empty());
        assert_eq!(config.seed, 0);
    }
}
