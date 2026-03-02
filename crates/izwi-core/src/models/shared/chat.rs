//! Shared chat message types across text-chat model families.

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

const QWEN35_THINKING_CONTROL_PREFIX: &str = "__izwi_qwen35_enable_thinking=";
const QWEN35_TOOLS_CONTROL_PREFIX: &str = "__izwi_qwen35_tools_json=";

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

#[cfg(test)]
mod tests {
    use super::{
        parse_qwen35_thinking_control_content, parse_qwen35_tools_control_content,
        qwen35_thinking_control_content, qwen35_tools_control_content,
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
}
