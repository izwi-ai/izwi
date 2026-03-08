//! Shared chat message types across text-chat model families.

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::GenerationParams;
use crate::model::ModelVariant;

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
    pub presence_penalty: f32,
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
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
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

pub fn qwen35_effective_enable_thinking(
    variant: ModelVariant,
    messages: &[ChatMessage],
) -> Option<bool> {
    let default = match variant {
        ModelVariant::Qwen3508B
        | ModelVariant::Qwen352B
        | ModelVariant::Qwen354B
        | ModelVariant::Qwen359B => false,
        _ => return None,
    };

    let override_value = messages.iter().filter_map(|message| {
        if matches!(message.role, ChatRole::System) {
            parse_qwen35_thinking_control_content(&message.content)
        } else {
            None
        }
    });

    Some(override_value.last().unwrap_or(default))
}

pub fn qwen35_sampling_defaults(
    variant: ModelVariant,
    enable_thinking: bool,
) -> Option<Qwen35SamplingDefaults> {
    match variant {
        ModelVariant::Qwen3508B
        | ModelVariant::Qwen352B
        | ModelVariant::Qwen354B
        | ModelVariant::Qwen359B => {
            if enable_thinking {
                Some(Qwen35SamplingDefaults {
                    // Qwen3.5 official guidance for thinking mode.
                    temperature: 0.6,
                    top_p: 0.95,
                    top_k: 20,
                    repetition_penalty: 1.0,
                    presence_penalty: 0.0,
                })
            } else {
                Some(Qwen35SamplingDefaults {
                    // Qwen3.5 official guidance for non-thinking mode.
                    temperature: 0.7,
                    top_p: 0.8,
                    top_k: 20,
                    repetition_penalty: 1.0,
                    presence_penalty: 0.0,
                })
            }
        }
        _ => None,
    }
}

pub fn qwen35_recommended_generation_params(
    variant: ModelVariant,
    messages: &[ChatMessage],
    max_tokens: usize,
) -> Option<GenerationParams> {
    let enable_thinking = qwen35_effective_enable_thinking(variant, messages)?;
    let defaults = qwen35_sampling_defaults(variant, enable_thinking)?;
    let mut params = GenerationParams::default();
    params.temperature = defaults.temperature;
    params.top_p = defaults.top_p;
    params.top_k = defaults.top_k;
    params.repetition_penalty = defaults.repetition_penalty;
    params.presence_penalty = defaults.presence_penalty;
    params.max_tokens = max_tokens.max(1);
    Some(params)
}

#[cfg(test)]
mod tests {
    use super::{
        parse_qwen35_multimodal_control_content, parse_qwen35_thinking_control_content,
        parse_qwen35_tools_control_content, qwen35_multimodal_control_content,
        qwen35_recommended_generation_params, qwen35_sampling_defaults,
        qwen35_thinking_control_content, qwen35_tools_control_content, ChatGenerationConfig,
        ChatMessage, ChatRole, Qwen35MultimodalInput, Qwen35MultimodalKind,
    };
    use crate::model::ModelVariant;
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
        assert_eq!(config.presence_penalty, 0.0);
        assert!(config.stop_token_ids.is_empty());
        assert_eq!(config.seed, 0);
    }

    #[test]
    fn qwen35_small_model_defaults_to_non_thinking_sampling() {
        let params = qwen35_recommended_generation_params(
            ModelVariant::Qwen352B,
            &[ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            64,
        )
        .expect("qwen3.5 params");

        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.8);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.presence_penalty, 0.0);
        assert_eq!(params.max_tokens, 64);
    }

    #[test]
    fn qwen35_large_model_defaults_to_non_thinking_sampling() {
        let params = qwen35_recommended_generation_params(
            ModelVariant::Qwen354B,
            &[ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            32,
        )
        .expect("qwen3.5 params");

        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.8);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.presence_penalty, 0.0);
        assert_eq!(params.max_tokens, 32);
    }

    #[test]
    fn qwen35_thinking_control_switches_large_model_to_thinking_defaults() {
        let params = qwen35_recommended_generation_params(
            ModelVariant::Qwen354B,
            &[
                ChatMessage {
                    role: ChatRole::System,
                    content: qwen35_thinking_control_content(true),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "hello".to_string(),
                },
            ],
            16,
        )
        .expect("qwen3.5 params");

        assert_eq!(params.temperature, 0.6);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.presence_penalty, 0.0);
    }

    #[test]
    fn qwen35_sampling_defaults_ignore_non_qwen35_variants() {
        assert!(qwen35_sampling_defaults(ModelVariant::Qwen306B, false).is_none());
        assert!(qwen35_recommended_generation_params(ModelVariant::Qwen306B, &[], 8).is_none());
    }
}
