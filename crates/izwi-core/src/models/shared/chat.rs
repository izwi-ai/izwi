//! Shared chat message types across text-chat model families.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatGenerationFinishReason {
    MaxTokens,
    StopToken,
    StopSequence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatMediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMediaInput {
    pub kind: ChatMediaKind,
    pub source: String,
}

/// Chat-specific request metadata consumed by native backends.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequestConfig {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub media_inputs: Vec<ChatMediaInput>,
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
    pub stop_sequences: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub seed: u64,
    pub request: ChatRequestConfig,
}

impl Default for ChatGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
            seed: 0,
            request: ChatRequestConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatGenerationConfig, ChatRequestConfig};

    #[test]
    fn chat_generation_config_default_is_greedy() {
        let config = ChatGenerationConfig::default();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.presence_penalty, 0.0);
        assert!(config.stop_sequences.is_empty());
        assert!(config.stop_token_ids.is_empty());
        assert_eq!(config.seed, 0);
        assert_eq!(config.request, ChatRequestConfig::default());
    }
}
