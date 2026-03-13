//! Shared chat message types across text-chat model families.

use serde::{Deserialize, Serialize};

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

#[cfg(test)]
mod tests {
    use super::ChatGenerationConfig;

    #[test]
    fn chat_generation_config_default_is_greedy() {
        let config = ChatGenerationConfig::default();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.presence_penalty, 0.0);
        assert!(config.stop_token_ids.is_empty());
        assert_eq!(config.seed, 0);
    }
}
