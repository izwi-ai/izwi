//! Shared chat message types across text-chat model families.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::catalog::ChatReasoningEffort;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    /// A tool result. Families whose template has a tool turn (LFM2.5)
    /// render it natively; the Qwen and Gemma prompt builders render it as
    /// the user turn their templates use.
    Tool,
}

impl ChatRole {
    pub fn as_prompt_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
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

/// Hugging Face-compatible keyword arguments accepted alongside the direct
/// chat request fields. Server adapters resolve these into `ChatRequestConfig`
/// before native model execution.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatTemplateKwargs {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ChatReasoningEffort>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
}

/// Chat-specific request metadata consumed by native backends.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequestConfig {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ChatReasoningEffort>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub media_inputs: Vec<ChatMediaInput>,
    #[serde(default)]
    pub truncation: ChatTruncation,
}

/// What a chat request does when it does not fit the model's context
/// (OpenAI Responses' `truncation`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatTruncation {
    /// Reject the request with `ContextLengthExceeded` when the prompt does
    /// not fit, or when an explicit `max_tokens` does not fit beside it. An
    /// unset `max_tokens` takes the rest of the context.
    #[default]
    Disabled,
    /// Drop the oldest complete turns until the prompt fits, and shorten
    /// the output budget to the remaining context.
    Auto,
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
    pub request: ChatRequestConfig,
    /// DS9.3: collect per-token logprobs of the raw model distribution.
    pub logprobs: bool,
    /// DS9.3: top alternatives per token (0..=20); only read when `logprobs`.
    pub top_logprobs: usize,
    /// DS9.2: constrain generation to one valid JSON value
    /// (`response_format: json_object`). Only honored by samplers that opt in.
    pub constrain_json_object: bool,
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
            request: ChatRequestConfig::default(),
            logprobs: false,
            top_logprobs: 0,
            constrain_json_object: false,
        }
    }
}

/// Tokens `[start, end)` of a capacity replay's journal: the prompt followed
/// by the generated tokens.
pub(crate) fn replay_journal_span(
    prompt_ids: &[u32],
    generated_ids: &[u32],
    start: usize,
    end: usize,
) -> crate::error::Result<Vec<u32>> {
    if start > end || end > prompt_ids.len() + generated_ids.len() {
        return Err(crate::error::Error::InvalidInput(format!(
            "replay span [{start},{end}) exceeds its {}-token journal",
            prompt_ids.len() + generated_ids.len()
        )));
    }
    Ok(prompt_ids
        .iter()
        .chain(generated_ids)
        .skip(start)
        .take(end - start)
        .copied()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{ChatGenerationConfig, ChatReasoningEffort, ChatRequestConfig, ChatTemplateKwargs};

    #[test]
    fn replay_journal_spans_cross_from_prompt_to_generated_tokens() {
        let prompt = [1, 2, 3];
        let generated = [7, 8];
        assert_eq!(
            super::replay_journal_span(&prompt, &generated, 1, 5).unwrap(),
            [2, 3, 7, 8]
        );
        assert_eq!(
            super::replay_journal_span(&prompt, &generated, 3, 4).unwrap(),
            [7]
        );
        assert!(super::replay_journal_span(&prompt, &generated, 2, 6).is_err());
    }

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
        assert_eq!(config.request, ChatRequestConfig::default());
    }

    #[test]
    fn reasoning_effort_accepts_only_pinned_qwen_values() {
        let kwargs: ChatTemplateKwargs =
            serde_json::from_str(r#"{"reasoning_effort":"xhigh","preserve_thinking":true}"#)
                .expect("supported effort");
        assert_eq!(kwargs.reasoning_effort, Some(ChatReasoningEffort::Xhigh));
        assert_eq!(kwargs.preserve_thinking, Some(true));

        assert!(
            serde_json::from_str::<ChatTemplateKwargs>(r#"{"reasoning_effort":"high"}"#).is_err()
        );
    }
}
