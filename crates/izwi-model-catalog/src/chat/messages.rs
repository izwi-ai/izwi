//! Chat message and request types shared by every chat entry path.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::catalog::ChatReasoningEffort;

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
