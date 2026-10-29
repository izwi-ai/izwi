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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// One leading system message holding every system message's text, in
/// order, with the rest of the conversation unchanged. Chat templates that
/// accept a system message only first (Qwen3.5/3.6/3.8) get this instead of
/// a rejected request when a client sends instructions mid-conversation.
pub fn fold_system_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let folded_needed = messages
        .iter()
        .enumerate()
        .any(|(index, message)| index > 0 && message.role == ChatRole::System);
    if !folded_needed {
        return messages;
    }
    let (system, conversation): (Vec<_>, Vec<_>) = messages
        .into_iter()
        .partition(|message| message.role == ChatRole::System);
    let content = system
        .into_iter()
        .map(|message| message.content)
        .filter(|content| !content.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    std::iter::once(ChatMessage {
        role: ChatRole::System,
        content,
    })
    .chain(conversation)
    .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatMediaKind {
    Image,
    Video,
    /// A speech clip (LFM3-Audio), its `source` a `data:audio/...;base64,`
    /// URL.
    Audio,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMediaInput {
    pub kind: ChatMediaKind,
    pub source: String,
}

/// Where a message's text holds each of its audio clips, in order: the
/// token LFM3-Audio's template writes for an audio content part.
pub const CHAT_AUDIO_PLACEHOLDER: &str = "<|audio|>";

/// The most audio clips one chat request may carry.
pub const CHAT_MAX_AUDIO_CLIPS: usize = 8;

/// The longest audio clip a chat request may carry, in seconds.
pub const CHAT_MAX_AUDIO_CLIP_SECONDS: u32 = 120;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: ChatRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
        }
    }

    #[test]
    fn extra_system_messages_join_the_leading_one() {
        use ChatRole::{Assistant, System, User};
        let folded = fold_system_messages(vec![
            message(System, "Be brief."),
            message(User, "Hi"),
            message(Assistant, "Hello"),
            message(System, "Answer in French."),
            message(User, "Name a planet"),
        ]);
        assert_eq!(
            folded,
            vec![
                message(System, "Be brief.\n\nAnswer in French."),
                message(User, "Hi"),
                message(Assistant, "Hello"),
                message(User, "Name a planet"),
            ]
        );
        // A system message after the first turn and none before it.
        assert_eq!(
            fold_system_messages(vec![message(User, "Hi"), message(System, "Be brief.")]),
            vec![message(System, "Be brief."), message(User, "Hi")]
        );
    }

    #[test]
    fn a_conversation_already_in_shape_is_unchanged() {
        use ChatRole::{System, User};
        let messages = vec![message(System, "Be brief."), message(User, "Hi")];
        assert_eq!(fold_system_messages(messages.clone()), messages);
        let messages = vec![message(User, "Hi")];
        assert_eq!(fold_system_messages(messages.clone()), messages);
    }
}
