//! The chat API every entry path shares: message and request types, what a
//! generation returns, and how a request's sampling and output budget
//! resolve for its model.

mod generation;
mod messages;
pub mod reasoning;
mod sampling;
pub mod tools;

pub use generation::{
    ChatGeneration, FinishReason, LatencyBreakdown, TokenLogprob, TopTokenLogprob,
};
pub use messages::{
    fold_system_messages, ChatMediaInput, ChatMediaKind, ChatMessage, ChatRequestConfig, ChatRole,
    ChatTemplateKwargs, ChatTruncation, CHAT_AUDIO_PLACEHOLDER,
};
pub use sampling::{
    chat_supports_json_object, chat_supports_tools, resolve_chat_max_tokens, resolve_chat_sampling,
    ChatGenerationControls, ChatSamplingRequest, ResolvedChatSampling, CHAT_MAX_TOKENS_UNSET,
};
