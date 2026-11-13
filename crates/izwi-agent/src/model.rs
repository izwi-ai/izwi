use crate::errors::Result;
use crate::memory::MemoryMessage;
use crate::session::TurnEvents;
use crate::tools::ToolCall;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    pub model_id: String,
    pub messages: Vec<MemoryMessage>,
    /// The function tools the model may call, in the chat completions form.
    pub tools: Vec<serde_json::Value>,
    pub max_output_tokens: usize,
    /// The round of the turn this request is, from zero.
    pub round: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOutput {
    /// The reply's text, without the markup of its tool calls.
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
    /// Whether the reply stopped at its token limit.
    #[serde(default)]
    pub truncated: bool,
}

/// One model round. A backend sends the reply's text to `events` as it is
/// generated ([`TurnEvents::text_delta`]) and returns the whole reply with
/// the calls it parsed from it.
#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn generate(&self, request: ModelRequest, events: &TurnEvents) -> Result<ModelOutput>;
}
