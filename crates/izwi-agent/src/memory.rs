use crate::errors::Result;
use crate::session::ToolCallRecord;
use crate::tools::ToolCall;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMessageRole {
    System,
    User,
    Assistant,
    /// A tool's result, answering the call named by `tool_call_id`.
    Tool,
}

impl MemoryMessageRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMessageMeta {
    pub model_id: Option<String>,
    pub tokens_generated: Option<usize>,
    pub generation_time_ms: Option<f64>,
    /// The calls the turn made before an assistant reply, with what came
    /// of them, for a store that keeps them.
    pub tool_calls: Vec<ToolCallRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryMessage {
    pub role: MemoryMessageRole,
    pub content: String,
    /// The calls an assistant message made.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The call a tool message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl MemoryMessage {
    pub fn new(role: MemoryMessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MemoryMessageRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MemoryMessageRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MemoryMessageRole::Assistant, content)
    }

    /// An assistant message that called tools.
    pub fn assistant_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            tool_calls,
            ..Self::assistant(content)
        }
    }

    /// The result of the call `call_id`.
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(call_id.into()),
            ..Self::new(MemoryMessageRole::Tool, content)
        }
    }
}

/// A thread's stored conversation: user and assistant messages. A store may
/// keep the calls a reply made ([`MemoryMessageMeta::tool_calls`]) and load
/// them back as the assistant and tool messages the turn's rounds read
/// ([`turn_messages`]); one that does not lets them live for their turn only.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn load_messages(&self, thread_id: &str) -> Result<Vec<MemoryMessage>>;

    async fn append_message(
        &self,
        thread_id: &str,
        role: MemoryMessageRole,
        content: String,
        meta: MemoryMessageMeta,
    ) -> Result<()>;

    /// A summary of the thread's messages older than the newest
    /// `kept_messages`, when the store keeps one.
    async fn load_summary(
        &self,
        _thread_id: &str,
        _kept_messages: usize,
    ) -> Result<Option<String>> {
        Ok(None)
    }
}

/// The most of one call's output a stored turn keeps.
pub const MAX_REMEMBERED_OUTPUT_CHARS: usize = 1_500;

/// A stored assistant reply and the calls made before it, as the messages
/// its turn's rounds read: for each round that called tools, the assistant
/// message that made the calls and a tool message for each result, then the
/// reply's text. Calls that were cancelled are left out; a long output is
/// cut short.
pub fn turn_messages(reply: &str, calls: &[ToolCallRecord]) -> Vec<MemoryMessage> {
    let mut messages: Vec<MemoryMessage> = Vec::new();
    let mut calls: Vec<&ToolCallRecord> = calls
        .iter()
        .filter(|call| call.status != "cancelled")
        .collect();
    calls.sort_by_key(|call| call.round);
    for round in calls.chunk_by(|a, b| a.round == b.round) {
        messages.push(MemoryMessage::assistant_calls(
            "",
            round
                .iter()
                .map(|call| ToolCall {
                    id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect(),
        ));
        for call in round {
            let mut output: String = call
                .output
                .chars()
                .take(MAX_REMEMBERED_OUTPUT_CHARS)
                .collect();
            if output.len() < call.output.len() {
                output.push_str("\n[result cut short]");
            }
            messages.push(MemoryMessage::tool(call.call_id.clone(), output));
        }
    }
    if !reply.trim().is_empty() {
        messages.push(MemoryMessage::assistant(reply));
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, round: usize, status: &str, output: &str) -> ToolCallRecord {
        ToolCallRecord {
            call_id: id.to_string(),
            name: "lookup".to_string(),
            arguments: "{}".to_string(),
            status: status.to_string(),
            output: output.to_string(),
            round,
        }
    }

    #[test]
    fn a_stored_turn_reads_back_as_its_rounds() {
        let long = "x".repeat(MAX_REMEMBERED_OUTPUT_CHARS + 1);
        let messages = turn_messages(
            "It is in stock.",
            &[
                call("a", 0, "ok", "12 left"),
                call("b", 0, "cancelled", "The turn was interrupted."),
                call("c", 1, "error", &long),
            ],
        );
        let shape: Vec<_> = messages
            .iter()
            .map(|message| (message.role, message.tool_calls.len()))
            .collect();
        assert_eq!(
            shape,
            [
                (MemoryMessageRole::Assistant, 1),
                (MemoryMessageRole::Tool, 0),
                (MemoryMessageRole::Assistant, 1),
                (MemoryMessageRole::Tool, 0),
                (MemoryMessageRole::Assistant, 0),
            ]
        );
        assert_eq!(messages[1].content, "12 left");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("a"));
        assert!(messages[3].content.ends_with("[result cut short]"));
        assert_eq!(messages[4].content, "It is in stock.");

        // A reply with no words still keeps what it looked up.
        assert_eq!(
            turn_messages(" ", &[call("a", 0, "ok", "12 left")]).len(),
            2
        );
        assert_eq!(turn_messages("Hi.", &[]), [MemoryMessage::assistant("Hi.")]);
    }
}
