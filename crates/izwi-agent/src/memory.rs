use crate::errors::Result;
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

/// A thread's stored conversation: user and assistant messages. The tool
/// calls of a turn live for that turn only.
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
