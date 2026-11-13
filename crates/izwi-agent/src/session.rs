use crate::tools::ToolCall;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    /// The agent that holds the conversation.
    pub agent_id: String,
    pub thread_id: String,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnInput {
    pub text: String,
}

/// One call a turn made and what came of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    /// `ok`, `error`, `timeout`, `cancelled` or `started` (a background
    /// tool).
    pub status: String,
    pub output: String,
    /// The model round of the turn that made the call.
    #[serde(default)]
    pub round: usize,
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnFinish {
    Completed,
    /// The reply stopped at its token limit.
    MaxTokens,
    /// The reply called tools the caller runs; they are the result's
    /// `pending_calls`.
    PendingCalls,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AgentEvent {
    TurnStarted {
        session_id: String,
        thread_id: String,
    },
    /// A piece of the reply's text. Sent to the turn's listener only; a
    /// turn's result does not repeat them.
    TextDelta {
        text: String,
    },
    /// A model round began. Sent to the turn's listener only, like the
    /// text deltas.
    ModelRoundStarted {
        round: usize,
    },
    /// A model round ended with this many tokens and calls. Listener only.
    ModelRoundCompleted {
        round: usize,
        tokens_generated: usize,
        tool_calls: usize,
    },
    ToolCallStarted {
        call_id: String,
        name: String,
        arguments: String,
        /// Who runs the call: a tool's [`crate::Tool::executor`], or `none`
        /// for a tool the turn does not offer.
        executor: String,
    },
    /// The call has run for the turn's soft timeout and is still running:
    /// the moment to tell the user it is taking a while.
    ToolCallSlow {
        call_id: String,
        name: String,
    },
    ToolCallCompleted {
        call_id: String,
        name: String,
        status: String,
        output: String,
    },
    /// A background tool finished after its turn.
    BackgroundToolCompleted {
        call_id: String,
        name: String,
        status: String,
    },
    Handoff {
        from: String,
        to: String,
    },
    TaskCompleted {
        name: String,
        result: serde_json::Value,
    },
    AssistantMessage {
        content: String,
    },
    TurnCompleted {
        session_id: String,
        thread_id: String,
        finish: TurnFinish,
    },
}

/// Where a turn reports what happens while it runs.
#[derive(Debug, Clone, Default)]
pub struct TurnEvents {
    listener: Option<mpsc::UnboundedSender<AgentEvent>>,
}

impl TurnEvents {
    /// A turn nobody listens to.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn new(listener: mpsc::UnboundedSender<AgentEvent>) -> Self {
        Self {
            listener: Some(listener),
        }
    }

    pub fn emit(&self, event: AgentEvent) {
        if let Some(listener) = &self.listener {
            let _ = listener.send(event);
        }
    }

    /// A piece of the reply's text, as it is generated.
    pub fn text_delta(&self, text: &str) {
        if !text.is_empty() {
            self.emit(AgentEvent::TextDelta {
                text: text.to_string(),
            });
        }
    }
}

/// A task a turn recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletedTask {
    pub name: String,
    pub result: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTurnResult {
    /// The reply's text over all its rounds.
    pub assistant_text: String,
    pub model_id: String,
    pub finish: TurnFinish,
    /// The agent that holds the conversation after the turn.
    pub agent_id: String,
    pub tool_calls: Vec<ToolCallRecord>,
    /// Calls to tools the caller runs.
    pub pending_calls: Vec<ToolCall>,
    pub completed_tasks: Vec<CompletedTask>,
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
    /// The turn's events, without its text deltas.
    pub events: Vec<AgentEvent>,
}
