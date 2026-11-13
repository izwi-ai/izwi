//! The agent turn runtime of Izwi: one streaming, cancellable loop of model
//! rounds and tool calls, with handoffs between agents and structured
//! tasks. It knows nothing of audio; a voice session drives it with text
//! and listens to its events.

pub mod agent;
pub mod cancel;
pub mod engine;
pub mod errors;
pub mod memory;
pub mod model;
pub mod reasoning;
pub mod session;
pub mod tools;

pub use agent::{AgentDefinition, AgentSet, TaskSpec};
pub use cancel::TurnCancel;
pub use engine::{AgentEngine, AgentTurnOptions, HistoryBudget, TurnContext};
pub use errors::{AgentError, Result};
pub use memory::{MemoryMessage, MemoryMessageMeta, MemoryMessageRole, MemoryStore};
pub use model::{ModelBackend, ModelOutput, ModelRequest};
pub use reasoning::{strip_reasoning, ReasoningFilter};
pub use session::{
    AgentEvent, AgentSession, AgentTurnResult, CompletedTask, ToolCallRecord, TurnEvents,
    TurnFinish, TurnInput,
};
pub use tools::{
    function_definition, BackgroundResult, BackgroundResults, TimeTool, Tool, ToolCall,
    ToolContext, ToolExecution, ToolOutput, ToolRegistry,
};
