use thiserror::Error;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid agent: {0}")]
    InvalidAgent(String),
    #[error("Memory error: {0}")]
    Memory(String),
    #[error("Tool error: {0}")]
    Tool(String),
    /// The reply kept calling tools without answering.
    #[error("{0}")]
    ToolLoop(String),
    #[error("Model error: {0}")]
    Model(String),
}
