//! Qwen3.5 family implementations.

pub(crate) mod cache;
pub mod chat;
mod fast_path;
pub(crate) mod telemetry;
pub(crate) mod text;
mod vision;

pub use vision::{media_resource_estimate, Qwen35MediaResourceEstimate};
