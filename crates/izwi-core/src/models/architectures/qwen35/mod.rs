//! Qwen3.5 family implementations.

pub mod chat;
mod text;
mod vision;

pub use vision::{media_resource_estimate, Qwen35MediaResourceEstimate};
