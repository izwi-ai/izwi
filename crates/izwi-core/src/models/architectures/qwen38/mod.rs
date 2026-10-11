//! Qwen3.8 text-chat runtime.
//!
//! Qwen3.8 uses the upstream Qwen3.5 tensor architecture, but owns its loader,
//! graph, cache state, and chat behavior so the two product families can be
//! optimized independently.

mod cache;
pub mod chat;
mod fast_path;
pub(crate) mod mtp;
pub mod native;
pub(crate) mod residency;
mod telemetry;
mod text;
