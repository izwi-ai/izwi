//! Compatibility facade for the engine KV-cache subsystem.
//!
//! New engine-internal code should prefer `crate::engine::cache::kv`.

pub use super::cache::kv::*;
