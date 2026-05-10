//! Compatibility facade for the Metal KV-cache subsystem.
//!
//! New engine-internal code should prefer `crate::engine::cache::metal`.

pub use super::cache::metal::*;
