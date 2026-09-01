//! Shared model infrastructure used by all architecture implementations.
//!
//! This module holds reusable components that should not depend on any
//! specific model family.

pub mod attention;
pub mod chat;
pub mod config;
pub mod memory;
pub mod sampling;
pub mod speculative_sampling;
pub(crate) mod state;
pub mod telemetry;
pub mod weights;
