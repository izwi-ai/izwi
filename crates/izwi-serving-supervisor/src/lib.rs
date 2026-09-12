//! Accelerator-free primitives for assigning and launching local Izwi workers.
//!
//! This crate deliberately does not start child processes. It validates a bounded
//! node configuration, resolves it against an injected host inventory and binary
//! catalog, produces deterministic child launch specifications, and owns the
//! advisory lock primitives used to fence supervisors and workers.

mod config;
mod launch;
mod locks;

pub use config::*;
pub use launch::*;
pub use locks::*;
