//! Accelerator-free primitives for assigning and launching local Izwi workers.
//!
//! It validates a bounded node configuration, resolves it against an injected
//! host inventory and binary catalog, produces deterministic child launch
//! specifications, and supervises worker processes without initializing an
//! inference backend in the supervisor.

mod config;
mod launch;
mod lifecycle;
mod locks;

pub use config::*;
pub use launch::*;
pub use lifecycle::*;
pub use locks::*;
