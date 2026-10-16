//! Accelerator-free primitives for assigning and launching local Izwi workers.
//!
//! It validates a bounded node configuration, resolves it against an injected
//! host inventory and binary catalog, produces deterministic child launch
//! specifications, and supervises worker processes without initializing an
//! inference backend in the supervisor.

pub mod actor;
pub mod approvals_file;
pub mod autoscale;
mod config;
pub mod env;
mod launch;
mod lifecycle;
mod locks;
pub mod probe;
mod quarantine;
pub mod rollout;
pub mod signals;

pub use config::*;
pub use launch::*;
pub use lifecycle::*;
pub use locks::*;
pub use quarantine::*;
