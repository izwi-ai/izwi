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
/// The supervisor process behind the binary.
pub mod daemon;
pub mod env;
pub mod gateway_ack;
mod launch;
mod lifecycle;
pub mod probe;
mod quarantine;
pub mod rollout;
pub mod signals;

pub use config::*;
pub use izwi_serving_fence::*;
pub use launch::*;
pub use lifecycle::*;
pub use quarantine::*;
