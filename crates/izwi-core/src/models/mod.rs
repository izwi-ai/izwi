//! Native model implementations and registry.
//!
//! Canonical module layout:
//! - `architectures::*` for model-family implementations
//! - `shared::*` for reusable runtime/model utilities
//! - `registry` for loaded native model handles

pub mod architectures;
pub mod registry;
pub mod shared;

pub use registry::ModelRegistry;
pub use crate::backends::{DeviceProfile, DeviceSelector};
