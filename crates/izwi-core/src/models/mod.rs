//! Internal implementation backing the canonical `crate::runtime_models` API.
//!
//! Canonical module layout:
//! - `architectures::*` for model-family implementations
//! - `shared::*` for reusable runtime/model utilities
//! - `registry` for loaded native model handles

pub mod architectures;
pub mod registry;
pub mod shared;

pub use crate::backends::DeviceSelector;
pub use registry::ModelRegistry;
