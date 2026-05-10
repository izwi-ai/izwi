//! Internal implementation backing the canonical `crate::runtime_models` API.
//!
//! Canonical module layout:
//! - `architectures::*` for model-family implementations
//! - `shared::*` for reusable runtime/model utilities
//! - `registry` for loaded native model handles

pub mod architectures;
pub mod families;
pub mod registry;
pub mod shared;

pub use families::{
    model_family_registrations, registration_for_variant, registrations_for_capability,
    FamilyRegistration, MODEL_FAMILY_REGISTRATIONS,
};
pub use registry::{LoadedModelRegistry, ModelRegistry};
