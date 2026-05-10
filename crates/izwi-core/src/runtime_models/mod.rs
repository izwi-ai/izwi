//! Canonical surface for native runtime model implementations and registries.
//!
//! Public callers should prefer this namespace over the internal `models`
//! implementation module.

pub use crate::models::{
    architectures, families, model_family_registrations, registration_for_variant,
    registrations_for_capability, registry, shared, FamilyRegistration, LoadedModelRegistry,
    ModelRegistry, MODEL_FAMILY_REGISTRATIONS,
};
