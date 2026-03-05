//! Canonical surface for downloaded model artifacts and artifact lifecycle helpers.
//!
//! This module owns the public API for model downloads, cached weights, and
//! artifact-management state. Legacy `crate::model` imports remain available as
//! compatibility shims.

pub use crate::model::{
    DownloadProgress, ModelDownloader, ModelManager, ModelResidency, ModelResidencyState,
    ModelWeights,
};
