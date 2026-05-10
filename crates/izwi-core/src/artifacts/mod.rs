//! Canonical surface for downloaded model artifacts and artifact lifecycle helpers.
//!
//! This module owns the public API for model downloads, cached weights, and
//! artifact-management state. Legacy `crate::model` imports remain available as
//! compatibility shims.

pub mod downloader;

pub use downloader::{DownloadProgress, DownloadState, ModelDownloader};
pub use crate::model::{
    ModelArtifactState, ModelLifecycleSnapshot, ModelManager, ModelResidency, ModelResidencyState,
    ModelWeights,
};
