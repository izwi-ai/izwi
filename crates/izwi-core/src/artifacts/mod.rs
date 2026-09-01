//! Canonical surface for downloaded model artifacts and artifact lifecycle helpers.
//!
//! This module owns the public API for model downloads, cached weights, and
//! artifact-management state. Legacy `crate::model` imports remain available as
//! compatibility shims.

pub mod downloader;

pub use crate::model::{
    ModelArtifactState, ModelLifecycleSnapshot, ModelManager, ModelResidency, ModelResidencyState,
    ModelWeights,
};
pub use downloader::{
    read_artifact_manifest, ArtifactManifest, DownloadProgress, DownloadState, ModelDownloader,
    ARTIFACT_MANIFEST_FILE,
};
