//! Legacy compatibility surface for model metadata and artifact management.
//!
//! Prefer `crate::catalog` for model metadata/parsing and `crate::artifacts`
//! for downloads, manager state, and weights.

pub mod download;
mod manager;
mod residency;
pub mod weights;

pub use crate::catalog::{ModelInfo, ModelStatus, ModelVariant};
pub use download::{DownloadProgress, ModelDownloader};
pub use manager::ModelManager;
pub use residency::{ModelResidency, ModelResidencyState};
pub use weights::ModelWeights;
