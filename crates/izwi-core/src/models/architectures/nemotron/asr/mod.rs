//! Nemotron 3.5 ASR artifact and native inference support.
//!
//! Phase 1 intentionally exposes only artifact extraction and config inventory.
//! The FastConformer-RNNT network loader is added in later phases.

pub mod config;
pub mod nemo;

pub use config::NemotronConfigInventory;
pub use nemo::{ensure_nemotron_artifacts, NemotronArtifacts, NEMOTRON_NEMO_FILENAME};
