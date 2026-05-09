//! Explicit residency tracking for models loaded into runtime memory.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::catalog::{ModelInfo, ModelStatus, ModelVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ModelResidencyState {
    #[default]
    NotResident,
    Loading,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArtifactState {
    Missing,
    Downloading,
    Available,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLifecycleSnapshot {
    pub variant: ModelVariant,
    pub artifact_state: ModelArtifactState,
    pub residency_state: ModelResidencyState,
    pub local_path: Option<PathBuf>,
    pub download_progress: Option<f32>,
    pub error_message: Option<String>,
}

impl ModelLifecycleSnapshot {
    pub fn from_model_info(info: ModelInfo) -> Self {
        let artifact_state = match info.status {
            ModelStatus::NotDownloaded => ModelArtifactState::Missing,
            ModelStatus::Downloading => ModelArtifactState::Downloading,
            ModelStatus::Downloaded | ModelStatus::Loading | ModelStatus::Ready => {
                ModelArtifactState::Available
            }
            ModelStatus::Error => ModelArtifactState::Error,
        };

        let residency_state = match info.status {
            ModelStatus::Loading => ModelResidencyState::Loading,
            ModelStatus::Ready => ModelResidencyState::Ready,
            ModelStatus::NotDownloaded
            | ModelStatus::Downloading
            | ModelStatus::Downloaded
            | ModelStatus::Error => ModelResidencyState::NotResident,
        };

        Self {
            variant: info.variant,
            artifact_state,
            residency_state,
            local_path: info.local_path,
            download_progress: info.download_progress,
            error_message: info.error_message,
        }
    }
}

#[derive(Debug, Default)]
pub struct ModelResidency {
    states: RwLock<HashMap<ModelVariant, ModelResidencyState>>,
}

impl ModelResidency {
    pub async fn state(&self, variant: ModelVariant) -> ModelResidencyState {
        self.states
            .read()
            .await
            .get(&variant)
            .copied()
            .unwrap_or(ModelResidencyState::NotResident)
    }

    pub async fn mark_loading(&self, variant: ModelVariant) {
        self.states
            .write()
            .await
            .insert(variant, ModelResidencyState::Loading);
    }

    pub async fn mark_ready(&self, variant: ModelVariant) {
        self.states
            .write()
            .await
            .insert(variant, ModelResidencyState::Ready);
    }

    pub async fn clear(&self, variant: ModelVariant) {
        self.states.write().await.remove(&variant);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_snapshot_splits_downloaded_from_resident() {
        let mut info = ModelInfo::new(ModelVariant::Kokoro82M);
        info.status = ModelStatus::Downloaded;

        let snapshot = ModelLifecycleSnapshot::from_model_info(info);

        assert_eq!(snapshot.artifact_state, ModelArtifactState::Available);
        assert_eq!(snapshot.residency_state, ModelResidencyState::NotResident);
    }

    #[test]
    fn lifecycle_snapshot_maps_ready_to_artifact_and_residency_ready() {
        let mut info = ModelInfo::new(ModelVariant::Kokoro82M);
        info.status = ModelStatus::Ready;

        let snapshot = ModelLifecycleSnapshot::from_model_info(info);

        assert_eq!(snapshot.artifact_state, ModelArtifactState::Available);
        assert_eq!(snapshot.residency_state, ModelResidencyState::Ready);
    }
}
