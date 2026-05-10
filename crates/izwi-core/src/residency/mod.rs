//! Explicit residency tracking for models loaded into runtime memory.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
    lease_counts: Arc<Mutex<HashMap<ModelVariant, usize>>>,
}

#[derive(Debug)]
pub struct ModelResidencyLease {
    variant: ModelVariant,
    lease_counts: Arc<Mutex<HashMap<ModelVariant, usize>>>,
    active: bool,
}

impl ModelResidencyLease {
    fn new(variant: ModelVariant, lease_counts: Arc<Mutex<HashMap<ModelVariant, usize>>>) -> Self {
        {
            let mut counts = lease_counts
                .lock()
                .expect("model residency lease counts lock poisoned");
            *counts.entry(variant).or_insert(0) += 1;
        }

        Self {
            variant,
            lease_counts,
            active: true,
        }
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }
}

impl Drop for ModelResidencyLease {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;

        let mut counts = self
            .lease_counts
            .lock()
            .expect("model residency lease counts lock poisoned");
        let Some(count) = counts.get_mut(&self.variant) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            counts.remove(&self.variant);
        }
    }
}

impl ModelResidency {
    pub fn acquire_lease(&self, variant: ModelVariant) -> ModelResidencyLease {
        ModelResidencyLease::new(variant, self.lease_counts.clone())
    }

    pub fn active_leases(&self, variant: ModelVariant) -> usize {
        self.lease_counts
            .lock()
            .expect("model residency lease counts lock poisoned")
            .get(&variant)
            .copied()
            .unwrap_or(0)
    }

    pub fn has_active_leases(&self, variant: ModelVariant) -> bool {
        self.active_leases(variant) > 0
    }

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

    #[test]
    fn residency_leases_count_active_model_use_until_drop() {
        let residency = ModelResidency::default();
        assert_eq!(residency.active_leases(ModelVariant::Kokoro82M), 0);

        let first = residency.acquire_lease(ModelVariant::Kokoro82M);
        assert_eq!(first.variant(), ModelVariant::Kokoro82M);
        assert_eq!(residency.active_leases(ModelVariant::Kokoro82M), 1);
        assert!(residency.has_active_leases(ModelVariant::Kokoro82M));

        {
            let _second = residency.acquire_lease(ModelVariant::Kokoro82M);
            assert_eq!(residency.active_leases(ModelVariant::Kokoro82M), 2);
        }

        assert_eq!(residency.active_leases(ModelVariant::Kokoro82M), 1);
        drop(first);
        assert_eq!(residency.active_leases(ModelVariant::Kokoro82M), 0);
        assert!(!residency.has_active_leases(ModelVariant::Kokoro82M));
    }
}
