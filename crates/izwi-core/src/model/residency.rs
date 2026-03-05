//! Explicit residency tracking for models loaded into runtime memory.

use std::collections::HashMap;

use tokio::sync::RwLock;

use crate::catalog::ModelVariant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelResidencyState {
    #[default]
    NotResident,
    Loading,
    Ready,
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
