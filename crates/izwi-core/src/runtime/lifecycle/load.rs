use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{info, warn};

use crate::error::Result;
use crate::model::ModelStatus;
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn select_lru_eviction_candidate(
    loaded_variants: &[ModelVariant],
    requested_variant: ModelVariant,
    active_variants: &HashSet<ModelVariant>,
    last_used: &HashMap<ModelVariant, u64>,
) -> Option<ModelVariant> {
    loaded_variants
        .iter()
        .copied()
        .filter(|variant| *variant != requested_variant && !active_variants.contains(variant))
        .min_by(|left, right| {
            last_used
                .get(left)
                .copied()
                .unwrap_or(0)
                .cmp(&last_used.get(right).copied().unwrap_or(0))
                .then_with(|| left.to_string().cmp(&right.to_string()))
        })
}

impl RuntimeService {
    pub(super) async fn touch_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.insert(variant, now_unix_millis());
    }

    pub(super) async fn forget_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.remove(&variant);
    }

    async fn evict_idle_model_for_budget(&self, requested_variant: ModelVariant) -> Result<()> {
        let Some(max_loaded_models) = self.max_loaded_models else {
            return Ok(());
        };

        let loaded_variants = self
            .model_manager
            .list_models()
            .await
            .into_iter()
            .filter(|info| matches!(info.status, ModelStatus::Ready))
            .map(|info| info.variant)
            .collect::<Vec<_>>();
        if loaded_variants.len() <= max_loaded_models {
            return Ok(());
        }

        let active_variants = self.core_engine.active_model_variants().await;
        let last_used = self.model_last_used.lock().await.clone();
        let Some(victim) = select_lru_eviction_candidate(
            &loaded_variants,
            requested_variant,
            &active_variants,
            &last_used,
        ) else {
            return Ok(());
        };

        info!(
            requested_variant = %requested_variant,
            victim = %victim,
            max_loaded_models,
            "Evicting idle model to honor residency budget"
        );
        self.unload_model(victim).await
    }

    /// Load a model for inference.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        let resolved = self.resolve_model_load(variant).await?;
        let acquired = self.acquire_model_artifacts(resolved).await?;
        let instantiated = self.instantiate_model(acquired).await?;
        self.publish_loaded_model(instantiated).await?;
        self.touch_model_usage(variant).await;

        if let Err(err) = self.evict_idle_model_for_budget(variant).await {
            warn!(
                current_variant = %variant,
                "Model residency budget eviction failed: {err}"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::select_lru_eviction_candidate;
    use crate::model::ModelVariant;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn select_lru_eviction_candidate_skips_requested_and_active_models() {
        let loaded_variants = vec![
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ModelVariant::Qwen38BGguf,
            ModelVariant::Lfm25Audio15B,
        ];
        let requested_variant = ModelVariant::Lfm25Audio15B;
        let active_variants = HashSet::from([ModelVariant::Qwen38BGguf]);
        let last_used = HashMap::from([
            (ModelVariant::Qwen3Tts12Hz06BCustomVoice, 10_u64),
            (ModelVariant::Qwen38BGguf, 5_u64),
            (ModelVariant::Lfm25Audio15B, 20_u64),
        ]);

        let candidate = select_lru_eviction_candidate(
            &loaded_variants,
            requested_variant,
            &active_variants,
            &last_used,
        );

        assert_eq!(candidate, Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice));
    }
}
