use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::info;

use crate::backends::BackendKind;
use crate::engine::{
    ReservationClass, ReservationOwner, ResourceAmount, ResourceLease, ResourceVector,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn select_lru_eviction_candidate(
    resident_variants: &[ModelVariant],
    requested_variant: ModelVariant,
    active_variants: &HashSet<ModelVariant>,
    last_used: &HashMap<ModelVariant, u64>,
) -> Option<ModelVariant> {
    resident_variants
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

fn residency_budget_has_capacity(
    resident_variants: &[ModelVariant],
    requested_variant: ModelVariant,
    max_loaded_models: usize,
) -> bool {
    resident_variants.contains(&requested_variant) || resident_variants.len() < max_loaded_models
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

    async fn ensure_model_budget_before_load(&self, requested_variant: ModelVariant) -> Result<()> {
        let Some(max_loaded_models) = self.max_loaded_models else {
            return Ok(());
        };

        loop {
            let resident_variants = self.model_manager.resident_variants().await;
            if residency_budget_has_capacity(
                &resident_variants,
                requested_variant,
                max_loaded_models,
            ) {
                return Ok(());
            }

            let mut active_variants = self.core_engine.active_model_variants().await;
            active_variants.extend(
                resident_variants
                    .iter()
                    .copied()
                    .filter(|variant| self.active_model_residency_leases(*variant) > 0),
            );
            let mut ready_variants = Vec::with_capacity(resident_variants.len());
            for variant in &resident_variants {
                if self.model_manager.is_ready(*variant).await {
                    ready_variants.push(*variant);
                }
            }
            let last_used = self.model_last_used.lock().await.clone();
            let Some(victim) = select_lru_eviction_candidate(
                &ready_variants,
                requested_variant,
                &active_variants,
                &last_used,
            ) else {
                return Err(Error::ModelLoadError(format!(
                    "Cannot load {requested_variant}: the {max_loaded_models}-model residency budget is full and no resident model is idle and ready for eviction"
                )));
            };

            info!(
                requested_variant = %requested_variant,
                victim = %victim,
                max_loaded_models,
                "Evicting idle model before loading its replacement"
            );
            self.unload_model(victim).await?;
        }
    }

    fn model_resource_estimate(&self, variant: ModelVariant) -> ResourceVector {
        let bytes = (variant.memory_required_gb() as f64 * 1024_f64.powi(3)).ceil() as u64;
        let mut resources = ResourceVector::zero();
        match self.backend_router.context().backend_kind {
            BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(bytes),
            BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(bytes),
            BackendKind::Cuda => resources.device_bytes = ResourceAmount::Known(bytes),
        }
        resources
    }

    async fn reserve_model_resources(
        &self,
        requested_variant: ModelVariant,
    ) -> Result<ResourceLease> {
        loop {
            match self.coordinator.resource_authority().reserve(
                ReservationOwner::new(ReservationClass::Model, requested_variant.to_string()),
                self.model_resource_estimate(requested_variant),
            ) {
                Ok(lease) => return Ok(lease),
                Err(resource_error @ Error::Overloaded(_)) => {
                    let resident_variants = self.model_manager.resident_variants().await;
                    let mut active_variants = self.core_engine.active_model_variants().await;
                    active_variants.extend(
                        resident_variants
                            .iter()
                            .copied()
                            .filter(|variant| self.active_model_residency_leases(*variant) > 0),
                    );
                    let mut ready_variants = Vec::new();
                    for variant in &resident_variants {
                        if self.model_manager.is_ready(*variant).await {
                            ready_variants.push(*variant);
                        }
                    }
                    let last_used = self.model_last_used.lock().await.clone();
                    let Some(victim) = select_lru_eviction_candidate(
                        &ready_variants,
                        requested_variant,
                        &active_variants,
                        &last_used,
                    ) else {
                        return Err(Error::ModelLoadError(format!(
                            "Cannot reserve memory for {requested_variant}: {resource_error}"
                        )));
                    };
                    info!(
                        requested_variant = %requested_variant,
                        victim = %victim,
                        "Evicting idle model to satisfy the physical memory budget"
                    );
                    self.unload_model(victim).await?;
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub(crate) async fn load_model_for_inference(
        &self,
        variant: ModelVariant,
    ) -> Result<crate::model::ModelResidencyLease> {
        let _load_guard = self.model_load_lock.lock().await;
        if self.model_manager.is_ready(variant).await {
            self.touch_model_usage(variant).await;
            return Ok(self.acquire_model_residency_lease(variant));
        }

        let _coordinator_load = self
            .coordinator
            .begin_model_load(format!("model-load:{variant}"))?;
        let resolved = self.resolve_model_load(variant).await?;
        let acquired = self.acquire_model_artifacts(resolved).await?;

        self.ensure_model_budget_before_load(variant).await?;
        let resource_lease = self.reserve_model_resources(variant).await?;
        let instantiated = self.instantiate_model(acquired).await?;
        self.publish_loaded_model(instantiated).await?;
        self.model_resource_leases
            .lock()
            .await
            .insert(variant, resource_lease);
        self.touch_model_usage(variant).await;

        Ok(self.acquire_model_residency_lease(variant))
    }

    /// Load a model without retaining an inference pin.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        drop(self.load_model_for_inference(variant).await?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{residency_budget_has_capacity, select_lru_eviction_candidate};
    use crate::backends::BackendPreference;
    use crate::config::EngineConfig;
    use crate::error::Error;
    use crate::model::ModelVariant;
    use crate::runtime::service::RuntimeService;
    use std::collections::{HashMap, HashSet};
    use uuid::Uuid;

    #[test]
    fn select_lru_eviction_candidate_skips_requested_and_active_models() {
        let resident_variants = vec![
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ModelVariant::Qwen38BGguf,
            ModelVariant::Kokoro82M,
        ];
        let requested_variant = ModelVariant::Kokoro82M;
        let active_variants = HashSet::from([ModelVariant::Qwen38BGguf]);
        let last_used = HashMap::from([
            (ModelVariant::Qwen3Tts12Hz06BCustomVoice, 10_u64),
            (ModelVariant::Qwen38BGguf, 5_u64),
            (ModelVariant::Kokoro82M, 20_u64),
        ]);

        let candidate = select_lru_eviction_candidate(
            &resident_variants,
            requested_variant,
            &active_variants,
            &last_used,
        );

        assert_eq!(candidate, Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice));
    }

    #[test]
    fn residency_budget_requires_space_before_loading_a_replacement() {
        let resident_variants = vec![ModelVariant::Kokoro82M];

        assert!(!residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Qwen38BGguf,
            1,
        ));
        assert!(residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Kokoro82M,
            1,
        ));
        assert!(residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Qwen38BGguf,
            2,
        ));
    }

    #[tokio::test]
    async fn residency_budget_evicts_granite_before_loading_another_asr_model() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-residency-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let mut runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime.max_loaded_models = Some(1);
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;

        runtime
            .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
            .await
            .unwrap();

        assert!(runtime.model_manager.resident_variants().await.is_empty());
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn cold_model_load_is_rejected_before_artifact_work_during_drain() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-drain-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime.coordinator.begin_drain();

        assert!(matches!(
            runtime
                .load_model_for_inference(ModelVariant::Kokoro82M)
                .await,
            Err(Error::Overloaded(_))
        ));
        assert!(runtime.model_manager.resident_variants().await.is_empty());

        std::fs::remove_dir_all(models_dir).unwrap();
    }
}
