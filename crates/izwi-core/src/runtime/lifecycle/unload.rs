use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures::FutureExt;
use tokio::sync::oneshot;

use crate::audio::AudioCodec;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::memory::metal::MetalPoolManager;
use crate::runtime::lifecycle::controller::ModelLifecycleController;
use crate::runtime::service::RuntimeService;

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown model lifecycle panic".to_string()
}

impl ModelLifecycleController {
    async fn remove_registry_and_auxiliary_state(&self, variant: ModelVariant) {
        #[cfg(test)]
        self.wait_at_unload_test_barrier().await;
        #[cfg(test)]
        self.maybe_panic_during_unload_cleanup();

        match variant.family() {
            ModelFamily::ParakeetAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3Asr
            | ModelFamily::VibeVoiceAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::GraniteSpeechAsr
            | ModelFamily::Qwen3ForcedAligner => {
                self.model_registry.unload_asr(variant).await;
            }
            ModelFamily::SortformerDiarization => {
                self.model_registry.unload_diarization(variant).await;
            }
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => {
                self.model_registry.unload_chat(variant).await;
            }
            ModelFamily::Voxtral => {
                self.model_registry.unload_voxtral(variant).await;
            }
            ModelFamily::Lfm25Audio => {
                self.model_registry.unload_audio_chat(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::Qwen3Tts => {
                self.model_registry.unload_qwen_tts(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::KokoroTts => {
                self.model_registry.unload_kokoro(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::VoxtralTts => {
                self.model_registry.unload_voxtral_tts(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::VibeVoiceTts => {
                self.model_registry.unload_vibevoice_tts(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::FishS2Tts => {
                self.model_registry.unload_fish_s2_tts(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::Tokenizer => {
                let mut tokenizer_guard = self.tokenizer.write().await;
                *tokenizer_guard = None;
                drop(tokenizer_guard);
                let mut codec_guard = self.codec.write().await;
                *codec_guard = AudioCodec::new();
            }
        }
    }

    async fn purge_executor_model_cache(&self, variant: ModelVariant) -> Result<()> {
        let release = self.core_engine.purge_model_cache(variant).await;
        if variant.family() == ModelFamily::Qwen35Chat && !release.confirmed {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 cache purge was not confirmed before unloading {variant}"
            )));
        }
        Ok(())
    }

    pub(super) async fn rollback_model_locked(&self, variant: ModelVariant) -> Result<()> {
        let model_instance = self.resident_instance_id(variant);
        let _ = self.core_engine.abort_requests_for_variant(variant).await;
        self.purge_executor_model_cache(variant).await?;
        if let Some(model_instance) = model_instance {
            self.core_engine
                .unload_managed_model_cache(model_instance)
                .await?;
        }
        self.model_manager.unload_model(variant).await?;
        self.remove_registry_and_auxiliary_state(variant).await;
        self.remove_resident_slot(variant);
        self.forget_model_usage(variant).await;
        Ok(())
    }

    pub(super) async fn rollback_model_after_panic_locked(
        &self,
        variant: ModelVariant,
    ) -> Result<()> {
        self.mark_slot_cleanup_required(variant);
        match AssertUnwindSafe(self.rollback_model_locked(variant))
            .catch_unwind()
            .await
        {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.mark_slot_cleanup_required(variant);
                Err(error)
            }
            Err(payload) => {
                self.mark_slot_cleanup_required(variant);
                Err(Error::ModelLoadError(format!(
                    "model rollback task panicked: {}",
                    panic_message(payload)
                )))
            }
        }
    }

    pub(super) async fn unload_model_locked(&self, variant: ModelVariant) -> Result<()> {
        let model_instance = self.resident_instance_id(variant);
        self.begin_unloading_slot(variant)?;
        let _ = self.core_engine.abort_requests_for_variant(variant).await;
        if let Err(error) = self.purge_executor_model_cache(variant).await {
            self.restore_ready_slot_after_failed_unload(variant);
            return Err(error);
        }
        if let Some(model_instance) = model_instance {
            if let Err(error) = self
                .core_engine
                .unload_managed_model_cache(model_instance)
                .await
            {
                self.restore_ready_slot_after_failed_unload(variant);
                return Err(error);
            }
        }

        // Clear externally visible Ready state before removing the physical
        // handle. The authoritative slot remains Unloading and retains its
        // resource lease until the handle is gone.
        if let Err(error) = self.model_manager.unload_model(variant).await {
            self.restore_ready_slot_after_failed_unload(variant);
            return Err(error);
        }

        self.remove_registry_and_auxiliary_state(variant).await;

        let has_other_authoritative_models = self
            .authoritative_resident_variants()
            .into_iter()
            .any(|resident| resident != variant);
        let has_other_manager_models = self
            .model_manager
            .resident_variants()
            .await
            .into_iter()
            .any(|resident| resident != variant);
        if !has_other_authoritative_models && !has_other_manager_models {
            MetalPoolManager::global().clear_all();
        }

        // Dropping the slot is the final step: it releases physical resource
        // accounting only after every published handle has been removed.
        self.remove_resident_slot(variant);
        self.forget_model_usage(variant).await;
        Ok(())
    }

    async fn run_unload(self: Arc<Self>, variant: ModelVariant) -> Result<()> {
        let _mutation_guard = self.mutation_gate.lock().await;
        self.supersede_registered_load_locked(variant, "explicit unload");
        self.unload_model_locked(variant).await
    }

    pub(super) async fn unload_model_detached(
        self: &Arc<Self>,
        variant: ModelVariant,
    ) -> Result<()> {
        let (result_tx, result_rx) = oneshot::channel();
        let controller = self.clone();
        tokio::spawn(async move {
            let result = match AssertUnwindSafe(controller.clone().run_unload(variant))
                .catch_unwind()
                .await
            {
                Ok(result) => result,
                Err(payload) => {
                    let message = panic_message(payload);
                    let _mutation_guard = controller.mutation_gate.lock().await;
                    if let Err(error) = controller.rollback_model_after_panic_locked(variant).await
                    {
                        tracing::error!(model = %variant, %error, "Panicked unload cleanup failed");
                    }
                    Err(Error::ModelLoadError(format!(
                        "model unload task panicked: {message}"
                    )))
                }
            };
            let _ = result_tx.send(result);
        });
        result_rx.await.map_err(|_| {
            Error::ModelLoadError("model unload task ended without a result".to_string())
        })?
    }

    async fn run_unload_all(self: Arc<Self>) -> Result<usize> {
        let _mutation_guard = self.mutation_gate.lock().await;
        self.supersede_all_registered_loads_locked("explicit unload-all");
        let mut variants = self.authoritative_resident_variants();
        variants.extend(self.model_manager.resident_variants().await);
        variants.sort_by_key(|variant| variant.to_string());
        variants.dedup();

        let mut unloaded = 0usize;
        let mut first_error = None;
        for variant in variants {
            match AssertUnwindSafe(self.unload_model_locked(variant))
                .catch_unwind()
                .await
            {
                Ok(Ok(())) => unloaded += 1,
                Ok(Err(error)) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(payload) => {
                    let message = panic_message(payload);
                    if let Err(error) = self.rollback_model_after_panic_locked(variant).await {
                        tracing::error!(
                            model = %variant,
                            %error,
                            "Panicked unload-all variant cleanup failed"
                        );
                    }
                    if first_error.is_none() {
                        first_error = Some(Error::ModelLoadError(format!(
                            "model unload task panicked: {message}"
                        )));
                    }
                }
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(unloaded)
        }
    }

    pub(super) async fn unload_all_models_detached(self: &Arc<Self>) -> Result<usize> {
        let (result_tx, result_rx) = oneshot::channel();
        let controller = self.clone();
        tokio::spawn(async move {
            let result = match AssertUnwindSafe(controller.clone().run_unload_all())
                .catch_unwind()
                .await
            {
                Ok(result) => result,
                Err(payload) => {
                    controller.recover_unloading_slots();
                    Err(Error::ModelLoadError(format!(
                        "model unload-all task panicked: {}",
                        panic_message(payload)
                    )))
                }
            };
            let _ = result_tx.send(result);
        });
        result_rx.await.map_err(|_| {
            Error::ModelLoadError("model unload-all task ended without a result".to_string())
        })?
    }
}

impl RuntimeService {
    /// Unload a model from memory. Once accepted, the operation continues to
    /// completion even if the requesting task or HTTP connection is cancelled.
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        self.model_lifecycle.unload_model_detached(variant).await
    }

    /// Unload every authoritatively resident model from memory.
    pub async fn unload_all_models(&self) -> Result<usize> {
        self.model_lifecycle.unload_all_models_detached().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::CodecConfig;
    use crate::backends::BackendPreference;
    use crate::config::EngineConfig;
    use crate::engine::{
        CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot, ReservationClass,
        ReservationOwner, ResourceAmount, ResourceAuthority, ResourceVector,
    };
    use crate::runtime::lifecycle::controller::ResidentPhase;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Barrier;
    use uuid::Uuid;

    #[derive(Debug)]
    struct TestCapacityProvider;

    impl PhysicalCapacityProvider for TestCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            let capacity = ResourceVector {
                host_bytes: ResourceAmount::Known(1024),
                ..ResourceVector::zero()
            };
            PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            }
        }
    }

    #[tokio::test]
    async fn cancelled_unload_waiter_keeps_lease_until_auxiliary_state_is_removed() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-cancelled-unload-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Qwen3TtsTokenizer12Hz;
        let resources = ResourceVector {
            host_bytes: ResourceAmount::Known(1),
            ..ResourceVector::zero()
        };
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider)));
        let resource_lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "cancelled-unload-test"),
                resources,
            )
            .expect("test resource reservation");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime
            .model_lifecycle
            .finalize_slot_materialization(variant, resources)
            .expect("materialized slot");
        runtime
            .model_lifecycle
            .mark_slot_ready(variant)
            .expect("ready slot");
        runtime.model_manager.mark_loaded(variant).await;
        *runtime.codec.write().await = AudioCodec::with_config(CodecConfig {
            sample_rate: 12_345,
            ..CodecConfig::default()
        });

        let reached = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        runtime
            .model_lifecycle
            .set_unload_test_barriers(reached.clone(), release.clone());
        let controller = runtime.model_lifecycle.clone();
        let waiter = tokio::spawn(async move { controller.unload_model_detached(variant).await });

        reached.wait().await;
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Unloading)
        );
        assert_eq!(authority.snapshot().reservations, 1);
        assert_eq!(runtime.codec.read().await.sample_rate(), 12_345);
        assert!(
            runtime
                .model_lifecycle
                .try_acquire_ready_lease(variant)
                .is_none(),
            "unloading state must reject new inference pins"
        );

        waiter.abort();
        assert!(waiter
            .await
            .expect_err("unload waiter should be cancelled")
            .is_cancelled());
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Unloading),
            "cancelling the waiter must not release the authoritative lease"
        );
        release.wait().await;

        tokio::time::timeout(Duration::from_secs(1), async {
            while runtime.model_lifecycle.resident_phase(variant).is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached unload must finish");

        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(runtime.codec.read().await.sample_rate(), 24_000);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn shutdown_unload_all_cleans_authoritative_loading_slots() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-loading-slot-shutdown-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider)));
        let resource_lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "loading-shutdown-test"),
                ResourceVector {
                    host_bytes: ResourceAmount::Known(1),
                    ..ResourceVector::zero()
                },
            )
            .expect("test resource reservation");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");

        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(authority.snapshot().reservations, 1);
        assert_eq!(
            runtime.unload_all_models().await.expect("shutdown unload"),
            1
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn unload_all_panic_leaves_retryable_cleanup_state_and_retains_lease() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-unload-panic-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let resources = ResourceVector {
            host_bytes: ResourceAmount::Known(1),
            ..ResourceVector::zero()
        };
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider)));
        let resource_lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "unload-panic-test"),
                resources,
            )
            .expect("test resource reservation");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime
            .model_lifecycle
            .finalize_slot_materialization(variant, resources)
            .expect("materialized slot");
        runtime.model_manager.mark_loaded(variant).await;
        runtime
            .model_lifecycle
            .mark_slot_ready(variant)
            .expect("ready slot");

        // The first panic interrupts the normal cleanup; the second interrupts
        // its immediate rollback. The slot must remain accounted and retryable.
        runtime.model_lifecycle.set_unload_test_panics(2);
        let error = runtime
            .unload_all_models()
            .await
            .expect_err("injected unload panic");
        assert!(matches!(error, Error::ModelLoadError(_)));
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::CleanupRequired)
        );
        assert_eq!(authority.snapshot().reservations, 1);
        assert!(
            runtime
                .model_lifecycle
                .try_acquire_ready_lease(variant)
                .is_none(),
            "cleanup-required state must reject new inference pins"
        );

        assert_eq!(runtime.unload_all_models().await.expect("retry cleanup"), 1);
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }
}
