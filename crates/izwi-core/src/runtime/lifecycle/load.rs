use std::collections::{HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use tracing::info;

use crate::backends::BackendKind;
use crate::engine::{
    ReservationClass, ReservationOwner, ResourceAmount, ResourceLease, ResourceVector,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::lifecycle::controller::{
    ModelLifecycleController, SharedLoadFailure, SharedLoadOutcome,
};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelResourcePlan {
    /// Maximum simultaneous memory authorized before physical instantiation.
    load_authorization: ResourceVector,
    /// Long-lived memory retained after publication completes.
    resident_authorization: ResourceVector,
}

fn model_resource_plan(backend: BackendKind, bytes: u64) -> ModelResourcePlan {
    let mut resident_authorization = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resident_authorization.host_bytes = ResourceAmount::Known(bytes);
        }
        BackendKind::Metal => {
            resident_authorization.unified_bytes = ResourceAmount::Known(bytes);
        }
        BackendKind::Cuda => {
            resident_authorization.device_bytes = ResourceAmount::Known(bytes);
        }
    }

    let mut load_authorization = resident_authorization;
    if backend == BackendKind::Cuda {
        // CUDA loaders materialize host-side artifact/tensor state before or
        // while copying the resident weights to the device. Authorize both
        // peaks up front; the host component is shed after publication.
        load_authorization.host_bytes = ResourceAmount::Known(bytes);
    }

    ModelResourcePlan {
        load_authorization,
        resident_authorization,
    }
}

impl ModelLifecycleController {
    pub(super) async fn touch_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.insert(variant, now_unix_millis());
    }

    pub(super) async fn forget_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.remove(&variant);
    }

    async fn known_resident_variants(&self) -> Vec<ModelVariant> {
        let mut variants = self.authoritative_resident_variants();
        variants.extend(self.model_manager.resident_variants().await);
        variants.sort_by_key(|variant| variant.to_string());
        variants.dedup();
        variants
    }

    pub(super) async fn ensure_model_budget_before_load(
        &self,
        requested_variant: ModelVariant,
        max_loaded_models: Option<usize>,
    ) -> Result<()> {
        let Some(max_loaded_models) = max_loaded_models else {
            return Ok(());
        };

        loop {
            let resident_variants = self.known_resident_variants().await;
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
                    .filter(|variant| self.model_manager.active_residency_leases(*variant) > 0),
            );
            let mut ready_variants = Vec::with_capacity(resident_variants.len());
            for variant in &resident_variants {
                if self.resident_phase(*variant)
                    == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                    || self.model_manager.is_ready(*variant).await
                {
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
            self.unload_model_locked(victim).await?;
        }
    }

    fn model_resource_plan(&self, variant: ModelVariant) -> ModelResourcePlan {
        let bytes = (variant.memory_required_gb() as f64 * 1024_f64.powi(3)).ceil() as u64;
        model_resource_plan(self.backend_router.context().backend_kind, bytes)
    }

    async fn reserve_model_resources(
        &self,
        requested_variant: ModelVariant,
        load_authorization: ResourceVector,
    ) -> Result<ResourceLease> {
        loop {
            match self.coordinator.resource_authority().reserve(
                ReservationOwner::new(ReservationClass::Model, requested_variant.to_string()),
                load_authorization,
            ) {
                Ok(lease) => return Ok(lease),
                Err(resource_error @ Error::Overloaded(_)) => {
                    let resident_variants = self.known_resident_variants().await;
                    let mut active_variants = self.core_engine.active_model_variants().await;
                    active_variants.extend(resident_variants.iter().copied().filter(|variant| {
                        self.model_manager.active_residency_leases(*variant) > 0
                    }));
                    let mut ready_variants = Vec::new();
                    for variant in &resident_variants {
                        if self.resident_phase(*variant)
                            == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                            || self.model_manager.is_ready(*variant).await
                        {
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
                    self.unload_model_locked(victim).await?;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn run_load_transaction_locked(
        &self,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
    ) -> Result<()> {
        if self.resident_phase(variant)
            == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
        {
            return Ok(());
        }
        #[cfg(test)]
        self.maybe_panic_during_load();

        let resolved = self.resolve_model_load(variant).await?;
        let acquired = self.acquire_model_artifacts(resolved).await?;

        self.ensure_model_budget_before_load(variant, max_loaded_models)
            .await?;
        let resource_plan = self.model_resource_plan(variant);
        let resource_lease = self
            .reserve_model_resources(variant, resource_plan.load_authorization)
            .await?;
        self.install_loading_slot(variant, resource_lease)?;

        let publication = async {
            // This is the first operation allowed to allocate model tensors;
            // the peak host/device authorization and authoritative Loading slot
            // are both installed above.
            let instantiated = self.instantiate_model(acquired).await?;
            self.publish_loaded_model(instantiated).await?;
            // The physical allocation is now visible to the live provider.
            // Reconcile before Ready publication so it is no longer counted as
            // both pending ledger work and observed backend memory. CUDA drops
            // its transient host-side authorization at this commit point while
            // retaining the immutable device residency authorization.
            self.finalize_slot_materialization(variant, resource_plan.resident_authorization)?;
            // Install the legacy manager projection before the authoritative
            // commit. Inference pins consult the slot, so no caller can observe
            // Ready while this await is still in progress.
            self.model_manager.mark_loaded(variant).await;
            self.mark_slot_ready(variant)?;
            self.touch_model_usage(variant).await;
            Ok(())
        }
        .await;

        if let Err(error) = publication {
            if let Err(rollback_error) = self.rollback_model_locked(variant).await {
                self.mark_slot_cleanup_required(variant);
                tracing::error!(
                    model = %variant,
                    error = %rollback_error,
                    "Model load rollback failed"
                );
            }
            return Err(error);
        }

        Ok(())
    }

    pub(crate) fn spawn_load_transaction(
        self: &Arc<Self>,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
        leader: crate::runtime::lifecycle::controller::LoadLeader,
    ) -> tokio::task::JoinHandle<()> {
        let controller = self.clone();
        tokio::spawn(async move {
            // Publication of both the Ready slot and the shared terminal
            // outcome is one mutation-gated transaction. Explicit unload can
            // neither erase a successful load before waiters are notified nor
            // observe a half-published failure rollback.
            let _mutation_guard = controller.mutation_gate.lock().await;
            if !controller.is_current_load_generation_locked(variant, leader.generation) {
                return;
            }
            let _coordinator_load = match controller
                .coordinator
                .begin_model_load(format!("model-load:{variant}"))
            {
                Ok(load) => load,
                Err(error) => {
                    controller.finish_load_locked(
                        variant,
                        leader.generation,
                        &leader.completion,
                        SharedLoadOutcome::Failed(SharedLoadFailure::from_error(error)),
                    );
                    return;
                }
            };
            let outcome = match AssertUnwindSafe(
                controller.run_load_transaction_locked(variant, max_loaded_models),
            )
            .catch_unwind()
            .await
            {
                Ok(Ok(())) => SharedLoadOutcome::Ready,
                Ok(Err(error)) => SharedLoadOutcome::Failed(SharedLoadFailure::from_error(error)),
                Err(payload) => {
                    let message = if let Some(message) = payload.downcast_ref::<&str>() {
                        (*message).to_string()
                    } else if let Some(message) = payload.downcast_ref::<String>() {
                        message.clone()
                    } else {
                        "unknown model load panic".to_string()
                    };
                    if let Err(error) = controller.rollback_model_after_panic_locked(variant).await
                    {
                        tracing::error!(model = %variant, %error, "Panicked model load rollback failed");
                    }
                    SharedLoadOutcome::Failed(SharedLoadFailure::ModelLoad(format!(
                        "model load task panicked: {message}"
                    )))
                }
            };
            controller.finish_load_locked(variant, leader.generation, &leader.completion, outcome);
        })
    }
}

impl RuntimeService {
    pub(crate) async fn load_model_for_inference(
        &self,
        variant: ModelVariant,
    ) -> Result<crate::model::ModelResidencyLease> {
        loop {
            if let Some(lease) = self.model_lifecycle.try_acquire_ready_lease(variant) {
                self.model_lifecycle.touch_model_usage(variant).await;
                return Ok(lease);
            }

            let (waiter, leader) = self.model_lifecycle.join_or_start_load(variant);
            if let Some(leader) = leader {
                let _load_task = self.model_lifecycle.spawn_load_transaction(
                    variant,
                    self.max_loaded_models,
                    leader,
                );
            }
            waiter.wait().await?;
        }
    }

    /// Load a model without retaining an inference pin.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        drop(self.load_model_for_inference(variant).await?);
        Ok(())
    }

    async fn ensure_model_budget_before_load(&self, requested_variant: ModelVariant) -> Result<()> {
        let _mutation_guard = self.model_lifecycle.mutation_gate.lock().await;
        self.model_lifecycle
            .ensure_model_budget_before_load(requested_variant, self.max_loaded_models)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        model_resource_plan, residency_budget_has_capacity, select_lru_eviction_candidate,
    };
    use crate::backends::{BackendKind, BackendPreference};
    use crate::config::EngineConfig;
    use crate::engine::{
        CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot, ReservationClass,
        ReservationOwner, ResourceAmount, ResourceAuthority, ResourceLease, ResourceVector,
    };
    use crate::error::Error;
    use crate::model::ModelVariant;
    use crate::runtime::lifecycle::controller::{
        ResidentPhase, SharedLoadFailure, SharedLoadOutcome,
    };
    use crate::runtime::service::RuntimeService;
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::{Duration, Instant};
    use tokio::sync::{oneshot, Barrier};
    use uuid::Uuid;

    fn one_byte_host_reservation() -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(1),
            ..ResourceVector::zero()
        }
    }

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

    #[derive(Debug)]
    struct VectorCapacityProvider {
        snapshot: PhysicalCapacitySnapshot,
    }

    impl PhysicalCapacityProvider for VectorCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
        }
    }

    fn all_memory_capacity(bytes: u64) -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(bytes),
            device_bytes: ResourceAmount::Known(bytes),
            unified_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        }
    }

    fn vector_authority(capacity: ResourceVector) -> Arc<ResourceAuthority> {
        Arc::new(ResourceAuthority::new(Arc::new(VectorCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            },
        })))
    }

    fn isolated_resource_lease(key: &str) -> (Arc<ResourceAuthority>, ResourceLease) {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider)));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, key),
                one_byte_host_reservation(),
            )
            .expect("test resource reservation");
        (authority, lease)
    }

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

    #[test]
    fn model_load_resource_plan_authorizes_backend_specific_peaks() {
        let bytes = 64;

        let cpu = model_resource_plan(BackendKind::Cpu, bytes);
        assert_eq!(
            cpu.load_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(bytes),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(cpu.resident_authorization, cpu.load_authorization);

        let metal = model_resource_plan(BackendKind::Metal, bytes);
        assert_eq!(
            metal.load_authorization,
            ResourceVector {
                unified_bytes: ResourceAmount::Known(bytes),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(metal.resident_authorization, metal.load_authorization);

        let cuda = model_resource_plan(BackendKind::Cuda, bytes);
        assert_eq!(
            cuda.load_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(bytes),
                device_bytes: ResourceAmount::Known(bytes),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            cuda.resident_authorization,
            ResourceVector {
                device_bytes: ResourceAmount::Known(bytes),
                ..ResourceVector::zero()
            }
        );
    }

    #[test]
    fn cuda_load_is_rejected_when_only_device_peak_has_capacity() {
        let plan = model_resource_plan(BackendKind::Cuda, 64);
        let capacity = ResourceVector {
            host_bytes: ResourceAmount::Known(63),
            device_bytes: ResourceAmount::Known(64),
            ..ResourceVector::zero()
        };
        let authority = vector_authority(capacity);

        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "cuda-host-peak"),
                plan.load_authorization,
            ),
            Err(Error::Overloaded(_))
        ));
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[tokio::test]
    async fn published_models_retain_only_backend_residency_authorization() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-resource-finalize-test-{}",
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

        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let plan = model_resource_plan(backend, 64);
            let authority = vector_authority(all_memory_capacity(1024));
            let resource_lease = authority
                .reserve(
                    ReservationOwner::new(
                        ReservationClass::Model,
                        format!("{backend:?}-publication"),
                    ),
                    plan.load_authorization,
                )
                .expect("peak load authorization");
            runtime
                .model_lifecycle
                .install_loading_slot(variant, resource_lease)
                .expect("loading slot");

            runtime
                .model_lifecycle
                .finalize_slot_materialization(variant, plan.resident_authorization)
                .expect("publication resource finalization");
            assert_eq!(
                authority.snapshot().reserved,
                plan.resident_authorization,
                "{backend:?} retained the wrong resident authorization"
            );

            assert!(runtime.model_lifecycle.remove_resident_slot(variant));
            assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
        }

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_unload_supersedes_registered_load_before_spawn() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-pre-gate-load-unload-test-{}",
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

        // Register the generation but deliberately delay spawning its detached
        // transaction. This is the exact window where unload used to return
        // before the stale task acquired the gate and published the model.
        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("registered load leader");
        let stale_generation = leader.generation;

        runtime
            .unload_model(variant)
            .await
            .expect("explicit unload supersedes pending load");
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
        }
        let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
            .await
            .expect("superseded waiter timed out")
            .expect_err("superseded load must fail");
        assert!(matches!(
            error,
            Error::Cancelled(message)
                if message.contains("superseded by explicit unload")
        ));

        let stale_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader,
        );
        tokio::time::timeout(Duration::from_secs(1), stale_task)
            .await
            .expect("stale detached load timed out")
            .expect("stale detached load task");

        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 0);

        // Removal of the stale registration must allow a later request to own
        // a new generation rather than coalescing with cancelled work.
        let (retry_waiter, retry_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let retry_leader = retry_leader.expect("new load generation after unload");
        assert_ne!(retry_leader.generation, stale_generation);
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                retry_leader.generation,
                &retry_leader.completion,
                SharedLoadOutcome::Failed(SharedLoadFailure::Cancelled("test cleanup".to_string())),
            );
        }
        assert!(matches!(
            retry_waiter.wait().await,
            Err(Error::Cancelled(_))
        ));

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unload_all_supersedes_every_registered_load_before_spawn() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-pre-gate-load-unload-all-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variants = [ModelVariant::Kokoro82M, ModelVariant::Qwen38BGguf];
        let mut registrations = Vec::new();
        for variant in variants {
            let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
            registrations.push((variant, waiter, leader.expect("registered load leader")));
        }

        assert_eq!(
            runtime
                .unload_all_models()
                .await
                .expect("unload-all supersedes pending loads"),
            0
        );

        for (variant, waiter, leader) in registrations {
            let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
                .await
                .expect("superseded unload-all waiter timed out")
                .expect_err("superseded load must fail");
            assert!(matches!(
                error,
                Error::Cancelled(message)
                    if message.contains("superseded by explicit unload-all")
            ));
            let stale_task = runtime.model_lifecycle.spawn_load_transaction(
                variant,
                runtime.max_loaded_models,
                leader,
            );
            tokio::time::timeout(Duration::from_secs(1), stale_task)
                .await
                .expect("stale unload-all load timed out")
                .expect("stale unload-all task");
            assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
            assert!(!runtime.model_manager.is_ready(variant).await);
        }
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 0);

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ready_outcome_is_published_before_explicit_unload_enters_the_gate() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-load-publication-race-test-{}",
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
        let resources = one_byte_host_reservation();
        let (authority, resource_lease) = isolated_resource_lease("publication-race");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime
            .model_lifecycle
            .finalize_slot_materialization(variant, resources)
            .expect("materialized slot");
        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("load leader");

        let publication_reached = Arc::new(Barrier::new(2));
        let publication_release = Arc::new(Barrier::new(2));
        let events = Arc::new(StdMutex::new(Vec::new()));
        let controller = runtime.model_lifecycle.clone();
        let reached = publication_reached.clone();
        let release = publication_release.clone();
        let publication_events = events.clone();
        let publisher = tokio::spawn(async move {
            let _mutation_guard = controller.mutation_gate.lock().await;
            controller.model_manager.mark_loaded(variant).await;
            controller.mark_slot_ready(variant).expect("ready slot");
            reached.wait().await;
            release.wait().await;
            controller.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
            publication_events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push("outcome");
        });

        publication_reached.wait().await;
        let unload_controller = runtime.model_lifecycle.clone();
        let unload_events = events.clone();
        let mut unload = tokio::spawn(async move {
            unload_controller
                .unload_model_detached(variant)
                .await
                .expect("explicit unload");
            unload_events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push("unload");
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(25), &mut unload)
                .await
                .is_err(),
            "unload must not cross the gate before the load outcome is published"
        );

        publication_release.wait().await;
        waiter.wait().await.expect("shared ready outcome");
        publisher.await.expect("publisher task");
        unload.await.expect("unload task");
        assert_eq!(
            *events.lock().unwrap_or_else(|poison| poison.into_inner()),
            vec!["outcome", "unload"]
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn detached_load_panic_rolls_back_before_publishing_failure() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-panic-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let (authority, resource_lease) = isolated_resource_lease("load-panic");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime.model_manager.mark_loaded(variant).await;
        runtime.model_lifecycle.set_load_test_panics(1);

        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let _load_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader.expect("load leader"),
        );
        let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
            .await
            .expect("panic outcome timed out")
            .expect_err("injected load panic must fail");
        assert!(
            matches!(error, Error::ModelLoadError(message) if message.contains("injected model load panic"))
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(authority.snapshot().reservations, 0);

        let (cleanup_waiter, cleanup_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let cleanup_leader = cleanup_leader.expect("failed generation must be removable");
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                cleanup_leader.generation,
                &cleanup_leader.completion,
                SharedLoadOutcome::Failed(SharedLoadFailure::ModelLoad("test cleanup".to_string())),
            );
        }
        assert!(cleanup_waiter.wait().await.is_err());

        std::fs::remove_dir_all(models_dir).unwrap();
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

    #[tokio::test]
    async fn cancelled_waiter_keeps_shared_load_accounted_and_visible_to_drain() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-cancelled-load-test-{}",
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
        let resources = one_byte_host_reservation();
        let (authority, resource_lease) = isolated_resource_lease("cancelled-load-test");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");

        let (first_waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("first waiter leads the load");
        let (second_waiter, second_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        assert!(second_leader.is_none(), "second waiter must coalesce");

        let first_task = tokio::spawn(first_waiter.wait());
        let second_task = tokio::spawn(second_waiter.wait());
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let loader_calls = Arc::new(AtomicUsize::new(0));
        let controller = runtime.model_lifecycle.clone();
        let calls = loader_calls.clone();
        let transaction = tokio::spawn(async move {
            let _mutation_guard = controller.mutation_gate.lock().await;
            let _coordinator_load = controller
                .coordinator
                .begin_model_load("cancelled-shared-load")
                .expect("model load admission");
            calls.fetch_add(1, Ordering::AcqRel);
            let _ = started_tx.send(());
            let _ = release_rx.await;
            controller
                .finalize_slot_materialization(variant, resources)
                .expect("materialized lease");
            controller.model_manager.mark_loaded(variant).await;
            controller.mark_slot_ready(variant).expect("ready slot");
            controller.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
        });

        started_rx.await.expect("fake load started");
        first_task.abort();
        assert!(first_task
            .await
            .expect_err("first waiter should be cancelled")
            .is_cancelled());
        assert_eq!(loader_calls.load(Ordering::Acquire), 1);
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Loading)
        );
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 1);
        assert_eq!(authority.snapshot().reservations, 1);

        runtime.begin_drain();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(25),
                runtime
                    .coordinator
                    .wait_for_idle(Instant::now() + Duration::from_secs(1)),
            )
            .await
            .is_err(),
            "drain must still observe the detached load"
        );

        release_tx.send(()).expect("release fake load");
        second_task
            .await
            .expect("second waiter join")
            .expect("coalesced waiter succeeds");
        transaction.await.expect("fake load transaction");
        runtime
            .coordinator
            .wait_for_idle(Instant::now() + Duration::from_secs(1))
            .await
            .expect("drain after load completion");
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Ready)
        );
        assert!(runtime.model_manager.is_ready(variant).await);

        assert_eq!(
            runtime.unload_all_models().await.expect("shutdown unload"),
            1
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn failed_publication_rolls_back_slot_before_releasing_lease() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-load-rollback-test-{}",
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
        let (authority, resource_lease) = isolated_resource_lease("rollback-test");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");

        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Loading)
        );
        assert_eq!(authority.snapshot().reservations, 1);
        let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
        runtime
            .model_lifecycle
            .rollback_model_locked(variant)
            .await
            .expect("rollback");

        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }
}
