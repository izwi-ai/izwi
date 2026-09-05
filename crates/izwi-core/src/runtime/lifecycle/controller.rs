use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use tokio::sync::Barrier;
use tokio::sync::{watch, Mutex, RwLock};

use crate::artifacts::ModelManager;
use crate::audio::AudioCodec;
use crate::backends::BackendRouter;
use crate::config::EngineConfig;
use crate::engine::{Engine as CoreEngine, ModelInstanceId, ResourceLease, ResourceVector};
use crate::error::{Error, Result};
use crate::model::{ModelResidencyLease, ModelVariant};
use crate::runtime::adapters::{
    CapabilityKind, LoadedModelBundle, LoadedModelBundleDraft, LoadedStatePublication,
    RuntimeAdapterRegistry,
};
use crate::runtime::coordinator::InferenceCoordinator;
use crate::runtime_models::ModelRegistry;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResidentPhase {
    Loading,
    Ready,
    Unloading,
    CleanupRequired,
}

#[derive(Debug)]
struct ResidentSlot {
    phase: ResidentPhase,
    model_instance_id: ModelInstanceId,
    bundle: Option<Arc<LoadedModelBundle>>,
    resource_lease: ResourceLease,
}

#[derive(Debug, Clone)]
pub(super) enum SharedLoadFailure {
    InvalidInput(String),
    ModelNotFound(String),
    Overloaded(String),
    Timeout(String),
    Cancelled(String),
    Config(String),
    ModelLoad(String),
}

impl SharedLoadFailure {
    pub(super) fn from_error(error: Error) -> Self {
        match error {
            Error::InvalidInput(message) => Self::InvalidInput(message),
            Error::ModelNotFound(message) => Self::ModelNotFound(message),
            Error::Overloaded(message) => Self::Overloaded(message),
            Error::Timeout(message) => Self::Timeout(message),
            Error::Cancelled(message) => Self::Cancelled(message),
            Error::ConfigError(message) => Self::Config(message),
            Error::ModelLoadError(message) => Self::ModelLoad(message),
            other => Self::ModelLoad(other.to_string()),
        }
    }

    fn into_error(self) -> Error {
        match self {
            Self::InvalidInput(message) => Error::InvalidInput(message),
            Self::ModelNotFound(message) => Error::ModelNotFound(message),
            Self::Overloaded(message) => Error::Overloaded(message),
            Self::Timeout(message) => Error::Timeout(message),
            Self::Cancelled(message) => Error::Cancelled(message),
            Self::Config(message) => Error::ConfigError(message),
            Self::ModelLoad(message) => Error::ModelLoadError(message),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum SharedLoadOutcome {
    Ready,
    Failed(SharedLoadFailure),
}

#[derive(Debug)]
struct InFlightLoad {
    generation: u64,
    completion: watch::Sender<Option<SharedLoadOutcome>>,
}

#[derive(Debug, Default)]
struct LifecycleState {
    residents: HashMap<ModelVariant, ResidentSlot>,
    loads: HashMap<ModelVariant, InFlightLoad>,
    next_generation: u64,
}

pub(crate) struct LoadWaiter {
    completion: watch::Receiver<Option<SharedLoadOutcome>>,
}

impl LoadWaiter {
    pub(crate) async fn wait(mut self) -> Result<()> {
        loop {
            if let Some(outcome) = self.completion.borrow().clone() {
                return match outcome {
                    SharedLoadOutcome::Ready => Ok(()),
                    SharedLoadOutcome::Failed(error) => Err(error.into_error()),
                };
            }
            self.completion.changed().await.map_err(|_| {
                Error::ModelLoadError(
                    "model load operation ended without a terminal outcome".to_string(),
                )
            })?;
        }
    }
}

pub(crate) struct LoadLeader {
    pub(super) generation: u64,
    pub(super) completion: watch::Sender<Option<SharedLoadOutcome>>,
}

pub(crate) struct ModelLifecycleController {
    pub(super) config: EngineConfig,
    pub(super) backend_router: BackendRouter,
    pub(super) adapter_registry: Arc<RuntimeAdapterRegistry>,
    pub(super) model_manager: Arc<ModelManager>,
    pub(super) model_registry: Arc<ModelRegistry>,
    pub(super) core_engine: Arc<CoreEngine>,
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    pub(super) codec: Arc<RwLock<AudioCodec>>,
    pub(super) loaded_tts_variant: Arc<RwLock<Option<ModelVariant>>>,
    pub(super) realtime_asr_sequence_capacity: u32,
    pub(super) model_last_used: Mutex<HashMap<ModelVariant, u64>>,
    pub(super) mutation_gate: Mutex<()>,
    state: StdMutex<LifecycleState>,
    #[cfg(test)]
    load_test_panics: AtomicUsize,
    #[cfg(test)]
    unload_test_barriers: StdMutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
    #[cfg(test)]
    unload_test_panics: AtomicUsize,
}

impl ModelLifecycleController {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: EngineConfig,
        backend_router: BackendRouter,
        adapter_registry: Arc<RuntimeAdapterRegistry>,
        model_manager: Arc<ModelManager>,
        model_registry: Arc<ModelRegistry>,
        core_engine: Arc<CoreEngine>,
        coordinator: Arc<InferenceCoordinator>,
        tokenizer: Arc<RwLock<Option<Tokenizer>>>,
        codec: Arc<RwLock<AudioCodec>>,
        loaded_tts_variant: Arc<RwLock<Option<ModelVariant>>>,
        realtime_asr_sequence_capacity: u32,
    ) -> Self {
        Self {
            config,
            backend_router,
            adapter_registry,
            model_manager,
            model_registry,
            core_engine,
            coordinator,
            tokenizer,
            codec,
            loaded_tts_variant,
            realtime_asr_sequence_capacity,
            model_last_used: Mutex::new(HashMap::new()),
            mutation_gate: Mutex::new(()),
            state: StdMutex::new(LifecycleState::default()),
            #[cfg(test)]
            load_test_panics: AtomicUsize::new(0),
            #[cfg(test)]
            unload_test_barriers: StdMutex::new(None),
            #[cfg(test)]
            unload_test_panics: AtomicUsize::new(0),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, LifecycleState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    pub(crate) fn try_acquire_ready_lease(
        &self,
        variant: ModelVariant,
    ) -> Option<ModelResidencyLease> {
        let state = self.state();
        let slot = state.residents.get(&variant)?;
        (slot.phase == ResidentPhase::Ready).then(|| {
            self.model_manager
                .acquire_instance_residency_lease(variant, slot.model_instance_id)
        })
    }

    pub(super) fn resident_instance_id(&self, variant: ModelVariant) -> Option<ModelInstanceId> {
        self.state()
            .residents
            .get(&variant)
            .map(|slot| slot.model_instance_id)
    }

    pub(super) fn resident_variant_for_instance(
        &self,
        model_instance_id: ModelInstanceId,
    ) -> Option<ModelVariant> {
        self.state().residents.iter().find_map(|(variant, slot)| {
            (slot.model_instance_id == model_instance_id).then_some(*variant)
        })
    }

    pub(crate) fn try_get_ready_bundle(
        &self,
        variant: ModelVariant,
    ) -> Option<Arc<LoadedModelBundle>> {
        let state = self.state();
        let slot = state.residents.get(&variant)?;
        (slot.phase == ResidentPhase::Ready)
            .then(|| slot.bundle.clone())
            .flatten()
    }

    pub(super) fn resident_phase(&self, variant: ModelVariant) -> Option<ResidentPhase> {
        self.state().residents.get(&variant).map(|slot| slot.phase)
    }

    pub(super) fn authoritative_resident_variants(&self) -> Vec<ModelVariant> {
        let mut variants = self.state().residents.keys().copied().collect::<Vec<_>>();
        variants.sort_by_key(|variant| variant.to_string());
        variants
    }

    pub(super) fn install_loading_slot(
        &self,
        variant: ModelVariant,
        resource_lease: ResourceLease,
    ) -> Result<ModelInstanceId> {
        let mut state = self.state();
        if state.residents.contains_key(&variant) {
            return Err(Error::ModelLoadError(format!(
                "model {variant} already has authoritative residency state"
            )));
        }
        let generation = if let Some(load) = state.loads.get(&variant) {
            load.generation
        } else {
            // Tests and legacy manager projections can install an authoritative
            // slot without first joining the detached-load protocol. They
            // still receive a unique lifecycle instance identity.
            state.next_generation = state.next_generation.wrapping_add(1).max(1);
            state.next_generation
        };
        let model_instance_id = ModelInstanceId::new(generation);
        state.residents.insert(
            variant,
            ResidentSlot {
                phase: ResidentPhase::Loading,
                model_instance_id,
                bundle: None,
                resource_lease,
            },
        );
        Ok(model_instance_id)
    }

    #[cfg(test)]
    pub(super) fn mark_slot_ready(&self, variant: ModelVariant) -> Result<()> {
        let model_instance_id = self.resident_instance_id(variant).ok_or_else(|| {
            Error::ModelLoadError(format!(
                "model {variant} lost its resource lease before publication"
            ))
        })?;
        if self
            .state()
            .residents
            .get(&variant)
            .is_some_and(|slot| slot.bundle.is_none())
        {
            self.bind_loaded_model_bundle(variant, model_instance_id)?;
        }
        self.mark_slot_ready_for_instance(variant, model_instance_id)
    }

    pub(super) fn bind_loaded_model_bundle(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
    ) -> Result<Arc<LoadedModelBundle>> {
        self.bind_loaded_model_bundle_with_state_publications(
            variant,
            model_instance_id,
            HashMap::new(),
        )
    }

    pub(super) fn bind_loaded_model_bundle_with_state_publications(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        state_publications: HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<Arc<LoadedModelBundle>> {
        let draft = self.draft_loaded_model_bundle(variant, model_instance_id)?;
        self.bind_loaded_model_bundle_draft(draft, variant, model_instance_id, state_publications)
    }

    /// Freeze the exact adapter instances and every selectable stage graph
    /// before model-derived physical state is planned or allocated.
    pub(super) fn draft_loaded_model_bundle(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
    ) -> Result<LoadedModelBundleDraft> {
        let state = self.state();
        let slot = state.residents.get(&variant).ok_or_else(|| {
            Error::ModelLoadError(format!(
                "model {variant} lost its resource lease before adapter drafting"
            ))
        })?;
        if slot.phase != ResidentPhase::Loading || slot.model_instance_id != model_instance_id {
            return Err(Error::ModelLoadError(format!(
                "model {variant} changed lifecycle identity before adapter drafting"
            )));
        }
        if slot.bundle.is_some() {
            return Err(Error::ModelLoadError(format!(
                "model {variant} already has a loaded execution bundle"
            )));
        }
        drop(state);
        LoadedModelBundleDraft::build(
            &self.adapter_registry,
            self.coordinator.execution_group_id(),
            model_instance_id,
            variant,
            self.backend_router.context().backend_kind,
        )
    }

    pub(super) fn bind_loaded_model_bundle_draft(
        &self,
        draft: LoadedModelBundleDraft,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        state_publications: HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<Arc<LoadedModelBundle>> {
        let bundle = Arc::new(draft.seal(state_publications)?);
        if bundle.model_instance_id() != model_instance_id || bundle.model_variant() != variant {
            return Err(Error::ModelLoadError(format!(
                "model {variant} draft sealed a different lifecycle identity"
            )));
        }
        let mut state = self.state();
        let slot = state.residents.get_mut(&variant).ok_or_else(|| {
            Error::ModelLoadError(format!(
                "model {variant} lost its resource lease before adapter binding"
            ))
        })?;
        if slot.phase != ResidentPhase::Loading || slot.model_instance_id != model_instance_id {
            return Err(Error::ModelLoadError(format!(
                "model {variant} changed lifecycle identity before adapter binding"
            )));
        }
        if slot.bundle.is_some() {
            return Err(Error::ModelLoadError(format!(
                "model {variant} already has a loaded execution bundle"
            )));
        }
        slot.bundle = Some(bundle.clone());
        Ok(bundle)
    }

    pub(super) fn mark_slot_ready_for_instance(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
    ) -> Result<()> {
        let mut state = self.state();
        let slot = state.residents.get_mut(&variant).ok_or_else(|| {
            Error::ModelLoadError(format!(
                "model {variant} lost its resource lease before publication"
            ))
        })?;
        if slot.phase != ResidentPhase::Loading {
            return Err(Error::ModelLoadError(format!(
                "model {variant} cannot transition from {:?} to ready",
                slot.phase
            )));
        }
        if slot.model_instance_id != model_instance_id {
            return Err(Error::ModelLoadError(format!(
                "model {variant} load generation changed before ready publication"
            )));
        }
        let bundle = slot.bundle.as_ref().ok_or_else(|| {
            Error::ModelLoadError(format!(
                "model {variant} cannot become ready without an execution bundle"
            ))
        })?;
        if bundle.model_instance_id() != model_instance_id || bundle.model_variant() != variant {
            return Err(Error::ModelLoadError(format!(
                "model {variant} execution bundle does not match its lifecycle slot"
            )));
        }
        slot.phase = ResidentPhase::Ready;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn finalize_slot_materialization(
        &self,
        variant: ModelVariant,
        resident_resources: ResourceVector,
    ) -> Result<()> {
        self.finalize_slot_materialization_with_pending(
            variant,
            resident_resources,
            ResourceVector::zero(),
        )
    }

    pub(super) fn finalize_slot_materialization_with_pending(
        &self,
        variant: ModelVariant,
        resident_resources: ResourceVector,
        deferred_resources: ResourceVector,
    ) -> Result<()> {
        let materialized = resident_resources.checked_sub(deferred_resources)?;
        let mut state = self.state();
        let slot = state.residents.get_mut(&variant).ok_or_else(|| {
            Error::ModelLoadError(format!(
                "model {variant} lost its resource lease during physical instantiation"
            ))
        })?;
        // Allocation was authorized by the immutable peak reservation before
        // instantiation. Record the retained physical residency first, then
        // shed only authorization that was transient to model construction.
        slot.resource_lease.reconcile_materialized(materialized)?;
        if slot.resource_lease.resources() != resident_resources {
            slot.resource_lease.resize(resident_resources)?;
        }
        Ok(())
    }

    pub(super) fn begin_unloading_slot(&self, variant: ModelVariant) -> Result<bool> {
        let mut state = self.state();
        let active_leases = self.model_manager.active_residency_leases(variant);
        if active_leases > 0 {
            return Err(Error::InferenceError(format!(
                "Cannot unload model {variant}: {active_leases} active inference lease(s) are still held"
            )));
        }
        let Some(slot) = state.residents.get_mut(&variant) else {
            return Ok(false);
        };
        match slot.phase {
            // Callers hold the lifecycle mutation gate. A Loading slot at this
            // point has no physical initializer still running and represents
            // rollback/shutdown cleanup that must not leak its lease.
            ResidentPhase::Loading | ResidentPhase::Ready | ResidentPhase::CleanupRequired => {
                slot.phase = ResidentPhase::Unloading;
                Ok(true)
            }
            ResidentPhase::Unloading => Ok(true),
        }
    }

    /// Remove the immutable execution/state bundle before physical arenas are
    /// drained. Retiring the publication first prevents new adapter selection
    /// from racing teardown and drops any publication-owned state handles
    /// before their authoritative managers perform ownership checks.
    pub(super) fn retire_loaded_model_bundle(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
    ) -> Result<bool> {
        let bundle = {
            let mut state = self.state();
            let slot = state.residents.get_mut(&variant).ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "model {variant} lost its lifecycle slot before physical-state retirement"
                ))
            })?;
            if slot.model_instance_id != model_instance_id {
                return Err(Error::ModelLoadError(format!(
                    "model {variant} changed generation before physical-state retirement"
                )));
            }
            if !matches!(
                slot.phase,
                ResidentPhase::Loading | ResidentPhase::Unloading | ResidentPhase::CleanupRequired
            ) {
                return Err(Error::ModelLoadError(format!(
                    "model {variant} cannot retire its execution bundle while {:?}",
                    slot.phase
                )));
            }
            slot.bundle.take()
        };
        let retired = bundle.is_some();
        drop(bundle);
        Ok(retired)
    }

    pub(super) fn remove_resident_slot(&self, variant: ModelVariant) -> bool {
        let removed = self.state().residents.remove(&variant);
        let was_resident = removed.is_some();
        drop(removed);
        was_resident
    }

    pub(super) fn restore_ready_slot_after_failed_unload(&self, variant: ModelVariant) {
        if let Some(slot) = self.state().residents.get_mut(&variant) {
            if slot.phase == ResidentPhase::Unloading {
                slot.phase = ResidentPhase::Ready;
            }
        }
    }

    pub(super) fn mark_slot_cleanup_required(&self, variant: ModelVariant) {
        if let Some(slot) = self.state().residents.get_mut(&variant) {
            slot.phase = ResidentPhase::CleanupRequired;
        }
    }

    pub(super) fn recover_unloading_slots(&self) {
        let mut state = self.state();
        for slot in state.residents.values_mut() {
            if slot.phase == ResidentPhase::Unloading {
                slot.phase = ResidentPhase::CleanupRequired;
            }
        }
    }

    pub(crate) fn join_or_start_load(
        &self,
        variant: ModelVariant,
    ) -> (LoadWaiter, Option<LoadLeader>) {
        let mut state = self.state();
        if let Some(load) = state.loads.get(&variant) {
            return (
                LoadWaiter {
                    completion: load.completion.subscribe(),
                },
                None,
            );
        }

        state.next_generation = state.next_generation.wrapping_add(1).max(1);
        let generation = state.next_generation;
        let (completion, receiver) = watch::channel(None);
        state.loads.insert(
            variant,
            InFlightLoad {
                generation,
                completion: completion.clone(),
            },
        );
        (
            LoadWaiter {
                completion: receiver,
            },
            Some(LoadLeader {
                generation,
                completion,
            }),
        )
    }

    /// Check that a detached load still owns the registered generation after
    /// acquiring the lifecycle mutation gate. An unload that acquired the
    /// gate first removes the registration and publishes cancellation, fencing
    /// the stale task before it can instantiate or publish model state.
    pub(super) fn is_current_load_generation_locked(
        &self,
        variant: ModelVariant,
        generation: u64,
    ) -> bool {
        self.state()
            .loads
            .get(&variant)
            .is_some_and(|load| load.generation == generation)
    }

    /// Supersede one registered load while holding the lifecycle mutation
    /// gate. If the load already owns the gate, the caller cannot reach this
    /// method until that generation has published its terminal outcome.
    pub(super) fn supersede_registered_load_locked(
        &self,
        variant: ModelVariant,
        operation: &str,
    ) -> bool {
        let mut state = self.state();
        let Some(load) = state.loads.remove(&variant) else {
            return false;
        };
        load.completion.send_replace(Some(SharedLoadOutcome::Failed(
            SharedLoadFailure::Cancelled(format!(
                "model load {variant} was superseded by {operation}"
            )),
        )));
        true
    }

    /// Supersede every registered load while holding the lifecycle mutation
    /// gate. Later load requests receive a new generation and are not affected.
    pub(super) fn supersede_all_registered_loads_locked(&self, operation: &str) -> usize {
        let mut state = self.state();
        let loads = std::mem::take(&mut state.loads);
        let count = loads.len();
        for (variant, load) in loads {
            load.completion.send_replace(Some(SharedLoadOutcome::Failed(
                SharedLoadFailure::Cancelled(format!(
                    "model load {variant} was superseded by {operation}"
                )),
            )));
        }
        count
    }

    /// Publish a terminal load outcome while the caller still owns the
    /// lifecycle mutation gate. This prevents unload from separating the Ready
    /// slot commit from the outcome observed by coalesced waiters.
    pub(super) fn finish_load_locked(
        &self,
        variant: ModelVariant,
        generation: u64,
        completion: &watch::Sender<Option<SharedLoadOutcome>>,
        outcome: SharedLoadOutcome,
    ) {
        let mut state = self.state();
        if state
            .loads
            .get(&variant)
            .is_some_and(|load| load.generation == generation)
        {
            completion.send_replace(Some(outcome));
            state.loads.remove(&variant);
        }
    }

    #[cfg(test)]
    pub(super) fn set_load_test_panics(&self, count: usize) {
        self.load_test_panics.store(count, Ordering::Release);
    }

    #[cfg(test)]
    pub(super) fn maybe_panic_during_load(&self) {
        if self
            .load_test_panics
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            panic!("injected model load panic");
        }
    }

    #[cfg(test)]
    pub(super) fn set_unload_test_barriers(&self, reached: Arc<Barrier>, release: Arc<Barrier>) {
        *self
            .unload_test_barriers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some((reached, release));
    }

    #[cfg(test)]
    pub(super) async fn wait_at_unload_test_barrier(&self) {
        let barriers = self
            .unload_test_barriers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        if let Some((reached, release)) = barriers {
            reached.wait().await;
            release.wait().await;
        }
    }

    #[cfg(test)]
    pub(super) fn set_unload_test_panics(&self, count: usize) {
        self.unload_test_panics.store(count, Ordering::Release);
    }

    #[cfg(test)]
    pub(super) fn maybe_panic_during_unload_cleanup(&self) {
        if self
            .unload_test_panics
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            panic!("injected model unload cleanup panic");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn concurrent_waiters_join_the_same_generation() {
        let (first_tx, first_rx) = watch::channel(None);
        let mut loads = HashMap::new();
        loads.insert(
            ModelVariant::Kokoro82M,
            InFlightLoad {
                generation: 7,
                completion: first_tx.clone(),
            },
        );
        let state = LifecycleState {
            next_generation: 7,
            loads,
            ..Default::default()
        };
        let first = LoadWaiter {
            completion: first_rx,
        };
        let second = LoadWaiter {
            completion: state
                .loads
                .get(&ModelVariant::Kokoro82M)
                .expect("in-flight load")
                .completion
                .subscribe(),
        };

        first_tx.send_replace(Some(SharedLoadOutcome::Ready));

        first.wait().await.expect("first waiter");
        second.wait().await.expect("second waiter");
    }

    #[tokio::test]
    async fn shared_failure_preserves_overload_classification() {
        let (completion, receiver) = watch::channel(None);
        completion.send_replace(Some(SharedLoadOutcome::Failed(
            SharedLoadFailure::Overloaded("memory budget exhausted".to_string()),
        )));

        let error = LoadWaiter {
            completion: receiver,
        }
        .wait()
        .await
        .expect_err("load must fail");
        assert!(matches!(error, Error::Overloaded(_)));
    }
}
