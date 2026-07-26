//! Live binding between loaded-model KV contracts and physical engine state.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use candle_core::{DType, Device, DeviceLocation};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::coordinator::{
    KvBlockIntent, KvCacheCoordinator, KvCoordinatorCommitPlan, KvCoordinatorError,
    KvGroupReservation, KvReserveRequest, KvSnapshot, KvWindowReserveRequest, KvWriteReceipt,
};
use super::prefix::{
    CoordinatedPrefixIndex, KvPrefixNamespace, KvPrefixPageKey, KvPrefixPublication,
    StagedPrefixCommit,
};
use super::telemetry::{ManagedKvTelemetry, ManagedKvTelemetrySnapshot};
#[cfg(feature = "cuda")]
use crate::backends::kv::CudaKvBackendRuntime;
#[cfg(feature = "metal")]
use crate::backends::kv::MetalKvBackendRuntime;
use crate::backends::kv::{
    CpuKvBackendRuntime, KvArena, KvArenaConfig, KvBackendPlanRequest, KvBackendRuntime,
    KvLayerConfig,
};
use crate::backends::state::{
    negotiate_state_plan, PhysicalStateSequenceId, PhysicalStateTransactionId,
    StateBackendPlanRequest, TensorStateArena,
};
use crate::backends::BackendKind;
use crate::engine::{
    EngineCoreRequest, ManagedCacheDomainReservation, ManagedCacheReceipt, ManagedCacheReservation,
    ManagedTensorStateReservation, ModelInstanceId, PlanId, ReservationClass, ReservationOwner,
    ResourceAmount, ResourceAuthority, ResourceLease, ResourceVector, SessionKey, WorkUnit,
};
use crate::error::{Error, Result};
use crate::kv::v2::{upgrade_kv_contract_v1, InferenceStateContract, ResolvedStatePlan};
#[cfg(test)]
use crate::kv::CacheDomainId;
use crate::kv::{
    AttentionSemantics, CacheCapability, KvArenaId, KvCacheContract, KvDomainSpec, KvGroupId,
    KvPrefixSemantics, KvStorageDType, ResolvedKvGroupKind, ResolvedKvPlan,
};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ManagedKvOperationSnapshot {
    pub slot_write_dispatches: u64,
    pub paged_decode_dispatches: u64,
    pub page_zero_dispatches: u64,
    pub page_copy_dispatches: u64,
    pub host_synchronizations: u64,
}

impl ManagedKvOperationSnapshot {
    fn add_assign(&mut self, other: Self) {
        self.slot_write_dispatches = self
            .slot_write_dispatches
            .saturating_add(other.slot_write_dispatches);
        self.paged_decode_dispatches = self
            .paged_decode_dispatches
            .saturating_add(other.paged_decode_dispatches);
        self.page_zero_dispatches = self
            .page_zero_dispatches
            .saturating_add(other.page_zero_dispatches);
        self.page_copy_dispatches = self
            .page_copy_dispatches
            .saturating_add(other.page_copy_dispatches);
        self.host_synchronizations = self
            .host_synchronizations
            .saturating_add(other.host_synchronizations);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ManagedKvCoordinatorSnapshot {
    pub capacity_pages: u64,
    pub allocated_pages: u64,
    pub free_pages: u64,
    pub table_refs: u64,
    pub prefix_refs: u64,
    pub execution_pins: u64,
    pub transfer_pins: u64,
    pub reservations: u64,
    pub active_transactions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagedKvArenaRuntimeSnapshot {
    pub generation: u32,
    pub group_id: u32,
    pub domain_id: u32,
    pub device_ordinal: Option<u32>,
    pub page_tokens: u32,
    pub bytes_per_page: u64,
    pub physical_bytes: u64,
    pub coordinator: ManagedKvCoordinatorSnapshot,
    pub operations: ManagedKvOperationSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagedKvModelRuntimeSnapshot {
    pub model_instance: ModelInstanceId,
    pub plan_fingerprint: String,
    pub state_plan_v2_fingerprint: String,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    pub physical_bytes: u64,
    pub registered_sessions: u64,
    pub arenas: Vec<ManagedKvArenaRuntimeSnapshot>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ManagedKvRuntimeTotalsSnapshot {
    pub models: u64,
    pub arenas: u64,
    pub registered_sessions: u64,
    pub physical_bytes: u64,
    pub coordinator: ManagedKvCoordinatorSnapshot,
    pub operations: ManagedKvOperationSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagedKvRuntimeSnapshot {
    pub memory_accounting: &'static str,
    pub totals: ManagedKvRuntimeTotalsSnapshot,
    pub counters: ManagedKvTelemetrySnapshot,
    pub models: Vec<ManagedKvModelRuntimeSnapshot>,
}

impl Default for ManagedKvRuntimeSnapshot {
    fn default() -> Self {
        Self {
            memory_accounting: "physical_arena_backing",
            totals: ManagedKvRuntimeTotalsSnapshot::default(),
            counters: ManagedKvTelemetrySnapshot::default(),
            models: Vec::new(),
        }
    }
}

/// Immutable model-level plan and physical arenas shared by all its sessions.
pub(crate) struct ManagedKvModelRuntime {
    plan: Arc<ResolvedKvPlan>,
    state_plan_v2: Arc<ResolvedStatePlan>,
    arenas: HashMap<KvArenaId, Arc<dyn KvArena>>,
    tensor_state: Option<Arc<TensorStateArena>>,
    physical_bytes: u64,
}

impl fmt::Debug for ManagedKvModelRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedKvModelRuntime")
            .field("plan", &self.plan.id)
            .field("state_plan_v2", &self.state_plan_v2.id)
            .field("arena_count", &self.arenas.len())
            .field("physical_bytes", &self.physical_bytes)
            .finish()
    }
}

impl ManagedKvModelRuntime {
    pub(crate) fn plan(&self) -> &ResolvedKvPlan {
        &self.plan
    }

    pub(crate) fn state_plan_v2(&self) -> &ResolvedStatePlan {
        &self.state_plan_v2
    }

    pub(crate) fn arena(&self, id: KvArenaId) -> Option<&Arc<dyn KvArena>> {
        self.arenas.get(&id)
    }

    pub(crate) fn tensor_state(&self) -> Option<&Arc<TensorStateArena>> {
        self.tensor_state.as_ref()
    }

    pub(crate) fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }
}

struct ManagedKvModelState {
    contract: KvCacheContract,
    runtime: Arc<ManagedKvModelRuntime>,
    coordinators: HashMap<KvArenaId, KvCacheCoordinator>,
    prefix_indexes: HashMap<KvArenaId, CoordinatedPrefixIndex>,
    pending_prefixes: HashMap<PlanId, Vec<PendingPrefixCommit>>,
    registered_sessions: HashSet<SessionKey>,
    tensor_sequences: HashMap<SessionKey, PhysicalStateSequenceId>,
    resource_lease: Option<ResourceLease>,
}

#[derive(Clone)]
struct PendingPrefixCommit {
    arena: KvArenaId,
    page_tokens: u32,
    publications: Vec<KvPrefixPublication>,
}

/// Engine-owned managed-cache registry. Arena backing is allocated once per
/// exact model instance; row transactions only change page ownership.
pub(crate) struct ManagedKvCacheManager {
    models: HashMap<ModelInstanceId, ManagedKvModelState>,
    resource_authority: Option<Arc<ResourceAuthority>>,
    next_arena_generation: u32,
    next_tensor_sequence: u64,
    telemetry: Arc<ManagedKvTelemetry>,
    prefix_cache_salt: Option<[u8; 32]>,
    worker_backend: BackendKind,
    worker_device_location: DeviceLocation,
    worker_device: Device,
    worker_device_ordinal: Option<u32>,
    backend_runtime: Option<Arc<dyn KvBackendRuntime>>,
    backend_unavailable: Option<String>,
}

impl Default for ManagedKvCacheManager {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ManagedKvCacheManager {
    #[cfg(test)]
    pub(crate) fn model_count(&self) -> usize {
        self.models.len()
    }

    pub(crate) fn new(resource_authority: Option<Arc<ResourceAuthority>>) -> Self {
        Self::for_worker(resource_authority, BackendKind::Cpu, Device::Cpu)
    }

    pub(crate) fn for_worker(
        resource_authority: Option<Arc<ResourceAuthority>>,
        backend: BackendKind,
        device: Device,
    ) -> Self {
        let (backend_runtime, backend_unavailable) = managed_backend_runtime(backend, &device);
        Self {
            models: HashMap::new(),
            resource_authority,
            next_arena_generation: 1,
            next_tensor_sequence: 1,
            telemetry: Arc::new(ManagedKvTelemetry::default()),
            prefix_cache_salt: None,
            worker_backend: backend,
            worker_device_location: device.location(),
            worker_device: device.clone(),
            worker_device_ordinal: managed_device_ordinal(&device),
            backend_runtime,
            backend_unavailable,
        }
    }

    pub(crate) fn with_prefix_cache_salt(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
    ) -> Self {
        let mut manager = Self::new(resource_authority);
        manager.prefix_cache_salt = salt;
        manager
    }

    pub(crate) fn for_worker_with_prefix_cache_salt(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
        backend: BackendKind,
        device: Device,
    ) -> Self {
        let mut manager = Self::for_worker(resource_authority, backend, device);
        manager.prefix_cache_salt = salt;
        manager
    }

    pub(crate) fn telemetry_snapshot(&self) -> ManagedKvTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    pub(crate) fn runtime_snapshot(&self) -> ManagedKvRuntimeSnapshot {
        let mut totals = ManagedKvRuntimeTotalsSnapshot {
            models: usize_to_u64(self.models.len()),
            ..ManagedKvRuntimeTotalsSnapshot::default()
        };
        let mut models = self
            .models
            .iter()
            .map(|(model_instance, state)| {
                totals.registered_sessions = totals
                    .registered_sessions
                    .saturating_add(usize_to_u64(state.registered_sessions.len()));
                totals.physical_bytes = totals
                    .physical_bytes
                    .saturating_add(state.runtime.physical_bytes);
                let mut arenas = state
                    .runtime
                    .plan
                    .groups
                    .iter()
                    .map(|group| {
                        let coordinator = state
                            .coordinators
                            .get(&group.arena)
                            .expect("resolved managed arena has a coordinator")
                            .stats();
                        let coordinator = ManagedKvCoordinatorSnapshot {
                            capacity_pages: usize_to_u64(coordinator.capacity_pages),
                            allocated_pages: usize_to_u64(coordinator.allocated_pages),
                            free_pages: usize_to_u64(coordinator.free_pages),
                            table_refs: usize_to_u64(coordinator.table_refs),
                            prefix_refs: usize_to_u64(coordinator.prefix_refs),
                            execution_pins: usize_to_u64(coordinator.execution_pins),
                            transfer_pins: usize_to_u64(coordinator.transfer_pins),
                            reservations: usize_to_u64(coordinator.reservations),
                            active_transactions: usize_to_u64(coordinator.active_transactions),
                        };
                        let arena = state
                            .runtime
                            .arenas
                            .get(&group.arena)
                            .expect("resolved managed arena has physical storage");
                        let operation_stats = arena.operation_stats();
                        let operations = ManagedKvOperationSnapshot {
                            slot_write_dispatches: operation_stats.slot_write_dispatches,
                            paged_decode_dispatches: operation_stats.paged_decode_dispatches,
                            page_zero_dispatches: operation_stats.page_zero_dispatches,
                            page_copy_dispatches: operation_stats.page_copy_dispatches,
                            host_synchronizations: operation_stats.host_synchronizations,
                        };
                        add_coordinator_stats(&mut totals.coordinator, &coordinator);
                        totals.operations.add_assign(operations.clone());
                        totals.arenas = totals.arenas.saturating_add(1);
                        ManagedKvArenaRuntimeSnapshot {
                            generation: group.arena.generation,
                            group_id: group.id.get(),
                            domain_id: group.domain.get(),
                            device_ordinal: group.arena.device_ordinal,
                            page_tokens: group.page_tokens,
                            bytes_per_page: group.bytes_per_page,
                            physical_bytes: group
                                .bytes_per_page
                                .saturating_mul(u64::from(group.capacity_pages)),
                            coordinator,
                            operations,
                        }
                    })
                    .collect::<Vec<_>>();
                arenas.sort_by_key(|arena| (arena.generation, arena.group_id));
                ManagedKvModelRuntimeSnapshot {
                    model_instance: *model_instance,
                    plan_fingerprint: state.runtime.plan.fingerprint().to_string(),
                    state_plan_v2_fingerprint: state
                        .runtime
                        .state_plan_v2
                        .fingerprint()
                        .to_string(),
                    backend: state.runtime.plan.backend,
                    device_ordinal: state.runtime.plan.device_ordinal,
                    physical_bytes: state.runtime.physical_bytes,
                    registered_sessions: usize_to_u64(state.registered_sessions.len()),
                    arenas,
                }
            })
            .collect::<Vec<_>>();
        models.sort_by_key(|model| model.model_instance);
        ManagedKvRuntimeSnapshot {
            memory_accounting: "physical_arena_backing",
            totals,
            counters: self.telemetry.snapshot(),
            models,
        }
    }

    pub(crate) fn worker_backend(&self) -> BackendKind {
        self.worker_backend
    }

    pub(crate) fn bind_request(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity_pages: usize,
        page_tokens_hint: usize,
        capability: &CacheCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let state_contract_v2 = upgrade_kv_contract_v1(contract)?;
        self.bind_model_state(
            model_instance,
            backend,
            capacity_pages,
            page_tokens_hint,
            contract,
            &state_contract_v2,
        )
        .map(Some)
    }

    pub(crate) fn bind_model_state(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity_pages: usize,
        page_tokens_hint: usize,
        contract: &KvCacheContract,
        state_contract_v2: &InferenceStateContract,
    ) -> Result<Arc<ManagedKvModelRuntime>> {
        validate_sliding_contract(contract, backend)?;
        if let Some(state) = self.models.get(&model_instance) {
            if &state.contract != contract
                || state.runtime.plan.backend != backend
                || state.runtime.state_plan_v2.contract_fingerprint
                    != state_contract_v2.fingerprint()?
            {
                return Err(Error::InvalidInput(
                    "one loaded model instance published incompatible managed KV contracts"
                        .to_string(),
                ));
            }
            return Ok(state.runtime.clone());
        }
        if backend != self.worker_backend {
            return Err(Error::InvalidInput(format!(
                "managed KV request targets {backend:?}, but its worker is bound to {:?}",
                self.worker_backend
            )));
        }
        let backend_runtime = self.backend_runtime.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                self.backend_unavailable
                    .clone()
                    .unwrap_or_else(|| format!("managed KV is unavailable for {backend:?}")),
            )
        })?;
        let capacity_pages = u32::try_from(capacity_pages)
            .map_err(|_| Error::InvalidInput("managed KV page capacity exceeds u32".to_string()))?;
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".to_string()))?;
        let first_arena_generation = self.next_arena_generation;
        let paged_contract = paged_only_contract(contract)?;
        let plan = backend_runtime.negotiate(
            &paged_contract,
            &KvBackendPlanRequest {
                model_instance,
                backend,
                device_ordinal: self.worker_device_ordinal,
                capacity_pages,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
                first_arena_generation,
            },
        )?;
        let state_plan_v2 = negotiate_state_plan(
            state_contract_v2,
            &StateBackendPlanRequest {
                backend,
                device_ordinal: self.worker_device_ordinal,
                // Resolve against the exact page geometry selected by the
                // allocating backend, not merely the configuration hint.
                page_tokens_hint: plan.groups.first().map(|group| group.page_tokens),
                storage_dtype_hint: None,
            },
        )?;
        validate_v2_physical_equivalence(&plan, &state_plan_v2)?;
        let tensor_state = (!state_plan_v2.non_paged.is_empty())
            .then(|| {
                TensorStateArena::new(Arc::new(state_plan_v2.clone()), self.worker_device.clone())
            })
            .transpose()?
            .map(Arc::new);

        let physical_bytes = plan_physical_bytes(&plan)?;
        let resources = managed_arena_resources(backend, physical_bytes);
        let resource_lease = self
            .resource_authority
            .as_ref()
            .map(|authority| reserve_managed_arena(authority, model_instance, backend, resources))
            .transpose()?;
        let mut arenas = HashMap::with_capacity(plan.groups.len());
        let mut coordinators = HashMap::with_capacity(plan.groups.len());
        let mut prefix_indexes = HashMap::with_capacity(plan.groups.len());
        for group in &plan.groups {
            let config = arena_config(contract, group)?;
            let arena = backend_runtime.allocate_arena(config)?;
            if arena.backend_kind() != self.worker_backend {
                return Err(Error::InferenceError(format!(
                    "managed KV runtime allocated a {:?} arena for a {:?} worker",
                    arena.backend_kind(),
                    self.worker_backend
                )));
            }
            if arena.device_location() != self.worker_device_location {
                return Err(Error::InferenceError(format!(
                    "managed KV runtime allocated arena device {:?} for exact worker device {:?}",
                    arena.device_location(),
                    self.worker_device_location
                )));
            }
            if arenas.insert(group.arena, arena).is_some() {
                return Err(Error::InferenceError(
                    "resolved KV plan reused one arena identity".to_string(),
                ));
            }
            self.telemetry.record_backing_allocation();
            coordinators.insert(
                group.arena,
                KvCacheCoordinator::new(group.arena, group.capacity_pages as usize),
            );
            prefix_indexes.insert(
                group.arena,
                CoordinatedPrefixIndex::with_telemetry(
                    group.capacity_pages as usize,
                    self.telemetry.clone(),
                ),
            );
        }
        if let Some(lease) = resource_lease.as_ref() {
            lease.record_materialized_usage(resources)?;
        }
        let runtime = Arc::new(ManagedKvModelRuntime {
            plan: Arc::new(plan),
            state_plan_v2: Arc::new(state_plan_v2),
            arenas,
            tensor_state,
            physical_bytes,
        });
        self.models.insert(
            model_instance,
            ManagedKvModelState {
                contract: contract.clone(),
                runtime: runtime.clone(),
                coordinators,
                prefix_indexes,
                pending_prefixes: HashMap::new(),
                registered_sessions: HashSet::new(),
                tensor_sequences: HashMap::new(),
                resource_lease,
            },
        );
        self.next_arena_generation = first_arena_generation
            .checked_add(u32::try_from(runtime.plan.groups.len()).map_err(|_| {
                Error::InvalidInput("managed KV arena count exceeds u32".to_string())
            })?)
            .ok_or_else(|| Error::InvalidInput("managed KV arena generation overflow".into()))?;
        Ok(runtime)
    }

    /// Resolve an arena runtime that was allocated by model loading. Request
    /// admission must never create backing storage or expand model residency.
    pub(crate) fn require_loaded_runtime(
        &self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capability: &CacheCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let state = self.models.get(&model_instance).ok_or_else(|| {
            Error::InferenceError(
                "loaded adapter published managed KV without load-time physical allocation"
                    .to_string(),
            )
        })?;
        if backend != self.worker_backend
            || state.runtime.plan.backend != backend
            || &state.contract != contract
        {
            return Err(Error::InferenceError(
                "load-time managed KV runtime does not match the request capability".to_string(),
            ));
        }
        Ok(Some(state.runtime.clone()))
    }

    pub(crate) fn prepare(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
    ) -> Result<Option<ManagedCacheReservation>> {
        let WorkUnit::SequenceStep { input, .. } = work else {
            return Ok(None);
        };
        let target_committed_tokens = u32::try_from(input.end).map_err(|_| {
            Error::InvalidInput("managed KV token position exceeds u32".to_string())
        })?;
        let namespace = managed_prefix_namespace(request, runtime, self.prefix_cache_salt)?;
        let tensor_sequence_candidate = if runtime.tensor_state().is_some() {
            let candidate = PhysicalStateSequenceId::new(self.next_tensor_sequence)?;
            self.next_tensor_sequence = self
                .next_tensor_sequence
                .checked_add(1)
                .ok_or_else(|| Error::InferenceError("tensor-state sequence id overflow".into()))?;
            Some(candidate)
        } else {
            None
        };
        let state = self
            .models
            .get_mut(&runtime.plan.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model runtime is missing".into()))?;
        if state.runtime.plan.id != runtime.plan.id {
            return Err(Error::InferenceError(
                "request carries a stale managed KV runtime".to_string(),
            ));
        }
        ensure_session_tables(state, session)?;

        let mut domains = Vec::with_capacity(runtime.plan.groups.len());
        let mut pending_prefixes = Vec::new();
        for group in &runtime.plan.groups {
            let coordinator = state
                .coordinators
                .get_mut(&group.arena)
                .expect("resolved arena has a coordinator");
            let snapshot = coordinator
                .snapshot(session, group.domain)
                .map_err(coordinator_error)?;
            if target_committed_tokens < snapshot.committed_tokens {
                abort_domains(state, txn_id, &domains);
                return Err(Error::InferenceError(
                    "scheduled KV target regressed behind the committed cache table".to_string(),
                ));
            }
            let prefix_eligible = snapshot.committed_tokens == 0
                && input.start == 0
                && request.is_some_and(|request| input.end == request.prompt_tokens.len())
                && target_committed_tokens > 1
                && prefix_enabled_for_domain(&state.contract, group.domain);
            let prefix_match = if prefix_eligible {
                if let Some(namespace) = namespace.as_ref() {
                    let reusable_tokens =
                        usize::try_from(target_committed_tokens - 1).unwrap_or(usize::MAX);
                    state
                        .prefix_indexes
                        .get_mut(&group.arena)
                        .expect("resolved arena has a prefix index")
                        .lookup_longest(
                            namespace,
                            &request
                                .expect("prefix namespace requires a request")
                                .prompt_tokens[..reusable_tokens],
                            group.page_tokens,
                        )
                        .map_err(prefix_error)?
                } else {
                    Default::default()
                }
            } else {
                Default::default()
            };
            let execution_start_tokens = snapshot.committed_tokens.max(prefix_match.reused_tokens);
            let sliding_window = sliding_window_for_domain(&state.contract, group.domain)?;
            let target_window_start = sliding_window
                .map(|window| {
                    target_committed_tokens
                        .saturating_sub(window)
                        .min(u32::try_from(input.start).unwrap_or(u32::MAX))
                })
                .unwrap_or(0);
            let established_window_table = sliding_window.is_some()
                && snapshot.groups.iter().any(|table| table.group == group.id);
            let reserve_request = if established_window_table {
                None
            } else {
                let reservation = reservation_for_group(
                    group.id,
                    group.page_tokens,
                    &snapshot,
                    target_committed_tokens,
                    &prefix_match.blocks,
                )?;
                Some(KvReserveRequest {
                    txn_id,
                    expected: snapshot.clone(),
                    target_committed_tokens,
                    target_window_start: 0,
                    groups: vec![reservation],
                })
            };
            let reserve_once = |coordinator: &mut KvCacheCoordinator| {
                if established_window_table {
                    coordinator.reserve_window(KvWindowReserveRequest {
                        txn_id,
                        expected: snapshot.clone(),
                        target_committed_tokens,
                        target_window_start,
                        page_tokens: group.page_tokens,
                    })
                } else {
                    let request = reserve_request
                        .as_ref()
                        .expect("non-window reservation exists")
                        .clone();
                    match coordinator.reserve(request.clone()) {
                        Err(KvCoordinatorError::WriteConflict) => {
                            let mut copy_on_write = request;
                            for intent in &mut copy_on_write.groups[0].blocks {
                                if let KvBlockIntent::Writable(source) = *intent {
                                    *intent = KvBlockIntent::CopyOnWrite(source);
                                }
                            }
                            coordinator.reserve(copy_on_write)
                        }
                        result => result,
                    }
                }
            };
            let mut reserved = reserve_once(coordinator);
            while matches!(reserved, Err(KvCoordinatorError::Capacity)) {
                let protected = prefix_match.blocks.iter().copied().collect::<HashSet<_>>();
                let evicted = state
                    .prefix_indexes
                    .get_mut(&group.arena)
                    .expect("resolved arena has a prefix index")
                    .evict_lru_excluding(coordinator, &protected)
                    .map_err(prefix_error)?;
                if evicted.is_empty() {
                    break;
                }
                reserved = reserve_once(coordinator);
            }
            if let Err(error) = reserved {
                abort_domains(state, txn_id, &domains);
                if matches!(error, KvCoordinatorError::Capacity) {
                    return Err(Error::Backpressure(
                        "managed KV arena has no reservable pages".to_string(),
                    ));
                }
                return Err(coordinator_error(error));
            }
            let prepared = match coordinator.prepare(txn_id) {
                Ok(prepared) => prepared,
                Err(error) => {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(coordinator_error(error));
                }
            };
            let old = prepared
                .expected
                .groups
                .iter()
                .flat_map(|group| group.blocks.iter().copied())
                .collect::<HashSet<_>>();
            let fresh = prepared
                .writable_blocks
                .iter()
                .copied()
                .filter(|block| !old.contains(block))
                .collect::<Vec<_>>();
            if !fresh.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = arena.zero_pages(&fresh).and_then(|fence| fence.wait()) {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
                self.telemetry.record_zero(fresh.len());
            }
            if !prepared.page_copies.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = arena
                    .copy_pages(&prepared.page_copies)
                    .and_then(|fence| fence.wait())
                {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
                self.telemetry.record_copy(prepared.page_copies.len());
            }
            domains.push(ManagedCacheDomainReservation {
                arena: group.arena,
                domain: group.domain,
                expected_version: prepared.expected.version,
                expected_committed_tokens: prepared.expected.committed_tokens,
                execution_start_tokens,
                target_committed_tokens: prepared.target_committed_tokens,
                target_window_start: prepared.target_window_start,
                first_page_offset: prepared.target_window_start % group.page_tokens,
                provisional_groups: prepared.provisional_groups,
                writable_blocks: prepared.writable_blocks,
            });
            if prefix_eligible {
                let Some(namespace) = namespace.as_ref() else {
                    continue;
                };
                let publications = prefix_publications(
                    namespace,
                    &request
                        .expect("prefix namespace requires a request")
                        .prompt_tokens,
                    group.page_tokens,
                    execution_start_tokens,
                    target_committed_tokens,
                    domains
                        .last()
                        .expect("domain reservation was just appended"),
                    group.id,
                )?;
                if !publications.is_empty() {
                    pending_prefixes.push(PendingPrefixCommit {
                        arena: group.arena,
                        page_tokens: group.page_tokens,
                        publications,
                    });
                }
            }
        }
        if !pending_prefixes.is_empty()
            && state
                .pending_prefixes
                .insert(txn_id, pending_prefixes)
                .is_some()
        {
            abort_domains(state, txn_id, &domains);
            return Err(Error::InferenceError(
                "managed KV transaction duplicated pending prefix publication".into(),
            ));
        }
        let tensor_state = if let Some(arena) = runtime.tensor_state() {
            let sequence = if let Some(sequence) = state.tensor_sequences.get(session).copied() {
                sequence
            } else {
                let sequence = tensor_sequence_candidate.expect("tensor arena has a candidate");
                if let Err(error) = arena.register(sequence) {
                    abort_domains(state, txn_id, &domains);
                    state.pending_prefixes.remove(&txn_id);
                    return Err(error);
                }
                state.tensor_sequences.insert(session.clone(), sequence);
                sequence
            };
            let transaction = PhysicalStateTransactionId::new(txn_id)?;
            if let Err(error) = arena.begin(transaction, sequence) {
                abort_domains(state, txn_id, &domains);
                state.pending_prefixes.remove(&txn_id);
                return Err(error);
            }
            Some(ManagedTensorStateReservation {
                sequence: sequence.get(),
            })
        } else {
            None
        };
        Ok(Some(ManagedCacheReservation {
            txn_id,
            session: session.clone(),
            domains,
            tensor_state,
        }))
    }

    pub(crate) fn finalize(
        &mut self,
        reservation: &ManagedCacheReservation,
        receipt: Option<&ManagedCacheReceipt>,
        commit: bool,
    ) -> Result<()> {
        let model_instance = reservation
            .domains
            .first()
            .map(|domain| domain.arena.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV reservation is empty".into()))?;
        let state = self
            .models
            .get_mut(&model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model state is missing".into()))?;
        if !commit {
            abort_reservation(state, reservation);
            self.telemetry.record_abort();
            return Ok(());
        }
        let receipt = match receipt {
            Some(receipt) => receipt,
            None => {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "committing managed KV row omitted its write receipt".into(),
                ));
            }
        };
        if &receipt.reservation != reservation {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV receipt crossed a row transaction fence".to_string(),
            ));
        }

        // Mark every live transaction written. This changes no table/index
        // ownership and is rolled back by abort if any later validation fails.
        for domain in &reservation.domains {
            let Some(written) = receipt
                .domains
                .iter()
                .find(|receipt| receipt.arena == domain.arena && receipt.domain == domain.domain)
            else {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "managed KV receipt omitted a cache domain".into(),
                ));
            };
            let completed = state
                .coordinators
                .get_mut(&domain.arena)
                .expect("reservation arena has a coordinator")
                .complete_write(KvWriteReceipt {
                    txn_id: reservation.txn_id,
                    committed_tokens: domain.target_committed_tokens,
                    written_blocks: written.written_blocks.clone(),
                });
            if let Err(error) = completed {
                abort_reservation(state, reservation);
                return Err(coordinator_error(error));
            }
        }
        let mut pending = state
            .pending_prefixes
            .get(&reservation.txn_id)
            .cloned()
            .unwrap_or_default();
        let mut staged = Vec::<(
            KvArenaId,
            KvCoordinatorCommitPlan,
            Option<StagedPrefixCommit>,
        )>::with_capacity(reservation.domains.len());
        for domain in &reservation.domains {
            let prefix = if let Some(index) = pending
                .iter()
                .position(|publication| publication.arena == domain.arena)
            {
                let publication = pending.swap_remove(index);
                let staged_prefix = state
                    .prefix_indexes
                    .get(&domain.arena)
                    .expect("reservation arena has a prefix index")
                    .stage_transaction(publication.page_tokens, &publication.publications);
                Some(match staged_prefix {
                    Ok(staged) => staged,
                    Err(error) => {
                        abort_reservation(state, reservation);
                        return Err(prefix_error(error));
                    }
                })
            } else {
                None
            };
            let coordinator = state
                .coordinators
                .get(&domain.arena)
                .expect("reservation arena has a coordinator");
            let commit = coordinator.stage_commit_with_prefix_updates(
                reservation.txn_id,
                prefix
                    .as_ref()
                    .map(StagedPrefixCommit::retained)
                    .unwrap_or(&[]),
                prefix
                    .as_ref()
                    .map(StagedPrefixCommit::released)
                    .unwrap_or(&[]),
            );
            match commit {
                Ok(commit) => staged.push((domain.arena, commit, prefix)),
                Err(error) => {
                    abort_reservation(state, reservation);
                    return Err(coordinator_error(error));
                }
            }
        }
        if !pending.is_empty() {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV transaction contains a prefix publication for an unknown domain".into(),
            ));
        }
        if reservation.tensor_state.is_some() {
            let arena = state.runtime.tensor_state().ok_or_else(|| {
                Error::InferenceError("tensor-state reservation lost its physical arena".into())
            })?;
            let target_cursor = reservation
                .domains
                .first()
                .map(|domain| u64::from(domain.target_committed_tokens))
                .ok_or_else(|| Error::InferenceError("managed KV reservation is empty".into()))?;
            if reservation
                .domains
                .iter()
                .any(|domain| u64::from(domain.target_committed_tokens) != target_cursor)
            {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "one managed state transaction resolved divergent domain cursors".into(),
                ));
            }
            if let Err(error) = arena.commit(
                PhysicalStateTransactionId::new(reservation.txn_id)?,
                target_cursor,
            ) {
                abort_reservation(state, reservation);
                return Err(error);
            }
        }
        // Every fallible operation has succeeded. Applying these plans cannot
        // fail, and the engine state lock prevents an interleaving mutation.
        for (arena, commit, prefix) in staged {
            state
                .coordinators
                .get_mut(&arena)
                .expect("staged arena has a coordinator")
                .apply_staged_commit(commit);
            if let Some(prefix) = prefix {
                state
                    .prefix_indexes
                    .get_mut(&arena)
                    .expect("staged arena has a prefix index")
                    .apply_staged(prefix);
            }
        }
        state.pending_prefixes.remove(&reservation.txn_id);
        self.telemetry.record_commit();
        Ok(())
    }

    pub(crate) fn release_session(&mut self, session: &SessionKey) -> Result<()> {
        for state in self.models.values_mut() {
            if !state.registered_sessions.contains(session) {
                continue;
            }
            for group in &state.runtime.plan.groups {
                state
                    .coordinators
                    .get(&group.arena)
                    .expect("resolved arena has a coordinator")
                    .validate_table_release(session, group.domain)
                    .map_err(coordinator_error)?;
            }
            if let Some(sequence) = state.tensor_sequences.get(session).copied() {
                state
                    .runtime
                    .tensor_state()
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "registered tensor-state sequence lost its arena".into(),
                        )
                    })?
                    .validate_release(sequence)?;
            }
            for group in &state.runtime.plan.groups {
                state
                    .coordinators
                    .get_mut(&group.arena)
                    .expect("resolved arena has a coordinator")
                    .release_table(session, group.domain)
                    .map_err(coordinator_error)?;
            }
            if let Some(sequence) = state.tensor_sequences.remove(session) {
                state
                    .runtime
                    .tensor_state()
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "registered tensor-state sequence lost its arena".into(),
                        )
                    })?
                    .release(sequence)?;
            }
            state.registered_sessions.remove(session);
        }
        Ok(())
    }

    /// Drain and retire every arena belonging to one exact loaded-model
    /// generation. The model-scoped physical lease is retained until no
    /// session, row transaction, device fence, or external runtime handle can
    /// still reference the backing storage.
    pub(crate) fn unload_model(&mut self, model_instance: ModelInstanceId) -> Result<bool> {
        let Some(state) = self.models.get_mut(&model_instance) else {
            return Ok(false);
        };
        if !state.registered_sessions.is_empty() {
            return Err(Error::InferenceError(format!(
                "managed KV model {} still has registered sessions",
                model_instance.get()
            )));
        }
        if Arc::strong_count(&state.runtime) != 1 {
            return Err(Error::InferenceError(format!(
                "managed KV model {} still has live runtime handles",
                model_instance.get()
            )));
        }
        for group in &state.runtime.plan.groups {
            loop {
                let evicted = state
                    .prefix_indexes
                    .get_mut(&group.arena)
                    .expect("resolved arena has a prefix index")
                    .evict_lru(
                        state
                            .coordinators
                            .get_mut(&group.arena)
                            .expect("resolved arena has a coordinator"),
                    )
                    .map_err(prefix_error)?;
                if evicted.is_empty() {
                    break;
                }
            }
        }
        for coordinator in state.coordinators.values() {
            let stats = coordinator.stats();
            if stats.allocated_pages != 0
                || stats.table_refs != 0
                || stats.prefix_refs != 0
                || stats.execution_pins != 0
                || stats.transfer_pins != 0
                || stats.reservations != 0
                || stats.active_transactions != 0
            {
                return Err(Error::InferenceError(format!(
                    "managed KV model {} still has live page ownership or transactions",
                    model_instance.get()
                )));
            }
        }
        for arena in state.runtime.arenas.values() {
            arena.drain()?;
        }
        if let Some(lease) = state.resource_lease.as_ref() {
            lease.prepare_materialized_release(ResourceVector::zero())?;
        }
        let removed = self
            .models
            .remove(&model_instance)
            .expect("managed KV state was validated under exclusive manager access");
        drop(removed);
        Ok(true)
    }

    #[cfg(test)]
    pub(crate) fn snapshot(
        &self,
        model_instance: ModelInstanceId,
        session: &SessionKey,
        domain: CacheDomainId,
    ) -> Option<KvSnapshot> {
        let state = self.models.get(&model_instance)?;
        state
            .coordinators
            .values()
            .find_map(|coordinator| coordinator.snapshot(session, domain).ok())
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn add_coordinator_stats(
    totals: &mut ManagedKvCoordinatorSnapshot,
    arena: &ManagedKvCoordinatorSnapshot,
) {
    totals.capacity_pages = totals.capacity_pages.saturating_add(arena.capacity_pages);
    totals.allocated_pages = totals.allocated_pages.saturating_add(arena.allocated_pages);
    totals.free_pages = totals.free_pages.saturating_add(arena.free_pages);
    totals.table_refs = totals.table_refs.saturating_add(arena.table_refs);
    totals.prefix_refs = totals.prefix_refs.saturating_add(arena.prefix_refs);
    totals.execution_pins = totals.execution_pins.saturating_add(arena.execution_pins);
    totals.transfer_pins = totals.transfer_pins.saturating_add(arena.transfer_pins);
    totals.reservations = totals.reservations.saturating_add(arena.reservations);
    totals.active_transactions = totals
        .active_transactions
        .saturating_add(arena.active_transactions);
}

pub(super) fn managed_backend_runtime(
    backend: BackendKind,
    device: &Device,
) -> (Option<Arc<dyn KvBackendRuntime>>, Option<String>) {
    let wrong_device = || {
        (
            None,
            Some(format!(
                "managed {backend:?} KV cannot bind worker device {:?}",
                device.location()
            )),
        )
    };
    match backend {
        BackendKind::Cpu => {
            if !device.is_cpu() {
                return wrong_device();
            }
            (Some(Arc::new(CpuKvBackendRuntime)), None)
        }
        BackendKind::Metal => {
            if !device.is_metal() {
                return wrong_device();
            }
            #[cfg(feature = "metal")]
            {
                match MetalKvBackendRuntime::new(device.clone()) {
                    Ok(runtime) => (Some(Arc::new(runtime)), None),
                    Err(error) => (None, Some(error.to_string())),
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                (
                    None,
                    Some(
                        "managed Metal KV requires the metal feature and direct paged attention"
                            .to_string(),
                    ),
                )
            }
        }
        BackendKind::Cuda => {
            if !device.is_cuda() {
                return wrong_device();
            }
            #[cfg(feature = "cuda")]
            {
                match CudaKvBackendRuntime::new(device.clone()) {
                    Ok(runtime) => (Some(Arc::new(runtime)), None),
                    Err(error) => (None, Some(error.to_string())),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                (
                    None,
                    Some(
                        "managed CUDA KV requires the cuda feature and direct paged attention"
                            .to_string(),
                    ),
                )
            }
        }
    }
}

pub(super) fn managed_device_ordinal(device: &Device) -> Option<u32> {
    match device.location() {
        DeviceLocation::Cpu => None,
        DeviceLocation::Cuda { gpu_id } => u32::try_from(gpu_id).ok(),
        // Candle reports Metal's registry id rather than the selector ordinal.
        // Fold the exact device identity into the plan's compact device tag;
        // the runtime itself retains the exact Candle device handle.
        DeviceLocation::Metal { gpu_id } => {
            let id = gpu_id as u64;
            Some((id ^ (id >> 32)) as u32)
        }
    }
}

/// During the declaration migration both planners inspect the same semantic
/// model geometry. Refuse Ready publication unless v2 resolves to the exact
/// physical page shape already allocated by the proven arena implementation.
/// This check disappears together with the v1 planner after the final model
/// declaration is native v2.
fn validate_v2_physical_equivalence(legacy: &ResolvedKvPlan, v2: &ResolvedStatePlan) -> Result<()> {
    if legacy.backend != v2.backend
        || legacy.device_ordinal != v2.device_ordinal
        || legacy.groups.len() != v2.paged_attention.len()
    {
        return Err(Error::ModelLoadError(
            "v2 state plan does not match the allocating KV backend".to_string(),
        ));
    }

    for (legacy_group, v2_group) in legacy.groups.iter().zip(&v2.paged_attention) {
        let ResolvedKvGroupKind::PagedAttention { layers } = &legacy_group.kind else {
            return Err(Error::ModelLoadError(
                "v1 model-state groups require a native v2 backend allocator".to_string(),
            ));
        };
        let same_layers = layers.len() == v2_group.layers.len()
            && layers.iter().zip(&v2_group.layers).all(|(left, right)| {
                left.model_layer == right.model_layer && left.physical_layer == right.physical_layer
            });
        if legacy_group.page_tokens != v2_group.page_tokens
            || legacy_group.bytes_per_page != v2_group.bytes_per_page
            || !same_layers
        {
            return Err(Error::ModelLoadError(
                "v2 state plan resolved different physical KV geometry".to_string(),
            ));
        }
    }
    Ok(())
}

fn paged_only_contract(contract: &KvCacheContract) -> Result<KvCacheContract> {
    let paged = KvCacheContract {
        abi: contract.abi,
        domains: contract
            .domains
            .iter()
            .filter(|domain| matches!(domain, KvDomainSpec::PagedAttention(_)))
            .cloned()
            .collect(),
    };
    if paged.domains.is_empty() {
        return Err(Error::ModelLoadError(
            "retained tensor-only state requires the native v2 runtime loader".into(),
        ));
    }
    paged.validate()?;
    Ok(paged)
}

fn plan_physical_bytes(plan: &ResolvedKvPlan) -> Result<u64> {
    plan.groups.iter().try_fold(0_u64, |total, group| {
        let group_bytes = group
            .bytes_per_page
            .checked_mul(u64::from(group.capacity_pages))
            .ok_or_else(|| Error::Overloaded("managed KV byte total overflow".into()))?;
        total
            .checked_add(group_bytes)
            .ok_or_else(|| Error::Overloaded("managed KV byte total overflow".into()))
    })
}

fn managed_arena_resources(backend: BackendKind, bytes: u64) -> ResourceVector {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(bytes),
        BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(bytes),
        BackendKind::Cuda => resources.device_bytes = ResourceAmount::Known(bytes),
    }
    resources
}

fn reserve_managed_arena(
    authority: &Arc<ResourceAuthority>,
    model_instance: ModelInstanceId,
    backend: BackendKind,
    resources: ResourceVector,
) -> Result<ResourceLease> {
    let owner = ReservationOwner::new(
        ReservationClass::Model,
        format!("managed-kv:{}:{backend:?}", model_instance.get()),
    );
    match backend {
        BackendKind::Cpu | BackendKind::Metal => authority.track_advisory(owner, resources),
        BackendKind::Cuda => authority.reserve(owner, resources),
    }
}

fn ensure_session_tables(state: &mut ManagedKvModelState, session: &SessionKey) -> Result<()> {
    if !state.registered_sessions.insert(session.clone()) {
        return Ok(());
    }
    let mut registered = Vec::new();
    for group in &state.runtime.plan.groups {
        let coordinator = state
            .coordinators
            .get_mut(&group.arena)
            .expect("resolved arena has a coordinator");
        if let Err(error) = coordinator.register_table(session.clone(), group.domain) {
            for (arena, domain) in registered {
                let _ = state
                    .coordinators
                    .get_mut(&arena)
                    .expect("registered arena exists")
                    .release_table(session, domain);
            }
            state.registered_sessions.remove(session);
            return Err(coordinator_error(error));
        }
        registered.push((group.arena, group.domain));
    }
    Ok(())
}

fn managed_prefix_namespace(
    request: Option<&EngineCoreRequest>,
    runtime: &ManagedKvModelRuntime,
    cache_salt: Option<[u8; 32]>,
) -> Result<Option<KvPrefixNamespace>> {
    let (Some(request), Some(cache_salt)) = (request, cache_salt) else {
        return Ok(None);
    };
    let Some(binding) = request.execution_adapter_binding() else {
        return Ok(None);
    };
    if request.task_type != crate::engine::TaskType::Chat
        || request.model_instance_id() != Some(runtime.plan.model_instance)
        || binding.model_instance_id != runtime.plan.model_instance
    {
        return Ok(None);
    }

    // The current managed producer is text-only Qwen3. Each digest is derived
    // from exact lifecycle/adapter facts already sealed onto this request. The
    // model-generation fence deliberately prevents reuse across reloads until
    // artifact revisions and tokenizer ABIs become first-class lifecycle data.
    let model_revision = digest_parts(
        b"izwi.kv.loaded-model-generation.v1\0",
        &[&runtime.plan.model_instance.get().to_le_bytes()],
    );
    let adapter_abi = digest_parts(
        b"izwi.kv.loaded-adapter-abi.v1\0",
        &[
            &binding.adapter_instance_id.get().to_le_bytes(),
            &binding.adapter_abi_revision.get().to_le_bytes(),
            binding.capability_id.as_bytes(),
        ],
    );
    let tokenizer_or_input_encoding = digest_parts(
        b"izwi.kv.loaded-input-encoding.v1\0",
        &[
            &runtime.plan.model_instance.get().to_le_bytes(),
            &binding.adapter_instance_id.get().to_le_bytes(),
            binding.capability_id.as_bytes(),
        ],
    );
    let position_semantics = digest_parts(
        b"izwi.kv.position-semantics.v1\0",
        &[&runtime.plan.contract_fingerprint],
    );
    Ok(Some(KvPrefixNamespace {
        model_instance: runtime.plan.model_instance,
        model_revision,
        adapter_abi,
        tokenizer_or_input_encoding,
        position_semantics,
        plan: runtime.plan.fingerprint(),
        multimodal_artifact: None,
        cache_salt,
    }))
}

fn digest_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn prefix_enabled_for_domain(
    contract: &KvCacheContract,
    domain_id: crate::kv::CacheDomainId,
) -> bool {
    contract.domains.iter().any(|domain| {
        if domain.id() != domain_id {
            return false;
        }
        match domain {
            KvDomainSpec::PagedAttention(spec) => {
                spec.layers
                    .iter()
                    .all(|layer| matches!(layer.attention, AttentionSemantics::Full))
                    && matches!(
                        spec.prefix_semantics,
                        KvPrefixSemantics::CommittedFullPages { .. }
                    )
            }
            KvDomainSpec::ModelState(spec) => matches!(
                spec.prefix_semantics,
                KvPrefixSemantics::CommittedFullPages { .. }
            ),
        }
    })
}

fn sliding_window_for_domain(
    contract: &KvCacheContract,
    domain_id: crate::kv::CacheDomainId,
) -> Result<Option<u32>> {
    let Some(domain) = contract
        .domains
        .iter()
        .find(|domain| domain.id() == domain_id)
    else {
        return Err(Error::InferenceError(
            "managed KV plan references a missing semantic domain".into(),
        ));
    };
    let KvDomainSpec::PagedAttention(spec) = domain else {
        return Ok(None);
    };
    let first = spec.layers.first().ok_or_else(|| {
        Error::InvalidInput("managed paged-attention domain has no layers".into())
    })?;
    if spec
        .layers
        .iter()
        .any(|layer| layer.attention != first.attention)
    {
        return Err(Error::InvalidInput(format!(
            "managed KV domain {} mixes full and sliding attention semantics",
            domain_id.get()
        )));
    }
    Ok(match first.attention {
        AttentionSemantics::Full => None,
        AttentionSemantics::SlidingWindow { window_tokens } => Some(window_tokens),
    })
}

fn validate_sliding_contract(contract: &KvCacheContract, backend: BackendKind) -> Result<()> {
    for domain in &contract.domains {
        if let Some(window_tokens) = sliding_window_for_domain(contract, domain.id())? {
            if backend == BackendKind::Cuda {
                return Err(Error::InvalidInput(format!(
                    "managed CUDA KV cannot safely consume sliding window {window_tokens}: its paged kernel ABI has no first-page offset"
                )));
            }
        }
    }
    Ok(())
}

fn prefix_publications(
    namespace: &KvPrefixNamespace,
    tokens: &[u32],
    page_tokens: u32,
    execution_start_tokens: u32,
    target_tokens: u32,
    reservation: &ManagedCacheDomainReservation,
    group: KvGroupId,
) -> Result<Vec<KvPrefixPublication>> {
    let page_tokens_usize = usize::try_from(page_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix page size exceeds usize".into()))?;
    let target = usize::try_from(target_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix target exceeds usize".into()))?;
    if target > tokens.len() || page_tokens_usize == 0 {
        return Ok(Vec::new());
    }
    let table = reservation
        .provisional_groups
        .iter()
        .find(|table| table.group == group)
        .ok_or_else(|| Error::InferenceError("managed prefix group table is missing".into()))?;
    let first_new_page = usize::try_from(execution_start_tokens / page_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix start exceeds usize".into()))?;
    let complete_pages = target / page_tokens_usize;
    let mut previous = None;
    let mut publications = Vec::new();
    for page_index in 0..complete_pages {
        let start = page_index * page_tokens_usize;
        let key = KvPrefixPageKey::new(
            namespace,
            previous,
            start as u64,
            tokens[start..start + page_tokens_usize].to_vec(),
        )
        .map_err(prefix_error)?;
        previous = Some(key.digest());
        if page_index < first_new_page {
            continue;
        }
        let block = table.blocks.get(page_index).copied().ok_or_else(|| {
            Error::InferenceError("managed prefix publication exceeds its block table".into())
        })?;
        publications.push(KvPrefixPublication { key, block });
    }
    Ok(publications)
}

fn reservation_for_group(
    group: KvGroupId,
    page_tokens: u32,
    snapshot: &KvSnapshot,
    target_tokens: u32,
    shared_prefix: &[crate::kv::CacheBlockRef],
) -> Result<KvGroupReservation> {
    let required_pages = target_tokens
        .checked_add(page_tokens - 1)
        .ok_or_else(|| Error::Overloaded("managed KV page count overflow".into()))?
        / page_tokens;
    let required_pages = usize::try_from(required_pages)
        .map_err(|_| Error::Overloaded("managed KV page count exceeds usize".into()))?;
    let existing = snapshot
        .groups
        .iter()
        .find(|table| table.group == group)
        .map(|table| table.blocks.as_slice())
        .unwrap_or_default();
    let mut blocks = Vec::with_capacity(required_pages);
    if snapshot.committed_tokens == 0 {
        blocks.extend(shared_prefix.iter().copied().map(KvBlockIntent::Shared));
    } else if !shared_prefix.is_empty() {
        return Err(Error::InferenceError(
            "managed KV prefix pages cannot replace a committed request table".into(),
        ));
    }
    for (index, block) in existing.iter().take(required_pages).copied().enumerate() {
        let is_partial_tail = snapshot.committed_tokens % page_tokens != 0
            && index + 1 == existing.len()
            && target_tokens > snapshot.committed_tokens;
        blocks.push(if is_partial_tail {
            KvBlockIntent::Writable(block)
        } else {
            KvBlockIntent::Existing(block)
        });
    }
    blocks.extend(
        std::iter::repeat(KvBlockIntent::Fresh).take(required_pages.saturating_sub(blocks.len())),
    );
    Ok(KvGroupReservation { group, blocks })
}

fn abort_domains(
    state: &mut ManagedKvModelState,
    txn_id: PlanId,
    domains: &[ManagedCacheDomainReservation],
) {
    for domain in domains {
        let _ = state
            .coordinators
            .get_mut(&domain.arena)
            .expect("reservation arena has a coordinator")
            .abort(txn_id);
    }
}

fn abort_reservation(state: &mut ManagedKvModelState, reservation: &ManagedCacheReservation) {
    state.pending_prefixes.remove(&reservation.txn_id);
    abort_domains(state, reservation.txn_id, &reservation.domains);
    if reservation.tensor_state.is_some() {
        if let (Some(arena), Ok(transaction)) = (
            state.runtime.tensor_state(),
            PhysicalStateTransactionId::new(reservation.txn_id),
        ) {
            let _ = arena.abort(transaction);
        }
    }
}

fn arena_config(
    contract: &KvCacheContract,
    group: &crate::kv::ResolvedKvGroup,
) -> Result<KvArenaConfig> {
    let domain = contract
        .domains
        .iter()
        .find(|domain| domain.id() == group.domain)
        .ok_or_else(|| {
            Error::InferenceError("resolved KV group lost its semantic domain".into())
        })?;
    let (spec, bindings) = match (domain, &group.kind) {
        (KvDomainSpec::PagedAttention(spec), ResolvedKvGroupKind::PagedAttention { layers }) => {
            (spec, layers)
        }
        _ => {
            return Err(Error::InvalidInput(
                "dense paged KV arena requires a paged-attention domain".to_string(),
            ));
        }
    };
    let mut layers = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let layer = spec
            .layers
            .iter()
            .find(|layer| layer.model_layer == binding.model_layer)
            .ok_or_else(|| Error::InferenceError("resolved KV layer binding is stale".into()))?;
        layers.push(KvLayerConfig {
            binding: *binding,
            num_kv_heads: layer.num_kv_heads,
            key_head_dim: layer.key_head_dim,
            value_head_dim: layer.value_head_dim,
        });
    }
    Ok(KvArenaConfig {
        id: group.arena,
        group: group.id,
        page_tokens: group.page_tokens,
        capacity_pages: group.capacity_pages,
        dtype: candle_dtype(group.storage.dtype())?,
        layers,
    })
}

fn candle_dtype(dtype: KvStorageDType) -> Result<DType> {
    match dtype {
        KvStorageDType::F32 => Ok(DType::F32),
        KvStorageDType::F16 => Ok(DType::F16),
        KvStorageDType::Bf16 => Ok(DType::BF16),
        KvStorageDType::I8 | KvStorageDType::Q4 => Err(Error::InvalidInput(
            "dense KV arena cannot allocate quantized storage".to_string(),
        )),
    }
}

fn coordinator_error(error: impl fmt::Display) -> Error {
    Error::InferenceError(format!(
        "managed KV coordinator rejected transaction: {error}"
    ))
}

fn prefix_error(error: impl fmt::Display) -> Error {
    Error::InferenceError(format!(
        "managed KV prefix index rejected operation: {error}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{
        PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue,
    };
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, CapacitySource, ExecutionAdapterBinding,
        ExecutionGroupId, ExecutionMode, ExecutionProfile, InputRange, NativeBatchMode,
        PhysicalCapacityProvider, PhysicalCapacitySnapshot, SequencePhase, StageDescriptor,
        StageId,
    };
    use crate::kv::{test_contract, CacheBlockRef, KvSlotRef};
    use crate::kv::{
        CacheTokenAxis, KvDomainSpec as LegacyDomainSpec, KvStorageRequest, ModelStateDomainSpec,
        ModelStateKind, ModelStateLayerSpec,
    };
    use crate::model::ModelVariant;
    use crate::models::shared::chat::{ChatMessage, ChatRole};
    use candle_core::Tensor;

    #[derive(Debug)]
    struct TestCapacityProvider {
        snapshot: PhysicalCapacitySnapshot,
    }

    impl PhysicalCapacityProvider for TestCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
        }
    }

    fn authority_with_capacity(bytes: u64) -> Arc<ResourceAuthority> {
        let capacity = ResourceVector {
            host_bytes: ResourceAmount::Known(bytes),
            device_bytes: ResourceAmount::Known(bytes),
            unified_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        };
        Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            },
        })))
    }

    #[test]
    fn managed_runtime_rejects_a_backend_device_mismatch() {
        let mut manager = ManagedKvCacheManager::for_worker(None, BackendKind::Metal, Device::Cpu);
        let error = manager
            .bind_request(
                ModelInstanceId::new(700),
                BackendKind::Metal,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("cannot bind worker device"));
        assert_eq!(manager.model_count(), 0);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn managed_metal_runtime_allocates_on_the_exact_worker_device() -> Result<()> {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return Ok(());
        };
        let expected_location = device.location();
        let mut manager =
            ManagedKvCacheManager::for_worker(None, BackendKind::Metal, device.clone());
        let runtime = manager
            .bind_request(
                ModelInstanceId::new(701),
                BackendKind::Metal,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )?
            .expect("managed contract should activate on compiled Metal");
        assert_eq!(runtime.plan().backend, BackendKind::Metal);
        assert_eq!(
            runtime.plan().device_ordinal,
            managed_device_ordinal(&device)
        );
        for group in &runtime.plan().groups {
            let arena = runtime.arena(group.arena).expect("resolved arena");
            assert_eq!(arena.backend_kind(), BackendKind::Metal);
            assert_eq!(arena.device_location(), expected_location);
        }
        Ok(())
    }

    fn sequence_work(start: usize, end: usize) -> WorkUnit {
        WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange { start, end },
            max_output_steps: end.saturating_sub(start).max(1),
        }
    }

    fn sliding_contract(window_tokens: u32) -> KvCacheContract {
        let mut contract = test_contract();
        let KvDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        for layer in &mut domain.layers {
            layer.attention = AttentionSemantics::SlidingWindow { window_tokens };
        }
        domain.prefix_semantics = KvPrefixSemantics::Disabled;
        contract
    }

    fn composite_tensor_contract() -> KvCacheContract {
        let mut contract = test_contract();
        contract
            .domains
            .push(LegacyDomainSpec::ModelState(ModelStateDomainSpec {
                id: CacheDomainId::new(2),
                token_axis: CacheTokenAxis::DecoderTokens,
                layers: vec![ModelStateLayerSpec {
                    model_layer: 0,
                    kind: ModelStateKind::Recurrent,
                    elements_per_sequence: 4,
                }],
                storage: KvStorageRequest {
                    dtypes: vec![KvStorageDType::F32],
                    allow_quantized: false,
                },
                prefix_semantics: KvPrefixSemantics::Disabled,
            }));
        contract.validate().unwrap();
        contract
    }

    fn prefix_request(model: ModelInstanceId, tokens: Vec<u32>) -> EngineCoreRequest {
        let variant = ModelVariant::Qwen306B;
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
        let stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "qwen3.managed",
            &profile,
            NativeBatchMode::None,
        );
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "prefix".into(),
        }])
        .with_model_variant(variant);
        request.prompt_tokens = tokens;
        request
            .bind_execution_adapter(ExecutionAdapterBinding {
                execution_group_id: ExecutionGroupId::new(1),
                model_instance_id: model,
                adapter_instance_id: AdapterInstanceId::new(2),
                adapter_abi_revision: AdapterAbiRevision::new(9),
                model_variant: variant,
                capability_id: "chat".into(),
                stages: Arc::from([stage]),
            })
            .expect("adapter binding");
        request
    }

    #[test]
    fn live_manager_commits_aborts_and_releases_exact_session_tables() {
        let model = ModelInstanceId::new(41);
        let session = SessionKey::new("managed-live".to_string(), 7);
        let domain = CacheDomainId::new(1);
        let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([5; 32]));
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                4,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("bind")
            .expect("managed runtime");
        assert!(runtime.physical_bytes() > 0);

        let first = manager
            .prepare(&runtime, 1, &session, &sequence_work(0, 5), None)
            .expect("prepare")
            .expect("reservation");
        assert_eq!(first.domains.len(), 1);
        assert_eq!(first.domains[0].writable_blocks.len(), 1);
        manager
            .finalize(
                &first,
                Some(&first.completed_write_receipt_for_test()),
                true,
            )
            .expect("commit");
        let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.committed_tokens, 5);

        let second = manager
            .prepare(&runtime, 2, &session, &sequence_work(5, 17), None)
            .expect("prepare")
            .expect("reservation");
        assert_eq!(second.domains[0].writable_blocks.len(), 2);
        manager.finalize(&second, None, false).expect("abort");
        let unchanged = manager.snapshot(model, &session, domain).expect("snapshot");
        assert_eq!(unchanged.version, 1);
        assert_eq!(unchanged.committed_tokens, 5);

        manager.release_session(&session).expect("release");
        assert!(manager.snapshot(model, &session, domain).is_none());
    }

    #[test]
    fn session_release_is_retryable_while_a_row_transaction_is_active() {
        let model = ModelInstanceId::new(53);
        let session = SessionKey::new("managed-release-retry".into(), 1);
        let domain = CacheDomainId::new(1);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .unwrap()
            .unwrap();
        let reservation = manager
            .prepare(&runtime, 41, &session, &sequence_work(0, 1), None)
            .unwrap()
            .unwrap();

        assert!(manager.release_session(&session).is_err());
        assert!(manager.models[&model]
            .registered_sessions
            .contains(&session));
        assert!(manager.snapshot(model, &session, domain).is_some());

        manager.finalize(&reservation, None, false).unwrap();
        manager.release_session(&session).unwrap();
        assert!(!manager.models[&model]
            .registered_sessions
            .contains(&session));
        assert!(manager.snapshot(model, &session, domain).is_none());
    }

    #[test]
    fn runtime_snapshot_reports_exact_physical_state_and_serializes() {
        let model = ModelInstanceId::new(44);
        let session = SessionKey::new("managed-telemetry".to_string(), 3);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("bind")
            .expect("managed runtime");
        let reservation = manager
            .prepare(&runtime, 11, &session, &sequence_work(0, 1), None)
            .expect("prepare")
            .expect("reservation");

        let prepared = manager.runtime_snapshot();
        assert_eq!(prepared.memory_accounting, "physical_arena_backing");
        assert_eq!(prepared.totals.models, 1);
        assert_eq!(prepared.totals.arenas, 1);
        assert_eq!(prepared.totals.physical_bytes, runtime.physical_bytes());
        assert_eq!(prepared.totals.coordinator.capacity_pages, 2);
        assert_eq!(prepared.totals.coordinator.allocated_pages, 1);
        assert_eq!(prepared.totals.coordinator.active_transactions, 1);
        assert_eq!(prepared.totals.operations.page_zero_dispatches, 1);
        assert_eq!(prepared.counters.pages_zeroed, 1);
        assert_eq!(prepared.counters.backing_allocations, 1);
        assert_eq!(prepared.models[0].model_instance, model);
        assert_eq!(
            prepared.models[0].arenas[0].physical_bytes,
            runtime.physical_bytes()
        );

        let encoded = serde_json::to_value(&prepared).expect("serialize managed KV telemetry");
        assert_eq!(encoded["memory_accounting"], "physical_arena_backing");
        assert_eq!(encoded["totals"]["coordinator"]["allocated_pages"], 1);
        assert_eq!(encoded["models"][0]["backend"], "cpu");

        manager.finalize(&reservation, None, false).expect("abort");
        let aborted = manager.runtime_snapshot();
        assert_eq!(aborted.counters.transaction_aborts, 1);
        assert_eq!(aborted.totals.coordinator.allocated_pages, 0);
        assert_eq!(aborted.totals.coordinator.active_transactions, 0);
    }

    #[test]
    fn live_sliding_window_table_stays_bounded_and_carries_first_page_offset() {
        let model = ModelInstanceId::new(48);
        let session = SessionKey::new("managed-window".into(), 1);
        let domain = CacheDomainId::new(1);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                4,
                16,
                &CacheCapability::Managed(sliding_contract(32)),
            )
            .expect("bind")
            .expect("managed runtime");

        let mut observed_nonzero_offset = false;
        for target in 1..=256_usize {
            let reservation = manager
                .prepare(
                    &runtime,
                    target as u64,
                    &session,
                    &sequence_work(target - 1, target),
                    None,
                )
                .expect("prepare")
                .expect("reservation");
            let row = &reservation.domains[0];
            assert_eq!(
                row.target_window_start,
                u32::try_from(target.saturating_sub(32)).unwrap()
            );
            assert_eq!(row.first_page_offset, row.target_window_start % 16);
            observed_nonzero_offset |= row.first_page_offset != 0;
            assert!(row.provisional_groups[0].blocks.len() <= 3);
            manager
                .finalize(
                    &reservation,
                    Some(&reservation.completed_write_receipt_for_test()),
                    true,
                )
                .expect("commit");
        }

        assert!(observed_nonzero_offset);
        let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
        assert_eq!(snapshot.committed_tokens, 256);
        assert_eq!(snapshot.window_start, 224);
        assert!(snapshot.groups[0].blocks.len() <= 3);
    }

    #[test]
    fn salted_prefix_reuse_attaches_shared_pages_and_skips_their_prefill() {
        let model = ModelInstanceId::new(45);
        let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([9; 32]));
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                4,
                32,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("bind")
            .expect("runtime");
        let tokens = (0..65).collect::<Vec<u32>>();
        let first_request = prefix_request(model, tokens.clone());
        let first_session = SessionKey::new("prefix-first".into(), 1);
        let first = manager
            .prepare(
                &runtime,
                21,
                &first_session,
                &sequence_work(0, tokens.len()),
                Some(&first_request),
            )
            .expect("first prepare")
            .expect("first reservation");
        assert_eq!(first.domains[0].execution_start_tokens, 0);
        manager
            .finalize(
                &first,
                Some(&first.completed_write_receipt_for_test()),
                true,
            )
            .expect("first commit");
        manager
            .release_session(&first_session)
            .expect("first release");

        let mut second_tokens = tokens;
        *second_tokens.last_mut().unwrap() = 999;
        let second_request = prefix_request(model, second_tokens.clone());
        let second_session = SessionKey::new("prefix-second".into(), 1);
        let second = manager
            .prepare(
                &runtime,
                22,
                &second_session,
                &sequence_work(0, second_tokens.len()),
                Some(&second_request),
            )
            .expect("second prepare")
            .expect("second reservation");
        assert_eq!(second.domains[0].execution_start_tokens, 64);
        assert_eq!(second.domains[0].writable_blocks.len(), 1);
        let telemetry = manager.telemetry_snapshot();
        assert_eq!(telemetry.prefix_hits, 1);
        assert_eq!(telemetry.reused_tokens, 64);
        assert_eq!(telemetry.avoided_prefill_tokens, 64);
        manager.finalize(&second, None, false).expect("abort");
        assert_eq!(manager.telemetry_snapshot().transaction_aborts, 1);
    }

    #[test]
    fn managed_prefix_reuse_is_disabled_without_an_explicit_salt() {
        let model = ModelInstanceId::new(46);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                4,
                32,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("bind")
            .expect("runtime");
        let request = prefix_request(model, (0..33).collect());
        let session = SessionKey::new("prefix-disabled".into(), 1);
        let reservation = manager
            .prepare(
                &runtime,
                23,
                &session,
                &sequence_work(0, 33),
                Some(&request),
            )
            .expect("prepare")
            .expect("reservation");
        assert_eq!(reservation.domains[0].execution_start_tokens, 0);
        assert_eq!(manager.telemetry_snapshot().prefix_hits, 0);
        assert_eq!(manager.telemetry_snapshot().prefix_misses, 0);
    }

    #[test]
    fn aborted_managed_prefill_never_publishes_prefix_pages() {
        let model = ModelInstanceId::new(47);
        let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([7; 32]));
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                3,
                32,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("bind")
            .expect("runtime");
        let request = prefix_request(model, (100..133).collect());
        let first_session = SessionKey::new("aborted-prefix".into(), 1);
        let first = manager
            .prepare(
                &runtime,
                24,
                &first_session,
                &sequence_work(0, 33),
                Some(&request),
            )
            .expect("prepare")
            .expect("reservation");
        manager.finalize(&first, None, false).expect("abort");
        manager.release_session(&first_session).expect("release");

        let second_session = SessionKey::new("after-abort".into(), 1);
        let second = manager
            .prepare(
                &runtime,
                25,
                &second_session,
                &sequence_work(0, 33),
                Some(&request),
            )
            .expect("prepare after abort")
            .expect("reservation");
        assert_eq!(second.domains[0].execution_start_tokens, 0);
        let telemetry = manager.telemetry_snapshot();
        assert_eq!(telemetry.prefix_hits, 0);
        assert_eq!(telemetry.prefix_misses, 2);
    }

    #[test]
    fn one_model_instance_cannot_change_contract_after_arena_allocation() {
        let model = ModelInstanceId::new(42);
        let mut manager = ManagedKvCacheManager::default();
        manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("first binding");
        let mut changed = test_contract();
        if let KvDomainSpec::PagedAttention(domain) = &mut changed.domains[0] {
            domain.layers[0].num_kv_heads = 2;
        }
        assert!(manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(changed),
            )
            .is_err());
    }

    #[test]
    fn composite_domain_receipt_publishes_every_table_under_one_row_fence() {
        let model = ModelInstanceId::new(43);
        let session = SessionKey::new("managed-composite".to_string(), 3);
        let mut contract = test_contract();
        let mut second = contract.domains[0].clone();
        if let KvDomainSpec::PagedAttention(domain) = &mut second {
            domain.id = CacheDomainId::new(2);
        }
        contract.domains.push(second);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(contract),
            )
            .expect("bind")
            .expect("runtime");
        let reservation = manager
            .prepare(&runtime, 8, &session, &sequence_work(0, 8), None)
            .expect("prepare")
            .expect("reservation");
        assert_eq!(reservation.domains.len(), 2);
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .expect("composite commit");
        for domain in [CacheDomainId::new(1), CacheDomainId::new(2)] {
            let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
            assert_eq!(snapshot.version, 1);
            assert_eq!(snapshot.committed_tokens, 8);
        }
    }

    #[test]
    fn composite_domain_failure_publishes_no_table() {
        let model = ModelInstanceId::new(45);
        let session = SessionKey::new("managed-composite-failure".to_string(), 1);
        let mut contract = test_contract();
        let mut second = contract.domains[0].clone();
        if let KvDomainSpec::PagedAttention(domain) = &mut second {
            domain.id = CacheDomainId::new(2);
        }
        contract.domains.push(second);
        let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([5; 32]));
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(contract),
            )
            .expect("bind")
            .expect("runtime");
        let request = prefix_request(model, (0..16).collect());
        let reservation = manager
            .prepare(&runtime, 9, &session, &sequence_work(0, 16), Some(&request))
            .expect("prepare")
            .expect("reservation");
        let receipt = reservation.completed_write_receipt_for_test();
        let state = manager.models.get_mut(&model).expect("model state");
        let pending = state
            .pending_prefixes
            .get_mut(&9)
            .expect("pending prefixes");
        assert_eq!(pending.len(), 2);
        assert!(!pending[1].publications.is_empty());
        pending[1].publications[0].block = reservation.domains[0].writable_blocks[0];

        assert!(manager
            .finalize(&reservation, Some(&receipt), true)
            .is_err());
        for domain in [CacheDomainId::new(1), CacheDomainId::new(2)] {
            let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
            assert_eq!(snapshot.version, 0);
            assert_eq!(snapshot.committed_tokens, 0);
        }
        assert!(manager
            .runtime_snapshot()
            .models
            .iter()
            .flat_map(|model| &model.arenas)
            .all(|arena| arena.coordinator.active_transactions == 0));
    }

    #[test]
    fn paged_and_tensor_state_commit_or_abort_under_one_row_fence() {
        let model = ModelInstanceId::new(52);
        let session = SessionKey::new("managed-tensor-composite".into(), 1);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(composite_tensor_contract()),
            )
            .unwrap()
            .unwrap();
        let arena = runtime.tensor_state().unwrap().clone();
        let tensor_domain = crate::kv::v2::StateDomainId::new(3);

        let aborted = manager
            .prepare(&runtime, 31, &session, &sequence_work(0, 1), None)
            .unwrap()
            .unwrap();
        let sequence =
            PhysicalStateSequenceId::new(aborted.tensor_state.unwrap().sequence).unwrap();
        let transaction = PhysicalStateTransactionId::new(aborted.txn_id).unwrap();
        arena
            .stage_replace(
                transaction,
                tensor_domain,
                0,
                1,
                vec![StateComponentValue {
                    component: crate::kv::v2::StateComponentId::new(1),
                    tensor: Some(Tensor::from_slice(&[1.0_f32], 1, &Device::Cpu).unwrap()),
                }],
            )
            .unwrap();
        manager.finalize(&aborted, None, false).unwrap();
        assert!(arena.read(sequence, tensor_domain).unwrap().is_none());
        assert_eq!(
            manager
                .snapshot(model, &session, CacheDomainId::new(1))
                .unwrap()
                .committed_tokens,
            0
        );

        let committed = manager
            .prepare(&runtime, 32, &session, &sequence_work(0, 1), None)
            .unwrap()
            .unwrap();
        arena
            .stage_replace(
                PhysicalStateTransactionId::new(committed.txn_id).unwrap(),
                tensor_domain,
                0,
                1,
                vec![StateComponentValue {
                    component: crate::kv::v2::StateComponentId::new(1),
                    tensor: Some(Tensor::from_slice(&[2.0_f32], 1, &Device::Cpu).unwrap()),
                }],
            )
            .unwrap();
        manager
            .finalize(
                &committed,
                Some(&committed.completed_write_receipt_for_test()),
                true,
            )
            .unwrap();
        assert_eq!(
            arena
                .read(sequence, tensor_domain)
                .unwrap()
                .unwrap()
                .components[0]
                .tensor
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.0]
        );
        assert_eq!(
            manager
                .snapshot(model, &session, CacheDomainId::new(1))
                .unwrap()
                .committed_tokens,
            1
        );
    }

    #[test]
    fn arena_accounting_is_once_per_model_and_survives_session_release() {
        let model = ModelInstanceId::new(44);
        let session = SessionKey::new("managed-accounting".to_string(), 1);
        let authority = authority_with_capacity(u64::MAX);
        let mut manager = ManagedKvCacheManager::new(Some(authority.clone()));
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("bind")
            .expect("runtime");
        let physical_bytes = runtime.physical_bytes();
        assert_eq!(authority.snapshot().reservations, 1);
        assert_eq!(
            authority.snapshot().reserved.host_bytes,
            ResourceAmount::Known(physical_bytes)
        );

        let same = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("repeat bind")
            .expect("same runtime");
        assert_eq!(same.plan().id, runtime.plan().id);
        assert_eq!(authority.snapshot().reservations, 1);

        let reservation = manager
            .prepare(&runtime, 10, &session, &sequence_work(0, 1), None)
            .expect("prepare")
            .expect("reservation");
        manager.finalize(&reservation, None, false).expect("abort");
        drop(same);
        drop(runtime);
        assert!(manager.unload_model(model).is_err());
        manager.release_session(&session).expect("session release");
        assert_eq!(authority.snapshot().reservations, 1);
        assert_eq!(
            authority.snapshot().reserved.host_bytes,
            ResourceAmount::Known(physical_bytes)
        );

        assert!(manager.unload_model(model).expect("model unload"));
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(
            authority.snapshot().reserved.host_bytes,
            ResourceAmount::Known(0)
        );
    }

    #[test]
    fn replacement_arena_rejects_handles_from_the_unloaded_generation() {
        let model = ModelInstanceId::new(45);
        let mut manager = ManagedKvCacheManager::default();
        let old_runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("old bind")
            .expect("old runtime");
        let old_group = old_runtime.plan().groups[0].clone();
        assert!(manager.unload_model(model).is_err());
        drop(old_runtime);
        assert!(manager.unload_model(model).expect("old unload"));

        let replacement = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(test_contract()),
            )
            .expect("replacement bind")
            .expect("replacement runtime");
        let replacement_group = &replacement.plan().groups[0];
        assert_ne!(old_group.arena, replacement_group.arena);
        let stale = KvSlotRef {
            block: CacheBlockRef {
                arena: old_group.arena,
                group: old_group.id,
                index: 0,
                slot_generation: 1,
            },
            offset: 0,
        };
        assert!(replacement
            .arena(replacement_group.arena)
            .expect("replacement arena")
            .lower_slots(&[stale])
            .is_err());
    }

    #[test]
    fn cuda_arena_accounting_is_guarded_while_cpu_and_metal_are_advisory() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            let authority = authority_with_capacity(1);
            let resources = managed_arena_resources(backend, 2);
            let lease =
                reserve_managed_arena(&authority, ModelInstanceId::new(46), backend, resources)
                    .expect("advisory arena accounting");
            assert_eq!(lease.resources(), resources);
        }

        let authority = authority_with_capacity(1);
        assert!(reserve_managed_arena(
            &authority,
            ModelInstanceId::new(47),
            BackendKind::Cuda,
            managed_arena_resources(BackendKind::Cuda, 2),
        )
        .is_err());
    }
}
