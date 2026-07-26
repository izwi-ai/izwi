//! Lifecycle-owned physical inference-state allocations beyond retained KV.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, DeviceLocation};
use serde::Serialize;

use crate::backends::kv::{KvArenaConfig, KvBackendRuntime, KvLayerConfig};
use crate::backends::state::{
    negotiate_state_plan, PhysicalStateSequenceId, PhysicalStateTransactionId,
    StateBackendPlanRequest, StateBackendRegistry, StateComponentValue, StateDomainSnapshot,
    TensorStateArena,
};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::v2::{
    invocation_paged_workspace_backing_v2, InferenceStateContract, InvocationStateCapacity,
    InvocationWorkspaceBackingV2, InvocationWorkspaceDomain, ResolvedStatePlan, StateDType,
    StateDomainId, StateDomainSpec, StatePlanId, StateScope,
};
use crate::kv::{KvArenaId, KvGroupId, KvLayerBinding};

use super::invocation::{InvocationPagedKvPoolHandle, InvocationPagedKvPoolOwner};
use super::invocation_tensor::InvocationTensorPoolOwner;
use super::managed::{managed_backend_runtime, managed_device_ordinal};
use crate::engine::{
    AdapterInstanceId, ModelInstanceId, ReservationClass, ReservationOwner, ResourceAmount,
    ResourceAuthority, ResourceLease, ResourceVector, StageId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InvocationPhysicalKey {
    pub(crate) adapter_instance: AdapterInstanceId,
    pub(crate) stage_graph: [u8; 32],
    pub(crate) stage: StageId,
    pub(crate) domain: StateDomainId,
}

struct OwnedInvocationPool {
    owner: InvocationPagedKvPoolOwner,
    resource_lease: Option<ResourceLease>,
    resources: ResourceVector,
}

struct OwnedInvocationTensorPool {
    owner: InvocationTensorPoolOwner,
    resource_lease: Option<ResourceLease>,
    resources: ResourceVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct RetainedTensorStateRuntimeIdV2 {
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) allocation_generation: u32,
    pub(crate) state_plan: StatePlanId,
}

/// Lifecycle-owned physical runtime for a retained contract containing only
/// transactional tensor/append/ring/static-tensor domains.
pub(crate) struct RetainedTensorStateRuntimeV2 {
    id: RetainedTensorStateRuntimeIdV2,
    state_plan: Arc<ResolvedStatePlan>,
    arena: Arc<TensorStateArena>,
    next_sequence: AtomicU64,
    next_transaction: AtomicU64,
    active_sequences: AtomicU32,
    sequence_capacity: u32,
    maximum_bytes: u64,
}

impl std::fmt::Debug for RetainedTensorStateRuntimeV2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedTensorStateRuntimeV2")
            .field("id", &self.id)
            .field("sequence_capacity", &self.sequence_capacity)
            .field("maximum_bytes", &self.maximum_bytes)
            .finish_non_exhaustive()
    }
}

impl RetainedTensorStateRuntimeV2 {
    pub(crate) const fn id(&self) -> RetainedTensorStateRuntimeIdV2 {
        self.id
    }

    pub(crate) fn state_plan_v2(&self) -> &ResolvedStatePlan {
        &self.state_plan
    }

    pub(crate) const fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }

    pub(crate) const fn sequence_capacity(&self) -> u32 {
        self.sequence_capacity
    }

    pub(crate) fn register_sequence(&self) -> Result<PhysicalStateSequenceId> {
        self.active_sequences
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < self.sequence_capacity).then_some(active + 1)
            })
            .map_err(|_| invalid("retained tensor sequence capacity is exhausted"))?;
        let registered = (|| {
            let sequence = PhysicalStateSequenceId::new(next_identity(&self.next_sequence)?)?;
            self.arena.register(sequence)?;
            Ok(sequence)
        })();
        if registered.is_err() {
            self.active_sequences.fetch_sub(1, Ordering::AcqRel);
        }
        registered
    }

    pub(crate) fn begin_transaction(
        &self,
        sequence: PhysicalStateSequenceId,
    ) -> Result<PhysicalStateTransactionId> {
        let transaction = PhysicalStateTransactionId::new(next_identity(&self.next_transaction)?)?;
        self.arena.begin(transaction, sequence)?;
        Ok(transaction)
    }

    pub(crate) fn read(
        &self,
        sequence: PhysicalStateSequenceId,
        domain: StateDomainId,
    ) -> Result<Option<StateDomainSnapshot>> {
        self.arena.read(sequence, domain)
    }

    pub(crate) fn read_transaction_base(
        &self,
        transaction: PhysicalStateTransactionId,
        domain: StateDomainId,
    ) -> Result<Option<StateDomainSnapshot>> {
        self.arena.read_transaction_base(transaction, domain)
    }

    pub(crate) fn stage_replace(
        &self,
        transaction: PhysicalStateTransactionId,
        domain: StateDomainId,
        expected_cursor: u64,
        target_cursor: u64,
        components: Vec<StateComponentValue>,
    ) -> Result<()> {
        self.arena.stage_replace(
            transaction,
            domain,
            expected_cursor,
            target_cursor,
            components,
        )
    }

    pub(crate) fn commit_transaction(
        &self,
        transaction: PhysicalStateTransactionId,
        expected_cursor: u64,
    ) -> Result<()> {
        self.arena.commit(transaction, expected_cursor)
    }

    pub(crate) fn abort_transaction(&self, transaction: PhysicalStateTransactionId) -> Result<()> {
        self.arena.abort(transaction)
    }

    pub(crate) fn release_sequence(&self, sequence: PhysicalStateSequenceId) -> Result<()> {
        self.arena.release(sequence)?;
        self.active_sequences
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                active.checked_sub(1)
            })
            .map_err(|_| {
                Error::InferenceError(
                    "retained tensor sequence accounting underflowed after release".into(),
                )
            })?;
        Ok(())
    }

    fn close_and_validate_drained(&self) -> Result<()> {
        self.arena.close_and_validate_drained()
    }
}

struct OwnedRetainedTensorState {
    runtime: Arc<RetainedTensorStateRuntimeV2>,
    resource_lease: Option<ResourceLease>,
    resources: ResourceVector,
}

#[derive(Default)]
struct ModelPhysicalState {
    invocation_paged: HashMap<InvocationPhysicalKey, OwnedInvocationPool>,
    invocation_tensor: HashMap<InvocationPhysicalKey, OwnedInvocationTensorPool>,
    retained_tensor: Option<OwnedRetainedTensorState>,
}

/// Worker-local owner for capability-authored invocation state. Planning and
/// allocation happen while the model is Loading; request admission receives
/// weak generation handles only.
pub(crate) struct PhysicalStateManager {
    models: HashMap<ModelInstanceId, ModelPhysicalState>,
    resource_authority: Option<Arc<ResourceAuthority>>,
    next_allocation_generation: u32,
    worker_backend: BackendKind,
    worker_device: Device,
    worker_device_location: DeviceLocation,
    worker_device_ordinal: Option<u32>,
    backend_runtime: Option<Arc<dyn KvBackendRuntime>>,
    backend_unavailable: Option<String>,
}

impl PhysicalStateManager {
    pub(crate) fn for_worker(
        resource_authority: Option<Arc<ResourceAuthority>>,
        backend: BackendKind,
        device: Device,
    ) -> Self {
        let (backend_runtime, backend_unavailable) = managed_backend_runtime(backend, &device);
        Self {
            models: HashMap::new(),
            resource_authority,
            next_allocation_generation: 1,
            worker_backend: backend,
            worker_device: device.clone(),
            worker_device_location: device.location(),
            worker_device_ordinal: managed_device_ordinal(&device),
            backend_runtime,
            backend_unavailable,
        }
    }

    pub(crate) fn cpu(resource_authority: Option<Arc<ResourceAuthority>>) -> Self {
        Self::for_worker(resource_authority, BackendKind::Cpu, Device::Cpu)
    }

    pub(crate) fn allocate_retained_tensor(
        &mut self,
        model_instance: ModelInstanceId,
        contract: &InferenceStateContract,
        sequence_capacity: u32,
    ) -> Result<Arc<RetainedTensorStateRuntimeV2>> {
        contract.validate()?;
        if sequence_capacity == 0
            || contract.domains.is_empty()
            || contract
                .domains
                .iter()
                .any(|domain| domain.scope() != StateScope::Retained)
            || contract.domains.iter().any(|domain| {
                !matches!(
                    domain,
                    StateDomainSpec::Tensor(_)
                        | StateDomainSpec::Append(_)
                        | StateDomainSpec::Ring(_)
                        | StateDomainSpec::StaticTensor(_)
                )
            })
        {
            return Err(invalid(
                "retained tensor allocation requires non-zero sequence capacity and retained tensor, append, ring, or static-tensor domains only",
            ));
        }
        if let Some(existing) = self
            .models
            .get(&model_instance)
            .and_then(|model| model.retained_tensor.as_ref())
        {
            if existing.runtime.state_plan_v2().contract_fingerprint != contract.fingerprint()?
                || existing.runtime.sequence_capacity() != sequence_capacity
            {
                return Err(invalid(
                    "one model generation requested incompatible retained tensor allocation",
                ));
            }
            return Ok(existing.runtime.clone());
        }
        let state_plan = Arc::new(negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend: self.worker_backend,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: None,
                storage_dtype_hint: None,
            },
        )?);
        if !state_plan.paged_attention.is_empty() || state_plan.non_paged.is_empty() {
            return Err(invalid(
                "retained tensor allocation resolved an invalid physical domain set",
            ));
        }
        let per_sequence_bytes = state_plan
            .non_paged
            .iter()
            .try_fold(0_u64, |total, domain| {
                total
                    .checked_add(domain.maximum_bytes())
                    .ok_or_else(|| invalid("retained tensor byte bound overflow"))
            })?;
        let maximum_bytes = retained_capacity_bytes(per_sequence_bytes, sequence_capacity)?;
        let generation = self.next_allocation_generation;
        if generation == 0 {
            return Err(invalid("physical retained allocation generation exhausted"));
        }
        let next_generation = generation
            .checked_add(1)
            .ok_or_else(|| invalid("physical retained allocation generation overflow"))?;
        let resources = arena_resources(self.worker_backend, maximum_bytes);
        let resource_lease = self
            .resource_authority
            .as_ref()
            .map(|authority| {
                reserve_retained_tensor(authority, model_instance, self.worker_backend, resources)
            })
            .transpose()?;
        let arena = Arc::new(TensorStateArena::new(
            state_plan.clone(),
            self.worker_device.clone(),
        )?);
        let runtime = Arc::new(RetainedTensorStateRuntimeV2 {
            id: RetainedTensorStateRuntimeIdV2 {
                model_instance,
                allocation_generation: generation,
                state_plan: state_plan.id,
            },
            state_plan,
            arena,
            next_sequence: AtomicU64::new(1),
            next_transaction: AtomicU64::new(1),
            active_sequences: AtomicU32::new(0),
            sequence_capacity,
            maximum_bytes,
        });
        let model = self.models.entry(model_instance).or_default();
        if model
            .retained_tensor
            .replace(OwnedRetainedTensorState {
                runtime: runtime.clone(),
                resource_lease,
                resources,
            })
            .is_some()
        {
            return Err(invalid(
                "one model generation allocated retained tensor state twice",
            ));
        }
        self.next_allocation_generation = next_generation;
        Ok(runtime)
    }

    pub(crate) fn allocate_invocation_workspace(
        &mut self,
        model_instance: ModelInstanceId,
        key: InvocationPhysicalKey,
        contract: &InferenceStateContract,
        plan: Arc<ResolvedStatePlan>,
        workspace_domain: &InvocationWorkspaceDomain,
        slot_count: u32,
    ) -> Result<Arc<dyn InvocationWorkspaceBackingV2>> {
        validate_invocation_allocation(
            self,
            model_instance,
            key,
            contract,
            plan.as_ref(),
            workspace_domain,
            slot_count,
        )?;
        match workspace_domain {
            InvocationWorkspaceDomain::Scratch { .. } => Err(invalid(
                "scratch invocation workspace is stage-owned and has no persistent allocator",
            )),
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::PagedAttention(_),
                ..
            } => {
                let pool = self.allocate_invocation_paged(
                    model_instance,
                    key,
                    plan.as_ref(),
                    workspace_domain,
                    slot_count,
                )?;
                Ok(invocation_paged_workspace_backing_v2(pool))
            }
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::StaticAttention(_),
                ..
            } => Err(invalid(
                "static attention requires a direct install-and-attend physical allocator",
            )),
            InvocationWorkspaceDomain::State {
                state:
                    StateDomainSpec::Tensor(_)
                    | StateDomainSpec::Append(_)
                    | StateDomainSpec::Ring(_)
                    | StateDomainSpec::StaticTensor(_),
                ..
            } => self.allocate_invocation_tensor(
                model_instance,
                key,
                contract,
                plan,
                workspace_domain,
                slot_count,
            ),
        }
    }

    pub(crate) fn resolve_and_allocate_invocation_workspace(
        &mut self,
        model_instance: ModelInstanceId,
        key: InvocationPhysicalKey,
        contract: &InferenceStateContract,
        workspace_domain: &InvocationWorkspaceDomain,
        slot_count: u32,
    ) -> Result<Arc<dyn InvocationWorkspaceBackingV2>> {
        let plan = Arc::new(negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend: self.worker_backend,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: match workspace_domain {
                    InvocationWorkspaceDomain::State {
                        state: StateDomainSpec::PagedAttention(domain),
                        ..
                    } => Some(domain.page_size.preferred_tokens),
                    _ => None,
                },
                storage_dtype_hint: None,
            },
        )?);
        self.allocate_invocation_workspace(
            model_instance,
            key,
            contract,
            plan,
            workspace_domain,
            slot_count,
        )
    }

    fn allocate_invocation_tensor(
        &mut self,
        model_instance: ModelInstanceId,
        key: InvocationPhysicalKey,
        contract: &InferenceStateContract,
        plan: Arc<ResolvedStatePlan>,
        workspace_domain: &InvocationWorkspaceDomain,
        slot_count: u32,
    ) -> Result<Arc<dyn InvocationWorkspaceBackingV2>> {
        let generation = self.next_allocation_generation;
        if generation == 0 {
            return Err(invalid(
                "physical invocation allocation generation exhausted",
            ));
        }
        let next_generation = generation
            .checked_add(1)
            .ok_or_else(|| invalid("physical invocation allocation generation overflow"))?;
        let matches = plan
            .non_paged
            .iter()
            .filter(|resolved| resolved.domain() == key.domain)
            .collect::<Vec<_>>();
        let [resolved] = matches.as_slice() else {
            return Err(invalid(
                "invocation tensor allocator requires one exact resolved non-paged domain",
            ));
        };
        let maximum_bytes = resolved
            .maximum_bytes()
            .checked_mul(u64::from(slot_count))
            .ok_or_else(|| invalid("invocation tensor pool byte accounting overflow"))?;
        let resources = arena_resources(self.worker_backend, maximum_bytes);
        let resource_lease = self
            .resource_authority
            .as_ref()
            .map(|authority| {
                reserve_arena(
                    authority,
                    model_instance,
                    key,
                    self.worker_backend,
                    resources,
                )
            })
            .transpose()?;
        let owner = InvocationTensorPoolOwner::new(
            contract,
            plan,
            workspace_domain.clone(),
            self.worker_device.clone(),
            model_instance,
            slot_count,
            generation,
        )?;
        if owner.maximum_bytes() != maximum_bytes {
            return Err(Error::InferenceError(
                "invocation tensor allocation does not match its reserved byte bound".to_string(),
            ));
        }
        if let Some(lease) = resource_lease.as_ref() {
            lease.record_materialized_usage(resources)?;
        }
        let backing = owner.backing();
        let replaced = self
            .models
            .entry(model_instance)
            .or_default()
            .invocation_tensor
            .insert(
                key,
                OwnedInvocationTensorPool {
                    owner,
                    resource_lease,
                    resources,
                },
            );
        if replaced.is_some() {
            return Err(Error::InferenceError(
                "physical invocation identity changed during exclusive allocation".to_string(),
            ));
        }
        self.next_allocation_generation = next_generation;
        Ok(backing)
    }

    fn allocate_invocation_paged(
        &mut self,
        model_instance: ModelInstanceId,
        key: InvocationPhysicalKey,
        plan: &ResolvedStatePlan,
        workspace_domain: &InvocationWorkspaceDomain,
        slot_count: u32,
    ) -> Result<InvocationPagedKvPoolHandle> {
        validate_key(key)?;
        if plan.backend != self.worker_backend
            || plan.device_ordinal != self.worker_device_ordinal
            || key.domain != workspace_domain.id()
            || slot_count == 0
        {
            return Err(invalid(
                "invocation allocation does not match its worker, domain, or lease multiplicity",
            ));
        }
        if self
            .models
            .get(&model_instance)
            .is_some_and(|model| model.invocation_paged.contains_key(&key))
        {
            return Err(invalid(
                "one physical invocation identity was allocated more than once",
            ));
        }
        let runtime = self.backend_runtime.as_ref().ok_or_else(|| {
            Error::ModelLoadError(self.backend_unavailable.clone().unwrap_or_else(|| {
                format!(
                    "physical paged state is unavailable for {:?}",
                    self.worker_backend
                )
            }))
        })?;
        let resolved = plan
            .paged_attention
            .iter()
            .find(|candidate| candidate.domain == key.domain)
            .ok_or_else(|| invalid("invocation paged domain is absent from its resolved plan"))?;
        let InvocationWorkspaceDomain::State {
            state: StateDomainSpec::PagedAttention(semantic),
            capacity: InvocationStateCapacity::PagedTokens { max_tokens },
            ..
        } = workspace_domain
        else {
            return Err(invalid(
                "physical paged allocator requires a paged token-capacity domain",
            ));
        };
        let pages_per_slot = pages_for_tokens(*max_tokens, resolved.page_tokens)?;
        let capacity_pages = pages_per_slot
            .checked_mul(slot_count)
            .ok_or_else(|| invalid("invocation paged arena capacity overflow"))?;
        let physical_bytes = resolved
            .bytes_per_page
            .checked_mul(u64::from(capacity_pages))
            .ok_or_else(|| invalid("invocation paged arena byte size overflow"))?;
        let generation = self.next_allocation_generation;
        if generation == 0 {
            return Err(invalid(
                "physical invocation allocation generation exhausted",
            ));
        }
        let next_generation = generation
            .checked_add(1)
            .ok_or_else(|| invalid("physical invocation allocation generation overflow"))?;
        let resources = arena_resources(self.worker_backend, physical_bytes);
        let resource_lease = self
            .resource_authority
            .as_ref()
            .map(|authority| {
                reserve_arena(
                    authority,
                    model_instance,
                    key,
                    self.worker_backend,
                    resources,
                )
            })
            .transpose()?;
        let arena_id = KvArenaId {
            model_instance,
            backend: self.worker_backend,
            device_ordinal: self.worker_device_ordinal,
            generation,
        };
        let config = invocation_arena_config(
            arena_id,
            resolved,
            semantic,
            capacity_pages,
            resolved.storage.dtype(),
        )?;
        let arena = runtime.allocate_arena(config)?;
        if arena.backend_kind() != self.worker_backend
            || arena.device_location() != self.worker_device_location
        {
            let _ = arena.drain();
            return Err(Error::ModelLoadError(
                "invocation arena was allocated on a different worker device".to_string(),
            ));
        }
        let owner = match InvocationPagedKvPoolOwner::new(
            plan,
            workspace_domain,
            arena.clone(),
            0,
            pages_per_slot,
            slot_count,
            generation,
        ) {
            Ok(owner) => owner,
            Err(error) => {
                let _ = arena.drain();
                return Err(error);
            }
        };
        if let Some(lease) = resource_lease.as_ref() {
            lease.record_materialized_usage(resources)?;
        }
        let handle = owner.handle();
        self.models
            .entry(model_instance)
            .or_default()
            .invocation_paged
            .insert(
                key,
                OwnedInvocationPool {
                    owner,
                    resource_lease,
                    resources,
                },
            );
        self.next_allocation_generation = next_generation;
        Ok(handle)
    }

    /// Close every pool first so a failed active-lease drain cannot admit new
    /// work. Removal and resource release occur only after all pools fence.
    pub(crate) fn unload_model(&mut self, model_instance: ModelInstanceId) -> Result<bool> {
        let Some(model) = self.models.get(&model_instance) else {
            return Ok(false);
        };
        let mut drain_error = None;
        for pool in model.invocation_paged.values() {
            if let Err(error) = pool.owner.close_and_drain() {
                drain_error.get_or_insert(error);
            }
        }
        for pool in model.invocation_tensor.values() {
            if let Err(error) = pool.owner.close_and_drain() {
                drain_error.get_or_insert(error);
            }
        }
        if let Some(retained) = model.retained_tensor.as_ref() {
            if let Err(error) = retained.runtime.close_and_validate_drained() {
                drain_error.get_or_insert(error);
            }
        }
        if let Some(error) = drain_error {
            return Err(error);
        }
        for pool in model.invocation_paged.values() {
            if let Some(lease) = pool.resource_lease.as_ref() {
                if lease.resources() != pool.resources {
                    return Err(Error::InferenceError(
                        "invocation state resource lease changed after allocation".to_string(),
                    ));
                }
                lease.prepare_materialized_release(ResourceVector::zero())?;
            }
        }
        for pool in model.invocation_tensor.values() {
            if let Some(lease) = pool.resource_lease.as_ref() {
                if lease.resources() != pool.resources {
                    return Err(Error::InferenceError(
                        "invocation tensor resource lease changed after allocation".to_string(),
                    ));
                }
                lease.prepare_materialized_release(ResourceVector::zero())?;
            }
        }
        if let Some(retained) = model.retained_tensor.as_ref() {
            if let Some(lease) = retained.resource_lease.as_ref() {
                if lease.resources() != retained.resources {
                    return Err(Error::InferenceError(
                        "retained tensor resource lease changed after allocation".to_string(),
                    ));
                }
                lease.prepare_materialized_release(ResourceVector::zero())?;
            }
        }
        let removed = self
            .models
            .remove(&model_instance)
            .expect("physical state record was validated under exclusive access");
        drop(removed);
        Ok(true)
    }

    #[cfg(test)]
    fn model_count(&self) -> usize {
        self.models.len()
    }
}

fn next_identity(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
            value.checked_add(1)
        })
        .map_err(|_| invalid("physical state identity space exhausted"))
}

fn retained_capacity_bytes(per_sequence_bytes: u64, sequence_capacity: u32) -> Result<u64> {
    per_sequence_bytes
        .checked_mul(u64::from(sequence_capacity))
        .ok_or_else(|| invalid("retained tensor sequence capacity byte bound overflow"))
}

fn validate_key(key: InvocationPhysicalKey) -> Result<()> {
    if key.adapter_instance.get() == 0
        || key.stage_graph.iter().all(|byte| *byte == 0)
        || key.stage.get() == 0
        || key.domain.get() == 0
    {
        return Err(invalid("physical invocation key is incomplete"));
    }
    Ok(())
}

fn validate_invocation_allocation(
    manager: &PhysicalStateManager,
    model_instance: ModelInstanceId,
    key: InvocationPhysicalKey,
    contract: &InferenceStateContract,
    plan: &ResolvedStatePlan,
    workspace_domain: &InvocationWorkspaceDomain,
    slot_count: u32,
) -> Result<()> {
    validate_key(key)?;
    contract.validate()?;
    if model_instance.get() == 0
        || slot_count == 0
        || plan.backend != manager.worker_backend
        || plan.device_ordinal != manager.worker_device_ordinal
        || plan.contract_fingerprint != contract.fingerprint()?
        || key.domain != workspace_domain.id()
    {
        return Err(invalid(
            "invocation allocation does not match its model, worker, contract, domain, or lease multiplicity",
        ));
    }
    plan.validate_against(
        contract,
        &StateBackendRegistry::new(manager.worker_backend, manager.worker_device_ordinal)?,
    )?;
    if manager.models.get(&model_instance).is_some_and(|model| {
        model.invocation_paged.contains_key(&key) || model.invocation_tensor.contains_key(&key)
    }) {
        return Err(invalid(
            "one physical invocation identity was allocated more than once",
        ));
    }
    let InvocationWorkspaceDomain::State {
        state,
        capacity,
        placement,
        ..
    } = workspace_domain
    else {
        return Ok(());
    };
    let canonical = contract
        .domains
        .iter()
        .find(|candidate| candidate.id() == state.id())
        .ok_or_else(|| invalid("invocation workspace domain is absent from its contract"))?;
    if canonical != state
        || state.scope() != StateScope::Invocation
        || state.header().placement != *placement
    {
        return Err(invalid(
            "invocation workspace is not the canonical invocation-scoped contract domain",
        ));
    }
    let capacity_matches = matches!(
        (state, capacity),
        (
            StateDomainSpec::PagedAttention(_),
            InvocationStateCapacity::PagedTokens { .. }
        ) | (
            StateDomainSpec::StaticAttention(_)
                | StateDomainSpec::Tensor(_)
                | StateDomainSpec::Append(_)
                | StateDomainSpec::Ring(_)
                | StateDomainSpec::StaticTensor(_),
            InvocationStateCapacity::SemanticBounded
        )
    );
    if !capacity_matches {
        return Err(invalid(
            "invocation workspace capacity does not match its semantic domain",
        ));
    }
    Ok(())
}

fn pages_for_tokens(max_tokens: u64, page_tokens: u32) -> Result<u32> {
    if max_tokens == 0 || page_tokens == 0 {
        return Err(invalid("paged invocation capacity must be non-zero"));
    }
    let pages = max_tokens
        .checked_add(u64::from(page_tokens) - 1)
        .and_then(|tokens| tokens.checked_div(u64::from(page_tokens)))
        .ok_or_else(|| invalid("paged invocation capacity overflow"))?;
    u32::try_from(pages).map_err(|_| invalid("paged invocation capacity exceeds u32"))
}

fn invocation_arena_config(
    id: KvArenaId,
    resolved: &crate::kv::v2::ResolvedPagedAttentionGroup,
    semantic: &crate::kv::v2::PagedAttentionDomainSpec,
    capacity_pages: u32,
    dtype: StateDType,
) -> Result<KvArenaConfig> {
    let mut layers = Vec::with_capacity(resolved.layers.len());
    for binding in &resolved.layers {
        let layer = semantic
            .layers
            .iter()
            .find(|layer| layer.model_layer == binding.model_layer)
            .ok_or_else(|| invalid("resolved invocation layer lost its semantic geometry"))?;
        layers.push(KvLayerConfig {
            binding: KvLayerBinding {
                model_layer: binding.model_layer,
                physical_layer: binding.physical_layer,
            },
            num_kv_heads: layer.kv_heads,
            key_head_dim: layer.key_head_dim,
            value_head_dim: layer.value_head_dim,
        });
    }
    Ok(KvArenaConfig {
        id,
        group: KvGroupId::new(resolved.group.get()),
        page_tokens: resolved.page_tokens,
        capacity_pages,
        dtype: candle_dtype(dtype)?,
        layers,
    })
}

fn candle_dtype(dtype: StateDType) -> Result<DType> {
    match dtype {
        StateDType::F32 => Ok(DType::F32),
        StateDType::F16 => Ok(DType::F16),
        StateDType::Bf16 => Ok(DType::BF16),
        StateDType::I8 | StateDType::Q4 => Err(invalid(
            "dense invocation arenas cannot allocate quantized state",
        )),
    }
}

fn arena_resources(backend: BackendKind, bytes: u64) -> ResourceVector {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(bytes),
        BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(bytes),
        BackendKind::Cuda => resources.device_bytes = ResourceAmount::Known(bytes),
    }
    resources
}

fn reserve_arena(
    authority: &Arc<ResourceAuthority>,
    model_instance: ModelInstanceId,
    key: InvocationPhysicalKey,
    backend: BackendKind,
    resources: ResourceVector,
) -> Result<ResourceLease> {
    let owner = ReservationOwner::new(
        ReservationClass::Model,
        format!(
            "invocation-state:{}:{}:{}:{}:{backend:?}",
            model_instance.get(),
            key.adapter_instance.get(),
            key.stage.get(),
            key.domain.get()
        ),
    );
    match backend {
        BackendKind::Cpu | BackendKind::Metal => authority.track_advisory(owner, resources),
        BackendKind::Cuda => authority.reserve(owner, resources),
    }
}

fn reserve_retained_tensor(
    authority: &Arc<ResourceAuthority>,
    model_instance: ModelInstanceId,
    backend: BackendKind,
    resources: ResourceVector,
) -> Result<ResourceLease> {
    let owner = ReservationOwner::new(
        ReservationClass::Model,
        format!("retained-tensor-state:{}:{backend:?}", model_instance.get()),
    );
    match backend {
        BackendKind::Cpu | BackendKind::Metal => authority.track_advisory(owner, resources),
        BackendKind::Cuda => authority.reserve(owner, resources),
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::engine::{CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot};
    use crate::kv::v2::{
        test_contract, BoundedShape, CheckpointPolicy, InvocationStateCapacity, PlacementPolicy,
        PrefixPolicy, ShapeAxis, ShapeDimension, ShapeExtent, StateClock, StateComponentId,
        StateDomainHeader, StateGroupId, StateGroupSpec, TensorComponentSpec, TensorRole,
        TensorStateDomainSpec, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
    };
    use candle_core::Tensor;

    #[derive(Debug)]
    struct TestCapacityProvider {
        capacity: ResourceVector,
    }

    impl PhysicalCapacityProvider for TestCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: self.capacity,
                available: self.capacity,
                source: CapacitySource::Test,
            }
        }
    }

    fn paged_invocation_contract() -> InferenceStateContract {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        domain.header.scope = StateScope::Invocation;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::None;
        domain.accepted_dtypes = vec![StateDType::F32];
        domain.layers[0].query_heads = 2;
        domain.layers[0].kv_heads = 2;
        domain.layers[0].key_head_dim = 4;
        domain.layers[0].value_head_dim = 4;
        domain.layers[0].key_encoding = crate::kv::v2::KeyEncoding::Rotary { rotary_dim: 4 };
        contract.groups[0].prefix_shareable = false;
        contract
    }

    fn invocation_plan() -> (ResolvedStatePlan, InvocationWorkspaceDomain) {
        let contract = paged_invocation_contract();
        let workspace = InvocationWorkspaceDomain::State {
            state: contract.domains[0].clone(),
            capacity: InvocationStateCapacity::PagedTokens { max_tokens: 17 },
            placement: PlacementPolicy::BackendLocalWithHostOffload,
            formula: WorkspaceFormula {
                fixed_bytes: 1024 * 1024,
                dimensions: vec![],
                terms: vec![],
            },
        };
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(16),
                storage_dtype_hint: Some(StateDType::F32),
            },
        )
        .unwrap();
        (plan, workspace)
    }

    fn tensor_invocation_domain(id: u32, maximum_elements: u64) -> StateDomainSpec {
        StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: StateDomainId::new(id),
                scope: StateScope::Invocation,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::None,
            },
            components: vec![TensorComponentSpec {
                id: StateComponentId::new(1),
                role: TensorRole::Control,
                shape: BoundedShape {
                    dimensions: vec![ShapeDimension {
                        axis: ShapeAxis::Hidden,
                        extent: ShapeExtent::RuntimeBounded {
                            min: 1,
                            max: maximum_elements,
                        },
                    }],
                },
                accepted_dtypes: vec![StateDType::F32],
            }],
        })
    }

    fn mixed_invocation_contract() -> InferenceStateContract {
        let mut contract = paged_invocation_contract();
        contract.domains.push(tensor_invocation_domain(2, 8));
        contract.groups.push(StateGroupSpec {
            id: StateGroupId::new(2),
            domains: vec![StateDomainId::new(2)],
            prefix_shareable: false,
        });
        contract
    }

    fn tensor_invocation_plan_and_workspace() -> (
        InferenceStateContract,
        Arc<ResolvedStatePlan>,
        InvocationWorkspaceDomain,
    ) {
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![tensor_invocation_domain(1, 8)],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        };
        let plan = Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: Some(StateDType::F32),
                },
            )
            .unwrap(),
        );
        let workspace = InvocationWorkspaceDomain::State {
            state: contract.domains[0].clone(),
            capacity: InvocationStateCapacity::SemanticBounded,
            placement: PlacementPolicy::BackendLocal,
            formula: WorkspaceFormula {
                fixed_bytes: plan.non_paged[0].maximum_bytes(),
                dimensions: vec![],
                terms: vec![],
            },
        };
        (contract, plan, workspace)
    }

    fn mixed_plan_and_workspaces() -> (
        InferenceStateContract,
        Arc<ResolvedStatePlan>,
        InvocationWorkspaceDomain,
        InvocationWorkspaceDomain,
    ) {
        let contract = mixed_invocation_contract();
        let plan = Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: Some(16),
                    storage_dtype_hint: Some(StateDType::F32),
                },
            )
            .unwrap(),
        );
        let paged = InvocationWorkspaceDomain::State {
            state: contract.domains[0].clone(),
            capacity: InvocationStateCapacity::PagedTokens { max_tokens: 17 },
            placement: PlacementPolicy::BackendLocalWithHostOffload,
            formula: WorkspaceFormula {
                fixed_bytes: 1024 * 1024,
                dimensions: vec![],
                terms: vec![],
            },
        };
        let tensor_bytes = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == StateDomainId::new(2))
            .unwrap()
            .maximum_bytes();
        let tensor = InvocationWorkspaceDomain::State {
            state: contract.domains[1].clone(),
            capacity: InvocationStateCapacity::SemanticBounded,
            placement: PlacementPolicy::BackendLocal,
            formula: WorkspaceFormula {
                fixed_bytes: tensor_bytes,
                dimensions: vec![],
                terms: vec![],
            },
        };
        (contract, plan, paged, tensor)
    }

    fn key() -> InvocationPhysicalKey {
        InvocationPhysicalKey {
            adapter_instance: AdapterInstanceId::new(3),
            stage_graph: [7; 32],
            stage: StageId::new(2),
            domain: StateDomainId::new(1),
        }
    }

    fn retained_tensor_contract(maximum_elements: u64) -> InferenceStateContract {
        InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: StateDomainHeader {
                    id: StateDomainId::new(1),
                    scope: StateScope::Retained,
                    clock: StateClock::DecoderTokens,
                    placement: PlacementPolicy::BackendLocal,
                    prefix: PrefixPolicy::Disabled,
                    checkpoint: CheckpointPolicy::Transactional,
                },
                components: vec![TensorComponentSpec {
                    id: StateComponentId::new(1),
                    role: TensorRole::RecurrentHidden,
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::RuntimeBounded {
                                min: 1,
                                max: maximum_elements,
                            },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                }],
            })],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        }
    }

    #[test]
    fn allocator_uses_exact_page_rounding_and_slot_multiplicity() {
        let (plan, domain) = invocation_plan();
        let mut manager = PhysicalStateManager::cpu(None);
        let handle = manager
            .allocate_invocation_paged(ModelInstanceId::new(41), key(), &plan, &domain, 2)
            .unwrap();
        let first = handle.lease().unwrap();
        let second = handle.lease().unwrap();
        assert_eq!(first.slot().page_count, 2);
        assert_eq!(second.slot().page_count, 2);
        assert_ne!(first.slot().first_page, second.slot().first_page);
        assert!(handle.lease().is_err());
        drop(first);
        drop(second);
        assert!(manager.unload_model(ModelInstanceId::new(41)).unwrap());
        assert!(handle.lease().is_err());
    }

    #[test]
    fn unload_closes_against_new_leases_and_retries_after_release() {
        let (plan, domain) = invocation_plan();
        let model = ModelInstanceId::new(42);
        let mut manager = PhysicalStateManager::cpu(None);
        let handle = manager
            .allocate_invocation_paged(model, key(), &plan, &domain, 1)
            .unwrap();
        assert!(manager
            .allocate_invocation_paged(model, key(), &plan, &domain, 1)
            .is_err());
        let lease = handle.lease().unwrap();
        assert!(manager.unload_model(model).is_err());
        assert!(handle.lease().is_err());
        drop(lease);
        assert!(manager.unload_model(model).unwrap());
        assert_eq!(manager.model_count(), 0);

        let replacement = manager
            .allocate_invocation_paged(model, key(), &plan, &domain, 1)
            .unwrap();
        assert_ne!(replacement.id(), handle.id());
        assert!(handle.lease().is_err());
        manager.unload_model(model).unwrap();
    }

    #[test]
    fn unload_closes_every_physical_owner_before_reporting_backpressure() {
        let (plan, domain) = invocation_plan();
        let model = ModelInstanceId::new(45);
        let mut manager = PhysicalStateManager::cpu(None);
        let first = manager
            .allocate_invocation_paged(model, key(), &plan, &domain, 1)
            .unwrap();
        let mut second_key = key();
        second_key.adapter_instance = AdapterInstanceId::new(4);
        let second = manager
            .allocate_invocation_paged(model, second_key, &plan, &domain, 1)
            .unwrap();
        let retained = manager
            .allocate_retained_tensor(model, &retained_tensor_contract(8), 1)
            .unwrap();
        let invocation_lease = first.lease().unwrap();
        let sequence = retained.register_sequence().unwrap();

        assert!(manager.unload_model(model).is_err());
        assert!(first.lease().is_err());
        assert!(second.lease().is_err());
        assert!(retained.register_sequence().is_err());

        drop(invocation_lease);
        retained.release_sequence(sequence).unwrap();
        assert!(manager.unload_model(model).unwrap());
    }

    #[test]
    fn generic_allocator_publishes_mixed_paged_and_tensor_backings_with_exact_resources() {
        let (contract, plan, paged, tensor) = mixed_plan_and_workspaces();
        let model = ModelInstanceId::new(46);
        let mut manager = PhysicalStateManager::cpu(None);
        let paged_key = key();
        let mut tensor_key = key();
        tensor_key.domain = StateDomainId::new(2);
        let expected_paged_bytes = plan.paged_attention[0].bytes_per_page * 2 * 2;
        let expected_tensor_bytes = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == StateDomainId::new(2))
            .unwrap()
            .maximum_bytes()
            * 3;

        let paged_backing = manager
            .allocate_invocation_workspace(model, paged_key, &contract, plan.clone(), &paged, 2)
            .unwrap();
        let tensor_backing = manager
            .allocate_invocation_workspace(model, tensor_key, &contract, plan, &tensor, 3)
            .unwrap();
        let physical = manager.models.get(&model).unwrap();
        let paged_owned = physical.invocation_paged.get(&paged_key).unwrap();
        let tensor_owned = physical.invocation_tensor.get(&tensor_key).unwrap();
        assert_eq!(
            paged_owned.resources,
            arena_resources(BackendKind::Cpu, expected_paged_bytes)
        );
        assert_eq!(
            tensor_owned.resources,
            arena_resources(BackendKind::Cpu, expected_tensor_bytes)
        );
        assert_eq!(
            tensor_owned.resources.host_bytes,
            ResourceAmount::Known(tensor_owned.owner.maximum_bytes())
        );
        assert_ne!(paged_backing.identity(), tensor_backing.identity());

        let runtime = crate::kv::v2::InvocationWorkspaceRuntimeV2::new(vec![
            crate::kv::v2::InvocationWorkspaceBindingV2 {
                key: crate::kv::v2::InvocationWorkspaceKeyV2 {
                    stage_graph: paged_key.stage_graph,
                    stage: paged_key.stage,
                    domain: paged_key.domain,
                },
                backing: paged_backing,
            },
            crate::kv::v2::InvocationWorkspaceBindingV2 {
                key: crate::kv::v2::InvocationWorkspaceKeyV2 {
                    stage_graph: tensor_key.stage_graph,
                    stage: tensor_key.stage,
                    domain: tensor_key.domain,
                },
                backing: tensor_backing,
            },
        ])
        .unwrap();
        drop(runtime);
        assert!(manager.unload_model(model).unwrap());
    }

    #[test]
    fn generic_allocator_rejects_duplicate_keys_across_physical_kinds() {
        let (tensor_contract, tensor_plan, tensor) = tensor_invocation_plan_and_workspace();
        let paged_contract = paged_invocation_contract();
        let (paged_plan, paged) = invocation_plan();
        let model = ModelInstanceId::new(47);
        let physical_key = key();
        let mut manager = PhysicalStateManager::cpu(None);
        manager
            .allocate_invocation_workspace(
                model,
                physical_key,
                &tensor_contract,
                tensor_plan,
                &tensor,
                1,
            )
            .unwrap();
        let generation = manager.next_allocation_generation;
        assert!(manager
            .allocate_invocation_workspace(
                model,
                physical_key,
                &paged_contract,
                Arc::new(paged_plan),
                &paged,
                1,
            )
            .is_err());
        assert_eq!(manager.next_allocation_generation, generation);
        assert_eq!(
            manager.models.get(&model).unwrap().invocation_tensor.len(),
            1
        );
        assert!(manager
            .models
            .get(&model)
            .unwrap()
            .invocation_paged
            .is_empty());
        manager.unload_model(model).unwrap();
    }

    #[test]
    fn active_typed_lease_blocks_unload_then_allows_retry() {
        let (contract, plan, _paged, tensor) = mixed_plan_and_workspaces();
        let model = ModelInstanceId::new(48);
        let mut tensor_key = key();
        tensor_key.domain = StateDomainId::new(2);
        let mut manager = PhysicalStateManager::cpu(None);
        let backing = manager
            .allocate_invocation_workspace(model, tensor_key, &contract, plan, &tensor, 1)
            .unwrap();
        let lease = backing.lease().unwrap();
        assert!(manager.unload_model(model).is_err());
        assert!(backing.lease().is_err());
        drop(lease);
        assert!(manager.unload_model(model).unwrap());
        assert!(backing.validate_live().is_err());
    }

    #[test]
    fn mixed_unload_closes_paged_and_typed_owners_before_retry() {
        let (contract, plan, paged, tensor) = mixed_plan_and_workspaces();
        let model = ModelInstanceId::new(50);
        let paged_key = key();
        let mut tensor_key = key();
        tensor_key.domain = StateDomainId::new(2);
        let mut manager = PhysicalStateManager::cpu(None);
        let paged_backing = manager
            .allocate_invocation_workspace(model, paged_key, &contract, plan.clone(), &paged, 1)
            .unwrap();
        let tensor_backing = manager
            .allocate_invocation_workspace(model, tensor_key, &contract, plan, &tensor, 1)
            .unwrap();
        let paged_lease = paged_backing.lease().unwrap();
        let tensor_lease = tensor_backing.lease().unwrap();

        assert!(manager.unload_model(model).is_err());
        assert!(paged_backing.lease().is_err());
        assert!(tensor_backing.lease().is_err());

        drop((paged_lease, tensor_lease));
        assert!(manager.unload_model(model).unwrap());
        assert_eq!(manager.model_count(), 0);
    }

    #[test]
    fn typed_allocation_failure_does_not_publish_or_advance_generation() {
        let (contract, plan, _paged, tensor) = mixed_plan_and_workspaces();
        let mut tensor_key = key();
        tensor_key.domain = StateDomainId::new(2);
        let mut manager = PhysicalStateManager::cpu(None);
        manager.next_allocation_generation = u32::MAX;
        assert!(manager
            .allocate_invocation_workspace(
                ModelInstanceId::new(49),
                tensor_key,
                &contract,
                plan,
                &tensor,
                1,
            )
            .is_err());
        assert_eq!(manager.next_allocation_generation, u32::MAX);
        assert_eq!(manager.model_count(), 0);
    }

    #[test]
    fn typed_reservation_denial_precedes_materialization_and_publication() {
        let (contract, plan, tensor) = tensor_invocation_plan_and_workspace();
        let capacity = ResourceVector {
            host_bytes: ResourceAmount::Known(u64::MAX),
            ..ResourceVector::zero()
        };
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
            capacity,
        })));
        let saturated = authority
            .track_model("saturated-test-ledger", capacity)
            .unwrap();
        let mut manager = PhysicalStateManager::cpu(Some(authority.clone()));
        let generation = manager.next_allocation_generation;

        assert!(manager
            .allocate_invocation_workspace(
                ModelInstanceId::new(51),
                key(),
                &contract,
                plan,
                &tensor,
                1,
            )
            .is_err());
        assert_eq!(manager.next_allocation_generation, generation);
        assert_eq!(manager.model_count(), 0);
        assert_eq!(authority.snapshot().reservations, 1);
        drop(saturated);
        assert_eq!(authority.snapshot().reservations, 0);
    }

    #[test]
    fn retained_tensor_runtime_reuses_exact_contract_and_exposes_transactions() {
        let model = ModelInstanceId::new(43);
        let contract = retained_tensor_contract(8);
        let mut manager = PhysicalStateManager::cpu(None);
        let runtime = manager
            .allocate_retained_tensor(model, &contract, 2)
            .unwrap();
        let reused = manager
            .allocate_retained_tensor(model, &contract, 2)
            .unwrap();
        assert!(Arc::ptr_eq(&runtime, &reused));
        assert!(runtime.state_plan_v2().paged_attention.is_empty());
        assert_eq!(runtime.state_plan_v2().non_paged.len(), 1);
        assert_eq!(runtime.sequence_capacity(), 2);
        assert!(runtime.maximum_bytes() >= 2 * 8 * 4);
        assert!(manager
            .allocate_retained_tensor(model, &retained_tensor_contract(4), 2)
            .is_err());
        assert!(manager
            .allocate_retained_tensor(model, &contract, 1)
            .is_err());

        let sequence = runtime.register_sequence().unwrap();
        let second_sequence = runtime.register_sequence().unwrap();
        assert!(runtime.register_sequence().is_err());
        let transaction = runtime.begin_transaction(sequence).unwrap();
        runtime
            .stage_replace(
                transaction,
                StateDomainId::new(1),
                0,
                1,
                vec![crate::backends::state::StateComponentValue {
                    component: StateComponentId::new(1),
                    tensor: Some(Tensor::from_slice(&[1.0_f32, 2.0], 2, &Device::Cpu).unwrap()),
                }],
            )
            .unwrap();
        runtime.commit_transaction(transaction, 1).unwrap();
        assert_eq!(
            runtime
                .read(sequence, StateDomainId::new(1))
                .unwrap()
                .unwrap()
                .cursor,
            1
        );

        assert!(manager.unload_model(model).is_err());
        assert!(runtime.register_sequence().is_err());
        runtime.release_sequence(sequence).unwrap();
        runtime.release_sequence(second_sequence).unwrap();
        assert!(manager.unload_model(model).unwrap());
        assert_eq!(manager.model_count(), 0);
    }

    #[test]
    fn retained_tensor_capacity_bytes_reject_overflow() {
        assert!(retained_capacity_bytes(u64::MAX, 2).is_err());
        assert_eq!(retained_capacity_bytes(32, 3).unwrap(), 96);
    }

    #[test]
    fn retained_tensor_generation_overflow_does_not_publish_an_owner() {
        let mut manager = PhysicalStateManager::cpu(None);
        manager.next_allocation_generation = u32::MAX;
        assert!(manager
            .allocate_retained_tensor(ModelInstanceId::new(44), &retained_tensor_contract(8), 1,)
            .is_err());
        assert_eq!(manager.model_count(), 0);
    }
}
