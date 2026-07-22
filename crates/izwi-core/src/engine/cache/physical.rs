//! Lifecycle-owned physical inference-state allocations beyond retained KV.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, DeviceLocation};

use crate::backends::kv::{KvArenaConfig, KvBackendRuntime, KvLayerConfig};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::v2::{
    InvocationStateCapacity, InvocationWorkspaceDomain, ResolvedStatePlan, StateDType,
    StateDomainId, StateDomainSpec,
};
use crate::kv::{KvArenaId, KvGroupId, KvLayerBinding};

use super::invocation::{InvocationPagedKvPoolHandle, InvocationPagedKvPoolOwner};
use super::managed::{managed_backend_runtime, managed_device_ordinal};
use crate::engine::{
    ModelInstanceId, ReservationClass, ReservationOwner, ResourceAmount, ResourceAuthority,
    ResourceLease, ResourceVector, StageId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InvocationPhysicalKey {
    pub(crate) stage_graph: [u8; 32],
    pub(crate) stage: StageId,
    pub(crate) domain: StateDomainId,
}

struct OwnedInvocationPool {
    owner: InvocationPagedKvPoolOwner,
    resource_lease: Option<ResourceLease>,
    resources: ResourceVector,
}

#[derive(Default)]
struct ModelPhysicalState {
    invocation_paged: HashMap<InvocationPhysicalKey, OwnedInvocationPool>,
}

/// Worker-local owner for capability-authored invocation state. Planning and
/// allocation happen while the model is Loading; request admission receives
/// weak generation handles only.
pub(crate) struct PhysicalStateManager {
    models: HashMap<ModelInstanceId, ModelPhysicalState>,
    resource_authority: Option<Arc<ResourceAuthority>>,
    next_allocation_generation: u32,
    worker_backend: BackendKind,
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
            worker_device_location: device.location(),
            worker_device_ordinal: managed_device_ordinal(&device),
            backend_runtime,
            backend_unavailable,
        }
    }

    pub(crate) fn cpu(resource_authority: Option<Arc<ResourceAuthority>>) -> Self {
        Self::for_worker(resource_authority, BackendKind::Cpu, Device::Cpu)
    }

    pub(crate) fn allocate_invocation_paged(
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
        let generation = self.next_allocation_generation;
        if generation == 0 {
            return Err(invalid(
                "physical invocation allocation generation exhausted",
            ));
        }
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
        self.next_allocation_generation = generation
            .checked_add(1)
            .ok_or_else(|| invalid("physical invocation allocation generation overflow"))?;
        Ok(handle)
    }

    /// Close every pool first so a failed active-lease drain cannot admit new
    /// work. Removal and resource release occur only after all pools fence.
    pub(crate) fn unload_model(&mut self, model_instance: ModelInstanceId) -> Result<bool> {
        let Some(model) = self.models.get(&model_instance) else {
            return Ok(false);
        };
        for pool in model.invocation_paged.values() {
            pool.owner.close_and_drain()?;
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

fn validate_key(key: InvocationPhysicalKey) -> Result<()> {
    if key.stage_graph.iter().all(|byte| *byte == 0)
        || key.stage.get() == 0
        || key.domain.get() == 0
    {
        return Err(invalid("physical invocation key is incomplete"));
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
            "invocation-state:{}:{}:{}:{backend:?}",
            model_instance.get(),
            key.stage.get(),
            key.domain.get()
        ),
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
    use crate::kv::v2::{
        test_contract, CheckpointPolicy, InvocationStateCapacity, PlacementPolicy, PrefixPolicy,
        StateScope, WorkspaceFormula,
    };

    fn invocation_plan() -> (ResolvedStatePlan, InvocationWorkspaceDomain) {
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

    fn key() -> InvocationPhysicalKey {
        InvocationPhysicalKey {
            stage_graph: [7; 32],
            stage: StageId::new(2),
            domain: StateDomainId::new(1),
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
}
