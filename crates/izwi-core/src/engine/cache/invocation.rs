//! Invocation-scoped physical paged-attention workspace.
//!
//! Unlike retained KV, these pages have no scheduler table, prefix ownership,
//! checkpoint, or session lifetime. A loaded model runtime owns one dedicated
//! arena allocation and leases fixed disjoint page ranges for one invocation
//! at a time. Every range is zeroed and fenced before a model can observe it.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use candle_core::DType;

use crate::backends::kv::{KvArena, KvWriteBatchCompletion};
use crate::error::{Error, Result};
use crate::kv::v2::{
    InvocationWorkspaceDomain, ResolvedStatePlan, StateDType, StateDomainId, StateDomainSpec,
    StateGroupId, StatePhysicalLayout, StatePlanId, StateScope,
};
use crate::kv::{CacheBlockRef, KvArenaId, KvLayerBinding};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InvocationPagedKvPoolId {
    pub(crate) plan: StatePlanId,
    pub(crate) domain: StateDomainId,
    pub(crate) group: StateGroupId,
    pub(crate) arena: KvArenaId,
    /// Generation of this invocation-pool allocation, independent from each
    /// slot reuse generation and non-zero for every loaded runtime instance.
    pub(crate) allocation_generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InvocationPagedKvSlotRef {
    pub(crate) pool: InvocationPagedKvPoolId,
    pub(crate) slot: u32,
    pub(crate) slot_generation: u32,
    pub(crate) first_page: u32,
    pub(crate) page_count: u32,
}

/// Fixed page-pool metadata over an already allocated, invocation-exclusive
/// arena. Constructing this type does not allocate device memory; the v2
/// lifecycle allocator must supply a dedicated arena that is never registered
/// with the retained-state coordinator.
#[derive(Debug, Clone)]
pub(crate) struct InvocationPagedKvPool {
    inner: Arc<InvocationPagedKvPoolInner>,
}

struct InvocationPagedKvPoolInner {
    id: InvocationPagedKvPoolId,
    arena: Arc<dyn KvArena>,
    layer_bindings: Vec<KvLayerBinding>,
    first_page: u32,
    pages_per_slot: u32,
    slots: Mutex<Vec<InvocationPagedKvSlotState>>,
}

impl std::fmt::Debug for InvocationPagedKvPoolInner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationPagedKvPoolInner")
            .field("id", &self.id)
            .field("first_page", &self.first_page)
            .field("pages_per_slot", &self.pages_per_slot)
            .field("slot_count", &self.slots.lock().map(|slots| slots.len()))
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationPagedKvSlotState {
    Vacant { generation: u32 },
    Preparing { generation: u32 },
    Leased { generation: u32 },
}

impl InvocationPagedKvSlotState {
    const fn generation(self) -> u32 {
        match self {
            Self::Vacant { generation }
            | Self::Preparing { generation }
            | Self::Leased { generation } => generation,
        }
    }
}

impl InvocationPagedKvPool {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plan: &ResolvedStatePlan,
        workspace_domain: &InvocationWorkspaceDomain,
        arena: Arc<dyn KvArena>,
        first_page: u32,
        pages_per_slot: u32,
        slot_count: u32,
        allocation_generation: u32,
    ) -> Result<Self> {
        let InvocationWorkspaceDomain::State {
            state: StateDomainSpec::PagedAttention(semantic),
            placement,
            formula,
        } = workspace_domain
        else {
            return Err(invalid(
                "invocation paged workspace requires a typed invocation paged-attention domain",
            ));
        };
        let domain = semantic.header.id;
        if domain.get() == 0
            || arena.id().generation == 0
            || pages_per_slot == 0
            || slot_count == 0
            || allocation_generation == 0
        {
            return Err(invalid(
                "invocation paged workspace requires non-zero identities and capacity",
            ));
        }
        if plan.backend != arena.backend_kind()
            || plan.device_ordinal != arena.id().device_ordinal
            || arena.id() != arena.config().id
        {
            return Err(invalid(
                "invocation paged workspace arena does not match its resolved backend plan",
            ));
        }
        let resolved = plan
            .paged_attention
            .iter()
            .find(|candidate| candidate.domain == domain)
            .ok_or_else(|| invalid("invocation paged workspace domain is not resolved"))?;
        if plan
            .paged_attention
            .iter()
            .filter(|candidate| candidate.domain == domain)
            .count()
            != 1
        {
            return Err(invalid(
                "invocation paged workspace domain is not uniquely resolved",
            ));
        }
        if semantic.header.scope != StateScope::Invocation
            || semantic.header.placement != *placement
            || !semantic.accepted_dtypes.contains(&resolved.storage.dtype())
            || !semantic.page_size.accepts(resolved.page_tokens)
            || semantic.layers.len() != resolved.layers.len()
            || semantic
                .layers
                .iter()
                .zip(&resolved.layers)
                .any(|(expected, actual)| expected.model_layer != actual.model_layer)
        {
            return Err(invalid(
                "invocation paged workspace semantic domain does not match its resolved plan",
            ));
        }
        let requested_backing = resolved
            .bytes_per_page
            .checked_mul(u64::from(pages_per_slot))
            .ok_or_else(|| invalid("invocation paged workspace backing size overflow"))?;
        if requested_backing > formula.maximum_bytes()? {
            return Err(invalid(
                "invocation paged workspace pages exceed the domain formula maximum",
            ));
        }
        validate_arena_geometry(resolved, semantic, arena.as_ref())?;

        let total_pages = pages_per_slot
            .checked_mul(slot_count)
            .and_then(|pages| first_page.checked_add(pages))
            .ok_or_else(|| invalid("invocation paged workspace page range overflow"))?;
        if total_pages > arena.config().capacity_pages {
            return Err(invalid(
                "invocation paged workspace exceeds its dedicated arena capacity",
            ));
        }
        let slot_count = usize::try_from(slot_count)
            .map_err(|_| invalid("invocation paged workspace slot count exceeds usize"))?;
        let id = InvocationPagedKvPoolId {
            plan: plan.id,
            domain,
            group: resolved.group,
            arena: arena.id(),
            allocation_generation,
        };
        Ok(Self {
            inner: Arc::new(InvocationPagedKvPoolInner {
                id,
                arena,
                layer_bindings: resolved
                    .layers
                    .iter()
                    .map(|binding| KvLayerBinding {
                        model_layer: binding.model_layer,
                        physical_layer: binding.physical_layer,
                    })
                    .collect(),
                first_page,
                pages_per_slot,
                slots: Mutex::new(vec![
                    InvocationPagedKvSlotState::Vacant { generation: 0 };
                    slot_count
                ]),
            }),
        })
    }

    pub(crate) fn id(&self) -> InvocationPagedKvPoolId {
        self.inner.id
    }

    pub(crate) fn maximum_tokens_per_lease(&self) -> Result<u64> {
        u64::from(self.inner.pages_per_slot)
            .checked_mul(u64::from(self.inner.arena.config().page_tokens))
            .ok_or_else(|| invalid("invocation paged workspace token capacity overflow"))
    }

    /// Acquire, zero, and fence one fixed page range before exposing it as a
    /// model-facing physical cache. A failed zero/fence returns the exact slot
    /// generation to the pool without exposing partially scrubbed storage.
    pub(crate) fn lease(&self) -> Result<InvocationPagedKvLease> {
        let slot = self.begin_lease()?;
        let blocks = blocks_for_slot(self.inner.as_ref(), slot)?;
        let prepared = self
            .inner
            .arena
            .zero_pages(&blocks)
            .and_then(|fence| fence.wait())
            .and_then(|()| {
                PhysicalPagedKvCache::new(
                    self.inner.arena.clone(),
                    self.inner.layer_bindings.clone(),
                    blocks.clone(),
                    0,
                )
            });
        let cache = match prepared {
            Ok(cache) => cache,
            Err(error) => {
                release_slot(
                    self.inner.as_ref(),
                    slot,
                    InvocationPagedKvSlotKind::Preparing,
                );
                return Err(error);
            }
        };
        if !transition_to_leased(self.inner.as_ref(), slot) {
            release_slot(
                self.inner.as_ref(),
                slot,
                InvocationPagedKvSlotKind::Preparing,
            );
            return Err(Error::InferenceError(
                "invocation paged workspace lost its preparing generation".to_string(),
            ));
        }
        Ok(InvocationPagedKvLease {
            inner: self.inner.clone(),
            slot,
            cache: Some(cache),
            released: false,
        })
    }

    pub(crate) fn contains_active_lease(&self, slot: InvocationPagedKvSlotRef) -> bool {
        if slot.pool != self.inner.id
            || slot.slot_generation == 0
            || slot.first_page
                != self
                    .inner
                    .first_page
                    .saturating_add(slot.slot.saturating_mul(self.inner.pages_per_slot))
            || slot.page_count != self.inner.pages_per_slot
        {
            return false;
        }
        self.inner
            .slots
            .lock()
            .ok()
            .and_then(|slots| slots.get(slot.slot as usize).copied())
            == Some(InvocationPagedKvSlotState::Leased {
                generation: slot.slot_generation,
            })
    }

    fn begin_lease(&self) -> Result<InvocationPagedKvSlotRef> {
        let mut slots = self
            .inner
            .slots
            .lock()
            .map_err(|_| invalid("invocation paged workspace slot state is poisoned"))?;
        let (slot_index, state) = slots
            .iter_mut()
            .enumerate()
            .find(|(_, state)| matches!(state, InvocationPagedKvSlotState::Vacant { .. }))
            .ok_or_else(|| {
                Error::Backpressure("invocation paged workspace has no free slot".to_string())
            })?;
        let generation = state
            .generation()
            .checked_add(1)
            .ok_or_else(|| invalid("invocation paged workspace slot generation exhausted"))?;
        *state = InvocationPagedKvSlotState::Preparing { generation };
        let slot = u32::try_from(slot_index)
            .map_err(|_| invalid("invocation paged workspace slot index exceeds u32"))?;
        let first_page = slot
            .checked_mul(self.inner.pages_per_slot)
            .and_then(|offset| self.inner.first_page.checked_add(offset))
            .ok_or_else(|| invalid("invocation paged workspace slot range overflow"))?;
        Ok(InvocationPagedKvSlotRef {
            pool: self.inner.id,
            slot,
            slot_generation: generation,
            first_page,
            page_count: self.inner.pages_per_slot,
        })
    }
}

/// Unique generation pin over one zeroed invocation page range. The physical
/// cache and its authenticated backend completions remain owned here until an
/// explicit release or Drop; the type is intentionally not Clone.
pub(crate) struct InvocationPagedKvLease {
    inner: Arc<InvocationPagedKvPoolInner>,
    slot: InvocationPagedKvSlotRef,
    cache: Option<PhysicalPagedKvCache>,
    released: bool,
}

impl std::fmt::Debug for InvocationPagedKvLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationPagedKvLease")
            .field("slot", &self.slot)
            .field("released", &self.released)
            .finish_non_exhaustive()
    }
}

impl InvocationPagedKvLease {
    pub(crate) const fn slot(&self) -> InvocationPagedKvSlotRef {
        self.slot
    }

    pub(crate) fn cache(&self) -> &PhysicalPagedKvCache {
        self.cache
            .as_ref()
            .expect("live invocation paged lease must retain its cache")
    }

    pub(crate) fn cache_mut(&mut self) -> &mut PhysicalPagedKvCache {
        self.cache
            .as_mut()
            .expect("live invocation paged lease must retain its cache")
    }

    /// Release the slot and return only backend-sealed write completions that
    /// authenticate slots belonging to this exact lease generation.
    pub(crate) fn release(mut self) -> Result<InvocationPagedKvCompletion> {
        let mut cache = self
            .cache
            .take()
            .ok_or_else(|| invalid("invocation paged workspace lease was already released"))?;
        let writes = cache.take_completed_writes();
        let validation = validate_completions(self.inner.as_ref(), self.slot, &writes);
        release_slot(
            self.inner.as_ref(),
            self.slot,
            InvocationPagedKvSlotKind::Leased,
        );
        self.released = true;
        validation?;
        Ok(InvocationPagedKvCompletion {
            slot: self.slot,
            writes,
        })
    }
}

impl Drop for InvocationPagedKvLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // Dropping the cache first drops every retained authenticated
        // completion before the generation can be reused. The next lease
        // still performs an unconditional zero+fence before exposure.
        self.cache.take();
        release_slot(
            self.inner.as_ref(),
            self.slot,
            InvocationPagedKvSlotKind::Leased,
        );
        self.released = true;
    }
}

#[derive(Debug)]
pub(crate) struct InvocationPagedKvCompletion {
    pub(crate) slot: InvocationPagedKvSlotRef,
    pub(crate) writes: Vec<Arc<KvWriteBatchCompletion>>,
}

#[derive(Debug, Clone, Copy)]
enum InvocationPagedKvSlotKind {
    Preparing,
    Leased,
}

fn transition_to_leased(
    inner: &InvocationPagedKvPoolInner,
    slot: InvocationPagedKvSlotRef,
) -> bool {
    if slot.pool != inner.id {
        return false;
    }
    let Ok(mut slots) = inner.slots.lock() else {
        return false;
    };
    let Some(state) = slots.get_mut(slot.slot as usize) else {
        return false;
    };
    if *state
        != (InvocationPagedKvSlotState::Preparing {
            generation: slot.slot_generation,
        })
    {
        return false;
    }
    *state = InvocationPagedKvSlotState::Leased {
        generation: slot.slot_generation,
    };
    true
}

fn release_slot(
    inner: &InvocationPagedKvPoolInner,
    slot: InvocationPagedKvSlotRef,
    kind: InvocationPagedKvSlotKind,
) {
    if slot.pool != inner.id {
        return;
    }
    let Ok(mut slots) = inner.slots.lock() else {
        return;
    };
    let Some(state) = slots.get_mut(slot.slot as usize) else {
        return;
    };
    let expected = match kind {
        InvocationPagedKvSlotKind::Preparing => InvocationPagedKvSlotState::Preparing {
            generation: slot.slot_generation,
        },
        InvocationPagedKvSlotKind::Leased => InvocationPagedKvSlotState::Leased {
            generation: slot.slot_generation,
        },
    };
    if *state == expected {
        *state = InvocationPagedKvSlotState::Vacant {
            generation: slot.slot_generation,
        };
    }
}

fn blocks_for_slot(
    inner: &InvocationPagedKvPoolInner,
    slot: InvocationPagedKvSlotRef,
) -> Result<Vec<CacheBlockRef>> {
    let expected_first_page = slot
        .slot
        .checked_mul(inner.pages_per_slot)
        .and_then(|offset| inner.first_page.checked_add(offset));
    if slot.pool != inner.id
        || slot.slot as usize >= inner.slots.lock().map(|slots| slots.len()).unwrap_or(0)
        || slot.slot_generation == 0
        || slot.page_count != inner.pages_per_slot
        || Some(slot.first_page) != expected_first_page
    {
        return Err(invalid(
            "invocation paged workspace slot belongs to another pool",
        ));
    }
    (0..slot.page_count)
        .map(|offset| {
            Ok(CacheBlockRef {
                arena: inner.id.arena,
                group: inner.arena.config().group,
                index: slot
                    .first_page
                    .checked_add(offset)
                    .ok_or_else(|| invalid("invocation paged workspace block overflow"))?,
                slot_generation: slot.slot_generation,
            })
        })
        .collect()
}

fn validate_completions(
    inner: &InvocationPagedKvPoolInner,
    slot: InvocationPagedKvSlotRef,
    writes: &[Arc<KvWriteBatchCompletion>],
) -> Result<()> {
    let blocks = blocks_for_slot(inner, slot)?
        .into_iter()
        .collect::<HashSet<_>>();
    for completion in writes {
        if completion.arena() != inner.id.arena
            || completion.layers() != inner.layer_bindings.as_slice()
            || completion.page_tokens() != inner.arena.config().page_tokens
            || completion.slots().iter().any(|written| {
                written.block.slot_generation != slot.slot_generation
                    || !blocks.contains(&written.block)
            })
        {
            return Err(Error::InferenceError(
                "invocation paged completion crossed its lease generation".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_arena_geometry(
    resolved: &crate::kv::v2::ResolvedPagedAttentionGroup,
    semantic: &crate::kv::v2::PagedAttentionDomainSpec,
    arena: &dyn KvArena,
) -> Result<()> {
    if resolved.layout != StatePhysicalLayout::PageTokenHeadDim
        || resolved.page_tokens != arena.config().page_tokens
        || resolved.group.get() != arena.config().group.get()
    {
        return Err(invalid(
            "invocation paged workspace arena has incompatible group, page, or layout geometry",
        ));
    }
    let expected_dtype = match resolved.storage.dtype() {
        StateDType::F32 => DType::F32,
        StateDType::F16 => DType::F16,
        StateDType::Bf16 => DType::BF16,
        StateDType::I8 | StateDType::Q4 => {
            return Err(invalid(
                "invocation paged workspace has no dense physical arena ABI for quantized state",
            ));
        }
    };
    if expected_dtype != arena.config().dtype
        || resolved.layers.len() != arena.config().layers.len()
        || resolved
            .layers
            .iter()
            .zip(&semantic.layers)
            .zip(&arena.config().layers)
            .any(|((expected, semantic), actual)| {
                expected.model_layer != actual.binding.model_layer
                    || expected.physical_layer != actual.binding.physical_layer
                    || semantic.model_layer != expected.model_layer
                    || semantic.kv_heads != actual.num_kv_heads
                    || semantic.key_head_dim != actual.key_head_dim
                    || semantic.value_head_dim != actual.value_head_dim
            })
    {
        return Err(invalid(
            "invocation paged workspace arena does not match resolved dtype or layer bindings",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::engine::ModelInstanceId;
    use crate::kv::v2::{
        test_contract, CheckpointPolicy, InvocationWorkspaceDomain, PlacementPolicy, PrefixPolicy,
        StateDomainSpec, StateScope, WorkspaceFormula,
    };
    use crate::kv::KvGroupId;

    fn arena(plan: &ResolvedStatePlan, capacity_pages: u32) -> Arc<dyn KvArena> {
        let resolved = &plan.paged_attention[0];
        Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: KvArenaId {
                    model_instance: ModelInstanceId::new(7),
                    backend: crate::backends::BackendKind::Cpu,
                    device_ordinal: None,
                    generation: 11,
                },
                group: KvGroupId::new(resolved.group.get()),
                page_tokens: resolved.page_tokens,
                capacity_pages,
                dtype: DType::F32,
                layers: resolved
                    .layers
                    .iter()
                    .map(|binding| KvLayerConfig {
                        binding: KvLayerBinding {
                            model_layer: binding.model_layer,
                            physical_layer: binding.physical_layer,
                        },
                        num_kv_heads: 2,
                        key_head_dim: 4,
                        value_head_dim: 4,
                    })
                    .collect(),
            })
            .unwrap(),
        )
    }

    fn plan() -> (ResolvedStatePlan, InvocationWorkspaceDomain) {
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
        let workspace_domain = InvocationWorkspaceDomain::State {
            state: contract.domains[0].clone(),
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
                backend: crate::backends::BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(16),
                storage_dtype_hint: Some(StateDType::F32),
            },
        )
        .unwrap();
        (plan, workspace_domain)
    }

    #[test]
    fn lease_zeroes_before_exposure_and_reuses_with_a_new_generation() {
        let (plan, workspace_domain) = plan();
        let arena = arena(&plan, 2);
        let pool = InvocationPagedKvPool::new(&plan, &workspace_domain, arena.clone(), 0, 2, 1, 4)
            .unwrap();
        assert_eq!(pool.maximum_tokens_per_lease().unwrap(), 32);
        let first = pool.lease().unwrap();
        let first_slot = first.slot();
        assert!(pool.contains_active_lease(first_slot));
        assert_eq!(arena.operation_stats().page_zero_dispatches, 1);
        assert!(pool.lease().is_err());
        drop(first);

        let second = pool.lease().unwrap();
        assert_eq!(second.slot().slot, first_slot.slot);
        assert!(second.slot().slot_generation > first_slot.slot_generation);
        assert!(!pool.contains_active_lease(first_slot));
        assert_eq!(arena.operation_stats().page_zero_dispatches, 2);
    }

    #[test]
    fn explicit_release_returns_only_authenticated_backend_completions() {
        let (plan, workspace_domain) = plan();
        let arena = arena(&plan, 1);
        let pool = InvocationPagedKvPool::new(&plan, &workspace_domain, arena, 0, 1, 1, 5).unwrap();
        let mut lease = pool.lease().unwrap();
        let mut prepared = lease.cache().prepare_append(0, 1).unwrap();
        let queries = Tensor::from_vec(vec![1_f32; 8], (1, 2, 4), &Device::Cpu).unwrap();
        let keys = Tensor::from_vec(vec![1_f32; 8], (1, 2, 4), &Device::Cpu).unwrap();
        let values = Tensor::from_vec(vec![1_f32; 8], (1, 2, 4), &Device::Cpu).unwrap();
        lease
            .cache()
            .write_and_attend(0, &mut prepared, &queries, &keys, &values, 0.5)
            .unwrap();
        lease.cache_mut().commit_prepared(prepared).unwrap();
        let completion = lease.release().unwrap();
        assert_eq!(completion.writes.len(), 1);
        assert_eq!(completion.slot.pool, pool.id());

        let next = pool.lease().unwrap();
        assert!(next.slot().slot_generation > completion.slot.slot_generation);
    }

    #[test]
    fn pool_rejects_nonmatching_domain_or_overlapping_capacity() {
        let (plan, workspace_domain) = plan();
        let arena = arena(&plan, 1);
        let mut wrong_domain = workspace_domain.clone();
        let InvocationWorkspaceDomain::State {
            state: StateDomainSpec::PagedAttention(domain),
            ..
        } = &mut wrong_domain
        else {
            unreachable!()
        };
        domain.header.id = StateDomainId::new(99);
        assert!(
            InvocationPagedKvPool::new(&plan, &wrong_domain, arena.clone(), 0, 1, 1, 1,).is_err()
        );
        assert!(InvocationPagedKvPool::new(&plan, &workspace_domain, arena, 0, 1, 2, 1,).is_err());
    }
}
