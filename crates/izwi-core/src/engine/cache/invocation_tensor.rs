//! Load-owned generational pools for invocation-scoped typed tensor state.
//!
//! Every slot owns one load-time allocated arena. Request admission only
//! reserves, scrubs, and fences an existing slot; it never allocates backing
//! tensors. Completion receipts are one-use pool-state capabilities.

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use candle_core::Device;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::backends::state::{
    InvocationTensorArena, InvocationTensorChronologicalSegment, InvocationTensorDomainKind,
    InvocationTensorSnapshot, InvocationTensorUpdateV2,
};
use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};
use crate::kv::v2::{
    DomainStepIntent, InferenceStateContract, InvocationStateBackingKindV2,
    InvocationWorkspaceBackingIdentityV2, InvocationWorkspaceBackingV2, InvocationWorkspaceDomain,
    InvocationWorkspacePhysicalCompletionV2, InvocationWorkspacePhysicalLeaseV2, ResolvedStatePlan,
    StateDomainId, StatePlanId,
};

const RECEIPT_DOMAIN: &[u8] = b"izwi.invocation-tensor-receipt.v1\0";
static NEXT_POOL_INSTANCE_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct InvocationTensorPoolId {
    pub(crate) plan: StatePlanId,
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) backend: BackendKind,
    pub(crate) domain: StateDomainId,
    pub(crate) kind: InvocationStateBackingKindV2,
    pub(crate) allocation_generation: u32,
}

impl InvocationTensorPoolId {
    const fn backing_identity(self) -> InvocationWorkspaceBackingIdentityV2 {
        InvocationWorkspaceBackingIdentityV2::Typed {
            kind: self.kind,
            model_instance: self.model_instance,
            backend: self.backend,
            domain: self.domain,
            allocation_generation: self.allocation_generation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct InvocationTensorSlotRef {
    pub(crate) pool: InvocationTensorPoolId,
    pub(crate) slot: u32,
    pub(crate) slot_generation: u32,
    pub(crate) nonce: u64,
}

#[derive(Debug)]
pub(crate) struct InvocationTensorPoolOwner {
    inner: Arc<InvocationTensorPoolInner>,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationTensorPoolHandle {
    id: InvocationTensorPoolId,
    workspace_domain: InvocationWorkspaceDomain,
    maximum_bytes: u64,
    inner: Weak<InvocationTensorPoolInner>,
}

struct InvocationTensorPoolInner {
    id: InvocationTensorPoolId,
    instance_nonce: u64,
    workspace_domain: InvocationWorkspaceDomain,
    maximum_bytes: u64,
    arenas: Vec<Mutex<InvocationTensorArena>>,
    state: Mutex<InvocationTensorPoolState>,
}

impl std::fmt::Debug for InvocationTensorPoolInner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationTensorPoolInner")
            .field("id", &self.id)
            .field("maximum_bytes", &self.maximum_bytes)
            .field("slot_count", &self.arenas.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct InvocationTensorPoolState {
    lifecycle: InvocationTensorPoolLifecycle,
    owner_alive: bool,
    next_nonce: u64,
    drain_cursor: usize,
    slots: Vec<InvocationTensorSlotState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationTensorPoolLifecycle {
    Accepting,
    Draining,
    DrainInFlight,
    Drained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationTensorSlotState {
    Vacant {
        generation: u32,
    },
    Preparing {
        generation: u32,
        nonce: u64,
    },
    Leased {
        generation: u32,
        nonce: u64,
    },
    Completing {
        generation: u32,
        nonce: u64,
        receipt: [u8; 32],
    },
}

impl InvocationTensorSlotState {
    const fn generation(self) -> u32 {
        match self {
            Self::Vacant { generation }
            | Self::Preparing { generation, .. }
            | Self::Leased { generation, .. }
            | Self::Completing { generation, .. } => generation,
        }
    }

    const fn is_vacant(self) -> bool {
        matches!(self, Self::Vacant { .. })
    }
}

impl InvocationTensorPoolOwner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        contract: &InferenceStateContract,
        plan: Arc<ResolvedStatePlan>,
        workspace_domain: InvocationWorkspaceDomain,
        device: Device,
        model_instance: ModelInstanceId,
        slot_count: u32,
        allocation_generation: u32,
    ) -> Result<Self> {
        if model_instance.get() == 0 || slot_count == 0 || allocation_generation == 0 {
            return Err(invalid(
                "invocation tensor pool requires non-zero model, slot, and allocation identities",
            ));
        }
        let slot_count_usize = usize::try_from(slot_count)
            .map_err(|_| invalid("invocation tensor pool slot count exceeds usize"))?;
        let mut arenas = Vec::with_capacity(slot_count_usize);
        for _ in 0..slot_count_usize {
            arenas.push(Mutex::new(InvocationTensorArena::new(
                contract,
                plan.clone(),
                workspace_domain.clone(),
                device.clone(),
            )?));
        }
        let first = arenas
            .first()
            .ok_or_else(|| invalid("invocation tensor pool has no physical arena"))?
            .lock()
            .map_err(|_| invalid("invocation tensor arena lock is poisoned"))?;
        let per_slot_bytes = first.maximum_bytes();
        let domain = first.domain();
        let kind = backing_kind(first.kind());
        drop(first);
        let maximum_bytes = per_slot_bytes
            .checked_mul(u64::from(slot_count))
            .ok_or_else(|| invalid("invocation tensor pool byte accounting overflow"))?;
        for arena in &arenas {
            let arena = arena
                .lock()
                .map_err(|_| invalid("invocation tensor arena lock is poisoned"))?;
            if arena.domain() != domain
                || backing_kind(arena.kind()) != kind
                || arena.maximum_bytes() != per_slot_bytes
            {
                return Err(invalid(
                    "invocation tensor pool slots do not share exact physical geometry",
                ));
            }
        }
        let id = InvocationTensorPoolId {
            plan: plan.id,
            model_instance,
            backend: plan.backend,
            domain,
            kind,
            allocation_generation,
        };
        let instance_nonce = NEXT_POOL_INSTANCE_NONCE
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |nonce| {
                nonce.checked_add(1)
            })
            .map_err(|_| invalid("invocation tensor pool instance nonce exhausted"))?;
        Ok(Self {
            inner: Arc::new(InvocationTensorPoolInner {
                id,
                instance_nonce,
                workspace_domain,
                maximum_bytes,
                arenas,
                state: Mutex::new(InvocationTensorPoolState {
                    lifecycle: InvocationTensorPoolLifecycle::Accepting,
                    owner_alive: true,
                    next_nonce: 0,
                    drain_cursor: 0,
                    slots: vec![
                        InvocationTensorSlotState::Vacant { generation: 0 };
                        slot_count_usize
                    ],
                }),
            }),
        })
    }

    pub(crate) fn id(&self) -> InvocationTensorPoolId {
        self.inner.id
    }

    pub(crate) fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        &self.inner.workspace_domain
    }

    pub(crate) fn maximum_bytes(&self) -> u64 {
        self.inner.maximum_bytes
    }

    pub(crate) fn handle(&self) -> InvocationTensorPoolHandle {
        InvocationTensorPoolHandle {
            id: self.inner.id,
            workspace_domain: self.inner.workspace_domain.clone(),
            maximum_bytes: self.inner.maximum_bytes,
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub(crate) fn backing(&self) -> Arc<dyn InvocationWorkspaceBackingV2> {
        Arc::new(self.handle())
    }

    pub(crate) fn lease(&self) -> Result<InvocationTensorLease> {
        lease_from_inner(self.inner.clone())
    }

    pub(crate) fn close_and_drain(&self) -> Result<()> {
        close_and_drain(self.inner.as_ref())
    }

    #[cfg(test)]
    fn is_drained(&self) -> bool {
        self.inner
            .state
            .lock()
            .map(|state| state.lifecycle == InvocationTensorPoolLifecycle::Drained)
            .unwrap_or(false)
    }
}

impl Drop for InvocationTensorPoolOwner {
    fn drop(&mut self) {
        let should_drain = {
            let mut state = lifecycle_state(self.inner.as_ref());
            state.owner_alive = false;
            if state.lifecycle == InvocationTensorPoolLifecycle::Accepting {
                state.lifecycle = InvocationTensorPoolLifecycle::Draining;
            }
            begin_drain_if_idle(&mut state)
        };
        if should_drain {
            let _ = finish_drain(self.inner.as_ref());
        }
    }
}

impl InvocationTensorPoolHandle {
    pub(crate) const fn id(&self) -> InvocationTensorPoolId {
        self.id
    }

    pub(crate) fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        &self.workspace_domain
    }

    pub(crate) const fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }

    pub(crate) fn validate_live(&self) -> Result<()> {
        let inner = self.inner.upgrade().ok_or_else(|| {
            invalid("invocation tensor pool refers to a retired physical generation")
        })?;
        let state = inner
            .state
            .lock()
            .map_err(|_| invalid("invocation tensor pool state is poisoned"))?;
        if !state.owner_alive || state.lifecycle != InvocationTensorPoolLifecycle::Accepting {
            return Err(invalid(
                "invocation tensor pool is not accepting new leases",
            ));
        }
        Ok(())
    }

    pub(crate) fn lease(&self) -> Result<InvocationTensorLease> {
        self.validate_live()?;
        let inner = self
            .inner
            .upgrade()
            .ok_or_else(|| invalid("invocation tensor pool retired during admission"))?;
        lease_from_inner(inner)
    }

    pub(crate) fn backing(&self) -> Arc<dyn InvocationWorkspaceBackingV2> {
        Arc::new(self.clone())
    }
}

impl InvocationWorkspaceBackingV2 for InvocationTensorPoolHandle {
    fn identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
        self.id.backing_identity()
    }

    fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        &self.workspace_domain
    }

    fn validate_live(&self) -> Result<()> {
        InvocationTensorPoolHandle::validate_live(self)
    }

    fn lease(&self) -> Result<Box<dyn InvocationWorkspacePhysicalLeaseV2>> {
        Ok(Box::new(InvocationTensorPoolHandle::lease(self)?))
    }

    fn authenticate_completion(
        &self,
        completion: &InvocationWorkspacePhysicalCompletionV2,
    ) -> Result<()> {
        let InvocationWorkspacePhysicalCompletionV2::Typed {
            backing,
            authentication,
        } = completion
        else {
            return Err(invalid(
                "invocation tensor completion has a non-typed physical kind",
            ));
        };
        if *backing != self.id.backing_identity() {
            return Err(invalid(
                "invocation tensor completion belongs to another backing",
            ));
        }
        let inner = self.inner.upgrade().ok_or_else(|| {
            invalid("invocation tensor completion refers to a retired generation")
        })?;
        authenticate_receipt(inner.as_ref(), *authentication)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationTensorLeasePhase {
    Leased,
    Completing { receipt: [u8; 32] },
    Released,
}

pub(crate) struct InvocationTensorLease {
    inner: Arc<InvocationTensorPoolInner>,
    slot: InvocationTensorSlotRef,
    phase: InvocationTensorLeasePhase,
}

impl std::fmt::Debug for InvocationTensorLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationTensorLease")
            .field("slot", &self.slot)
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

impl InvocationTensorLease {
    pub(crate) const fn slot(&self) -> InvocationTensorSlotRef {
        self.slot
    }

    pub(crate) fn apply_intent(
        &mut self,
        intent: &DomainStepIntent,
        update: InvocationTensorUpdateV2,
    ) -> Result<()> {
        self.require_leased()?;
        let mut arena = lock_arena(self.inner.as_ref(), self.slot.slot)?;
        arena.apply_intent(intent, update)
    }

    pub(crate) fn read_snapshot(&self) -> Result<InvocationTensorSnapshot> {
        self.require_leased()?;
        lock_arena(self.inner.as_ref(), self.slot.slot)?.read_snapshot()
    }

    pub(crate) fn read_chronological_segments(
        &self,
    ) -> Result<Vec<InvocationTensorChronologicalSegment>> {
        self.require_leased()?;
        lock_arena(self.inner.as_ref(), self.slot.slot)?.read_chronological_segments()
    }

    fn require_leased(&self) -> Result<()> {
        if self.phase != InvocationTensorLeasePhase::Leased {
            return Err(invalid("invocation tensor lease is no longer model-facing"));
        }
        Ok(())
    }

    fn begin_completion(&mut self) -> Result<InvocationWorkspacePhysicalCompletionV2> {
        self.require_leased()?;
        let receipt = transition_to_completing(self.inner.as_ref(), self.slot)?;
        self.phase = InvocationTensorLeasePhase::Completing { receipt };
        Ok(InvocationWorkspacePhysicalCompletionV2::Typed {
            backing: self.inner.id.backing_identity(),
            authentication: receipt,
        })
    }

    fn abort_once(&mut self) {
        if self.phase == InvocationTensorLeasePhase::Released {
            return;
        }
        release_slot(self.inner.as_ref(), self.slot);
        self.phase = InvocationTensorLeasePhase::Released;
    }
}

impl InvocationWorkspacePhysicalLeaseV2 for InvocationTensorLease {
    fn backing_identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
        self.inner.id.backing_identity()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn complete(&mut self) -> Result<InvocationWorkspacePhysicalCompletionV2> {
        self.begin_completion()
    }

    fn abort(&mut self) {
        self.abort_once();
    }
}

impl Drop for InvocationTensorLease {
    fn drop(&mut self) {
        self.abort_once();
    }
}

fn lease_from_inner(inner: Arc<InvocationTensorPoolInner>) -> Result<InvocationTensorLease> {
    let slot = begin_lease(inner.as_ref())?;
    let mut reservation = PreparingSlotGuard::new(inner.clone(), slot);
    reset_arena_for_reuse(inner.as_ref(), slot.slot)?;
    if !transition_to_leased(inner.as_ref(), slot) {
        return Err(invalid(
            "invocation tensor slot lost its preparing generation",
        ));
    }
    reservation.disarm();
    Ok(InvocationTensorLease {
        inner,
        slot,
        phase: InvocationTensorLeasePhase::Leased,
    })
}

struct PreparingSlotGuard {
    inner: Arc<InvocationTensorPoolInner>,
    slot: InvocationTensorSlotRef,
    armed: bool,
}

impl PreparingSlotGuard {
    fn new(inner: Arc<InvocationTensorPoolInner>, slot: InvocationTensorSlotRef) -> Self {
        Self {
            inner,
            slot,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PreparingSlotGuard {
    fn drop(&mut self) {
        if self.armed {
            release_slot(self.inner.as_ref(), self.slot);
        }
    }
}

fn begin_lease(inner: &InvocationTensorPoolInner) -> Result<InvocationTensorSlotRef> {
    let mut state = inner
        .state
        .lock()
        .map_err(|_| invalid("invocation tensor pool state is poisoned"))?;
    if !state.owner_alive || state.lifecycle != InvocationTensorPoolLifecycle::Accepting {
        return Err(invalid("invocation tensor pool is closed for admission"));
    }
    let slot_index = state
        .slots
        .iter()
        .position(|slot| slot.is_vacant())
        .ok_or_else(|| {
            Error::Backpressure("invocation tensor pool has no free slot".to_string())
        })?;
    let generation = state.slots[slot_index]
        .generation()
        .checked_add(1)
        .ok_or_else(|| invalid("invocation tensor slot generation exhausted"))?;
    state.next_nonce = state
        .next_nonce
        .checked_add(1)
        .ok_or_else(|| invalid("invocation tensor lease nonce exhausted"))?;
    let nonce = state.next_nonce;
    state.slots[slot_index] = InvocationTensorSlotState::Preparing { generation, nonce };
    Ok(InvocationTensorSlotRef {
        pool: inner.id,
        slot: u32::try_from(slot_index)
            .map_err(|_| invalid("invocation tensor slot index exceeds u32"))?,
        slot_generation: generation,
        nonce,
    })
}

fn transition_to_leased(inner: &InvocationTensorPoolInner, slot: InvocationTensorSlotRef) -> bool {
    if slot.pool != inner.id {
        return false;
    }
    let Ok(mut state) = inner.state.lock() else {
        return false;
    };
    let Some(current) = state.slots.get_mut(slot.slot as usize) else {
        return false;
    };
    if *current
        != (InvocationTensorSlotState::Preparing {
            generation: slot.slot_generation,
            nonce: slot.nonce,
        })
    {
        return false;
    }
    *current = InvocationTensorSlotState::Leased {
        generation: slot.slot_generation,
        nonce: slot.nonce,
    };
    true
}

fn transition_to_completing(
    inner: &InvocationTensorPoolInner,
    slot: InvocationTensorSlotRef,
) -> Result<[u8; 32]> {
    if slot.pool != inner.id {
        return Err(invalid("invocation tensor lease belongs to another pool"));
    }
    let receipt = receipt_for(inner, slot)?;
    let mut state = inner
        .state
        .lock()
        .map_err(|_| invalid("invocation tensor pool state is poisoned"))?;
    let current = state
        .slots
        .get_mut(slot.slot as usize)
        .ok_or_else(|| invalid("invocation tensor slot index is out of bounds"))?;
    if *current
        != (InvocationTensorSlotState::Leased {
            generation: slot.slot_generation,
            nonce: slot.nonce,
        })
    {
        return Err(invalid(
            "invocation tensor lease generation is no longer active",
        ));
    }
    *current = InvocationTensorSlotState::Completing {
        generation: slot.slot_generation,
        nonce: slot.nonce,
        receipt,
    };
    Ok(receipt)
}

fn authenticate_receipt(inner: &InvocationTensorPoolInner, receipt: [u8; 32]) -> Result<()> {
    let mut state = inner
        .state
        .lock()
        .map_err(|_| invalid("invocation tensor pool state is poisoned"))?;
    let (slot_index, generation) = state
        .slots
        .iter()
        .enumerate()
        .find_map(|(index, slot)| match *slot {
            InvocationTensorSlotState::Completing {
                generation,
                receipt: candidate,
                ..
            } if candidate == receipt => Some((index, generation)),
            _ => None,
        })
        .ok_or_else(|| invalid("invocation tensor completion receipt is stale or foreign"))?;
    state.slots[slot_index] = InvocationTensorSlotState::Vacant { generation };
    Ok(())
}

fn release_slot(inner: &InvocationTensorPoolInner, slot: InvocationTensorSlotRef) {
    if slot.pool != inner.id {
        return;
    }
    let should_drain = {
        let mut state = lifecycle_state(inner);
        let Some(current) = state.slots.get_mut(slot.slot as usize) else {
            return;
        };
        let matches_generation = matches!(
            *current,
            InvocationTensorSlotState::Preparing { generation, nonce }
                | InvocationTensorSlotState::Leased { generation, nonce }
                | InvocationTensorSlotState::Completing {
                    generation,
                    nonce,
                    ..
                } if generation == slot.slot_generation && nonce == slot.nonce
        );
        if matches_generation {
            *current = InvocationTensorSlotState::Vacant {
                generation: slot.slot_generation,
            };
        }
        !state.owner_alive && begin_drain_if_idle(&mut state)
    };
    if should_drain {
        let _ = finish_drain(inner);
    }
}

fn begin_drain_if_idle(state: &mut InvocationTensorPoolState) -> bool {
    if state.lifecycle == InvocationTensorPoolLifecycle::Draining
        && state.slots.iter().all(|slot| slot.is_vacant())
    {
        state.lifecycle = InvocationTensorPoolLifecycle::DrainInFlight;
        true
    } else {
        false
    }
}

fn close_and_drain(inner: &InvocationTensorPoolInner) -> Result<()> {
    let should_drain = {
        let mut state = lifecycle_state(inner);
        match state.lifecycle {
            InvocationTensorPoolLifecycle::Drained => return Ok(()),
            InvocationTensorPoolLifecycle::DrainInFlight => {
                return Err(Error::Backpressure(
                    "invocation tensor pool drain is already in flight".to_string(),
                ));
            }
            InvocationTensorPoolLifecycle::Accepting => {
                state.lifecycle = InvocationTensorPoolLifecycle::Draining;
            }
            InvocationTensorPoolLifecycle::Draining => {}
        }
        if !begin_drain_if_idle(&mut state) {
            return Err(Error::Backpressure(
                "invocation tensor pool still has active or completing leases".to_string(),
            ));
        }
        true
    };
    debug_assert!(should_drain);
    finish_drain(inner)
}

fn finish_drain(inner: &InvocationTensorPoolInner) -> Result<()> {
    loop {
        let next = {
            let state = lifecycle_state(inner);
            if state.drain_cursor == inner.arenas.len() {
                None
            } else {
                Some(state.drain_cursor)
            }
        };
        let Some(index) = next else {
            let mut state = lifecycle_state(inner);
            state.lifecycle = InvocationTensorPoolLifecycle::Drained;
            return Ok(());
        };
        let result = u32::try_from(index)
            .map_err(|_| invalid("invocation tensor drain index exceeds u32"))
            .and_then(|slot| reset_arena_for_reuse(inner, slot));
        let mut state = lifecycle_state(inner);
        match result {
            Ok(()) => {
                if state.drain_cursor == index {
                    state.drain_cursor += 1;
                }
            }
            Err(error) => {
                state.lifecycle = InvocationTensorPoolLifecycle::Draining;
                return Err(error);
            }
        }
    }
}

fn lock_arena(
    inner: &InvocationTensorPoolInner,
    slot: u32,
) -> Result<std::sync::MutexGuard<'_, InvocationTensorArena>> {
    inner
        .arenas
        .get(slot as usize)
        .ok_or_else(|| invalid("invocation tensor slot index is out of bounds"))?
        .lock()
        .map_err(|_| invalid("invocation tensor arena lock is poisoned"))
}

fn reset_arena_for_reuse(inner: &InvocationTensorPoolInner, slot: u32) -> Result<()> {
    let mutex = inner
        .arenas
        .get(slot as usize)
        .ok_or_else(|| invalid("invocation tensor slot index is out of bounds"))?;
    match mutex.lock() {
        Ok(mut arena) => arena.reset_for_reuse(),
        Err(poisoned) => {
            let mut arena = poisoned.into_inner();
            let result = arena.reset_for_reuse();
            drop(arena);
            if result.is_ok() {
                mutex.clear_poison();
            }
            result
        }
    }
}

fn lifecycle_state(
    inner: &InvocationTensorPoolInner,
) -> std::sync::MutexGuard<'_, InvocationTensorPoolState> {
    match inner.state.lock() {
        Ok(state) => state,
        Err(poisoned) => {
            let state = poisoned.into_inner();
            inner.state.clear_poison();
            state
        }
    }
}

fn receipt_for(
    inner: &InvocationTensorPoolInner,
    slot: InvocationTensorSlotRef,
) -> Result<[u8; 32]> {
    let encoded = serde_json::to_vec(&slot).map_err(|error| {
        invalid(format!(
            "failed to encode invocation tensor receipt: {error}"
        ))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(inner.instance_nonce.to_le_bytes());
    hasher.update(encoded);
    Ok(hasher.finalize().into())
}

const fn backing_kind(kind: InvocationTensorDomainKind) -> InvocationStateBackingKindV2 {
    match kind {
        InvocationTensorDomainKind::StaticTensor => InvocationStateBackingKindV2::StaticTensor,
        InvocationTensorDomainKind::Tensor => InvocationStateBackingKindV2::Tensor,
        InvocationTensorDomainKind::Append => InvocationStateBackingKindV2::Append,
        InvocationTensorDomainKind::Ring => InvocationStateBackingKindV2::Ring,
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use candle_core::Tensor;

    use super::*;
    use crate::backends::state::{
        negotiate_state_plan, InvocationTensorComponentValue, StateBackendPlanRequest,
    };
    use crate::kv::v2::{
        test_contract, BoundedShape, CheckpointPolicy, ComponentShapeInstantiation,
        InferenceStateContract, InvocationStateCapacity, PlacementPolicy, PrefixPolicy, ShapeAxis,
        ShapeDimension, ShapeDimensionValue, ShapeExtent, StateClock, StateComponentId, StateDType,
        StateDomainHeader, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope,
        StateUpdateKind, TensorComponentSpec, TensorRole, TensorStateDomainSpec, WorkspaceFormula,
        CURRENT_INFERENCE_STATE_ABI,
    };

    fn component(max: u64) -> TensorComponentSpec {
        TensorComponentSpec {
            id: StateComponentId::new(1),
            role: TensorRole::Control,
            shape: BoundedShape {
                dimensions: vec![ShapeDimension {
                    axis: ShapeAxis::Hidden,
                    extent: ShapeExtent::RuntimeBounded { min: 1, max },
                }],
            },
            accepted_dtypes: vec![StateDType::F32],
        }
    }

    fn tensor_state(id: u32, max: u64) -> StateDomainSpec {
        StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: StateDomainId::new(id),
                scope: StateScope::Invocation,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::None,
            },
            components: vec![component(max)],
        })
    }

    fn contract(state: StateDomainSpec) -> InferenceStateContract {
        let domain = state.id();
        InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![state],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![domain],
                prefix_shareable: false,
            }],
        }
    }

    fn plan_and_workspace(
        contract: &InferenceStateContract,
        domain: StateDomainId,
    ) -> (Arc<ResolvedStatePlan>, InvocationWorkspaceDomain) {
        let plan = Arc::new(
            negotiate_state_plan(
                contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: None,
                },
            )
            .unwrap(),
        );
        let state = contract
            .domains
            .iter()
            .find(|state| state.id() == domain)
            .unwrap()
            .clone();
        let fixed_bytes = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == domain)
            .unwrap()
            .maximum_bytes();
        (
            plan,
            InvocationWorkspaceDomain::State {
                state,
                capacity: InvocationStateCapacity::SemanticBounded,
                placement: PlacementPolicy::BackendLocal,
                formula: WorkspaceFormula {
                    fixed_bytes,
                    dimensions: vec![],
                    terms: vec![],
                },
            },
        )
    }

    fn owner(slot_count: u32, allocation_generation: u32) -> InvocationTensorPoolOwner {
        owner_for_model(slot_count, allocation_generation, ModelInstanceId::new(7))
    }

    fn owner_for_model(
        slot_count: u32,
        allocation_generation: u32,
        model_instance: ModelInstanceId,
    ) -> InvocationTensorPoolOwner {
        let contract = contract(tensor_state(1, 4));
        let (plan, workspace) = plan_and_workspace(&contract, StateDomainId::new(1));
        InvocationTensorPoolOwner::new(
            &contract,
            plan,
            workspace,
            Device::Cpu,
            model_instance,
            slot_count,
            allocation_generation,
        )
        .unwrap()
    }

    fn declared(units: u64) -> ComponentShapeInstantiation {
        ComponentShapeInstantiation {
            component: StateComponentId::new(1),
            dimensions: vec![ShapeDimensionValue {
                axis: ShapeAxis::Hidden,
                units,
            }],
        }
    }

    fn value(values: &[f32]) -> InvocationTensorComponentValue {
        InvocationTensorComponentValue {
            component: StateComponentId::new(1),
            tensor: Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap(),
        }
    }

    fn replace_intent(expected: u64, target: u64, units: u64) -> DomainStepIntent {
        DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: expected,
            target_cursor: target,
            update: StateUpdateKind::TensorReplace {
                components: vec![declared(units)],
            },
        }
    }

    fn replace_update(values: &[f32]) -> InvocationTensorUpdateV2 {
        InvocationTensorUpdateV2::TensorReplace {
            components: vec![value(values)],
        }
    }

    #[test]
    fn slot_exhaustion_reuse_generation_and_scrub_are_exact() {
        let owner = owner(1, 11);
        let mut first = owner.lease().unwrap();
        let first_slot = first.slot();
        first
            .apply_intent(&replace_intent(0, 1, 2), replace_update(&[1.0, 2.0]))
            .unwrap();
        let held = first.read_snapshot().unwrap();
        assert!(matches!(owner.lease(), Err(Error::Backpressure(_))));
        drop(first);

        let mut second = owner.lease().unwrap();
        assert_eq!(second.slot().slot, first_slot.slot);
        assert!(second.slot().slot_generation > first_slot.slot_generation);
        assert_ne!(second.slot().nonce, first_slot.nonce);
        assert!(second.read_snapshot().is_err());
        second
            .apply_intent(&replace_intent(0, 1, 1), replace_update(&[9.0]))
            .unwrap();
        assert_eq!(
            second.read_snapshot().unwrap().components[0]
                .tensor
                .to_vec1::<f32>()
                .unwrap(),
            vec![9.0]
        );
        assert_eq!(
            held.components[0].tensor.to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn completion_receipt_rejects_foreign_backing_and_replay() {
        let owner = owner_for_model(1, 12, ModelInstanceId::new(7));
        let foreign = owner_for_model(1, 13, ModelInstanceId::new(8));
        let backing = owner.backing();
        let foreign_backing = foreign.backing();
        let mut physical = backing.lease().unwrap();
        let typed = physical
            .as_any_mut()
            .downcast_mut::<InvocationTensorLease>()
            .unwrap();
        typed
            .apply_intent(&replace_intent(0, 1, 1), replace_update(&[1.0]))
            .unwrap();
        let completion = physical.complete().unwrap();
        assert!(foreign_backing
            .authenticate_completion(&completion)
            .is_err());
        let authentication = match &completion {
            InvocationWorkspacePhysicalCompletionV2::Typed { authentication, .. } => {
                *authentication
            }
            InvocationWorkspacePhysicalCompletionV2::Paged(_) => unreachable!(),
        };
        let forged_generation = InvocationWorkspacePhysicalCompletionV2::Typed {
            backing: InvocationWorkspaceBackingIdentityV2::Typed {
                kind: InvocationStateBackingKindV2::Tensor,
                model_instance: ModelInstanceId::new(7),
                backend: BackendKind::Cpu,
                domain: StateDomainId::new(1),
                allocation_generation: 99,
            },
            authentication,
        };
        assert!(backing.authenticate_completion(&forged_generation).is_err());
        backing.authenticate_completion(&completion).unwrap();
        assert!(backing.authenticate_completion(&completion).is_err());
        drop(physical);

        let mut aborted = backing.lease().unwrap();
        let aborted_completion = aborted.complete().unwrap();
        aborted.abort();
        aborted.abort();
        assert!(backing
            .authenticate_completion(&aborted_completion)
            .is_err());
        drop(aborted);
        let next = owner.lease().unwrap();
        assert!(next.slot().slot_generation > 2);
    }

    #[test]
    fn identical_logical_pool_ids_have_distinct_one_use_receipts() {
        let first_owner = owner(1, 18);
        let second_owner = owner(1, 18);
        assert_eq!(first_owner.id(), second_owner.id());
        let first_backing = first_owner.backing();
        let second_backing = second_owner.backing();
        let mut first = first_backing.lease().unwrap();
        let mut second = second_backing.lease().unwrap();
        let first_completion = first.complete().unwrap();
        let second_completion = second.complete().unwrap();
        let receipt = |completion: &InvocationWorkspacePhysicalCompletionV2| match completion {
            InvocationWorkspacePhysicalCompletionV2::Typed { authentication, .. } => {
                *authentication
            }
            InvocationWorkspacePhysicalCompletionV2::Paged(_) => unreachable!(),
        };
        assert_ne!(receipt(&first_completion), receipt(&second_completion));
        assert!(second_backing
            .authenticate_completion(&first_completion)
            .is_err());
        first_backing
            .authenticate_completion(&first_completion)
            .unwrap();
        second_backing
            .authenticate_completion(&second_completion)
            .unwrap();
    }

    #[test]
    fn close_is_retryable_for_leased_and_completing_slots() {
        let leased_owner = owner(1, 14);
        let lease = leased_owner.lease().unwrap();
        assert!(matches!(
            leased_owner.close_and_drain(),
            Err(Error::Backpressure(_))
        ));
        assert!(leased_owner.handle().lease().is_err());
        drop(lease);
        leased_owner.close_and_drain().unwrap();
        assert!(leased_owner.is_drained());
        leased_owner.close_and_drain().unwrap();

        let completing_owner = owner(1, 15);
        let backing = completing_owner.backing();
        let mut physical = backing.lease().unwrap();
        let completion = physical.complete().unwrap();
        assert!(matches!(
            completing_owner.close_and_drain(),
            Err(Error::Backpressure(_))
        ));
        backing.authenticate_completion(&completion).unwrap();
        drop(physical);
        completing_owner.close_and_drain().unwrap();
        assert!(completing_owner.is_drained());
    }

    #[test]
    fn weak_handle_and_unwind_cannot_own_or_leak_a_generation() {
        let owner = owner(1, 16);
        let handle = owner.handle();
        let first_generation = catch_unwind(AssertUnwindSafe(|| {
            let lease = handle.lease().unwrap();
            let generation = lease.slot().slot_generation;
            panic!("model execution panic after acquisition: {generation}");
        }));
        assert!(first_generation.is_err());
        let next = handle.lease().unwrap();
        assert!(next.slot().slot_generation > 1);
        drop(next);
        drop(owner);
        assert!(handle.lease().is_err());
        assert!(handle.validate_live().is_err());
    }

    #[test]
    fn preparing_rollback_and_poisoned_arena_scrub_restore_capacity() {
        let owner = owner(1, 19);
        let reserved = begin_lease(owner.inner.as_ref()).unwrap();
        let unwind = catch_unwind(AssertUnwindSafe({
            let inner = owner.inner.clone();
            move || {
                let _reservation = PreparingSlotGuard::new(inner, reserved);
                panic!("panic after reserving a preparing slot");
            }
        }));
        assert!(unwind.is_err());
        let after_rollback = owner.lease().unwrap();
        assert!(after_rollback.slot().slot_generation > reserved.slot_generation);
        drop(after_rollback);

        let poison = catch_unwind(AssertUnwindSafe({
            let inner = owner.inner.clone();
            move || {
                let _arena = inner.arenas[0].lock().unwrap();
                panic!("panic while a model-facing arena guard is held");
            }
        }));
        assert!(poison.is_err());
        assert!(owner.inner.arenas[0].is_poisoned());
        let recovered = owner.lease().unwrap();
        assert!(!owner.inner.arenas[0].is_poisoned());
        assert!(recovered.read_snapshot().is_err());
        drop(recovered);
        owner.close_and_drain().unwrap();
        assert!(owner.is_drained());
    }

    #[test]
    fn lifecycle_recovers_poisoned_pool_state_for_close() {
        let owner = owner(1, 20);
        let poison = catch_unwind(AssertUnwindSafe({
            let inner = owner.inner.clone();
            move || {
                let _state = inner.state.lock().unwrap();
                panic!("panic while pool lifecycle state is held");
            }
        }));
        assert!(poison.is_err());
        assert!(owner.inner.state.is_poisoned());
        owner.close_and_drain().unwrap();
        assert!(!owner.inner.state.is_poisoned());
        assert!(owner.is_drained());
    }

    #[test]
    fn mixed_plan_constructs_only_the_selected_domain_and_accounts_all_slots() {
        let selected = tensor_state(2, 4);
        let other = tensor_state(3, 2);
        let mut contract = test_contract();
        contract.domains.push(selected.clone());
        contract.domains.push(other);
        contract.groups.push(StateGroupSpec {
            id: StateGroupId::new(2),
            domains: vec![StateDomainId::new(2)],
            prefix_shareable: false,
        });
        contract.groups.push(StateGroupSpec {
            id: StateGroupId::new(3),
            domains: vec![StateDomainId::new(3)],
            prefix_shareable: false,
        });
        let (plan, workspace) = plan_and_workspace(&contract, StateDomainId::new(2));
        let per_slot = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == StateDomainId::new(2))
            .unwrap()
            .maximum_bytes();
        let owner = InvocationTensorPoolOwner::new(
            &contract,
            plan,
            workspace,
            Device::Cpu,
            ModelInstanceId::new(9),
            3,
            17,
        )
        .unwrap();
        assert_eq!(owner.id().domain, StateDomainId::new(2));
        assert_eq!(owner.maximum_bytes(), per_slot * 3);
        assert_eq!(owner.handle().maximum_bytes(), per_slot * 3);
        let first = owner.lease().unwrap();
        let second = owner.lease().unwrap();
        let third = owner.lease().unwrap();
        assert!(matches!(owner.lease(), Err(Error::Backpressure(_))));
        drop((first, second, third));
    }
}
