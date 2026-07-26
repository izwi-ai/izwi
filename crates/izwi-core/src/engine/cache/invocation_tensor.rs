//! Reusable load-owned generational pools for invocation-scoped state.
//!
//! Every slot owns one load-time allocated arena. Request admission only
//! reserves, scrubs, and fences an existing slot; it never allocates backing
//! storage. Completion receipts are one-use pool-state capabilities. The
//! tensor aliases preserve the original API while other physical arenas reuse
//! the same lifecycle machinery.

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use candle_core::{Device, Tensor};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::backends::state::{
    InvocationRingDepthwiseConvTransaction, InvocationStaticAttentionArena, InvocationTensorArena,
    InvocationTensorChronologicalSegment, InvocationTensorDomainKind, InvocationTensorSnapshot,
    InvocationTensorUpdateV2, StaticAttentionLayerValue, StaticAttentionMetadata,
    StaticAttentionRaggedRow,
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

const RECEIPT_DOMAIN: &[u8] = b"izwi.invocation-slot-pool-receipt.v2\0";
static NEXT_POOL_INSTANCE_NONCE: AtomicU64 = AtomicU64::new(1);

/// Crate-sealed arena surface consumed by the reusable invocation slot pool.
/// Model-facing state operations remain inherent methods on each concrete
/// arena and its specialized lease.
pub(crate) mod slot_arena_sealed {
    pub(crate) trait Sealed {}
}

pub(crate) trait InvocationSlotArena:
    slot_arena_sealed::Sealed + std::fmt::Debug + Send + 'static
{
    fn plan(&self) -> &ResolvedStatePlan;

    fn workspace_domain(&self) -> &InvocationWorkspaceDomain;

    fn domain(&self) -> StateDomainId;

    fn backing_kind(&self) -> InvocationStateBackingKindV2;

    fn maximum_bytes(&self) -> u64;

    fn reset_for_reuse(&mut self) -> Result<()>;

    fn prepare_completion(&mut self) -> Result<()>;
}

impl slot_arena_sealed::Sealed for InvocationTensorArena {}

impl InvocationSlotArena for InvocationTensorArena {
    fn plan(&self) -> &ResolvedStatePlan {
        InvocationTensorArena::plan(self)
    }

    fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        InvocationTensorArena::workspace_domain(self)
    }

    fn domain(&self) -> StateDomainId {
        InvocationTensorArena::domain(self)
    }

    fn backing_kind(&self) -> InvocationStateBackingKindV2 {
        backing_kind(InvocationTensorArena::kind(self))
    }

    fn maximum_bytes(&self) -> u64 {
        InvocationTensorArena::maximum_bytes(self)
    }

    fn reset_for_reuse(&mut self) -> Result<()> {
        InvocationTensorArena::reset_for_reuse(self)
    }

    fn prepare_completion(&mut self) -> Result<()> {
        InvocationTensorArena::prepare_completion(self)
    }
}

impl slot_arena_sealed::Sealed for InvocationStaticAttentionArena {}

impl InvocationSlotArena for InvocationStaticAttentionArena {
    fn plan(&self) -> &ResolvedStatePlan {
        InvocationStaticAttentionArena::plan(self)
    }

    fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        InvocationStaticAttentionArena::workspace_domain(self)
    }

    fn domain(&self) -> StateDomainId {
        InvocationStaticAttentionArena::domain(self)
    }

    fn backing_kind(&self) -> InvocationStateBackingKindV2 {
        InvocationStateBackingKindV2::StaticAttention
    }

    fn maximum_bytes(&self) -> u64 {
        InvocationStaticAttentionArena::maximum_bytes(self)
    }

    fn reset_for_reuse(&mut self) -> Result<()> {
        InvocationStaticAttentionArena::reset_for_reuse(self)
    }

    fn prepare_completion(&mut self) -> Result<()> {
        InvocationStaticAttentionArena::prepare_completion(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct InvocationSlotPoolId {
    pub(crate) plan: StatePlanId,
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) backend: BackendKind,
    pub(crate) domain: StateDomainId,
    pub(crate) kind: InvocationStateBackingKindV2,
    pub(crate) allocation_generation: u32,
}

pub(crate) type InvocationTensorPoolId = InvocationSlotPoolId;

impl InvocationSlotPoolId {
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
pub(crate) struct InvocationSlotRef {
    pub(crate) pool: InvocationSlotPoolId,
    pub(crate) slot: u32,
    pub(crate) slot_generation: u32,
    pub(crate) nonce: u64,
}

pub(crate) type InvocationTensorSlotRef = InvocationSlotRef;

#[derive(Debug)]
pub(crate) struct InvocationSlotPoolOwner<A: InvocationSlotArena> {
    inner: Arc<InvocationSlotPoolInner<A>>,
}

pub(crate) type InvocationTensorPoolOwner = InvocationSlotPoolOwner<InvocationTensorArena>;
pub(crate) type InvocationStaticAttentionPoolOwner =
    InvocationSlotPoolOwner<InvocationStaticAttentionArena>;

#[derive(Debug)]
pub(crate) struct InvocationSlotPoolHandle<A: InvocationSlotArena> {
    id: InvocationSlotPoolId,
    workspace_domain: InvocationWorkspaceDomain,
    maximum_bytes: u64,
    inner: Weak<InvocationSlotPoolInner<A>>,
}

pub(crate) type InvocationTensorPoolHandle = InvocationSlotPoolHandle<InvocationTensorArena>;
pub(crate) type InvocationStaticAttentionPoolHandle =
    InvocationSlotPoolHandle<InvocationStaticAttentionArena>;

impl<A: InvocationSlotArena> Clone for InvocationSlotPoolHandle<A> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            workspace_domain: self.workspace_domain.clone(),
            maximum_bytes: self.maximum_bytes,
            inner: self.inner.clone(),
        }
    }
}

struct InvocationSlotPoolInner<A: InvocationSlotArena> {
    id: InvocationSlotPoolId,
    instance_nonce: u64,
    workspace_domain: InvocationWorkspaceDomain,
    maximum_bytes: u64,
    arenas: Vec<Mutex<A>>,
    state: Mutex<InvocationSlotPoolState>,
}

impl<A: InvocationSlotArena> std::fmt::Debug for InvocationSlotPoolInner<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationSlotPoolInner")
            .field("id", &self.id)
            .field("maximum_bytes", &self.maximum_bytes)
            .field("slot_count", &self.arenas.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct InvocationSlotPoolState {
    lifecycle: InvocationSlotPoolLifecycle,
    owner_alive: bool,
    next_nonce: u64,
    admission_cursor: usize,
    drain_cursor: usize,
    slots: Vec<InvocationSlotState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationSlotPoolLifecycle {
    Accepting,
    Draining,
    DrainInFlight,
    Drained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationSlotState {
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

impl InvocationSlotState {
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

impl InvocationSlotPoolOwner<InvocationTensorArena> {
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
            arenas.push(InvocationTensorArena::new(
                contract,
                plan.clone(),
                workspace_domain.clone(),
                device.clone(),
            )?);
        }
        Self::from_arenas(
            plan.as_ref(),
            workspace_domain,
            model_instance,
            allocation_generation,
            arenas,
        )
    }
}

impl InvocationSlotPoolOwner<InvocationStaticAttentionArena> {
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
                "invocation static-attention pool requires non-zero model, slot, and allocation identities",
            ));
        }
        let slot_count_usize = usize::try_from(slot_count)
            .map_err(|_| invalid("invocation static-attention slot count exceeds usize"))?;
        let mut arenas = Vec::with_capacity(slot_count_usize);
        for _ in 0..slot_count_usize {
            arenas.push(InvocationStaticAttentionArena::new(
                contract,
                plan.clone(),
                workspace_domain.clone(),
                device.clone(),
            )?);
        }
        Self::from_arenas(
            plan.as_ref(),
            workspace_domain,
            model_instance,
            allocation_generation,
            arenas,
        )
    }
}

impl<A: InvocationSlotArena> InvocationSlotPoolOwner<A> {
    pub(crate) fn from_arenas(
        plan: &ResolvedStatePlan,
        workspace_domain: InvocationWorkspaceDomain,
        model_instance: ModelInstanceId,
        allocation_generation: u32,
        arenas: Vec<A>,
    ) -> Result<Self> {
        if model_instance.get() == 0 || arenas.is_empty() || allocation_generation == 0 {
            return Err(invalid(
                "invocation slot pool requires non-zero model, slot, and allocation identities",
            ));
        }
        let slot_count = u32::try_from(arenas.len())
            .map_err(|_| invalid("invocation slot count exceeds u32"))?;
        let first = arenas
            .first()
            .ok_or_else(|| invalid("invocation slot pool has no physical arena"))?;
        let per_slot_bytes = first.maximum_bytes();
        if per_slot_bytes == 0 {
            return Err(invalid(
                "invocation slot pool requires non-zero per-slot bytes",
            ));
        }
        let domain = first.domain();
        let kind = first.backing_kind();
        let maximum_bytes = per_slot_bytes
            .checked_mul(u64::from(slot_count))
            .ok_or_else(|| invalid("invocation slot pool byte accounting overflow"))?;
        for arena in &arenas {
            if arena.plan() != plan
                || arena.domain() != domain
                || arena.backing_kind() != kind
                || arena.workspace_domain() != &workspace_domain
                || arena.maximum_bytes() != per_slot_bytes
            {
                return Err(invalid(
                    "invocation slot pool arenas do not share the exact plan, physical geometry, and workspace",
                ));
            }
        }
        let slot_count = usize::try_from(slot_count)
            .map_err(|_| invalid("invocation slot count exceeds usize"))?;
        let id = InvocationSlotPoolId {
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
            .map_err(|_| invalid("invocation slot pool instance nonce exhausted"))?;
        Ok(Self {
            inner: Arc::new(InvocationSlotPoolInner {
                id,
                instance_nonce,
                workspace_domain,
                maximum_bytes,
                arenas: arenas.into_iter().map(Mutex::new).collect(),
                state: Mutex::new(InvocationSlotPoolState {
                    lifecycle: InvocationSlotPoolLifecycle::Accepting,
                    owner_alive: true,
                    next_nonce: 0,
                    admission_cursor: 0,
                    drain_cursor: 0,
                    slots: vec![InvocationSlotState::Vacant { generation: 0 }; slot_count],
                }),
            }),
        })
    }

    pub(crate) fn id(&self) -> InvocationSlotPoolId {
        self.inner.id
    }

    pub(crate) fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        &self.inner.workspace_domain
    }

    pub(crate) fn maximum_bytes(&self) -> u64 {
        self.inner.maximum_bytes
    }

    pub(crate) fn handle(&self) -> InvocationSlotPoolHandle<A> {
        InvocationSlotPoolHandle {
            id: self.inner.id,
            workspace_domain: self.inner.workspace_domain.clone(),
            maximum_bytes: self.inner.maximum_bytes,
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub(crate) fn backing(&self) -> Arc<dyn InvocationWorkspaceBackingV2> {
        Arc::new(self.handle())
    }

    pub(crate) fn lease(&self) -> Result<InvocationSlotLease<A>> {
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
            .map(|state| state.lifecycle == InvocationSlotPoolLifecycle::Drained)
            .unwrap_or(false)
    }
}

impl<A: InvocationSlotArena> Drop for InvocationSlotPoolOwner<A> {
    fn drop(&mut self) {
        let should_drain = {
            let mut state = lifecycle_state(self.inner.as_ref());
            state.owner_alive = false;
            if state.lifecycle == InvocationSlotPoolLifecycle::Accepting {
                state.lifecycle = InvocationSlotPoolLifecycle::Draining;
            }
            begin_drain_if_idle(&mut state)
        };
        if should_drain {
            let _ = finish_drain(self.inner.as_ref());
        }
    }
}

impl<A: InvocationSlotArena> InvocationSlotPoolHandle<A> {
    pub(crate) const fn id(&self) -> InvocationSlotPoolId {
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
            invalid("invocation slot pool refers to a retired physical generation")
        })?;
        let state = inner
            .state
            .lock()
            .map_err(|_| invalid("invocation slot pool state is poisoned"))?;
        if !state.owner_alive || state.lifecycle != InvocationSlotPoolLifecycle::Accepting {
            return Err(invalid("invocation slot pool is not accepting new leases"));
        }
        Ok(())
    }

    pub(crate) fn lease(&self) -> Result<InvocationSlotLease<A>> {
        self.validate_live()?;
        let inner = self
            .inner
            .upgrade()
            .ok_or_else(|| invalid("invocation slot pool retired during admission"))?;
        lease_from_inner(inner)
    }

    pub(crate) fn backing(&self) -> Arc<dyn InvocationWorkspaceBackingV2> {
        Arc::new(self.clone())
    }
}

impl<A: InvocationSlotArena> InvocationWorkspaceBackingV2 for InvocationSlotPoolHandle<A> {
    fn identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
        self.id.backing_identity()
    }

    fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        &self.workspace_domain
    }

    fn validate_live(&self) -> Result<()> {
        InvocationSlotPoolHandle::validate_live(self)
    }

    fn lease(&self) -> Result<Box<dyn InvocationWorkspacePhysicalLeaseV2>> {
        Ok(Box::new(InvocationSlotPoolHandle::lease(self)?))
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
                "invocation slot completion has a non-typed physical kind",
            ));
        };
        if *backing != self.id.backing_identity() {
            return Err(invalid(
                "invocation slot completion belongs to another backing",
            ));
        }
        let inner = self
            .inner
            .upgrade()
            .ok_or_else(|| invalid("invocation slot completion refers to a retired generation"))?;
        authenticate_receipt(inner.as_ref(), *authentication)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationSlotLeasePhase {
    Leased,
    Completing { receipt: [u8; 32] },
    Released,
}

pub(crate) struct InvocationSlotLease<A: InvocationSlotArena> {
    inner: Arc<InvocationSlotPoolInner<A>>,
    slot: InvocationSlotRef,
    phase: InvocationSlotLeasePhase,
}

pub(crate) type InvocationTensorLease = InvocationSlotLease<InvocationTensorArena>;
pub(crate) type InvocationStaticAttentionLease =
    InvocationSlotLease<InvocationStaticAttentionArena>;

impl<A: InvocationSlotArena> std::fmt::Debug for InvocationSlotLease<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationSlotLease")
            .field("slot", &self.slot)
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

impl<A: InvocationSlotArena> InvocationSlotLease<A> {
    pub(crate) const fn slot(&self) -> InvocationSlotRef {
        self.slot
    }

    pub(crate) const fn domain(&self) -> StateDomainId {
        self.slot.pool.domain
    }

    pub(crate) fn arena(&self) -> Result<std::sync::MutexGuard<'_, A>> {
        self.require_leased()?;
        lock_arena(self.inner.as_ref(), self.slot.slot)
    }

    pub(crate) fn arena_mut(&mut self) -> Result<std::sync::MutexGuard<'_, A>> {
        self.require_leased()?;
        lock_arena(self.inner.as_ref(), self.slot.slot)
    }

    fn require_leased(&self) -> Result<()> {
        if self.phase != InvocationSlotLeasePhase::Leased {
            return Err(invalid("invocation slot lease is no longer model-facing"));
        }
        Ok(())
    }

    fn begin_completion(&mut self) -> Result<InvocationWorkspacePhysicalCompletionV2> {
        self.require_leased()?;
        {
            let mut arena = lock_arena(self.inner.as_ref(), self.slot.slot)?;
            arena.prepare_completion()?;
        }
        let receipt = transition_to_completing(self.inner.as_ref(), self.slot)?;
        self.phase = InvocationSlotLeasePhase::Completing { receipt };
        Ok(InvocationWorkspacePhysicalCompletionV2::Typed {
            backing: self.inner.id.backing_identity(),
            authentication: receipt,
        })
    }

    fn abort_once(&mut self) {
        if self.phase == InvocationSlotLeasePhase::Released {
            return;
        }
        release_slot(self.inner.as_ref(), self.slot);
        self.phase = InvocationSlotLeasePhase::Released;
    }
}

impl InvocationSlotLease<InvocationTensorArena> {
    pub(crate) fn apply_intent(
        &mut self,
        intent: &DomainStepIntent,
        update: InvocationTensorUpdateV2,
    ) -> Result<()> {
        let mut arena = self.arena_mut()?;
        arena.apply_intent(intent, update)
    }

    pub(crate) fn read_snapshot(&self) -> Result<InvocationTensorSnapshot> {
        self.arena()?.read_snapshot()
    }

    pub(crate) fn read_chronological_segments(
        &self,
    ) -> Result<Vec<InvocationTensorChronologicalSegment>> {
        self.arena()?.read_chronological_segments()
    }

    /// Reuse this invocation-exclusive backing for a new nested logical
    /// sequence without releasing its authenticated pool slot.
    pub(crate) fn reset_invocation(&mut self) -> Result<()> {
        self.arena_mut()?.reset_for_reuse()
    }

    pub(crate) fn with_ring_depthwise_conv<T>(
        &mut self,
        intent: &DomainStepIntent,
        run: impl FnOnce(&mut InvocationRingDepthwiseConvTransaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut arena = self.arena_mut()?;
        let mut transaction = arena.begin_ring_depthwise_conv(intent)?;
        let output = run(&mut transaction)?;
        transaction.commit()?;
        Ok(output)
    }
}

impl InvocationSlotLease<InvocationStaticAttentionArena> {
    pub(crate) fn begin_install(&mut self, intent: &DomainStepIntent) -> Result<()> {
        self.arena_mut()?.begin_install(intent)
    }

    pub(crate) fn install_layer(&mut self, layer: StaticAttentionLayerValue) -> Result<()> {
        self.arena_mut()?.install_layer(layer)
    }

    pub(crate) fn commit_install(&mut self) -> Result<()> {
        self.arena_mut()?.commit_install()
    }

    pub(crate) fn install(
        &mut self,
        intent: &DomainStepIntent,
        layers: Vec<StaticAttentionLayerValue>,
    ) -> Result<()> {
        self.arena_mut()?.install_from_intent(intent, layers)
    }

    pub(crate) fn attend(
        &self,
        model_layer: u32,
        queries: &Tensor,
        rows: &[StaticAttentionRaggedRow],
        softmax_scale: f32,
    ) -> Result<Tensor> {
        self.arena()?
            .attend(model_layer, queries, rows, softmax_scale)
    }

    pub(crate) fn metadata(&self) -> Result<Option<StaticAttentionMetadata>> {
        self.arena()?.metadata()
    }
}

impl<A: InvocationSlotArena> InvocationWorkspacePhysicalLeaseV2 for InvocationSlotLease<A> {
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

impl<A: InvocationSlotArena> Drop for InvocationSlotLease<A> {
    fn drop(&mut self) {
        self.abort_once();
    }
}

fn lease_from_inner<A: InvocationSlotArena>(
    inner: Arc<InvocationSlotPoolInner<A>>,
) -> Result<InvocationSlotLease<A>> {
    let mut attempted = vec![false; inner.arenas.len()];
    let mut first_reset_error = None;
    loop {
        let Some(slot) = reserve_next_vacant(inner.as_ref(), &attempted)? else {
            return match first_reset_error {
                Some(error) => Err(error),
                None => Err(Error::Backpressure(
                    "invocation slot pool has no free slot".to_string(),
                )),
            };
        };
        attempted[slot.slot as usize] = true;
        let mut reservation = PreparingSlotGuard::new(inner.clone(), slot);
        match reset_arena_for_reuse(inner.as_ref(), slot.slot) {
            Ok(()) => {}
            Err(error) => {
                first_reset_error.get_or_insert(error);
                drop(reservation);
                continue;
            }
        }
        if !transition_to_leased(inner.as_ref(), slot) {
            return Err(invalid("invocation slot lost its preparing generation"));
        }
        reservation.disarm();
        return Ok(InvocationSlotLease {
            inner,
            slot,
            phase: InvocationSlotLeasePhase::Leased,
        });
    }
}

struct PreparingSlotGuard<A: InvocationSlotArena> {
    inner: Arc<InvocationSlotPoolInner<A>>,
    slot: InvocationSlotRef,
    armed: bool,
}

impl<A: InvocationSlotArena> PreparingSlotGuard<A> {
    fn new(inner: Arc<InvocationSlotPoolInner<A>>, slot: InvocationSlotRef) -> Self {
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

impl<A: InvocationSlotArena> Drop for PreparingSlotGuard<A> {
    fn drop(&mut self) {
        if self.armed {
            release_slot(self.inner.as_ref(), self.slot);
        }
    }
}

fn begin_lease<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
) -> Result<InvocationSlotRef> {
    let attempted = vec![false; inner.arenas.len()];
    reserve_next_vacant(inner, &attempted)?
        .ok_or_else(|| Error::Backpressure("invocation slot pool has no free slot".to_string()))
}

fn reserve_next_vacant<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    attempted: &[bool],
) -> Result<Option<InvocationSlotRef>> {
    let mut state = inner
        .state
        .lock()
        .map_err(|_| invalid("invocation slot pool state is poisoned"))?;
    if !state.owner_alive || state.lifecycle != InvocationSlotPoolLifecycle::Accepting {
        return Err(invalid("invocation slot pool is closed for admission"));
    }
    if attempted.len() != state.slots.len() {
        return Err(invalid(
            "invocation slot admission mask has a mismatched capacity",
        ));
    }
    let slot_count = state.slots.len();
    let slot_index = (0..slot_count)
        .map(|offset| (state.admission_cursor + offset) % slot_count)
        .find(|index| !attempted[*index] && state.slots[*index].is_vacant());
    let Some(slot_index) = slot_index else {
        return Ok(None);
    };
    let generation = state.slots[slot_index]
        .generation()
        .checked_add(1)
        .ok_or_else(|| invalid("invocation slot generation exhausted"))?;
    state.next_nonce = state
        .next_nonce
        .checked_add(1)
        .ok_or_else(|| invalid("invocation slot lease nonce exhausted"))?;
    let nonce = state.next_nonce;
    state.slots[slot_index] = InvocationSlotState::Preparing { generation, nonce };
    state.admission_cursor = (slot_index + 1) % slot_count;
    Ok(Some(InvocationSlotRef {
        pool: inner.id,
        slot: u32::try_from(slot_index)
            .map_err(|_| invalid("invocation slot index exceeds u32"))?,
        slot_generation: generation,
        nonce,
    }))
}

fn transition_to_leased<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    slot: InvocationSlotRef,
) -> bool {
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
        != (InvocationSlotState::Preparing {
            generation: slot.slot_generation,
            nonce: slot.nonce,
        })
    {
        return false;
    }
    *current = InvocationSlotState::Leased {
        generation: slot.slot_generation,
        nonce: slot.nonce,
    };
    true
}

fn transition_to_completing<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    slot: InvocationSlotRef,
) -> Result<[u8; 32]> {
    if slot.pool != inner.id {
        return Err(invalid("invocation slot lease belongs to another pool"));
    }
    let receipt = receipt_for(inner, slot)?;
    let mut state = inner
        .state
        .lock()
        .map_err(|_| invalid("invocation slot pool state is poisoned"))?;
    let current = state
        .slots
        .get_mut(slot.slot as usize)
        .ok_or_else(|| invalid("invocation slot index is out of bounds"))?;
    if *current
        != (InvocationSlotState::Leased {
            generation: slot.slot_generation,
            nonce: slot.nonce,
        })
    {
        return Err(invalid(
            "invocation slot lease generation is no longer active",
        ));
    }
    *current = InvocationSlotState::Completing {
        generation: slot.slot_generation,
        nonce: slot.nonce,
        receipt,
    };
    Ok(receipt)
}

fn authenticate_receipt<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    receipt: [u8; 32],
) -> Result<()> {
    let mut state = inner
        .state
        .lock()
        .map_err(|_| invalid("invocation slot pool state is poisoned"))?;
    let (slot_index, generation) = state
        .slots
        .iter()
        .enumerate()
        .find_map(|(index, slot)| match *slot {
            InvocationSlotState::Completing {
                generation,
                receipt: candidate,
                ..
            } if candidate == receipt => Some((index, generation)),
            _ => None,
        })
        .ok_or_else(|| invalid("invocation slot completion receipt is stale or foreign"))?;
    state.slots[slot_index] = InvocationSlotState::Vacant { generation };
    Ok(())
}

fn release_slot<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    slot: InvocationSlotRef,
) {
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
            InvocationSlotState::Preparing { generation, nonce }
                | InvocationSlotState::Leased { generation, nonce }
                | InvocationSlotState::Completing {
                    generation,
                    nonce,
                    ..
                } if generation == slot.slot_generation && nonce == slot.nonce
        );
        if matches_generation {
            *current = InvocationSlotState::Vacant {
                generation: slot.slot_generation,
            };
        }
        !state.owner_alive && begin_drain_if_idle(&mut state)
    };
    if should_drain {
        let _ = finish_drain(inner);
    }
}

fn begin_drain_if_idle(state: &mut InvocationSlotPoolState) -> bool {
    if state.lifecycle == InvocationSlotPoolLifecycle::Draining
        && state.slots.iter().all(|slot| slot.is_vacant())
    {
        state.lifecycle = InvocationSlotPoolLifecycle::DrainInFlight;
        true
    } else {
        false
    }
}

fn close_and_drain<A: InvocationSlotArena>(inner: &InvocationSlotPoolInner<A>) -> Result<()> {
    let should_drain = {
        let mut state = lifecycle_state(inner);
        match state.lifecycle {
            InvocationSlotPoolLifecycle::Drained => return Ok(()),
            InvocationSlotPoolLifecycle::DrainInFlight => {
                return Err(Error::Backpressure(
                    "invocation slot pool drain is already in flight".to_string(),
                ));
            }
            InvocationSlotPoolLifecycle::Accepting => {
                state.lifecycle = InvocationSlotPoolLifecycle::Draining;
            }
            InvocationSlotPoolLifecycle::Draining => {}
        }
        if !begin_drain_if_idle(&mut state) {
            return Err(Error::Backpressure(
                "invocation slot pool still has active or completing leases".to_string(),
            ));
        }
        true
    };
    debug_assert!(should_drain);
    finish_drain(inner)
}

fn finish_drain<A: InvocationSlotArena>(inner: &InvocationSlotPoolInner<A>) -> Result<()> {
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
            state.lifecycle = InvocationSlotPoolLifecycle::Drained;
            return Ok(());
        };
        let result = u32::try_from(index)
            .map_err(|_| invalid("invocation slot drain index exceeds u32"))
            .and_then(|slot| reset_arena_for_reuse(inner, slot));
        let mut state = lifecycle_state(inner);
        match result {
            Ok(()) => {
                if state.drain_cursor == index {
                    state.drain_cursor += 1;
                }
            }
            Err(error) => {
                state.lifecycle = InvocationSlotPoolLifecycle::Draining;
                return Err(error);
            }
        }
    }
}

fn lock_arena<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    slot: u32,
) -> Result<std::sync::MutexGuard<'_, A>> {
    inner
        .arenas
        .get(slot as usize)
        .ok_or_else(|| invalid("invocation slot index is out of bounds"))?
        .lock()
        .map_err(|_| invalid("invocation slot arena lock is poisoned"))
}

fn reset_arena_for_reuse<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    slot: u32,
) -> Result<()> {
    let mutex = inner
        .arenas
        .get(slot as usize)
        .ok_or_else(|| invalid("invocation slot index is out of bounds"))?;
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

fn lifecycle_state<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
) -> std::sync::MutexGuard<'_, InvocationSlotPoolState> {
    match inner.state.lock() {
        Ok(state) => state,
        Err(poisoned) => {
            let state = poisoned.into_inner();
            inner.state.clear_poison();
            state
        }
    }
}

fn receipt_for<A: InvocationSlotArena>(
    inner: &InvocationSlotPoolInner<A>,
    slot: InvocationSlotRef,
) -> Result<[u8; 32]> {
    let encoded = serde_json::to_vec(&slot)
        .map_err(|error| invalid(format!("failed to encode invocation slot receipt: {error}")))?;
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

    use candle_core::Tensor;

    use super::*;
    use crate::backends::state::{
        negotiate_state_plan, InvocationTensorComponentValue, StateBackendPlanRequest,
    };
    use crate::kv::v2::{
        test_contract, BoundedShape, CheckpointPolicy, ComponentShapeInstantiation,
        InferenceStateContract, InvocationStateCapacity, PlacementPolicy, PrefixPolicy,
        RingStateDomainSpec, ShapeAxis, ShapeDimension, ShapeDimensionValue, ShapeExtent,
        StateClock, StateComponentId, StateDType, StateDomainHeader, StateDomainSpec, StateGroupId,
        StateGroupSpec, StateScope, StateUpdateKind, StaticAttentionDomainSpec,
        StaticAttentionLayerSpec, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
        WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
    };

    #[derive(Debug)]
    struct TestSlotArena {
        plan: Arc<ResolvedStatePlan>,
        workspace: InvocationWorkspaceDomain,
        resets: Arc<AtomicUsize>,
        fail_reset: Arc<AtomicBool>,
        fail_completion: Arc<AtomicBool>,
        maximum_bytes: u64,
    }

    impl slot_arena_sealed::Sealed for TestSlotArena {}

    impl InvocationSlotArena for TestSlotArena {
        fn plan(&self) -> &ResolvedStatePlan {
            &self.plan
        }

        fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
            &self.workspace
        }

        fn domain(&self) -> StateDomainId {
            self.workspace.id()
        }

        fn backing_kind(&self) -> InvocationStateBackingKindV2 {
            InvocationStateBackingKindV2::Tensor
        }

        fn maximum_bytes(&self) -> u64 {
            self.maximum_bytes
        }

        fn reset_for_reuse(&mut self) -> Result<()> {
            self.resets.fetch_add(1, AtomicOrdering::AcqRel);
            if self.fail_reset.load(AtomicOrdering::Acquire) {
                return Err(Error::InferenceError(
                    "injected invocation slot reset failure".to_string(),
                ));
            }
            Ok(())
        }

        fn prepare_completion(&mut self) -> Result<()> {
            if self.fail_completion.load(AtomicOrdering::Acquire) {
                return Err(Error::InferenceError(
                    "injected invocation slot completion fence failure".to_string(),
                ));
            }
            Ok(())
        }
    }

    fn test_slot_arena(
        plan: Arc<ResolvedStatePlan>,
        workspace: InvocationWorkspaceDomain,
        resets: Arc<AtomicUsize>,
        maximum_bytes: u64,
    ) -> TestSlotArena {
        TestSlotArena {
            plan,
            workspace,
            resets,
            fail_reset: Arc::new(AtomicBool::new(false)),
            fail_completion: Arc::new(AtomicBool::new(false)),
            maximum_bytes,
        }
    }

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

    fn static_attention_state(id: u32, max_memory_tokens: u64) -> StateDomainSpec {
        StateDomainSpec::StaticAttention(StaticAttentionDomainSpec {
            header: StateDomainHeader {
                id: StateDomainId::new(id),
                scope: StateScope::Invocation,
                clock: StateClock::EncoderTokens,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::None,
            },
            layers: vec![StaticAttentionLayerSpec {
                model_layer: 0,
                query_heads: 4,
                kv_heads: 2,
                key_head_dim: 2,
                value_head_dim: 2,
                key_encoding: crate::kv::v2::KeyEncoding::Raw,
            }],
            max_memory_tokens,
            accepted_dtypes: vec![StateDType::F32],
        })
    }

    fn shortconv_ring_state(id: u32) -> StateDomainSpec {
        StateDomainSpec::Ring(RingStateDomainSpec {
            header: StateDomainHeader {
                id: StateDomainId::new(id),
                scope: StateScope::Invocation,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::None,
            },
            components_per_step: vec![TensorComponentSpec {
                id: StateComponentId::new(1),
                role: TensorRole::ConvolutionState,
                shape: BoundedShape {
                    dimensions: vec![
                        ShapeDimension {
                            axis: ShapeAxis::Batch,
                            extent: ShapeExtent::Fixed { value: 1 },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed { value: 2 },
                        },
                    ],
                },
                accepted_dtypes: vec![StateDType::F32],
            }],
            capacity_steps: 3,
        })
    }

    fn static_attention_values(memory_tokens: usize) -> Vec<StaticAttentionLayerValue> {
        let elements = memory_tokens * 2 * 2;
        let keys = (0..elements)
            .map(|index| (index + 1) as f32 / elements as f32)
            .collect::<Vec<_>>();
        let values = (0..elements)
            .map(|index| (index + 1) as f32)
            .collect::<Vec<_>>();
        vec![StaticAttentionLayerValue {
            model_layer: 0,
            keys: Tensor::from_vec(keys, (memory_tokens, 2, 2), &Device::Cpu).unwrap(),
            values: Tensor::from_vec(values, (memory_tokens, 2, 2), &Device::Cpu).unwrap(),
        }]
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

    fn shortconv_owner() -> InvocationTensorPoolOwner {
        let contract = contract(shortconv_ring_state(1));
        let (plan, workspace) = plan_and_workspace(&contract, StateDomainId::new(1));
        InvocationTensorPoolOwner::new(
            &contract,
            plan,
            workspace,
            Device::Cpu,
            ModelInstanceId::new(8),
            1,
            1,
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
    fn lease_scopes_the_complete_depthwise_transaction_under_one_arena_lock() {
        let owner = shortconv_owner();
        let mut lease = owner.lease().unwrap();
        let intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 2,
            update: StateUpdateKind::RingAdvance {
                steps: 2,
                components_per_step: vec![ComponentShapeInstantiation {
                    component: StateComponentId::new(1),
                    dimensions: vec![
                        ShapeDimensionValue {
                            axis: ShapeAxis::Batch,
                            units: 1,
                        },
                        ShapeDimensionValue {
                            axis: ShapeAxis::Hidden,
                            units: 2,
                        },
                    ],
                }],
            },
        };
        let input = Tensor::from_slice(&[1.0f32, 2.0, 0.5, 1.0], (1, 2, 2), &Device::Cpu).unwrap();
        let weight =
            Tensor::from_slice(&[0.1f32, 0.2, 0.3, -0.5, 0.25, 0.75], (2, 3), &Device::Cpu)
                .unwrap();
        let output = lease
            .with_ring_depthwise_conv(&intent, |transaction| {
                transaction.apply(StateComponentId::new(1), &input, &weight)
            })
            .unwrap();
        assert_eq!(output.dims(), [1, 2, 2]);
        assert_eq!(lease.arena().unwrap().absolute_cursor(), 2);

        let rejected: Result<()> = lease.with_ring_depthwise_conv(
            &DomainStepIntent {
                domain: StateDomainId::new(1),
                expected_cursor: 2,
                target_cursor: 3,
                update: StateUpdateKind::RingAdvance {
                    steps: 1,
                    components_per_step: vec![ComponentShapeInstantiation {
                        component: StateComponentId::new(1),
                        dimensions: vec![
                            ShapeDimensionValue {
                                axis: ShapeAxis::Batch,
                                units: 1,
                            },
                            ShapeDimensionValue {
                                axis: ShapeAxis::Hidden,
                                units: 2,
                            },
                        ],
                    }],
                },
            },
            |_transaction| Err(Error::InferenceError("injected model failure".into())),
        );
        assert!(rejected.is_err());
        assert_eq!(lease.arena().unwrap().absolute_cursor(), 2);
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
    fn active_tensor_lease_resets_between_nested_logical_sequences() {
        let owner = owner(1, 12);
        let mut lease = owner.lease().unwrap();
        lease
            .apply_intent(&replace_intent(0, 1, 2), replace_update(&[1.0, 2.0]))
            .unwrap();
        assert_eq!(lease.arena().unwrap().absolute_cursor(), 1);

        lease.reset_invocation().unwrap();
        assert_eq!(lease.arena().unwrap().absolute_cursor(), 0);
        assert!(lease.read_snapshot().is_err());
        lease
            .apply_intent(&replace_intent(0, 1, 1), replace_update(&[9.0]))
            .unwrap();
        assert_eq!(
            lease.read_snapshot().unwrap().components[0]
                .tensor
                .to_vec1::<f32>()
                .unwrap(),
            vec![9.0]
        );
    }

    #[test]
    fn static_attention_pool_authenticates_install_attends_and_reuses_exact_slots() {
        let contract = contract(static_attention_state(1, 2));
        let (plan, workspace) = plan_and_workspace(&contract, StateDomainId::new(1));
        let per_slot_bytes = plan.non_paged[0].maximum_bytes();
        let owner = InvocationStaticAttentionPoolOwner::new(
            &contract,
            plan,
            workspace,
            Device::Cpu,
            ModelInstanceId::new(15),
            2,
            27,
        )
        .unwrap();
        assert_eq!(owner.maximum_bytes(), per_slot_bytes * 2);
        let backing = owner.backing();
        let mut physical = backing.lease().unwrap();
        let mismatched = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 1,
            update: StateUpdateKind::StaticInitialize {
                source_identity: [7; 32],
                components: vec![],
            },
        };
        let intent = DomainStepIntent {
            target_cursor: 2,
            ..mismatched.clone()
        };
        {
            let lease = physical
                .as_any_mut()
                .downcast_mut::<InvocationStaticAttentionLease>()
                .unwrap();
            assert!(lease
                .install(&mismatched, static_attention_values(2))
                .is_err());
            assert_eq!(lease.metadata().unwrap(), None);
            lease.begin_install(&intent).unwrap();
            for layer in static_attention_values(2) {
                lease.install_layer(layer).unwrap();
            }
        }
        assert!(physical.complete().is_err());
        let lease = physical
            .as_any_mut()
            .downcast_mut::<InvocationStaticAttentionLease>()
            .unwrap();
        lease.commit_install().unwrap();
        assert_eq!(
            lease.metadata().unwrap(),
            Some(StaticAttentionMetadata {
                source_identity: [7; 32],
                absolute_cursor: 2,
            })
        );
        let queries = Tensor::from_slice(
            &[1.0_f32, 0.0, 0.5, 0.5, 0.0, 1.0, -0.5, 0.5],
            (1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let output = lease
            .attend(
                0,
                &queries,
                &[StaticAttentionRaggedRow {
                    query_start: 0,
                    query_len: 1,
                }],
                1.0,
            )
            .unwrap();
        assert_eq!(output.dims(), &[1, 4, 2]);

        let completion = physical.complete().unwrap();
        backing.authenticate_completion(&completion).unwrap();
        assert!(backing.authenticate_completion(&completion).is_err());
        drop(physical);
        let reused = owner.lease().unwrap();
        assert_eq!(reused.metadata().unwrap(), None);
        drop(reused);
        owner.close_and_drain().unwrap();
    }

    #[test]
    fn generic_slot_pool_reuses_lifecycle_receipts_and_arena_reset_contract() {
        let contract = contract(tensor_state(1, 4));
        let (plan, workspace) = plan_and_workspace(&contract, StateDomainId::new(1));
        let resets = Arc::new(AtomicUsize::new(0));
        let owner = InvocationSlotPoolOwner::from_arenas(
            plan.as_ref(),
            workspace.clone(),
            ModelInstanceId::new(10),
            21,
            vec![
                test_slot_arena(plan.clone(), workspace.clone(), resets.clone(), 64),
                test_slot_arena(plan.clone(), workspace, resets.clone(), 64),
            ],
        )
        .unwrap();
        assert_eq!(owner.maximum_bytes(), 128);

        let backing = owner.backing();
        let mut first = backing.lease().unwrap();
        assert!(first
            .as_any()
            .downcast_ref::<InvocationSlotLease<TestSlotArena>>()
            .is_some());
        let completion = first.complete().unwrap();
        backing.authenticate_completion(&completion).unwrap();
        assert_eq!(resets.load(AtomicOrdering::Acquire), 1);

        let second = owner.lease().unwrap();
        assert!(second.slot().slot_generation >= 1);
        drop(second);
        owner.close_and_drain().unwrap();
        assert_eq!(resets.load(AtomicOrdering::Acquire), 4);
    }

    #[test]
    fn generic_slot_pool_rejects_foreign_plan_and_zero_byte_arenas() {
        let base_contract = contract(tensor_state(1, 4));
        let (plan, workspace) = plan_and_workspace(&base_contract, StateDomainId::new(1));
        let foreign_contract = contract(tensor_state(1, 8));
        let (foreign_plan, _) = plan_and_workspace(&foreign_contract, StateDomainId::new(1));
        assert_ne!(plan.as_ref(), foreign_plan.as_ref());
        let resets = Arc::new(AtomicUsize::new(0));

        assert!(InvocationSlotPoolOwner::from_arenas(
            plan.as_ref(),
            workspace.clone(),
            ModelInstanceId::new(11),
            22,
            vec![
                test_slot_arena(plan.clone(), workspace.clone(), resets.clone(), 64),
                test_slot_arena(foreign_plan, workspace.clone(), resets.clone(), 64),
            ],
        )
        .is_err());
        assert!(InvocationSlotPoolOwner::from_arenas(
            plan.as_ref(),
            workspace.clone(),
            ModelInstanceId::new(11),
            23,
            vec![test_slot_arena(plan.clone(), workspace, resets, 0)],
        )
        .is_err());
    }

    #[test]
    fn completion_fence_failure_issues_no_receipt_and_abort_recovers_slot() {
        let base_contract = contract(tensor_state(1, 4));
        let (plan, workspace) = plan_and_workspace(&base_contract, StateDomainId::new(1));
        let resets = Arc::new(AtomicUsize::new(0));
        let fail_completion = Arc::new(AtomicBool::new(true));
        let mut arena = test_slot_arena(plan.clone(), workspace.clone(), resets.clone(), 64);
        arena.fail_completion = fail_completion.clone();
        let owner = InvocationSlotPoolOwner::from_arenas(
            plan.as_ref(),
            workspace,
            ModelInstanceId::new(12),
            24,
            vec![arena],
        )
        .unwrap();
        let backing = owner.backing();
        let mut lease = backing.lease().unwrap();

        assert!(lease.complete().is_err());
        let forged = InvocationWorkspacePhysicalCompletionV2::Typed {
            backing: backing.identity(),
            authentication: [9; 32],
        };
        assert!(backing.authenticate_completion(&forged).is_err());
        lease.abort();
        fail_completion.store(false, AtomicOrdering::Release);

        let mut recovered = backing.lease().unwrap();
        let completion = recovered.complete().unwrap();
        backing.authenticate_completion(&completion).unwrap();
        assert_eq!(resets.load(AtomicOrdering::Acquire), 2);
        drop((lease, recovered));
        owner.close_and_drain().unwrap();
    }

    #[test]
    fn reset_failure_does_not_starve_later_vacant_slots_or_retry_forever() {
        let base_contract = contract(tensor_state(1, 4));
        let (plan, workspace) = plan_and_workspace(&base_contract, StateDomainId::new(1));
        let resets = Arc::new(AtomicUsize::new(0));
        let fail_first = Arc::new(AtomicBool::new(true));
        let mut first = test_slot_arena(plan.clone(), workspace.clone(), resets.clone(), 64);
        first.fail_reset = fail_first.clone();
        let second = test_slot_arena(plan.clone(), workspace.clone(), resets.clone(), 64);
        let owner = InvocationSlotPoolOwner::from_arenas(
            plan.as_ref(),
            workspace.clone(),
            ModelInstanceId::new(13),
            25,
            vec![first, second],
        )
        .unwrap();

        let mut lease = owner.lease().unwrap();
        assert_eq!(lease.slot().slot, 1);
        {
            let state = owner.inner.state.lock().unwrap();
            assert_eq!(state.slots[0].generation(), 1);
            assert_eq!(state.slots[1].generation(), 1);
        }
        let completion = lease.complete().unwrap();
        owner
            .backing()
            .authenticate_completion(&completion)
            .unwrap();
        drop(lease);
        fail_first.store(false, AtomicOrdering::Release);
        owner.close_and_drain().unwrap();

        let fail_a = Arc::new(AtomicBool::new(true));
        let fail_b = Arc::new(AtomicBool::new(true));
        let mut first = test_slot_arena(plan.clone(), workspace.clone(), resets.clone(), 64);
        first.fail_reset = fail_a.clone();
        let mut second = test_slot_arena(plan.clone(), workspace.clone(), resets, 64);
        second.fail_reset = fail_b.clone();
        let all_failing = InvocationSlotPoolOwner::from_arenas(
            plan.as_ref(),
            workspace,
            ModelInstanceId::new(14),
            26,
            vec![first, second],
        )
        .unwrap();
        let error = all_failing.lease().unwrap_err();
        assert!(matches!(
            error,
            Error::InferenceError(message)
                if message.contains("injected invocation slot reset failure")
        ));
        {
            let state = all_failing.inner.state.lock().unwrap();
            assert_eq!(state.slots[0].generation(), 1);
            assert_eq!(state.slots[1].generation(), 1);
            assert!(state.slots.iter().all(|slot| slot.is_vacant()));
        }
        fail_a.store(false, AtomicOrdering::Release);
        fail_b.store(false, AtomicOrdering::Release);
        all_failing.close_and_drain().unwrap();
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
