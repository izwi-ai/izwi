use std::fmt;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::backends::BackendKind;
use crate::engine::StageId;
use crate::error::{Error, Result};

use super::contract::{PlacementPolicy, StateDomainId};
use super::descriptor::InvocationWorkspaceDomain;

const SCRATCH_PLAN_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.scratch-workspace-plan.v2\0";

/// Exact capability/stage authority allowed to acquire one scratch pool.
/// Keeping this identity in the plan prevents equal byte shapes from silently
/// becoming shareable across model generations or execution graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct ScratchWorkspaceOwner {
    pub(crate) capability_runtime: [u8; 32],
    pub(crate) stage_graph: [u8; 32],
    pub(crate) stage: StageId,
}

impl ScratchWorkspaceOwner {
    fn validate(self) -> Result<()> {
        if self.capability_runtime.iter().all(|byte| *byte == 0)
            || self.stage_graph.iter().all(|byte| *byte == 0)
        {
            return Err(invalid(
                "scratch workspace owner has an incomplete identity",
            ));
        }
        Ok(())
    }
}

/// Physical placement selected for scratch bytes. Host offload remains an
/// explicit permission rather than being collapsed into the primary target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct ResolvedScratchPlacement {
    pub(crate) primary: ScratchMemoryDomain,
    pub(crate) host_offload_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ScratchMemoryDomain {
    Host,
    BackendLocal,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub(crate) struct ScratchWorkspacePlanId([u8; 32]);

impl fmt::Debug for ScratchWorkspacePlanId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ScratchWorkspacePlanId(")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str(")")
    }
}

/// Backend-resolved immutable scratch geometry for one invocation stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ResolvedScratchWorkspace {
    pub(crate) id: ScratchWorkspacePlanId,
    pub(crate) owner: ScratchWorkspaceOwner,
    pub(crate) domain: StateDomainId,
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) placement: ResolvedScratchPlacement,
    pub(crate) alignment_bytes: u64,
    /// Formula result at every declared dimension's sealed maximum.
    pub(crate) maximum_requested_bytes: u64,
    /// Per-slot backing size after the scratch domain's alignment is applied.
    pub(crate) maximum_allocated_bytes: u64,
    pub(crate) zero_on_release: bool,
}

impl ResolvedScratchWorkspace {
    pub(crate) fn resolve(
        owner: ScratchWorkspaceOwner,
        backend: BackendKind,
        device_ordinal: Option<u32>,
        domain: &InvocationWorkspaceDomain,
    ) -> Result<Self> {
        owner.validate()?;
        validate_backend_device(backend, device_ordinal)?;
        let InvocationWorkspaceDomain::Scratch {
            id,
            placement,
            alignment_bytes,
            zero_on_release,
            formula,
        } = domain
        else {
            return Err(invalid(
                "scratch workspace resolution requires a scratch domain",
            ));
        };
        if id.get() == 0 || *alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(invalid(
                "scratch workspace requires a non-zero id and power-of-two alignment",
            ));
        }
        let maximum_requested_bytes = formula.maximum_bytes()?;
        if maximum_requested_bytes == 0 {
            return Err(invalid("scratch workspace maximum cannot be zero"));
        }
        let maximum_allocated_bytes = align_up(maximum_requested_bytes, *alignment_bytes)?;
        let placement = resolve_placement(backend, *placement);
        let mut resolved = Self {
            id: ScratchWorkspacePlanId([0; 32]),
            owner,
            domain: *id,
            backend,
            device_ordinal,
            placement,
            alignment_bytes: *alignment_bytes,
            maximum_requested_bytes,
            maximum_allocated_bytes,
            zero_on_release: *zero_on_release,
        };
        resolved.id = resolved.compute_id()?;
        Ok(resolved)
    }

    pub(crate) fn validate_request_bytes(&self, requested_bytes: u64) -> Result<u64> {
        if requested_bytes == 0 || requested_bytes > self.maximum_requested_bytes {
            return Err(invalid(
                "scratch workspace request is zero or exceeds its sealed maximum",
            ));
        }
        align_up(requested_bytes, self.alignment_bytes)
    }

    fn compute_id(&self) -> Result<ScratchWorkspacePlanId> {
        #[derive(Serialize)]
        struct Payload {
            owner: ScratchWorkspaceOwner,
            domain: StateDomainId,
            backend: BackendKind,
            device_ordinal: Option<u32>,
            placement: ResolvedScratchPlacement,
            alignment_bytes: u64,
            maximum_requested_bytes: u64,
            maximum_allocated_bytes: u64,
            zero_on_release: bool,
        }
        let encoded = serde_json::to_vec(&Payload {
            owner: self.owner,
            domain: self.domain,
            backend: self.backend,
            device_ordinal: self.device_ordinal,
            placement: self.placement,
            alignment_bytes: self.alignment_bytes,
            maximum_requested_bytes: self.maximum_requested_bytes,
            maximum_allocated_bytes: self.maximum_allocated_bytes,
            zero_on_release: self.zero_on_release,
        })
        .map_err(|error| invalid(format!("failed to encode scratch workspace plan: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(SCRATCH_PLAN_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(ScratchWorkspacePlanId(hasher.finalize().into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ScratchWorkspaceAllocationId {
    pub(crate) plan: ScratchWorkspacePlanId,
    /// Non-zero generation of the backing allocation/pool incarnation.
    pub(crate) generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ScratchWorkspaceSlotRef {
    pub(crate) allocation: ScratchWorkspaceAllocationId,
    pub(crate) domain: StateDomainId,
    pub(crate) slot: u32,
    /// Incremented on every reuse of this slot.
    pub(crate) slot_generation: u32,
    pub(crate) requested_bytes: u64,
    pub(crate) allocated_bytes: u64,
}

/// Metadata authority for a fixed collection of already-backed scratch slots.
/// It deliberately does not allocate host/device bytes; a backend allocator
/// creates the backing and then constructs this pool with its generation.
#[derive(Debug, Clone)]
pub(crate) struct ScratchWorkspacePool {
    inner: Arc<ScratchWorkspacePoolInner>,
}

#[derive(Debug)]
struct ScratchWorkspacePoolInner {
    plan: ResolvedScratchWorkspace,
    allocation: ScratchWorkspaceAllocationId,
    slots: Mutex<Vec<ScratchSlotState>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScratchSlotState {
    Vacant { generation: u32 },
    Leased { generation: u32 },
    PendingScrub { generation: u32 },
}

impl ScratchSlotState {
    const fn generation(self) -> u32 {
        match self {
            Self::Vacant { generation }
            | Self::Leased { generation }
            | Self::PendingScrub { generation } => generation,
        }
    }
}

impl ScratchWorkspacePool {
    pub(crate) fn new(
        plan: ResolvedScratchWorkspace,
        allocation_generation: u32,
        slot_count: u32,
    ) -> Result<Self> {
        if plan.id != plan.compute_id()? {
            return Err(invalid("scratch workspace plan identity is stale"));
        }
        if allocation_generation == 0 || slot_count == 0 {
            return Err(invalid(
                "scratch workspace allocation requires non-zero generation and slots",
            ));
        }
        let slot_count = usize::try_from(slot_count)
            .map_err(|_| invalid("scratch workspace slot count exceeds usize"))?;
        let allocation = ScratchWorkspaceAllocationId {
            plan: plan.id,
            generation: allocation_generation,
        };
        Ok(Self {
            inner: Arc::new(ScratchWorkspacePoolInner {
                plan,
                allocation,
                slots: Mutex::new(vec![ScratchSlotState::Vacant { generation: 0 }; slot_count]),
            }),
        })
    }

    pub(crate) fn allocation(&self) -> ScratchWorkspaceAllocationId {
        self.inner.allocation
    }

    pub(crate) fn lease(&self, requested_bytes: u64) -> Result<ScratchWorkspaceLease> {
        let allocated_bytes = self.inner.plan.validate_request_bytes(requested_bytes)?;
        let mut slots = self
            .inner
            .slots
            .lock()
            .map_err(|_| invalid("scratch workspace slot state is poisoned"))?;
        let (slot, state) = slots
            .iter_mut()
            .enumerate()
            .find(|(_, state)| matches!(state, ScratchSlotState::Vacant { .. }))
            .ok_or_else(|| Error::Backpressure("scratch workspace has no free slot".to_string()))?;
        let generation = state
            .generation()
            .checked_add(1)
            .ok_or_else(|| invalid("scratch workspace slot generation exhausted"))?;
        *state = ScratchSlotState::Leased { generation };
        let slot =
            u32::try_from(slot).map_err(|_| invalid("scratch workspace slot index exceeds u32"))?;
        Ok(ScratchWorkspaceLease {
            inner: self.inner.clone(),
            slot: ScratchWorkspaceSlotRef {
                allocation: self.inner.allocation,
                domain: self.inner.plan.domain,
                slot,
                slot_generation: generation,
                requested_bytes,
                allocated_bytes,
            },
        })
    }

    /// Confirm that the backend has scrubbed a released sensitive slot. Until
    /// this exact generation is confirmed, the slot cannot be leased again.
    pub(crate) fn confirm_scrubbed(&self, slot: ScratchWorkspaceSlotRef) -> Result<()> {
        self.validate_slot_identity(slot)?;
        let mut slots = self
            .inner
            .slots
            .lock()
            .map_err(|_| invalid("scratch workspace slot state is poisoned"))?;
        let state = slots
            .get_mut(slot.slot as usize)
            .ok_or_else(|| invalid("scratch workspace slot index is out of range"))?;
        if *state
            != (ScratchSlotState::PendingScrub {
                generation: slot.slot_generation,
            })
        {
            return Err(invalid(
                "scratch scrub completion is stale or the slot is not awaiting scrub",
            ));
        }
        *state = ScratchSlotState::Vacant {
            generation: slot.slot_generation,
        };
        Ok(())
    }

    pub(crate) fn contains_active_lease(&self, slot: ScratchWorkspaceSlotRef) -> bool {
        if self.validate_slot_identity(slot).is_err() {
            return false;
        }
        self.inner
            .slots
            .lock()
            .ok()
            .and_then(|slots| slots.get(slot.slot as usize).copied())
            == Some(ScratchSlotState::Leased {
                generation: slot.slot_generation,
            })
    }

    fn validate_slot_identity(&self, slot: ScratchWorkspaceSlotRef) -> Result<()> {
        if slot.allocation != self.inner.allocation
            || slot.domain != self.inner.plan.domain
            || slot.slot_generation == 0
            || slot.requested_bytes == 0
            || slot.requested_bytes > self.inner.plan.maximum_requested_bytes
            || slot.allocated_bytes
                != align_up(slot.requested_bytes, self.inner.plan.alignment_bytes)?
        {
            return Err(invalid(
                "scratch workspace slot does not belong to this allocation generation",
            ));
        }
        Ok(())
    }
}

/// Unique RAII pin for one scratch slot generation. Backends must retain this
/// value until their final read/write fence completes. Dropping it releases
/// ordinary scratch immediately or quarantines sensitive scratch for scrub.
#[derive(Debug)]
pub(crate) struct ScratchWorkspaceLease {
    inner: Arc<ScratchWorkspacePoolInner>,
    slot: ScratchWorkspaceSlotRef,
}

impl ScratchWorkspaceLease {
    pub(crate) const fn slot(&self) -> ScratchWorkspaceSlotRef {
        self.slot
    }
}

impl Drop for ScratchWorkspaceLease {
    fn drop(&mut self) {
        let Ok(mut slots) = self.inner.slots.lock() else {
            return;
        };
        let Some(state) = slots.get_mut(self.slot.slot as usize) else {
            return;
        };
        if *state
            != (ScratchSlotState::Leased {
                generation: self.slot.slot_generation,
            })
        {
            return;
        }
        *state = if self.inner.plan.zero_on_release {
            ScratchSlotState::PendingScrub {
                generation: self.slot.slot_generation,
            }
        } else {
            ScratchSlotState::Vacant {
                generation: self.slot.slot_generation,
            }
        };
    }
}

fn validate_backend_device(backend: BackendKind, device_ordinal: Option<u32>) -> Result<()> {
    match (backend, device_ordinal) {
        (BackendKind::Cpu, None) | (BackendKind::Metal | BackendKind::Cuda, Some(_)) => Ok(()),
        (BackendKind::Cpu, Some(_)) => {
            Err(invalid("CPU scratch workspace cannot use a device ordinal"))
        }
        (BackendKind::Metal | BackendKind::Cuda, None) => Err(invalid(
            "accelerator scratch workspace requires a device ordinal",
        )),
    }
}

fn resolve_placement(backend: BackendKind, placement: PlacementPolicy) -> ResolvedScratchPlacement {
    match (backend, placement) {
        (_, PlacementPolicy::Host) | (BackendKind::Cpu, _) => ResolvedScratchPlacement {
            primary: ScratchMemoryDomain::Host,
            host_offload_allowed: false,
        },
        (_, PlacementPolicy::BackendLocal) => ResolvedScratchPlacement {
            primary: ScratchMemoryDomain::BackendLocal,
            host_offload_allowed: false,
        },
        (_, PlacementPolicy::BackendLocalWithHostOffload) => ResolvedScratchPlacement {
            primary: ScratchMemoryDomain::BackendLocal,
            host_offload_allowed: true,
        },
    }
}

fn align_up(bytes: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid(
            "scratch workspace alignment must be a power of two",
        ));
    }
    bytes
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| invalid("aligned scratch workspace size overflow"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::v2::{WorkspaceAxis, WorkspaceDimensionBound, WorkspaceFormula, WorkspaceTerm};

    fn owner() -> ScratchWorkspaceOwner {
        ScratchWorkspaceOwner {
            capability_runtime: [1; 32],
            stage_graph: [2; 32],
            stage: StageId::new(3),
        }
    }

    #[test]
    fn scratch_owner_accepts_the_canonical_scalar_stage_zero() {
        let mut scalar = owner();
        scalar.stage = StageId::new(0);
        ResolvedScratchWorkspace::resolve(scalar, BackendKind::Cpu, None, &scratch(false)).unwrap();
    }

    fn scratch(zero_on_release: bool) -> InvocationWorkspaceDomain {
        InvocationWorkspaceDomain::Scratch {
            id: StateDomainId::new(4),
            placement: PlacementPolicy::BackendLocalWithHostOffload,
            alignment_bytes: 64,
            zero_on_release,
            formula: WorkspaceFormula {
                fixed_bytes: 17,
                dimensions: vec![WorkspaceDimensionBound {
                    axis: WorkspaceAxis::CodebookSteps,
                    max_units: 15,
                }],
                terms: vec![WorkspaceTerm {
                    factors: vec![WorkspaceAxis::CodebookSteps],
                    bytes_per_element: 10,
                }],
            },
        }
    }

    #[test]
    fn resolves_maximum_formula_alignment_and_backend_placement() {
        let resolved = ResolvedScratchWorkspace::resolve(
            owner(),
            BackendKind::Metal,
            Some(0),
            &scratch(false),
        )
        .unwrap();
        assert_eq!(resolved.maximum_requested_bytes, 167);
        assert_eq!(resolved.maximum_allocated_bytes, 192);
        assert_eq!(
            resolved.placement.primary,
            ScratchMemoryDomain::BackendLocal
        );
        assert!(resolved.placement.host_offload_allowed);
        assert_eq!(resolved.validate_request_bytes(65).unwrap(), 128);

        let cpu =
            ResolvedScratchWorkspace::resolve(owner(), BackendKind::Cpu, None, &scratch(false))
                .unwrap();
        assert_eq!(cpu.placement.primary, ScratchMemoryDomain::Host);
        assert!(!cpu.placement.host_offload_allowed);
    }

    #[test]
    fn rejects_wrong_domain_device_and_request_bounds() {
        assert!(ResolvedScratchWorkspace::resolve(
            owner(),
            BackendKind::Metal,
            None,
            &scratch(false),
        )
        .is_err());
        let resolved =
            ResolvedScratchWorkspace::resolve(owner(), BackendKind::Cpu, None, &scratch(false))
                .unwrap();
        assert!(resolved.validate_request_bytes(0).is_err());
        assert!(resolved.validate_request_bytes(168).is_err());
    }

    #[test]
    fn raii_release_reuses_a_slot_with_a_new_generation() {
        let resolved =
            ResolvedScratchWorkspace::resolve(owner(), BackendKind::Cpu, None, &scratch(false))
                .unwrap();
        let pool = ScratchWorkspacePool::new(resolved, 7, 1).unwrap();
        let first = pool.lease(65).unwrap();
        let first_slot = first.slot();
        assert!(pool.contains_active_lease(first_slot));
        assert!(pool.lease(1).is_err());
        drop(first);
        assert!(!pool.contains_active_lease(first_slot));

        let second = pool.lease(1).unwrap();
        let second_slot = second.slot();
        assert_eq!(second_slot.slot, first_slot.slot);
        assert!(second_slot.slot_generation > first_slot.slot_generation);
        assert!(!pool.contains_active_lease(first_slot));
        assert!(pool.contains_active_lease(second_slot));
    }

    #[test]
    fn sensitive_slot_is_quarantined_until_exact_generation_is_scrubbed() {
        let resolved =
            ResolvedScratchWorkspace::resolve(owner(), BackendKind::Cpu, None, &scratch(true))
                .unwrap();
        let pool = ScratchWorkspacePool::new(resolved, 9, 1).unwrap();
        let lease = pool.lease(32).unwrap();
        let released = lease.slot();
        drop(lease);
        assert!(pool.lease(1).is_err());

        let mut stale = released;
        stale.slot_generation = stale.slot_generation.saturating_add(1);
        assert!(pool.confirm_scrubbed(stale).is_err());
        assert!(pool.lease(1).is_err());
        pool.confirm_scrubbed(released).unwrap();
        let next = pool.lease(1).unwrap();
        assert!(next.slot().slot_generation > released.slot_generation);
    }
}
