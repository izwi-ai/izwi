use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};

use super::contract::{StateDomainId, StateGroupId};
use super::resolved::{ResolvedPlacement, ResolvedStatePlan, StatePlanId};
use super::resolved_domains::ResolvedNonPagedDomainPlan;

const ALLOCATION_PLAN_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.allocation-plan.v2\0";

/// Hard-accounted resources for one loaded capability runtime. These classes
/// are deliberately additive and allocations are charged exactly once even on
/// unified-memory systems.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StateResourceVector {
    pub(crate) host_bytes: u64,
    pub(crate) device_bytes: u64,
    pub(crate) pinned_bytes: u64,
    pub(crate) metadata_bytes: u64,
}

impl StateResourceVector {
    pub(crate) fn checked_add(self, other: Self) -> Result<Self> {
        Ok(Self {
            host_bytes: checked_add(self.host_bytes, other.host_bytes, "host")?,
            device_bytes: checked_add(self.device_bytes, other.device_bytes, "device")?,
            pinned_bytes: checked_add(self.pinned_bytes, other.pinned_bytes, "pinned")?,
            metadata_bytes: checked_add(self.metadata_bytes, other.metadata_bytes, "metadata")?,
        })
    }

    pub(crate) const fn fits_within(self, limit: Self) -> bool {
        self.host_bytes <= limit.host_bytes
            && self.device_bytes <= limit.device_bytes
            && self.pinned_bytes <= limit.pinned_bytes
            && self.metadata_bytes <= limit.metadata_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum CapacityStrategy {
    /// Fully backed before Ready publication; preferred for stable graph and
    /// pointer identities.
    Fixed { blocks: u32 },
    /// Backing is demand-materialized inside a separately bounded arena. The
    /// complete logical envelope is admitted before Ready, but no allocation
    /// receipt is required because there is no monolithic initial backing.
    BoundedLazy { max_blocks: u32 },
    /// Stable address/indirection space reserved up front, with backing
    /// committed only during admission or a maintenance barrier.
    Reserved {
        initial_blocks: u32,
        max_blocks: u32,
    },
    /// Append-only immutable slabs. Existing backing is never reallocated or
    /// moved, and growth is forbidden from layer/decode hot paths.
    AdmissionGrowable {
        initial_blocks: u32,
        growth_quantum: u32,
        max_blocks: u32,
    },
}

impl CapacityStrategy {
    fn validate(self) -> Result<()> {
        match self {
            Self::Fixed { blocks } if blocks > 0 => Ok(()),
            Self::BoundedLazy { max_blocks } if max_blocks > 0 => Ok(()),
            Self::Reserved {
                initial_blocks,
                max_blocks,
            } if initial_blocks > 0 && initial_blocks <= max_blocks => Ok(()),
            Self::AdmissionGrowable {
                initial_blocks,
                growth_quantum,
                max_blocks,
            } if initial_blocks > 0
                && growth_quantum > 0
                && initial_blocks <= max_blocks
                && growth_quantum <= max_blocks
                && (max_blocks - initial_blocks) % growth_quantum == 0 =>
            {
                Ok(())
            }
            _ => Err(invalid("invalid state capacity strategy")),
        }
    }

    const fn initial_blocks(self) -> u32 {
        match self {
            Self::Fixed { blocks } => blocks,
            Self::BoundedLazy { .. } => 0,
            Self::Reserved { initial_blocks, .. }
            | Self::AdmissionGrowable { initial_blocks, .. } => initial_blocks,
        }
    }

    pub(crate) const fn maximum_blocks(self) -> u32 {
        match self {
            Self::Fixed { blocks } => blocks,
            Self::BoundedLazy { max_blocks } => max_blocks,
            Self::Reserved { max_blocks, .. } | Self::AdmissionGrowable { max_blocks, .. } => {
                max_blocks
            }
        }
    }

    const fn minimum_backing_allocations(self) -> u32 {
        match self {
            Self::BoundedLazy { .. } => 0,
            Self::Fixed { .. } | Self::Reserved { .. } => 1,
            Self::AdmissionGrowable {
                initial_blocks,
                growth_quantum,
                max_blocks,
            } => 1 + (max_blocks - initial_blocks) / growth_quantum,
        }
    }
}

pub(crate) struct GroupResourceQuery<'a> {
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) resolved: ResolvedCapacityDomain<'a>,
    pub(crate) strategy: CapacityStrategy,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ResolvedCapacityDomain<'a> {
    Paged(&'a super::resolved::ResolvedPagedAttentionGroup),
    NonPaged(&'a ResolvedNonPagedDomainPlan),
}

impl ResolvedCapacityDomain<'_> {
    const fn group(self) -> StateGroupId {
        match self {
            Self::Paged(plan) => plan.group,
            Self::NonPaged(plan) => plan.group(),
        }
    }

    const fn domain(self) -> StateDomainId {
        match self {
            Self::Paged(plan) => plan.domain,
            Self::NonPaged(plan) => plan.domain(),
        }
    }

    const fn bytes_per_block(self) -> u64 {
        match self {
            Self::Paged(plan) => plan.bytes_per_page,
            Self::NonPaged(plan) => plan.maximum_bytes(),
        }
    }

    const fn placement(self) -> ResolvedPlacement {
        match self {
            Self::Paged(plan) => plan.placement,
            Self::NonPaged(plan) => plan.placement(),
        }
    }
}

pub(crate) struct WorkspaceResourceQuery<'a> {
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) workspace: &'a WorkspaceContract,
}

/// Allocator-owned proof of the physical costs that lifecycle must admit.
/// Callers can request capacity, but cannot supply these values themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ResolvedGroupResourceEnvelope {
    pub(crate) allocator_alignment_bytes: u64,
    pub(crate) allocator_overhead_per_allocation: u64,
    pub(crate) max_backing_allocations: u32,
    pub(crate) reservation_metadata_bytes: u64,
    pub(crate) metadata_bytes_per_block: u64,
    pub(crate) pinned_bytes_per_block: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ResolvedWorkspaceResourceEnvelope {
    pub(crate) allocator_alignment_bytes: u64,
    pub(crate) allocator_overhead_per_slot: u64,
    pub(crate) max_concurrency_slots: u32,
}

/// Implemented by the selected backend allocator/operation registry. A hard
/// admission plan is invalid unless the backend attests every state group and
/// the complete invocation workspace shape.
pub(crate) trait StateResourceRegistry {
    fn resolve_group_resources(
        &self,
        query: &GroupResourceQuery<'_>,
    ) -> Result<ResolvedGroupResourceEnvelope>;

    fn resolve_workspace_resources(
        &self,
        query: &WorkspaceResourceQuery<'_>,
    ) -> Result<ResolvedWorkspaceResourceEnvelope>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GroupCapacityRequest {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) strategy: CapacityStrategy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct GroupCapacityPlan {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) strategy: CapacityStrategy,
    pub(crate) bytes_per_block: u64,
    pub(crate) resources: ResolvedGroupResourceEnvelope,
}

impl GroupCapacityPlan {
    fn initial_resources(
        &self,
        backend: BackendKind,
        placement: ResolvedPlacement,
    ) -> Result<StateResourceVector> {
        self.strategy.validate()?;
        let blocks = u64::from(self.strategy.initial_blocks());
        self.validate_resources()?;
        let metadata_blocks = if matches!(
            self.strategy,
            CapacityStrategy::Reserved { .. } | CapacityStrategy::BoundedLazy { .. }
        ) {
            u64::from(self.strategy.maximum_blocks())
        } else {
            blocks
        };
        let owned = self.rounded_owned_bytes(blocks)?;
        let metadata = metadata_blocks
            .checked_mul(self.resources.metadata_bytes_per_block)
            .and_then(|bytes| bytes.checked_add(self.resources.reservation_metadata_bytes))
            .ok_or_else(|| invalid("initial state metadata size overflow"))?;
        let pinned = blocks
            .checked_mul(self.resources.pinned_bytes_per_block)
            .ok_or_else(|| invalid("initial pinned state size overflow"))?;
        let mut resources = place_owned(owned, backend, placement)?;
        resources.pinned_bytes = pinned;
        resources.metadata_bytes = metadata;
        Ok(resources)
    }

    fn rounded_owned_bytes(&self, blocks: u64) -> Result<u64> {
        self.validate_resources()?;
        let rounded_block = self
            .bytes_per_block
            .checked_add(self.resources.allocator_alignment_bytes - 1)
            .map(|bytes| bytes & !(self.resources.allocator_alignment_bytes - 1))
            .ok_or_else(|| invalid("aligned state block size overflow"))?;
        blocks
            .checked_mul(rounded_block)
            .ok_or_else(|| invalid("state backing size overflow"))
    }

    fn maximum_resources(
        &self,
        backend: BackendKind,
        placement: ResolvedPlacement,
    ) -> Result<StateResourceVector> {
        self.strategy.validate()?;
        let blocks = u64::from(self.strategy.maximum_blocks());
        let owned = self
            .rounded_owned_bytes(blocks)?
            .checked_add(
                u64::from(self.resources.max_backing_allocations)
                    .checked_mul(self.resources.allocator_overhead_per_allocation)
                    .ok_or_else(|| invalid("allocator overhead bound overflow"))?,
            )
            .ok_or_else(|| invalid("maximum state backing size overflow"))?;
        let metadata = blocks
            .checked_mul(self.resources.metadata_bytes_per_block)
            .and_then(|bytes| bytes.checked_add(self.resources.reservation_metadata_bytes))
            .ok_or_else(|| invalid("maximum state metadata size overflow"))?;
        let pinned = blocks
            .checked_mul(self.resources.pinned_bytes_per_block)
            .ok_or_else(|| invalid("maximum pinned state size overflow"))?;
        let mut resources = place_owned(owned, backend, placement)?;
        resources.pinned_bytes = pinned;
        resources.metadata_bytes = metadata;
        Ok(resources)
    }

    fn maximum_requested_owned_bytes(&self) -> Result<u64> {
        u64::from(self.strategy.maximum_blocks())
            .checked_mul(self.bytes_per_block)
            .ok_or_else(|| invalid("maximum requested state size overflow"))
    }

    fn validate_receipt(&self, receipt: AllocationReceipt) -> Result<()> {
        if matches!(self.strategy, CapacityStrategy::BoundedLazy { .. }) {
            return Err(invalid(
                "bounded-lazy state is reconciled by arena occupancy, not allocation receipts",
            ));
        }
        receipt.validate()?;
        if receipt.requested_owned_bytes == 0
            || receipt.requested_owned_bytes % self.bytes_per_block != 0
        {
            return Err(invalid(
                "allocation receipt must request a non-zero whole number of blocks",
            ));
        }
        let receipt_blocks = receipt.requested_owned_bytes / self.bytes_per_block;
        let maximum_committed = self.rounded_owned_bytes(receipt_blocks)?;
        if receipt.requested_owned_bytes > self.maximum_requested_owned_bytes()?
            || receipt.committed_owned_bytes > maximum_committed
            || receipt.allocator_overhead_bytes > self.resources.allocator_overhead_per_allocation
        {
            return Err(invalid(
                "allocation receipt exceeds its sealed capacity or overhead bound",
            ));
        }
        if let ResidencyMeasurement::Reported { bytes } = receipt.residency {
            let authorized_residency = receipt
                .committed_owned_bytes
                .checked_add(receipt.allocator_overhead_bytes)
                .ok_or_else(|| invalid("allocation receipt residency bound overflow"))?;
            if bytes > authorized_residency {
                return Err(invalid(
                    "reported allocation residency exceeds committed backing and overhead",
                ));
            }
        }
        Ok(())
    }

    fn validate_resources(&self) -> Result<()> {
        if self.resources.allocator_alignment_bytes == 0
            || !self.resources.allocator_alignment_bytes.is_power_of_two()
            || self.resources.max_backing_allocations < self.strategy.minimum_backing_allocations()
        {
            return Err(invalid("invalid backend state resource envelope"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkspaceAxis {
    BatchRows,
    InputTokens,
    ContextTokens,
    AudioSamples,
    AudioFrames,
    CodecFrames,
    CodebookSteps,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct WorkspaceDimensionBound {
    pub(crate) axis: WorkspaceAxis,
    pub(crate) max_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct WorkspaceTerm {
    /// Product of these runtime dimensions, allowing batch×tokens and similar
    /// request-shaped peaks rather than an unsafe additive approximation.
    pub(crate) factors: Vec<WorkspaceAxis>,
    pub(crate) bytes_per_element: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkspacePlacement {
    BackendLocal,
    Host,
    Pinned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkspaceContract {
    pub(crate) fixed_bytes: u64,
    pub(crate) dimensions: Vec<WorkspaceDimensionBound>,
    pub(crate) terms: Vec<WorkspaceTerm>,
    pub(crate) placement: WorkspacePlacement,
    /// Number of independently admitted workspace leases that may coexist.
    pub(crate) concurrency_slots: u32,
}

impl WorkspaceContract {
    pub(crate) fn validate(&self) -> Result<()> {
        self.validate_structure()?;
        self.maximum_bytes_per_slot()?;
        Ok(())
    }

    fn validate_structure(&self) -> Result<()> {
        if self.concurrency_slots == 0 {
            return Err(invalid("workspace concurrency must be non-zero"));
        }
        let mut axes = HashSet::with_capacity(self.dimensions.len());
        for dimension in &self.dimensions {
            if dimension.max_units == 0 || !axes.insert(dimension.axis) {
                return Err(invalid(
                    "workspace dimensions require non-zero unique bounded axes",
                ));
            }
        }
        for term in &self.terms {
            if term.bytes_per_element == 0 || term.factors.is_empty() {
                return Err(invalid(
                    "workspace terms require factors and non-zero element bytes",
                ));
            }
            let mut term_axes = HashSet::with_capacity(term.factors.len());
            for factor in &term.factors {
                if !term_axes.insert(*factor) || !axes.contains(factor) {
                    return Err(invalid(
                        "workspace term factors must be unique declared dimensions",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn bytes_for(&self, dimensions: &[(WorkspaceAxis, u64)]) -> Result<u64> {
        self.validate_structure()?;
        let mut values = std::collections::HashMap::with_capacity(dimensions.len());
        for (axis, units) in dimensions {
            if values.insert(*axis, *units).is_some() {
                return Err(invalid("workspace dimensions repeat an axis"));
            }
            let bound = self
                .dimensions
                .iter()
                .find(|bound| bound.axis == *axis)
                .ok_or_else(|| invalid("workspace dimensions contain an undeclared axis"))?;
            if *units > bound.max_units {
                return Err(invalid("workspace dimensions exceed their sealed bound"));
            }
        }
        if values.len() != self.dimensions.len() {
            return Err(invalid(
                "workspace dimensions do not instantiate every declared axis",
            ));
        }

        let mut total = self.fixed_bytes;
        for term in &self.terms {
            let elements = term.factors.iter().try_fold(1_u64, |product, axis| {
                product
                    .checked_mul(values[axis])
                    .ok_or_else(|| invalid("workspace request element-count overflow"))
            })?;
            total = total
                .checked_add(
                    elements
                        .checked_mul(term.bytes_per_element)
                        .ok_or_else(|| invalid("workspace request size overflow"))?,
                )
                .ok_or_else(|| invalid("workspace request size overflow"))?;
        }
        Ok(total)
    }

    fn maximum_bytes_per_slot(&self) -> Result<u64> {
        self.bytes_for(
            &self
                .dimensions
                .iter()
                .map(|dimension| (dimension.axis, dimension.max_units))
                .collect::<Vec<_>>(),
        )
    }

    fn maximum_resources(
        &self,
        backend: BackendKind,
        resources: ResolvedWorkspaceResourceEnvelope,
    ) -> Result<StateResourceVector> {
        if resources.allocator_alignment_bytes == 0
            || !resources.allocator_alignment_bytes.is_power_of_two()
            || resources.max_concurrency_slots < self.concurrency_slots
        {
            return Err(invalid("invalid backend workspace resource envelope"));
        }
        let requested = self.maximum_bytes_per_slot()?;
        let aligned = requested
            .checked_add(resources.allocator_alignment_bytes - 1)
            .map(|bytes| bytes & !(resources.allocator_alignment_bytes - 1))
            .ok_or_else(|| invalid("aligned workspace size overflow"))?;
        let bytes = aligned
            .checked_add(resources.allocator_overhead_per_slot)
            .and_then(|bytes| bytes.checked_mul(u64::from(self.concurrency_slots)))
            .ok_or_else(|| invalid("concurrent workspace peak overflow"))?;
        match self.placement {
            WorkspacePlacement::BackendLocal => {
                place_owned(bytes, backend, ResolvedPlacement::BackendLocal)
            }
            WorkspacePlacement::Host => place_owned(bytes, backend, ResolvedPlacement::Host),
            WorkspacePlacement::Pinned => Ok(StateResourceVector {
                pinned_bytes: bytes,
                ..StateResourceVector::default()
            }),
        }
    }
}

/// Per-loaded-instance capacity negotiation. It references an immutable state
/// plan but cannot change its fingerprint or kernel/layout identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub(crate) struct StateAllocationPlanId([u8; 32]);

impl fmt::Debug for StateAllocationPlanId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StateAllocationPlanId(")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str(")")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct StateRuntimeAllocationPlan {
    pub(crate) id: StateAllocationPlanId,
    pub(crate) state_plan: StatePlanId,
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) groups: Vec<GroupCapacityPlan>,
    pub(crate) workspace: WorkspaceContract,
    pub(crate) workspace_resources: ResolvedWorkspaceResourceEnvelope,
    pub(crate) hard_limit: StateResourceVector,
}

impl StateRuntimeAllocationPlan {
    /// Build a sealed plan whose hard limit is exactly the allocator-attested
    /// maximum for its state groups and workspace.
    pub(crate) fn build_exact(
        state_plan: &ResolvedStatePlan,
        model_instance: ModelInstanceId,
        groups: Vec<GroupCapacityRequest>,
        workspace: WorkspaceContract,
        resource_registry: &dyn StateResourceRegistry,
    ) -> Result<Self> {
        let provisional = Self::build(
            state_plan,
            model_instance,
            groups.clone(),
            workspace.clone(),
            StateResourceVector {
                host_bytes: u64::MAX,
                device_bytes: u64::MAX,
                pinned_bytes: u64::MAX,
                metadata_bytes: u64::MAX,
            },
            resource_registry,
        )?;
        let exact_limit = provisional.maximum_resources(state_plan)?;
        Self::build(
            state_plan,
            model_instance,
            groups,
            workspace,
            exact_limit,
            resource_registry,
        )
    }

    pub(crate) fn build(
        state_plan: &ResolvedStatePlan,
        model_instance: ModelInstanceId,
        groups: Vec<GroupCapacityRequest>,
        workspace: WorkspaceContract,
        hard_limit: StateResourceVector,
        resource_registry: &dyn StateResourceRegistry,
    ) -> Result<Self> {
        let mut resolved_groups = Vec::with_capacity(groups.len());
        for request in groups {
            request.strategy.validate()?;
            let resolved = state_plan
                .paged_attention
                .iter()
                .find(|candidate| {
                    candidate.group == request.group && candidate.domain == request.domain
                })
                .map(ResolvedCapacityDomain::Paged)
                .or_else(|| {
                    state_plan
                        .non_paged
                        .iter()
                        .find(|candidate| {
                            candidate.group() == request.group
                                && candidate.domain() == request.domain
                        })
                        .map(ResolvedCapacityDomain::NonPaged)
                })
                .ok_or_else(|| invalid("capacity request references an unknown state group"))?;
            let resources = resource_registry.resolve_group_resources(&GroupResourceQuery {
                backend: state_plan.backend,
                device_ordinal: state_plan.device_ordinal,
                resolved,
                strategy: request.strategy,
            })?;
            resolved_groups.push(GroupCapacityPlan {
                group: resolved.group(),
                domain: resolved.domain(),
                strategy: request.strategy,
                bytes_per_block: resolved.bytes_per_block(),
                resources,
            });
        }
        let workspace_resources =
            resource_registry.resolve_workspace_resources(&WorkspaceResourceQuery {
                backend: state_plan.backend,
                device_ordinal: state_plan.device_ordinal,
                workspace: &workspace,
            })?;
        let plan = Self {
            id: StateAllocationPlanId([0; 32]),
            state_plan: state_plan.id,
            model_instance,
            groups: resolved_groups,
            workspace,
            workspace_resources,
            hard_limit,
        };
        let mut plan = plan;
        plan.id = plan.compute_id()?;
        plan.validate_against(state_plan)?;
        Ok(plan)
    }

    pub(crate) fn validate_against(&self, state_plan: &ResolvedStatePlan) -> Result<()> {
        if self.state_plan != state_plan.id {
            return Err(invalid(
                "runtime allocation plan belongs to a different state plan",
            ));
        }
        if self.id != self.compute_id()? {
            return Err(invalid("runtime allocation plan identity is stale"));
        }
        if self.model_instance.get() == 0 {
            return Err(invalid("runtime allocation plan has zero model instance"));
        }
        self.workspace.validate()?;

        let mut expected = state_plan
            .paged_attention
            .iter()
            .map(|group| {
                (
                    (group.group, group.domain),
                    group.bytes_per_page,
                    group.placement,
                )
            })
            .collect::<Vec<_>>();
        expected.extend(state_plan.non_paged.iter().map(|domain| {
            (
                (domain.group(), domain.domain()),
                domain.maximum_bytes(),
                domain.placement(),
            )
        }));
        expected.sort_unstable_by_key(|(key, _, _)| *key);
        if self.groups.len() != expected.len() {
            return Err(invalid(
                "runtime allocation plan does not cover every resolved state group",
            ));
        }
        let mut seen = HashSet::with_capacity(self.groups.len());
        let mut previous = None;
        let mut maximum = self
            .workspace
            .maximum_resources(state_plan.backend, self.workspace_resources)?;

        for group in &self.groups {
            let key = (group.group, group.domain);
            if previous.is_some_and(|previous| key <= previous) || !seen.insert(key) {
                return Err(invalid(
                    "runtime capacity groups must be in canonical unique order",
                ));
            }
            previous = Some(key);
            let expected_bytes = expected
                .iter()
                .find_map(|(candidate, bytes, placement)| {
                    (*candidate == key).then_some((*bytes, *placement))
                })
                .ok_or_else(|| invalid("runtime capacity references an unknown state group"))?;
            if group.bytes_per_block != expected_bytes.0 || group.bytes_per_block == 0 {
                return Err(invalid(
                    "runtime capacity bytes do not match the resolved state plan",
                ));
            }
            // Validate both the immediately committed backing and the maximum
            // authorized shape. The latter must fit the hard resource vector
            // even when backing is admitted incrementally.
            group.initial_resources(state_plan.backend, expected_bytes.1)?;
            maximum = maximum
                .checked_add(group.maximum_resources(state_plan.backend, expected_bytes.1)?)?;
        }
        if !maximum.fits_within(self.hard_limit) {
            return Err(invalid(
                "runtime state and peak workspace exceed the hard resource vector",
            ));
        }
        Ok(())
    }

    pub(crate) fn maximum_resources(
        &self,
        state_plan: &ResolvedStatePlan,
    ) -> Result<StateResourceVector> {
        if self.state_plan != state_plan.id {
            return Err(invalid(
                "runtime allocation plan belongs to a different state plan",
            ));
        }
        let mut maximum = self
            .workspace
            .maximum_resources(state_plan.backend, self.workspace_resources)?;
        for group in &self.groups {
            let placement = self.group_placement(state_plan, group)?;
            maximum =
                maximum.checked_add(group.maximum_resources(state_plan.backend, placement)?)?;
        }
        Ok(maximum)
    }

    /// Physical state backing that must exist before Ready publication.
    /// Transient workspace and bounded-lazy owned bytes are intentionally
    /// excluded; metadata reserved for the lazy arena remains included.
    pub(crate) fn initial_state_resources(
        &self,
        state_plan: &ResolvedStatePlan,
    ) -> Result<StateResourceVector> {
        if self.state_plan != state_plan.id {
            return Err(invalid(
                "runtime allocation plan belongs to a different state plan",
            ));
        }
        self.groups
            .iter()
            .try_fold(StateResourceVector::default(), |total, group| {
                let placement = self.group_placement(state_plan, group)?;
                total.checked_add(group.initial_resources(state_plan.backend, placement)?)
            })
    }

    fn group_placement(
        &self,
        state_plan: &ResolvedStatePlan,
        group: &GroupCapacityPlan,
    ) -> Result<ResolvedPlacement> {
        state_plan
            .paged_attention
            .iter()
            .find(|candidate| candidate.group == group.group && candidate.domain == group.domain)
            .map(|candidate| candidate.placement)
            .or_else(|| {
                state_plan
                    .non_paged
                    .iter()
                    .find(|candidate| {
                        candidate.group() == group.group && candidate.domain() == group.domain
                    })
                    .map(ResolvedNonPagedDomainPlan::placement)
            })
            .ok_or_else(|| invalid("runtime capacity references an unknown state group"))
    }

    pub(crate) fn group_capacity(
        &self,
        group: StateGroupId,
        domain: StateDomainId,
    ) -> Result<&GroupCapacityPlan> {
        self.groups
            .iter()
            .find(|candidate| candidate.group == group && candidate.domain == domain)
            .ok_or_else(|| invalid("allocation receipt references an unknown capacity group"))
    }

    fn compute_id(&self) -> Result<StateAllocationPlanId> {
        #[derive(Serialize)]
        struct Payload<'a> {
            state_plan: StatePlanId,
            model_instance: ModelInstanceId,
            groups: &'a [GroupCapacityPlan],
            workspace: &'a WorkspaceContract,
            workspace_resources: ResolvedWorkspaceResourceEnvelope,
            hard_limit: StateResourceVector,
        }
        let encoded = serde_json::to_vec(&Payload {
            state_plan: self.state_plan,
            model_instance: self.model_instance,
            groups: &self.groups,
            workspace: &self.workspace,
            workspace_resources: self.workspace_resources,
            hard_limit: self.hard_limit,
        })
        .map_err(|error| invalid(format!("failed to encode allocation plan: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(ALLOCATION_PLAN_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(StateAllocationPlanId(hasher.finalize().into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ResidencyMeasurement {
    Unknown,
    Estimated { bytes: u64 },
    Reported { bytes: u64 },
}

/// Allocation accounting produced by the backend. Requested/committed owned
/// bytes are exact; driver residency remains explicitly labeled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AllocationReceipt {
    pub(crate) requested_owned_bytes: u64,
    pub(crate) committed_owned_bytes: u64,
    pub(crate) allocator_overhead_bytes: u64,
    pub(crate) residency: ResidencyMeasurement,
}

impl AllocationReceipt {
    pub(crate) fn validate(self) -> Result<()> {
        if self.committed_owned_bytes < self.requested_owned_bytes {
            return Err(invalid(
                "allocation receipt committed fewer bytes than requested",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GroupAllocationTotals {
    pub(crate) allocations: u32,
    pub(crate) allocated_blocks: u32,
    pub(crate) requested_owned_bytes: u64,
    pub(crate) committed_owned_bytes: u64,
    pub(crate) allocator_overhead_bytes: u64,
}

/// Cumulative allocation truth for one sealed runtime plan. Receipts describe
/// allocation deltas; accepting them independently would let repeated growth
/// allocations exceed the admitted maximum.
#[derive(Debug)]
pub(crate) struct StateAllocationLedger {
    allocation_plan: StateAllocationPlanId,
    groups: HashMap<(StateGroupId, StateDomainId), GroupAllocationTotals>,
}

impl StateAllocationLedger {
    pub(crate) fn new(plan: &StateRuntimeAllocationPlan) -> Self {
        Self {
            allocation_plan: plan.id,
            groups: HashMap::with_capacity(plan.groups.len()),
        }
    }

    pub(crate) fn reconcile_group_receipt(
        &mut self,
        plan: &StateRuntimeAllocationPlan,
        group: StateGroupId,
        domain: StateDomainId,
        receipt: AllocationReceipt,
    ) -> Result<GroupAllocationTotals> {
        if self.allocation_plan != plan.id {
            return Err(invalid(
                "allocation ledger belongs to a different runtime plan",
            ));
        }
        let capacity = plan.group_capacity(group, domain)?;
        capacity.validate_receipt(receipt)?;
        let previous = self
            .groups
            .get(&(group, domain))
            .copied()
            .unwrap_or_default();
        let receipt_blocks =
            u32::try_from(receipt.requested_owned_bytes / capacity.bytes_per_block)
                .map_err(|_| invalid("allocation receipt block count overflow"))?;
        let expected_blocks = if previous.allocations == 0 {
            capacity.strategy.initial_blocks()
        } else {
            match capacity.strategy {
                CapacityStrategy::BoundedLazy { .. } => {
                    return Err(invalid(
                        "bounded-lazy state cannot publish allocation receipts",
                    ));
                }
                CapacityStrategy::Fixed { .. } => {
                    return Err(invalid("fixed state capacity cannot grow after Ready"));
                }
                CapacityStrategy::Reserved { .. } => receipt_blocks,
                CapacityStrategy::AdmissionGrowable { growth_quantum, .. } => growth_quantum,
            }
        };
        if receipt_blocks != expected_blocks {
            return Err(invalid(
                "allocation receipt does not match the required initial or growth block quantum",
            ));
        }
        let next = GroupAllocationTotals {
            allocations: previous
                .allocations
                .checked_add(1)
                .ok_or_else(|| invalid("allocation receipt count overflow"))?,
            allocated_blocks: previous
                .allocated_blocks
                .checked_add(receipt_blocks)
                .ok_or_else(|| invalid("allocated block count overflow"))?,
            requested_owned_bytes: checked_add(
                previous.requested_owned_bytes,
                receipt.requested_owned_bytes,
                "requested owned",
            )?,
            committed_owned_bytes: checked_add(
                previous.committed_owned_bytes,
                receipt.committed_owned_bytes,
                "committed owned",
            )?,
            allocator_overhead_bytes: checked_add(
                previous.allocator_overhead_bytes,
                receipt.allocator_overhead_bytes,
                "allocator overhead",
            )?,
        };
        let maximum_committed =
            capacity.rounded_owned_bytes(u64::from(capacity.strategy.maximum_blocks()))?;
        let maximum_overhead = u64::from(capacity.resources.max_backing_allocations)
            .checked_mul(capacity.resources.allocator_overhead_per_allocation)
            .ok_or_else(|| invalid("allocator overhead bound overflow"))?;
        if next.allocations > capacity.resources.max_backing_allocations
            || next.allocated_blocks > capacity.strategy.maximum_blocks()
            || next.requested_owned_bytes > capacity.maximum_requested_owned_bytes()?
            || next.committed_owned_bytes > maximum_committed
            || next.allocator_overhead_bytes > maximum_overhead
        {
            return Err(invalid(
                "cumulative allocation receipts exceed the sealed capacity envelope",
            ));
        }
        self.groups.insert((group, domain), next);
        Ok(next)
    }

    pub(crate) fn ensure_ready(&self, plan: &StateRuntimeAllocationPlan) -> Result<()> {
        if self.allocation_plan != plan.id {
            return Err(invalid(
                "allocation ledger belongs to a different runtime plan",
            ));
        }
        for capacity in &plan.groups {
            if matches!(capacity.strategy, CapacityStrategy::BoundedLazy { .. }) {
                continue;
            }
            let totals = self
                .groups
                .get(&(capacity.group, capacity.domain))
                .ok_or_else(|| invalid("runtime state backing is not ready"))?;
            if totals.allocated_blocks < capacity.strategy.initial_blocks() {
                return Err(invalid("runtime state backing is not ready"));
            }
        }
        Ok(())
    }
}

fn checked_add(left: u64, right: u64, class: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("{class} resource accounting overflow")))
}

fn place_owned(
    bytes: u64,
    backend: BackendKind,
    placement: ResolvedPlacement,
) -> Result<StateResourceVector> {
    match (placement, backend) {
        (ResolvedPlacement::Host, _) | (ResolvedPlacement::BackendLocal, BackendKind::Cpu) => {
            Ok(StateResourceVector {
                host_bytes: bytes,
                ..StateResourceVector::default()
            })
        }
        (ResolvedPlacement::BackendLocal, BackendKind::Metal | BackendKind::Cuda) => {
            Ok(StateResourceVector {
                device_bytes: bytes,
                ..StateResourceVector::default()
            })
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::v2::resolved::test_plan;
    use crate::kv::v2::resolved_domains::tests::{
        contract as tensor_contract, tensor_plan, TestRegistry as TensorOperationRegistry,
    };
    use crate::kv::v2::test_contract;

    struct TestResources;

    impl StateResourceRegistry for TestResources {
        fn resolve_group_resources(
            &self,
            query: &GroupResourceQuery<'_>,
        ) -> Result<ResolvedGroupResourceEnvelope> {
            let (max_backing_allocations, reservation_metadata_bytes) = match query.strategy {
                CapacityStrategy::Fixed { .. } => (1, 0),
                CapacityStrategy::BoundedLazy { .. } => (0, 0),
                CapacityStrategy::Reserved { .. } => (3, 64),
                CapacityStrategy::AdmissionGrowable { .. } => {
                    (query.strategy.minimum_backing_allocations(), 0)
                }
            };
            Ok(ResolvedGroupResourceEnvelope {
                allocator_alignment_bytes: 256,
                allocator_overhead_per_allocation: 4096,
                max_backing_allocations,
                reservation_metadata_bytes,
                metadata_bytes_per_block: 32,
                pinned_bytes_per_block: 16,
            })
        }

        fn resolve_workspace_resources(
            &self,
            _query: &WorkspaceResourceQuery<'_>,
        ) -> Result<ResolvedWorkspaceResourceEnvelope> {
            Ok(ResolvedWorkspaceResourceEnvelope {
                allocator_alignment_bytes: 256,
                allocator_overhead_per_slot: 128,
                max_concurrency_slots: 4,
            })
        }
    }

    fn workspace() -> WorkspaceContract {
        WorkspaceContract {
            fixed_bytes: 1024,
            dimensions: vec![
                WorkspaceDimensionBound {
                    axis: WorkspaceAxis::BatchRows,
                    max_units: 8,
                },
                WorkspaceDimensionBound {
                    axis: WorkspaceAxis::InputTokens,
                    max_units: 256,
                },
            ],
            terms: vec![
                WorkspaceTerm {
                    factors: vec![WorkspaceAxis::BatchRows],
                    bytes_per_element: 64,
                },
                WorkspaceTerm {
                    factors: vec![WorkspaceAxis::BatchRows, WorkspaceAxis::InputTokens],
                    bytes_per_element: 128,
                },
            ],
            placement: WorkspacePlacement::BackendLocal,
            concurrency_slots: 2,
        }
    }

    #[test]
    fn capacity_is_separate_from_resolved_plan_identity() {
        let contract = test_contract();
        let state_plan = test_plan(&contract);
        let fingerprint = state_plan.fingerprint();
        let group = &state_plan.paged_attention[0];
        let allocation = StateRuntimeAllocationPlan::build(
            &state_plan,
            ModelInstanceId::new(7),
            vec![GroupCapacityRequest {
                group: group.group,
                domain: group.domain,
                strategy: CapacityStrategy::AdmissionGrowable {
                    initial_blocks: 4,
                    growth_quantum: 4,
                    max_blocks: 32,
                },
            }],
            workspace(),
            StateResourceVector {
                host_bytes: 256 * group.bytes_per_page,
                pinned_bytes: 32 * 16,
                metadata_bytes: 32 * 32,
                ..StateResourceVector::default()
            },
            &TestResources,
        )
        .unwrap();

        allocation.validate_against(&state_plan).unwrap();
        let mut ledger = StateAllocationLedger::new(&allocation);
        ledger
            .reconcile_group_receipt(
                &allocation,
                group.group,
                group.domain,
                AllocationReceipt {
                    requested_owned_bytes: 4 * group.bytes_per_page,
                    committed_owned_bytes: 4 * group.bytes_per_page,
                    allocator_overhead_bytes: 4096,
                    residency: ResidencyMeasurement::Unknown,
                },
            )
            .unwrap();
        ledger.ensure_ready(&allocation).unwrap();
        let mut unready = StateAllocationLedger::new(&allocation);
        assert!(unready
            .reconcile_group_receipt(
                &allocation,
                group.group,
                group.domain,
                AllocationReceipt {
                    requested_owned_bytes: group.bytes_per_page,
                    committed_owned_bytes: group.bytes_per_page,
                    allocator_overhead_bytes: 4096,
                    residency: ResidencyMeasurement::Unknown,
                },
            )
            .is_err());
        assert!(unready.ensure_ready(&allocation).is_err());

        let other_allocation = StateRuntimeAllocationPlan::build(
            &state_plan,
            ModelInstanceId::new(7),
            vec![GroupCapacityRequest {
                group: group.group,
                domain: group.domain,
                strategy: CapacityStrategy::AdmissionGrowable {
                    initial_blocks: 4,
                    growth_quantum: 4,
                    max_blocks: 32,
                },
            }],
            workspace(),
            StateResourceVector {
                host_bytes: 512 * group.bytes_per_page,
                pinned_bytes: 32 * 16,
                metadata_bytes: 32 * 32,
                ..StateResourceVector::default()
            },
            &TestResources,
        )
        .unwrap();
        assert_ne!(allocation.id, other_allocation.id);
        assert!(ledger.ensure_ready(&other_allocation).is_err());
        assert!(ledger
            .reconcile_group_receipt(
                &allocation,
                group.group,
                group.domain,
                AllocationReceipt {
                    requested_owned_bytes: 4 * group.bytes_per_page,
                    committed_owned_bytes: 4 * group.bytes_per_page,
                    allocator_overhead_bytes: 9 * 4096,
                    residency: ResidencyMeasurement::Unknown,
                },
            )
            .is_err());
        assert!(ledger
            .reconcile_group_receipt(
                &allocation,
                group.group,
                group.domain,
                AllocationReceipt {
                    requested_owned_bytes: 4 * group.bytes_per_page,
                    committed_owned_bytes: 33 * group.bytes_per_page,
                    allocator_overhead_bytes: 4096,
                    residency: ResidencyMeasurement::Unknown,
                },
            )
            .is_err());
        for _ in 0..7 {
            ledger
                .reconcile_group_receipt(
                    &allocation,
                    group.group,
                    group.domain,
                    AllocationReceipt {
                        requested_owned_bytes: 4 * group.bytes_per_page,
                        committed_owned_bytes: 4 * group.bytes_per_page,
                        allocator_overhead_bytes: 4096,
                        residency: ResidencyMeasurement::Unknown,
                    },
                )
                .unwrap();
        }
        assert!(ledger
            .reconcile_group_receipt(
                &allocation,
                group.group,
                group.domain,
                AllocationReceipt {
                    requested_owned_bytes: 4 * group.bytes_per_page,
                    committed_owned_bytes: 4 * group.bytes_per_page,
                    allocator_overhead_bytes: 4096,
                    residency: ResidencyMeasurement::Unknown,
                },
            )
            .is_err());
        assert_eq!(state_plan.fingerprint(), fingerprint);
    }

    #[test]
    fn hard_limit_accounts_for_state_and_peak_workspace_together() {
        let contract = test_contract();
        let state_plan = test_plan(&contract);
        let group = &state_plan.paged_attention[0];
        let result = StateRuntimeAllocationPlan::build(
            &state_plan,
            ModelInstanceId::new(7),
            vec![GroupCapacityRequest {
                group: group.group,
                domain: group.domain,
                strategy: CapacityStrategy::Fixed { blocks: 4 },
            }],
            workspace(),
            StateResourceVector {
                host_bytes: 4 * group.bytes_per_page,
                metadata_bytes: 128,
                ..StateResourceVector::default()
            },
            &TestResources,
        );
        assert!(result.is_err());
    }

    #[test]
    fn workspace_instantiation_is_bounded_and_complete() {
        let workspace = workspace();
        assert_eq!(
            workspace
                .bytes_for(&[
                    (WorkspaceAxis::BatchRows, 2),
                    (WorkspaceAxis::InputTokens, 16),
                ])
                .unwrap(),
            1024 + 2 * 64 + 2 * 16 * 128
        );
        assert!(workspace
            .bytes_for(&[(WorkspaceAxis::BatchRows, 2)])
            .is_err());
        assert!(workspace
            .bytes_for(&[
                (WorkspaceAxis::BatchRows, 2),
                (WorkspaceAxis::InputTokens, 257),
            ])
            .is_err());
    }

    #[test]
    fn allocation_receipt_never_claims_unknown_residency_precision() {
        AllocationReceipt {
            requested_owned_bytes: 1024,
            committed_owned_bytes: 2048,
            allocator_overhead_bytes: 64,
            residency: ResidencyMeasurement::Unknown,
        }
        .validate()
        .unwrap();

        assert!(AllocationReceipt {
            requested_owned_bytes: 2048,
            committed_owned_bytes: 1024,
            allocator_overhead_bytes: 0,
            residency: ResidencyMeasurement::Reported { bytes: 4096 },
        }
        .validate()
        .is_err());

        let contract = test_contract();
        let state_plan = test_plan(&contract);
        let group = &state_plan.paged_attention[0];
        let capacity = GroupCapacityPlan {
            group: group.group,
            domain: group.domain,
            strategy: CapacityStrategy::Fixed { blocks: 1 },
            bytes_per_block: group.bytes_per_page,
            resources: ResolvedGroupResourceEnvelope {
                allocator_alignment_bytes: 256,
                allocator_overhead_per_allocation: 64,
                max_backing_allocations: 1,
                reservation_metadata_bytes: 0,
                metadata_bytes_per_block: 0,
                pinned_bytes_per_block: 0,
            },
        };
        assert!(capacity
            .validate_receipt(AllocationReceipt {
                requested_owned_bytes: group.bytes_per_page,
                committed_owned_bytes: group.bytes_per_page,
                allocator_overhead_bytes: 64,
                residency: ResidencyMeasurement::Reported {
                    bytes: group.bytes_per_page + 65,
                },
            })
            .is_err());
    }

    #[test]
    fn placement_concurrency_and_reserved_metadata_are_hard_accounted() {
        let reserved = GroupCapacityPlan {
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            strategy: CapacityStrategy::Reserved {
                initial_blocks: 2,
                max_blocks: 8,
            },
            bytes_per_block: 1024,
            resources: ResolvedGroupResourceEnvelope {
                allocator_alignment_bytes: 256,
                allocator_overhead_per_allocation: 4096,
                max_backing_allocations: 3,
                reservation_metadata_bytes: 64,
                metadata_bytes_per_block: 32,
                pinned_bytes_per_block: 16,
            },
        };
        let ready = reserved
            .initial_resources(BackendKind::Cuda, ResolvedPlacement::Host)
            .unwrap();
        assert_eq!(ready.host_bytes, 2 * 1024);
        assert_eq!(ready.device_bytes, 0);
        assert_eq!(ready.metadata_bytes, 8 * 32 + 64);
        assert_eq!(ready.pinned_bytes, 2 * 16);

        let workspace = workspace();
        let maximum = workspace
            .maximum_resources(
                BackendKind::Cuda,
                ResolvedWorkspaceResourceEnvelope {
                    allocator_alignment_bytes: 256,
                    allocator_overhead_per_slot: 128,
                    max_concurrency_slots: 4,
                },
            )
            .unwrap();
        assert_eq!(maximum.host_bytes, 0);
        let raw_per_slot = 1024 + 8 * 64 + 8 * 256 * 128;
        let aligned_per_slot = (raw_per_slot + 255) & !255;
        assert_eq!(maximum.device_bytes, 2 * (aligned_per_slot + 128));
    }

    #[test]
    fn growable_capacity_requires_exact_slab_quanta() {
        assert!(CapacityStrategy::AdmissionGrowable {
            initial_blocks: 5,
            growth_quantum: 4,
            max_blocks: 6,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn non_paged_state_is_hard_admitted_from_its_resolved_byte_bound() {
        let contract = tensor_contract();
        let state_plan = ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![],
            vec![tensor_plan(16)],
            &TensorOperationRegistry,
        )
        .unwrap();
        let allocation = StateRuntimeAllocationPlan::build_exact(
            &state_plan,
            ModelInstanceId::new(9),
            vec![GroupCapacityRequest {
                group: StateGroupId::new(1),
                domain: StateDomainId::new(1),
                strategy: CapacityStrategy::BoundedLazy { max_blocks: 3 },
            }],
            WorkspaceContract {
                fixed_bytes: 0,
                dimensions: vec![],
                terms: vec![],
                placement: WorkspacePlacement::Host,
                concurrency_slots: 1,
            },
            &TestResources,
        )
        .unwrap();
        assert_eq!(allocation.groups[0].bytes_per_block, 16);
        assert_eq!(allocation.groups[0].strategy.initial_blocks(), 0);
        assert_eq!(allocation.groups[0].strategy.maximum_blocks(), 3);
        assert_eq!(
            allocation.hard_limit,
            allocation.maximum_resources(&state_plan).unwrap()
        );
        let initial = allocation.groups[0]
            .initial_resources(BackendKind::Cpu, ResolvedPlacement::BackendLocal)
            .unwrap();
        assert_eq!(initial.host_bytes, 0);
        assert_eq!(initial.metadata_bytes, 3 * 32);

        let mut ledger = StateAllocationLedger::new(&allocation);
        ledger.ensure_ready(&allocation).unwrap();
        assert!(ledger
            .reconcile_group_receipt(
                &allocation,
                StateGroupId::new(1),
                StateDomainId::new(1),
                AllocationReceipt {
                    requested_owned_bytes: 16,
                    committed_owned_bytes: 16,
                    allocator_overhead_bytes: 0,
                    residency: ResidencyMeasurement::Unknown,
                },
            )
            .is_err());
    }
}
