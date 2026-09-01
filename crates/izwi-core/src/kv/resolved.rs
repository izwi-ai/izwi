use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};

use super::v2::{
    CapacityStrategy, ResolvedStatePlan, StateAllocationPlanId, StateDType, StateDomainId,
    StateLayerBinding, StatePhysicalLayout, StateRuntimeAllocationPlan, StateStorageFormat,
};

const PLAN_FINGERPRINT_DOMAIN: &[u8] = b"izwi.physical-kv.allocation-plan.v2\0";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KvPlanFingerprint([u8; 32]);

impl KvPlanFingerprint {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for KvPlanFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("KvPlanFingerprint(")?;
        hex_fingerprint(&self.0, formatter)?;
        formatter.write_str(")")
    }
}

impl fmt::Display for KvPlanFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        hex_fingerprint(&self.0, formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KvPlanId(KvPlanFingerprint);

impl KvPlanId {
    pub const fn new(fingerprint: KvPlanFingerprint) -> Self {
        Self(fingerprint)
    }

    pub const fn fingerprint(self) -> KvPlanFingerprint {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KvGroupId(u32);

impl KvGroupId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvArenaId {
    pub model_instance: ModelInstanceId,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    pub generation: u32,
}

/// Capacity-bearing allocation view derived from one already-negotiated V2
/// state plan. It introduces no second semantic or backend policy decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedKvPlan {
    pub id: KvPlanId,
    pub model_instance: ModelInstanceId,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    pub contract_fingerprint: [u8; 32],
    pub state_plan_fingerprint: [u8; 32],
    pub(crate) state_allocation_plan: StateAllocationPlanId,
    pub groups: Vec<ResolvedKvGroup>,
}

impl ResolvedKvPlan {
    pub(crate) fn from_runtime_allocation(
        first_arena_generation: u32,
        state_plan: &ResolvedStatePlan,
        allocation_plan: &StateRuntimeAllocationPlan,
    ) -> Result<Self> {
        if first_arena_generation == 0 {
            return Err(invalid(
                "first physical KV arena generation must be non-zero",
            ));
        }
        if state_plan.paged_attention.is_empty() {
            return Err(invalid(
                "physical KV allocation requires a resolved paged-attention domain",
            ));
        }
        allocation_plan.validate_against(state_plan)?;

        let mut groups = Vec::with_capacity(state_plan.paged_attention.len());
        for (ordinal, resolved) in state_plan.paged_attention.iter().enumerate() {
            let capacity = allocation_plan.group_capacity(resolved.group, resolved.domain)?;
            let capacity_pages = match capacity.strategy {
                CapacityStrategy::Fixed { blocks } => blocks,
                CapacityStrategy::AdmissionGrowable { max_blocks, .. } => max_blocks,
                CapacityStrategy::BoundedLazy { .. } | CapacityStrategy::Reserved { .. } => {
                    return Err(invalid(
                        "paged-attention capacity must use fixed or admission-growable allocation",
                    ));
                }
            };
            let ordinal = u32::try_from(ordinal)
                .map_err(|_| invalid("physical KV group count exceeds u32"))?;
            let generation = first_arena_generation
                .checked_add(ordinal)
                .ok_or_else(|| invalid("physical KV arena generation overflow"))?;
            let layout = match resolved.layout {
                StatePhysicalLayout::PageTokenHeadDim => KvPhysicalLayout::PageTokenHeadDim,
                StatePhysicalLayout::PageHeadTokenDim => KvPhysicalLayout::PageHeadTokenDim,
            };
            let storage = match resolved.storage {
                StateStorageFormat::Dense { dtype } => KvStorageFormat::Dense { dtype },
            };
            groups.push(ResolvedKvGroup {
                id: KvGroupId::new(resolved.group.get()),
                arena: KvArenaId {
                    model_instance: allocation_plan.model_instance,
                    backend: state_plan.backend,
                    device_ordinal: state_plan.device_ordinal,
                    generation,
                },
                domain: resolved.domain,
                page_tokens: resolved.page_tokens,
                capacity_pages,
                capacity_strategy: capacity.strategy,
                bytes_per_page: resolved.bytes_per_page,
                layout,
                storage,
                layers: resolved
                    .layers
                    .iter()
                    .map(|binding| KvLayerBinding {
                        model_layer: binding.model_layer,
                        physical_layer: binding.physical_layer,
                    })
                    .collect(),
            });
        }

        let mut plan = Self {
            id: KvPlanId::new(KvPlanFingerprint::new([0; 32])),
            model_instance: allocation_plan.model_instance,
            backend: state_plan.backend,
            device_ordinal: state_plan.device_ordinal,
            contract_fingerprint: state_plan.contract_fingerprint,
            state_plan_fingerprint: state_plan.fingerprint().bytes(),
            state_allocation_plan: allocation_plan.id,
            groups,
        };
        plan.id = KvPlanId::new(plan.compute_fingerprint()?);
        plan.validate_against_allocation(state_plan, allocation_plan)?;
        Ok(plan)
    }

    pub fn fingerprint(&self) -> KvPlanFingerprint {
        self.id.fingerprint()
    }

    pub(crate) fn validate_against(&self, state_plan: &ResolvedStatePlan) -> Result<()> {
        if self.backend != state_plan.backend
            || self.device_ordinal != state_plan.device_ordinal
            || self.contract_fingerprint != state_plan.contract_fingerprint
            || self.state_plan_fingerprint != state_plan.fingerprint().bytes()
            || self.groups.len() != state_plan.paged_attention.len()
        {
            return Err(invalid(
                "physical KV allocation was derived from a different V2 state plan",
            ));
        }
        if self.id.fingerprint() != self.compute_fingerprint()? {
            return Err(invalid(
                "physical KV allocation fingerprint is stale or invalid",
            ));
        }
        for (group, resolved) in self.groups.iter().zip(&state_plan.paged_attention) {
            let same_layers = group.layers.len() == resolved.layers.len()
                && group
                    .layers
                    .iter()
                    .zip(&resolved.layers)
                    .all(|(left, right)| {
                        left.model_layer == right.model_layer
                            && left.physical_layer == right.physical_layer
                    });
            if group.id.get() != resolved.group.get()
                || group.domain != resolved.domain
                || group.page_tokens != resolved.page_tokens
                || group.bytes_per_page != resolved.bytes_per_page
                || group.capacity_pages == 0
                || group.capacity_strategy.maximum_blocks() != group.capacity_pages
                || !matches!(
                    group.capacity_strategy,
                    CapacityStrategy::Fixed { .. } | CapacityStrategy::AdmissionGrowable { .. }
                )
                || !same_layers
            {
                return Err(invalid(
                    "physical KV allocation does not match its V2 resolved group",
                ));
            }
            if group.arena.model_instance != self.model_instance
                || group.arena.backend != self.backend
                || group.arena.device_ordinal != self.device_ordinal
                || group.arena.generation == 0
            {
                return Err(invalid(
                    "physical KV allocation group has an incompatible arena identity",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_against_allocation(
        &self,
        state_plan: &ResolvedStatePlan,
        allocation_plan: &StateRuntimeAllocationPlan,
    ) -> Result<()> {
        self.validate_against(state_plan)?;
        allocation_plan.validate_against(state_plan)?;
        if self.state_allocation_plan != allocation_plan.id
            || self.model_instance != allocation_plan.model_instance
        {
            return Err(invalid(
                "physical KV allocation belongs to a different runtime allocation plan",
            ));
        }
        for (group, resolved) in self.groups.iter().zip(&state_plan.paged_attention) {
            let capacity = allocation_plan.group_capacity(resolved.group, resolved.domain)?;
            if capacity.strategy != group.capacity_strategy
                || capacity.strategy.maximum_blocks() != group.capacity_pages
            {
                return Err(invalid(
                    "physical KV group capacity diverges from its runtime allocation plan",
                ));
            }
        }
        Ok(())
    }

    fn compute_fingerprint(&self) -> Result<KvPlanFingerprint> {
        #[derive(Serialize)]
        struct FingerprintPayload<'a> {
            model_instance: ModelInstanceId,
            backend: BackendKind,
            device_ordinal: Option<u32>,
            contract_fingerprint: &'a [u8; 32],
            state_plan_fingerprint: &'a [u8; 32],
            state_allocation_plan: StateAllocationPlanId,
            groups: &'a [ResolvedKvGroup],
        }

        let payload = FingerprintPayload {
            model_instance: self.model_instance,
            backend: self.backend,
            device_ordinal: self.device_ordinal,
            contract_fingerprint: &self.contract_fingerprint,
            state_plan_fingerprint: &self.state_plan_fingerprint,
            state_allocation_plan: self.state_allocation_plan,
            groups: &self.groups,
        };
        let encoded = serde_json::to_vec(&payload)
            .map_err(|error| invalid(format!("failed to encode physical KV plan: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(PLAN_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(KvPlanFingerprint::new(hasher.finalize().into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedKvGroup {
    pub id: KvGroupId,
    pub arena: KvArenaId,
    pub domain: StateDomainId,
    pub page_tokens: u32,
    pub capacity_pages: u32,
    pub(crate) capacity_strategy: CapacityStrategy,
    pub bytes_per_page: u64,
    pub layout: KvPhysicalLayout,
    pub storage: KvStorageFormat,
    pub layers: Vec<KvLayerBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvLayerBinding {
    pub model_layer: u32,
    pub physical_layer: u32,
}

impl From<StateLayerBinding> for KvLayerBinding {
    fn from(binding: StateLayerBinding) -> Self {
        Self {
            model_layer: binding.model_layer,
            physical_layer: binding.physical_layer,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPhysicalLayout {
    PageTokenHeadDim,
    PageHeadTokenDim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvStorageFormat {
    Dense { dtype: StateDType },
}

impl KvStorageFormat {
    pub const fn dtype(self) -> StateDType {
        match self {
            Self::Dense { dtype } => dtype,
        }
    }
}

fn hex_fingerprint(bytes: &[u8; 32], formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    for byte in bytes {
        write!(formatter, "{byte:02x}")?;
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{
        negotiate_state_plan, StateBackendPlanRequest, StateBackendRegistry,
    };
    use crate::kv::v2::{
        test_contract, GroupCapacityRequest, WorkspaceContract, WorkspacePlacement,
    };

    #[test]
    fn physical_allocation_is_derived_from_the_exact_state_plan() {
        let state_plan = negotiate_state_plan(
            &test_contract(),
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(16),
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let model_instance = ModelInstanceId::new(42);
        let allocation = StateRuntimeAllocationPlan::build_exact(
            &state_plan,
            model_instance,
            state_plan
                .paged_attention
                .iter()
                .map(|group| GroupCapacityRequest {
                    group: group.group,
                    domain: group.domain,
                    strategy: CapacityStrategy::Fixed { blocks: 128 },
                })
                .collect(),
            WorkspaceContract {
                fixed_bytes: 0,
                dimensions: vec![],
                terms: vec![],
                placement: WorkspacePlacement::Host,
                concurrency_slots: 1,
            },
            &StateBackendRegistry::new(BackendKind::Cpu, None).unwrap(),
        )
        .unwrap();
        let plan = ResolvedKvPlan::from_runtime_allocation(7, &state_plan, &allocation).unwrap();
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].capacity_pages, 128);
        assert_eq!(plan.groups[0].arena.generation, 7);
        assert_eq!(
            plan.state_plan_fingerprint,
            state_plan.fingerprint().bytes()
        );
        assert_eq!(plan.state_allocation_plan, allocation.id);

        let other = StateRuntimeAllocationPlan::build_exact(
            &state_plan,
            model_instance,
            state_plan
                .paged_attention
                .iter()
                .map(|group| GroupCapacityRequest {
                    group: group.group,
                    domain: group.domain,
                    strategy: CapacityStrategy::Fixed { blocks: 64 },
                })
                .collect(),
            WorkspaceContract {
                fixed_bytes: 0,
                dimensions: vec![],
                terms: vec![],
                placement: WorkspacePlacement::Host,
                concurrency_slots: 1,
            },
            &StateBackendRegistry::new(BackendKind::Cpu, None).unwrap(),
        )
        .unwrap();
        assert!(plan
            .validate_against_allocation(&state_plan, &other)
            .is_err());
    }

    #[test]
    fn admission_growable_capacity_survives_resolution_and_validation() {
        let state_plan = negotiate_state_plan(
            &test_contract(),
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(64),
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let model_instance = ModelInstanceId::new(43);
        let strategy = CapacityStrategy::AdmissionGrowable {
            initial_blocks: 64,
            growth_quantum: 64,
            max_blocks: 640,
        };
        let allocation = StateRuntimeAllocationPlan::build_exact(
            &state_plan,
            model_instance,
            state_plan
                .paged_attention
                .iter()
                .map(|group| GroupCapacityRequest {
                    group: group.group,
                    domain: group.domain,
                    strategy,
                })
                .collect(),
            WorkspaceContract {
                fixed_bytes: 0,
                dimensions: vec![],
                terms: vec![],
                placement: WorkspacePlacement::Host,
                concurrency_slots: 1,
            },
            &StateBackendRegistry::new(BackendKind::Cpu, None).unwrap(),
        )
        .unwrap();

        let plan = ResolvedKvPlan::from_runtime_allocation(9, &state_plan, &allocation).unwrap();
        assert_eq!(plan.groups[0].capacity_pages, 640);
        assert_eq!(plan.groups[0].capacity_strategy, strategy);
        plan.validate_against_allocation(&state_plan, &allocation)
            .unwrap();
    }
}
