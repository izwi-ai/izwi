use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::backends::BackendKind;
use crate::error::{Error, Result};

use super::contract::{
    AppendStateDomainSpec, RingStateDomainSpec, StateComponentId, StateDomainId, StateDomainSpec,
    StateGroupId, StaticAttentionDomainSpec, StaticTensorDomainSpec, TensorComponentSpec,
    TensorStateDomainSpec,
};
use super::resolved::{
    RegisteredOperationId, ResolvedPlacement, StateLayerBinding, StateStorageFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TensorPhysicalLayout {
    ContiguousRowMajor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedTensorComponent {
    pub(crate) component: StateComponentId,
    pub(crate) layout: TensorPhysicalLayout,
    pub(crate) storage: StateStorageFormat,
    pub(crate) alignment_bytes: u64,
    pub(crate) maximum_bytes: u64,
}

impl ResolvedTensorComponent {
    fn validate_against(&self, semantic: &TensorComponentSpec) -> Result<()> {
        if self.component != semantic.id {
            return Err(invalid("resolved tensor component identity mismatch"));
        }
        self.storage.validate()?;
        if !semantic.accepted_dtypes.contains(&self.storage.dtype()) {
            return Err(invalid("resolved tensor component dtype was not accepted"));
        }
        validate_alignment(self.alignment_bytes)?;
        let expected = self
            .storage
            .bytes_for_elements(semantic.shape.maximum_elements()?)?;
        let expected = align_bytes(expected, self.alignment_bytes)?;
        if self.maximum_bytes != expected || expected == 0 {
            return Err(invalid("resolved tensor component byte bound is incorrect"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StaticAttentionOperationSet {
    pub(crate) install: RegisteredOperationId,
    pub(crate) attend: RegisteredOperationId,
}

impl StaticAttentionOperationSet {
    fn validate(&self) -> Result<()> {
        self.install.validate()?;
        self.attend.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StaticTensorOperationSet {
    pub(crate) install: RegisteredOperationId,
    pub(crate) read: RegisteredOperationId,
}

impl StaticTensorOperationSet {
    fn validate(&self) -> Result<()> {
        self.install.validate()?;
        self.read.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TensorStateOperationSet {
    pub(crate) initialize: RegisteredOperationId,
    pub(crate) read: RegisteredOperationId,
    pub(crate) stage_replace: RegisteredOperationId,
    pub(crate) reset: RegisteredOperationId,
}

impl TensorStateOperationSet {
    fn validate(&self) -> Result<()> {
        self.initialize.validate()?;
        self.read.validate()?;
        self.stage_replace.validate()?;
        self.reset.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AppendStateOperationSet {
    pub(crate) initialize: RegisteredOperationId,
    pub(crate) read: RegisteredOperationId,
    pub(crate) append: RegisteredOperationId,
    pub(crate) reset: RegisteredOperationId,
}

impl AppendStateOperationSet {
    fn validate(&self) -> Result<()> {
        self.initialize.validate()?;
        self.read.validate()?;
        self.append.validate()?;
        self.reset.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RingStateOperationSet {
    pub(crate) initialize: RegisteredOperationId,
    pub(crate) read: RegisteredOperationId,
    pub(crate) advance: RegisteredOperationId,
    pub(crate) reset: RegisteredOperationId,
}

impl RingStateOperationSet {
    fn validate(&self) -> Result<()> {
        self.initialize.validate()?;
        self.read.validate()?;
        self.advance.validate()?;
        self.reset.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedStaticAttentionPlan {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) placement: ResolvedPlacement,
    pub(crate) layers: Vec<StateLayerBinding>,
    pub(crate) storage: StateStorageFormat,
    pub(crate) layout: TensorPhysicalLayout,
    pub(crate) alignment_bytes: u64,
    pub(crate) maximum_bytes: u64,
    pub(crate) operations: StaticAttentionOperationSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedStaticTensorPlan {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) placement: ResolvedPlacement,
    pub(crate) components: Vec<ResolvedTensorComponent>,
    pub(crate) maximum_bytes: u64,
    pub(crate) operations: StaticTensorOperationSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedTensorStatePlan {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) placement: ResolvedPlacement,
    pub(crate) components: Vec<ResolvedTensorComponent>,
    pub(crate) maximum_bytes: u64,
    pub(crate) operations: TensorStateOperationSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedAppendStatePlan {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) placement: ResolvedPlacement,
    pub(crate) components_per_step: Vec<ResolvedTensorComponent>,
    pub(crate) maximum_bytes: u64,
    pub(crate) operations: AppendStateOperationSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedRingStatePlan {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) placement: ResolvedPlacement,
    pub(crate) components_per_step: Vec<ResolvedTensorComponent>,
    pub(crate) maximum_bytes: u64,
    pub(crate) operations: RingStateOperationSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ResolvedNonPagedDomainPlan {
    StaticAttention(ResolvedStaticAttentionPlan),
    StaticTensor(ResolvedStaticTensorPlan),
    Tensor(ResolvedTensorStatePlan),
    Append(ResolvedAppendStatePlan),
    Ring(ResolvedRingStatePlan),
}

impl ResolvedNonPagedDomainPlan {
    pub(crate) const fn group(&self) -> StateGroupId {
        match self {
            Self::StaticAttention(plan) => plan.group,
            Self::StaticTensor(plan) => plan.group,
            Self::Tensor(plan) => plan.group,
            Self::Append(plan) => plan.group,
            Self::Ring(plan) => plan.group,
        }
    }

    pub(crate) const fn domain(&self) -> StateDomainId {
        match self {
            Self::StaticAttention(plan) => plan.domain,
            Self::StaticTensor(plan) => plan.domain,
            Self::Tensor(plan) => plan.domain,
            Self::Append(plan) => plan.domain,
            Self::Ring(plan) => plan.domain,
        }
    }

    pub(crate) const fn placement(&self) -> ResolvedPlacement {
        match self {
            Self::StaticAttention(plan) => plan.placement,
            Self::StaticTensor(plan) => plan.placement,
            Self::Tensor(plan) => plan.placement,
            Self::Append(plan) => plan.placement,
            Self::Ring(plan) => plan.placement,
        }
    }

    pub(crate) const fn maximum_bytes(&self) -> u64 {
        match self {
            Self::StaticAttention(plan) => plan.maximum_bytes,
            Self::StaticTensor(plan) => plan.maximum_bytes,
            Self::Tensor(plan) => plan.maximum_bytes,
            Self::Append(plan) => plan.maximum_bytes,
            Self::Ring(plan) => plan.maximum_bytes,
        }
    }

    pub(crate) fn validate_against(
        &self,
        semantic: &StateDomainSpec,
        backend: BackendKind,
        device_ordinal: Option<u32>,
        registry: &dyn NonPagedStateOperationRegistry,
    ) -> Result<()> {
        semantic.validate()?;
        match (self, semantic) {
            (Self::StaticAttention(plan), StateDomainSpec::StaticAttention(spec)) => {
                plan.validate_against(spec)?
            }
            (Self::StaticTensor(plan), StateDomainSpec::StaticTensor(spec)) => {
                plan.validate_against(spec)?
            }
            (Self::Tensor(plan), StateDomainSpec::Tensor(spec)) => plan.validate_against(spec)?,
            (Self::Append(plan), StateDomainSpec::Append(spec)) => plan.validate_against(spec)?,
            (Self::Ring(plan), StateDomainSpec::Ring(spec)) => plan.validate_against(spec)?,
            _ => return Err(invalid("resolved and semantic state domain kinds differ")),
        }
        if !registry.supports_non_paged(&NonPagedStateOperationQuery {
            backend,
            device_ordinal,
            resolved: self,
            semantic,
        }) {
            return Err(invalid(
                "selected backend did not attest the resolved non-paged operation set",
            ));
        }
        Ok(())
    }
}

pub(crate) struct NonPagedStateOperationQuery<'a> {
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) resolved: &'a ResolvedNonPagedDomainPlan,
    pub(crate) semantic: &'a StateDomainSpec,
}

pub(crate) trait NonPagedStateOperationRegistry {
    fn supports_non_paged(&self, query: &NonPagedStateOperationQuery<'_>) -> bool;
}

impl ResolvedStaticAttentionPlan {
    fn validate_against(&self, spec: &StaticAttentionDomainSpec) -> Result<()> {
        common(
            self.domain,
            self.placement,
            spec.header.id,
            spec.header.placement,
        )?;
        self.storage.validate()?;
        validate_alignment(self.alignment_bytes)?;
        if !spec.accepted_dtypes.contains(&self.storage.dtype())
            || self.layers.len() != spec.layers.len()
        {
            return Err(invalid(
                "invalid static-attention storage or layer coverage",
            ));
        }
        self.operations.validate()?;
        let semantic = spec
            .layers
            .iter()
            .map(|layer| (layer.model_layer, layer))
            .collect::<HashMap<_, _>>();
        let mut physical = HashSet::with_capacity(self.layers.len());
        let mut elements = 0_u64;
        let mut previous = None;
        for binding in &self.layers {
            if previous.is_some_and(|value| binding.model_layer <= value)
                || !physical.insert(binding.physical_layer)
            {
                return Err(invalid("static-attention layer bindings are not canonical"));
            }
            previous = Some(binding.model_layer);
            let layer = semantic
                .get(&binding.model_layer)
                .ok_or_else(|| invalid("unknown static-attention layer binding"))?;
            let per_token = u64::from(layer.kv_heads)
                .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
                .ok_or_else(|| invalid("static-attention element count overflow"))?;
            elements = elements
                .checked_add(
                    per_token
                        .checked_mul(spec.max_memory_tokens)
                        .ok_or_else(|| invalid("static-attention element count overflow"))?,
                )
                .ok_or_else(|| invalid("static-attention element count overflow"))?;
        }
        let expected = align_bytes(
            self.storage.bytes_for_elements(elements)?,
            self.alignment_bytes,
        )?;
        if self.maximum_bytes != expected || expected == 0 {
            return Err(invalid("static-attention byte bound is incorrect"));
        }
        Ok(())
    }
}

impl ResolvedStaticTensorPlan {
    fn validate_against(&self, spec: &StaticTensorDomainSpec) -> Result<()> {
        common(
            self.domain,
            self.placement,
            spec.header.id,
            spec.header.placement,
        )?;
        validate_components(&self.components, &spec.components)?;
        validate_aggregate_bytes(self.maximum_bytes, &self.components)?;
        self.operations.validate()
    }
}

impl ResolvedTensorStatePlan {
    fn validate_against(&self, spec: &TensorStateDomainSpec) -> Result<()> {
        common(
            self.domain,
            self.placement,
            spec.header.id,
            spec.header.placement,
        )?;
        validate_components(&self.components, &spec.components)?;
        validate_aggregate_bytes(self.maximum_bytes, &self.components)?;
        self.operations.validate()
    }
}

impl ResolvedAppendStatePlan {
    fn validate_against(&self, spec: &AppendStateDomainSpec) -> Result<()> {
        common(
            self.domain,
            self.placement,
            spec.header.id,
            spec.header.placement,
        )?;
        validate_components(&self.components_per_step, &spec.components_per_step)?;
        let expected = component_bytes(&self.components_per_step)?
            .checked_mul(spec.max_steps)
            .ok_or_else(|| invalid("append-state byte bound overflow"))?;
        if self.maximum_bytes != expected || expected == 0 {
            return Err(invalid("append-state byte bound is incorrect"));
        }
        self.operations.validate()
    }
}

impl ResolvedRingStatePlan {
    fn validate_against(&self, spec: &RingStateDomainSpec) -> Result<()> {
        common(
            self.domain,
            self.placement,
            spec.header.id,
            spec.header.placement,
        )?;
        validate_components(&self.components_per_step, &spec.components_per_step)?;
        let expected = component_bytes(&self.components_per_step)?
            .checked_mul(spec.capacity_steps)
            .ok_or_else(|| invalid("ring-state byte bound overflow"))?;
        if self.maximum_bytes != expected || expected == 0 {
            return Err(invalid("ring-state byte bound is incorrect"));
        }
        self.operations.validate()
    }
}

fn common(
    domain: StateDomainId,
    placement: ResolvedPlacement,
    semantic_domain: StateDomainId,
    policy: super::contract::PlacementPolicy,
) -> Result<()> {
    if domain != semantic_domain {
        return Err(invalid("resolved state domain identity mismatch"));
    }
    placement.validate_against(policy)
}

fn validate_components(
    resolved: &[ResolvedTensorComponent],
    semantic: &[TensorComponentSpec],
) -> Result<()> {
    if resolved.len() != semantic.len() {
        return Err(invalid("resolved tensor component coverage is incomplete"));
    }
    for (resolved, semantic) in resolved.iter().zip(semantic) {
        resolved.validate_against(semantic)?;
    }
    Ok(())
}

fn component_bytes(components: &[ResolvedTensorComponent]) -> Result<u64> {
    components.iter().try_fold(0_u64, |bytes, component| {
        bytes
            .checked_add(component.maximum_bytes)
            .ok_or_else(|| invalid("resolved component byte bound overflow"))
    })
}

fn validate_aggregate_bytes(aggregate: u64, components: &[ResolvedTensorComponent]) -> Result<()> {
    if aggregate == 0 || aggregate != component_bytes(components)? {
        return Err(invalid("resolved tensor aggregate byte bound is incorrect"));
    }
    Ok(())
}

fn validate_alignment(alignment: u64) -> Result<()> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid(
            "resolved tensor alignment must be a non-zero power of two",
        ));
    }
    Ok(())
}

pub(crate) fn align_bytes(bytes: u64, alignment: u64) -> Result<u64> {
    validate_alignment(alignment)?;
    bytes
        .checked_add(alignment - 1)
        .map(|bytes| bytes & !(alignment - 1))
        .ok_or_else(|| invalid("resolved tensor aligned byte bound overflow"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backends::BackendKind;
    use crate::kv::v2::contract::{
        BoundedShape, CheckpointPolicy, InferenceStateContract, PlacementPolicy, PrefixPolicy,
        ShapeAxis, ShapeDimension, ShapeExtent, StateClock, StateDType, StateDomainHeader,
        StateGroupSpec, StateScope, TensorRole, CURRENT_INFERENCE_STATE_ABI,
    };
    use crate::kv::v2::resolved::{
        OperationAbi, PagedAttentionOperationQuery, ResolvedStatePlan, StateOperationRegistry,
    };

    pub(crate) struct TestRegistry;

    impl NonPagedStateOperationRegistry for TestRegistry {
        fn supports_non_paged(&self, query: &NonPagedStateOperationQuery<'_>) -> bool {
            query.backend == BackendKind::Cpu
                && query.device_ordinal.is_none()
                && matches!(query.resolved, ResolvedNonPagedDomainPlan::Tensor(_))
                && matches!(query.semantic, StateDomainSpec::Tensor(_))
        }
    }

    impl StateOperationRegistry for TestRegistry {
        fn supports_paged_attention(&self, _query: &PagedAttentionOperationQuery<'_>) -> bool {
            false
        }
    }

    fn operation(name: &str) -> RegisteredOperationId {
        RegisteredOperationId::new(name, OperationAbi::new(1))
    }

    pub(crate) fn contract() -> InferenceStateContract {
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
                            extent: ShapeExtent::RuntimeBounded { min: 1, max: 8 },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F16],
                }],
            })],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        }
    }

    pub(crate) fn tensor_plan(maximum_bytes: u64) -> ResolvedNonPagedDomainPlan {
        ResolvedNonPagedDomainPlan::Tensor(ResolvedTensorStatePlan {
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            placement: ResolvedPlacement::BackendLocal,
            components: vec![ResolvedTensorComponent {
                component: StateComponentId::new(1),
                layout: TensorPhysicalLayout::ContiguousRowMajor,
                storage: StateStorageFormat::Dense {
                    dtype: StateDType::F16,
                },
                alignment_bytes: 8,
                maximum_bytes,
            }],
            maximum_bytes,
            operations: TensorStateOperationSet {
                initialize: operation("tensor_state_initialize"),
                read: operation("tensor_state_read"),
                stage_replace: operation("tensor_state_stage_replace"),
                reset: operation("tensor_state_reset"),
            },
        })
    }

    #[test]
    fn composite_plan_resolves_non_paged_domains_without_paged_fields() {
        let contract = contract();
        let plan = ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![],
            vec![tensor_plan(16)],
            &TestRegistry,
        )
        .unwrap();
        assert!(plan.paged_attention.is_empty());
        assert_eq!(plan.non_paged.len(), 1);

        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![],
            vec![tensor_plan(15)],
            &TestRegistry,
        )
        .is_err());
    }
}
