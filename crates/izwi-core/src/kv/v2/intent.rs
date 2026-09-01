use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::capacity::{WorkspaceAxis, WorkspaceContract};
use super::contract::{
    AttentionPattern, BoundedShape, InferenceStateContract, PrefixPolicy, ShapeAxis,
    StateComponentId, StateDomainId, StateDomainSpec, StateGroupId,
};
use super::resolved::ResolvedStatePlan;
use super::resolved_domains::{align_bytes, ResolvedNonPagedDomainPlan, ResolvedTensorComponent};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AdapterStateIntent {
    pub(crate) domains: Vec<DomainStepIntent>,
    pub(crate) prefixes: Vec<PrefixIntent>,
    pub(crate) workspace: WorkspaceShapeInstantiation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DomainStepIntent {
    pub(crate) domain: StateDomainId,
    pub(crate) expected_cursor: u64,
    pub(crate) target_cursor: u64,
    pub(crate) update: StateUpdateKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum StateUpdateKind {
    PagedAppend {
        input_tokens: u64,
        read_visible_start: u64,
        commit_visible_start: u64,
    },
    StaticInitialize {
        source_identity: [u8; 32],
        components: Vec<ComponentShapeInstantiation>,
    },
    TensorReplace {
        components: Vec<ComponentShapeInstantiation>,
    },
    Append {
        steps: u64,
        components_per_step: Vec<ComponentShapeInstantiation>,
    },
    RingAdvance {
        steps: u64,
        components_per_step: Vec<ComponentShapeInstantiation>,
    },
    Reset,
    NoOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PrefixIntent {
    pub(crate) group: StateGroupId,
    pub(crate) matched_cursor: u64,
    pub(crate) canonical_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkspaceShapeInstantiation {
    pub(crate) dimensions: Vec<WorkspaceDimensionValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkspaceDimensionValue {
    pub(crate) axis: WorkspaceAxis,
    pub(crate) units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ComponentShapeInstantiation {
    pub(crate) component: StateComponentId,
    pub(crate) dimensions: Vec<ShapeDimensionValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ShapeDimensionValue {
    pub(crate) axis: ShapeAxis,
    pub(crate) units: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntentResourceUsage {
    pub(crate) state_update_bytes: u64,
    pub(crate) workspace_bytes: u64,
}

impl WorkspaceShapeInstantiation {
    pub(crate) fn bytes(&self, contract: &WorkspaceContract) -> Result<u64> {
        contract.bytes_for(
            &self
                .dimensions
                .iter()
                .map(|dimension| (dimension.axis, dimension.units))
                .collect::<Vec<_>>(),
        )
    }
}

impl AdapterStateIntent {
    pub(crate) fn validate(
        &self,
        contract: &InferenceStateContract,
        plan: &ResolvedStatePlan,
        workspace: &WorkspaceContract,
    ) -> Result<IntentResourceUsage> {
        contract.validate()?;
        if plan.contract_fingerprint != contract.fingerprint()? {
            return Err(invalid(
                "adapter intent state plan belongs to a different semantic contract",
            ));
        }
        let domains = contract
            .domains
            .iter()
            .map(|domain| (domain.id(), domain))
            .collect::<HashMap<_, _>>();
        if self.domains.len() != domains.len() {
            return Err(invalid(
                "adapter intent must cover every state domain exactly once",
            ));
        }
        let mut seen = HashSet::with_capacity(self.domains.len());
        let mut previous = None;
        let mut state_update_bytes = 0_u64;
        for intent in &self.domains {
            if previous.is_some_and(|previous| intent.domain <= previous)
                || !seen.insert(intent.domain)
            {
                return Err(invalid(
                    "adapter state intents require canonical unique domain order",
                ));
            }
            previous = Some(intent.domain);
            let semantic = domains
                .get(&intent.domain)
                .ok_or_else(|| invalid("adapter intent references an unknown state domain"))?;
            let non_paged = plan
                .non_paged
                .iter()
                .find(|candidate| candidate.domain() == intent.domain);
            let paged = plan
                .paged_attention
                .iter()
                .find(|candidate| candidate.domain == intent.domain);
            state_update_bytes = state_update_bytes
                .checked_add(intent.validate_against(semantic, paged, non_paged)?)
                .ok_or_else(|| invalid("adapter state update byte bound overflow"))?;
        }
        self.validate_prefixes(contract, plan, &seen)?;
        Ok(IntentResourceUsage {
            state_update_bytes,
            workspace_bytes: self.workspace.bytes(workspace)?,
        })
    }

    fn validate_prefixes(
        &self,
        contract: &InferenceStateContract,
        plan: &ResolvedStatePlan,
        covered_domains: &HashSet<StateDomainId>,
    ) -> Result<()> {
        let intents = self
            .domains
            .iter()
            .map(|intent| (intent.domain, intent))
            .collect::<HashMap<_, _>>();
        let groups = contract
            .groups
            .iter()
            .map(|group| (group.id, group))
            .collect::<HashMap<_, _>>();
        let mut seen = HashSet::with_capacity(self.prefixes.len());
        let mut previous = None;
        for prefix in &self.prefixes {
            if prefix.matched_cursor == 0
                || previous.is_some_and(|previous| prefix.group <= previous)
                || !seen.insert(prefix.group)
            {
                return Err(invalid("prefix intents require canonical unique groups"));
            }
            previous = Some(prefix.group);
            let group = groups
                .get(&prefix.group)
                .ok_or_else(|| invalid("prefix intent references an unknown state group"))?;
            if !group.prefix_shareable {
                return Err(invalid(
                    "prefix intent references a non-shareable state group",
                ));
            }
            for domain in &group.domains {
                if !covered_domains.contains(domain)
                    || intents[domain].expected_cursor < prefix.matched_cursor
                {
                    return Err(invalid(
                        "prefix intent exceeds a consistency-group domain cursor",
                    ));
                }
                let semantic = contract
                    .domains
                    .iter()
                    .find(|candidate| candidate.id() == *domain)
                    .ok_or_else(|| invalid("prefix group contains an unknown domain"))?;
                let alignment = match semantic.prefix_policy() {
                    PrefixPolicy::CommittedPages { .. } => u64::from(
                        plan.paged_attention
                            .iter()
                            .find(|candidate| candidate.domain == *domain)
                            .ok_or_else(|| invalid("prefix domain is missing its resolved plan"))?
                            .page_tokens,
                    ),
                    PrefixPolicy::CommittedSnapshots { interval_steps } => *interval_steps,
                    PrefixPolicy::Disabled => {
                        return Err(invalid(
                            "prefix group contains a domain with sharing disabled",
                        ));
                    }
                };
                if prefix.matched_cursor % alignment != 0 {
                    return Err(invalid(
                        "prefix cursor does not end at a committed physical boundary",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl DomainStepIntent {
    fn validate_against(
        &self,
        semantic: &StateDomainSpec,
        paged: Option<&super::resolved::ResolvedPagedAttentionGroup>,
        non_paged: Option<&ResolvedNonPagedDomainPlan>,
    ) -> Result<u64> {
        if self.domain != semantic.id() {
            return Err(invalid("adapter intent domain identity mismatch"));
        }
        let bytes = match (&self.update, semantic) {
            (
                StateUpdateKind::PagedAppend {
                    input_tokens,
                    read_visible_start,
                    commit_visible_start,
                },
                StateDomainSpec::PagedAttention(spec),
            ) => {
                require_delta(self.expected_cursor, self.target_cursor, *input_tokens)?;
                if *read_visible_start > *commit_visible_start
                    || *commit_visible_start > self.target_cursor
                {
                    return Err(invalid("paged visible windows are inconsistent"));
                }
                for layer in &spec.layers {
                    match layer.pattern {
                        AttentionPattern::Full
                            if *read_visible_start != 0 || *commit_visible_start != 0 =>
                        {
                            return Err(invalid(
                                "full attention cannot discard an earlier visible token",
                            ));
                        }
                        AttentionPattern::SlidingWindow { window_tokens }
                            if *read_visible_start
                                > self
                                    .expected_cursor
                                    .saturating_add(1)
                                    .saturating_sub(u64::from(window_tokens))
                                || *commit_visible_start
                                    > self
                                        .target_cursor
                                        .saturating_sub(u64::from(window_tokens)) =>
                        {
                            return Err(invalid(
                                "sliding attention intent discards tokens required by its window",
                            ));
                        }
                        _ => {}
                    }
                }
                let resolved = paged.ok_or_else(|| {
                    invalid("paged state intent is missing its resolved domain plan")
                })?;
                resolved
                    .bytes_per_page
                    .checked_div(u64::from(resolved.page_tokens))
                    .and_then(|per_token| per_token.checked_mul(*input_tokens))
                    .ok_or_else(|| invalid("paged state update byte bound overflow"))?
            }
            (
                StateUpdateKind::StaticInitialize {
                    source_identity,
                    components,
                },
                StateDomainSpec::StaticAttention(spec),
            ) => {
                if self.expected_cursor != 0
                    || self.target_cursor == 0
                    || self.target_cursor > spec.max_memory_tokens
                    || source_identity.iter().all(|byte| *byte == 0)
                    || !components.is_empty()
                {
                    return Err(invalid("invalid static-attention initialization"));
                }
                let Some(ResolvedNonPagedDomainPlan::StaticAttention(resolved)) = non_paged else {
                    return Err(invalid(
                        "static-attention intent is missing its resolved domain plan",
                    ));
                };
                let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
                    let layer_elements = u64::from(layer.kv_heads)
                        .checked_mul(
                            u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim),
                        )
                        .ok_or_else(|| invalid("static-attention update element overflow"))?;
                    total
                        .checked_add(layer_elements)
                        .ok_or_else(|| invalid("static-attention update element overflow"))
                })?;
                align_bytes(
                    resolved.storage.bytes_for_elements(
                        elements_per_token
                            .checked_mul(self.target_cursor)
                            .ok_or_else(|| invalid("static-attention update element overflow"))?,
                    )?,
                    resolved.alignment_bytes,
                )?
            }
            (
                StateUpdateKind::StaticInitialize {
                    source_identity,
                    components,
                },
                StateDomainSpec::StaticTensor(spec),
            ) => {
                if self.expected_cursor != 0
                    || self.target_cursor == 0
                    || source_identity.iter().all(|byte| *byte == 0)
                {
                    return Err(invalid("invalid static-tensor initialization"));
                }
                let Some(ResolvedNonPagedDomainPlan::StaticTensor(resolved)) = non_paged else {
                    return Err(invalid(
                        "static-tensor intent is missing its resolved domain plan",
                    ));
                };
                validate_component_shapes(components, &spec.components, &resolved.components)?
            }
            (StateUpdateKind::TensorReplace { components }, StateDomainSpec::Tensor(spec)) => {
                if self.target_cursor <= self.expected_cursor {
                    return Err(invalid("tensor replacement must advance its logical clock"));
                }
                let Some(ResolvedNonPagedDomainPlan::Tensor(resolved)) = non_paged else {
                    return Err(invalid(
                        "tensor-state intent is missing its resolved domain plan",
                    ));
                };
                validate_component_shapes(components, &spec.components, &resolved.components)?
            }
            (
                StateUpdateKind::Append {
                    steps,
                    components_per_step,
                },
                StateDomainSpec::Append(spec),
            ) => {
                require_delta(self.expected_cursor, self.target_cursor, *steps)?;
                if self.target_cursor > spec.max_steps {
                    return Err(invalid("append intent exceeds its bounded capacity"));
                }
                let Some(ResolvedNonPagedDomainPlan::Append(resolved)) = non_paged else {
                    return Err(invalid(
                        "append-state intent is missing its resolved domain plan",
                    ));
                };
                validate_component_shapes(
                    components_per_step,
                    &spec.components_per_step,
                    &resolved.components_per_step,
                )?
                .checked_mul(*steps)
                .ok_or_else(|| invalid("append-state update byte bound overflow"))?
            }
            (
                StateUpdateKind::RingAdvance {
                    steps,
                    components_per_step,
                },
                StateDomainSpec::Ring(spec),
            ) => {
                require_delta(self.expected_cursor, self.target_cursor, *steps)?;
                let Some(ResolvedNonPagedDomainPlan::Ring(resolved)) = non_paged else {
                    return Err(invalid(
                        "ring-state intent is missing its resolved domain plan",
                    ));
                };
                validate_component_shapes(
                    components_per_step,
                    &spec.components_per_step,
                    &resolved.components_per_step,
                )?
                .checked_mul(*steps)
                .ok_or_else(|| invalid("ring-state update byte bound overflow"))?
            }
            (StateUpdateKind::Reset, StateDomainSpec::PagedAttention(_))
            | (StateUpdateKind::Reset, StateDomainSpec::Tensor(_))
            | (StateUpdateKind::Reset, StateDomainSpec::Append(_))
            | (StateUpdateKind::Reset, StateDomainSpec::Ring(_)) => {
                if self.target_cursor != 0 {
                    return Err(invalid("state reset target cursor must be zero"));
                }
                0
            }
            (StateUpdateKind::NoOp, _) => {
                if self.expected_cursor != self.target_cursor {
                    return Err(invalid("no-op state intent changed its cursor"));
                }
                0
            }
            _ => {
                return Err(invalid(
                    "state update kind does not match its semantic domain",
                ));
            }
        };
        Ok(bytes)
    }
}

fn validate_component_shapes(
    instantiated: &[ComponentShapeInstantiation],
    semantic: &[super::contract::TensorComponentSpec],
    resolved: &[ResolvedTensorComponent],
) -> Result<u64> {
    if instantiated.len() != semantic.len() || resolved.len() != semantic.len() {
        return Err(invalid(
            "runtime tensor shape must cover every component exactly once",
        ));
    }
    let mut total = 0_u64;
    for ((instantiated, semantic), resolved) in instantiated.iter().zip(semantic).zip(resolved) {
        if instantiated.component != semantic.id || resolved.component != semantic.id {
            return Err(invalid("runtime tensor component identity mismatch"));
        }
        let elements = validate_shape(&instantiated.dimensions, &semantic.shape)?;
        let bytes = align_bytes(
            resolved.storage.bytes_for_elements(elements)?,
            resolved.alignment_bytes,
        )?;
        if bytes > resolved.maximum_bytes {
            return Err(invalid(
                "runtime tensor shape exceeds its resolved component byte bound",
            ));
        }
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("runtime tensor update byte bound overflow"))?;
    }
    Ok(total)
}

fn validate_shape(instantiated: &[ShapeDimensionValue], semantic: &BoundedShape) -> Result<u64> {
    if instantiated.len() != semantic.dimensions.len() {
        return Err(invalid("runtime tensor shape rank mismatch"));
    }
    instantiated
        .iter()
        .zip(&semantic.dimensions)
        .try_fold(1_u64, |elements, (actual, expected)| {
            if actual.axis != expected.axis || !expected.extent.accepts(actual.units) {
                return Err(invalid(
                    "runtime tensor extent violates its semantic shape bound",
                ));
            }
            elements
                .checked_mul(actual.units)
                .ok_or_else(|| invalid("runtime tensor element count overflow"))
        })
}

fn require_delta(expected: u64, target: u64, delta: u64) -> Result<()> {
    if delta == 0 || expected.checked_add(delta) != Some(target) {
        return Err(invalid("state update cursor delta is invalid"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::BackendKind;
    use crate::kv::v2::resolved::test_plan;
    use crate::kv::v2::test_contract;
    use crate::kv::v2::{
        BoundedShape, CheckpointPolicy, NonPagedStateOperationQuery,
        NonPagedStateOperationRegistry, OperationAbi, PagedAttentionOperationQuery,
        PlacementPolicy, PrefixPolicy, RegisteredOperationId, ResolvedPlacement,
        ResolvedTensorStatePlan, ShapeDimension, ShapeExtent, StateClock, StateDType,
        StateDomainHeader, StateGroupSpec, StateOperationRegistry, StateScope, StateStorageFormat,
        TensorComponentSpec, TensorPhysicalLayout, TensorRole, TensorStateDomainSpec,
        TensorStateOperationSet, CURRENT_INFERENCE_STATE_ABI,
    };
    use crate::kv::v2::{WorkspaceDimensionBound, WorkspacePlacement, WorkspaceTerm};

    fn workspace() -> WorkspaceContract {
        WorkspaceContract {
            fixed_bytes: 64,
            dimensions: vec![WorkspaceDimensionBound {
                axis: WorkspaceAxis::InputTokens,
                max_units: 32,
            }],
            terms: vec![WorkspaceTerm {
                factors: vec![WorkspaceAxis::InputTokens],
                bytes_per_element: 16,
            }],
            placement: WorkspacePlacement::BackendLocal,
            concurrency_slots: 1,
        }
    }

    fn valid_intent() -> AdapterStateIntent {
        AdapterStateIntent {
            domains: vec![DomainStepIntent {
                domain: StateDomainId::new(1),
                expected_cursor: 16,
                target_cursor: 20,
                update: StateUpdateKind::PagedAppend {
                    input_tokens: 4,
                    read_visible_start: 0,
                    commit_visible_start: 0,
                },
            }],
            prefixes: vec![PrefixIntent {
                group: StateGroupId::new(1),
                matched_cursor: 16,
                canonical_digest: [7; 32],
            }],
            workspace: WorkspaceShapeInstantiation {
                dimensions: vec![WorkspaceDimensionValue {
                    axis: WorkspaceAxis::InputTokens,
                    units: 4,
                }],
            },
        }
    }

    struct TensorRegistry;

    impl NonPagedStateOperationRegistry for TensorRegistry {
        fn supports_non_paged(&self, query: &NonPagedStateOperationQuery<'_>) -> bool {
            query.backend == BackendKind::Cpu
                && matches!(query.resolved, ResolvedNonPagedDomainPlan::Tensor(_))
        }
    }

    impl StateOperationRegistry for TensorRegistry {
        fn supports_paged_attention(&self, _query: &PagedAttentionOperationQuery<'_>) -> bool {
            false
        }
    }

    fn tensor_operation(name: &str) -> RegisteredOperationId {
        RegisteredOperationId::new(name, OperationAbi::new(1))
    }

    fn tensor_contract_and_plan() -> (InferenceStateContract, ResolvedStatePlan) {
        let component = TensorComponentSpec {
            id: StateComponentId::new(1),
            role: TensorRole::RecurrentHidden,
            shape: BoundedShape {
                dimensions: vec![ShapeDimension {
                    axis: ShapeAxis::Hidden,
                    extent: ShapeExtent::RuntimeBounded { min: 2, max: 8 },
                }],
            },
            accepted_dtypes: vec![StateDType::F16],
        };
        let contract = InferenceStateContract {
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
                components: vec![component],
            })],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        };
        let plan = ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![],
            vec![ResolvedNonPagedDomainPlan::Tensor(
                ResolvedTensorStatePlan {
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
                        maximum_bytes: 16,
                    }],
                    maximum_bytes: 16,
                    operations: TensorStateOperationSet {
                        initialize: tensor_operation("tensor_initialize"),
                        read: tensor_operation("tensor_read"),
                        stage_replace: tensor_operation("tensor_replace"),
                        reset: tensor_operation("tensor_reset"),
                    },
                },
            )],
            &TensorRegistry,
        )
        .unwrap();
        (contract, plan)
    }

    #[test]
    fn intent_is_complete_typed_and_workspace_bounded() {
        let contract = test_contract();
        let plan = test_plan(&contract);
        assert_eq!(
            valid_intent()
                .validate(&contract, &plan, &workspace())
                .unwrap(),
            IntentResourceUsage {
                state_update_bytes: 4 * (16 * 4 * (64 + 64) * 2 / 16),
                workspace_bytes: 64 + 4 * 16,
            }
        );

        let mut missing = valid_intent();
        missing.domains.clear();
        assert!(missing.validate(&contract, &plan, &workspace()).is_err());

        let mut wrong_kind = valid_intent();
        wrong_kind.domains[0].update = StateUpdateKind::RingAdvance {
            steps: 4,
            components_per_step: vec![],
        };
        assert!(wrong_kind.validate(&contract, &plan, &workspace()).is_err());

        let mut oversized = valid_intent();
        oversized.workspace.dimensions[0].units = 33;
        assert!(oversized.validate(&contract, &plan, &workspace()).is_err());

        let mut partial_prefix = valid_intent();
        partial_prefix.prefixes[0].matched_cursor = 15;
        assert!(partial_prefix
            .validate(&contract, &plan, &workspace())
            .is_err());

        let mut truncated_full_attention = valid_intent();
        let StateUpdateKind::PagedAppend {
            read_visible_start, ..
        } = &mut truncated_full_attention.domains[0].update
        else {
            unreachable!()
        };
        *read_visible_start = 1;
        assert!(truncated_full_attention
            .validate(&contract, &plan, &workspace())
            .is_err());
    }

    #[test]
    fn runtime_tensor_extents_produce_exact_aligned_update_bytes() {
        let (contract, plan) = tensor_contract_and_plan();
        let intent = AdapterStateIntent {
            domains: vec![DomainStepIntent {
                domain: StateDomainId::new(1),
                expected_cursor: 4,
                target_cursor: 5,
                update: StateUpdateKind::TensorReplace {
                    components: vec![ComponentShapeInstantiation {
                        component: StateComponentId::new(1),
                        dimensions: vec![ShapeDimensionValue {
                            axis: ShapeAxis::Hidden,
                            units: 3,
                        }],
                    }],
                },
            }],
            prefixes: vec![],
            workspace: WorkspaceShapeInstantiation {
                dimensions: vec![WorkspaceDimensionValue {
                    axis: WorkspaceAxis::InputTokens,
                    units: 1,
                }],
            },
        };
        assert_eq!(
            intent.validate(&contract, &plan, &workspace()).unwrap(),
            IntentResourceUsage {
                state_update_bytes: 8,
                workspace_bytes: 80,
            }
        );

        let mut oversized = intent;
        let StateUpdateKind::TensorReplace { components } = &mut oversized.domains[0].update else {
            unreachable!()
        };
        components[0].dimensions[0].units = 9;
        assert!(oversized.validate(&contract, &plan, &workspace()).is_err());
    }

    #[test]
    fn chunked_sliding_attention_keeps_history_for_the_earliest_query() {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        domain.layers[0].pattern = AttentionPattern::SlidingWindow { window_tokens: 8 };
        let plan = test_plan(&contract);
        let mut intent = valid_intent();
        let StateUpdateKind::PagedAppend {
            read_visible_start,
            commit_visible_start,
            ..
        } = &mut intent.domains[0].update
        else {
            unreachable!()
        };
        *read_visible_start = 9;
        *commit_visible_start = 12;
        intent.validate(&contract, &plan, &workspace()).unwrap();

        let StateUpdateKind::PagedAppend {
            read_visible_start, ..
        } = &mut intent.domains[0].update
        else {
            unreachable!()
        };
        *read_visible_start = 10;
        assert!(intent.validate(&contract, &plan, &workspace()).is_err());
    }
}
