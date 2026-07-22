use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::engine::{StageDescriptor, StageId};
use crate::error::{Error, Result};

use super::capacity::{WorkspaceDimensionBound, WorkspaceTerm};
use super::contract::{
    InferenceStateAbi, InferenceStateContract, PlacementPolicy, StateDomainSpec, StateScope,
    CURRENT_INFERENCE_STATE_ABI,
};

const CAPABILITY_DESCRIPTOR_FINGERPRINT_DOMAIN: &[u8] =
    b"izwi.inference-state.capability-descriptor.v2\0";
const STAGE_GRAPH_FINGERPRINT_DOMAIN: &[u8] = b"izwi.execution-stage-graph.v2\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CapabilityStateDescriptorV2 {
    pub(crate) abi: InferenceStateAbi,
    pub(crate) retained: RetainedStateCapability,
    pub(crate) invocation: InvocationWorkspaceSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum RetainedStateCapability {
    Stateless,
    Managed { contract: InferenceStateContract },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum InvocationWorkspaceSet {
    None {
        stage_graph_fingerprints: Vec<[u8; 32]>,
    },
    Bounded {
        profiles: Vec<InvocationWorkspaceProfile>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InvocationWorkspaceProfile {
    pub(crate) stage_graph_fingerprint: [u8; 32],
    pub(crate) stages: Vec<InvocationStageWorkspace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InvocationStageWorkspace {
    pub(crate) stage: StageId,
    /// Empty is an affirmative zero-workspace declaration for this stage.
    pub(crate) domains: Vec<InvocationWorkspaceDomain>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InvocationWorkspaceDomain {
    pub(crate) state: StateDomainSpec,
    pub(crate) placement: PlacementPolicy,
    pub(crate) formula: WorkspaceFormula,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkspaceFormula {
    pub(crate) fixed_bytes: u64,
    pub(crate) dimensions: Vec<WorkspaceDimensionBound>,
    pub(crate) terms: Vec<WorkspaceTerm>,
}

impl CapabilityStateDescriptorV2 {
    pub(crate) fn validate_against_stages(&self, stages: &[StageDescriptor]) -> Result<()> {
        if self.abi != CURRENT_INFERENCE_STATE_ABI {
            return Err(invalid("unsupported capability state descriptor ABI"));
        }
        match &self.retained {
            RetainedStateCapability::Stateless => {}
            RetainedStateCapability::Managed { contract } => {
                contract.validate()?;
                if contract
                    .domains
                    .iter()
                    .any(|domain| domain.scope() != StateScope::Retained)
                {
                    return Err(invalid(
                        "managed retained-state contract contains an invocation-scoped domain",
                    ));
                }
            }
        }
        validate_stage_ids(stages)?;
        self.invocation.validate_against_stages(stages)
    }

    pub(crate) fn fingerprint(&self, stages: &[StageDescriptor]) -> Result<[u8; 32]> {
        self.validate_against_stages(stages)?;
        #[derive(Serialize)]
        struct Payload<'a> {
            descriptor: &'a CapabilityStateDescriptorV2,
            stages: &'a [StageDescriptor],
        }
        let encoded = serde_json::to_vec(&Payload {
            descriptor: self,
            stages,
        })
        .map_err(|error| {
            invalid(format!(
                "failed to encode capability state descriptor: {error}"
            ))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(CAPABILITY_DESCRIPTOR_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }

    pub(crate) const fn is_stateless(&self) -> bool {
        matches!(self.retained, RetainedStateCapability::Stateless)
    }

    pub(crate) fn has_zero_invocation_workspace_for(
        &self,
        stages: &[StageDescriptor],
    ) -> Result<bool> {
        self.validate_against_stages(stages)?;
        Ok(matches!(
            self.invocation,
            InvocationWorkspaceSet::None { .. }
        ))
    }

    #[cfg(test)]
    pub(crate) fn managed_for_stages_test(
        contract: InferenceStateContract,
        stages: &[StageDescriptor],
    ) -> Self {
        let invocation =
            if stages.iter().all(|stage| stage.max_workspace_bytes == 0) {
                InvocationWorkspaceSet::None {
                    stage_graph_fingerprints: vec![
                        stage_graph_fingerprint(stages).expect("test stages must serialize")
                    ],
                }
            } else {
                InvocationWorkspaceSet::Bounded {
                    profiles: vec![InvocationWorkspaceProfile {
                        stage_graph_fingerprint: stage_graph_fingerprint(stages)
                            .expect("test stages must serialize"),
                        stages: stages
                            .iter()
                            .map(|stage| InvocationStageWorkspace {
                                stage: stage.id,
                                domains: (stage.max_workspace_bytes > 0)
                                    .then(|| InvocationWorkspaceDomain {
                                        state: StateDomainSpec::StaticTensor(
                                            super::contract::StaticTensorDomainSpec {
                                                header: super::contract::StateDomainHeader {
                                                    id: super::contract::StateDomainId::new(
                                                        stage.id.get(),
                                                    ),
                                                    scope: StateScope::Invocation,
                                                    clock:
                                                        super::contract::StateClock::DecoderTokens,
                                                    placement: PlacementPolicy::BackendLocal,
                                                    prefix: super::contract::PrefixPolicy::Disabled,
                                                    checkpoint:
                                                        super::contract::CheckpointPolicy::None,
                                                },
                                                components:
                                                    vec![super::contract::TensorComponentSpec {
                                            id: super::contract::StateComponentId::new(1),
                                            role: super::contract::TensorRole::Control,
                                            shape: super::contract::BoundedShape {
                                                dimensions: vec![super::contract::ShapeDimension {
                                                    axis: super::contract::ShapeAxis::Hidden,
                                                    extent: super::contract::ShapeExtent::Fixed {
                                                        value: 1,
                                                    },
                                                }],
                                            },
                                            accepted_dtypes: vec![super::contract::StateDType::I8],
                                        }],
                                            },
                                        ),
                                        placement: PlacementPolicy::BackendLocal,
                                        formula: WorkspaceFormula {
                                            fixed_bytes: stage.max_workspace_bytes,
                                            dimensions: vec![],
                                            terms: vec![],
                                        },
                                    })
                                    .into_iter()
                                    .collect(),
                            })
                            .collect(),
                    }],
                }
            };
        Self {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Managed { contract },
            invocation,
        }
    }

    #[cfg(test)]
    pub(crate) fn stateless_for_test() -> Self {
        Self {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::None {
                stage_graph_fingerprints: vec![],
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn stateless_for_stages_test(stages: &[StageDescriptor]) -> Self {
        Self::stateless_for_stage_graphs_test(&[stages])
    }

    #[cfg(test)]
    pub(crate) fn stateless_for_stage_graphs_test(stage_graphs: &[&[StageDescriptor]]) -> Self {
        let mut stage_graph_fingerprints = stage_graphs
            .iter()
            .map(|stages| stage_graph_fingerprint(stages).expect("test stages must serialize"))
            .collect::<Vec<_>>();
        stage_graph_fingerprints.sort_unstable();
        stage_graph_fingerprints.dedup();
        Self {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::None {
                stage_graph_fingerprints,
            },
        }
    }
}

impl InvocationWorkspaceSet {
    fn validate_against_stages(&self, execution_stages: &[StageDescriptor]) -> Result<()> {
        match self {
            Self::None {
                stage_graph_fingerprints,
            } => {
                if execution_stages
                    .iter()
                    .any(|stage| stage.max_workspace_bytes != 0)
                {
                    return Err(invalid(
                        "zero invocation workspace contradicts an execution-stage workspace bound",
                    ));
                }
                let fingerprint = stage_graph_fingerprint(execution_stages)?;
                if stage_graph_fingerprints.is_empty()
                    || stage_graph_fingerprints
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || stage_graph_fingerprints
                        .binary_search(&fingerprint)
                        .is_err()
                {
                    return Err(invalid(
                        "zero invocation workspace does not seal the selected stage graph",
                    ));
                }
                Ok(())
            }
            Self::Bounded { profiles } => {
                if profiles.is_empty() {
                    return Err(invalid(
                        "bounded invocation workspace has no stage profiles",
                    ));
                }
                let fingerprint = stage_graph_fingerprint(execution_stages)?;
                let matching = profiles
                    .iter()
                    .filter(|profile| profile.stage_graph_fingerprint == fingerprint)
                    .collect::<Vec<_>>();
                if matching.len() != 1 {
                    return Err(invalid(
                        "invocation workspace has no unique profile for the selected stage graph",
                    ));
                }
                matching[0].validate_against_stages(execution_stages)
            }
        }
    }
}

impl InvocationWorkspaceProfile {
    fn validate_against_stages(&self, execution_stages: &[StageDescriptor]) -> Result<()> {
        if self.stages.len() != execution_stages.len() {
            return Err(invalid(
                "invocation workspace must declare every execution stage exactly once",
            ));
        }
        let execution_ids = execution_stages
            .iter()
            .map(|stage| stage.id)
            .collect::<HashSet<_>>();
        let mut previous = None;
        let mut has_domain = false;
        for stage in &self.stages {
            if previous.is_some_and(|previous| stage.stage <= previous)
                || !execution_ids.contains(&stage.stage)
            {
                return Err(invalid(
                    "invocation workspace stages are not canonical execution stages",
                ));
            }
            previous = Some(stage.stage);
            let execution = execution_stages
                .iter()
                .find(|candidate| candidate.id == stage.stage)
                .expect("execution stage membership was checked");
            let maximum = stage.validate()?;
            has_domain |= !stage.domains.is_empty();
            if maximum != execution.max_workspace_bytes {
                return Err(invalid(
                    "invocation workspace formula disagrees with the loaded execution stage",
                ));
            }
        }
        if !has_domain {
            return Err(invalid(
                "bounded invocation workspace has no physical workspace domains",
            ));
        }
        Ok(())
    }
}

impl InvocationStageWorkspace {
    fn validate(&self) -> Result<u64> {
        let mut previous = None;
        let mut maximum = 0_u64;
        for domain in &self.domains {
            let id = domain.state.id();
            if previous.is_some_and(|previous| id <= previous) {
                return Err(invalid(
                    "invocation workspace domains require canonical unique ids",
                ));
            }
            previous = Some(id);
            maximum = maximum
                .checked_add(domain.maximum_bytes()?)
                .ok_or_else(|| invalid("invocation workspace stage byte bound overflow"))?;
        }
        Ok(maximum)
    }
}

impl InvocationWorkspaceDomain {
    fn maximum_bytes(&self) -> Result<u64> {
        self.state.validate()?;
        if self.state.scope() != StateScope::Invocation {
            return Err(invalid(
                "invocation workspace contains a retained state domain",
            ));
        }
        // Semantic placement is stated once. Resolution may choose host or
        // backend-local only within this policy.
        if self.state.header().placement != self.placement {
            return Err(invalid(
                "invocation workspace placement disagrees with its state domain",
            ));
        }
        let maximum = self.formula.maximum_bytes()?;
        let physical_minimum = minimum_physical_bytes(&self.state)?;
        if maximum < physical_minimum {
            return Err(invalid(
                "invocation workspace formula is smaller than its physical state geometry",
            ));
        }
        Ok(maximum)
    }
}

impl WorkspaceFormula {
    pub(crate) fn maximum_bytes(&self) -> Result<u64> {
        let mut axes = HashSet::with_capacity(self.dimensions.len());
        for dimension in &self.dimensions {
            if dimension.max_units == 0 || !axes.insert(dimension.axis) {
                return Err(invalid(
                    "workspace formula dimensions require non-zero unique axes",
                ));
            }
        }
        let mut maximum = self.fixed_bytes;
        for term in &self.terms {
            if term.bytes_per_element == 0 || term.factors.is_empty() {
                return Err(invalid(
                    "workspace formula terms require factors and non-zero bytes",
                ));
            }
            let mut term_axes = HashSet::with_capacity(term.factors.len());
            let elements = term.factors.iter().try_fold(1_u64, |product, axis| {
                if !term_axes.insert(*axis) || !axes.contains(axis) {
                    return Err(invalid(
                        "workspace formula term references duplicate or undeclared axes",
                    ));
                }
                let units = self
                    .dimensions
                    .iter()
                    .find(|dimension| dimension.axis == *axis)
                    .expect("workspace axis membership was checked")
                    .max_units;
                product
                    .checked_mul(units)
                    .ok_or_else(|| invalid("workspace formula element bound overflow"))
            })?;
            maximum = maximum
                .checked_add(
                    elements
                        .checked_mul(term.bytes_per_element)
                        .ok_or_else(|| invalid("workspace formula byte bound overflow"))?,
                )
                .ok_or_else(|| invalid("workspace formula byte bound overflow"))?;
        }
        Ok(maximum)
    }
}

fn minimum_physical_bytes(state: &StateDomainSpec) -> Result<u64> {
    use super::contract::StateDomainSpec;
    match state {
        StateDomainSpec::PagedAttention(spec) => {
            let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
                let elements = u64::from(layer.kv_heads)
                    .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
                    .ok_or_else(|| invalid("paged workspace geometry overflow"))?;
                total
                    .checked_add(elements)
                    .ok_or_else(|| invalid("paged workspace geometry overflow"))
            })?;
            minimum_dtype_bytes(
                elements_per_token
                    .checked_mul(u64::from(spec.page_size.preferred_tokens))
                    .ok_or_else(|| invalid("paged workspace geometry overflow"))?,
                &spec.accepted_dtypes,
            )
        }
        StateDomainSpec::StaticAttention(spec) => {
            let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
                let elements = u64::from(layer.kv_heads)
                    .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
                    .ok_or_else(|| invalid("static-attention workspace geometry overflow"))?;
                total
                    .checked_add(elements)
                    .ok_or_else(|| invalid("static-attention workspace geometry overflow"))
            })?;
            minimum_dtype_bytes(
                elements_per_token
                    .checked_mul(spec.max_memory_tokens)
                    .ok_or_else(|| invalid("static-attention workspace geometry overflow"))?,
                &spec.accepted_dtypes,
            )
        }
        StateDomainSpec::Tensor(spec) => minimum_component_bytes(&spec.components),
        StateDomainSpec::StaticTensor(spec) => minimum_component_bytes(&spec.components),
        StateDomainSpec::Append(spec) => minimum_component_bytes(&spec.components_per_step)?
            .checked_mul(spec.max_steps)
            .ok_or_else(|| invalid("append workspace geometry overflow")),
        StateDomainSpec::Ring(spec) => minimum_component_bytes(&spec.components_per_step)?
            .checked_mul(spec.capacity_steps)
            .ok_or_else(|| invalid("ring workspace geometry overflow")),
    }
}

fn minimum_component_bytes(components: &[super::contract::TensorComponentSpec]) -> Result<u64> {
    components.iter().try_fold(0_u64, |total, component| {
        let bytes = minimum_dtype_bytes(
            component.shape.maximum_elements()?,
            &component.accepted_dtypes,
        )?;
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid("tensor workspace geometry overflow"))
    })
}

fn minimum_dtype_bytes(elements: u64, dtypes: &[super::contract::StateDType]) -> Result<u64> {
    dtypes
        .iter()
        .map(|dtype| match dtype {
            super::contract::StateDType::F32 => elements.checked_mul(4),
            super::contract::StateDType::F16 | super::contract::StateDType::Bf16 => {
                elements.checked_mul(2)
            }
            super::contract::StateDType::I8 => Some(elements),
            super::contract::StateDType::Q4 => elements.checked_add(1).map(|value| value / 2),
        })
        .collect::<Option<Vec<_>>>()
        .and_then(|bytes| bytes.into_iter().min())
        .ok_or_else(|| invalid("workspace dtype byte bound overflow or missing dtype"))
}

fn validate_stage_ids(stages: &[StageDescriptor]) -> Result<()> {
    if stages.is_empty() {
        return Err(invalid("loaded execution contract has no stages"));
    }
    let mut ids = HashSet::with_capacity(stages.len());
    if stages.iter().any(|stage| !ids.insert(stage.id)) {
        return Err(invalid("loaded execution contract repeats a stage id"));
    }
    Ok(())
}

pub(crate) fn stage_graph_fingerprint(stages: &[StageDescriptor]) -> Result<[u8; 32]> {
    validate_stage_ids(stages)?;
    let encoded = serde_json::to_vec(stages)
        .map_err(|error| invalid(format!("failed to encode execution stage graph: {error}")))?;
    let mut hasher = Sha256::new();
    hasher.update(STAGE_GRAPH_FINGERPRINT_DOMAIN);
    hasher.update(encoded);
    Ok(hasher.finalize().into())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::{
        ConcurrencyClass, ExecutionDomain, MembershipSafePoint, NativeBatchMode, OutputVisibility,
        StageProgressKind, StageShapePolicy, StageWorkSelector,
    };
    use crate::kv::v2::{
        BoundedShape, CheckpointPolicy, PrefixPolicy, ShapeAxis, ShapeDimension, ShapeExtent,
        StateClock, StateComponentId, StateDType, StateDomainHeader, StaticTensorDomainSpec,
        TensorComponentSpec, TensorRole, WorkspaceAxis,
    };

    fn stage(max_workspace_bytes: u64) -> StageDescriptor {
        StageDescriptor {
            id: StageId::new(1),
            name: "execute".to_string(),
            selector: StageWorkSelector::Any,
            domain: ExecutionDomain::ExecutionGroup,
            progress: StageProgressKind::Atomic,
            concurrency: ConcurrencyClass::Batchable,
            batch_mode: NativeBatchMode::Static,
            max_batch_size: 4,
            max_work_units: 32,
            workspace_base_bytes: 0,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
            shape_policy: StageShapePolicy::Independent,
            membership_safe_point: MembershipSafePoint::OperationBoundary,
            output_visibility: OutputVisibility::AfterQuantumCommit,
        }
    }

    fn workspace_domain() -> InvocationWorkspaceDomain {
        InvocationWorkspaceDomain {
            state: StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
                header: StateDomainHeader {
                    id: super::super::StateDomainId::new(1),
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
                            extent: ShapeExtent::Fixed { value: 8 },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F16],
                }],
            }),
            placement: PlacementPolicy::BackendLocal,
            formula: WorkspaceFormula {
                fixed_bytes: 64,
                dimensions: vec![WorkspaceDimensionBound {
                    axis: WorkspaceAxis::InputTokens,
                    max_units: 8,
                }],
                terms: vec![WorkspaceTerm {
                    factors: vec![WorkspaceAxis::InputTokens],
                    bytes_per_element: 16,
                }],
            },
        }
    }

    #[test]
    fn explicit_stateless_workspace_is_stage_complete_and_bounded() {
        let mut second_domain = workspace_domain();
        second_domain.formula = WorkspaceFormula {
            fixed_bytes: 64,
            dimensions: vec![],
            terms: vec![],
        };
        let descriptor = CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::Bounded {
                profiles: vec![
                    InvocationWorkspaceProfile {
                        stage_graph_fingerprint: stage_graph_fingerprint(&[stage(192)]).unwrap(),
                        stages: vec![InvocationStageWorkspace {
                            stage: StageId::new(1),
                            domains: vec![workspace_domain()],
                        }],
                    },
                    InvocationWorkspaceProfile {
                        stage_graph_fingerprint: stage_graph_fingerprint(&[stage(64)]).unwrap(),
                        stages: vec![InvocationStageWorkspace {
                            stage: StageId::new(1),
                            domains: vec![second_domain],
                        }],
                    },
                ],
            },
        };
        descriptor.validate_against_stages(&[stage(192)]).unwrap();
        descriptor.validate_against_stages(&[stage(64)]).unwrap();
        assert!(descriptor.is_stateless());
        assert!(descriptor.validate_against_stages(&[stage(191)]).is_err());
        assert_eq!(
            descriptor.fingerprint(&[stage(192)]).unwrap(),
            descriptor.fingerprint(&[stage(192)]).unwrap()
        );
    }

    #[test]
    fn missing_workspace_and_wrong_lifetime_fail_closed() {
        let none = CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::None {
                stage_graph_fingerprints: vec![stage_graph_fingerprint(&[stage(1)]).unwrap()],
            },
        };
        assert!(none.validate_against_stages(&[stage(1)]).is_err());

        let mut wrong = workspace_domain();
        match &mut wrong.state {
            StateDomainSpec::StaticTensor(spec) => spec.header.scope = StateScope::Retained,
            _ => unreachable!(),
        }
        let descriptor = CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::Bounded {
                profiles: vec![InvocationWorkspaceProfile {
                    stage_graph_fingerprint: stage_graph_fingerprint(&[stage(192)]).unwrap(),
                    stages: vec![InvocationStageWorkspace {
                        stage: StageId::new(1),
                        domains: vec![wrong],
                    }],
                }],
            },
        };
        assert!(descriptor.validate_against_stages(&[stage(192)]).is_err());

        let mut undersized = workspace_domain();
        undersized.formula = WorkspaceFormula {
            fixed_bytes: 15,
            dimensions: vec![],
            terms: vec![],
        };
        let descriptor = CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::Bounded {
                profiles: vec![InvocationWorkspaceProfile {
                    stage_graph_fingerprint: stage_graph_fingerprint(&[stage(15)]).unwrap(),
                    stages: vec![InvocationStageWorkspace {
                        stage: StageId::new(1),
                        domains: vec![undersized],
                    }],
                }],
            },
        };
        assert!(descriptor.validate_against_stages(&[stage(15)]).is_err());
    }
}
