//! Shared construction for invocation-only typed physical state.
//!
//! Model adapters remain responsible for authoring semantic domains and
//! consistency groups. This module only lowers those exact contracts into the
//! repeated per-stage workspace descriptor used by the lifecycle allocator.

use crate::engine::{NativeBatchMode, StageDescriptor, StageProgressKind};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    RetainedStateCapability, StateDType, StateDomainSpec, WorkspaceFormula,
    CURRENT_INFERENCE_STATE_ABI,
};

pub(crate) fn typed_invocation_descriptor(
    stage_graphs: &[&[StageDescriptor]],
    contract: &InferenceStateContract,
) -> Result<CapabilityStateDescriptorV2> {
    if stage_graphs.is_empty() {
        return Err(model_load(
            "typed invocation state has no selectable execution graph",
        ));
    }
    contract.validate()?;
    if contract.domains.is_empty()
        || contract
            .domains
            .iter()
            .any(|domain| domain.scope() != crate::kv::v2::StateScope::Invocation)
    {
        return Err(model_load(
            "typed invocation state requires non-empty invocation-scoped domains",
        ));
    }
    let max_domain = contract
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| model_load("typed invocation contract is empty"))?;

    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        if stages.is_empty() {
            return Err(model_load("typed invocation execution graph has no stages"));
        }
        for stage in *stages {
            stage.validate()?;
            if stage.progress != StageProgressKind::Atomic
                || stage.batch_mode != NativeBatchMode::None
            {
                return Err(model_load(
                    "typed invocation state requires independently scheduled atomic rows",
                ));
            }
        }

        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (stage_index, stage) in ordered.into_iter().enumerate() {
            let mut domains = contract
                .domains
                .iter()
                .cloned()
                .map(|state| {
                    let fixed_bytes = typed_domain_maximum_bytes(&state)?;
                    Ok(InvocationWorkspaceDomain::State {
                        placement: state.header().placement,
                        formula: fixed_formula(fixed_bytes),
                        state,
                        capacity: InvocationStateCapacity::SemanticBounded,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if stage.max_workspace_bytes > 0 {
                let ordinal = u32::try_from(stage_index + 1)
                    .map_err(|_| model_load("typed invocation stage count exceeds u32"))?;
                let scratch = max_domain
                    .checked_add(ordinal)
                    .ok_or_else(|| model_load("typed invocation scratch domain overflow"))?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: crate::kv::v2::StateDomainId::new(scratch),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: fixed_formula(stage.max_workspace_bytes),
                });
            }
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: contract.groups.clone(),
                domains,
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: invocation_stages,
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();

    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: RetainedStateCapability::Stateless,
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(descriptor)
}

fn typed_domain_maximum_bytes(domain: &StateDomainSpec) -> Result<u64> {
    let (components, steps) = match domain {
        StateDomainSpec::Tensor(spec) => (spec.components.as_slice(), 1),
        StateDomainSpec::StaticTensor(spec) => (spec.components.as_slice(), 1),
        StateDomainSpec::Append(spec) => (spec.components_per_step.as_slice(), spec.max_steps),
        StateDomainSpec::Ring(spec) => (spec.components_per_step.as_slice(), spec.capacity_steps),
        StateDomainSpec::PagedAttention(_) | StateDomainSpec::StaticAttention(_) => {
            return Err(model_load(
                "generic typed invocation lowering does not accept attention domains",
            ));
        }
    };
    let per_step = components.iter().try_fold(0_u64, |total, component| {
        let elements = component.shape.maximum_elements()?;
        let element_bytes = component
            .accepted_dtypes
            .iter()
            .copied()
            .map(dtype_bytes)
            .max()
            .ok_or_else(|| model_load("typed invocation component has no dtype"))?;
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or_else(|| model_load("typed invocation component byte bound overflow"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| model_load("typed invocation domain byte bound overflow"))
    })?;
    per_step
        .checked_mul(steps)
        .ok_or_else(|| model_load("typed invocation capacity byte bound overflow"))
}

const fn dtype_bytes(dtype: StateDType) -> u64 {
    match dtype {
        StateDType::F32 => 4,
        StateDType::F16 | StateDType::Bf16 => 2,
        StateDType::I64 => 8,
        StateDType::I8 => 1,
        StateDType::Q4 => 1,
    }
}

const fn fixed_formula(fixed_bytes: u64) -> WorkspaceFormula {
    WorkspaceFormula {
        fixed_bytes,
        dimensions: Vec::new(),
        terms: Vec::new(),
    }
}

fn model_load(message: impl Into<String>) -> Error {
    Error::ModelLoadError(message.into())
}
