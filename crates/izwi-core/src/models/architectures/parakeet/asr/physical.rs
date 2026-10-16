//! Physical recurrent state authored by the loaded Parakeet decoder.

use crate::engine::{StageDescriptor, StageWorkSelector};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, BoundedShape, CapabilityStateDescriptorV2, CheckpointPolicy,
    InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy, PrefixPolicy, ShapeAxis,
    ShapeDimension, ShapeExtent, StateClock, StateComponentId, StateDType, StateDomainHeader,
    StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope, TensorComponentSpec,
    TensorRole, TensorStateDomainSpec, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::state::typed_invocation_descriptor;

use super::PRED_HIDDEN;

pub(crate) const PARAKEET_PREDICTOR_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const PARAKEET_PREDICTOR_STATE_GROUP: StateGroupId = StateGroupId::new(1);

#[derive(Debug, Clone)]
pub(crate) struct ParakeetPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    /// Sequence graphs own the predictor tensors as retained state and
    /// publish scratch-only stage workspace.
    pub(crate) retained: Option<InferenceStateContract>,
    /// Atomic graphs (the long-form pipeline graph) lease the same predictor
    /// tensors per invocation, beside the retained state of sequence graphs.
    pub(crate) invocation: Option<InferenceStateContract>,
}

pub(crate) fn parakeet_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
) -> Result<ParakeetPhysicalStateSpec> {
    let invocation = parakeet_invocation_contract()?;
    let retained = parakeet_retained_contract(invocation.clone())?;
    let is_retained_stage = |stage: &StageDescriptor| {
        matches!(
            stage.selector,
            StageWorkSelector::SequencePrefill | StageWorkSelector::SequenceDecode
        )
    };
    let (atomic_graphs, retained_graphs): (Vec<_>, Vec<_>) =
        stage_graphs.iter().copied().partition(|stages| {
            stages
                .iter()
                .any(|stage| stage.selector == StageWorkSelector::Atomic)
        });
    if atomic_graphs
        .iter()
        .any(|stages| stages.iter().any(is_retained_stage))
    {
        return Err(Error::ModelLoadError(
            "a Parakeet graph cannot mix retained and atomic predictor lifetimes".into(),
        ));
    }
    let uses_retained = retained_graphs
        .iter()
        .any(|stages| stages.iter().any(is_retained_stage));
    let uses_atomic = !atomic_graphs.is_empty();
    let descriptor = if !uses_retained {
        typed_invocation_descriptor(stage_graphs, &invocation)?
    } else if !uses_atomic {
        CapabilityStateDescriptorV2::managed_for_stage_graphs(retained.clone(), stage_graphs)?
    } else {
        retained_with_atomic_profiles(
            CapabilityStateDescriptorV2::managed_for_stage_graphs(
                retained.clone(),
                &retained_graphs,
            )?,
            &retained_graphs,
            typed_invocation_descriptor(&atomic_graphs, &invocation)?,
            stage_graphs,
        )?
    };
    Ok(ParakeetPhysicalStateSpec {
        descriptor,
        retained: uses_retained.then_some(retained),
        invocation: uses_atomic.then_some(invocation),
    })
}

/// One publication for both lifetimes: the sequence graphs keep their
/// scratch-only profiles under the retained contract, and each atomic graph
/// adds its invocation predictor profile.
fn retained_with_atomic_profiles(
    retained: CapabilityStateDescriptorV2,
    retained_graphs: &[&[StageDescriptor]],
    atomic: CapabilityStateDescriptorV2,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<CapabilityStateDescriptorV2> {
    let mut profiles = match retained.invocation {
        InvocationWorkspaceSet::Bounded { profiles } => profiles,
        // Sequence graphs without stage scratch still need an exact,
        // domain-free profile once the publication is bounded.
        InvocationWorkspaceSet::None { .. } => retained_graphs
            .iter()
            .map(|stages| {
                let mut ids = stages.iter().map(|stage| stage.id).collect::<Vec<_>>();
                ids.sort_unstable();
                Ok(InvocationWorkspaceProfile {
                    stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
                    stages: ids
                        .into_iter()
                        .map(|stage| InvocationStageWorkspace {
                            stage,
                            lease_scope: InvocationLeaseScope::PerStageBatch,
                            groups: Vec::new(),
                            domains: Vec::new(),
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let InvocationWorkspaceSet::Bounded {
        profiles: atomic_profiles,
    } = atomic.invocation
    else {
        return Err(Error::ModelLoadError(
            "Parakeet atomic graph must bound its invocation predictor workspace".into(),
        ));
    };
    profiles.extend(atomic_profiles);
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: retained.retained,
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(descriptor)
}

fn parakeet_retained_contract(
    mut contract: InferenceStateContract,
) -> Result<InferenceStateContract> {
    for domain in &mut contract.domains {
        let StateDomainSpec::Tensor(domain) = domain else {
            return Err(Error::ModelLoadError(
                "Parakeet retained predictor state must be tensor state".into(),
            ));
        };
        domain.header.scope = StateScope::Retained;
        domain.header.checkpoint = CheckpointPolicy::Transactional;
        domain.header.prefix = PrefixPolicy::Disabled;
    }
    contract.validate().map_err(|error| {
        Error::ModelLoadError(format!("invalid retained Parakeet state contract: {error}"))
    })?;
    Ok(contract)
}

fn parakeet_invocation_contract() -> Result<InferenceStateContract> {
    let component = |id, role, dimensions| TensorComponentSpec {
        id: StateComponentId::new(id),
        role,
        shape: BoundedShape { dimensions },
        accepted_dtypes: vec![StateDType::F32],
    };
    let batch_hidden = || {
        vec![
            fixed(ShapeAxis::Batch, 1),
            fixed(ShapeAxis::Hidden, PRED_HIDDEN as u64),
        ]
    };
    let components = vec![
        component(1, TensorRole::RecurrentHidden, batch_hidden()),
        component(2, TensorRole::RecurrentCell, batch_hidden()),
        component(3, TensorRole::RecurrentHidden, batch_hidden()),
        component(4, TensorRole::RecurrentCell, batch_hidden()),
        component(
            5,
            TensorRole::Custom("tdt_predictor_output".into()),
            vec![
                fixed(ShapeAxis::Batch, 1),
                fixed(ShapeAxis::Sequence, 1),
                fixed(ShapeAxis::Hidden, PRED_HIDDEN as u64),
            ],
        ),
    ];
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: PARAKEET_PREDICTOR_STATE_DOMAIN,
                scope: StateScope::Invocation,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::None,
            },
            components,
        })],
        groups: vec![StateGroupSpec {
            id: PARAKEET_PREDICTOR_STATE_GROUP,
            domains: vec![PARAKEET_PREDICTOR_STATE_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate().map_err(|error| {
        Error::ModelLoadError(format!("invalid Parakeet physical state contract: {error}"))
    })?;
    Ok(contract)
}

const fn fixed(axis: ShapeAxis, value: u64) -> ShapeDimension {
    ShapeDimension {
        axis,
        extent: ShapeExtent::Fixed { value },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{
        negotiate_state_plan, PhysicalStateSequenceId, PhysicalStateTransactionId,
        StateBackendPlanRequest, StateComponentValue, TensorStateArena, TensorStateCapacity,
        TensorStateSelection,
    };
    use crate::backends::BackendKind;
    use crate::engine::{
        ExecutionMode, ExecutionProfile, NativeBatchMode, StageId, StageWorkSelector,
    };
    use crate::model::ModelVariant;
    use candle_core::{DType, Device, Tensor};
    use std::sync::Arc;

    fn stage(id: u32, selector: StageWorkSelector) -> StageDescriptor {
        let mut profile = ExecutionProfile::fail_closed(
            crate::backends::BackendKind::Cpu,
            Some(ModelVariant::ParakeetTdt06BV3),
            ExecutionMode::Sequence,
        );
        profile.resolved_from_loaded_model = true;
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(id),
            "parakeet.test",
            &profile,
            NativeBatchMode::None,
        );
        stage.selector = selector;
        stage
    }

    #[test]
    fn retained_contract_is_tensor_only_and_decoder_clocked() {
        let stages = vec![
            stage(1, StageWorkSelector::SequencePrefill),
            stage(2, StageWorkSelector::SequenceDecode),
        ];
        let spec = parakeet_physical_state_spec(&[&stages]).unwrap();
        assert!(spec.invocation.is_none());
        let retained = spec.retained.expect("retained Parakeet contract");
        assert_eq!(retained.domains.len(), 1);
        let StateDomainSpec::Tensor(domain) = &retained.domains[0] else {
            panic!("Parakeet recurrent state must never publish paged KV")
        };
        assert_eq!(domain.header.id, PARAKEET_PREDICTOR_STATE_DOMAIN);
        assert_eq!(domain.header.scope, StateScope::Retained);
        assert_eq!(domain.header.clock, StateClock::DecoderTokens);
        assert_eq!(domain.header.checkpoint, CheckpointPolicy::Transactional);
        assert_eq!(domain.components.len(), 5);
    }

    #[test]
    fn retained_and_atomic_graphs_share_one_physical_publication() {
        let retained = vec![
            stage(1, StageWorkSelector::SequencePrefill),
            stage(2, StageWorkSelector::SequenceDecode),
        ];
        let mut atomic_stage = stage(3, StageWorkSelector::Atomic);
        atomic_stage.progress = crate::engine::StageProgressKind::Atomic;
        let atomic = vec![atomic_stage];
        let spec = parakeet_physical_state_spec(&[&retained, &atomic]).unwrap();
        assert!(spec.retained.is_some());
        let invocation = spec.invocation.expect("atomic Parakeet predictor contract");
        assert!(matches!(
            spec.descriptor.retained,
            crate::kv::v2::RetainedStateCapability::Managed { .. }
        ));
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("mixed Parakeet publication must bound its invocation workspace")
        };
        let profile_for = |stages: &[StageDescriptor]| {
            let graph = crate::kv::v2::stage_graph_fingerprint(stages).unwrap();
            profiles
                .iter()
                .find(|profile| profile.stage_graph_fingerprint == graph)
                .expect("every Parakeet graph has a workspace profile")
        };
        let typed_domains = |stages: &[StageDescriptor]| {
            profile_for(stages)
                .stages
                .iter()
                .flat_map(|stage| &stage.domains)
                .filter_map(|domain| match domain {
                    crate::kv::v2::InvocationWorkspaceDomain::State { state, .. } => {
                        Some(state.clone())
                    }
                    crate::kv::v2::InvocationWorkspaceDomain::Scratch { .. } => None,
                })
                .collect::<Vec<_>>()
        };
        // Sequence graphs keep the retained predictor and lease no tensor;
        // the atomic graph leases exactly the invocation predictor domain.
        assert!(typed_domains(&retained).is_empty());
        assert_eq!(typed_domains(&atomic), invocation.domains);
        assert_eq!(
            profile_for(&atomic).stages[0].lease_scope,
            crate::kv::v2::InvocationLeaseScope::PerRow
        );
    }

    #[test]
    fn one_graph_cannot_mix_retained_and_atomic_predictor_lifetimes() {
        let mut atomic_stage = stage(3, StageWorkSelector::Atomic);
        atomic_stage.progress = crate::engine::StageProgressKind::Atomic;
        let mixed = vec![stage(1, StageWorkSelector::SequenceDecode), atomic_stage];
        let error = parakeet_physical_state_spec(&[&mixed])
            .expect_err("one graph cannot own both predictor lifetimes");
        assert!(error.to_string().contains("cannot mix"));
    }

    #[test]
    fn atomic_graph_publishes_invocation_predictor_state() {
        let mut atomic_stage = stage(1, StageWorkSelector::Atomic);
        atomic_stage.progress = crate::engine::StageProgressKind::Atomic;
        let atomic = vec![atomic_stage];
        let spec = parakeet_physical_state_spec(&[&atomic]).unwrap();
        assert!(spec.retained.is_none());
        assert_eq!(spec.invocation.unwrap().domains.len(), 1);
    }

    #[test]
    fn retained_tensor_abort_preserves_committed_predictor_state() {
        let contract = parakeet_retained_contract(parakeet_invocation_contract().unwrap()).unwrap();
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: None,
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let capacity = TensorStateCapacity::for_plan(&plan, 1, 1).unwrap();
        let arena =
            TensorStateArena::new_with_contract(Arc::new(plan), &contract, capacity, Device::Cpu)
                .unwrap();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        arena.register(sequence).unwrap();
        let values = |fill: f32| {
            (0_u32..5)
                .map(|index| StateComponentValue {
                    component: StateComponentId::new(index + 1),
                    tensor: Some(
                        Tensor::full(
                            fill,
                            if index < 4 {
                                vec![1, PRED_HIDDEN]
                            } else {
                                vec![1, 1, PRED_HIDDEN]
                            },
                            &Device::Cpu,
                        )
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap(),
                    ),
                })
                .collect::<Vec<_>>()
        };
        let selection = |expected_cursor, target_cursor| TensorStateSelection {
            group: PARAKEET_PREDICTOR_STATE_GROUP,
            clock: StateClock::DecoderTokens,
            expected_cursor,
            target_cursor,
        };

        let first = PhysicalStateTransactionId::new(1).unwrap();
        arena
            .begin_selected(first, sequence, &[selection(0, 1)])
            .unwrap();
        arena
            .stage_replace(first, PARAKEET_PREDICTOR_STATE_DOMAIN, 0, 1, values(0.0))
            .unwrap();
        let completion = arena.seal_selected_completion(first).unwrap();
        arena.commit_selected(first, &completion).unwrap();
        assert_eq!(
            arena
                .read(sequence, PARAKEET_PREDICTOR_STATE_DOMAIN)
                .unwrap()
                .unwrap()
                .cursor,
            1
        );

        let rollback = PhysicalStateTransactionId::new(2).unwrap();
        arena
            .begin_selected(rollback, sequence, &[selection(1, 2)])
            .unwrap();
        arena
            .stage_replace(rollback, PARAKEET_PREDICTOR_STATE_DOMAIN, 1, 2, values(1.0))
            .unwrap();
        arena.abort(rollback).unwrap();
        let after = arena
            .read(sequence, PARAKEET_PREDICTOR_STATE_DOMAIN)
            .unwrap()
            .unwrap();
        assert_eq!(after.cursor, 1);
        assert_eq!(
            after.components[0]
                .tensor
                .as_ref()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            0.0
        );
    }
}
