//! Physical recurrent state authored by the loaded Parakeet decoder.

use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    BoundedShape, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    PlacementPolicy, PrefixPolicy, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
    CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::state::typed_invocation_descriptor;

use super::PRED_HIDDEN;

pub(crate) const PARAKEET_PREDICTOR_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const PARAKEET_PREDICTOR_STATE_GROUP: StateGroupId = StateGroupId::new(1);

#[derive(Debug, Clone)]
pub(crate) struct ParakeetPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: Option<InferenceStateContract>,
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn parakeet_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
) -> Result<ParakeetPhysicalStateSpec> {
    let invocation = parakeet_invocation_contract()?;
    let retained = parakeet_retained_contract(invocation.clone())?;
    let uses_retained = stage_graphs.iter().any(|stages| {
        stages.iter().any(|stage| {
            matches!(
                stage.selector,
                crate::engine::StageWorkSelector::SequencePrefill
                    | crate::engine::StageWorkSelector::SequenceDecode
            )
        })
    });
    let uses_atomic = stage_graphs.iter().any(|stages| {
        stages
            .iter()
            .any(|stage| stage.selector == crate::engine::StageWorkSelector::Atomic)
    });
    if uses_retained && uses_atomic {
        return Err(Error::ModelLoadError(
            "Parakeet retained and atomic compatibility graphs must be published separately".into(),
        ));
    }
    let descriptor = if uses_retained {
        CapabilityStateDescriptorV2::managed_for_stage_graphs(retained.clone(), stage_graphs)?
    } else {
        typed_invocation_descriptor(stage_graphs, &invocation)?
    };
    Ok(ParakeetPhysicalStateSpec {
        descriptor,
        retained: uses_retained.then_some(retained),
        invocation,
    })
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
    fn retained_and_atomic_graphs_cannot_share_physical_publication() {
        let retained = vec![stage(1, StageWorkSelector::SequenceDecode)];
        let atomic = vec![stage(2, StageWorkSelector::Atomic)];
        let error = parakeet_physical_state_spec(&[&retained, &atomic])
            .expect_err("mixed Parakeet state lifetimes must fail closed");
        assert!(error.to_string().contains("must be published separately"));
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
