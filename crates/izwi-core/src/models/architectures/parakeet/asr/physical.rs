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
