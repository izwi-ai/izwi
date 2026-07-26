//! Physical invocation state authored by the loaded Parakeet decoder.

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
const PARAKEET_PREDICTOR_STATE_GROUP: StateGroupId = StateGroupId::new(1);

#[derive(Debug, Clone)]
pub(crate) struct ParakeetPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn parakeet_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
) -> Result<ParakeetPhysicalStateSpec> {
    let invocation = parakeet_invocation_contract()?;
    let descriptor = typed_invocation_descriptor(stage_graphs, &invocation)?;
    Ok(ParakeetPhysicalStateSpec {
        descriptor,
        invocation,
    })
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
