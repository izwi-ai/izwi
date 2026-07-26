//! Invocation-only physical state for Nemotron's offline RNNT decoder.

use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    BoundedShape, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    PlacementPolicy, PrefixPolicy, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, StaticTensorDomainSpec, TensorComponentSpec, TensorRole,
    TensorStateDomainSpec, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::state::typed_invocation_descriptor;

use super::{NemotronRealtimeStateShape, DEFAULT_MAX_AUDIO_SECONDS_HINT, SAMPLE_RATE};

pub(crate) const NEMOTRON_OFFLINE_PREDICTOR_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN: StateDomainId = StateDomainId::new(2);
const NEMOTRON_OFFLINE_GROUP: StateGroupId = StateGroupId::new(1);

#[derive(Debug, Clone)]
pub(crate) struct NemotronOfflinePhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn nemotron_offline_physical_state_spec(
    shape: NemotronRealtimeStateShape,
    dtype: StateDType,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<NemotronOfflinePhysicalStateSpec> {
    let invocation = nemotron_offline_invocation_contract(shape, dtype)?;
    let descriptor = typed_invocation_descriptor(stage_graphs, &invocation)?;
    Ok(NemotronOfflinePhysicalStateSpec {
        descriptor,
        invocation,
    })
}

fn nemotron_offline_invocation_contract(
    shape: NemotronRealtimeStateShape,
    dtype: StateDType,
) -> Result<InferenceStateContract> {
    if shape.predictor_hidden == 0
        || shape.joint_hidden == 0
        || shape.hop_length == 0
        || shape.subsampling_factor == 0
    {
        return Err(model_load("Nemotron offline state has zero geometry"));
    }
    let batch_hidden = |hidden| {
        vec![
            fixed(ShapeAxis::Batch, 1),
            fixed(ShapeAxis::Hidden, hidden as u64),
        ]
    };
    let batch_sequence_hidden = |hidden| {
        vec![
            fixed(ShapeAxis::Batch, 1),
            fixed(ShapeAxis::Sequence, 1),
            fixed(ShapeAxis::Hidden, hidden as u64),
        ]
    };
    let component = |id, role, dimensions| TensorComponentSpec {
        id: StateComponentId::new(id),
        role,
        shape: BoundedShape { dimensions },
        accepted_dtypes: vec![dtype],
    };
    let predictor = vec![
        component(
            1,
            TensorRole::RecurrentHidden,
            batch_hidden(shape.predictor_hidden),
        ),
        component(
            2,
            TensorRole::RecurrentCell,
            batch_hidden(shape.predictor_hidden),
        ),
        component(
            3,
            TensorRole::RecurrentHidden,
            batch_hidden(shape.predictor_hidden),
        ),
        component(
            4,
            TensorRole::RecurrentCell,
            batch_hidden(shape.predictor_hidden),
        ),
        component(
            5,
            TensorRole::RetainedEmbedding,
            batch_sequence_hidden(shape.predictor_hidden),
        ),
        component(
            6,
            TensorRole::Custom("rnnt_predictor_projection".into()),
            batch_sequence_hidden(shape.joint_hidden),
        ),
    ];
    let max_samples = (DEFAULT_MAX_AUDIO_SECONDS_HINT * SAMPLE_RATE as f32).ceil() as usize;
    let max_feature_frames = max_samples.div_ceil(shape.hop_length).max(1);
    let max_encoded_frames = max_feature_frames.div_ceil(shape.subsampling_factor).max(1);
    let acoustic = vec![component(
        1,
        TensorRole::EncoderMemory,
        vec![
            fixed(ShapeAxis::Batch, 1),
            ShapeDimension {
                axis: ShapeAxis::Sequence,
                extent: ShapeExtent::RuntimeBounded {
                    min: 1,
                    max: max_encoded_frames as u64,
                },
            },
            fixed(ShapeAxis::Hidden, shape.joint_hidden as u64),
        ],
    )];
    let header = |id, clock| StateDomainHeader {
        id,
        scope: StateScope::Invocation,
        clock,
        placement: PlacementPolicy::BackendLocal,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::None,
    };
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![
            StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: header(NEMOTRON_OFFLINE_PREDICTOR_DOMAIN, StateClock::DecoderTokens),
                components: predictor,
            }),
            StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
                header: header(NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN, StateClock::EncoderTokens),
                components: acoustic,
            }),
        ],
        groups: vec![StateGroupSpec {
            id: NEMOTRON_OFFLINE_GROUP,
            domains: vec![
                NEMOTRON_OFFLINE_PREDICTOR_DOMAIN,
                NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN,
            ],
            prefix_shareable: false,
        }],
    };
    contract
        .validate()
        .map_err(|error| model_load(format!("invalid Nemotron offline state: {error}")))?;
    Ok(contract)
}

const fn fixed(axis: ShapeAxis, value: u64) -> ShapeDimension {
    ShapeDimension {
        axis,
        extent: ShapeExtent::Fixed { value },
    }
}

fn model_load(message: impl Into<String>) -> Error {
    Error::ModelLoadError(message.into())
}
