//! Invocation-only physical state for Nemotron's offline RNNT decoder.

use crate::engine::InvocationTensorLease;
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

use super::network::causal_subsampled_len;
use super::{NemotronRealtimeStateShape, DEFAULT_MAX_AUDIO_SECONDS_HINT, SAMPLE_RATE};

pub(crate) const NEMOTRON_OFFLINE_PREDICTOR_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN: StateDomainId = StateDomainId::new(2);
const NEMOTRON_OFFLINE_GROUP: StateGroupId = StateGroupId::new(1);
const NEMOTRON_OFFLINE_ENCODER_MEMORY: StateComponentId = StateComponentId::new(1);

/// Encoder frames the offline encoder emits for `samples_16khz` samples:
/// NeMo `get_seq_len` (floor) followed by the encoder's causal subsampling.
pub(super) fn nemotron_offline_encoded_frames(
    samples_16khz: usize,
    shape: NemotronRealtimeStateShape,
) -> usize {
    let feature_frames = (samples_16khz / shape.hop_length.max(1)).max(1);
    causal_subsampled_len(feature_frames, shape.subsampling_factor)
}

/// Rejects audio whose encoded length exceeds the leased acoustic state's
/// sequence bound. Callers run this before encoding, so over-bound input
/// never enqueues encoder work that would only fail at state install.
pub(super) fn ensure_offline_acoustic_capacity(
    acoustic: &InvocationTensorLease,
    encoded_frames: usize,
) -> Result<()> {
    let bound = acoustic
        .arena()?
        .component_axis_max(NEMOTRON_OFFLINE_ENCODER_MEMORY, &ShapeAxis::Sequence)
        .ok_or_else(|| {
            Error::InferenceError("Nemotron offline acoustic state has no sequence bound".into())
        })?;
    if u64::try_from(encoded_frames).map_or(true, |frames| frames > bound) {
        return Err(Error::InvalidInput(format!(
            "audio encodes to {encoded_frames} frames, above the Nemotron offline invocation \
             bound of {bound} (about {DEFAULT_MAX_AUDIO_SECONDS_HINT} s); chunk it"
        )));
    }
    Ok(())
}

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
    // The same causal subsampling the encoder applies (len / 2 + 1 per
    // stride-2 stage): 3000 feature frames encode to 376, not 375.
    let max_encoded_frames =
        causal_subsampled_len(max_feature_frames, shape.subsampling_factor).max(1);
    let acoustic = vec![component(
        NEMOTRON_OFFLINE_ENCODER_MEMORY.get(),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::Device;

    use super::*;
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::backends::BackendKind;
    use crate::engine::{InvocationTensorPoolOwner, ModelInstanceId};
    use crate::kv::v2::{InvocationStateCapacity, InvocationWorkspaceDomain, WorkspaceFormula};
    use crate::models::architectures::nemotron::asr::default_realtime_state_shape;

    fn acoustic_sequence_max(contract: &InferenceStateContract) -> u64 {
        let StateDomainSpec::StaticTensor(acoustic) = contract
            .domains
            .iter()
            .find(|domain| domain.id() == NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN)
            .expect("acoustic domain")
        else {
            panic!("acoustic domain must be static tensor state");
        };
        acoustic.components[0]
            .shape
            .dimensions
            .iter()
            .find(|dimension| dimension.axis == ShapeAxis::Sequence)
            .expect("sequence axis")
            .extent
            .max()
    }

    fn acoustic_pool(contract: &InferenceStateContract) -> InvocationTensorPoolOwner {
        let plan = Arc::new(
            negotiate_state_plan(
                contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: None,
                },
            )
            .expect("plan"),
        );
        let state = contract
            .domains
            .iter()
            .find(|domain| domain.id() == NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN)
            .expect("acoustic domain")
            .clone();
        let fixed_bytes = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN)
            .expect("resolved acoustic domain")
            .maximum_bytes();
        InvocationTensorPoolOwner::new(
            contract,
            plan,
            InvocationWorkspaceDomain::State {
                state,
                capacity: InvocationStateCapacity::SemanticBounded,
                placement: PlacementPolicy::BackendLocal,
                formula: WorkspaceFormula {
                    fixed_bytes,
                    dimensions: vec![],
                    terms: vec![],
                },
            },
            Device::Cpu,
            ModelInstanceId::new(1),
            1,
            1,
        )
        .expect("acoustic pool")
    }

    #[test]
    fn offline_acoustic_bound_covers_the_encoder_frames_of_its_audio_hint() {
        let shape = default_realtime_state_shape();
        let contract = nemotron_offline_invocation_contract(shape, StateDType::F32).unwrap();
        let hint_samples = (DEFAULT_MAX_AUDIO_SECONDS_HINT * SAMPLE_RATE as f32) as usize;
        let emitted = nemotron_offline_encoded_frames(hint_samples, shape);
        assert_eq!(emitted, 376);
        assert!(acoustic_sequence_max(&contract) >= emitted as u64);
    }

    #[test]
    fn offline_acoustic_capacity_rejects_over_bound_audio_before_install() {
        let shape = default_realtime_state_shape();
        let contract = nemotron_offline_invocation_contract(shape, StateDType::F32).unwrap();
        let pool = acoustic_pool(&contract);
        let lease = pool.lease().expect("acoustic lease");
        let frames = |seconds: usize| nemotron_offline_encoded_frames(seconds * 16_000, shape);

        assert!(ensure_offline_acoustic_capacity(&lease, frames(30)).is_ok());
        let error = ensure_offline_acoustic_capacity(&lease, frames(31))
            .expect_err("31 s must be rejected");
        assert!(matches!(error, Error::InvalidInput(_)), "{error:?}");
        let error = ensure_offline_acoustic_capacity(&lease, frames(600))
            .expect_err("600 s must be rejected");
        assert!(matches!(error, Error::InvalidInput(_)), "{error:?}");

        let arena = lease.arena().expect("arena");
        assert_eq!(arena.absolute_cursor(), 0);
        assert!(
            !arena.is_dirty(),
            "a rejected request must not touch the state"
        );
    }
}
