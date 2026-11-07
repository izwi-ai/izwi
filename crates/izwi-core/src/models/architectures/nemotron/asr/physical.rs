//! Physical state for Nemotron's offline RNNT decoder: the retained
//! predictor of short clips decoded as rows, and the invocation predictor
//! and acoustic state of the atomic long-form graph.

use crate::engine::InvocationTensorLease;
use crate::engine::{StageDescriptor, StageWorkSelector};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, BoundedShape, CapabilityStateDescriptorV2, CheckpointPolicy,
    InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy, PrefixPolicy, ShapeAxis,
    ShapeDimension, ShapeExtent, StateClock, StateComponentId, StateDType, StateDomainHeader,
    StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope,
    StaticTensorDomainSpec, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
    CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::state::typed_invocation_descriptor;

use super::network::causal_subsampled_len;
use super::{NemotronRealtimeStateShape, DEFAULT_MAX_AUDIO_SECONDS_HINT, SAMPLE_RATE};

pub(crate) const NEMOTRON_OFFLINE_PREDICTOR_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const NEMOTRON_OFFLINE_ACOUSTIC_DOMAIN: StateDomainId = StateDomainId::new(2);
pub(crate) const NEMOTRON_OFFLINE_GROUP: StateGroupId = StateGroupId::new(1);
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
    /// Sequence graphs (short clips decoded as rows) own the predictor
    /// tensors as retained state and publish scratch-free stage workspace.
    pub(crate) retained: Option<InferenceStateContract>,
    /// Atomic graphs (long audio and diarization) lease the predictor and
    /// the encoder projection per invocation.
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn nemotron_offline_physical_state_spec(
    shape: NemotronRealtimeStateShape,
    dtype: StateDType,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<NemotronOfflinePhysicalStateSpec> {
    let invocation = nemotron_offline_invocation_contract(shape, dtype)?;
    let retained = nemotron_retained_predictor_contract(&invocation)?;
    let is_retained_stage = |stage: &StageDescriptor| {
        matches!(
            stage.selector,
            StageWorkSelector::SequencePrefill | StageWorkSelector::SequenceDecode
        )
    };
    let (retained_graphs, atomic_graphs): (Vec<_>, Vec<_>) = stage_graphs
        .iter()
        .copied()
        .partition(|stages| stages.iter().any(is_retained_stage));
    let descriptor = match (retained_graphs.is_empty(), atomic_graphs.is_empty()) {
        (true, _) => typed_invocation_descriptor(stage_graphs, &invocation)?,
        (false, true) => {
            CapabilityStateDescriptorV2::managed_for_stage_graphs(retained.clone(), stage_graphs)?
        }
        (false, false) => retained_with_atomic_profiles(
            CapabilityStateDescriptorV2::managed_for_stage_graphs(
                retained.clone(),
                &retained_graphs,
            )?,
            &retained_graphs,
            typed_invocation_descriptor(&atomic_graphs, &invocation)?,
            stage_graphs,
        )?,
    };
    Ok(NemotronOfflinePhysicalStateSpec {
        descriptor,
        retained: (!retained_graphs.is_empty()).then_some(retained),
        invocation,
    })
}

/// One publication for both lifetimes: the sequence graphs keep their
/// scratch-free profiles under the retained contract, and each atomic graph
/// adds its invocation predictor and acoustic profile.
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
        return Err(model_load(
            "Nemotron atomic graph must bound its invocation workspace",
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

/// The predictor domain of the invocation contract, retained and
/// transactional: the recurrent state a row's decode carries between
/// scheduler steps.
fn nemotron_retained_predictor_contract(
    invocation: &InferenceStateContract,
) -> Result<InferenceStateContract> {
    let predictor = invocation
        .domains
        .iter()
        .find(|domain| domain.id() == NEMOTRON_OFFLINE_PREDICTOR_DOMAIN)
        .cloned()
        .ok_or_else(|| model_load("Nemotron offline state has no predictor domain"))?;
    let StateDomainSpec::Tensor(mut predictor) = predictor else {
        return Err(model_load(
            "Nemotron retained predictor state must be tensor state",
        ));
    };
    predictor.header.scope = StateScope::Retained;
    predictor.header.checkpoint = CheckpointPolicy::Transactional;
    predictor.header.prefix = PrefixPolicy::Disabled;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::Tensor(predictor)],
        groups: vec![StateGroupSpec {
            id: NEMOTRON_OFFLINE_GROUP,
            domains: vec![NEMOTRON_OFFLINE_PREDICTOR_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract
        .validate()
        .map_err(|error| model_load(format!("invalid retained Nemotron state: {error}")))?;
    Ok(contract)
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

    /// The offline graphs publish the retained predictor for the graph that
    /// decodes short clips as rows, beside the invocation predictor and
    /// encoder projection the atomic graphs lease.
    #[test]
    fn offline_spec_retains_the_predictor_for_rows_beside_the_atomic_invocation() {
        use crate::engine::{ExecutionGroupId, ExecutionMode};
        use crate::kv::v2::{
            stage_graph_fingerprint, InvocationWorkspaceDomain, InvocationWorkspaceSet,
        };
        use crate::runtime::{CapabilityKind, LoadedModelBundleDraft, RuntimeAdapterRegistry};

        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in_with_execution_limits(4, 4).unwrap(),
            ExecutionGroupId::new(1),
            ModelInstanceId::new(1),
            crate::model::ModelVariant::Nemotron35AsrStreaming06B,
            BackendKind::Cpu,
        )
        .unwrap();
        let contracts = draft.execution_contracts(CapabilityKind::Asr).unwrap();
        let graphs = contracts
            .iter()
            .map(|contract| contract.stages.as_ref())
            .collect::<Vec<_>>();
        let spec = nemotron_offline_physical_state_spec(
            default_realtime_state_shape(),
            StateDType::F32,
            &graphs,
        )
        .unwrap();
        let retained = spec
            .retained
            .as_ref()
            .expect("row graphs retain the predictor");
        let [StateDomainSpec::Tensor(predictor)] = retained.domains.as_slice() else {
            panic!("the retained state is the predictor's tensor domain alone");
        };
        assert_eq!(predictor.header.id, NEMOTRON_OFFLINE_PREDICTOR_DOMAIN);
        assert_eq!(predictor.header.scope, StateScope::Retained);
        assert_eq!(predictor.header.checkpoint, CheckpointPolicy::Transactional);
        assert_eq!(predictor.components.len(), 6);
        for graph in &graphs {
            spec.descriptor.validate_against_stages(graph).unwrap();
        }
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("the atomic graphs bound their invocation workspace");
        };
        let mut atomic_graphs = 0;
        for contract in &contracts {
            let fingerprint = stage_graph_fingerprint(&contract.stages).unwrap();
            let profile = profiles
                .iter()
                .find(|profile| profile.stage_graph_fingerprint == fingerprint)
                .expect("every graph has a profile");
            let domains = profile
                .stages
                .iter()
                .flat_map(|stage| &stage.domains)
                .filter(|domain| matches!(domain, InvocationWorkspaceDomain::State { .. }))
                .count();
            if contract.execution_profile.mode == ExecutionMode::Atomic {
                atomic_graphs += 1;
                assert_eq!(domains, 2, "predictor and encoder projection");
            } else {
                assert_eq!(domains, 0, "a row's predictor is retained");
            }
        }
        assert!(atomic_graphs > 0);
    }
}
