//! Physical invocation state for Sortformer's bounded streaming speaker cache.

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

use super::{SortformerStreamingConfig, MAX_SUPPORTED_SPEAKERS};

pub(crate) const SORTFORMER_STREAMING_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
const SORTFORMER_STREAMING_STATE_GROUP: StateGroupId = StateGroupId::new(1);

#[derive(Debug, Clone)]
pub(crate) struct SortformerPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn sortformer_physical_state_spec(
    cfg: SortformerStreamingConfig,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<SortformerPhysicalStateSpec> {
    cfg.validate()?;
    let invocation = sortformer_invocation_contract(cfg)?;
    let descriptor = typed_invocation_descriptor(stage_graphs, &invocation)?;
    Ok(SortformerPhysicalStateSpec {
        descriptor,
        invocation,
    })
}

fn sortformer_invocation_contract(
    cfg: SortformerStreamingConfig,
) -> Result<InferenceStateContract> {
    let component = |id, role, dimensions| TensorComponentSpec {
        id: StateComponentId::new(id),
        role,
        shape: BoundedShape { dimensions },
        accepted_dtypes: vec![StateDType::F32],
    };
    let rows = |count, width_axis, width| {
        vec![
            fixed(ShapeAxis::Frames, count as u64),
            fixed(width_axis, width as u64),
        ]
    };
    let mut components = vec![
        component(
            1,
            TensorRole::RetainedEmbedding,
            rows(cfg.spkcache_len, ShapeAxis::Hidden, cfg.fc_d_model),
        ),
        component(
            2,
            TensorRole::RetainedLogits,
            rows(
                cfg.spkcache_len,
                ShapeAxis::Custom("speakers".into()),
                MAX_SUPPORTED_SPEAKERS,
            ),
        ),
        component(
            5,
            TensorRole::RetainedEmbedding,
            vec![fixed(ShapeAxis::Hidden, cfg.fc_d_model as u64)],
        ),
        component(
            6,
            TensorRole::Control,
            vec![fixed(ShapeAxis::Custom("control".into()), 4)],
        ),
    ];
    if cfg.fifo_len > 0 {
        components.insert(
            2,
            component(
                3,
                TensorRole::EncoderMemory,
                rows(cfg.fifo_len, ShapeAxis::Hidden, cfg.fc_d_model),
            ),
        );
        components.insert(
            3,
            component(
                4,
                TensorRole::RetainedLogits,
                rows(
                    cfg.fifo_len,
                    ShapeAxis::Custom("speakers".into()),
                    MAX_SUPPORTED_SPEAKERS,
                ),
            ),
        );
    }
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: SORTFORMER_STREAMING_STATE_DOMAIN,
                scope: StateScope::Invocation,
                clock: StateClock::AudioFrames,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::None,
            },
            components,
        })],
        groups: vec![StateGroupSpec {
            id: SORTFORMER_STREAMING_STATE_GROUP,
            domains: vec![SORTFORMER_STREAMING_STATE_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract
        .validate()
        .map_err(|error| model_load(format!("invalid Sortformer physical state: {error}")))?;
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
    use super::*;

    #[test]
    fn contract_covers_every_streaming_cache_component() {
        let contract =
            sortformer_invocation_contract(super::super::production_workspace_streaming_config())
                .unwrap();
        assert_eq!(contract.domains.len(), 1);
        let StateDomainSpec::Tensor(state) = &contract.domains[0] else {
            panic!("Sortformer state must be a tensor domain");
        };
        assert_eq!(state.components.len(), 6);
        assert_eq!(
            contract.groups[0].domains,
            vec![SORTFORMER_STREAMING_STATE_DOMAIN]
        );
    }

    #[test]
    fn zero_fifo_model_profile_omits_empty_tensor_components() {
        let cfg = super::super::resolve_streaming_config(
            crate::catalog::ModelVariant::DiarStreamingSortformer4SpkV21,
            &super::super::SortformerModulesConfig::default(),
            512,
        )
        .unwrap();
        assert_eq!(cfg.fifo_len, 0);

        let contract = sortformer_invocation_contract(cfg).unwrap();
        let StateDomainSpec::Tensor(state) = &contract.domains[0] else {
            panic!("Sortformer state must be a tensor domain");
        };
        assert_eq!(
            state
                .components
                .iter()
                .map(|component| component.id.get())
                .collect::<Vec<_>>(),
            vec![1, 2, 5, 6]
        );
    }
}
