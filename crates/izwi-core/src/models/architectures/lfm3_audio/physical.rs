//! LFM3-Audio's invocation state: the language model's paged attention and
//! ShortConv ring, owned by each atomic request. The MaskGIT depth decoder
//! is stateless and the encoder and codec streams are bounded host state,
//! so no other domain exists. The domain ids are LFM2.5-Audio's, so the
//! engine's audio-chat leases serve both families.

use crate::engine::{NativeBatchMode, StageDescriptor};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InvocationLeaseScope,
    InvocationStageWorkspace, InvocationStateCapacity, InvocationWorkspaceDomain,
    InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy, RetainedStateCapability,
    StateCapacityAxis, StateCapacityBinding, StateDomainId, StateDomainSpec, WorkspaceFormula,
    CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::lfm2::config::Lfm2BackboneConfig;
use crate::models::architectures::lfm2::physical::{
    lfm2_main_invocation_contract, paged_f32_invocation_bytes, ring_f32_invocation_bytes,
    Lfm2StateIds, Lfm2StateLayout,
};
use crate::models::architectures::lfm25_audio::physical::{
    Lfm25AudioPhysicalStateSpec, LFM25_MAIN_ATTENTION_STATE_DOMAIN,
    LFM25_MAIN_SHORTCONV_STATE_DOMAIN, LFM25_MAIN_STATE_GROUP,
};

/// The invocation workspace of one scalar LFM3-Audio request.
pub(crate) fn lfm3_audio_physical_state_spec(
    lm_config: &Lfm2BackboneConfig,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<Lfm25AudioPhysicalStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "LFM3-Audio physical state has no execution graph".into(),
        ));
    }
    for stages in stage_graphs {
        if stages.len() != 1 {
            return Err(Error::ModelLoadError(
                "LFM3-Audio invocation state requires one scalar execution stage".into(),
            ));
        }
        for stage in *stages {
            stage.validate()?;
            if !stage.runs_independent_invocations() || stage.batch_mode != NativeBatchMode::None {
                return Err(Error::ModelLoadError(
                    "LFM3-Audio invocation state requires scalar atomic execution stages".into(),
                ));
            }
        }
    }
    let layout = Lfm2StateLayout::from_config(lm_config)?;
    let invocation = lfm2_main_invocation_contract(
        lm_config,
        &layout,
        Lfm2StateIds {
            attention: LFM25_MAIN_ATTENTION_STATE_DOMAIN,
            shortconv: LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
            main_group: LFM25_MAIN_STATE_GROUP,
        },
    )?;
    let context = u64::try_from(lm_config.context_length)
        .map_err(|_| Error::ModelLoadError("LFM3-Audio context exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("LFM3-Audio state contract is empty".into()))?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let stage = &stages[0];
        let mut domains = invocation
            .domains
            .iter()
            .cloned()
            .map(|state| {
                let (fixed_bytes, capacity) = match &state {
                    StateDomainSpec::PagedAttention(_) => {
                        let capacity = InvocationStateCapacity::AxisBoundPagedTokens {
                            binding: StateCapacityBinding::new(
                                StateCapacityAxis::DecoderContext,
                                1,
                                context,
                            )?,
                        };
                        (
                            paged_f32_invocation_bytes(
                                &state,
                                capacity.paged_max_tokens().expect("paged capacity"),
                            )?,
                            capacity,
                        )
                    }
                    StateDomainSpec::Ring(_) => (
                        ring_f32_invocation_bytes(&state)?,
                        InvocationStateCapacity::SemanticBounded,
                    ),
                    _ => {
                        return Err(Error::ModelLoadError(
                            "LFM3-Audio contract contains an unsupported state kind".into(),
                        ))
                    }
                };
                Ok(InvocationWorkspaceDomain::State {
                    placement: state.header().placement,
                    formula: WorkspaceFormula {
                        fixed_bytes,
                        dimensions: vec![],
                        terms: vec![],
                    },
                    state,
                    capacity,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if stage.max_workspace_bytes > 0 {
            domains.push(InvocationWorkspaceDomain::Scratch {
                id: StateDomainId::new(max_domain_id + 1),
                placement: PlacementPolicy::BackendLocal,
                alignment_bytes: 64,
                zero_on_release: false,
                formula: WorkspaceFormula {
                    fixed_bytes: stage.max_workspace_bytes,
                    dimensions: vec![],
                    terms: vec![],
                },
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: vec![InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: invocation.groups.clone(),
                domains,
            }],
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
    Ok(Lfm25AudioPhysicalStateSpec {
        descriptor,
        invocation,
    })
}
