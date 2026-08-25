//! Physical invocation-state contract for the LFM2.5 Audio capability family.

use crate::engine::{
    NativeBatchMode, OutputVisibility, StageDescriptor, StageProgressKind, StageWorkSelector,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, AttentionMask, AttentionPattern, CapabilityStateDescriptorV2,
    InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationStateCapacity, InvocationWorkspaceDomain, InvocationWorkspaceProfile,
    InvocationWorkspaceSet, KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec,
    PagedAttentionLayerSpec, PlacementPolicy, RetainedStateCapability, StateCapacityAxis,
    StateCapacityBinding, StateClock, StateDType, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::lfm2::physical::{
    invocation_header, lfm2_main_invocation_contract, lfm2_managed_cache_contract,
    paged_f32_invocation_bytes, ring_f32_invocation_bytes, Lfm2StateIds, Lfm2StateLayout,
};
use crate::models::shared::attention::paged::default_kv_page_size;

use super::config::{
    Lfm25AudioDecoderConfig, Lfm2BackboneConfig, LFM25_DEPTHFORMER_KV_HEADS,
    LFM25_DEPTHFORMER_QUERY_HEADS,
};
use super::state::Lfm25AudioRetainedMode;

pub(crate) const LFM25_MAIN_ATTENTION_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const LFM25_MAIN_SHORTCONV_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);
pub(crate) const LFM25_DEPTHFORMER_STATE_DOMAIN: StateDomainId = StateDomainId::new(3);
pub(crate) const LFM25_MAIN_STATE_GROUP: StateGroupId = StateGroupId::new(1);
pub(crate) const LFM25_DEPTHFORMER_STATE_GROUP: StateGroupId = StateGroupId::new(2);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lfm25AudioStateMode {
    MainOnly,
    MainAndDepthformer,
}

#[derive(Debug, Clone)]
pub(crate) struct Lfm25AudioPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

#[derive(Debug, Clone)]
pub(crate) struct Lfm25AudioRetainedStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
    /// TTS resets this invocation cache for every emitted frame. ASR has no
    /// Depthformer and therefore no invocation state contract here.
    pub(crate) depthformer_invocation: Option<InferenceStateContract>,
}

pub(crate) fn lfm25_audio_physical_state_spec(
    main_config: &Lfm2BackboneConfig,
    decoder_config: &Lfm25AudioDecoderConfig,
    mode: Lfm25AudioStateMode,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<Lfm25AudioPhysicalStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "LFM2.5 Audio physical state has no execution graph".into(),
        ));
    }
    for stages in stage_graphs {
        if stages.len() != 1 {
            return Err(Error::ModelLoadError(
                "LFM2.5 Audio invocation state requires one scalar execution stage".into(),
            ));
        }
        for stage in *stages {
            stage.validate()?;
            if stage.progress != StageProgressKind::Atomic
                || stage.batch_mode != NativeBatchMode::None
            {
                return Err(Error::ModelLoadError(
                    "LFM2.5 Audio invocation state requires scalar atomic execution stages".into(),
                ));
            }
        }
    }
    let layout = Lfm2StateLayout::from_config(main_config)?;
    let mut invocation = lfm2_main_invocation_contract(
        main_config,
        &layout,
        Lfm2StateIds {
            attention: LFM25_MAIN_ATTENTION_STATE_DOMAIN,
            shortconv: LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
            main_group: LFM25_MAIN_STATE_GROUP,
        },
    )?;
    if mode == Lfm25AudioStateMode::MainAndDepthformer {
        invocation.domains.push(depthformer_domain(decoder_config)?);
        invocation.groups.push(StateGroupSpec {
            id: LFM25_DEPTHFORMER_STATE_GROUP,
            domains: vec![LFM25_DEPTHFORMER_STATE_DOMAIN],
            prefix_shareable: false,
        });
        invocation.validate()?;
    }

    let main_tokens = u64::try_from(main_config.context_length)
        .map_err(|_| Error::ModelLoadError("LFM2.5 Audio main context exceeds u64".into()))?;
    let codebook_steps = u64::try_from(decoder_config.codebooks)
        .map_err(|_| Error::ModelLoadError("LFM2.5 Audio codebook count exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("LFM2.5 Audio state contract is empty".into()))?
        .max(LFM25_DEPTHFORMER_STATE_DOMAIN.get());
    let mut profiles = Vec::with_capacity(stage_graphs.len());

    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let mut domains = invocation
                .domains
                .iter()
                .cloned()
                .map(|state| {
                    let (fixed_bytes, capacity) = match &state {
                        StateDomainSpec::PagedAttention(_) => {
                            let capacity = if state.id() == LFM25_DEPTHFORMER_STATE_DOMAIN {
                                InvocationStateCapacity::PagedTokens {
                                    max_tokens: codebook_steps,
                                }
                            } else {
                                InvocationStateCapacity::AxisBoundPagedTokens {
                                    binding: StateCapacityBinding::new(
                                        StateCapacityAxis::DecoderContext,
                                        1,
                                        main_tokens,
                                    )?,
                                }
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
                                "LFM2.5 Audio contract contains an unsupported state kind".into(),
                            ));
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
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError(
                            "LFM2.5 Audio execution stage count exceeds u32".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("LFM2.5 Audio scratch domain id overflow".into())
                    })?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
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
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: invocation.groups.clone(),
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
    Ok(Lfm25AudioPhysicalStateSpec {
        descriptor,
        invocation,
    })
}

/// Build the fail-closed retained state descriptor used by future ASR/TTS
/// sequence handlers. The main backbone is scheduler-owned and transactional;
/// Depthformer remains invocation-owned because the model resets it once per
/// generated audio frame.
pub(crate) fn lfm25_audio_retained_state_spec(
    main_config: &Lfm2BackboneConfig,
    decoder_config: &Lfm25AudioDecoderConfig,
    mode: Lfm25AudioRetainedMode,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<Lfm25AudioRetainedStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "LFM2.5 Audio retained state has no execution graph".into(),
        ));
    }
    let retained = lfm2_managed_cache_contract(main_config)?;
    let retained_max_tokens = main_config.context_length;
    let depthformer_invocation = if mode == Lfm25AudioRetainedMode::Tts {
        let domain = depthformer_domain(decoder_config)?;
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![domain],
            groups: vec![StateGroupSpec {
                id: LFM25_DEPTHFORMER_STATE_GROUP,
                domains: vec![LFM25_DEPTHFORMER_STATE_DOMAIN],
                prefix_shareable: false,
            }],
        };
        contract.validate()?;
        Some(contract)
    } else {
        None
    };
    let depthformer_bytes = depthformer_invocation
        .as_ref()
        .map(|contract| {
            paged_f32_invocation_bytes(
                contract
                    .domains
                    .first()
                    .expect("Depthformer invocation contract has one domain"),
                u64::try_from(decoder_config.codebooks).map_err(|_| {
                    Error::ModelLoadError("LFM2.5 Audio codebook count exceeds u64".into())
                })?,
            )
        })
        .transpose()?;

    let mut profiles = Vec::with_capacity(stage_graphs.len());
    let mut has_invocation_workspace = false;
    for stages in stage_graphs {
        validate_retained_stage_graph(mode, stages)?;
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let owns_depthformer = mode == Lfm25AudioRetainedMode::Tts
                && stage.selector == StageWorkSelector::SequenceDecode;
            let mut groups = Vec::new();
            let mut domains = Vec::new();
            if owns_depthformer {
                let contract = depthformer_invocation
                    .as_ref()
                    .expect("TTS retained topology has Depthformer invocation state");
                groups = contract.groups.clone();
                let state = contract.domains[0].clone();
                domains.push(InvocationWorkspaceDomain::State {
                    placement: state.header().placement,
                    formula: WorkspaceFormula {
                        fixed_bytes: depthformer_bytes
                            .expect("TTS Depthformer invocation bytes were computed"),
                        dimensions: vec![],
                        terms: vec![],
                    },
                    state,
                    capacity: InvocationStateCapacity::PagedTokens {
                        max_tokens: u64::try_from(decoder_config.codebooks).map_err(|_| {
                            Error::ModelLoadError("LFM2.5 Audio codebook count exceeds u64".into())
                        })?,
                    },
                });
            }
            if stage.max_workspace_bytes > 0 {
                let scratch_id = LFM25_DEPTHFORMER_STATE_DOMAIN
                    .get()
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError(
                            "LFM2.5 Audio retained stage count exceeds u32".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError(
                            "LFM2.5 Audio retained scratch domain id overflow".into(),
                        )
                    })?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
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
            has_invocation_workspace |= !domains.is_empty();
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups,
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
    let invocation = if has_invocation_workspace {
        InvocationWorkspaceSet::Bounded { profiles }
    } else {
        InvocationWorkspaceSet::None {
            stage_graph_fingerprints: profiles
                .into_iter()
                .map(|profile| profile.stage_graph_fingerprint)
                .collect(),
        }
    };
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: RetainedStateCapability::Managed {
            contract: retained.clone(),
        },
        invocation,
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(Lfm25AudioRetainedStateSpec {
        descriptor,
        retained,
        retained_max_tokens,
        depthformer_invocation,
    })
}

fn validate_retained_stage_graph(
    mode: Lfm25AudioRetainedMode,
    stages: &[StageDescriptor],
) -> Result<()> {
    let mut preparations = 0usize;
    let mut prefills = 0usize;
    let mut decodes = 0usize;
    for stage in stages {
        stage.validate()?;
        if stage.batch_mode != NativeBatchMode::None || stage.max_batch_size != 1 {
            return Err(Error::ModelLoadError(
                "LFM2.5 Audio retained handlers are dormant and cannot advertise native batching"
                    .into(),
            ));
        }
        if stage.output_visibility != OutputVisibility::AfterQuantumCommit {
            return Err(Error::ModelLoadError(
                "LFM2.5 Audio retained state requires post-commit output visibility".into(),
            ));
        }
        if stage.retained_state_selections.is_some() {
            return Err(Error::ModelLoadError(
                "LFM2.5 Audio retained main state advances only on the decoder-token clock".into(),
            ));
        }
        match stage.selector {
            StageWorkSelector::PreSequencePreparation
                if mode == Lfm25AudioRetainedMode::Asr
                    && stage.progress == StageProgressKind::Atomic =>
            {
                preparations += 1;
            }
            StageWorkSelector::SequencePrefill | StageWorkSelector::SequenceDecode
                if stage.progress == StageProgressKind::Iterative =>
            {
                if stage.selector == StageWorkSelector::SequencePrefill {
                    prefills += 1;
                } else {
                    decodes += 1;
                }
            }
            _ => {
                return Err(Error::ModelLoadError(
                    "LFM2.5 Audio retained graph contains an unsupported stage".into(),
                ));
            }
        }
    }
    let preparation_valid = match mode {
        Lfm25AudioRetainedMode::Asr => preparations == 1,
        Lfm25AudioRetainedMode::Tts => preparations == 0,
    };
    if !preparation_valid || prefills != 1 || decodes != 1 {
        return Err(Error::ModelLoadError(
            "LFM2.5 Audio retained graph has the wrong preparation/prefill/decode topology".into(),
        ));
    }
    Ok(())
}

fn depthformer_domain(config: &Lfm25AudioDecoderConfig) -> Result<StateDomainSpec> {
    if config.codebooks == 0
        || config.depthformer_layers == 0
        || config.depthformer_dim == 0
        || !config
            .depthformer_dim
            .is_multiple_of(LFM25_DEPTHFORMER_QUERY_HEADS)
        || !(config.depthformer_dim / LFM25_DEPTHFORMER_QUERY_HEADS).is_multiple_of(2)
        || !LFM25_DEPTHFORMER_QUERY_HEADS.is_multiple_of(LFM25_DEPTHFORMER_KV_HEADS)
    {
        return Err(Error::ModelLoadError(
            "LFM2.5 Audio physical state received invalid Depthformer geometry".into(),
        ));
    }
    let query_heads = u32::try_from(LFM25_DEPTHFORMER_QUERY_HEADS)
        .map_err(|_| Error::ModelLoadError("Depthformer query-head count exceeds u32".into()))?;
    let kv_heads = u32::try_from(LFM25_DEPTHFORMER_KV_HEADS)
        .map_err(|_| Error::ModelLoadError("Depthformer KV-head count exceeds u32".into()))?;
    let head_dim = u32::try_from(config.depthformer_dim / LFM25_DEPTHFORMER_QUERY_HEADS)
        .map_err(|_| Error::ModelLoadError("Depthformer head dimension exceeds u32".into()))?;
    let layers = (0..config.depthformer_layers)
        .map(|model_layer| {
            Ok(PagedAttentionLayerSpec {
                model_layer: u32::try_from(model_layer).map_err(|_| {
                    Error::ModelLoadError("Depthformer layer index exceeds u32".into())
                })?,
                query_heads,
                kv_heads,
                key_head_dim: head_dim,
                value_head_dim: head_dim,
                pattern: AttentionPattern::Full,
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Rotary {
                    rotary_dim: head_dim,
                },
                attention_logit_softcap: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let preferred_tokens = u32::try_from(default_kv_page_size())
        .map_err(|_| Error::ModelLoadError("Depthformer page size exceeds u32".into()))?;
    Ok(StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
        header: invocation_header(LFM25_DEPTHFORMER_STATE_DOMAIN, StateClock::CodebookSteps),
        layers,
        page_size: PageSizeConstraint {
            min_tokens: 1,
            preferred_tokens,
            max_tokens: preferred_tokens.max(256),
            multiple_of: 1,
        },
        accepted_dtypes: vec![StateDType::F32],
    }))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::{
        ConcurrencyClass, ExecutionDomain, MembershipSafePoint, NativeBatchMode, OutputVisibility,
        StageDescriptor, StageId, StageProgressKind, StageShapePolicy, StageWorkSelector,
    };

    fn main_config() -> Lfm2BackboneConfig {
        Lfm2BackboneConfig {
            architecture: "lfm2".into(),
            block_count: 5,
            context_length: 17,
            embedding_length: 64,
            embedding_length_out: None,
            feed_forward_length: Some(128),
            attention_head_count: 4,
            attention_head_count_kv: vec![0, 2, 0, 2, 0],
            attention_layer_norm_rms_epsilon: 1e-5,
            attention_sliding_window: Some(8),
            rope_freq_base: 1_000_000.0,
            shortconv_l_cache: 3,
        }
    }

    fn decoder_config() -> Lfm25AudioDecoderConfig {
        Lfm25AudioDecoderConfig {
            codebooks: 8,
            audio_vocab_size: 2_049,
            audio_end_token_id: 2_048,
            depthformer_layers: 3,
            depthformer_dim: 64,
            output_sample_rate: 24_000,
            output_n_fft: 1_280,
            output_hop_length: 320,
            detokenizer_upsample_factor: 6,
            interleaved_n_text: 6,
            interleaved_n_audio: 12,
        }
    }

    fn stage() -> StageDescriptor {
        StageDescriptor {
            id: StageId::new(0),
            name: "lfm25_audio.scalar".into(),
            domain: ExecutionDomain::ExecutionGroup,
            selector: StageWorkSelector::Atomic,
            progress: StageProgressKind::Atomic,
            batch_mode: NativeBatchMode::None,
            concurrency: ConcurrencyClass::Exclusive,
            physical_launch_policy: crate::engine::PhysicalLaunchPolicy::ExecutionGroupExclusive,
            max_batch_size: 1,
            max_work_units: 1,
            workspace_base_bytes: 256,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes: 256,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
            shape_policy: StageShapePolicy::Exact,
            membership_safe_point: MembershipSafePoint::OperationBoundary,
            output_visibility: OutputVisibility::AfterQuantumCommit,
            retained_state_selections: None,
        }
    }

    fn retained_stage(id: u32, selector: StageWorkSelector, workspace: u64) -> StageDescriptor {
        let mut stage = stage();
        stage.id = StageId::new(id);
        stage.name = format!("lfm25_audio.retained.{id}");
        stage.selector = selector;
        stage.progress = if selector == StageWorkSelector::PreSequencePreparation {
            StageProgressKind::Atomic
        } else {
            StageProgressKind::Iterative
        };
        stage.workspace_base_bytes = workspace;
        stage.max_workspace_bytes = workspace;
        stage
    }

    #[test]
    fn main_only_contract_excludes_depthformer_state() {
        let stage = stage();
        let spec = lfm25_audio_physical_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioStateMode::MainOnly,
            &[&[stage]],
        )
        .unwrap();

        assert_eq!(spec.invocation.domains.len(), 2);
        assert_eq!(spec.invocation.groups.len(), 1);
        assert_eq!(
            spec.invocation.groups[0].domains,
            vec![
                LFM25_MAIN_ATTENTION_STATE_DOMAIN,
                LFM25_MAIN_SHORTCONV_STATE_DOMAIN
            ]
        );
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("audio state must use bounded invocation workspaces");
        };
        let workspace_domains = &profiles[0].stages[0].domains;
        assert_eq!(workspace_domains.len(), 3);
        assert_eq!(workspace_domains[0].id(), LFM25_MAIN_ATTENTION_STATE_DOMAIN);
        assert_eq!(workspace_domains[1].id(), LFM25_MAIN_SHORTCONV_STATE_DOMAIN);
        assert_eq!(
            workspace_domains[2].id(),
            StateDomainId::new(LFM25_DEPTHFORMER_STATE_DOMAIN.get() + 1)
        );
    }

    #[test]
    fn generation_contract_separates_main_and_per_frame_state() {
        let stage = stage();
        let spec = lfm25_audio_physical_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioStateMode::MainAndDepthformer,
            &[&[stage]],
        )
        .unwrap();

        assert_eq!(spec.invocation.domains.len(), 3);
        assert_eq!(spec.invocation.groups.len(), 2);
        assert_eq!(
            spec.invocation.groups[1].domains,
            vec![LFM25_DEPTHFORMER_STATE_DOMAIN]
        );
        let StateDomainSpec::PagedAttention(depthformer) = &spec.invocation.domains[2] else {
            panic!("third domain must be Depthformer paged attention");
        };
        assert_eq!(depthformer.header.clock, StateClock::CodebookSteps);
        assert_eq!(depthformer.layers.len(), 3);
        assert!(depthformer
            .layers
            .iter()
            .all(|layer| layer.query_heads == 32 && layer.kv_heads == 8));
        assert!(depthformer.layers.iter().all(|layer| {
            layer.pattern == AttentionPattern::Full
                && layer.mask == AttentionMask::Causal
                && layer.key_head_dim == 2
                && layer.value_head_dim == 2
                && layer.key_encoding == KeyEncoding::Rotary { rotary_dim: 2 }
        }));
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("audio state must use bounded invocation workspaces");
        };
        let InvocationWorkspaceDomain::State { capacity, .. } = &profiles[0].stages[0].domains[2]
        else {
            panic!("Depthformer workspace must be state");
        };
        assert_eq!(
            *capacity,
            InvocationStateCapacity::PagedTokens { max_tokens: 8 }
        );
    }

    #[test]
    fn invalid_depthformer_geometry_fails_closed() {
        let stage = stage();
        let mut decoder = decoder_config();
        decoder.depthformer_dim = 63;
        let error = lfm25_audio_physical_state_spec(
            &main_config(),
            &decoder,
            Lfm25AudioStateMode::MainAndDepthformer,
            &[&[stage]],
        )
        .expect_err("invalid geometry must be rejected");
        assert!(error.to_string().contains("Depthformer geometry"));
    }

    #[test]
    fn odd_depthformer_head_dimension_fails_closed() {
        let stage = stage();
        let mut decoder = decoder_config();
        decoder.depthformer_dim = 96;
        let error = lfm25_audio_physical_state_spec(
            &main_config(),
            &decoder,
            Lfm25AudioStateMode::MainAndDepthformer,
            &[&[stage]],
        )
        .expect_err("odd rotary head dimension must be rejected");
        assert!(error.to_string().contains("Depthformer geometry"));
    }

    #[test]
    fn non_scalar_or_multi_stage_graphs_fail_closed() {
        let mut batched = stage();
        batched.batch_mode = NativeBatchMode::Static;
        batched.concurrency = ConcurrencyClass::Batchable;
        batched.max_batch_size = 2;
        let batched_error = lfm25_audio_physical_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioStateMode::MainOnly,
            &[&[batched]],
        )
        .expect_err("native batching must be rejected");
        assert!(batched_error.to_string().contains("scalar atomic"));

        let first = stage();
        let mut second = stage();
        second.id = StageId::new(2);
        let multi_error = lfm25_audio_physical_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioStateMode::MainOnly,
            &[&[first, second]],
        )
        .expect_err("multi-stage ownership must be rejected");
        assert!(multi_error.to_string().contains("one scalar"));
    }

    #[test]
    fn retained_asr_contract_moves_main_state_out_of_invocation_workspaces() {
        let preparation = retained_stage(0, StageWorkSelector::PreSequencePreparation, 128);
        let prefill = retained_stage(1, StageWorkSelector::SequencePrefill, 64);
        let decode = retained_stage(2, StageWorkSelector::SequenceDecode, 32);
        let stages = [preparation, prefill, decode];
        let spec = lfm25_audio_retained_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioRetainedMode::Asr,
            &[&stages],
        )
        .expect("retained ASR contract");

        assert_eq!(spec.retained_max_tokens, 17);
        assert!(spec.depthformer_invocation.is_none());
        assert!(spec.retained.domains.iter().all(|domain| {
            domain.header().scope == crate::kv::v2::StateScope::Retained
                && domain.header().checkpoint == crate::kv::v2::CheckpointPolicy::Transactional
        }));
        assert!(matches!(
            spec.descriptor.retained,
            RetainedStateCapability::Managed { .. }
        ));
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("ASR stage scratch must be bounded");
        };
        assert!(profiles[0].stages.iter().all(|stage| stage
            .domains
            .iter()
            .all(|domain| matches!(domain, InvocationWorkspaceDomain::Scratch { .. }))));
    }

    #[test]
    fn retained_tts_keeps_depthformer_on_decode_invocation_only() {
        let prefill = retained_stage(1, StageWorkSelector::SequencePrefill, 0);
        let decode = retained_stage(2, StageWorkSelector::SequenceDecode, 64);
        let stages = [prefill, decode];
        let spec = lfm25_audio_retained_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioRetainedMode::Tts,
            &[&stages],
        )
        .expect("retained TTS contract");

        assert_eq!(spec.retained.domains.len(), 2);
        assert!(spec
            .retained
            .domains
            .iter()
            .all(|domain| domain.id() != LFM25_DEPTHFORMER_STATE_DOMAIN));
        let depthformer = spec
            .depthformer_invocation
            .as_ref()
            .expect("TTS Depthformer invocation contract");
        assert_eq!(depthformer.domains.len(), 1);
        assert_eq!(
            depthformer.domains[0].header().scope,
            crate::kv::v2::StateScope::Invocation
        );
        assert_eq!(
            depthformer.domains[0].header().clock,
            StateClock::CodebookSteps
        );

        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("TTS decode invocation state must be bounded");
        };
        assert!(profiles[0].stages[0].domains.is_empty());
        assert!(matches!(
            profiles[0].stages[1].domains[0],
            InvocationWorkspaceDomain::State { .. }
        ));
    }

    #[test]
    fn retained_contract_rejects_unimplemented_native_batching_and_atomic_routes() {
        let prefill = retained_stage(1, StageWorkSelector::SequencePrefill, 0);
        let mut decode = retained_stage(2, StageWorkSelector::SequenceDecode, 0);
        decode.batch_mode = NativeBatchMode::Continuous;
        decode.concurrency = ConcurrencyClass::Batchable;
        decode.max_batch_size = 2;
        decode.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        decode.shape_policy = StageShapePolicy::Ragged;
        let stages = [prefill.clone(), decode];
        assert!(lfm25_audio_retained_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioRetainedMode::Tts,
            &[&stages],
        )
        .is_err());

        let atomic = stage();
        let stages = [atomic, prefill];
        assert!(lfm25_audio_retained_state_spec(
            &main_config(),
            &decoder_config(),
            Lfm25AudioRetainedMode::Asr,
            &[&stages],
        )
        .is_err());
    }
}
