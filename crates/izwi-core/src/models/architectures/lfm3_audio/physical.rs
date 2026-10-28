//! LFM3-Audio's invocation state: the language model's paged attention and
//! ShortConv ring, owned by each atomic request. The MaskGIT depth decoder
//! is stateless and the encoder and codec streams are bounded host state,
//! so no other domain exists. The domain ids are LFM2.5-Audio's, so the
//! engine's audio-chat leases serve both families.
//!
//! Each workload leases only the context it can use (a 30 s ASR chunk, one
//! TTS segment, a conversation), and paged KV is F16 on Metal and CUDA, so
//! the four capabilities of one loaded model reserve a few hundred MiB per
//! row instead of four full 32K-token contexts.

use crate::backends::BackendKind;
use crate::engine::{NativeBatchMode, StageDescriptor};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InvocationLeaseScope,
    InvocationStageWorkspace, InvocationStateCapacity, InvocationWorkspaceDomain,
    InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy, RetainedStateCapability,
    StateCapacityAxis, StateCapacityBinding, StateDType, StateDomainId, StateDomainSpec,
    WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::lfm2::config::Lfm2BackboneConfig;
use crate::models::architectures::lfm2::physical::{
    lfm2_main_invocation_contract_with_kv, paged_invocation_bytes, ring_f32_invocation_bytes,
    Lfm2StateIds, Lfm2StateLayout,
};
use crate::models::architectures::lfm25_audio::physical::{
    Lfm25AudioPhysicalStateSpec, Lfm25AudioRetainedStateSpec, LFM25_MAIN_ATTENTION_STATE_DOMAIN,
    LFM25_MAIN_SHORTCONV_STATE_DOMAIN, LFM25_MAIN_STATE_GROUP,
};

/// What a capability's invocation holds, which bounds its context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lfm3AudioWorkload {
    /// One ASR chunk of at most 30 s: the prompt, 375 encoder tokens and
    /// at most 512 text tokens.
    Asr,
    /// One TTS segment: the prompt with its text and at most 2,048 frames.
    Tts,
    /// A voice-chat turn: the history, the spoken turn and the reply.
    Conversation,
}

impl Lfm3AudioWorkload {
    /// Tokens of language-model context the workload's invocation leases.
    pub(crate) const fn context_tokens(self) -> usize {
        match self {
            Self::Asr => 2_048,
            Self::Tts => 4_096,
            Self::Conversation => 16_384,
        }
    }
}

/// The paged KV storage LFM3-Audio uses on `backend`: F16 on Metal and
/// CUDA, F32 on the CPU (as LFM chat stores it).
pub(crate) const fn lfm3_audio_kv_dtype(backend: BackendKind) -> StateDType {
    match backend {
        BackendKind::Cpu => StateDType::F32,
        BackendKind::Metal | BackendKind::Cuda => StateDType::F16,
    }
}

/// The invocation workspace of one scalar LFM3-Audio request of `workload`.
/// Speech as retained rows: the language model's paged attention and
/// ShortConv ring stay in the managed arena between quanta, sized for one
/// TTS segment per row. The depth decoder and codec keep no state between
/// quanta; each stage leases its activations as per-row scratch.
pub(crate) fn lfm3_audio_retained_tts_state_spec(
    lm_config: &Lfm2BackboneConfig,
    kv_dtype: StateDType,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<Lfm25AudioRetainedStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "LFM3-Audio retained TTS state has no execution graph".into(),
        ));
    }
    let retained = crate::models::architectures::lfm2::physical::lfm2_managed_cache_contract(
        lm_config, kv_dtype,
    )?;
    let max_domain_id = retained
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .unwrap_or(0);
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let per_row = if stage.workspace_per_row_bytes == 0 {
                stage.max_workspace_bytes / u64::try_from(stage.max_batch_size.max(1)).unwrap_or(1)
            } else {
                stage.workspace_per_row_bytes
            };
            let domains = if per_row == 0 {
                Vec::new()
            } else {
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError("LFM3-Audio stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("LFM3-Audio scratch domain overflow".into())
                    })?;
                vec![InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: per_row,
                        dimensions: vec![],
                        terms: vec![],
                    },
                }]
            };
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: Vec::new(),
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
        retained: RetainedStateCapability::Managed {
            contract: retained.clone(),
        },
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(Lfm25AudioRetainedStateSpec {
        descriptor,
        retained,
        retained_max_tokens: Lfm3AudioWorkload::Tts
            .context_tokens()
            .min(lm_config.context_length),
        main_invocation: None,
        depthformer_invocation: None,
    })
}

pub(crate) fn lfm3_audio_physical_state_spec(
    lm_config: &Lfm2BackboneConfig,
    workload: Lfm3AudioWorkload,
    kv_dtype: StateDType,
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
    let invocation = lfm2_main_invocation_contract_with_kv(
        lm_config,
        &layout,
        Lfm2StateIds {
            attention: LFM25_MAIN_ATTENTION_STATE_DOMAIN,
            shortconv: LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
            main_group: LFM25_MAIN_STATE_GROUP,
        },
        kv_dtype,
    )?;
    let context = u64::try_from(workload.context_tokens().min(lm_config.context_length))
        .map_err(|_| Error::ModelLoadError("LFM3-Audio context exceeds u64".into()))?;
    let kv_element_bytes = match kv_dtype {
        StateDType::F32 => 4,
        StateDType::F16 | StateDType::Bf16 => 2,
        other => {
            return Err(Error::ModelLoadError(format!(
                "LFM3-Audio paged KV cannot be stored as {other:?}"
            )))
        }
    };
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
                            paged_invocation_bytes(
                                &state,
                                capacity.paged_max_tokens().expect("paged capacity"),
                                kv_element_bytes,
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
