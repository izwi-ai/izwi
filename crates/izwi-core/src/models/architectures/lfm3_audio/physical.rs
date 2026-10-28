//! LFM3-Audio's state: the language model's paged attention and ShortConv
//! ring. Rows of every capability keep them in one managed arena between
//! quanta; long-form transcription leases them for one atomic invocation.
//! The MaskGIT depth decoder is stateless and the encoder and codec streams
//! are bounded host state, so no other domain exists. The domain ids are
//! LFM2.5-Audio's, so the engine's audio-chat leases serve both families.
//!
//! Each workload holds only the context it can use (a 30 s ASR chunk, one
//! TTS segment, a conversation), and paged KV is F16 on Metal and CUDA.

use crate::backends::BackendKind;
use crate::engine::{StageDescriptor, StageWorkSelector};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    RetainedStateCapability, StateCapacityAxis, StateCapacityBinding, StateDType, StateDomainId,
    StateDomainSpec, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::lfm2::config::Lfm2BackboneConfig;
use crate::models::architectures::lfm2::physical::{
    lfm2_main_invocation_contract_with_kv, paged_invocation_bytes, ring_f32_invocation_bytes,
    Lfm2StateIds, Lfm2StateLayout,
};
use crate::models::architectures::lfm25_audio::physical::{
    Lfm25AudioRetainedStateSpec, LFM25_MAIN_ATTENTION_STATE_DOMAIN,
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

/// The language model's invocation workspace under the LFM2.5-Audio main
/// ids: its paged attention and ShortConv ring, in `kv_dtype`. The atomic
/// routes and the diarization pipeline lease it.
fn main_invocation_contract(
    lm_config: &Lfm2BackboneConfig,
    kv_dtype: StateDType,
) -> Result<InferenceStateContract> {
    let layout = Lfm2StateLayout::from_config(lm_config)?;
    lfm2_main_invocation_contract_with_kv(
        lm_config,
        &layout,
        Lfm2StateIds {
            attention: LFM25_MAIN_ATTENTION_STATE_DOMAIN,
            shortconv: LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
            main_group: LFM25_MAIN_STATE_GROUP,
        },
        kv_dtype,
    )
}

/// The workspace domains of one scalar invocation of `invocation` over
/// `context` tokens.
fn invocation_state_domains(
    invocation: &InferenceStateContract,
    context: usize,
    kv_dtype: StateDType,
) -> Result<Vec<InvocationWorkspaceDomain>> {
    let context = u64::try_from(context)
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
    invocation
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
        .collect()
}

/// Per-row scratch for `bytes` of a stage's activations, as domain `id`.
fn scratch_domain(id: u32, bytes: u64) -> InvocationWorkspaceDomain {
    InvocationWorkspaceDomain::Scratch {
        id: StateDomainId::new(id),
        placement: PlacementPolicy::BackendLocal,
        alignment_bytes: 64,
        zero_on_release: false,
        formula: WorkspaceFormula {
            fixed_bytes: bytes,
            dimensions: vec![],
            terms: vec![],
        },
    }
}

/// Rows of `workload` (speech, transcription or voice-chat replies): the
/// language model's paged attention and ShortConv ring stay in the managed
/// arena between quanta, at most one `workload` context per row. The depth
/// decoder and codec keep no state between quanta, so each row stage leases
/// its activations as per-row scratch. Transcription's long-form graph (one
/// atomic stage) also leases the language model's main invocation
/// workspace.
pub(crate) fn lfm3_audio_retained_state_spec(
    lm_config: &Lfm2BackboneConfig,
    workload: Lfm3AudioWorkload,
    kv_dtype: StateDType,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<Lfm25AudioRetainedStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "LFM3-Audio retained state has no execution graph".into(),
        ));
    }
    let retained = crate::models::architectures::lfm2::physical::lfm2_managed_cache_contract(
        lm_config, kv_dtype,
    )?;
    let context = workload.context_tokens().min(lm_config.context_length);
    let main_invocation = (workload == Lfm3AudioWorkload::Asr)
        .then(|| main_invocation_contract(lm_config, kv_dtype))
        .transpose()?;
    let max_domain_id = retained
        .domains
        .iter()
        .chain(
            main_invocation
                .iter()
                .flat_map(|contract| contract.domains.iter()),
        )
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
            let scratch_id = max_domain_id
                .checked_add(u32::try_from(index + 1).map_err(|_| {
                    Error::ModelLoadError("LFM3-Audio stage count exceeds u32".into())
                })?)
                .ok_or_else(|| {
                    Error::ModelLoadError("LFM3-Audio scratch domain overflow".into())
                })?;
            let long_form = stage.selector == StageWorkSelector::Atomic;
            let (groups, mut domains) = match (&main_invocation, long_form) {
                (Some(main), true) => (
                    main.groups.clone(),
                    invocation_state_domains(main, context, kv_dtype)?,
                ),
                (None, true) => {
                    return Err(Error::ModelLoadError(
                        "LFM3-Audio speech has no atomic graph".into(),
                    ))
                }
                _ => (Vec::new(), Vec::new()),
            };
            if per_row > 0 {
                domains.push(scratch_domain(scratch_id, per_row));
            }
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
        retained_max_tokens: context,
        main_invocation,
        depthformer_invocation: None,
    })
}
