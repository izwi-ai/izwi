//! Native VibeVoice model components.

use std::collections::BTreeMap;

use candle_core::DType;

use crate::engine::{
    ClockedStateSelection, ConcurrencyClass, ExecutionDomain, MembershipSafePoint, NativeBatchMode,
    OutputVisibility, PhysicalLaunchPolicy, StageDescriptor, StageId, StageProgressKind,
    StageShapePolicy, StageWorkSelector,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
    WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::qwen3::core::Qwen3Model;
use crate::models::architectures::vibevoice::tokenizer::VibeVoiceTokenizerStateComponentGeometry;

pub mod asr;
pub mod config;
pub mod connector;
pub mod diffusion;
pub mod prompt;
pub mod tokenizer;
pub mod tts;

pub(crate) const VIBEVOICE_ASR_DECODER_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const VIBEVOICE_ASR_ACOUSTIC_DOMAIN: StateDomainId = StateDomainId::new(2);
pub(crate) const VIBEVOICE_ASR_SEMANTIC_DOMAIN: StateDomainId = StateDomainId::new(3);
pub(crate) const VIBEVOICE_ASR_DECODER_GROUP: StateGroupId = StateGroupId::new(1);
pub(crate) const VIBEVOICE_ASR_TOKENIZER_GROUP: StateGroupId = StateGroupId::new(2);
pub(crate) const VIBEVOICE_ASR_PREPARATION_STAGE: &str = "asr.encoder.vibevoice";
pub(crate) const VIBEVOICE_ASR_PREFILL_STAGE: &str = "asr.prefill.tensor_static";
pub(crate) const VIBEVOICE_ASR_DECODE_STAGE: &str = "asr.decode.tensor_continuous";
pub(crate) const VIBEVOICE_ASR_LEGACY_STAGE: &str = "asr.scalar";
pub(crate) const VIBEVOICE_TTS_POSITIVE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const VIBEVOICE_TTS_NEGATIVE_DOMAIN: StateDomainId = StateDomainId::new(2);
pub(crate) const VIBEVOICE_TTS_ACOUSTIC_DOMAIN: StateDomainId = StateDomainId::new(3);
pub(crate) const VIBEVOICE_TTS_SEMANTIC_DOMAIN: StateDomainId = StateDomainId::new(4);

#[derive(Debug, Clone)]
pub(crate) struct VibeVoicePhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    /// Present only for the authenticated normal VibeVoice-ASR sequence graph.
    /// Legacy VibeVoice ASR and VibeVoice TTS remain invocation-only.
    pub(crate) retained: Option<InferenceStateContract>,
    pub(crate) retained_max_tokens: Option<usize>,
    pub(crate) invocation: InferenceStateContract,
}

#[derive(Debug, Clone)]
pub(crate) struct VibeVoiceTokenizerStateDomain {
    domain: StateDomainId,
    group: StateGroupId,
    clock: StateClock,
    components: Vec<VibeVoiceTokenizerStateComponentGeometry>,
}

impl VibeVoiceTokenizerStateDomain {
    pub(crate) fn new(
        domain: StateDomainId,
        group: StateGroupId,
        clock: StateClock,
        components: Vec<VibeVoiceTokenizerStateComponentGeometry>,
    ) -> Result<Self> {
        if domain.get() == 0 || group.get() == 0 || components.is_empty() {
            return Err(Error::ModelLoadError(
                "VibeVoice tokenizer state requires non-zero identities and components".into(),
            ));
        }
        if components
            .iter()
            .any(|component| component.channels == 0 || component.frames == 0)
        {
            return Err(Error::ModelLoadError(
                "VibeVoice tokenizer state has zero-sized convolution geometry".into(),
            ));
        }
        Ok(Self {
            domain,
            group,
            clock,
            components,
        })
    }
}

pub(crate) fn vibevoice_invocation_contract(
    model: &Qwen3Model,
    dtype: DType,
    preferred_page_tokens: usize,
    domains: &[StateDomainId],
    tokenizer_domains: &[VibeVoiceTokenizerStateDomain],
) -> Result<InferenceStateContract> {
    if domains.is_empty() {
        return Err(Error::ModelLoadError(
            "VibeVoice invocation state has no cache domains".into(),
        ));
    }
    let mut state_domains = Vec::with_capacity(domains.len());
    let mut groups = Vec::with_capacity(domains.len());
    for domain in domains {
        let contract =
            model.managed_inference_state_contract(*domain, dtype, preferred_page_tokens)?;
        state_domains.extend(contract.domains);
        groups.extend(contract.groups);
    }
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: state_domains,
        groups,
    };
    vibevoice_invocation_contract_from_state(contract, dtype, tokenizer_domains)
}

fn vibevoice_invocation_contract_from_state(
    mut contract: InferenceStateContract,
    dtype: DType,
    tokenizer_domains: &[VibeVoiceTokenizerStateDomain],
) -> Result<InferenceStateContract> {
    for domain in &mut contract.domains {
        let StateDomainSpec::PagedAttention(domain) = domain else {
            return Err(Error::ModelLoadError(
                "VibeVoice invocation state must be paged attention".into(),
            ));
        };
        domain.header.scope = StateScope::Invocation;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::None;
    }
    for group in &mut contract.groups {
        group.prefix_shareable = false;
    }
    append_tokenizer_state_domains(&mut contract, dtype, tokenizer_domains)?;
    contract.validate()?;
    Ok(contract)
}

fn append_tokenizer_state_domains(
    contract: &mut InferenceStateContract,
    dtype: DType,
    tokenizer_domains: &[VibeVoiceTokenizerStateDomain],
) -> Result<()> {
    if tokenizer_domains.is_empty() {
        return Ok(());
    }
    let state_dtype = vibevoice_state_dtype(dtype)?;
    let mut grouped = BTreeMap::<StateGroupId, Vec<StateDomainId>>::new();
    for authored in tokenizer_domains {
        if contract
            .domains
            .iter()
            .any(|domain| domain.id() == authored.domain)
        {
            return Err(Error::ModelLoadError(format!(
                "VibeVoice tokenizer domain {} collides with decoder state",
                authored.domain.get()
            )));
        }
        let components = authored
            .components
            .iter()
            .enumerate()
            .map(|(index, geometry)| {
                let id = u32::try_from(index + 1).map_err(|_| {
                    Error::ModelLoadError("VibeVoice tokenizer component count exceeds u32".into())
                })?;
                Ok(TensorComponentSpec {
                    id: StateComponentId::new(id),
                    role: TensorRole::ConvolutionState,
                    shape: crate::kv::v2::BoundedShape {
                        dimensions: vec![
                            ShapeDimension {
                                axis: ShapeAxis::Batch,
                                extent: ShapeExtent::Fixed { value: 1 },
                            },
                            ShapeDimension {
                                axis: ShapeAxis::Channels,
                                extent: ShapeExtent::Fixed {
                                    value: u64::try_from(geometry.channels).map_err(|_| {
                                        Error::ModelLoadError(
                                            "VibeVoice tokenizer channels exceed u64".into(),
                                        )
                                    })?,
                                },
                            },
                            ShapeDimension {
                                axis: ShapeAxis::Frames,
                                extent: ShapeExtent::Fixed {
                                    value: u64::try_from(geometry.frames).map_err(|_| {
                                        Error::ModelLoadError(
                                            "VibeVoice tokenizer frames exceed u64".into(),
                                        )
                                    })?,
                                },
                            },
                        ],
                    },
                    accepted_dtypes: vec![state_dtype],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        contract
            .domains
            .push(StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: StateDomainHeader {
                    id: authored.domain,
                    scope: StateScope::Invocation,
                    clock: authored.clock.clone(),
                    placement: PlacementPolicy::BackendLocal,
                    prefix: PrefixPolicy::Disabled,
                    checkpoint: CheckpointPolicy::None,
                },
                components,
            }));
        grouped
            .entry(authored.group)
            .or_default()
            .push(authored.domain);
    }
    contract.domains.sort_unstable_by_key(StateDomainSpec::id);
    for (group, mut domains) in grouped {
        if contract.groups.iter().any(|existing| existing.id == group) {
            return Err(Error::ModelLoadError(format!(
                "VibeVoice tokenizer group {} collides with decoder state",
                group.get()
            )));
        }
        domains.sort_unstable();
        contract.groups.push(StateGroupSpec {
            id: group,
            domains,
            prefix_shareable: false,
        });
    }
    contract.groups.sort_unstable_by_key(|group| group.id);
    Ok(())
}

fn vibevoice_state_dtype(dtype: DType) -> Result<StateDType> {
    match dtype {
        DType::F32 => Ok(StateDType::F32),
        DType::F16 => Ok(StateDType::F16),
        DType::BF16 => Ok(StateDType::Bf16),
        other => Err(Error::ModelLoadError(format!(
            "VibeVoice tokenizer state requires F32, F16, or BF16, got {other:?}"
        ))),
    }
}

pub(crate) fn vibevoice_invocation_descriptor(
    stage_graphs: &[&[StageDescriptor]],
    contract: &InferenceStateContract,
    max_context_tokens: usize,
) -> Result<CapabilityStateDescriptorV2> {
    if stage_graphs.is_empty() || max_context_tokens == 0 {
        return Err(Error::ModelLoadError(
            "VibeVoice invocation state requires stages and a non-zero context".into(),
        ));
    }
    let max_tokens = u64::try_from(max_context_tokens)
        .map_err(|_| Error::ModelLoadError("VibeVoice context exceeds u64".into()))?;
    let max_domain_id = contract
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("VibeVoice invocation contract is empty".into()))?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let mut domains = contract
                .domains
                .iter()
                .cloned()
                .map(|state| {
                    let (fixed_bytes, capacity) = match &state {
                        StateDomainSpec::PagedAttention(_) => (
                            vibevoice_paged_invocation_bytes(&state, max_tokens)?,
                            InvocationStateCapacity::decoder_context(max_tokens)?,
                        ),
                        StateDomainSpec::Tensor(_) => (
                            vibevoice_tensor_invocation_bytes(&state)?,
                            InvocationStateCapacity::SemanticBounded,
                        ),
                        _ => {
                            return Err(Error::ModelLoadError(
                                "VibeVoice invocation workspace contains an unsupported state kind"
                                    .into(),
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
                        Error::ModelLoadError("VibeVoice execution stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("VibeVoice scratch domain id overflow".into())
                    })?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: crate::kv::v2::StateDomainId::new(scratch_id),
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
                groups: contract.groups.clone(),
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
    Ok(descriptor)
}

pub(crate) fn vibevoice_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    invocation: InferenceStateContract,
    max_context_tokens: usize,
) -> Result<VibeVoicePhysicalStateSpec> {
    let has_asr_stage_identity =
        stage_graphs
            .iter()
            .flat_map(|stages| stages.iter())
            .any(|stage| {
                matches!(
                    stage.name.as_str(),
                    VIBEVOICE_ASR_PREPARATION_STAGE
                        | VIBEVOICE_ASR_PREFILL_STAGE
                        | VIBEVOICE_ASR_DECODE_STAGE
                        | VIBEVOICE_ASR_LEGACY_STAGE
                )
            });
    if has_vibevoice_asr_domain_topology(&invocation) || has_asr_stage_identity {
        return vibevoice_asr_physical_state_spec(stage_graphs, invocation, max_context_tokens);
    }
    let descriptor =
        vibevoice_invocation_descriptor(stage_graphs, &invocation, max_context_tokens)?;
    Ok(VibeVoicePhysicalStateSpec {
        descriptor,
        retained: None,
        retained_max_tokens: None,
        invocation,
    })
}

fn has_vibevoice_asr_domain_topology(contract: &InferenceStateContract) -> bool {
    contract.domains.len() == 3
        && contract.groups.len() == 2
        && contract.groups[0].id == VIBEVOICE_ASR_DECODER_GROUP
        && contract.groups[0].domains == [VIBEVOICE_ASR_DECODER_DOMAIN]
        && contract.groups[1].id == VIBEVOICE_ASR_TOKENIZER_GROUP
        && contract.groups[1].domains
            == [VIBEVOICE_ASR_ACOUSTIC_DOMAIN, VIBEVOICE_ASR_SEMANTIC_DOMAIN]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VibeVoiceAsrGraphKind {
    Normal,
    Legacy,
}

fn vibevoice_asr_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    invocation: InferenceStateContract,
    max_context_tokens: usize,
) -> Result<VibeVoicePhysicalStateSpec> {
    if stage_graphs.is_empty() || max_context_tokens == 0 {
        return Err(Error::ModelLoadError(
            "VibeVoice ASR physical state requires execution graphs and a non-zero context".into(),
        ));
    }
    let retained = vibevoice_asr_retained_contract(&invocation)?;
    let max_tokens = u64::try_from(max_context_tokens)
        .map_err(|_| Error::ModelLoadError("VibeVoice ASR context exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| {
            Error::ModelLoadError("VibeVoice ASR invocation contract is empty".into())
        })?;

    let mut profiles = Vec::with_capacity(stage_graphs.len());
    let mut saw_normal = false;
    let mut saw_legacy = false;
    for stages in stage_graphs {
        let kind = authenticate_vibevoice_asr_graph(stages)?;
        saw_normal |= kind == VibeVoiceAsrGraphKind::Normal;
        saw_legacy |= kind == VibeVoiceAsrGraphKind::Legacy;
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let owns_legacy_state = kind == VibeVoiceAsrGraphKind::Legacy;
            let mut domains = if owns_legacy_state {
                invocation
                    .domains
                    .iter()
                    .cloned()
                    .map(|state| {
                        let (fixed_bytes, capacity) = match &state {
                            StateDomainSpec::PagedAttention(_) => (
                                vibevoice_paged_invocation_bytes(&state, max_tokens)?,
                                InvocationStateCapacity::decoder_context(max_tokens)?,
                            ),
                            StateDomainSpec::Tensor(_) => (
                                vibevoice_tensor_invocation_bytes(&state)?,
                                InvocationStateCapacity::SemanticBounded,
                            ),
                            _ => {
                                return Err(Error::ModelLoadError(
                                    "VibeVoice ASR legacy state contains an unsupported domain"
                                        .into(),
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
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            if stage.max_workspace_bytes > 0 {
                let scratch_bytes = if stage.workspace_per_row_bytes > 0 {
                    stage.workspace_per_row_bytes
                } else {
                    stage.max_workspace_bytes
                };
                let ordinal = u32::try_from(index + 1).map_err(|_| {
                    Error::ModelLoadError("VibeVoice ASR stage count exceeds u32".into())
                })?;
                let scratch_id = max_domain_id.checked_add(ordinal).ok_or_else(|| {
                    Error::ModelLoadError("VibeVoice ASR scratch domain id overflow".into())
                })?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: scratch_bytes,
                        dimensions: vec![],
                        terms: vec![],
                    },
                });
            }
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: owns_legacy_state
                    .then(|| invocation.groups.clone())
                    .unwrap_or_default(),
                domains,
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: invocation_stages,
        });
    }
    if !saw_normal || !saw_legacy {
        return Err(Error::ModelLoadError(
            "VibeVoice ASR must seal both its normal retained and legacy atomic graphs".into(),
        ));
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
    Ok(VibeVoicePhysicalStateSpec {
        descriptor,
        retained: Some(retained),
        retained_max_tokens: Some(max_context_tokens),
        invocation,
    })
}

fn authenticate_vibevoice_asr_graph(stages: &[StageDescriptor]) -> Result<VibeVoiceAsrGraphKind> {
    for stage in stages {
        stage.validate()?;
    }
    let mut ordered = stages.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|stage| stage.id);
    if let [stage] = ordered.as_slice() {
        if stage.id == StageId::new(0)
            && stage.name == VIBEVOICE_ASR_LEGACY_STAGE
            && stage.selector == StageWorkSelector::Atomic
            && stage.progress == StageProgressKind::Atomic
            && stage.batch_mode == NativeBatchMode::None
            && stage.shape_policy == StageShapePolicy::Exact
        {
            return Ok(VibeVoiceAsrGraphKind::Legacy);
        }
    }
    if let [preparation, prefill, decode] = ordered.as_slice() {
        let native_prefill = prefill.batch_mode == NativeBatchMode::Static
            && prefill.shape_policy == StageShapePolicy::Padded
            && prefill.concurrency == ConcurrencyClass::Batchable
            && prefill.max_batch_size > 1
            && prefill.workspace_base_bytes == 0
            && prefill.workspace_per_work_unit_bytes == 0
            && prefill.workspace_per_row_bytes > 0
            && u64::try_from(prefill.max_batch_size)
                .ok()
                .and_then(|width| prefill.workspace_per_row_bytes.checked_mul(width))
                == Some(prefill.max_workspace_bytes);
        let scalar_prefill = prefill.batch_mode == NativeBatchMode::None
            && prefill.shape_policy == StageShapePolicy::Exact
            && prefill.concurrency == ConcurrencyClass::Exclusive
            && prefill.max_batch_size == 1
            && prefill.workspace_per_row_bytes == 0;
        let valid = preparation.id == StageId::new(0)
            && preparation.name == VIBEVOICE_ASR_PREPARATION_STAGE
            && preparation.selector == StageWorkSelector::PreSequencePreparation
            && preparation.domain == ExecutionDomain::ExecutionGroup
            && preparation.progress == StageProgressKind::Atomic
            && preparation.batch_mode == NativeBatchMode::None
            && preparation.shape_policy == StageShapePolicy::Exact
            && preparation.concurrency == ConcurrencyClass::Exclusive
            && preparation.physical_launch_policy == PhysicalLaunchPolicy::ExecutionGroupExclusive
            && preparation.membership_safe_point == MembershipSafePoint::OperationBoundary
            && preparation.output_visibility == OutputVisibility::AfterQuantumCommit
            && preparation.max_batch_size == 1
            && preparation.max_workspace_bytes > 0
            && prefill.id == StageId::new(1)
            && prefill.name == VIBEVOICE_ASR_PREFILL_STAGE
            && prefill.selector == StageWorkSelector::SequencePrefill
            && prefill.domain == ExecutionDomain::ExecutionGroup
            && prefill.progress == StageProgressKind::Iterative
            && (native_prefill || scalar_prefill)
            && prefill.physical_launch_policy == PhysicalLaunchPolicy::ExecutionGroupExclusive
            && prefill.membership_safe_point == MembershipSafePoint::QuantumBoundary
            && prefill.output_visibility == OutputVisibility::AfterQuantumCommit
            && prefill.max_work_units > 0
            && prefill.max_workspace_bytes > 0
            && prefill.retained_state_selections.as_deref()
                == Some(
                    [ClockedStateSelection::new(
                        VIBEVOICE_ASR_TOKENIZER_GROUP,
                        StateClock::AudioSamples,
                    )?]
                    .as_slice(),
                )
            && decode.id == StageId::new(2)
            && decode.name == VIBEVOICE_ASR_DECODE_STAGE
            && decode.selector == StageWorkSelector::SequenceDecode
            && decode.domain == ExecutionDomain::ExecutionGroup
            && decode.progress == StageProgressKind::Iterative
            && decode.batch_mode == NativeBatchMode::Continuous
            && decode.shape_policy == StageShapePolicy::Ragged
            && decode.concurrency == ConcurrencyClass::Batchable
            && decode.physical_launch_policy == PhysicalLaunchPolicy::ExecutionGroupExclusive
            && decode.membership_safe_point == MembershipSafePoint::QuantumBoundary
            && decode.output_visibility == OutputVisibility::AfterQuantumCommit
            && decode.max_batch_size > 1
            && decode.retained_state_selections.as_deref() == Some(&[]);
        if valid {
            return Ok(VibeVoiceAsrGraphKind::Normal);
        }
    }
    Err(Error::ModelLoadError(
        "VibeVoice ASR execution graph does not match the sealed normal or legacy graph".into(),
    ))
}

fn vibevoice_asr_retained_contract(
    invocation: &InferenceStateContract,
) -> Result<InferenceStateContract> {
    invocation.validate()?;
    if !has_vibevoice_asr_domain_topology(invocation) {
        return Err(Error::ModelLoadError(
            "VibeVoice ASR legacy contract must contain exact decoder and coupled tokenizer domains"
                .into(),
        ));
    }
    let decoder = invocation
        .domains
        .iter()
        .find(|domain| domain.id() == VIBEVOICE_ASR_DECODER_DOMAIN)
        .ok_or_else(|| Error::ModelLoadError("VibeVoice ASR decoder domain is missing".into()))?;
    let StateDomainSpec::PagedAttention(mut decoder) = decoder.clone() else {
        return Err(Error::ModelLoadError(
            "VibeVoice ASR decoder domain must be paged attention".into(),
        ));
    };
    if decoder.header.scope != StateScope::Invocation
        || decoder.header.clock != StateClock::DecoderTokens
        || decoder.header.prefix != PrefixPolicy::Disabled
        || decoder.header.checkpoint != CheckpointPolicy::None
    {
        return Err(Error::ModelLoadError(
            "VibeVoice ASR legacy decoder domain has unexpected state semantics".into(),
        ));
    }
    for id in [VIBEVOICE_ASR_ACOUSTIC_DOMAIN, VIBEVOICE_ASR_SEMANTIC_DOMAIN] {
        let domain = invocation
            .domains
            .iter()
            .find(|domain| domain.id() == id)
            .ok_or_else(|| {
                Error::ModelLoadError("VibeVoice ASR tokenizer domain is missing".into())
            })?;
        if !matches!(domain, StateDomainSpec::Tensor(_))
            || domain.header().scope != StateScope::Invocation
            || domain.header().clock != StateClock::AudioSamples
            || domain.header().prefix != PrefixPolicy::Disabled
            || domain.header().checkpoint != CheckpointPolicy::None
        {
            return Err(Error::ModelLoadError(
                "VibeVoice ASR tokenizer domains have unexpected state semantics".into(),
            ));
        }
    }
    decoder.header.scope = StateScope::Retained;
    decoder.header.prefix = PrefixPolicy::Disabled;
    decoder.header.checkpoint = CheckpointPolicy::Transactional;
    let mut retained_domains = vec![StateDomainSpec::PagedAttention(decoder)];
    for id in [VIBEVOICE_ASR_ACOUSTIC_DOMAIN, VIBEVOICE_ASR_SEMANTIC_DOMAIN] {
        let StateDomainSpec::Tensor(mut tokenizer) = invocation
            .domains
            .iter()
            .find(|domain| domain.id() == id)
            .cloned()
            .ok_or_else(|| {
                Error::ModelLoadError("VibeVoice ASR tokenizer domain is missing".into())
            })?
        else {
            return Err(Error::ModelLoadError(
                "VibeVoice ASR tokenizer domain must be tensor state".into(),
            ));
        };
        tokenizer.header.scope = StateScope::Retained;
        tokenizer.header.prefix = PrefixPolicy::Disabled;
        tokenizer.header.checkpoint = CheckpointPolicy::Transactional;
        retained_domains.push(StateDomainSpec::Tensor(tokenizer));
    }
    let retained = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: retained_domains,
        groups: invocation.groups.clone(),
    };
    retained.validate()?;
    Ok(retained)
}

fn vibevoice_paged_invocation_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(Error::ModelLoadError(
            "VibeVoice invocation workspace is not paged attention".into(),
        ));
    };
    let page_tokens = u64::from(spec.page_size.preferred_tokens);
    let rounded_tokens = max_tokens
        .checked_add(page_tokens.saturating_sub(1))
        .and_then(|tokens| tokens.checked_div(page_tokens))
        .and_then(|pages| pages.checked_mul(page_tokens))
        .ok_or_else(|| Error::ModelLoadError("VibeVoice page capacity overflow".into()))?;
    let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
        let layer_elements = u64::from(layer.kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| Error::ModelLoadError("VibeVoice KV geometry overflow".into()))?;
        total
            .checked_add(layer_elements)
            .ok_or_else(|| Error::ModelLoadError("VibeVoice KV geometry overflow".into()))
    })?;
    let element_bytes = spec
        .accepted_dtypes
        .iter()
        .map(|dtype| match dtype {
            StateDType::F32 => Ok(4_u64),
            StateDType::F16 | StateDType::Bf16 => Ok(2_u64),
            StateDType::I64 | StateDType::I8 | StateDType::Q4 => Err(Error::ModelLoadError(
                "VibeVoice invocation paging requires a dense loaded KV dtype".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .min()
        .ok_or_else(|| Error::ModelLoadError("VibeVoice KV dtype set is empty".into()))?;
    elements_per_token
        .checked_mul(rounded_tokens)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| Error::ModelLoadError("VibeVoice invocation byte bound overflow".into()))
}

fn vibevoice_tensor_invocation_bytes(state: &StateDomainSpec) -> Result<u64> {
    let StateDomainSpec::Tensor(spec) = state else {
        return Err(Error::ModelLoadError(
            "VibeVoice invocation workspace is not tensor state".into(),
        ));
    };
    spec.components.iter().try_fold(0_u64, |total, component| {
        let elements = component.shape.maximum_elements()?;
        let element_bytes = component
            .accepted_dtypes
            .first()
            .map(|dtype| match dtype {
                StateDType::F32 => Ok(4_u64),
                StateDType::F16 | StateDType::Bf16 => Ok(2_u64),
                StateDType::I64 | StateDType::I8 | StateDType::Q4 => Err(Error::ModelLoadError(
                    "VibeVoice tokenizer state requires a dense dtype".into(),
                )),
            })
            .transpose()?
            .ok_or_else(|| {
                Error::ModelLoadError("VibeVoice tokenizer state has no dtype".into())
            })?;
        total
            .checked_add(elements.checked_mul(element_bytes).ok_or_else(|| {
                Error::ModelLoadError("VibeVoice tokenizer component bytes overflow".into())
            })?)
            .ok_or_else(|| {
                Error::ModelLoadError("VibeVoice tokenizer domain bytes overflow".into())
            })
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::{
        ConcurrencyClass, ExecutionDomain, MembershipSafePoint, NativeBatchMode, OutputVisibility,
        StageId, StageProgressKind, StageShapePolicy, StageWorkSelector,
    };
    use crate::kv::v2::PositionSemantics;
    use crate::kv::v2::{StateDomainId, StateGroupId};
    use crate::models::architectures::qwen3::core::{
        qwen3_decoder_cache_domain, Qwen3DecoderCacheGeometry,
    };

    fn stage() -> StageDescriptor {
        StageDescriptor {
            id: StageId::new(1),
            name: "vibevoice.generate".into(),
            selector: StageWorkSelector::Atomic,
            domain: ExecutionDomain::ExecutionGroup,
            progress: StageProgressKind::Atomic,
            concurrency: ConcurrencyClass::Batchable,
            physical_launch_policy: crate::engine::PhysicalLaunchPolicy::ExecutionGroupExclusive,
            batch_mode: NativeBatchMode::None,
            max_batch_size: 2,
            max_work_units: 1,
            workspace_base_bytes: 0,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes: 0,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
            shape_policy: StageShapePolicy::Independent,
            membership_safe_point: MembershipSafePoint::OperationBoundary,
            output_visibility: OutputVisibility::AfterQuantumCommit,
            retained_state_selections: None,
        }
    }

    fn state_contract(domain_count: u32, dtype: DType) -> InferenceStateContract {
        let domains = (1..=domain_count)
            .map(|domain| {
                let id = StateDomainId::new(domain);
                qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
                    domain: id,
                    clock: StateClock::DecoderTokens,
                    num_layers: 2,
                    num_query_heads: 4,
                    num_kv_heads: 2,
                    key_head_dim: 8,
                    value_head_dim: 8,
                    sliding_window: None,
                    storage_dtype: dtype,
                    preferred_page_tokens: 16,
                    prefix: PrefixPolicy::CommittedPages {
                        positions: PositionSemantics::Absolute,
                    },
                })
                .map(|domain| (id, StateDomainSpec::PagedAttention(domain)))
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();
        InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: domains.iter().map(|(_, domain)| domain.clone()).collect(),
            groups: domains
                .iter()
                .map(|(id, _)| StateGroupSpec {
                    id: StateGroupId::new(id.get()),
                    domains: vec![*id],
                    prefix_shareable: true,
                })
                .collect(),
        }
    }

    fn invocation_contract(domain_count: u32) -> InferenceStateContract {
        vibevoice_invocation_contract_from_state(
            state_contract(domain_count, DType::F32),
            DType::F32,
            &[],
        )
        .unwrap()
    }

    fn tokenizer_domain(
        domain: u32,
        group: u32,
        clock: StateClock,
        geometry: &[(usize, usize)],
    ) -> VibeVoiceTokenizerStateDomain {
        VibeVoiceTokenizerStateDomain::new(
            StateDomainId::new(domain),
            StateGroupId::new(group),
            clock,
            geometry
                .iter()
                .map(
                    |(channels, frames)| VibeVoiceTokenizerStateComponentGeometry {
                        channels: *channels,
                        frames: *frames,
                    },
                )
                .collect(),
        )
        .unwrap()
    }

    fn asr_legacy_stage() -> StageDescriptor {
        let mut legacy = stage();
        legacy.id = StageId::new(0);
        legacy.name = VIBEVOICE_ASR_LEGACY_STAGE.into();
        legacy.shape_policy = StageShapePolicy::Exact;
        legacy.concurrency = ConcurrencyClass::Exclusive;
        legacy.max_batch_size = 1;
        legacy
    }

    fn asr_normal_stages() -> [StageDescriptor; 3] {
        let mut preparation = stage();
        preparation.id = StageId::new(0);
        preparation.name = VIBEVOICE_ASR_PREPARATION_STAGE.into();
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.max_batch_size = 1;
        preparation.max_workspace_bytes = 4096;

        let mut prefill = stage();
        prefill.id = StageId::new(1);
        prefill.name = VIBEVOICE_ASR_PREFILL_STAGE.into();
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.progress = StageProgressKind::Iterative;
        prefill.batch_mode = NativeBatchMode::Static;
        prefill.shape_policy = StageShapePolicy::Padded;
        prefill.concurrency = ConcurrencyClass::Batchable;
        prefill.max_batch_size = 2;
        prefill.max_work_units = 2;
        prefill.max_padding_basis_points = 10_000;
        prefill.workspace_per_row_bytes = 4096;
        prefill.max_workspace_bytes = 8192;
        prefill.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        prefill.retained_state_selections = Some(vec![ClockedStateSelection::new(
            VIBEVOICE_ASR_TOKENIZER_GROUP,
            StateClock::AudioSamples,
        )
        .unwrap()]);

        let mut decode = stage();
        decode.id = StageId::new(2);
        decode.name = VIBEVOICE_ASR_DECODE_STAGE.into();
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.progress = StageProgressKind::Iterative;
        decode.batch_mode = NativeBatchMode::Continuous;
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        decode.retained_state_selections = Some(vec![]);
        [preparation, prefill, decode]
    }

    fn asr_complete_invocation_contract() -> InferenceStateContract {
        let tokenizer = [
            tokenizer_domain(2, 2, StateClock::AudioSamples, &[(2, 3), (4, 5)]),
            tokenizer_domain(3, 2, StateClock::AudioSamples, &[(6, 7)]),
        ];
        vibevoice_invocation_contract_from_state(
            state_contract(1, DType::F32),
            DType::F32,
            &tokenizer,
        )
        .unwrap()
    }

    #[test]
    fn asr_descriptor_is_stateless_with_one_exact_invocation_domain() {
        let contract = invocation_contract(1);
        let execution = stage();
        let descriptor =
            vibevoice_invocation_descriptor(&[std::slice::from_ref(&execution)], &contract, 4096)
                .unwrap();
        assert!(descriptor.is_stateless());
        let InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
            panic!("ASR invocation pages must be bounded");
        };
        let workspace = &profiles[0].stages[0];
        assert_eq!(workspace.lease_scope, InvocationLeaseScope::PerRow);
        assert_eq!(workspace.groups.len(), 1);
        assert_eq!(workspace.domains.len(), 1);
        let InvocationWorkspaceDomain::State {
            capacity, formula, ..
        } = &workspace.domains[0]
        else {
            panic!("expected paged state")
        };
        assert_eq!(capacity.paged_max_tokens(), Some(4096));
        assert_eq!(formula.fixed_bytes, 1_048_576);
        descriptor.validate_against_stages(&[execution]).unwrap();
    }

    #[test]
    fn tts_cfg_descriptor_keeps_positive_and_negative_domains_isolated() {
        let contract = invocation_contract(2);
        assert_eq!(
            contract
                .groups
                .iter()
                .map(|group| (group.id, group.domains.clone()))
                .collect::<Vec<_>>(),
            vec![
                (StateGroupId::new(1), vec![StateDomainId::new(1)]),
                (StateGroupId::new(2), vec![StateDomainId::new(2)]),
            ]
        );
        assert!(contract
            .domains
            .iter()
            .all(|domain| domain.scope() == StateScope::Invocation));

        let execution = stage();
        let descriptor = vibevoice_invocation_descriptor(&[&[execution]], &contract, 8192).unwrap();
        let InvocationWorkspaceSet::Bounded { profiles } = descriptor.invocation else {
            panic!("TTS invocation pages must be bounded");
        };
        let workspace = &profiles[0].stages[0];
        assert_eq!(workspace.groups, contract.groups);
        assert_eq!(workspace.domains.len(), 2);
        assert!(workspace.domains.iter().all(|domain| {
            matches!(domain, InvocationWorkspaceDomain::State { capacity, formula, .. }
                if capacity.paged_max_tokens() == Some(8192)
                    && formula.fixed_bytes == 2_097_152)
        }));
    }

    #[test]
    fn asr_descriptor_couples_loaded_acoustic_and_semantic_tensor_state() {
        let tokenizer = [
            tokenizer_domain(2, 2, StateClock::AudioSamples, &[(2, 3), (4, 5)]),
            tokenizer_domain(3, 2, StateClock::AudioSamples, &[(6, 7)]),
        ];
        let contract = vibevoice_invocation_contract_from_state(
            state_contract(1, DType::F32),
            DType::F32,
            &tokenizer,
        )
        .unwrap();
        assert_eq!(
            contract
                .groups
                .iter()
                .map(|group| (group.id, group.domains.clone()))
                .collect::<Vec<_>>(),
            vec![
                (StateGroupId::new(1), vec![StateDomainId::new(1)]),
                (
                    StateGroupId::new(2),
                    vec![StateDomainId::new(2), StateDomainId::new(3)]
                ),
            ]
        );
        assert!(matches!(
            &contract.domains[1],
            StateDomainSpec::Tensor(spec)
                if spec.components.len() == 2
                    && spec.components[0].id == StateComponentId::new(1)
                    && spec.components[1].id == StateComponentId::new(2)
                    && spec.header.clock == StateClock::AudioSamples
        ));

        let execution = stage();
        let descriptor = vibevoice_invocation_descriptor(&[&[execution]], &contract, 4096).unwrap();
        let InvocationWorkspaceSet::Bounded { profiles } = descriptor.invocation else {
            panic!("ASR complete invocation state must be bounded");
        };
        let workspace = &profiles[0].stages[0];
        assert_eq!(workspace.domains.len(), 3);
        assert!(matches!(
            &workspace.domains[1],
            InvocationWorkspaceDomain::State {
                capacity: InvocationStateCapacity::SemanticBounded,
                formula: WorkspaceFormula {
                    fixed_bytes: 104,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            &workspace.domains[2],
            InvocationWorkspaceDomain::State {
                capacity: InvocationStateCapacity::SemanticBounded,
                formula: WorkspaceFormula {
                    fixed_bytes: 168,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn tts_descriptor_keeps_cfg_pages_isolated_and_couples_tokenizer_state() {
        let tokenizer = [
            tokenizer_domain(3, 3, StateClock::CodecFrames, &[(2, 3)]),
            tokenizer_domain(4, 3, StateClock::CodecFrames, &[(4, 5)]),
        ];
        let contract = vibevoice_invocation_contract_from_state(
            state_contract(2, DType::F32),
            DType::F32,
            &tokenizer,
        )
        .unwrap();
        assert_eq!(
            contract
                .groups
                .iter()
                .map(|group| (group.id, group.domains.clone()))
                .collect::<Vec<_>>(),
            vec![
                (StateGroupId::new(1), vec![StateDomainId::new(1)]),
                (StateGroupId::new(2), vec![StateDomainId::new(2)]),
                (
                    StateGroupId::new(3),
                    vec![StateDomainId::new(3), StateDomainId::new(4)]
                ),
            ]
        );
        assert!(contract.domains[2..].iter().all(|domain| matches!(
            domain,
            StateDomainSpec::Tensor(spec) if spec.header.clock == StateClock::CodecFrames
        )));

        let execution = stage();
        let descriptor = vibevoice_invocation_descriptor(&[&[execution]], &contract, 8192).unwrap();
        let InvocationWorkspaceSet::Bounded { profiles } = descriptor.invocation else {
            panic!("TTS complete invocation state must be bounded");
        };
        let workspace = &profiles[0].stages[0];
        assert_eq!(workspace.groups, contract.groups);
        assert_eq!(workspace.domains.len(), 4);
    }

    #[test]
    fn tokenizer_tensor_bytes_follow_the_exact_loaded_dtype() {
        let tokenizer = [tokenizer_domain(
            2,
            2,
            StateClock::CodecFrames,
            &[(2, 3), (4, 5)],
        )];
        let contract = vibevoice_invocation_contract_from_state(
            state_contract(1, DType::F16),
            DType::F16,
            &tokenizer,
        )
        .unwrap();
        assert_eq!(
            vibevoice_tensor_invocation_bytes(&contract.domains[1]).unwrap(),
            52
        );
    }

    #[test]
    fn asr_dual_graph_spec_retains_decoder_and_tokenizers_and_projects_exact_workspaces() {
        let invocation = asr_complete_invocation_contract();
        let normal = asr_normal_stages();
        let legacy = [asr_legacy_stage()];
        assert_eq!(
            normal[1].retained_state_selections.as_deref(),
            Some(
                [ClockedStateSelection::new(
                    VIBEVOICE_ASR_TOKENIZER_GROUP,
                    StateClock::AudioSamples,
                )
                .unwrap()]
                .as_slice()
            )
        );
        assert_eq!(
            normal[2].retained_state_selections.as_deref(),
            Some(&[][..])
        );
        let spec = vibevoice_physical_state_spec(
            &[normal.as_slice(), legacy.as_slice()],
            invocation.clone(),
            4096,
        )
        .unwrap();

        assert_eq!(spec.invocation, invocation);
        assert_eq!(spec.retained_max_tokens, Some(4096));
        let retained = spec.retained.as_ref().expect("normal retained contract");
        assert_eq!(retained.domains.len(), 3);
        assert_eq!(retained.groups.len(), 2);
        assert_eq!(retained.groups[0].domains, [VIBEVOICE_ASR_DECODER_DOMAIN]);
        assert_eq!(
            retained.groups[1].domains,
            [VIBEVOICE_ASR_ACOUSTIC_DOMAIN, VIBEVOICE_ASR_SEMANTIC_DOMAIN]
        );
        assert!(matches!(
            &retained.domains[0],
            StateDomainSpec::PagedAttention(domain)
                if domain.header.scope == StateScope::Retained
                    && domain.header.clock == StateClock::DecoderTokens
                    && domain.header.prefix == PrefixPolicy::Disabled
                    && domain.header.checkpoint == CheckpointPolicy::Transactional
        ));
        assert!(retained.domains[1..].iter().all(|domain| matches!(
            domain,
            StateDomainSpec::Tensor(tokenizer)
                if tokenizer.header.scope == StateScope::Retained
                    && tokenizer.header.clock == StateClock::AudioSamples
                    && tokenizer.header.prefix == PrefixPolicy::Disabled
                    && tokenizer.header.checkpoint == CheckpointPolicy::Transactional
        )));
        assert!(matches!(
            &spec.descriptor.retained,
            RetainedStateCapability::Managed { contract } if contract == retained
        ));

        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("dual graph ASR must have bounded graph-specific workspaces")
        };
        let normal_fingerprint = stage_graph_fingerprint(&normal).unwrap();
        let normal_profile = profiles
            .iter()
            .find(|profile| profile.stage_graph_fingerprint == normal_fingerprint)
            .unwrap();
        assert_eq!(normal_profile.stages.len(), 3);
        assert_eq!(
            normal_profile.stages[0].lease_scope,
            InvocationLeaseScope::PerRow
        );
        assert!(normal_profile.stages[0].groups.is_empty());
        assert!(matches!(
            normal_profile.stages[0].domains.as_slice(),
            [InvocationWorkspaceDomain::Scratch {
                formula: WorkspaceFormula {
                    fixed_bytes: 4096,
                    ..
                },
                ..
            }]
        ));
        assert!(matches!(
            normal_profile.stages[1].domains.as_slice(),
            [InvocationWorkspaceDomain::Scratch {
                formula: WorkspaceFormula {
                    fixed_bytes: 4096,
                    ..
                },
                ..
            }]
        ));
        assert!(normal_profile.stages[2].domains.is_empty());

        let legacy_fingerprint = stage_graph_fingerprint(&legacy).unwrap();
        let legacy_profile = profiles
            .iter()
            .find(|profile| profile.stage_graph_fingerprint == legacy_fingerprint)
            .unwrap();
        assert_eq!(legacy_profile.stages.len(), 1);
        assert_eq!(legacy_profile.stages[0].groups, invocation.groups);
        assert_eq!(legacy_profile.stages[0].domains.len(), 3);
        assert!(matches!(
            &legacy_profile.stages[0].domains[0],
            InvocationWorkspaceDomain::State {
                capacity,
                formula: WorkspaceFormula { fixed_bytes: 1_048_576, .. },
                ..
            } if capacity.paged_max_tokens() == Some(4096)
        ));
        assert!(matches!(
            &legacy_profile.stages[0].domains[1],
            InvocationWorkspaceDomain::State {
                capacity: InvocationStateCapacity::SemanticBounded,
                formula: WorkspaceFormula {
                    fixed_bytes: 104,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            &legacy_profile.stages[0].domains[2],
            InvocationWorkspaceDomain::State {
                capacity: InvocationStateCapacity::SemanticBounded,
                formula: WorkspaceFormula {
                    fixed_bytes: 168,
                    ..
                },
                ..
            }
        ));
        spec.descriptor.validate_against_stages(&normal).unwrap();
        spec.descriptor.validate_against_stages(&legacy).unwrap();
    }

    #[test]
    fn asr_dual_graph_spec_rejects_unsealed_decode_shape() {
        let invocation = asr_complete_invocation_contract();
        let mut normal = asr_normal_stages();
        normal[2].shape_policy = StageShapePolicy::Padded;
        let legacy = [asr_legacy_stage()];
        let error = vibevoice_physical_state_spec(
            &[normal.as_slice(), legacy.as_slice()],
            invocation,
            4096,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match the sealed normal or legacy graph"));
    }

    #[test]
    fn asr_dual_graph_spec_rejects_unimplemented_encoder_batching() {
        let invocation = asr_complete_invocation_contract();
        let mut normal = asr_normal_stages();
        normal[0].batch_mode = NativeBatchMode::Static;
        normal[0].shape_policy = StageShapePolicy::Padded;
        normal[0].concurrency = ConcurrencyClass::Batchable;
        let legacy = [asr_legacy_stage()];
        let error = vibevoice_physical_state_spec(
            &[normal.as_slice(), legacy.as_slice()],
            invocation,
            4096,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match the sealed normal or legacy graph"));
    }

    #[test]
    fn asr_dual_graph_spec_accepts_width_one_prefill_fallback() {
        let invocation = asr_complete_invocation_contract();
        let mut normal = asr_normal_stages();
        normal[1].batch_mode = NativeBatchMode::None;
        normal[1].shape_policy = StageShapePolicy::Exact;
        normal[1].concurrency = ConcurrencyClass::Exclusive;
        normal[1].max_batch_size = 1;
        normal[1].max_work_units = 1;
        normal[1].max_padding_basis_points = 0;
        normal[1].workspace_per_row_bytes = 0;
        normal[1].max_workspace_bytes = 4096;
        let legacy = [asr_legacy_stage()];

        vibevoice_physical_state_spec(&[normal.as_slice(), legacy.as_slice()], invocation, 4096)
            .unwrap();
    }

    #[test]
    fn asr_dual_graph_spec_authenticates_transaction_boundaries() {
        fn assert_rejected(normal: [StageDescriptor; 3]) {
            let legacy = [asr_legacy_stage()];
            assert!(vibevoice_physical_state_spec(
                &[normal.as_slice(), legacy.as_slice()],
                asr_complete_invocation_contract(),
                4096,
            )
            .is_err());
        }

        let mut wrong_visibility = asr_normal_stages();
        wrong_visibility[1].output_visibility = OutputVisibility::IncrementalCommitted;
        assert_rejected(wrong_visibility);

        let mut wrong_domain = asr_normal_stages();
        wrong_domain[0].domain = ExecutionDomain::Host;
        assert_rejected(wrong_domain);

        let mut wrong_safe_point = asr_normal_stages();
        wrong_safe_point[2].membership_safe_point = MembershipSafePoint::OperationBoundary;
        assert_rejected(wrong_safe_point);

        let mut scalar_decode_width = asr_normal_stages();
        scalar_decode_width[2].max_batch_size = 1;
        assert_rejected(scalar_decode_width);

        let mut scalar_prefill = asr_normal_stages();
        scalar_prefill[1].batch_mode = NativeBatchMode::None;
        scalar_prefill[1].shape_policy = StageShapePolicy::Independent;
        scalar_prefill[1].max_padding_basis_points = 0;
        assert_rejected(scalar_prefill);

        let mut implicit_prefill_state = asr_normal_stages();
        implicit_prefill_state[1].retained_state_selections = None;
        assert_rejected(implicit_prefill_state);

        let mut wrong_prefill_clock = asr_normal_stages();
        wrong_prefill_clock[1].retained_state_selections = Some(vec![ClockedStateSelection::new(
            VIBEVOICE_ASR_TOKENIZER_GROUP,
            StateClock::DecoderTokens,
        )
        .unwrap()]);
        assert_rejected(wrong_prefill_clock);

        let mut decoder_advances_tokenizer = asr_normal_stages();
        decoder_advances_tokenizer[2].retained_state_selections =
            Some(vec![ClockedStateSelection::new(
                VIBEVOICE_ASR_TOKENIZER_GROUP,
                StateClock::AudioSamples,
            )
            .unwrap()]);
        assert_rejected(decoder_advances_tokenizer);

        let mut implicit_decode_state = asr_normal_stages();
        implicit_decode_state[2].retained_state_selections = None;
        assert_rejected(implicit_decode_state);
    }

    #[test]
    fn asr_dual_graph_spec_requires_both_authenticated_graphs() {
        let invocation = asr_complete_invocation_contract();
        let normal = asr_normal_stages();
        let error =
            vibevoice_physical_state_spec(&[normal.as_slice()], invocation, 4096).unwrap_err();
        assert!(error
            .to_string()
            .contains("both its normal retained and legacy atomic graphs"));
    }

    #[test]
    fn asr_dual_graph_spec_rejects_incomplete_legacy_state() {
        let invocation = invocation_contract(1);
        let normal = asr_normal_stages();
        let legacy = [asr_legacy_stage()];
        let error = vibevoice_physical_state_spec(
            &[normal.as_slice(), legacy.as_slice()],
            invocation,
            4096,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("exact decoder and coupled tokenizer domains"));
    }

    #[test]
    fn tts_non_atomic_selector_does_not_enter_asr_dual_graph_contract() {
        let tokenizer = [
            tokenizer_domain(3, 3, StateClock::CodecFrames, &[(2, 3)]),
            tokenizer_domain(4, 3, StateClock::CodecFrames, &[(4, 5)]),
        ];
        let invocation = vibevoice_invocation_contract_from_state(
            state_contract(2, DType::F32),
            DType::F32,
            &tokenizer,
        )
        .unwrap();
        let mut tts = stage();
        tts.selector = StageWorkSelector::Any;
        let spec = vibevoice_physical_state_spec(&[&[tts]], invocation, 8192).unwrap();
        assert!(spec.retained.is_none());
        assert!(spec.retained_max_tokens.is_none());
        assert!(spec.descriptor.is_stateless());
    }
}
