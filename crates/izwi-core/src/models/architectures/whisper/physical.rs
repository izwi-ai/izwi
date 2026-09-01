//! Model-authored physical inference-state contract for Whisper.

use candle_core::DType;
use candle_transformers::models::whisper::Config;

use crate::backends::BackendKind;
use crate::engine::{NativeBatchMode, StageDescriptor, StageProgressKind, StageWorkSelector};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, AttentionMask, AttentionPattern, CapabilityStateDescriptorV2,
    CheckpointPolicy, InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationStateCapacity, InvocationWorkspaceDomain, InvocationWorkspaceProfile,
    InvocationWorkspaceSet, KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec,
    PagedAttentionLayerSpec, PlacementPolicy, PrefixPolicy, RetainedStateCapability, StateClock,
    StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec,
    StateScope, StaticAttentionDomainSpec, StaticAttentionLayerSpec, WorkspaceFormula,
    CURRENT_INFERENCE_STATE_ABI,
};

pub(crate) const WHISPER_SELF_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const WHISPER_CROSS_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);
pub(crate) const WHISPER_SELF_STATE_GROUP: StateGroupId = StateGroupId::new(1);
pub(crate) const WHISPER_CROSS_STATE_GROUP: StateGroupId = StateGroupId::new(2);

const WHISPER_PAGE_TOKENS: u32 = 64;
const CUDA_STATIC_ATTENTION_TOKEN_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub(crate) struct WhisperPhysicalStateSpec {
    pub(crate) retained: InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
    pub(crate) retained_static_domain: StateDomainId,
    pub(crate) retained_static_group: StateGroupId,
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn whisper_physical_state_spec(
    config: &Config,
    dtype: DType,
    backend: BackendKind,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<WhisperPhysicalStateSpec> {
    if stage_graphs.is_empty() {
        return Err(model_load("Whisper physical state has no execution graph"));
    }
    validate_config(config)?;
    let state_dtype = whisper_state_dtype(dtype, backend)?;
    let self_capacity = usize_to_u64(
        config.max_target_positions,
        "Whisper decoder context exceeds u64",
    )?;
    let cross_capacity = whisper_cross_capacity(config.max_source_positions, backend)?;
    let retained = whisper_state_contract(
        config,
        state_dtype,
        cross_capacity,
        StateScope::Retained,
        CheckpointPolicy::Transactional,
    )?;
    let invocation = whisper_state_contract(
        config,
        state_dtype,
        cross_capacity,
        StateScope::Invocation,
        CheckpointPolicy::None,
    )?;
    let self_state = invocation
        .domains
        .iter()
        .find(|domain| domain.id() == WHISPER_SELF_STATE_DOMAIN)
        .cloned()
        .ok_or_else(|| model_load("Whisper contract is missing decoder self-attention state"))?;
    let cross_state = invocation
        .domains
        .iter()
        .find(|domain| domain.id() == WHISPER_CROSS_STATE_DOMAIN)
        .cloned()
        .ok_or_else(|| model_load("Whisper contract is missing decoder cross-attention state"))?;
    let self_bytes = whisper_paged_bytes(&self_state, self_capacity)?;
    let cross_bytes = whisper_static_attention_bytes(&cross_state)?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| model_load("Whisper invocation contract is empty"))?;

    let mut profiles = Vec::with_capacity(stage_graphs.len());
    let mut has_invocation_workspace = false;
    for stages in stage_graphs {
        if stages.is_empty() {
            return Err(model_load("Whisper execution graph has no stages"));
        }
        for stage in *stages {
            stage.validate()?;
            match stage.selector {
                StageWorkSelector::Atomic => {
                    if stage.progress != StageProgressKind::Atomic
                        || stage.batch_mode != NativeBatchMode::None
                    {
                        return Err(model_load(
                            "Whisper long-form state requires independently scheduled atomic rows",
                        ));
                    }
                }
                StageWorkSelector::PreSequencePreparation => {
                    if stage.batch_mode != NativeBatchMode::Static {
                        return Err(model_load(
                            "Whisper encoder preparation requires a static native batch",
                        ));
                    }
                }
                StageWorkSelector::SequencePrefill | StageWorkSelector::SequenceDecode => {
                    if stage.batch_mode != NativeBatchMode::None {
                        return Err(model_load(
                            "Whisper scalar sequence stages cannot advertise native decoder batching",
                        ));
                    }
                }
                _ => {
                    return Err(model_load(
                        "Whisper execution graph contains an unsupported stage selector",
                    ));
                }
            }
        }

        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let owns_legacy_state = stage.selector == StageWorkSelector::Atomic;
            has_invocation_workspace |= owns_legacy_state || stage.max_workspace_bytes > 0;
            let mut domains = Vec::new();
            let mut groups = Vec::new();
            if owns_legacy_state {
                groups = invocation.groups.clone();
                domains.extend([
                    InvocationWorkspaceDomain::State {
                        placement: self_state.header().placement,
                        formula: fixed_formula(self_bytes),
                        state: self_state.clone(),
                        capacity: InvocationStateCapacity::decoder_context(self_capacity)?,
                    },
                    InvocationWorkspaceDomain::State {
                        placement: cross_state.header().placement,
                        formula: fixed_formula(cross_bytes),
                        state: cross_state.clone(),
                        capacity: InvocationStateCapacity::SemanticBounded,
                    },
                ]);
            }
            if stage.max_workspace_bytes > 0 {
                let ordinal = u32::try_from(index + 1)
                    .map_err(|_| model_load("Whisper execution stage count exceeds u32"))?;
                let scratch_id = max_domain_id
                    .checked_add(ordinal)
                    .ok_or_else(|| model_load("Whisper scratch domain id overflow"))?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: fixed_formula(stage.max_workspace_bytes),
                });
            }
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: if stage.selector == StageWorkSelector::PreSequencePreparation {
                    InvocationLeaseScope::PerStageBatch
                } else {
                    InvocationLeaseScope::PerRow
                },
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

    let invocation_workspaces = if has_invocation_workspace {
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
        invocation: invocation_workspaces,
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(WhisperPhysicalStateSpec {
        retained,
        retained_max_tokens: config.max_target_positions,
        retained_static_domain: WHISPER_CROSS_STATE_DOMAIN,
        retained_static_group: WHISPER_CROSS_STATE_GROUP,
        descriptor,
        invocation,
    })
}

fn whisper_state_contract(
    config: &Config,
    dtype: StateDType,
    cross_capacity: u64,
    scope: StateScope,
    checkpoint: CheckpointPolicy,
) -> Result<InferenceStateContract> {
    let query_heads = usize_to_u32(
        config.decoder_attention_heads,
        "Whisper decoder attention-head count exceeds u32",
    )?;
    let head_dim = usize_to_u32(
        config.d_model / config.decoder_attention_heads,
        "Whisper decoder head dimension exceeds u32",
    )?;
    let self_layers = (0..config.decoder_layers)
        .map(|model_layer| {
            Ok(PagedAttentionLayerSpec {
                model_layer: usize_to_u32(model_layer, "Whisper decoder layer count exceeds u32")?,
                query_heads,
                kv_heads: query_heads,
                key_head_dim: head_dim,
                value_head_dim: head_dim,
                pattern: AttentionPattern::Full,
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Raw,
                attention_logit_softcap: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let cross_layers = (0..config.decoder_layers)
        .map(|model_layer| {
            Ok(StaticAttentionLayerSpec {
                model_layer: usize_to_u32(model_layer, "Whisper decoder layer count exceeds u32")?,
                query_heads,
                kv_heads: query_heads,
                key_head_dim: head_dim,
                value_head_dim: head_dim,
                key_encoding: KeyEncoding::Raw,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let header = |id, clock| StateDomainHeader {
        id,
        scope,
        clock,
        placement: PlacementPolicy::BackendLocal,
        prefix: PrefixPolicy::Disabled,
        checkpoint,
    };
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![
            StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
                header: header(WHISPER_SELF_STATE_DOMAIN, StateClock::DecoderTokens),
                layers: self_layers,
                page_size: PageSizeConstraint {
                    min_tokens: WHISPER_PAGE_TOKENS,
                    preferred_tokens: WHISPER_PAGE_TOKENS,
                    max_tokens: WHISPER_PAGE_TOKENS,
                    multiple_of: WHISPER_PAGE_TOKENS,
                },
                accepted_dtypes: vec![dtype],
            }),
            StateDomainSpec::StaticAttention(StaticAttentionDomainSpec {
                header: header(WHISPER_CROSS_STATE_DOMAIN, StateClock::EncoderTokens),
                layers: cross_layers,
                max_memory_tokens: cross_capacity,
                accepted_dtypes: vec![dtype],
            }),
        ],
        groups: vec![
            StateGroupSpec {
                id: WHISPER_SELF_STATE_GROUP,
                domains: vec![WHISPER_SELF_STATE_DOMAIN],
                prefix_shareable: false,
            },
            StateGroupSpec {
                id: WHISPER_CROSS_STATE_GROUP,
                domains: vec![WHISPER_CROSS_STATE_DOMAIN],
                prefix_shareable: false,
            },
        ],
    };
    contract.validate()?;
    Ok(contract)
}

fn validate_config(config: &Config) -> Result<()> {
    if config.d_model == 0
        || config.decoder_attention_heads == 0
        || config.decoder_layers == 0
        || config.max_target_positions == 0
        || config.max_source_positions == 0
    {
        return Err(model_load(
            "Whisper physical state requires non-zero decoder geometry and capacities",
        ));
    }
    if !config
        .d_model
        .is_multiple_of(config.decoder_attention_heads)
    {
        return Err(model_load(
            "Whisper decoder width is not divisible by its attention-head count",
        ));
    }
    Ok(())
}

fn whisper_state_dtype(dtype: DType, backend: BackendKind) -> Result<StateDType> {
    match (backend, dtype) {
        (BackendKind::Cpu, DType::F32) | (BackendKind::Metal, DType::F32) => Ok(StateDType::F32),
        (BackendKind::Cpu | BackendKind::Metal | BackendKind::Cuda, DType::F16) => {
            Ok(StateDType::F16)
        }
        (BackendKind::Cpu | BackendKind::Cuda, DType::BF16) => Ok(StateDType::Bf16),
        _ => Err(model_load(format!(
            "Whisper {backend:?} physical state does not support exact loaded dtype {dtype:?}"
        ))),
    }
}

fn whisper_cross_capacity(max_source_positions: usize, backend: BackendKind) -> Result<u64> {
    let capacity = usize_to_u64(max_source_positions, "Whisper encoder context exceeds u64")?;
    if backend != BackendKind::Cuda {
        return Ok(capacity);
    }
    capacity
        .checked_add(CUDA_STATIC_ATTENTION_TOKEN_ALIGNMENT - 1)
        .and_then(|tokens| tokens.checked_div(CUDA_STATIC_ATTENTION_TOKEN_ALIGNMENT))
        .and_then(|groups| groups.checked_mul(CUDA_STATIC_ATTENTION_TOKEN_ALIGNMENT))
        .ok_or_else(|| model_load("Whisper CUDA cross-attention capacity overflow"))
}

fn whisper_paged_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(model_load(
            "Whisper decoder self-attention workspace is not paged attention",
        ));
    };
    let page_tokens = u64::from(spec.page_size.preferred_tokens);
    let rounded_tokens = max_tokens
        .checked_add(page_tokens - 1)
        .and_then(|tokens| tokens.checked_div(page_tokens))
        .and_then(|pages| pages.checked_mul(page_tokens))
        .ok_or_else(|| model_load("Whisper decoder page capacity overflow"))?;
    attention_bytes(
        spec.layers
            .iter()
            .map(|layer| (layer.kv_heads, layer.key_head_dim, layer.value_head_dim)),
        rounded_tokens,
        &spec.accepted_dtypes,
    )
}

fn whisper_static_attention_bytes(state: &StateDomainSpec) -> Result<u64> {
    let StateDomainSpec::StaticAttention(spec) = state else {
        return Err(model_load(
            "Whisper decoder cross-attention workspace is not static attention",
        ));
    };
    attention_bytes(
        spec.layers
            .iter()
            .map(|layer| (layer.kv_heads, layer.key_head_dim, layer.value_head_dim)),
        spec.max_memory_tokens,
        &spec.accepted_dtypes,
    )
}

fn attention_bytes<I>(mut layers: I, tokens: u64, accepted_dtypes: &[StateDType]) -> Result<u64>
where
    I: Iterator<Item = (u32, u32, u32)>,
{
    let elements_per_token = layers.try_fold(0_u64, |total, (kv_heads, key_dim, value_dim)| {
        let dimensions = u64::from(key_dim)
            .checked_add(u64::from(value_dim))
            .ok_or_else(|| model_load("Whisper attention geometry overflow"))?;
        let layer_elements = u64::from(kv_heads)
            .checked_mul(dimensions)
            .ok_or_else(|| model_load("Whisper attention geometry overflow"))?;
        total
            .checked_add(layer_elements)
            .ok_or_else(|| model_load("Whisper attention geometry overflow"))
    })?;
    let element_bytes = match accepted_dtypes {
        [StateDType::F32] => 4,
        [StateDType::F16] | [StateDType::Bf16] => 2,
        _ => {
            return Err(model_load(
                "Whisper physical state requires one exact dense loaded dtype",
            ))
        }
    };
    elements_per_token
        .checked_mul(tokens)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| model_load("Whisper attention workspace byte bound overflow"))
}

fn fixed_formula(fixed_bytes: u64) -> WorkspaceFormula {
    WorkspaceFormula {
        fixed_bytes,
        dimensions: vec![],
        terms: vec![],
    }
}

fn usize_to_u32(value: usize, overflow: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| model_load(overflow))
}

fn usize_to_u64(value: usize, overflow: &'static str) -> Result<u64> {
    u64::try_from(value).map_err(|_| model_load(overflow))
}

fn model_load(message: impl Into<String>) -> Error {
    Error::ModelLoadError(message.into())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::{
        ConcurrencyClass, ExecutionDomain, MembershipSafePoint, NativeBatchMode, OutputVisibility,
        StageId, StageShapePolicy, StageWorkSelector,
    };

    fn config() -> Config {
        Config {
            num_mel_bins: 80,
            max_source_positions: 33,
            d_model: 16,
            encoder_attention_heads: 4,
            encoder_layers: 2,
            vocab_size: 128,
            max_target_positions: 65,
            decoder_attention_heads: 4,
            decoder_layers: 2,
            suppress_tokens: vec![],
        }
    }

    fn stage(max_workspace_bytes: u64) -> StageDescriptor {
        StageDescriptor {
            id: StageId::new(1),
            name: "whisper.transcribe".into(),
            selector: StageWorkSelector::Atomic,
            domain: ExecutionDomain::ExecutionGroup,
            progress: StageProgressKind::Atomic,
            concurrency: ConcurrencyClass::Exclusive,
            physical_launch_policy: crate::engine::PhysicalLaunchPolicy::ExecutionGroupExclusive,
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            max_work_units: 1,
            workspace_base_bytes: max_workspace_bytes,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
            shape_policy: StageShapePolicy::Exact,
            membership_safe_point: MembershipSafePoint::OperationBoundary,
            output_visibility: OutputVisibility::AfterQuantumCommit,
            retained_state_selections: None,
        }
    }

    #[test]
    fn contracts_are_exact_for_cpu_metal_and_cuda() {
        for (
            backend,
            dtype,
            expected_dtype,
            expected_cross_capacity,
            expected_self_bytes,
            expected_cross_bytes,
        ) in [
            (
                BackendKind::Cpu,
                DType::F32,
                StateDType::F32,
                33,
                32_768,
                8_448,
            ),
            (
                BackendKind::Metal,
                DType::F16,
                StateDType::F16,
                33,
                16_384,
                4_224,
            ),
            (
                BackendKind::Cuda,
                DType::BF16,
                StateDType::Bf16,
                64,
                16_384,
                8_192,
            ),
        ] {
            let execution = stage(0);
            let spec = whisper_physical_state_spec(
                &config(),
                dtype,
                backend,
                &[std::slice::from_ref(&execution)],
            )
            .expect("physical state");
            assert!(matches!(
                &spec.descriptor.retained,
                RetainedStateCapability::Managed { contract } if contract == &spec.retained
            ));
            assert_eq!(spec.retained_max_tokens, 65);
            assert_eq!(spec.retained_static_domain, WHISPER_CROSS_STATE_DOMAIN);
            assert_eq!(spec.retained_static_group, WHISPER_CROSS_STATE_GROUP);
            assert!(spec.retained.domains.iter().all(|domain| {
                domain.scope() == StateScope::Retained
                    && domain.header().checkpoint == CheckpointPolicy::Transactional
            }));
            assert_eq!(
                spec.invocation.groups,
                vec![
                    StateGroupSpec {
                        id: WHISPER_SELF_STATE_GROUP,
                        domains: vec![WHISPER_SELF_STATE_DOMAIN],
                        prefix_shareable: false,
                    },
                    StateGroupSpec {
                        id: WHISPER_CROSS_STATE_GROUP,
                        domains: vec![WHISPER_CROSS_STATE_DOMAIN],
                        prefix_shareable: false,
                    },
                ]
            );

            let StateDomainSpec::PagedAttention(self_state) = &spec.invocation.domains[0] else {
                panic!("self state must be paged attention");
            };
            assert_eq!(self_state.header.id, WHISPER_SELF_STATE_DOMAIN);
            assert_eq!(self_state.header.clock, StateClock::DecoderTokens);
            assert_eq!(
                self_state.page_size,
                PageSizeConstraint {
                    min_tokens: 64,
                    preferred_tokens: 64,
                    max_tokens: 64,
                    multiple_of: 64,
                }
            );
            assert_eq!(self_state.accepted_dtypes, vec![expected_dtype]);
            assert_eq!(self_state.layers.len(), 2);
            assert!(self_state.layers.iter().all(|layer| {
                layer.query_heads == 4
                    && layer.kv_heads == 4
                    && layer.key_head_dim == 4
                    && layer.value_head_dim == 4
                    && layer.pattern == AttentionPattern::Full
                    && layer.mask == AttentionMask::Causal
                    && layer.key_encoding == KeyEncoding::Raw
            }));

            let StateDomainSpec::StaticAttention(cross_state) = &spec.invocation.domains[1] else {
                panic!("cross state must be static attention");
            };
            assert_eq!(cross_state.header.id, WHISPER_CROSS_STATE_DOMAIN);
            assert_eq!(cross_state.header.clock, StateClock::EncoderTokens);
            assert_eq!(cross_state.max_memory_tokens, expected_cross_capacity);
            assert_eq!(cross_state.accepted_dtypes, vec![expected_dtype]);
            assert_eq!(cross_state.layers.len(), 2);
            assert!(cross_state.layers.iter().all(|layer| {
                layer.query_heads == 4
                    && layer.kv_heads == 4
                    && layer.key_head_dim == 4
                    && layer.value_head_dim == 4
                    && layer.key_encoding == KeyEncoding::Raw
            }));
            let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
                panic!("Whisper invocation state must be bounded");
            };
            assert!(matches!(
                &profiles[0].stages[0].domains[0],
                InvocationWorkspaceDomain::State {
                    formula: WorkspaceFormula { fixed_bytes, .. },
                    ..
                } if *fixed_bytes == expected_self_bytes
            ));
            assert!(matches!(
                &profiles[0].stages[0].domains[1],
                InvocationWorkspaceDomain::State {
                    formula: WorkspaceFormula { fixed_bytes, .. },
                    ..
                } if *fixed_bytes == expected_cross_bytes
            ));
            spec.descriptor
                .validate_against_stages(&[execution])
                .expect("descriptor");
        }
    }

    #[test]
    fn scalar_atomic_workspace_has_exact_state_formulas_and_authored_scratch() {
        let execution = stage(96);
        let spec = whisper_physical_state_spec(
            &config(),
            DType::F32,
            BackendKind::Cpu,
            &[std::slice::from_ref(&execution)],
        )
        .expect("physical state");
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("Whisper invocation state must be bounded");
        };
        let workspace = &profiles[0].stages[0];
        assert_eq!(workspace.lease_scope, InvocationLeaseScope::PerRow);
        assert_eq!(workspace.groups, spec.invocation.groups);
        assert_eq!(workspace.domains.len(), 3);
        assert!(matches!(
            &workspace.domains[0],
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::PagedAttention(_),
                capacity,
                formula: WorkspaceFormula { fixed_bytes: 32_768, .. },
                ..
            } if capacity.paged_max_tokens() == Some(65)
        ));
        assert!(matches!(
            &workspace.domains[1],
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::StaticAttention(_),
                capacity: InvocationStateCapacity::SemanticBounded,
                formula: WorkspaceFormula {
                    fixed_bytes: 8_448,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            &workspace.domains[2],
            InvocationWorkspaceDomain::Scratch {
                formula: WorkspaceFormula {
                    fixed_bytes: 96,
                    ..
                },
                ..
            }
        ));
        spec.descriptor
            .validate_against_stages(&[execution])
            .expect("descriptor");
    }

    #[test]
    fn accepts_independent_row_parallelism_but_rejects_native_batching() {
        let mut parallel = stage(0);
        parallel.concurrency = ConcurrencyClass::Batchable;
        parallel.shape_policy = StageShapePolicy::Independent;
        parallel.max_batch_size = 8;
        whisper_physical_state_spec(&config(), DType::F32, BackendKind::Cpu, &[&[parallel]])
            .expect("parallel scalar rows");

        let mut native_batch = stage(0);
        native_batch.batch_mode = NativeBatchMode::Static;
        native_batch.concurrency = ConcurrencyClass::Batchable;
        native_batch.shape_policy = StageShapePolicy::Padded;
        native_batch.max_batch_size = 2;
        assert!(whisper_physical_state_spec(
            &config(),
            DType::F32,
            BackendKind::Cpu,
            &[&[native_batch]],
        )
        .is_err());
    }

    #[test]
    fn normal_graph_uses_retained_state_and_no_legacy_invocation_domains() {
        let mut encoder = stage(64);
        encoder.selector = StageWorkSelector::PreSequencePreparation;
        encoder.batch_mode = NativeBatchMode::Static;
        encoder.max_batch_size = 4;
        encoder.max_work_units = 4;
        encoder.concurrency = ConcurrencyClass::Batchable;
        encoder.shape_policy = StageShapePolicy::Padded;

        let mut prefill = stage(0);
        prefill.id = StageId::new(2);
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.progress = StageProgressKind::Iterative;
        let mut decode = prefill.clone();
        decode.id = StageId::new(3);
        decode.selector = StageWorkSelector::SequenceDecode;

        let stages = [encoder, prefill, decode];
        let spec = whisper_physical_state_spec(&config(), DType::F32, BackendKind::Cpu, &[&stages])
            .expect("normal retained Whisper graph");
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("normal graph must authenticate encoder scratch");
        };
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].stages[0].domains.len(), 1);
        assert!(matches!(
            profiles[0].stages[0].domains[0],
            InvocationWorkspaceDomain::Scratch { .. }
        ));
        assert!(profiles[0].stages[1].domains.is_empty());
        assert!(profiles[0].stages[2].domains.is_empty());
    }

    #[test]
    fn rejects_non_atomic_and_duplicate_stage_graphs() {
        let mut iterative = stage(0);
        iterative.progress = StageProgressKind::Iterative;
        assert!(whisper_physical_state_spec(
            &config(),
            DType::F32,
            BackendKind::Cpu,
            &[&[iterative]],
        )
        .is_err());

        let duplicate = stage(0);
        assert!(whisper_physical_state_spec(
            &config(),
            DType::F32,
            BackendKind::Cpu,
            &[&[duplicate.clone(), duplicate]],
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_geometry_and_backend_dtype() {
        let execution = stage(0);
        let mut invalid = config();
        invalid.d_model = 15;
        assert!(whisper_physical_state_spec(
            &invalid,
            DType::F32,
            BackendKind::Cpu,
            &[std::slice::from_ref(&execution)],
        )
        .is_err());
        assert!(whisper_physical_state_spec(
            &config(),
            DType::BF16,
            BackendKind::Metal,
            &[std::slice::from_ref(&execution)],
        )
        .is_err());
        assert!(whisper_physical_state_spec(
            &config(),
            DType::F32,
            BackendKind::Cuda,
            &[&[execution]],
        )
        .is_err());
    }
}
