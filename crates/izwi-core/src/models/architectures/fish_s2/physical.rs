//! Model-authored physical inference-state contract for Fish S2 DualAR.

use candle_core::DType;

use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, AttentionMask, AttentionPattern, CapabilityStateDescriptorV2,
    CheckpointPolicy, InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationStateCapacity, InvocationWorkspaceDomain, InvocationWorkspaceProfile,
    InvocationWorkspaceSet, KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec,
    PagedAttentionLayerSpec, PlacementPolicy, PrefixPolicy, RetainedStateCapability, StateClock,
    StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec,
    StateScope, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::attention::paged::default_kv_page_size;
#[cfg(test)]
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::FishS2Config;

pub(crate) const FISH_S2_SLOW_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const FISH_S2_FAST_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);
pub(crate) const FISH_S2_SLOW_STATE_GROUP: StateGroupId = StateGroupId::new(1);
pub(crate) const FISH_S2_FAST_STATE_GROUP: StateGroupId = StateGroupId::new(2);

#[derive(Debug, Clone)]
pub(crate) struct FishS2PhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    /// Slow decoder state persists between complete semantic frames. Fast
    /// codebook state is scoped to one decode invocation and resets each frame.
    pub(crate) retained: Option<InferenceStateContract>,
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn fish_s2_physical_state_spec(
    config: &FishS2Config,
    dtype: DType,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<FishS2PhysicalStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "Fish S2 physical state has no execution graph".into(),
        ));
    }
    config.validate()?;
    let page_tokens = u32::try_from(default_kv_page_size())
        .map_err(|_| Error::ModelLoadError("Fish S2 KV page size exceeds u32".into()))?;
    let state_dtype = fish_s2_state_dtype(dtype)?;
    let fast_capacity = u64::try_from(config.num_codebooks)
        .map_err(|_| Error::ModelLoadError("Fish S2 codebook count exceeds u64".into()))?;
    let invocation = fish_s2_invocation_contract(config, state_dtype, page_tokens)?;
    let retained = fish_s2_retained_contract(config, state_dtype, page_tokens)?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("Fish S2 invocation contract is empty".into()))?;

    let mut profiles = Vec::with_capacity(stage_graphs.len());
    let mut has_invocation_workspace = false;
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            if stage.selector == crate::engine::StageWorkSelector::Atomic {
                return Err(Error::ModelLoadError(
                    "Fish S2 requires its retained sequence execution graph".into(),
                ));
            }
            let uses_invocation_state =
                stage.selector == crate::engine::StageWorkSelector::SequenceDecode;
            let mut domains = if uses_invocation_state {
                invocation
                    .domains
                    .iter()
                    .cloned()
                    .map(|state| {
                        Ok(InvocationWorkspaceDomain::State {
                            placement: state.header().placement,
                            formula: WorkspaceFormula {
                                fixed_bytes: fish_s2_paged_invocation_bytes(&state, fast_capacity)?,
                                dimensions: vec![],
                                terms: vec![],
                            },
                            state,
                            capacity: InvocationStateCapacity::PagedTokens {
                                max_tokens: fast_capacity,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            if stage.max_workspace_bytes > 0 {
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError("Fish S2 execution stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("Fish S2 scratch domain id overflow".into())
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
                groups: if uses_invocation_state {
                    invocation.groups.clone()
                } else {
                    Vec::new()
                },
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

    let uses_retained = stage_graphs.iter().any(|stages| {
        stages.iter().any(|stage| {
            matches!(
                stage.selector,
                crate::engine::StageWorkSelector::SequencePrefill
                    | crate::engine::StageWorkSelector::SequenceDecode
            )
        })
    });
    let invocation_workspace = if has_invocation_workspace {
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
        retained: if uses_retained {
            RetainedStateCapability::Managed {
                contract: retained.clone(),
            }
        } else {
            RetainedStateCapability::Stateless
        },
        invocation: invocation_workspace,
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(FishS2PhysicalStateSpec {
        descriptor,
        retained: uses_retained.then_some(retained),
        invocation,
    })
}

fn fish_s2_retained_contract(
    config: &FishS2Config,
    dtype: StateDType,
    page_tokens: u32,
) -> Result<InferenceStateContract> {
    let mut slow = fish_s2_paged_domain(
        FISH_S2_SLOW_STATE_DOMAIN,
        StateClock::DecoderTokens,
        config.text_config.num_hidden_layers,
        config.text_config.num_attention_heads,
        config.text_config.num_key_value_heads,
        config
            .text_config
            .head_dim
            .unwrap_or(config.text_config.hidden_size / config.text_config.num_attention_heads),
        dtype,
        page_tokens,
    )?;
    let StateDomainSpec::PagedAttention(domain) = &mut slow else {
        unreachable!("Fish S2 slow state is paged attention");
    };
    domain.header.scope = StateScope::Retained;
    domain.header.checkpoint = CheckpointPolicy::Transactional;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![slow],
        groups: vec![StateGroupSpec {
            id: FISH_S2_SLOW_STATE_GROUP,
            domains: vec![FISH_S2_SLOW_STATE_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

fn fish_s2_invocation_contract(
    config: &FishS2Config,
    dtype: StateDType,
    page_tokens: u32,
) -> Result<InferenceStateContract> {
    let audio = &config.audio_decoder_config;
    let fast = fish_s2_paged_domain(
        FISH_S2_FAST_STATE_DOMAIN,
        StateClock::CodebookSteps,
        audio.num_hidden_layers,
        audio.num_attention_heads,
        audio.num_key_value_heads,
        audio
            .head_dim
            .unwrap_or(audio.hidden_size / audio.num_attention_heads),
        dtype,
        page_tokens,
    )?;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![fast],
        groups: vec![StateGroupSpec {
            id: FISH_S2_FAST_STATE_GROUP,
            domains: vec![FISH_S2_FAST_STATE_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

fn fish_s2_paged_domain(
    id: StateDomainId,
    clock: StateClock,
    num_layers: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    dtype: StateDType,
    page_tokens: u32,
) -> Result<StateDomainSpec> {
    let query_heads = u32::try_from(query_heads)
        .map_err(|_| Error::ModelLoadError("Fish S2 query-head count exceeds u32".into()))?;
    let kv_heads = u32::try_from(kv_heads)
        .map_err(|_| Error::ModelLoadError("Fish S2 KV-head count exceeds u32".into()))?;
    let head_dim = u32::try_from(head_dim)
        .map_err(|_| Error::ModelLoadError("Fish S2 head dimension exceeds u32".into()))?;
    let max_page_tokens = page_tokens.max(256);
    let layers = (0..num_layers)
        .map(|model_layer| {
            Ok(PagedAttentionLayerSpec {
                model_layer: u32::try_from(model_layer)
                    .map_err(|_| Error::ModelLoadError("Fish S2 layer count exceeds u32".into()))?,
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
    Ok(StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
        header: StateDomainHeader {
            id,
            scope: StateScope::Invocation,
            clock,
            placement: PlacementPolicy::BackendLocal,
            prefix: PrefixPolicy::Disabled,
            checkpoint: CheckpointPolicy::None,
        },
        layers,
        page_size: PageSizeConstraint {
            min_tokens: 1,
            preferred_tokens: page_tokens,
            max_tokens: max_page_tokens,
            multiple_of: 1,
        },
        accepted_dtypes: vec![dtype],
    }))
}

fn fish_s2_state_dtype(dtype: DType) -> Result<StateDType> {
    match dtype {
        DType::F32 => Ok(StateDType::F32),
        DType::F16 => Ok(StateDType::F16),
        DType::BF16 => Ok(StateDType::Bf16),
        other => Err(Error::ModelLoadError(format!(
            "Fish S2 physical paging does not support {other:?} KV storage"
        ))),
    }
}

fn fish_s2_paged_invocation_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(Error::ModelLoadError(
            "Fish S2 invocation workspace is not paged attention".into(),
        ));
    };
    let page_tokens = u64::from(spec.page_size.preferred_tokens);
    let rounded_tokens = max_tokens
        .checked_add(page_tokens.saturating_sub(1))
        .and_then(|tokens| tokens.checked_div(page_tokens))
        .and_then(|pages| pages.checked_mul(page_tokens))
        .ok_or_else(|| Error::ModelLoadError("Fish S2 page capacity overflow".into()))?;
    let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
        let layer_elements = u64::from(layer.kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| Error::ModelLoadError("Fish S2 KV geometry overflow".into()))?;
        total
            .checked_add(layer_elements)
            .ok_or_else(|| Error::ModelLoadError("Fish S2 KV geometry overflow".into()))
    })?;
    let element_bytes = spec
        .accepted_dtypes
        .first()
        .map(|dtype| match dtype {
            StateDType::F32 => Ok(4_u64),
            StateDType::F16 | StateDType::Bf16 => Ok(2_u64),
            StateDType::I64 | StateDType::I8 | StateDType::Q4 => Err(Error::ModelLoadError(
                "Fish S2 physical paging requires a dense loaded KV dtype".into(),
            )),
        })
        .transpose()?
        .ok_or_else(|| Error::ModelLoadError("Fish S2 KV dtype set is empty".into()))?;
    elements_per_token
        .checked_mul(rounded_tokens)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| Error::ModelLoadError("Fish S2 invocation byte bound overflow".into()))
}

#[cfg(test)]
pub(super) fn test_physical_cache(
    model_instance: u64,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    capacity_tokens: usize,
) -> PhysicalPagedKvCache {
    use std::sync::Arc;

    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    let page_tokens = 4usize;
    let capacity_pages = capacity_tokens.div_ceil(page_tokens);
    let arena_id = KvArenaId {
        model_instance: ModelInstanceId::new(model_instance),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let bindings = (0..num_layers)
        .map(|model_layer| KvLayerBinding {
            model_layer: u32::try_from(model_layer).expect("test model layer"),
            physical_layer: u32::try_from(model_layer).expect("test physical layer"),
        })
        .collect::<Vec<_>>();
    let arena: Arc<dyn KvArena> = Arc::new(
        CpuKvArena::new(KvArenaConfig {
            id: arena_id,
            group,
            page_tokens: u32::try_from(page_tokens).expect("test page tokens"),
            capacity_pages: u32::try_from(capacity_pages).expect("test capacity pages"),
            growth: None,
            dtype: DType::F32,
            layers: bindings
                .iter()
                .copied()
                .map(|binding| KvLayerConfig {
                    binding,
                    num_kv_heads: u32::try_from(num_kv_heads).expect("test KV heads"),
                    key_head_dim: u32::try_from(head_dim).expect("test key head dimension"),
                    value_head_dim: u32::try_from(head_dim).expect("test value head dimension"),
                })
                .collect(),
        })
        .expect("test CPU KV arena"),
    );
    let blocks = (0..capacity_pages)
        .map(|index| CacheBlockRef {
            arena: arena_id,
            group,
            index: u32::try_from(index).expect("test page index"),
            slot_generation: 1,
        })
        .collect();
    PhysicalPagedKvCache::new(arena, bindings, blocks, 0).expect("test physical cache")
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::{
        ClockedStateSelection, ConcurrencyClass, ExecutionDomain, MembershipSafePoint,
        NativeBatchMode, OutputVisibility, StageId, StageProgressKind, StageShapePolicy,
        StageWorkSelector,
    };
    use crate::models::architectures::fish_s2::config::current_config;

    fn stage() -> StageDescriptor {
        StageDescriptor {
            id: StageId::new(1),
            name: "fish_s2.generate".into(),
            selector: StageWorkSelector::Atomic,
            domain: ExecutionDomain::ExecutionGroup,
            progress: StageProgressKind::Atomic,
            concurrency: ConcurrencyClass::Batchable,
            physical_launch_policy: crate::engine::PhysicalLaunchPolicy::ExecutionGroupExclusive,
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
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

    fn retained_stage(id: u32, selector: StageWorkSelector) -> StageDescriptor {
        let mut stage = stage();
        stage.id = StageId::new(id);
        stage.name = format!("fish_s2.retained.{id}");
        stage.selector = selector;
        stage.progress = StageProgressKind::Iterative;
        stage.concurrency = ConcurrencyClass::Exclusive;
        stage.shape_policy = StageShapePolicy::Exact;
        stage.retained_state_selections = Some(vec![ClockedStateSelection::new(
            FISH_S2_SLOW_STATE_GROUP,
            StateClock::DecoderTokens,
        )
        .expect("slow state selection")]);
        stage
    }

    #[test]
    fn staged_graph_retains_only_slow_and_leases_one_fast_frame() {
        let config = current_config();
        let stages = [
            retained_stage(1, StageWorkSelector::SequencePrefill),
            retained_stage(2, StageWorkSelector::SequenceDecode),
        ];
        let spec = fish_s2_physical_state_spec(&config, DType::F16, &[&stages]).unwrap();
        let retained = spec.retained.as_ref().unwrap();
        assert_eq!(retained.domains.len(), 1);
        assert_eq!(retained.groups.len(), 1);
        let slow = &retained.domains[0];
        assert_eq!(slow.id(), FISH_S2_SLOW_STATE_DOMAIN);
        assert_eq!(slow.header().clock, StateClock::DecoderTokens);
        assert_eq!(slow.header().scope, StateScope::Retained);
        assert_eq!(slow.header().checkpoint, CheckpointPolicy::Transactional);
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("bounded fast cache");
        };
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0].stages[0].domains.is_empty());
        let decode = &profiles[0].stages[1];
        assert_eq!(decode.domains.len(), 1);
        let InvocationWorkspaceDomain::State {
            state,
            capacity,
            formula,
            ..
        } = &decode.domains[0]
        else {
            panic!("fast state");
        };
        assert_eq!(state.id(), FISH_S2_FAST_STATE_DOMAIN);
        assert_eq!(state.header().clock, StateClock::CodebookSteps);
        assert_eq!(capacity.paged_max_tokens(), Some(10));
        assert_eq!(
            formula.fixed_bytes,
            fish_s2_paged_invocation_bytes(state, 10).unwrap()
        );
        assert_eq!(formula.fixed_bytes, 1024 * 1024); // one 64-slot half-precision page
        spec.descriptor.validate_against_stages(&stages).unwrap();
    }

    #[test]
    fn rejects_unreachable_atomic_graph_instead_of_allocating_duplicate_slow_cache() {
        assert!(fish_s2_physical_state_spec(&current_config(), DType::F16, &[&[stage()]]).is_err());
    }
}
