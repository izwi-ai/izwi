//! Native VibeVoice model components.

use candle_core::DType;

use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, upgrade_kv_contract_v1, CapabilityStateDescriptorV2, CheckpointPolicy,
    InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationStateCapacity, InvocationWorkspaceDomain, InvocationWorkspaceProfile,
    InvocationWorkspaceSet, PlacementPolicy, PrefixPolicy, RetainedStateCapability, StateDType,
    StateDomainSpec, StateScope, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::kv::{CacheDomainId, KvCacheContract};
use crate::models::architectures::qwen3::core::Qwen3Model;

pub mod asr;
pub mod config;
pub mod connector;
pub mod diffusion;
pub mod prompt;
pub mod tokenizer;
pub mod tts;

#[derive(Debug, Clone)]
pub(crate) struct VibeVoicePhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

pub(crate) fn vibevoice_invocation_contract(
    model: &Qwen3Model,
    dtype: DType,
    preferred_page_tokens: usize,
    domains: &[CacheDomainId],
) -> Result<InferenceStateContract> {
    if domains.is_empty() {
        return Err(Error::ModelLoadError(
            "VibeVoice invocation state has no cache domains".into(),
        ));
    }
    let mut legacy_domains = Vec::with_capacity(domains.len());
    for domain in domains {
        let contract = model.managed_kv_cache_contract(*domain, dtype, preferred_page_tokens)?;
        legacy_domains.extend(contract.domains);
    }
    let legacy = KvCacheContract {
        abi: crate::kv::CURRENT_KV_CONTRACT_ABI,
        domains: legacy_domains,
    };
    vibevoice_invocation_contract_from_legacy(&legacy)
}

fn vibevoice_invocation_contract_from_legacy(
    legacy: &KvCacheContract,
) -> Result<InferenceStateContract> {
    legacy.validate()?;
    let mut contract = upgrade_kv_contract_v1(legacy)?;
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
    contract.validate()?;
    Ok(contract)
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
                    Ok(InvocationWorkspaceDomain::State {
                        placement: state.header().placement,
                        formula: WorkspaceFormula {
                            fixed_bytes: vibevoice_paged_invocation_bytes(&state, max_tokens)?,
                            dimensions: vec![],
                            terms: vec![],
                        },
                        state,
                        capacity: InvocationStateCapacity::PagedTokens { max_tokens },
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
    let descriptor =
        vibevoice_invocation_descriptor(stage_graphs, &invocation, max_context_tokens)?;
    Ok(VibeVoicePhysicalStateSpec {
        descriptor,
        invocation,
    })
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
            StateDType::I8 | StateDType::Q4 => Err(Error::ModelLoadError(
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::{
        ConcurrencyClass, ExecutionDomain, MembershipSafePoint, NativeBatchMode, OutputVisibility,
        StageId, StageProgressKind, StageShapePolicy, StageWorkSelector,
    };
    use crate::kv::v2::{StateDomainId, StateGroupId};
    use crate::kv::{CacheTokenAxis, KvDomainSpec, KvPrefixSemantics, PositionSemantics};
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
        }
    }

    fn invocation_contract(domain_count: u32) -> InferenceStateContract {
        let domains = (0..domain_count)
            .map(|domain| {
                qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
                    domain: CacheDomainId::new(domain),
                    token_axis: CacheTokenAxis::DecoderTokens,
                    num_layers: 2,
                    num_query_heads: 4,
                    num_kv_heads: 2,
                    key_head_dim: 8,
                    value_head_dim: 8,
                    sliding_window: None,
                    storage_dtype: DType::F32,
                    preferred_page_tokens: 16,
                    prefix_semantics: KvPrefixSemantics::CommittedFullPages {
                        positions: PositionSemantics::Absolute,
                    },
                })
                .map(KvDomainSpec::PagedAttention)
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();
        vibevoice_invocation_contract_from_legacy(&KvCacheContract {
            abi: crate::kv::CURRENT_KV_CONTRACT_ABI,
            domains,
        })
        .unwrap()
    }

    #[test]
    fn asr_descriptor_is_stateless_with_one_exact_invocation_domain() {
        let contract = invocation_contract(1);
        let execution = stage();
        let descriptor =
            vibevoice_invocation_descriptor(&[&[execution.clone()]], &contract, 4096).unwrap();
        assert!(descriptor.is_stateless());
        let InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
            panic!("ASR invocation pages must be bounded");
        };
        let workspace = &profiles[0].stages[0];
        assert_eq!(workspace.lease_scope, InvocationLeaseScope::PerRow);
        assert_eq!(workspace.groups.len(), 1);
        assert_eq!(workspace.domains.len(), 1);
        assert!(matches!(
            &workspace.domains[0],
            InvocationWorkspaceDomain::State {
                capacity: InvocationStateCapacity::PagedTokens { max_tokens: 4096 },
                formula: WorkspaceFormula {
                    fixed_bytes: 1_048_576,
                    ..
                },
                ..
            }
        ));
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
        assert!(workspace.domains.iter().all(|domain| matches!(
            domain,
            InvocationWorkspaceDomain::State {
                capacity: InvocationStateCapacity::PagedTokens { max_tokens: 8192 },
                formula: WorkspaceFormula {
                    fixed_bytes: 2_097_152,
                    ..
                },
                ..
            }
        )));
    }
}
