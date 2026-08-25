//! Voxtral family implementations.

use candle_core::DType;

use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, StateDType, StateDomainId, StateDomainSpec, StateScope,
    WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};

mod layers;
pub mod lm;
pub mod realtime;
pub mod tts;

use lm::VoxtralLM;

#[derive(Debug, Clone)]
pub(crate) struct VoxtralPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

#[derive(Debug, Clone)]
pub(crate) struct VoxtralRealtimePhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
}

pub(crate) fn voxtral_invocation_contract(
    model: &VoxtralLM,
    dtype: DType,
    preferred_page_tokens: usize,
    domains: &[StateDomainId],
) -> Result<InferenceStateContract> {
    if domains.is_empty() {
        return Err(Error::ModelLoadError(
            "Voxtral invocation state has no cache domains".into(),
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
    let mut contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: state_domains,
        groups,
    };
    for domain in &mut contract.domains {
        let StateDomainSpec::PagedAttention(domain) = domain else {
            return Err(Error::ModelLoadError(
                "Voxtral invocation state must be paged attention".into(),
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

pub(crate) fn voxtral_retained_contract(
    mut contract: InferenceStateContract,
) -> Result<InferenceStateContract> {
    for domain in &mut contract.domains {
        let StateDomainSpec::PagedAttention(domain) = domain else {
            return Err(Error::ModelLoadError(
                "Voxtral retained state must be paged attention".into(),
            ));
        };
        domain.header.scope = StateScope::Retained;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::Transactional;
    }
    for group in &mut contract.groups {
        group.prefix_shareable = false;
    }
    contract.validate()?;
    Ok(contract)
}

pub(crate) fn voxtral_realtime_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    retained: InferenceStateContract,
    max_context_tokens: usize,
) -> Result<VoxtralRealtimePhysicalStateSpec> {
    if stage_graphs.is_empty() || max_context_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral retained realtime state requires stages and a non-zero context".into(),
        ));
    }
    let max_domain_id = retained
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("Voxtral retained contract is empty".into()))?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let domains = if stage.max_workspace_bytes == 0 {
                Vec::new()
            } else {
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError(
                            "Voxtral realtime execution stage count exceeds u32".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("Voxtral realtime scratch domain overflow".into())
                    })?;
                vec![InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: stage.max_workspace_bytes,
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
    Ok(VoxtralRealtimePhysicalStateSpec {
        descriptor,
        retained,
        retained_max_tokens: max_context_tokens,
    })
}

pub(crate) fn voxtral_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    invocation: InferenceStateContract,
    max_context_tokens: usize,
) -> Result<VoxtralPhysicalStateSpec> {
    if stage_graphs.is_empty() || max_context_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral invocation state requires stages and a non-zero context".into(),
        ));
    }
    let max_tokens = u64::try_from(max_context_tokens)
        .map_err(|_| Error::ModelLoadError("Voxtral context exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("Voxtral invocation contract is empty".into()))?;
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
                    Ok(InvocationWorkspaceDomain::State {
                        placement: state.header().placement,
                        formula: WorkspaceFormula {
                            fixed_bytes: voxtral_paged_invocation_bytes(&state, max_tokens)?,
                            dimensions: vec![],
                            terms: vec![],
                        },
                        state,
                        capacity: InvocationStateCapacity::decoder_context(max_tokens)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if stage.max_workspace_bytes > 0 {
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError("Voxtral execution stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("Voxtral scratch domain id overflow".into())
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
    Ok(VoxtralPhysicalStateSpec {
        descriptor,
        invocation,
    })
}

fn voxtral_paged_invocation_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(Error::ModelLoadError(
            "Voxtral invocation workspace is not paged attention".into(),
        ));
    };
    let page_tokens = u64::from(spec.page_size.preferred_tokens);
    let rounded_tokens = max_tokens
        .checked_add(page_tokens.saturating_sub(1))
        .and_then(|tokens| tokens.checked_div(page_tokens))
        .and_then(|pages| pages.checked_mul(page_tokens))
        .ok_or_else(|| Error::ModelLoadError("Voxtral page capacity overflow".into()))?;
    let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
        let layer_elements = u64::from(layer.kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| Error::ModelLoadError("Voxtral KV geometry overflow".into()))?;
        total
            .checked_add(layer_elements)
            .ok_or_else(|| Error::ModelLoadError("Voxtral KV geometry overflow".into()))
    })?;
    let element_bytes = spec
        .accepted_dtypes
        .iter()
        .map(|dtype| match dtype {
            StateDType::F32 => Ok(4_u64),
            StateDType::F16 | StateDType::Bf16 => Ok(2_u64),
            StateDType::I64 | StateDType::I8 | StateDType::Q4 => Err(Error::ModelLoadError(
                "Voxtral invocation paging requires a dense loaded KV dtype".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .min()
        .ok_or_else(|| Error::ModelLoadError("Voxtral KV dtype set is empty".into()))?;
    elements_per_token
        .checked_mul(rounded_tokens)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| Error::ModelLoadError("Voxtral invocation byte bound overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::voxtral_retained_contract;
    use crate::kv::v2::{
        test_contract, CheckpointPolicy, PrefixPolicy, RetainedStateCapability, StateScope,
    };

    #[test]
    fn realtime_contract_is_transactional_retained_and_not_prefix_shareable() {
        let mut invocation = test_contract();
        for domain in &mut invocation.domains {
            match domain {
                crate::kv::v2::StateDomainSpec::PagedAttention(domain) => {
                    domain.header.scope = StateScope::Invocation;
                    domain.header.checkpoint = CheckpointPolicy::None;
                }
                other => panic!("unexpected test domain: {other:?}"),
            }
        }

        let retained = voxtral_retained_contract(invocation).unwrap();

        assert!(retained.domains.iter().all(|domain| {
            domain.header().scope == StateScope::Retained
                && domain.header().checkpoint == CheckpointPolicy::Transactional
                && domain.header().prefix == PrefixPolicy::Disabled
        }));
        assert!(retained.groups.iter().all(|group| !group.prefix_shareable));
        let capability = RetainedStateCapability::Managed {
            contract: retained.clone(),
        };
        assert!(matches!(
            capability,
            RetainedStateCapability::Managed { contract } if contract == retained
        ));
    }
}
