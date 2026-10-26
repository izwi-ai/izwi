//! Chatterbox's ABI-v2 state contract.
//!
//! T3 keeps one paged K/V domain per guidance branch: the conditional
//! sequence (domain and group 1) and the unconditional one (domain and group
//! 2). Each group gets its own arena, so the two branches never share pages,
//! and a request's claim prices both. Both are retained across the
//! request's quanta with transactional checkpoints; the stages hold only
//! per-row scratch (the flow and vocoder workspace of the acoustic stage).

use candle_core::DType;

use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationWorkspaceDomain,
    InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy, PrefixPolicy,
    RetainedStateCapability, StateClock, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::qwen3::core::{
    qwen3_decoder_cache_domain, Qwen3DecoderCacheGeometry,
};

use super::config::T3Config;

pub(crate) const CHATTERBOX_COND_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const CHATTERBOX_UNCOND_DOMAIN: StateDomainId = StateDomainId::new(2);

#[derive(Debug, Clone)]
pub(crate) struct ChatterboxPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
}

/// T3's retained K/V: one domain and group per guidance branch.
pub(crate) fn chatterbox_retained_contract(
    cfg: &T3Config,
    storage_dtype: DType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    let mut domains = Vec::with_capacity(2);
    let mut groups = Vec::with_capacity(2);
    for domain in [CHATTERBOX_COND_DOMAIN, CHATTERBOX_UNCOND_DOMAIN] {
        let mut spec = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
            domain,
            clock: StateClock::DecoderTokens,
            num_layers: cfg.num_hidden_layers,
            num_query_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_attention_heads,
            key_head_dim: cfg.head_dim,
            value_head_dim: cfg.head_dim,
            sliding_window: None,
            storage_dtype,
            preferred_page_tokens,
            prefix: PrefixPolicy::Disabled,
        })?;
        spec.header.scope = StateScope::Retained;
        spec.header.prefix = PrefixPolicy::Disabled;
        spec.header.checkpoint = CheckpointPolicy::Transactional;
        domains.push(StateDomainSpec::PagedAttention(spec));
        groups.push(StateGroupSpec {
            id: StateGroupId::new(domain.get()),
            domains: vec![domain],
            prefix_shareable: false,
        });
    }
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains,
        groups,
    };
    contract.validate()?;
    Ok(contract)
}

pub(crate) fn chatterbox_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    retained: InferenceStateContract,
    retained_max_tokens: usize,
) -> Result<ChatterboxPhysicalStateSpec> {
    if stage_graphs.is_empty() || retained_max_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Chatterbox retained state requires stages and a non-zero context".into(),
        ));
    }
    let max_domain_id = retained
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("Chatterbox retained contract is empty".into()))?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let per_row_workspace = if stage.workspace_per_row_bytes == 0 {
                stage.max_workspace_bytes / u64::try_from(stage.max_batch_size.max(1)).unwrap_or(1)
            } else {
                stage.workspace_per_row_bytes
            };
            let domains = if per_row_workspace == 0 {
                Vec::new()
            } else {
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError("Chatterbox stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("Chatterbox scratch domain overflow".into())
                    })?;
                vec![InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: per_row_workspace,
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
    Ok(ChatterboxPhysicalStateSpec {
        descriptor,
        retained,
        retained_max_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_contract_has_one_group_per_guidance_branch() {
        let contract = chatterbox_retained_contract(&T3Config::english(), DType::F32, 16).unwrap();
        assert_eq!(contract.domains.len(), 2);
        assert_eq!(contract.groups.len(), 2);
        for (group, domain) in contract
            .groups
            .iter()
            .zip([CHATTERBOX_COND_DOMAIN, CHATTERBOX_UNCOND_DOMAIN])
        {
            assert_eq!(group.domains, vec![domain]);
            assert!(!group.prefix_shareable);
        }
        for domain in &contract.domains {
            let StateDomainSpec::PagedAttention(spec) = domain else {
                panic!("T3 state is paged attention");
            };
            assert_eq!(spec.header.scope, StateScope::Retained);
            assert_eq!(spec.header.checkpoint, CheckpointPolicy::Transactional);
            assert_eq!(spec.layers.len(), 30);
            assert!(spec
                .layers
                .iter()
                .all(|layer| layer.kv_heads == 16 && layer.key_head_dim == 64));
        }
    }
}
