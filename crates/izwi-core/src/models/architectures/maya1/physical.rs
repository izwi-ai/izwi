//! Maya1's ABI-v2 state contract.
//!
//! The decoder keeps one paged K/V domain (domain and group 1), retained
//! across the request's quanta with transactional checkpoints. Stages hold
//! only per-row scratch (the SNAC window workspace of the audio stage).

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

use super::config::Maya1LmConfig;

pub(crate) const MAYA1_DOMAIN: StateDomainId = StateDomainId::new(1);

#[derive(Debug, Clone)]
pub(crate) struct Maya1PhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
}

/// The decoder's retained K/V: one domain and group.
pub(crate) fn maya1_retained_contract(
    cfg: &Maya1LmConfig,
    storage_dtype: DType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    let mut spec = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
        domain: MAYA1_DOMAIN,
        clock: StateClock::DecoderTokens,
        num_layers: cfg.num_hidden_layers,
        num_query_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
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
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::PagedAttention(spec)],
        groups: vec![StateGroupSpec {
            id: StateGroupId::new(MAYA1_DOMAIN.get()),
            domains: vec![MAYA1_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

pub(crate) fn maya1_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    retained: InferenceStateContract,
    retained_max_tokens: usize,
) -> Result<Maya1PhysicalStateSpec> {
    if stage_graphs.is_empty() || retained_max_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Maya1 retained state requires stages and a non-zero context".into(),
        ));
    }
    let max_domain_id = MAYA1_DOMAIN.get();
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
                        Error::ModelLoadError("Maya1 stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| Error::ModelLoadError("Maya1 scratch domain overflow".into()))?;
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
    Ok(Maya1PhysicalStateSpec {
        descriptor,
        retained,
        retained_max_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_contract_is_one_gqa_group() {
        let contract = maya1_retained_contract(&Maya1LmConfig::pinned(), DType::F16, 16).unwrap();
        assert_eq!(contract.domains.len(), 1);
        assert_eq!(contract.groups.len(), 1);
        assert!(!contract.groups[0].prefix_shareable);
        let StateDomainSpec::PagedAttention(spec) = &contract.domains[0] else {
            panic!("decoder state is paged attention");
        };
        assert_eq!(spec.header.scope, StateScope::Retained);
        assert_eq!(spec.header.checkpoint, CheckpointPolicy::Transactional);
        assert_eq!(spec.layers.len(), 28);
        assert!(spec
            .layers
            .iter()
            .all(|layer| layer.kv_heads == 8 && layer.key_head_dim == 128));
    }
}
