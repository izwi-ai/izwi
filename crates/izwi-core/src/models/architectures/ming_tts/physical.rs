//! Ming's ABI-v2 state contract.
//!
//! The language model keeps one paged K/V domain (domain and group 1),
//! retained across the request's quanta with transactional checkpoints.
//! Stages hold only per-row scratch (the VAE window workspace of the audio
//! stage).

use candle_core::DType;

use super::config::MingStackConfig;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, AttentionMask, AttentionPattern, CapabilityStateDescriptorV2,
    CheckpointPolicy, InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, KeyEncoding,
    PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, StateClock, StateDType, StateDomainHeader,
    StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope, WorkspaceFormula,
    CURRENT_INFERENCE_STATE_ABI,
};

pub(crate) const MING_DOMAIN: StateDomainId = StateDomainId::new(1);

#[derive(Debug, Clone)]
pub(crate) struct MingPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
}

/// The decoder's retained K/V: one domain and group.
pub(crate) fn ming_retained_contract(
    cfg: &MingStackConfig,
    storage_dtype: DType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    let dim = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| Error::InvalidInput(format!("Ming {what} exceeds u32")))
    };
    let storage = match storage_dtype {
        DType::F32 => StateDType::F32,
        DType::F16 => StateDType::F16,
        DType::BF16 => StateDType::Bf16,
        dtype => {
            return Err(Error::InvalidInput(format!(
                "Ming managed KV does not support {dtype:?} storage"
            )))
        }
    };
    let preferred = dim(preferred_page_tokens.max(1), "page size")?;
    let head_dim = dim(cfg.head_dim, "head dimension")?;
    let layers = (0..dim(cfg.num_hidden_layers, "layer count")?)
        .map(|model_layer| {
            Ok(PagedAttentionLayerSpec {
                model_layer,
                query_heads: dim(cfg.num_attention_heads, "query head count")?,
                kv_heads: dim(cfg.num_key_value_heads, "KV head count")?,
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
    let mut spec = PagedAttentionDomainSpec {
        header: StateDomainHeader {
            id: MING_DOMAIN,
            scope: StateScope::Retained,
            clock: StateClock::DecoderTokens,
            placement: PlacementPolicy::BackendLocalWithHostOffload,
            checkpoint: CheckpointPolicy::Transactional,
            prefix: PrefixPolicy::Disabled,
        },
        layers,
        page_size: PageSizeConstraint {
            min_tokens: 1,
            preferred_tokens: preferred,
            max_tokens: preferred.max(256),
            multiple_of: 1,
        },
        accepted_dtypes: vec![storage],
    };
    spec.header.scope = StateScope::Retained;
    spec.header.prefix = PrefixPolicy::Disabled;
    spec.header.checkpoint = CheckpointPolicy::Transactional;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::PagedAttention(spec)],
        groups: vec![StateGroupSpec {
            id: StateGroupId::new(MING_DOMAIN.get()),
            domains: vec![MING_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

pub(crate) fn ming_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    retained: InferenceStateContract,
    retained_max_tokens: usize,
) -> Result<MingPhysicalStateSpec> {
    if stage_graphs.is_empty() || retained_max_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Ming retained state requires stages and a non-zero context".into(),
        ));
    }
    let max_domain_id = MING_DOMAIN.get();
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
                        Error::ModelLoadError("Ming stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| Error::ModelLoadError("Ming scratch domain overflow".into()))?;
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
    Ok(MingPhysicalStateSpec {
        descriptor,
        retained,
        retained_max_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::ming_tts::config::MingTtsConfig;

    #[test]
    fn retained_contract_is_one_gqa_group() {
        let cfg = MingTtsConfig::pinned().llm;
        let contract = ming_retained_contract(&cfg, DType::F16, 16).unwrap();
        assert_eq!(contract.domains.len(), 1);
        assert_eq!(contract.groups.len(), 1);
        assert!(!contract.groups[0].prefix_shareable);
        let StateDomainSpec::PagedAttention(spec) = &contract.domains[0] else {
            panic!("language-model state is paged attention");
        };
        assert_eq!(spec.header.scope, StateScope::Retained);
        assert_eq!(spec.header.checkpoint, CheckpointPolicy::Transactional);
        assert_eq!(spec.layers.len(), 24);
        assert!(spec.layers.iter().all(|layer| layer.kv_heads == 2
            && layer.query_heads == 14
            && layer.key_head_dim == 64));
        assert!(ming_retained_contract(&cfg, DType::U8, 16).is_err());
    }
}
