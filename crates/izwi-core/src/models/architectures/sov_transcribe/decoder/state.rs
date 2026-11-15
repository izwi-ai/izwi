//! A row's decoder state contract.
//!
//! One consistency group with one domain: paged attention over all 30
//! layers, rotated keys. The layers do not share a geometry: sliding layers
//! store 8 KV heads of 256 dims and see a 1024-token window, full layers
//! store 2 KV heads of 512 dims and see everything. Prefixes are not shared
//! (prompts share about 20 tokens before the audio). The domain is
//! transactional: a quantum's KV pages commit or roll back together with
//! the row's host decode loop.

use candle_core::DType;

use super::super::config::{SovDecoderConfig, SovLayerKind};
use crate::error::{Error, Result};
use crate::kv::v2::{
    AttentionMask, AttentionPattern, CheckpointPolicy, InferenceStateContract, KeyEncoding,
    PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec, PlacementPolicy,
    PrefixPolicy, StateClock, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec,
    StateGroupId, StateGroupSpec, StateScope, CURRENT_INFERENCE_STATE_ABI,
};

pub const ATTENTION_DOMAIN: StateDomainId = StateDomainId::new(1);
pub const STATE_GROUP: StateGroupId = StateGroupId::new(1);

fn as_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::InvalidInput(format!("SOV-Transcribe {what} exceeds u32")))
}

/// The decoder's retained-state contract with KV stored as `kv_dtype`.
pub(crate) fn decoder_state_contract(
    config: &SovDecoderConfig,
    kv_dtype: StateDType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    let preferred = as_u32(preferred_page_tokens.max(1), "page size")?;
    let layers = config
        .layer_kinds
        .iter()
        .enumerate()
        .map(|(layer, kind)| {
            let g = config.geometry(layer);
            Ok(PagedAttentionLayerSpec {
                model_layer: as_u32(layer, "model layer")?,
                query_heads: as_u32(g.heads, "query heads")?,
                kv_heads: as_u32(g.kv_heads, "KV heads")?,
                key_head_dim: as_u32(g.head_dim, "head dim")?,
                value_head_dim: as_u32(g.head_dim, "head dim")?,
                pattern: match (kind, g.window) {
                    (SovLayerKind::Sliding, Some(window)) => AttentionPattern::SlidingWindow {
                        window_tokens: as_u32(window, "window")?,
                    },
                    _ => AttentionPattern::Full,
                },
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Rotary {
                    rotary_dim: as_u32(2 * g.rotary_pairs, "rotary dim")?,
                },
                attention_logit_softcap: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
            header: StateDomainHeader {
                id: ATTENTION_DOMAIN,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::Transactional,
            },
            layers,
            page_size: PageSizeConstraint {
                min_tokens: 1,
                preferred_tokens: preferred,
                max_tokens: preferred.max(256),
                multiple_of: 1,
            },
            accepted_dtypes: vec![kv_dtype],
        })],
        groups: vec![StateGroupSpec {
            id: STATE_GROUP,
            domains: vec![ATTENTION_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

/// The paged KV dtype per backend: the compute dtype (BF16 on CUDA, F32 on
/// the CPU test reference). Metal never loads the model; F16 is what its
/// arena would take.
pub(crate) fn kv_state_dtype(backend: crate::backends::BackendKind) -> StateDType {
    match backend {
        crate::backends::BackendKind::Cpu => StateDType::F32,
        crate::backends::BackendKind::Metal => StateDType::F16,
        crate::backends::BackendKind::Cuda => StateDType::Bf16,
    }
}

/// The capability descriptor of the retained rows: the managed contract,
/// and per row of each stage graph its activations as scratch.
pub(crate) fn retained_state_descriptor(
    contract: &InferenceStateContract,
    stage_graphs: &[&[crate::engine::StageDescriptor]],
) -> Result<crate::kv::v2::CapabilityStateDescriptorV2> {
    use crate::kv::v2::{
        stage_graph_fingerprint, CapabilityStateDescriptorV2, InvocationLeaseScope,
        InvocationStageWorkspace, InvocationWorkspaceDomain, InvocationWorkspaceProfile,
        InvocationWorkspaceSet, RetainedStateCapability, WorkspaceFormula,
    };
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "SOV-Transcribe retained state has no execution graph".into(),
        ));
    }
    let max_domain = contract
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .unwrap_or(0);
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut workspace = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let per_row = if stage.workspace_per_row_bytes == 0 {
                stage.max_workspace_bytes / u64::try_from(stage.max_batch_size.max(1)).unwrap_or(1)
            } else {
                stage.workspace_per_row_bytes
            };
            let mut domains = Vec::new();
            if per_row > 0 {
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(max_domain + 1 + as_u32(index, "stage")?),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: per_row,
                        dimensions: vec![],
                        terms: vec![],
                    },
                });
            }
            workspace.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: Vec::new(),
                domains,
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: workspace,
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: RetainedStateCapability::Managed {
            contract: contract.clone(),
        },
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(descriptor)
}

/// KV bytes per token across every layer in `kv_dtype`.
pub fn kv_bytes_per_token(config: &SovDecoderConfig, kv_dtype: DType) -> u64 {
    (0..config.layers())
        .map(|layer| {
            let g = config.geometry(layer);
            (2 * g.kv_heads * g.head_dim * kv_dtype.size_in_bytes()) as u64
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_contract_mixes_sliding_and_full_layers() {
        let config = super::super::super::config::tests::published();
        let contract = decoder_state_contract(&config.decoder, StateDType::Bf16, 16).unwrap();
        let StateDomainSpec::PagedAttention(attention) = &contract.domains[0] else {
            panic!("attention first");
        };
        assert_eq!(attention.layers.len(), 30);
        for (layer, spec) in attention.layers.iter().enumerate() {
            let full = [5, 11, 17, 23, 29].contains(&layer);
            assert_eq!(spec.query_heads, 16);
            assert_eq!(spec.kv_heads, if full { 2 } else { 8 });
            assert_eq!(spec.key_head_dim, if full { 512 } else { 256 });
            assert_eq!(
                spec.pattern,
                if full {
                    AttentionPattern::Full
                } else {
                    AttentionPattern::SlidingWindow {
                        window_tokens: 1024,
                    }
                }
            );
            assert_eq!(
                spec.key_encoding,
                KeyEncoding::Rotary {
                    rotary_dim: if full { 128 } else { 256 }
                }
            );
            assert!(spec.attention_logit_softcap.is_none());
        }
        // 220 KiB of BF16 KV per token: 25 × 8 × 256 × 2 + 5 × 2 × 512 × 2 values.
        assert_eq!(kv_bytes_per_token(&config.decoder, DType::BF16), 225_280);
        assert!(contract
            .domains
            .iter()
            .all(|d| d.header().prefix == PrefixPolicy::Disabled));
    }
}
