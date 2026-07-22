//! Temporary semantic upgrader used while model declarations move from the
//! KV-only ABI to the unified inference-state ABI. It performs no runtime
//! fallback: the result is a complete v2 contract and is negotiated only by
//! v2 backend policy. This module is deleted with the v1 contract types after
//! the final model migration.

use crate::error::{Error, Result};
use crate::kv::{
    AttentionSemantics as V1Attention, CacheTokenAxis, KeyEncoding as V1KeyEncoding,
    KvCacheContract, KvDomainSpec, KvPrefixSemantics, KvStorageDType, ModelStateKind,
    PositionSemantics as V1Position,
};

use super::{
    AttentionMask, AttentionPattern, BoundedShape, CheckpointPolicy, InferenceStateContract,
    KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec,
    PlacementPolicy, PositionSemantics, PrefixPolicy, ShapeAxis, ShapeDimension, ShapeExtent,
    StateClock, StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec,
    StateGroupId, StateGroupSpec, StateScope, TensorComponentSpec, TensorRole,
    TensorStateDomainSpec, CURRENT_INFERENCE_STATE_ABI,
};

pub(crate) fn upgrade_kv_contract_v1(contract: &KvCacheContract) -> Result<InferenceStateContract> {
    contract.validate()?;
    let mut domains = Vec::with_capacity(contract.domains.len());
    let mut groups = Vec::with_capacity(contract.domains.len());
    for source in &contract.domains {
        let domain_id = StateDomainId::new(
            source
                .id()
                .get()
                .checked_add(1)
                .ok_or_else(|| invalid("v1 cache domain id cannot be represented in v2"))?,
        );
        let (domain, prefix_shareable) = match source {
            KvDomainSpec::PagedAttention(spec) => {
                let prefix = paged_prefix(&spec.prefix_semantics);
                let prefix_shareable = !matches!(prefix, PrefixPolicy::Disabled);
                let layers = spec
                    .layers
                    .iter()
                    .map(|layer| {
                        Ok(PagedAttentionLayerSpec {
                            model_layer: layer.model_layer,
                            query_heads: layer.num_query_heads,
                            kv_heads: layer.num_kv_heads,
                            key_head_dim: layer.key_head_dim,
                            value_head_dim: layer.value_head_dim,
                            pattern: match layer.attention {
                                V1Attention::Full => AttentionPattern::Full,
                                V1Attention::SlidingWindow { window_tokens } => {
                                    AttentionPattern::SlidingWindow { window_tokens }
                                }
                            },
                            mask: AttentionMask::Causal,
                            key_encoding: match layer.key_encoding {
                                V1KeyEncoding::Raw => KeyEncoding::Raw,
                                V1KeyEncoding::Rotary { rotary_dim } => {
                                    KeyEncoding::Rotary { rotary_dim }
                                }
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                (
                    StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
                        header: StateDomainHeader {
                            id: domain_id,
                            scope: StateScope::Retained,
                            clock: clock(&spec.token_axis),
                            placement: PlacementPolicy::BackendLocalWithHostOffload,
                            prefix,
                            checkpoint: if prefix_shareable {
                                CheckpointPolicy::CopyOnWrite
                            } else {
                                CheckpointPolicy::Transactional
                            },
                        },
                        layers,
                        page_size: PageSizeConstraint {
                            min_tokens: spec.page_tokens.min,
                            preferred_tokens: spec.page_tokens.preferred,
                            max_tokens: spec.page_tokens.max,
                            multiple_of: spec.page_tokens.multiple_of,
                        },
                        accepted_dtypes: spec.storage.dtypes.iter().copied().map(dtype).collect(),
                    }),
                    prefix_shareable,
                )
            }
            KvDomainSpec::ModelState(spec) => {
                let prefix_shareable = matches!(
                    spec.prefix_semantics,
                    KvPrefixSemantics::CommittedFullPages { .. }
                );
                let components = spec
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(index, layer)| {
                        Ok(TensorComponentSpec {
                            id: StateComponentId::new(u32::try_from(index + 1).map_err(|_| {
                                invalid("v1 model-state component count exceeds u32")
                            })?),
                            role: match &layer.kind {
                                ModelStateKind::Recurrent => TensorRole::RecurrentHidden,
                                ModelStateKind::Convolution => TensorRole::ConvolutionState,
                                ModelStateKind::CrossAttention => TensorRole::EncoderMemory,
                                ModelStateKind::Custom(name) => TensorRole::Custom(name.clone()),
                            },
                            shape: BoundedShape {
                                dimensions: vec![ShapeDimension {
                                    axis: ShapeAxis::Hidden,
                                    extent: ShapeExtent::Fixed {
                                        value: layer.elements_per_sequence,
                                    },
                                }],
                            },
                            accepted_dtypes: spec
                                .storage
                                .dtypes
                                .iter()
                                .copied()
                                .map(dtype)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                (
                    StateDomainSpec::Tensor(TensorStateDomainSpec {
                        header: StateDomainHeader {
                            id: domain_id,
                            scope: StateScope::Retained,
                            clock: clock(&spec.token_axis),
                            placement: PlacementPolicy::BackendLocalWithHostOffload,
                            prefix: if prefix_shareable {
                                PrefixPolicy::CommittedSnapshots { interval_steps: 1 }
                            } else {
                                PrefixPolicy::Disabled
                            },
                            checkpoint: CheckpointPolicy::CopyOnWrite,
                        },
                        components,
                    }),
                    prefix_shareable,
                )
            }
        };
        domains.push(domain);
        groups.push(StateGroupSpec {
            id: StateGroupId::new(domain_id.get()),
            domains: vec![domain_id],
            prefix_shareable,
        });
    }
    domains.sort_unstable_by_key(StateDomainSpec::id);
    groups.sort_unstable_by_key(|group| group.id);
    let upgraded = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains,
        groups,
    };
    upgraded.validate()?;
    Ok(upgraded)
}

fn dtype(dtype: KvStorageDType) -> StateDType {
    match dtype {
        KvStorageDType::F32 => StateDType::F32,
        KvStorageDType::F16 => StateDType::F16,
        KvStorageDType::Bf16 => StateDType::Bf16,
        KvStorageDType::I8 => StateDType::I8,
        KvStorageDType::Q4 => StateDType::Q4,
    }
}

fn clock(axis: &CacheTokenAxis) -> StateClock {
    match axis {
        CacheTokenAxis::DecoderTokens => StateClock::DecoderTokens,
        CacheTokenAxis::EncoderTokens | CacheTokenAxis::CrossAttentionMemory => {
            StateClock::EncoderTokens
        }
        CacheTokenAxis::Custom(name) => StateClock::Custom(name.clone()),
    }
}

fn paged_prefix(prefix: &KvPrefixSemantics) -> PrefixPolicy {
    match prefix {
        KvPrefixSemantics::Disabled => PrefixPolicy::Disabled,
        KvPrefixSemantics::CommittedFullPages { positions } => PrefixPolicy::CommittedPages {
            positions: match positions {
                V1Position::Absolute => PositionSemantics::Absolute,
                V1Position::WindowRelative => PositionSemantics::WindowRelative,
                V1Position::ModelDefined(name) => PositionSemantics::ModelDefined(name.clone()),
            },
        },
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upgrades_zero_based_paged_contract_to_complete_v2_semantics() {
        let upgraded = upgrade_kv_contract_v1(&crate::kv::test_contract()).unwrap();
        assert_eq!(upgraded.domains[0].id(), StateDomainId::new(2));
        assert_eq!(upgraded.groups[0].id, StateGroupId::new(2));
        let StateDomainSpec::PagedAttention(paged) = &upgraded.domains[0] else {
            panic!("expected paged attention")
        };
        assert_eq!(paged.layers[0].query_heads, 16);
        assert_eq!(paged.layers[0].kv_heads, 4);
        assert_eq!(
            paged.accepted_dtypes,
            vec![StateDType::F16, StateDType::Bf16]
        );
        assert!(matches!(
            paged.header.prefix,
            PrefixPolicy::CommittedPages { .. }
        ));
    }
}
