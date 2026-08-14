//! Semantic cache domains for Qwen3.8's hybrid decoder.
//!
//! Full-attention pages, recurrent state, and convolution history advance in
//! one consistency group and are committed by the managed runtime transaction.

use candle_core::DType;

use crate::error::{Error, Result};
use crate::kv::v2::{
    AttentionMask, AttentionPattern, BoundedShape, CheckpointPolicy, InferenceStateContract,
    KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec,
    PlacementPolicy, PrefixPolicy, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
    CURRENT_INFERENCE_STATE_ABI,
};

use super::chat::Qwen38TextConfig;

pub(crate) const FULL_ATTENTION_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const RECURRENT_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);
pub(crate) const CONVOLUTION_STATE_DOMAIN: StateDomainId = StateDomainId::new(3);

pub(crate) fn qwen38_composite_cache_contract(
    config: &Qwen38TextConfig,
    attention_dtype: DType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    if config.full_attention_interval == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.8 composite cache requires a non-zero full-attention interval".into(),
        ));
    }
    if config.ssm_time_step_rank == 0
        || !config
            .ssm_inner_size
            .is_multiple_of(config.ssm_time_step_rank)
    {
        return Err(Error::InvalidInput(
            "Qwen3.8 composite cache has invalid recurrent head geometry".into(),
        ));
    }

    let dtype = storage_dtype(attention_dtype)?;
    let query_heads = as_u32(config.attention_head_count, "query head count")?;
    let kv_heads = as_u32(config.attention_head_count_kv, "KV head count")?;
    let key_head_dim = as_u32(config.attention_key_length, "key head dimension")?;
    let value_head_dim = as_u32(config.attention_value_length, "value head dimension")?;
    let rotary_dim = as_u32(config.rope_dimension_count, "rotary dimension")?;
    let preferred = as_u32(preferred_page_tokens.max(1), "page size")?;

    let mut attention_layers = Vec::new();
    let mut recurrent_layers = Vec::new();
    let mut convolution_layers = Vec::new();
    let recurrent_elements = checked_product(&[
        config.ssm_time_step_rank,
        config.ssm_state_size,
        config.ssm_inner_size / config.ssm_time_step_rank,
    ])?;
    let convolution_width = config
        .ssm_state_size
        .checked_mul(config.ssm_group_count)
        .and_then(|keys| keys.checked_mul(2))
        .and_then(|keys| keys.checked_add(config.ssm_inner_size))
        .ok_or_else(|| Error::InvalidInput("Qwen3.8 convolution width overflow".into()))?;
    let convolution_elements = convolution_width
        .checked_mul(config.ssm_conv_kernel.saturating_sub(1))
        .ok_or_else(|| Error::InvalidInput("Qwen3.8 convolution state overflow".into()))?;

    for layer in 0..config.block_count {
        let model_layer = as_u32(layer, "model layer")?;
        if (layer + 1).is_multiple_of(config.full_attention_interval) {
            attention_layers.push(PagedAttentionLayerSpec {
                model_layer,
                query_heads,
                kv_heads,
                key_head_dim,
                value_head_dim,
                pattern: AttentionPattern::Full,
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Rotary { rotary_dim },
                attention_logit_softcap: None,
            });
        } else {
            recurrent_layers.push((model_layer, recurrent_elements));
            convolution_layers.push((
                model_layer,
                u64::try_from(convolution_elements).map_err(|_| {
                    Error::InvalidInput("Qwen3.8 convolution state exceeds u64".into())
                })?,
            ));
        }
    }
    if attention_layers.is_empty() || recurrent_layers.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.8 composite cache requires both full-attention and linear layers".into(),
        ));
    }

    let retained_header = |id| StateDomainHeader {
        id,
        scope: StateScope::Retained,
        clock: StateClock::DecoderTokens,
        placement: PlacementPolicy::BackendLocalWithHostOffload,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::Transactional,
    };
    let tensor_components = |layers: &[(u32, u64)], role: TensorRole| {
        layers
            .iter()
            .enumerate()
            .map(|(index, (_, elements))| {
                Ok(TensorComponentSpec {
                    id: StateComponentId::new(u32::try_from(index + 1).map_err(|_| {
                        Error::InvalidInput("Qwen3.8 state component count exceeds u32".into())
                    })?),
                    role: role.clone(),
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed { value: *elements },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                })
            })
            .collect::<Result<Vec<_>>>()
    };
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![
            StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
                header: retained_header(FULL_ATTENTION_DOMAIN),
                layers: attention_layers,
                page_size: PageSizeConstraint {
                    min_tokens: 1,
                    preferred_tokens: preferred,
                    max_tokens: preferred.max(256),
                    multiple_of: 1,
                },
                accepted_dtypes: vec![dtype],
            }),
            StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: retained_header(RECURRENT_STATE_DOMAIN),
                components: tensor_components(&recurrent_layers, TensorRole::RecurrentHidden)?,
            }),
            StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: retained_header(CONVOLUTION_STATE_DOMAIN),
                components: tensor_components(&convolution_layers, TensorRole::ConvolutionState)?,
            }),
        ],
        groups: vec![StateGroupSpec {
            id: StateGroupId::new(1),
            domains: vec![
                FULL_ATTENTION_DOMAIN,
                RECURRENT_STATE_DOMAIN,
                CONVOLUTION_STATE_DOMAIN,
            ],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

fn storage_dtype(dtype: DType) -> Result<StateDType> {
    match dtype {
        DType::F32 => Ok(StateDType::F32),
        DType::F16 => Ok(StateDType::F16),
        DType::BF16 => Ok(StateDType::Bf16),
        dtype => Err(Error::InvalidInput(format!(
            "Qwen3.8 managed attention does not support {dtype:?} storage"
        ))),
    }
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidInput(format!("Qwen3.8 {label} exceeds u32")))
}

fn checked_product(values: &[usize]) -> Result<u64> {
    let product = values.iter().try_fold(1usize, |product, value| {
        product.checked_mul(*value).ok_or(())
    });
    u64::try_from(
        product.map_err(|_| Error::InvalidInput("Qwen3.8 recurrent state size overflow".into()))?,
    )
    .map_err(|_| Error::InvalidInput("Qwen3.8 recurrent state exceeds u64".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Qwen38TextConfig {
        Qwen38TextConfig {
            architecture: "qwen38".into(),
            block_count: 8,
            context_length: 4_096,
            embedding_length: 1_024,
            feed_forward_length: 3_072,
            attention_head_count: 16,
            attention_head_count_kv: 4,
            attention_key_length: 64,
            attention_value_length: 64,
            rope_dimension_sections: vec![8, 12, 12],
            rope_dimension_count: 64,
            rope_freq_base: 10_000.0,
            attention_layer_norm_rms_epsilon: 1e-6,
            ssm_conv_kernel: 4,
            ssm_state_size: 64,
            ssm_group_count: 4,
            ssm_time_step_rank: 8,
            ssm_inner_size: 1_024,
            full_attention_interval: 4,
        }
    }

    #[test]
    fn composite_contract_covers_every_hybrid_layer_without_prefix_claims() {
        let contract = qwen38_composite_cache_contract(&config(), DType::F16, 32).unwrap();
        assert_eq!(contract.domains.len(), 3);

        let StateDomainSpec::PagedAttention(attention) = &contract.domains[0] else {
            panic!("expected attention domain");
        };
        assert_eq!(attention.header.id, FULL_ATTENTION_DOMAIN);
        assert_eq!(
            attention
                .layers
                .iter()
                .map(|layer| layer.model_layer)
                .collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert_eq!(attention.header.prefix, PrefixPolicy::Disabled);

        let StateDomainSpec::Tensor(recurrent) = &contract.domains[1] else {
            panic!("expected recurrent domain");
        };
        let StateDomainSpec::Tensor(convolution) = &contract.domains[2] else {
            panic!("expected convolution domain");
        };
        assert_eq!(recurrent.components.len(), 6);
        assert_eq!(convolution.components.len(), 6);
        assert!(recurrent.components.iter().all(|component| component
            .shape
            .maximum_elements()
            .unwrap()
            == 65_536));
        assert!(convolution.components.iter().all(|component| component
            .shape
            .maximum_elements()
            .unwrap()
            == 4_608));
        assert_eq!(recurrent.header.prefix, PrefixPolicy::Disabled);
        assert_eq!(convolution.header.prefix, PrefixPolicy::Disabled);
    }

    #[test]
    fn composite_contract_allocates_paged_and_transactional_tensor_arenas() {
        use crate::backends::BackendKind;
        use crate::engine::{ManagedKvCacheManager, ModelInstanceId};
        use crate::kv::InferenceStateCapability;

        let contract = qwen38_composite_cache_contract(&config(), DType::F16, 32).unwrap();
        assert_eq!(contract.groups.len(), 1);
        assert_eq!(
            contract.groups[0].domains,
            vec![
                crate::kv::v2::StateDomainId::new(1),
                crate::kv::v2::StateDomainId::new(2),
                crate::kv::v2::StateDomainId::new(3),
            ]
        );
        assert!(!contract.groups[0].prefix_shareable);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                ModelInstanceId::new(71),
                BackendKind::Cpu,
                2,
                32,
                &InferenceStateCapability::Managed(contract),
            )
            .unwrap()
            .unwrap();

        assert_eq!(runtime.state_plan_v2().paged_attention.len(), 1);
        assert_eq!(runtime.state_plan_v2().non_paged.len(), 2);
        assert!(runtime.tensor_state().is_some());
    }
}
