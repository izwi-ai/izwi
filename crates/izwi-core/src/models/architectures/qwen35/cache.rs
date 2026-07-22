//! Semantic cache domains for Qwen3.5's hybrid decoder.
//!
//! Full-attention pages, recurrent state, and convolution history advance in
//! one consistency group and are committed by the managed runtime transaction.

use candle_core::DType;

use crate::error::{Error, Result};
use crate::kv::{
    AttentionSemantics, CacheDomainId, CacheTokenAxis, KeyEncoding, KvCacheContract, KvDomainSpec,
    KvPrefixSemantics, KvStorageDType, KvStorageRequest, ModelStateDomainSpec, ModelStateKind,
    ModelStateLayerSpec, PageTokenConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec,
    CURRENT_KV_CONTRACT_ABI,
};

use super::chat::Qwen35TextConfig;

pub(crate) const FULL_ATTENTION_DOMAIN: CacheDomainId = CacheDomainId::new(0);
pub(crate) const RECURRENT_STATE_DOMAIN: CacheDomainId = CacheDomainId::new(1);
pub(crate) const CONVOLUTION_STATE_DOMAIN: CacheDomainId = CacheDomainId::new(2);

pub(crate) fn qwen35_composite_cache_contract(
    config: &Qwen35TextConfig,
    attention_dtype: DType,
    preferred_page_tokens: usize,
) -> Result<KvCacheContract> {
    if config.full_attention_interval == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 composite cache requires a non-zero full-attention interval".into(),
        ));
    }
    if config.ssm_time_step_rank == 0
        || !config
            .ssm_inner_size
            .is_multiple_of(config.ssm_time_step_rank)
    {
        return Err(Error::InvalidInput(
            "Qwen3.5 composite cache has invalid recurrent head geometry".into(),
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
        .ok_or_else(|| Error::InvalidInput("Qwen3.5 convolution width overflow".into()))?;
    let convolution_elements = convolution_width
        .checked_mul(config.ssm_conv_kernel.saturating_sub(1))
        .ok_or_else(|| Error::InvalidInput("Qwen3.5 convolution state overflow".into()))?;

    for layer in 0..config.block_count {
        let model_layer = as_u32(layer, "model layer")?;
        if (layer + 1).is_multiple_of(config.full_attention_interval) {
            attention_layers.push(PagedAttentionLayerSpec {
                model_layer,
                num_query_heads: query_heads,
                num_kv_heads: kv_heads,
                key_head_dim,
                value_head_dim,
                attention: AttentionSemantics::Full,
                key_encoding: KeyEncoding::Rotary { rotary_dim },
            });
        } else {
            recurrent_layers.push(ModelStateLayerSpec {
                model_layer,
                kind: ModelStateKind::Recurrent,
                elements_per_sequence: recurrent_elements,
            });
            convolution_layers.push(ModelStateLayerSpec {
                model_layer,
                kind: ModelStateKind::Convolution,
                elements_per_sequence: u64::try_from(convolution_elements).map_err(|_| {
                    Error::InvalidInput("Qwen3.5 convolution state exceeds u64".into())
                })?,
            });
        }
    }
    if attention_layers.is_empty() || recurrent_layers.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.5 composite cache requires both full-attention and linear layers".into(),
        ));
    }

    let storage = KvStorageRequest {
        dtypes: vec![dtype],
        allow_quantized: false,
    };
    let token_axis = CacheTokenAxis::DecoderTokens;
    let contract = KvCacheContract {
        abi: CURRENT_KV_CONTRACT_ABI,
        domains: vec![
            KvDomainSpec::PagedAttention(PagedAttentionDomainSpec {
                id: FULL_ATTENTION_DOMAIN,
                token_axis: token_axis.clone(),
                layers: attention_layers,
                page_tokens: PageTokenConstraint {
                    min: 1,
                    preferred,
                    max: preferred.max(256),
                    multiple_of: 1,
                },
                storage: storage.clone(),
                // Attention-only reuse is invalid while recurrent and conv
                // state remain part of the same token transition.
                prefix_semantics: KvPrefixSemantics::Disabled,
            }),
            KvDomainSpec::ModelState(ModelStateDomainSpec {
                id: RECURRENT_STATE_DOMAIN,
                token_axis: token_axis.clone(),
                layers: recurrent_layers,
                storage: KvStorageRequest {
                    dtypes: vec![KvStorageDType::F32],
                    allow_quantized: false,
                },
                prefix_semantics: KvPrefixSemantics::Disabled,
            }),
            KvDomainSpec::ModelState(ModelStateDomainSpec {
                id: CONVOLUTION_STATE_DOMAIN,
                token_axis,
                layers: convolution_layers,
                storage: KvStorageRequest {
                    dtypes: vec![KvStorageDType::F32],
                    allow_quantized: false,
                },
                prefix_semantics: KvPrefixSemantics::Disabled,
            }),
        ],
    };
    contract.validate()?;
    Ok(contract)
}

fn storage_dtype(dtype: DType) -> Result<KvStorageDType> {
    match dtype {
        DType::F32 => Ok(KvStorageDType::F32),
        DType::F16 => Ok(KvStorageDType::F16),
        DType::BF16 => Ok(KvStorageDType::Bf16),
        dtype => Err(Error::InvalidInput(format!(
            "Qwen3.5 managed attention does not support {dtype:?} storage"
        ))),
    }
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidInput(format!("Qwen3.5 {label} exceeds u32")))
}

fn checked_product(values: &[usize]) -> Result<u64> {
    let product = values.iter().try_fold(1usize, |product, value| {
        product.checked_mul(*value).ok_or(())
    });
    u64::try_from(
        product.map_err(|_| Error::InvalidInput("Qwen3.5 recurrent state size overflow".into()))?,
    )
    .map_err(|_| Error::InvalidInput("Qwen3.5 recurrent state exceeds u64".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Qwen35TextConfig {
        Qwen35TextConfig {
            architecture: "qwen35".into(),
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
        let contract = qwen35_composite_cache_contract(&config(), DType::F16, 32).unwrap();
        assert_eq!(contract.domains.len(), 3);

        let KvDomainSpec::PagedAttention(attention) = &contract.domains[0] else {
            panic!("expected attention domain");
        };
        assert_eq!(attention.id, FULL_ATTENTION_DOMAIN);
        assert_eq!(
            attention
                .layers
                .iter()
                .map(|layer| layer.model_layer)
                .collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert_eq!(attention.prefix_semantics, KvPrefixSemantics::Disabled);

        let KvDomainSpec::ModelState(recurrent) = &contract.domains[1] else {
            panic!("expected recurrent domain");
        };
        let KvDomainSpec::ModelState(convolution) = &contract.domains[2] else {
            panic!("expected convolution domain");
        };
        assert_eq!(recurrent.layers.len(), 6);
        assert_eq!(convolution.layers.len(), 6);
        assert!(recurrent
            .layers
            .iter()
            .all(|layer| layer.elements_per_sequence == 65_536));
        assert!(convolution
            .layers
            .iter()
            .all(|layer| layer.elements_per_sequence == 4_608));
        assert_eq!(recurrent.prefix_semantics, KvPrefixSemantics::Disabled);
        assert_eq!(convolution.prefix_semantics, KvPrefixSemantics::Disabled);
    }

    #[test]
    fn composite_contract_allocates_paged_and_transactional_tensor_arenas() {
        use crate::backends::BackendKind;
        use crate::engine::{ManagedKvCacheManager, ModelInstanceId};
        use crate::kv::CacheCapability;

        let contract = qwen35_composite_cache_contract(&config(), DType::F16, 32).unwrap();
        let upgraded = crate::kv::v2::upgrade_kv_contract_v1(&contract).unwrap();
        assert_eq!(upgraded.groups.len(), 1);
        assert_eq!(
            upgraded.groups[0].domains,
            vec![
                crate::kv::v2::StateDomainId::new(1),
                crate::kv::v2::StateDomainId::new(2),
                crate::kv::v2::StateDomainId::new(3),
            ]
        );
        assert!(!upgraded.groups[0].prefix_shareable);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                ModelInstanceId::new(71),
                BackendKind::Cpu,
                2,
                32,
                &CacheCapability::Managed(contract),
            )
            .unwrap()
            .unwrap();

        assert_eq!(runtime.state_plan_v2().paged_attention.len(), 1);
        assert_eq!(runtime.state_plan_v2().non_paged.len(), 2);
        assert!(runtime.tensor_state().is_some());
    }
}
