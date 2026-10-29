//! A row's decoder state and its state contract.
//!
//! One consistency group with three domains:
//!
//! | Domain | Kind | Shape |
//! |---|---|---|
//! | 1 | paged attention, raw keys, 6 layers | KV heads × head_dim per token |
//! | 2 | tensor, recurrent hidden | one `[heads, head_dim, state]` F32 per Mamba layer |
//! | 3 | tensor, convolution state | one `[K - 1, conv_channels]` F32 per Mamba layer |
//!
//! The attention layers come from `layers_block_type`, never from an
//! interval. Prefixes are not shared (prompts share about 15 tokens before
//! the audio, and vLLM serves without prefix caching too). Every domain is
//! transactional: a quantum stages the tensor domains and the managed
//! runtime commits them with the KV pages, or rolls all three back.

use candle_core::{DType, Device, Tensor};

use super::super::config::{NemotronDecoderConfig, NemotronLayerKind};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    AttentionMask, AttentionPattern, BoundedShape, CheckpointPolicy, InferenceStateContract,
    KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec,
    PlacementPolicy, PrefixPolicy, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
    CURRENT_INFERENCE_STATE_ABI,
};

pub const ATTENTION_DOMAIN: StateDomainId = StateDomainId::new(1);
pub const RECURRENT_DOMAIN: StateDomainId = StateDomainId::new(2);
pub const CONVOLUTION_DOMAIN: StateDomainId = StateDomainId::new(3);
pub const STATE_GROUP: StateGroupId = StateGroupId::new(1);

fn as_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::InvalidInput(format!("Nemotron 3.5 Transcribe {what} exceeds u32")))
}

/// The decoder's retained-state contract with KV stored as `kv_dtype`.
pub(crate) fn decoder_state_contract(
    config: &NemotronDecoderConfig,
    kv_dtype: StateDType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    let a = config.attention;
    let m = config.mamba;
    let preferred = as_u32(preferred_page_tokens.max(1), "page size")?;
    let layers = config
        .layers_of(NemotronLayerKind::Attention)
        .into_iter()
        .map(|layer| {
            Ok(PagedAttentionLayerSpec {
                model_layer: as_u32(layer, "model layer")?,
                query_heads: as_u32(a.heads, "query heads")?,
                kv_heads: as_u32(a.kv_heads, "KV heads")?,
                key_head_dim: as_u32(a.head_dim, "head dim")?,
                value_head_dim: as_u32(a.head_dim, "head dim")?,
                pattern: AttentionPattern::Full,
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Raw,
                attention_logit_softcap: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let header = |id| StateDomainHeader {
        id,
        scope: StateScope::Retained,
        clock: StateClock::DecoderTokens,
        placement: PlacementPolicy::BackendLocalWithHostOffload,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::Transactional,
    };
    let mamba_layers = config.layers_of(NemotronLayerKind::Mamba);
    let components = |elements: usize, role: TensorRole| -> Result<Vec<TensorComponentSpec>> {
        (0..mamba_layers.len())
            .map(|index| {
                Ok(TensorComponentSpec {
                    id: StateComponentId::new(as_u32(index + 1, "component")?),
                    role: role.clone(),
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed {
                                value: elements as u64,
                            },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                })
            })
            .collect()
    };
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![
            StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
                header: header(ATTENTION_DOMAIN),
                layers,
                page_size: PageSizeConstraint {
                    min_tokens: 1,
                    preferred_tokens: preferred,
                    max_tokens: preferred.max(256),
                    multiple_of: 1,
                },
                accepted_dtypes: vec![kv_dtype],
            }),
            StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: header(RECURRENT_DOMAIN),
                components: components(
                    m.heads * m.head_dim * m.state,
                    TensorRole::RecurrentHidden,
                )?,
            }),
            StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: header(CONVOLUTION_DOMAIN),
                components: components(
                    (m.conv_kernel - 1) * m.conv_channels(),
                    TensorRole::ConvolutionState,
                )?,
            }),
        ],
        groups: vec![StateGroupSpec {
            id: STATE_GROUP,
            domains: vec![ATTENTION_DOMAIN, RECURRENT_DOMAIN, CONVOLUTION_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

/// Bytes of one row's Mamba state (both tensor domains), whatever its
/// length.
pub fn recurrent_state_bytes(config: &NemotronDecoderConfig) -> u64 {
    let m = config.mamba;
    let per_layer = m.heads * m.head_dim * m.state + (m.conv_kernel - 1) * m.conv_channels();
    (per_layer * config.layers_of(NemotronLayerKind::Mamba).len() * 4) as u64
}

/// KV bytes per token across the attention layers in `kv_dtype`.
pub fn kv_bytes_per_token(config: &NemotronDecoderConfig, kv_dtype: DType) -> u64 {
    let a = config.attention;
    (config.layers_of(NemotronLayerKind::Attention).len()
        * 2
        * a.kv_heads
        * a.head_dim
        * kv_dtype.size_in_bytes()) as u64
}

/// One row's Mamba state: per Mamba layer, the convolution history
/// `[1, K - 1, C]` and the SSM state `[1, heads, head_dim, state]`, F32.
/// Tensors are never written in place, so a clone is a checkpoint.
#[derive(Debug, Clone)]
pub struct NemotronMambaState {
    pub(crate) conv: Vec<Tensor>,
    pub(crate) ssm: Vec<Tensor>,
    /// Tokens the state has consumed.
    pub(crate) cursor: u64,
}

impl NemotronMambaState {
    pub fn zeros(config: &NemotronDecoderConfig, device: &Device) -> Result<Self> {
        let m = config.mamba;
        let layers = config.layers_of(NemotronLayerKind::Mamba).len();
        let conv = Tensor::zeros(
            (1, m.conv_kernel - 1, m.conv_channels()),
            DType::F32,
            device,
        )?;
        let ssm = Tensor::zeros((1, m.heads, m.head_dim, m.state), DType::F32, device)?;
        Ok(Self {
            conv: vec![conv; layers],
            ssm: vec![ssm; layers],
            cursor: 0,
        })
    }

    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Replaces the state from the arena's committed snapshot of
    /// `sequence` (a fresh sequence keeps zeros).
    pub(crate) fn restore(
        &mut self,
        arena: &TensorStateArena,
        sequence: PhysicalStateSequenceId,
        config: &NemotronDecoderConfig,
    ) -> Result<()> {
        let m = config.mamba;
        let recurrent = arena.read(sequence, RECURRENT_DOMAIN)?;
        let convolution = arena.read(sequence, CONVOLUTION_DOMAIN)?;
        match (recurrent, convolution) {
            (None, None) => {
                if self.cursor != 0 {
                    return Err(Error::InferenceError(
                        "Nemotron 3.5 Transcribe Mamba state disappeared".into(),
                    ));
                }
                Ok(())
            }
            (Some(recurrent), Some(convolution)) => {
                if recurrent.cursor != convolution.cursor
                    || recurrent.components.len() != self.ssm.len()
                    || convolution.components.len() != self.conv.len()
                {
                    return Err(Error::InferenceError(
                        "Nemotron 3.5 Transcribe Mamba state domains diverged".into(),
                    ));
                }
                let take =
                    |values: &[StateComponentValue], dims: &[usize]| -> Result<Vec<Tensor>> {
                        values
                            .iter()
                            .map(|value| {
                                let tensor = value.tensor.as_ref().ok_or_else(|| {
                                    Error::InferenceError("Mamba state component is absent".into())
                                })?;
                                Ok(tensor.reshape(dims)?)
                            })
                            .collect()
                    };
                self.ssm = take(&recurrent.components, &[1, m.heads, m.head_dim, m.state])?;
                self.conv = take(
                    &convolution.components,
                    &[1, m.conv_kernel - 1, m.conv_channels()],
                )?;
                self.cursor = recurrent.cursor;
                Ok(())
            }
            _ => Err(Error::InferenceError(
                "Nemotron 3.5 Transcribe Mamba state is half committed".into(),
            )),
        }
    }

    /// Stages both tensor domains into `transaction` at the state's cursor.
    pub(crate) fn stage(
        &self,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
    ) -> Result<()> {
        for (domain, tensors) in [
            (RECURRENT_DOMAIN, &self.ssm),
            (CONVOLUTION_DOMAIN, &self.conv),
        ] {
            let expected = arena
                .read_transaction_base(transaction, domain)?
                .map(|snapshot| snapshot.cursor)
                .unwrap_or(0);
            let values = tensors
                .iter()
                .enumerate()
                .map(|(index, tensor)| {
                    Ok(StateComponentValue {
                        component: StateComponentId::new(as_u32(index + 1, "component")?),
                        tensor: Some(tensor.flatten_all()?.contiguous()?),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            arena.stage_replace(transaction, domain, expected, self.cursor, values)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_contract_has_six_raw_attention_layers_and_two_tensor_domains() {
        let config = super::super::super::config::tests::published();
        let contract = decoder_state_contract(&config.decoder, StateDType::F16, 16).unwrap();
        let StateDomainSpec::PagedAttention(attention) = &contract.domains[0] else {
            panic!("attention first");
        };
        assert_eq!(
            attention
                .layers
                .iter()
                .map(|l| l.model_layer)
                .collect::<Vec<_>>(),
            vec![5, 12, 19, 26, 33, 42]
        );
        assert!(attention
            .layers
            .iter()
            .all(|l| l.key_encoding == KeyEncoding::Raw && l.kv_heads == 2 && l.query_heads == 32));
        let StateDomainSpec::Tensor(recurrent) = &contract.domains[1] else {
            panic!("recurrent second");
        };
        assert_eq!(recurrent.components.len(), 23);
        // 47.6 MiB of Mamba state per row; 6 KiB of 16-bit KV per token.
        assert_eq!(recurrent_state_bytes(&config.decoder), 49_930_240);
        assert_eq!(kv_bytes_per_token(&config.decoder, DType::F16), 6 * 1024);
        assert!(contract
            .domains
            .iter()
            .all(|d| d.header().prefix == PrefixPolicy::Disabled));
    }
}
