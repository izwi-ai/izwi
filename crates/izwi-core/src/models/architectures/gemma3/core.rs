//! Gemma 3 decoder backed exclusively by scheduler-owned physical KV pages.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear_b as linear, Embedding, Linear, VarBuilder};
use candle_transformers::models::gemma3::Config;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    AttentionLogitSoftcap, AttentionMask, AttentionPattern, CheckpointPolicy,
    InferenceStateContract, KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec,
    PagedAttentionLayerSpec, PlacementPolicy, PositionSemantics, PrefixPolicy, StateClock,
    StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec,
    StateScope, CURRENT_INFERENCE_STATE_ABI,
};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

#[derive(Debug, Clone)]
struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
    cuda_candle_kernel: bool,
}

impl GemmaRmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let cuda_candle_kernel = gemma_cuda_rms_norm_enabled(vb.device());
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
            cuda_candle_kernel,
        })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let adjusted_weight = (&self.weight + 1.0)?;
        if self.cuda_candle_kernel {
            return candle_nn::ops::rms_norm(input, &adjusted_weight, self.eps as f32);
        }
        let input_dtype = input.dtype();
        let internal_dtype = match input_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            dtype => dtype,
        };
        let hidden_size = input.dim(D::Minus1)?;
        let input = input.to_dtype(internal_dtype)?;
        let variance = (input.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        input
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .to_dtype(input_dtype)?
            .broadcast_mul(&adjusted_weight)
    }
}

fn gemma_cuda_rms_norm_enabled(device: &Device) -> bool {
    let override_enabled =
        std::env::var("IZWI_GEMMA3_CUDA_RMS_NORM")
            .ok()
            .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            });
    gemma_cuda_rms_norm_policy(device.is_cuda(), override_enabled)
}

fn gemma_cuda_rms_norm_policy(is_cuda: bool, override_enabled: Option<bool>) -> bool {
    is_cuda && override_enabled.unwrap_or(true)
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn load(
        dtype: DType,
        config: &Config,
        device: &Device,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let rope_frequency = if sliding_window.is_some() {
            config.rope_local_base_freq
        } else {
            config.rope_theta
        };
        let inverse_frequency = (0..config.head_dim)
            .step_by(2)
            .map(|index| 1.0f32 / rope_frequency.powf(index as f64 / config.head_dim as f64) as f32)
            .collect::<Vec<_>>();
        let inverse_frequency =
            Tensor::from_vec(inverse_frequency, (1, config.head_dim / 2), device)?
                .to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(dtype)?
            .reshape((config.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inverse_frequency)?;
        Ok(Self {
            sin: frequencies.sin()?,
            cos: frequencies.cos()?,
        })
    }

    fn apply(&self, query: &Tensor, key: &Tensor, position: usize) -> Result<(Tensor, Tensor)> {
        let sequence_len = query.dim(2)?;
        let cos = self.cos.narrow(0, position, sequence_len)?;
        let sin = self.sin.narrow(0, position, sequence_len)?;
        Ok((
            candle_nn::rotary_emb::rope(&query.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&key.contiguous()?, &cos, &sin)?,
        ))
    }
}

#[derive(Debug, Clone)]
struct GemmaMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    activation: candle_nn::Activation,
}

impl GemmaMlp {
    fn load(config: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear(
                config.hidden_size,
                config.intermediate_size,
                false,
                vb.pp("gate_proj"),
            )?,
            up: linear(
                config.hidden_size,
                config.intermediate_size,
                false,
                vb.pp("up_proj"),
            )?,
            down: linear(
                config.intermediate_size,
                config.hidden_size,
                false,
                vb.pp("down_proj"),
            )?,
            activation: config.hidden_activation,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = input.apply(&self.gate)?.apply(&self.activation)?;
        let up = input.apply(&self.up)?;
        (gate * up)?.apply(&self.down).map_err(Error::from)
    }
}

#[derive(Debug, Clone)]
struct GemmaAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_norm: GemmaRmsNorm,
    key_norm: GemmaRmsNorm,
    rotary: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
}

impl GemmaAttention {
    fn load(config: &Config, vb: VarBuilder, sliding_window: Option<usize>) -> Result<Self> {
        let bias = config.attention_bias;
        Ok(Self {
            query: linear(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                bias,
                vb.pp("q_proj"),
            )?,
            key: linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                bias,
                vb.pp("k_proj"),
            )?,
            value: linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                bias,
                vb.pp("v_proj"),
            )?,
            output: linear(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                bias,
                vb.pp("o_proj"),
            )?,
            query_norm: GemmaRmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            key_norm: GemmaRmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            rotary: Arc::new(RotaryEmbedding::load(
                vb.dtype(),
                config,
                vb.device(),
                sliding_window,
            )?),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            softmax_scale: 1.0 / (config.query_pre_attn_scalar as f32).sqrt(),
            softcap: config.attn_logit_softcapping.map(|value| value as f32),
            sliding_window,
        })
    }

    fn forward_physical(
        &self,
        input: &Tensor,
        position: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut crate::models::shared::attention::physical::PreparedPhysicalPagedStep,
        layer_index: usize,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = input.dims3()?;
        if batch != 1 || sequence == 0 {
            return Err(Error::InvalidInput(
                "Gemma physical attention requires one non-empty sequence".into(),
            ));
        }
        let query = self
            .query
            .forward(input)?
            .reshape((batch, sequence, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = self
            .key
            .forward(input)?
            .reshape((batch, sequence, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value = self
            .value
            .forward(input)?
            .reshape((sequence, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let query = self.query_norm.forward(&query)?;
        let key = self.key_norm.forward(&key)?;
        let (query, key) = self.rotary.apply(&query, &key, position)?;
        let query = query
            .transpose(1, 2)?
            .reshape((sequence, self.num_heads, self.head_dim))?
            .contiguous()?;
        let key = key
            .transpose(1, 2)?
            .reshape((sequence, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let attended = cache.write_and_attend_with_semantics(
            layer_index,
            prepared,
            &query,
            &key,
            &value,
            self.softmax_scale,
            self.softcap,
            self.sliding_window,
        )?;
        attended
            .reshape((batch, sequence, self.num_heads * self.head_dim))?
            .apply(&self.output)
            .map_err(Error::from)
    }

    fn forward_physical_batch(
        &self,
        input: &Tensor,
        positions: &[usize],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        completions: &mut KvWriteCompletionCollector,
        layer_index: usize,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = input.dims3()?;
        if sequence != 1 || batch == 0 || positions.len() != batch || caches.len() != batch {
            return Err(Error::InvalidInput(
                "Gemma physical decode batch dimensions do not match".into(),
            ));
        }
        let first = caches[0];
        if caches.iter().any(|cache| {
            !Arc::ptr_eq(cache.arena(), first.arena())
                || cache.layer_binding(layer_index).ok() != first.layer_binding(layer_index).ok()
        }) {
            return Err(Error::InvalidInput(
                "Gemma physical decode rows must share one arena and layer binding".into(),
            ));
        }
        let query = self
            .query
            .forward(input)?
            .reshape((batch, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = self
            .key
            .forward(input)?
            .reshape((batch, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value =
            self.value
                .forward(input)?
                .reshape((batch, self.num_kv_heads, self.head_dim))?;
        let query = self.query_norm.forward(&query)?;
        let key = self.key_norm.forward(&key)?;

        let mut query_rows = Vec::with_capacity(batch);
        let mut key_rows = Vec::with_capacity(batch);
        for row in 0..batch {
            let (query, key) = self.rotary.apply(
                &query.narrow(0, row, 1)?,
                &key.narrow(0, row, 1)?,
                positions[row],
            )?;
            query_rows.push(query.reshape((self.num_heads, self.head_dim))?);
            key_rows.push(key.reshape((self.num_kv_heads, self.head_dim))?);
        }
        let query_rows = query_rows.iter().collect::<Vec<_>>();
        let key_rows = key_rows.iter().collect::<Vec<_>>();
        let queries = Tensor::stack(&query_rows, 0)?.contiguous()?;
        let keys = Tensor::stack(&key_rows, 0)?.contiguous()?;
        let values = value.contiguous()?;
        if slots.arena_id() != first.arena().id() || slots.len() != batch {
            return Err(Error::InvalidInput(
                "Gemma physical decode received an incompatible slot map".into(),
            ));
        }
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| match self.sliding_window {
                    Some(window) => cache.sequence_table_with_window(positions[row] + 1, window),
                    None => cache.sequence_table(positions[row] + 1),
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let binding = first.layer_binding(layer_index)?;
        let completion = first.arena().write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let (attended, completion) = submit_ordered_after_write(completion, || {
            first.arena().paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &queries,
                    batch: &metadata,
                    softmax_scale: self.softmax_scale,
                    softcap: self.softcap,
                },
            )
        })?;
        completions.collect(completion)?;
        attended
            .reshape((batch, 1, self.num_heads * self.head_dim))?
            .apply(&self.output)
            .map_err(Error::from)
    }
}

#[derive(Debug, Clone)]
struct GemmaLayer {
    attention: GemmaAttention,
    mlp: GemmaMlp,
    input_norm: GemmaRmsNorm,
    post_attention_norm: GemmaRmsNorm,
    pre_feedforward_norm: GemmaRmsNorm,
    post_feedforward_norm: GemmaRmsNorm,
}

impl GemmaLayer {
    fn load(config: &Config, vb: VarBuilder, sliding_window: Option<usize>) -> Result<Self> {
        Ok(Self {
            attention: GemmaAttention::load(config, vb.pp("self_attn"), sliding_window)?,
            mlp: GemmaMlp::load(config, vb.pp("mlp"))?,
            input_norm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_norm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            pre_feedforward_norm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            post_feedforward_norm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?,
        })
    }

    fn forward_physical(
        &self,
        input: &Tensor,
        position: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut crate::models::shared::attention::physical::PreparedPhysicalPagedStep,
        layer_index: usize,
    ) -> Result<Tensor> {
        let residual = input;
        let attention = self.attention.forward_physical(
            &self.input_norm.forward(input)?,
            position,
            cache,
            prepared,
            layer_index,
        )?;
        let hidden = (residual + attention.apply(&self.post_attention_norm)?)?;
        let residual = &hidden;
        let feedforward = self
            .mlp
            .forward(&hidden.apply(&self.pre_feedforward_norm)?)?;
        (residual + feedforward.apply(&self.post_feedforward_norm)?).map_err(Error::from)
    }

    fn forward_physical_batch(
        &self,
        input: &Tensor,
        positions: &[usize],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        completions: &mut KvWriteCompletionCollector,
        layer_index: usize,
    ) -> Result<Tensor> {
        let residual = input;
        let attention = self.attention.forward_physical_batch(
            &self.input_norm.forward(input)?,
            positions,
            caches,
            slots,
            completions,
            layer_index,
        )?;
        let hidden = (residual + attention.apply(&self.post_attention_norm)?)?;
        let residual = &hidden;
        let feedforward = self
            .mlp
            .forward(&hidden.apply(&self.pre_feedforward_norm)?)?;
        (residual + feedforward.apply(&self.post_feedforward_norm)?).map_err(Error::from)
    }
}

pub(crate) struct Gemma3PhysicalModel {
    embedding: Embedding,
    layers: Vec<GemmaLayer>,
    norm: GemmaRmsNorm,
    lm_head: Linear,
    config: Config,
}

impl Gemma3PhysicalModel {
    pub(crate) fn load(config: Config, vb: VarBuilder) -> Result<Self> {
        if let Some(softcap) = config.attn_logit_softcapping {
            AttentionLogitSoftcap::new(softcap as f32).map_err(|_| {
                Error::ModelLoadError(
                    "Gemma attention-logit softcapping must be finite and positive".into(),
                )
            })?;
        }
        if config.sliding_window == 0 || config.sliding_window_pattern == 0 {
            return Err(Error::ModelLoadError(
                "Gemma physical attention requires non-zero sliding-window geometry".into(),
            ));
        }
        let model = vb.pp("model");
        let embedding = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model.pp("embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let sliding =
                ((index + 1) % config.sliding_window_pattern != 0).then_some(config.sliding_window);
            layers.push(GemmaLayer::load(
                &config,
                model.pp("layers").pp(index),
                sliding,
            )?);
        }
        let norm = GemmaRmsNorm::load(config.hidden_size, config.rms_norm_eps, model.pp("norm"))?;
        let lm_head = Linear::new(embedding.embeddings().clone(), None);
        Ok(Self {
            embedding,
            layers,
            norm,
            lm_head,
            config,
        })
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub(crate) fn managed_inference_state_contract(
        &self,
        domain: StateDomainId,
        storage_dtype: DType,
        preferred_page_tokens: usize,
    ) -> Result<InferenceStateContract> {
        let storage_dtype = match storage_dtype {
            DType::F32 => StateDType::F32,
            DType::F16 => StateDType::F16,
            DType::BF16 => StateDType::Bf16,
            dtype => {
                return Err(Error::InvalidInput(format!(
                    "Gemma physical KV does not support {dtype:?}"
                )))
            }
        };
        let preferred = u32::try_from(preferred_page_tokens.max(1))
            .map_err(|_| Error::InvalidInput("Gemma page size exceeds u32".into()))?;
        let head_dim = u32::try_from(self.config.head_dim)
            .map_err(|_| Error::InvalidInput("Gemma head dimension exceeds u32".into()))?;
        let query_heads = u32::try_from(self.config.num_attention_heads)
            .map_err(|_| Error::InvalidInput("Gemma query head count exceeds u32".into()))?;
        let kv_heads = u32::try_from(self.config.num_key_value_heads)
            .map_err(|_| Error::InvalidInput("Gemma KV head count exceeds u32".into()))?;
        let attention_logit_softcap = self
            .config
            .attn_logit_softcapping
            .map(|softcap| AttentionLogitSoftcap::new(softcap as f32))
            .transpose()?;
        let layers = (0..self.config.num_hidden_layers)
            .map(|index| PagedAttentionLayerSpec {
                model_layer: index as u32,
                query_heads,
                kv_heads,
                key_head_dim: head_dim,
                value_head_dim: head_dim,
                pattern: if (index + 1) % self.config.sliding_window_pattern != 0 {
                    AttentionPattern::SlidingWindow {
                        window_tokens: self.config.sliding_window as u32,
                    }
                } else {
                    AttentionPattern::Full
                },
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Rotary {
                    rotary_dim: head_dim,
                },
                attention_logit_softcap,
            })
            .collect();
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
                header: StateDomainHeader {
                    id: domain,
                    scope: StateScope::Retained,
                    clock: StateClock::DecoderTokens,
                    placement: PlacementPolicy::BackendLocalWithHostOffload,
                    prefix: PrefixPolicy::CommittedPages {
                        positions: PositionSemantics::Absolute,
                    },
                    checkpoint: CheckpointPolicy::CopyOnWrite,
                },
                layers,
                page_size: PageSizeConstraint {
                    min_tokens: 1,
                    preferred_tokens: preferred,
                    max_tokens: preferred.max(256),
                    multiple_of: 1,
                },
                accepted_dtypes: vec![storage_dtype],
            })],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(domain.get()),
                domains: vec![domain],
                prefix_shareable: true,
            }],
        };
        contract.validate()?;
        Ok(contract)
    }

    pub(crate) fn forward_physical(
        &self,
        input_ids: &Tensor,
        position: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let (batch, sequence) = input_ids.dims2()?;
        if batch != 1 || sequence == 0 {
            return Err(Error::InvalidInput(
                "Gemma physical decoder requires one non-empty token sequence".into(),
            ));
        }
        let end_position = position
            .checked_add(sequence)
            .ok_or_else(|| Error::InvalidInput("Gemma physical position overflow".into()))?;
        if end_position > self.config.max_position_embeddings {
            return Err(Error::InvalidInput(
                "Gemma physical sequence exceeds maximum position embeddings".into(),
            ));
        }
        cache.validate_model(
            self.layers.len(),
            self.config.num_key_value_heads,
            self.config.head_dim,
        )?;
        let mut prepared = cache.prepare_append(position, sequence)?;
        let mut hidden =
            (self.embedding.forward(input_ids)? * (self.config.hidden_size as f64).sqrt())?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden =
                layer.forward_physical(&hidden, position, cache, &mut prepared, layer_index)?;
        }
        let logits = hidden.apply(&self.norm)?.apply(&self.lm_head)?;
        let logits = match self.config.final_logit_softcapping {
            Some(softcap) => ((logits / softcap)?.tanh()? * softcap)?,
            None => logits,
        };
        let logits = logits.narrow(1, sequence - 1, 1)?;
        cache.commit_prepared(prepared)?;
        Ok(logits)
    }

    pub(crate) fn forward_physical_decode_batch(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let (batch, sequence) = input_ids.dims2()?;
        if sequence != 1 || batch == 0 || positions.len() != batch || caches.len() != batch {
            return Err(Error::InvalidInput(
                "Gemma physical decode expects matching [batch,1] rows".into(),
            ));
        }
        for (row, cache) in caches.iter().enumerate() {
            cache.validate_model(
                self.layers.len(),
                self.config.num_key_value_heads,
                self.config.head_dim,
            )?;
            cache.slots_for_append(positions[row], 1)?;
        }
        let first = &*caches[0];
        let combined_slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let slots = first.arena().lower_slots(&combined_slots)?;
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), slots.logical_slots())?;
        let mut hidden =
            (self.embedding.forward(input_ids)? * (self.config.hidden_size as f64).sqrt())?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            hidden = layer.forward_physical_batch(
                &hidden,
                positions,
                &cache_refs,
                slots.as_ref(),
                &mut completions,
                layer_index,
            )?;
        }
        let logits = hidden.apply(&self.norm)?.apply(&self.lm_head)?;
        let logits = match self.config.final_logit_softcapping {
            Some(softcap) => ((logits / softcap)?.tanh()? * softcap)?,
            None => logits,
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(positions[row], 1, completion.clone())?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};

    fn tiny_config() -> Config {
        Config {
            attention_bias: false,
            head_dim: 2,
            hidden_activation: candle_nn::Activation::Gelu,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 1_000.0,
            vocab_size: 16,
            final_logit_softcapping: Some(30.0),
            attn_logit_softcapping: None,
            query_pre_attn_scalar: 2,
            sliding_window: 2,
            sliding_window_pattern: 2,
            max_position_embeddings: 16,
        }
    }

    fn values(count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|index| (((index * 7 + 3) % 19) as f32 - 9.0) * scale)
            .collect()
    }

    #[test]
    fn gemma_cuda_rms_norm_defaults_on_with_explicit_rollback() {
        assert!(gemma_cuda_rms_norm_policy(true, None));
        assert!(gemma_cuda_rms_norm_policy(true, Some(true)));
        assert!(!gemma_cuda_rms_norm_policy(true, Some(false)));
        assert!(!gemma_cuda_rms_norm_policy(false, Some(true)));
    }

    #[test]
    fn candle_rms_norm_matches_gemma_reference_weight_transform() {
        let device = &Device::Cpu;
        let input = Tensor::from_vec(
            vec![0.5f32, -1.25, 2.0, 0.75, -0.25, 1.5, -0.5, 2.25],
            (2, 4),
            device,
        )
        .expect("input");
        let weight = Tensor::from_vec(vec![0.1f32, -0.2, 0.05, 0.3], 4, device).expect("weight");
        let reference = GemmaRmsNorm {
            weight: weight.clone(),
            eps: 1e-6,
            cuda_candle_kernel: false,
        }
        .forward(&input)
        .expect("reference");
        let candle = GemmaRmsNorm {
            weight,
            eps: 1e-6,
            cuda_candle_kernel: true,
        }
        .forward(&input)
        .expect("candle rms norm");

        let reference = reference.to_vec2::<f32>().expect("reference values");
        let candle = candle.to_vec2::<f32>().expect("candle values");
        for (reference, candle) in reference
            .into_iter()
            .flatten()
            .zip(candle.into_iter().flatten())
        {
            assert!((reference - candle).abs() < 1e-5, "{reference} != {candle}");
        }
    }

    fn tiny_weights(config: &Config) -> HashMap<String, Tensor> {
        let device = &Device::Cpu;
        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".into(),
            Tensor::from_vec(
                values(config.vocab_size * config.hidden_size, 0.025),
                (config.vocab_size, config.hidden_size),
                device,
            )
            .unwrap(),
        );
        weights.insert(
            "model.norm.weight".into(),
            Tensor::zeros(config.hidden_size, DType::F32, device).unwrap(),
        );
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for (name, output, input) in [
                (
                    "self_attn.q_proj.weight",
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                ),
                (
                    "self_attn.k_proj.weight",
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
                (
                    "self_attn.v_proj.weight",
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
                (
                    "self_attn.o_proj.weight",
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                ),
                (
                    "mlp.gate_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                ),
                (
                    "mlp.up_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                ),
                (
                    "mlp.down_proj.weight",
                    config.hidden_size,
                    config.intermediate_size,
                ),
            ] {
                weights.insert(
                    format!("{prefix}.{name}"),
                    Tensor::from_vec(
                        values(output * input, 0.01 + layer as f32 * 0.002),
                        (output, input),
                        device,
                    )
                    .unwrap(),
                );
            }
            for (name, width) in [
                ("self_attn.q_norm.weight", config.head_dim),
                ("self_attn.k_norm.weight", config.head_dim),
                ("input_layernorm.weight", config.hidden_size),
                ("post_attention_layernorm.weight", config.hidden_size),
                ("pre_feedforward_layernorm.weight", config.hidden_size),
                ("post_feedforward_layernorm.weight", config.hidden_size),
            ] {
                weights.insert(
                    format!("{prefix}.{name}"),
                    Tensor::zeros(width, DType::F32, device).unwrap(),
                );
            }
        }
        weights
    }

    fn tiny_cache(config: &Config) -> PhysicalPagedKvCache {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(91),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let bindings = (0..config.num_hidden_layers)
            .map(|layer| KvLayerBinding {
                model_layer: layer as u32,
                physical_layer: layer as u32,
            })
            .collect::<Vec<_>>();
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 4,
                dtype: DType::F32,
                layers: bindings
                    .iter()
                    .copied()
                    .map(|binding| KvLayerConfig {
                        binding,
                        num_kv_heads: config.num_key_value_heads as u32,
                        key_head_dim: config.head_dim as u32,
                        value_head_dim: config.head_dim as u32,
                    })
                    .collect(),
            })
            .unwrap(),
        );
        PhysicalPagedKvCache::new(
            arena,
            bindings,
            (0..4)
                .map(|index| CacheBlockRef {
                    arena: arena_id,
                    group,
                    index,
                    slot_generation: 1,
                })
                .collect(),
            0,
        )
        .unwrap()
    }

    fn tiny_shared_caches(config: &Config, rows: usize) -> Vec<PhysicalPagedKvCache> {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(92),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let bindings = (0..config.num_hidden_layers)
            .map(|layer| KvLayerBinding {
                model_layer: layer as u32,
                physical_layer: layer as u32,
            })
            .collect::<Vec<_>>();
        let pages_per_row = 4;
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: (rows * pages_per_row) as u32,
                dtype: DType::F32,
                layers: bindings
                    .iter()
                    .copied()
                    .map(|binding| KvLayerConfig {
                        binding,
                        num_kv_heads: config.num_key_value_heads as u32,
                        key_head_dim: config.head_dim as u32,
                        value_head_dim: config.head_dim as u32,
                    })
                    .collect(),
            })
            .unwrap(),
        );
        (0..rows)
            .map(|row| {
                PhysicalPagedKvCache::new(
                    arena.clone(),
                    bindings.clone(),
                    (row * pages_per_row..(row + 1) * pages_per_row)
                        .map(|index| CacheBlockRef {
                            arena: arena_id,
                            group,
                            index: index as u32,
                            slot_generation: 1,
                        })
                        .collect(),
                    0,
                )
                .unwrap()
            })
            .collect()
    }

    fn assert_close(left: &Tensor, right: &Tensor) {
        assert_eq!(left.dims(), right.dims());
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (left, right) in left.iter().zip(right.iter()) {
            assert!((left - right).abs() < 1e-4, "{left} != {right}");
        }
    }

    #[test]
    fn gemma_contract_preserves_loaded_global_and_local_layer_pattern() {
        let mut config = tiny_config();
        config.attn_logit_softcapping = Some(2.5);
        let model = Gemma3PhysicalModel::load(
            config.clone(),
            VarBuilder::from_tensors(tiny_weights(&config), DType::F32, &Device::Cpu),
        )
        .unwrap();
        let contract = model
            .managed_inference_state_contract(StateDomainId::new(3), DType::F32, 4)
            .unwrap();
        let StateDomainSpec::PagedAttention(domain) = &contract.domains[0] else {
            panic!("Gemma contract must contain paged attention");
        };
        assert_eq!(domain.header.id, StateDomainId::new(3));
        assert!(matches!(
            domain.layers[0].pattern,
            AttentionPattern::SlidingWindow { window_tokens: 2 }
        ));
        assert!(matches!(domain.layers[1].pattern, AttentionPattern::Full));
        assert!(domain.layers.iter().all(|layer| {
            layer
                .attention_logit_softcap
                .map(AttentionLogitSoftcap::get)
                == Some(2.5)
        }));
    }

    #[test]
    fn multi_token_physical_prefill_matches_softcapped_gemma_dependency() {
        let mut config = tiny_config();
        config.attn_logit_softcapping = Some(1.5);
        // Candle's batched Gemma mask treats `sliding_window` as prior tokens,
        // while its incremental cache treats it as total retained tokens. Keep
        // the window larger than this prompt so this parity test isolates packed
        // prefill and attention softcapping rather than that dependency mismatch.
        config.sliding_window = 8;
        let weights = tiny_weights(&config);
        let mut dependency = candle_transformers::models::gemma3::Model::new(
            false,
            &config,
            VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu),
        )
        .unwrap();
        let physical = Gemma3PhysicalModel::load(
            config.clone(),
            VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let mut cache = tiny_cache(&config);
        let input = Tensor::from_vec(vec![1u32, 7, 3, 11], (1, 4), &Device::Cpu).unwrap();

        let expected = dependency.forward(&input, 0).unwrap();
        let actual = physical.forward_physical(&input, 0, &mut cache).unwrap();

        assert_close(&actual, &expected);
        assert_eq!(cache.context_len(), 4);
    }

    #[test]
    fn physical_pages_match_dependency_cache_across_local_and_global_layers() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let dependency_builder =
            VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let physical_builder = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
        let mut dependency =
            candle_transformers::models::gemma3::Model::new(false, &config, dependency_builder)
                .unwrap();
        let physical = Gemma3PhysicalModel::load(config.clone(), physical_builder).unwrap();
        let mut cache = tiny_cache(&config);

        for (position, token) in [1u32, 7, 3, 11].into_iter().enumerate() {
            let input = Tensor::from_vec(vec![token], (1, 1), &Device::Cpu).unwrap();
            let expected = dependency.forward(&input, position).unwrap();
            let actual = physical
                .forward_physical(&input, position, &mut cache)
                .unwrap();
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn physical_decode_batch_matches_independent_gemma_sessions() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let mut dependency_a = candle_transformers::models::gemma3::Model::new(
            false,
            &config,
            VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu),
        )
        .unwrap();
        let mut dependency_b = candle_transformers::models::gemma3::Model::new(
            false,
            &config,
            VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu),
        )
        .unwrap();
        let physical = Gemma3PhysicalModel::load(
            config.clone(),
            VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let mut caches = tiny_shared_caches(&config, 2);
        for (position, tokens) in [[1u32, 3], [7, 11]].into_iter().enumerate() {
            dependency_a
                .forward(
                    &Tensor::from_vec(vec![tokens[0]], (1, 1), &Device::Cpu).unwrap(),
                    position,
                )
                .unwrap();
            dependency_b
                .forward(
                    &Tensor::from_vec(vec![tokens[1]], (1, 1), &Device::Cpu).unwrap(),
                    position,
                )
                .unwrap();
            for row in 0..2 {
                physical
                    .forward_physical(
                        &Tensor::from_vec(vec![tokens[row]], (1, 1), &Device::Cpu).unwrap(),
                        position,
                        &mut caches[row],
                    )
                    .unwrap();
            }
        }
        let expected_a = dependency_a
            .forward(
                &Tensor::from_vec(vec![5u32], (1, 1), &Device::Cpu).unwrap(),
                2,
            )
            .unwrap();
        let expected_b = dependency_b
            .forward(
                &Tensor::from_vec(vec![9u32], (1, 1), &Device::Cpu).unwrap(),
                2,
            )
            .unwrap();
        let mut cache_refs = caches.iter_mut().collect::<Vec<_>>();
        let actual = physical
            .forward_physical_decode_batch(
                &Tensor::from_vec(vec![5u32, 9], (2, 1), &Device::Cpu).unwrap(),
                &[2, 2],
                &mut cache_refs,
            )
            .unwrap();
        assert_close(&actual.narrow(0, 0, 1).unwrap(), &expected_a);
        assert_close(&actual.narrow(0, 1, 1).unwrap(), &expected_b);
    }
}
