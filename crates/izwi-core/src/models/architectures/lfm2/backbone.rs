use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;

use candle_transformers::utils::repeat_kv as candle_repeat_kv;

use crate::backends::state::InvocationRingDepthwiseConvTransaction;
use crate::engine::{InvocationTensorLease, StageDescriptor};
use crate::error::{Error, Result};
use crate::kernels::{
    try_fused_qk_rms_norm, try_fused_rms_norm, try_fused_rope_pair_bshd,
    try_fused_silu_mul_with_status, try_lfm_shortconv_decode3, try_lfm_shortconv_sequence3,
};
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention_with_options, CudaFlashAttentionOptions,
};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::telemetry::record_rope_kernel;
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::Lfm2BackboneConfig;
use super::physical::{lfm2_physical_state_spec, Lfm2PhysicalStateSpec, Lfm2StateLayout};

#[derive(Debug)]
struct Mlp {
    gate: QMatMul,
    down: QMatMul,
    up: QMatMul,
}

#[derive(Debug)]
struct AttentionLayer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: LfmRmsNorm,
    k_norm: LfmRmsNorm,
    qk_norm_weight: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    cos: Tensor,
    sin: Tensor,
    cos_sin: Tensor,
    neg_inf: Tensor,
    physical_layer: usize,
}

#[derive(Debug)]
struct ShortConvLayer {
    in_proj: QMatMul,
    out_proj: QMatMul,
    conv: Tensor,
    l_cache: usize,
    component: crate::kv::v2::StateComponentId,
}

#[derive(Debug)]
enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Debug)]
struct LayerWeights {
    operator_norm: LfmRmsNorm,
    ffn_norm: LfmRmsNorm,
    mlp: Mlp,
    kind: LayerKind,
}

#[derive(Debug)]
struct ProjectionHead {
    weight: QMatMul,
    bias: Option<Tensor>,
}

pub struct QuantizedLfm2Backbone {
    cfg: Lfm2BackboneConfig,
    token_embeddings: Embedding,
    output_head: ProjectionHead,
    layers: Vec<LayerWeights>,
    norm: LfmRmsNorm,
    vocab_size: usize,
    state_layout: Lfm2StateLayout,
}

#[derive(Debug, Clone)]
struct LfmRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl LfmRmsNorm {
    fn from_qtensor(weight: candle_core::quantized::QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device()).map_err(Error::from)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.device().is_metal() {
            let input = if input.is_contiguous() {
                input.clone()
            } else {
                input.contiguous()?
            };
            if let Some(output) = try_fused_rms_norm(&input, &self.weight, self.eps) {
                return Ok(output);
            }
        }
        candle_nn::ops::rms_norm(input, &self.weight, self.eps as f32).map_err(Error::from)
    }

    fn eps(&self) -> f64 {
        self.eps
    }

    fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Mlp {
    fn load(loader: &GgufLoader, device: &Device, prefix: &str) -> Result<Self> {
        Self::load_with_prefixes(loader, device, &[prefix.to_string()])
    }

    fn load_with_prefixes(
        loader: &GgufLoader,
        device: &Device,
        prefixes: &[String],
    ) -> Result<Self> {
        let mut gate_names = Vec::new();
        let mut down_names = Vec::new();
        let mut up_names = Vec::new();
        for prefix in prefixes {
            gate_names.extend([
                format!("{prefix}.ffn_gate.weight"),
                format!("{prefix}.feed_forward.w1.weight"),
                format!("{prefix}.mlp.gate_proj.weight"),
            ]);
            down_names.extend([
                format!("{prefix}.ffn_down.weight"),
                format!("{prefix}.feed_forward.w2.weight"),
                format!("{prefix}.mlp.down_proj.weight"),
            ]);
            up_names.extend([
                format!("{prefix}.ffn_up.weight"),
                format!("{prefix}.feed_forward.w3.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
            ]);
        }

        Ok(Self {
            gate: load_qmatmul_any(loader, device, &gate_names)?,
            down: load_qmatmul_any(loader, device, &down_names)?,
            up: load_qmatmul_any(loader, device, &up_names)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(hidden_states)?;
        let up = self.up.forward(hidden_states)?;
        let hidden = if let Some(fused) = try_fused_silu_mul_with_status(&gate, &up) {
            fused.tensor
        } else {
            let gate = candle_nn::ops::silu(&gate)?;
            gate.broadcast_mul(&up)?
        };
        self.down.forward(&hidden).map_err(Error::from)
    }
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        record_rope_kernel();
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Error::from)
    }

    fn try_apply_rotary_emb_pair_bshd(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let (_, seq_len, _, _) = q.dims4()?;
        let packed = self.cos_sin.narrow(0, index_pos, seq_len)?.contiguous()?;
        if let Some((q, k)) = try_fused_rope_pair_bshd(&q.contiguous()?, &k.contiguous()?, &packed)
        {
            record_rope_kernel();
            record_rope_kernel();
            return Ok(Some((q, k)));
        }
        Ok(None)
    }

    fn project_qkv(
        &self,
        hidden_states: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let query_states = self.wq.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ))?;
        let key_states = self.wk.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.n_kv_head,
            self.head_dim,
        ))?;
        let value_states = self
            .wv
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query_states = query_states.contiguous()?;
        let key_states = key_states.contiguous()?;
        let (query_states, key_states) = if seq_len == 1
            && (query_states.device().is_metal() || query_states.device().is_cuda())
        {
            if let Some((query_states, key_states)) = try_fused_qk_rms_norm(
                &query_states,
                &key_states,
                &self.qk_norm_weight,
                self.q_norm.eps(),
            ) {
                (query_states, key_states)
            } else {
                (
                    self.q_norm.forward(&query_states)?,
                    self.k_norm.forward(&key_states)?,
                )
            }
        } else {
            (
                self.q_norm.forward(&query_states)?,
                self.k_norm.forward(&key_states)?,
            )
        };
        let (query_states, key_states) = if let Some((query_states, key_states)) =
            self.try_apply_rotary_emb_pair_bshd(&query_states, &key_states, index_pos)?
        {
            (
                query_states.transpose(1, 2)?.contiguous()?,
                key_states.transpose(1, 2)?.contiguous()?,
            )
        } else {
            let query_states = query_states.transpose(1, 2)?.contiguous()?;
            let key_states = key_states.transpose(1, 2)?.contiguous()?;
            (
                self.apply_rotary_emb(&query_states, index_pos)?,
                self.apply_rotary_emb(&key_states, index_pos)?,
            )
        };
        Ok((query_states, key_states, value_states))
    }

    fn forward_stateless(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let (query_states, key_states, value_states) =
            self.project_qkv(hidden_states, index_pos)?;

        if query_states.device().is_cuda() && flash_attention_requested() {
            let cuda_options =
                lfm25_cuda_flash_attention_options(mask.is_some(), self.sliding_window);
            if let Some(attn_output) = try_fused_self_attention_with_options(
                &query_states,
                &key_states,
                &value_states,
                None,
                self.head_dim,
                true,
                cuda_options,
            )? {
                let attn_output =
                    attn_output
                        .transpose(1, 2)?
                        .reshape((batch_size, seq_len, hidden_size))?;
                return self.wo.forward(&attn_output).map_err(Error::from);
            }
        }

        if should_try_lfm25_metal_prefill_sdpa(
            &query_states,
            seq_len,
            index_pos,
            self.sliding_window,
        ) {
            if let Some(attn_output) = try_fused_self_attention_with_options(
                &query_states,
                &key_states,
                &value_states,
                None,
                self.head_dim,
                true,
                CudaFlashAttentionOptions::default(),
            )? {
                let attn_output =
                    attn_output
                        .transpose(1, 2)?
                        .reshape((batch_size, seq_len, hidden_size))?;
                return self.wo.forward(&attn_output).map_err(Error::from);
            }
        }

        let (key_states, value_states) = if self.n_head != self.n_kv_head {
            let repeats = self.n_head / self.n_kv_head;
            (
                candle_repeat_kv(key_states, repeats)?,
                candle_repeat_kv(value_states, repeats)?,
            )
        } else {
            (key_states, value_states)
        };

        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?.contiguous()?)?
            / (self.head_dim as f64).sqrt())?;
        let attn_weights = if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            masked_fill(&attn_weights, &mask, &self.neg_inf)?
        } else {
            attn_weights
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights
            .contiguous()?
            .matmul(&value_states.contiguous()?)?;
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((batch_size, seq_len, hidden_size))?;
        self.wo.forward(&attn_output).map_err(Error::from)
    }

    fn forward_physical(
        &self,
        hidden_states: &Tensor,
        index_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        if batch_size != 1 {
            return Err(Error::InvalidInput(
                "LFM2 physical attention currently requires batch size one".into(),
            ));
        }
        let (queries, keys, values) = self.project_qkv(hidden_states, index_pos)?;
        let queries = queries.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let keys = keys.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let values = values.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let softmax_scale = (1.0f64 / (self.head_dim as f64).sqrt()) as f32;
        let output = match self.sliding_window {
            Some(window_tokens) => cache.write_and_attend_with_window(
                self.physical_layer,
                prepared,
                &queries,
                &keys,
                &values,
                softmax_scale,
                window_tokens,
            )?,
            None => cache.write_and_attend(
                self.physical_layer,
                prepared,
                &queries,
                &keys,
                &values,
                softmax_scale,
            )?,
        };
        let output = output.reshape((batch_size, seq_len, hidden_size))?;
        self.wo.forward(&output).map_err(Error::from)
    }
}

impl ShortConvLayer {
    fn forward_stateless(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let projected = self.in_proj.forward(hidden_states)?.transpose(1, 2)?;
        let b = projected.narrow(1, 0, hidden_size)?;
        let c = projected.narrow(1, hidden_size, hidden_size)?;
        let x = projected.narrow(1, hidden_size * 2, hidden_size)?;
        let bx = (&b * &x)?.contiguous()?;

        let conv_weight = &self.conv;

        let conv_out = if seq_len == 1 {
            let state = Tensor::zeros(
                (batch_size, hidden_size, self.l_cache),
                bx.dtype(),
                bx.device(),
            )?;
            let fused_conv_out = if self.l_cache == 3 {
                try_lfm_shortconv_decode3(&state, &bx, conv_weight)
            } else {
                None
            };

            if let Some(conv_out) = fused_conv_out {
                conv_out
            } else {
                let state = if self.l_cache > 1 {
                    let tail = state.narrow(2, 1, self.l_cache - 1)?;
                    Tensor::cat(&[&tail, &bx], 2)?
                } else {
                    bx.clone()
                };
                (&state * &conv_weight.unsqueeze(0)?)?
                    .sum_keepdim(2)?
                    .contiguous()?
            }
        } else {
            let bx = bx.contiguous()?;
            let out = if self.l_cache == 3 {
                try_lfm_shortconv_sequence3(&bx, conv_weight)
            } else {
                None
            };
            let out = if let Some(out) = out {
                out
            } else {
                let conv = Conv1d::new(
                    conv_weight
                        .reshape((hidden_size, 1, self.l_cache))?
                        .contiguous()?,
                    None,
                    Conv1dConfig {
                        padding: self.l_cache.saturating_sub(1),
                        groups: hidden_size,
                        ..Default::default()
                    },
                );
                conv.forward(&bx)?.narrow(2, 0, seq_len)?
            };

            out
        };

        let conv_out = (&c * &conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out).map_err(Error::from)
    }

    fn forward_physical(
        &self,
        hidden_states: &Tensor,
        transaction: &mut InvocationRingDepthwiseConvTransaction<'_>,
    ) -> Result<Tensor> {
        let (_, _, hidden_size) = hidden_states.dims3()?;
        let projected = self.in_proj.forward(hidden_states)?.transpose(1, 2)?;
        let b = projected.narrow(1, 0, hidden_size)?;
        let c = projected.narrow(1, hidden_size, hidden_size)?;
        let x = projected.narrow(1, hidden_size * 2, hidden_size)?;
        let bx = (&b * &x)?.contiguous()?;
        let conv_out = transaction.apply(self.component, &bx, &self.conv)?;
        let conv_out = (&c * &conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out).map_err(Error::from)
    }
}

impl ProjectionHead {
    fn load(loader: &GgufLoader, device: &Device) -> Result<Self> {
        let weight = load_qmatmul_any(
            loader,
            device,
            &[
                "output.weight".to_string(),
                "lm_head.weight".to_string(),
                "model.output.weight".to_string(),
                "model.lm_head.weight".to_string(),
                "lfm.output.weight".to_string(),
                "lfm.lm_head.weight".to_string(),
                "dense_2_out.weight".to_string(),
                "dense_2.weight".to_string(),
                "lin.weight".to_string(),
                "token_embd.weight".to_string(),
                "tok_embeddings.weight".to_string(),
            ],
        )?;
        let bias = load_optional_bias_any(
            loader,
            device,
            &[
                "output.bias".to_string(),
                "lm_head.bias".to_string(),
                "lfm.output.bias".to_string(),
                "lfm.lm_head.bias".to_string(),
                "dense_2_out.bias".to_string(),
                "dense_2.bias".to_string(),
                "lin.bias".to_string(),
            ],
        )?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let projected = self.weight.forward(hidden_states)?;
        if let Some(bias) = &self.bias {
            projected.broadcast_add(bias).map_err(Error::from)
        } else {
            Ok(projected)
        }
    }
}

impl QuantizedLfm2Backbone {
    pub fn load(loader: &GgufLoader, cfg: Lfm2BackboneConfig, device: &Device) -> Result<Self> {
        let state_layout = Lfm2StateLayout::from_config(&cfg)?;
        let (cos, sin) = precompute_freqs(
            cfg.embedding_length / cfg.attention_head_count,
            cfg.rope_freq_base as f32,
            cfg.context_length,
            device,
        )?;
        let cos_sin = Tensor::cat(&[&cos, &sin], 1)?.contiguous()?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let token_embedding_q = load_qtensor_any(
            loader,
            device,
            &[
                "token_embd.weight".to_string(),
                "tok_embeddings.weight".to_string(),
                "model.embed_tokens.weight".to_string(),
                "lfm.embed_tokens.weight".to_string(),
                "emb.emb.weight".to_string(),
                "emb.weight".to_string(),
            ],
        )?;
        let token_embeddings_weight = token_embedding_q.dequantize(device).map_err(Error::from)?;
        let (vocab_size, hidden_size) = token_embeddings_weight.dims2()?;
        if hidden_size != cfg.embedding_length {
            return Err(Error::ModelLoadError(format!(
                "LFM2 embedding width mismatch: GGUF has {hidden_size}, metadata says {}",
                cfg.embedding_length
            )));
        }

        let token_embeddings = Embedding::new(token_embeddings_weight, hidden_size);
        let norm = LfmRmsNorm::from_qtensor(
            load_qtensor_any(
                loader,
                device,
                &[
                    "output_norm.weight".to_string(),
                    "embedding_norm.weight".to_string(),
                    "model.embedding_norm.weight".to_string(),
                    "model.embedding_norm".to_string(),
                    "token_embd_norm.weight".to_string(),
                    "lfm.embedding_norm.weight".to_string(),
                ],
            )?,
            cfg.attention_layer_norm_rms_epsilon,
        )?;
        let output_head = ProjectionHead::load(loader, device)?;

        let mut layers = Vec::with_capacity(cfg.block_count);
        for layer_idx in 0..cfg.block_count {
            let prefix = format!("blk.{layer_idx}");
            let legacy_prefix = format!("lfm.layers.{layer_idx}");
            let operator_norm = LfmRmsNorm::from_qtensor(
                load_qtensor_any(
                    loader,
                    device,
                    &[
                        format!("{prefix}.attn_norm.weight"),
                        format!("{prefix}.operator_norm.weight"),
                        format!("{prefix}.attention_norm.weight"),
                        format!("{legacy_prefix}.operator_norm.weight"),
                    ],
                )?,
                cfg.attention_layer_norm_rms_epsilon,
            )?;
            let ffn_norm = LfmRmsNorm::from_qtensor(
                load_qtensor_any(
                    loader,
                    device,
                    &[
                        format!("{prefix}.ffn_norm.weight"),
                        format!("{prefix}.ffn_norm"),
                        format!("{legacy_prefix}.ffn_norm.weight"),
                    ],
                )?,
                cfg.attention_layer_norm_rms_epsilon,
            )?;
            let mlp =
                Mlp::load_with_prefixes(loader, device, &[prefix.clone(), legacy_prefix.clone()])?;

            let is_attention = cfg
                .attention_head_count_kv
                .get(layer_idx)
                .copied()
                .unwrap_or(cfg.attention_head_count)
                > 0;
            let kind = if is_attention {
                let n_kv_head = cfg.attention_head_count_kv[layer_idx];
                let physical_layer = state_layout
                    .attention_model_layers
                    .iter()
                    .position(|model_layer| *model_layer as usize == layer_idx)
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "LFM2 attention layer {layer_idx} is absent from physical layout"
                        ))
                    })?;
                let q_norm = LfmRmsNorm::from_qtensor(
                    load_qtensor_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_q_norm.weight"),
                            format!("{prefix}.self_attn.q_layernorm.weight"),
                            format!("{prefix}.attention.q_norm.weight"),
                            format!("{legacy_prefix}.self_attn.q_layernorm.weight"),
                        ],
                    )?,
                    cfg.attention_layer_norm_rms_epsilon,
                )?;
                let k_norm = LfmRmsNorm::from_qtensor(
                    load_qtensor_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_k_norm.weight"),
                            format!("{prefix}.self_attn.k_layernorm.weight"),
                            format!("{prefix}.attention.k_norm.weight"),
                            format!("{legacy_prefix}.self_attn.k_layernorm.weight"),
                        ],
                    )?,
                    cfg.attention_layer_norm_rms_epsilon,
                )?;
                let qk_norm_weight =
                    Tensor::cat(&[q_norm.weight(), k_norm.weight()], 0)?.contiguous()?;
                LayerKind::Attention(AttentionLayer {
                    wq: load_qmatmul_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_q.weight"),
                            format!("{prefix}.self_attn.q_proj.weight"),
                            format!("{legacy_prefix}.self_attn.q_proj.weight"),
                        ],
                    )?,
                    wk: load_qmatmul_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_k.weight"),
                            format!("{prefix}.self_attn.k_proj.weight"),
                            format!("{legacy_prefix}.self_attn.k_proj.weight"),
                        ],
                    )?,
                    wv: load_qmatmul_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_v.weight"),
                            format!("{prefix}.self_attn.v_proj.weight"),
                            format!("{legacy_prefix}.self_attn.v_proj.weight"),
                        ],
                    )?,
                    wo: load_qmatmul_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.attn_output.weight"),
                            format!("{prefix}.self_attn.out_proj.weight"),
                            format!("{legacy_prefix}.self_attn.out_proj.weight"),
                        ],
                    )?,
                    q_norm,
                    k_norm,
                    qk_norm_weight,
                    n_head: cfg.attention_head_count,
                    n_kv_head,
                    head_dim: cfg.embedding_length / cfg.attention_head_count,
                    sliding_window: cfg.attention_sliding_window,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    cos_sin: cos_sin.clone(),
                    neg_inf: neg_inf.clone(),
                    physical_layer,
                })
            } else {
                LayerKind::ShortConv(ShortConvLayer {
                    in_proj: load_qmatmul_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.shortconv.in_proj.weight"),
                            format!("{prefix}.conv.in_proj.weight"),
                            format!("{legacy_prefix}.conv.in_proj.weight"),
                        ],
                    )?,
                    out_proj: load_qmatmul_any(
                        loader,
                        device,
                        &[
                            format!("{prefix}.shortconv.out_proj.weight"),
                            format!("{prefix}.conv.out_proj.weight"),
                            format!("{legacy_prefix}.conv.out_proj.weight"),
                        ],
                    )?,
                    conv: normalize_shortconv_weight(
                        load_dense_any(
                            loader,
                            device,
                            &[
                                format!("{prefix}.shortconv.conv.weight"),
                                format!("{prefix}.conv.conv.weight"),
                                format!("{prefix}.shortconv.conv"),
                                format!("{legacy_prefix}.conv.conv.weight"),
                            ],
                            Some(DType::F32),
                        )?,
                        cfg.shortconv_l_cache,
                        cfg.embedding_length,
                    )?,
                    l_cache: cfg.shortconv_l_cache,
                    component: state_layout.shortconv_component(layer_idx)?,
                })
            };

            layers.push(LayerWeights {
                operator_norm,
                ffn_norm,
                mlp,
                kind,
            });
        }

        Ok(Self {
            cfg,
            token_embeddings,
            output_head,
            layers,
            norm,
            vocab_size,
            state_layout,
        })
    }

    pub fn config(&self) -> &Lfm2BackboneConfig {
        &self.cfg
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.embedding_length
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Lfm2PhysicalStateSpec> {
        lfm2_physical_state_spec(&self.cfg, stage_graphs)
    }

    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.token_embeddings
            .forward(token_ids)
            .map_err(Error::from)
    }

    pub fn project_hidden(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.output_head.forward(hidden_states)
    }

    pub fn project_last_hidden(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        let last_hidden = hidden_states.i((.., seq_len - 1, ..))?;
        self.output_head.forward(&last_hidden)
    }

    pub(crate) fn forward_tokens_physical(
        &mut self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
    ) -> Result<Tensor> {
        let hidden_states = self.embed_tokens(token_ids)?;
        let hidden_states =
            self.forward_embeds_physical(&hidden_states, index_pos, cache, shortconv)?;
        self.project_last_hidden(&hidden_states)
    }

    pub(crate) fn forward_embeds_physical(
        &mut self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
    ) -> Result<Tensor> {
        let (batch, _seq_len, hidden) = input_embeds.dims3()?;
        if batch != 1 || hidden != self.cfg.embedding_length || index_pos != cache.context_len() {
            return Err(Error::InvalidInput(
                "LFM2 physical input does not match its invocation state".into(),
            ));
        }
        let head_dim = self.cfg.embedding_length / self.cfg.attention_head_count;
        let sparse_geometry = self
            .state_layout
            .attention_model_layers
            .iter()
            .map(|model_layer| {
                (
                    *model_layer,
                    self.cfg.attention_head_count_kv[*model_layer as usize],
                    head_dim,
                    head_dim,
                )
            })
            .collect::<Vec<_>>();
        cache.validate_sparse_model_layers(&sparse_geometry)?;

        self.forward_embeds_physical_chunk(
            input_embeds,
            index_pos,
            cache,
            shortconv,
            self.cfg.attention_sliding_window,
        )
    }

    fn forward_embeds_physical_chunk(
        &mut self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = input_embeds.dims3()?;
        let mut prepared = match sliding_window {
            Some(window_tokens) => {
                cache.prepare_append_with_window(index_pos, seq_len, window_tokens)?
            }
            None => cache.prepare_append(index_pos, seq_len)?,
        };
        let intent = self.state_layout.ring_step_intent(
            shortconv.domain(),
            index_pos,
            batch,
            hidden,
            seq_len,
        )?;
        let output = shortconv.with_ring_depthwise_conv(&intent, |transaction| {
            let mut hidden_states = input_embeds.clone();
            for layer in self.layers.iter_mut() {
                let residual = hidden_states.clone();
                let hidden = layer.operator_norm.forward(&hidden_states)?;
                let hidden = match &mut layer.kind {
                    LayerKind::Attention(attention) => {
                        attention.forward_physical(&hidden, index_pos, cache, &mut prepared)?
                    }
                    LayerKind::ShortConv(shortconv) => {
                        shortconv.forward_physical(&hidden, transaction)?
                    }
                };
                hidden_states = (&hidden + &residual)?;

                let residual = hidden_states.clone();
                let hidden = layer.ffn_norm.forward(&hidden_states)?;
                let hidden = layer.mlp.forward(&hidden)?;
                hidden_states = (&hidden + &residual)?;
            }
            self.norm.forward(&hidden_states).map_err(Error::from)
        })?;
        cache.commit_prepared(prepared)?;
        Ok(output)
    }

    pub(crate) fn forward_embeds_stateless(&self, input_embeds: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = input_embeds.dims3()?;
        if batch != 1 || hidden != self.cfg.embedding_length || seq_len == 0 {
            return Err(Error::InvalidInput(
                "LFM2 stateless input does not match the loaded backbone".into(),
            ));
        }
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.mask(seq_len, input_embeds.device())?)
        };

        let mut hidden_states = input_embeds.clone();
        for layer in &self.layers {
            let residual = hidden_states.clone();
            let hidden = layer.operator_norm.forward(&hidden_states)?;
            let hidden = match &layer.kind {
                LayerKind::Attention(attention) => {
                    attention.forward_stateless(&hidden, mask.as_ref(), 0)?
                }
                LayerKind::ShortConv(shortconv) => shortconv.forward_stateless(&hidden)?,
            };
            hidden_states = (&hidden + &residual)?;

            let residual = hidden_states.clone();
            let hidden = layer.ffn_norm.forward(&hidden_states)?;
            let hidden = layer.mlp.forward(&hidden)?;
            hidden_states = (&hidden + &residual)?;
        }
        self.norm.forward(&hidden_states).map_err(Error::from)
    }

    fn mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let sliding_window = self.cfg.attention_sliding_window;
        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|query| {
                (0..seq_len).map(move |key| {
                    u8::from(lfm2_attention_position_masked(query, key, sliding_window))
                })
            })
            .collect();
        Tensor::from_slice(&mask, (seq_len, seq_len), device).map_err(Error::from)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
        .map_err(Error::from)
}

fn lfm25_cuda_flash_attention_options(
    masked_prefill: bool,
    sliding_window: Option<usize>,
) -> CudaFlashAttentionOptions<'static> {
    CudaFlashAttentionOptions {
        window_size_left: if masked_prefill {
            sliding_window.map(|window| window.saturating_sub(1))
        } else {
            None
        },
        ..CudaFlashAttentionOptions::default()
    }
}

fn lfm2_attention_position_masked(query: usize, key: usize, sliding_window: Option<usize>) -> bool {
    key > query || sliding_window.is_some_and(|window| query.saturating_sub(key) >= window)
}

fn should_try_lfm25_metal_prefill_sdpa(
    query_states: &Tensor,
    seq_len: usize,
    index_pos: usize,
    sliding_window: Option<usize>,
) -> bool {
    seq_len > 1
        && index_pos == 0
        && sliding_window.is_none()
        && query_states.device().is_metal()
        && lfm25_metal_prefill_sdpa_enabled()
}

fn lfm25_metal_prefill_sdpa_enabled() -> bool {
    std::env::var("IZWI_LFM25_METAL_PREFILL_SDPA")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

fn normalize_shortconv_weight(
    mut conv_weight: Tensor,
    l_cache: usize,
    hidden_size: usize,
) -> Result<Tensor> {
    match conv_weight.rank() {
        3 => conv_weight = conv_weight.squeeze(1)?,
        2 => {}
        rank => {
            return Err(Error::ModelLoadError(format!(
                "Unexpected LFM2 shortconv weight rank: {rank}"
            )));
        }
    }

    let (mut rows, mut cols) = conv_weight.dims2()?;
    if rows == l_cache && cols == hidden_size {
        conv_weight = conv_weight.transpose(0, 1)?;
        (rows, cols) = conv_weight.dims2()?;
    }
    if rows != hidden_size || cols != l_cache {
        return Err(Error::ModelLoadError(format!(
            "Unexpected LFM2 shortconv weight shape: expected [{hidden_size}, {l_cache}], found [{rows}, {cols}]"
        )));
    }
    conv_weight.contiguous().map_err(Error::from)
}

fn precompute_freqs(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|idx| 1.0f32 / freq_base.powf(idx as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let angles = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((angles.cos()?, angles.sin()?))
}

fn load_qmatmul_any(loader: &GgufLoader, device: &Device, names: &[String]) -> Result<QMatMul> {
    let weights = Arc::new(load_qtensor_any(loader, device, names)?);
    QMatMul::from_weights(weights).map_err(Error::from)
}

fn load_qtensor_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<candle_core::quantized::QTensor> {
    for name in names {
        if loader.has_tensor(name) {
            return loader.load_qtensor(name, device);
        }
    }
    Err(Error::ModelLoadError(format!(
        "Missing GGUF tensor; tried {}",
        names.join(" | ")
    )))
}

fn load_dense_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    dtype: Option<DType>,
) -> Result<Tensor> {
    for name in names {
        if loader.has_tensor(name) {
            let mut tensor = loader.load_qtensor(name, device)?.dequantize(device)?;
            if let Some(dtype) = dtype {
                if tensor.dtype() != dtype {
                    tensor = tensor.to_dtype(dtype)?;
                }
            }
            return Ok(tensor);
        }
    }
    Err(Error::ModelLoadError(format!(
        "Missing GGUF tensor; tried {}",
        names.join(" | ")
    )))
}

fn load_optional_bias_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<Option<Tensor>> {
    for name in names {
        if loader.has_tensor(name) {
            return load_dense_any(loader, device, std::slice::from_ref(name), Some(DType::F32))
                .map(Some);
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{lfm25_cuda_flash_attention_options, lfm2_attention_position_masked};

    #[test]
    fn lfm25_cuda_flash_options_use_window_only_for_masked_prefill() {
        let options = lfm25_cuda_flash_attention_options(true, Some(512));
        assert_eq!(options.window_size_left, Some(511));
        assert_eq!(options.window_size_right, None);
        assert!(options.alibi_slopes.is_none());
        assert!(options.softcap.is_none());

        let decode_options = lfm25_cuda_flash_attention_options(false, Some(512));
        assert_eq!(decode_options.window_size_left, None);

        let full_causal_options = lfm25_cuda_flash_attention_options(true, None);
        assert_eq!(full_causal_options.window_size_left, None);
    }

    #[test]
    fn lfm2_sliding_window_counts_the_current_token() {
        let window = Some(3);
        assert!(!lfm2_attention_position_masked(4, 4, window));
        assert!(!lfm2_attention_position_masked(4, 2, window));
        assert!(lfm2_attention_position_masked(4, 1, window));
        assert!(lfm2_attention_position_masked(4, 5, window));
        assert!(!lfm2_attention_position_masked(4, 0, None));
    }
}
