use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;

use candle_transformers::utils::repeat_kv as candle_repeat_kv;

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::Lfm2BackboneConfig;

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
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

#[derive(Debug)]
struct ShortConvLayer {
    in_proj: QMatMul,
    out_proj: QMatMul,
    conv: Tensor,
    l_cache: usize,
    cache: Option<Tensor>,
}

#[derive(Debug)]
enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Debug)]
struct LayerWeights {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
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
    norm: RmsNorm,
    masks: HashMap<usize, Tensor>,
    vocab_size: usize,
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
        let gate = candle_nn::ops::silu(&self.gate.forward(hidden_states)?)?;
        let up = self.up.forward(hidden_states)?;
        self.down.forward(&(&gate * &up)?).map_err(Error::from)
    }
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Error::from)
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;

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
        let value_states = self.wv.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.n_kv_head,
            self.head_dim,
        ))?;

        let query_states = query_states.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.transpose(1, 2)?.contiguous()?;

        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;
        let query_states = self.apply_rotary_emb(&query_states, index_pos)?;
        let key_states = self.apply_rotary_emb(&key_states, index_pos)?;

        let (all_keys, all_values) = if let Some((cached_keys, cached_values)) = &self.kv_cache {
            if index_pos == 0 {
                (key_states, value_states)
            } else {
                (
                    Tensor::cat(&[cached_keys, &key_states], 2)?,
                    Tensor::cat(&[cached_values, &value_states], 2)?,
                )
            }
        } else {
            (key_states, value_states)
        };
        self.kv_cache = Some((all_keys.clone(), all_values.clone()));

        let (key_states, value_states) = if self.n_head != self.n_kv_head {
            let repeats = self.n_head / self.n_kv_head;
            (
                candle_repeat_kv(all_keys, repeats)?,
                candle_repeat_kv(all_values, repeats)?,
            )
        } else {
            (all_keys, all_values)
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

    fn reset_state(&mut self) {
        self.kv_cache = None;
    }
}

impl ShortConvLayer {
    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let projected = self.in_proj.forward(hidden_states)?.transpose(1, 2)?;
        let b = projected.narrow(1, 0, hidden_size)?;
        let c = projected.narrow(1, hidden_size, hidden_size)?;
        let x = projected.narrow(1, hidden_size * 2, hidden_size)?;
        let bx = (&b * &x)?.contiguous()?;

        let mut conv_weight = self.conv.clone();
        match conv_weight.rank() {
            3 => conv_weight = conv_weight.squeeze(1)?,
            2 => {
                let (d0, d1) = conv_weight.dims2()?;
                if d0 == self.l_cache && d1 == hidden_size {
                    conv_weight = conv_weight.transpose(0, 1)?.contiguous()?;
                }
            }
            _ => {}
        }
        let conv_weight = conv_weight.contiguous()?;

        let conv_out = if seq_len == 1 {
            let mut state = if let Some(cache) = &self.cache {
                cache.clone()
            } else {
                Tensor::zeros(
                    (batch_size, hidden_size, self.l_cache),
                    bx.dtype(),
                    bx.device(),
                )?
            };

            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[&tail, &bx], 2)?;
            } else {
                state = bx.clone();
            }
            self.cache = Some(state.clone());

            (&state * &conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
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
            let mut out = conv.forward(&bx.contiguous()?)?;
            out = out.narrow(2, 0, seq_len)?;

            if self.l_cache > 0 {
                let (_, _, cur_len) = bx.dims3()?;
                let start = cur_len.saturating_sub(self.l_cache);
                let mut cache = bx.narrow(2, start, cur_len - start)?;
                if cache.dims3()?.2 < self.l_cache {
                    let pad = self.l_cache - cache.dims3()?.2;
                    let zeros = Tensor::zeros(
                        (batch_size, hidden_size, pad),
                        cache.dtype(),
                        cache.device(),
                    )?;
                    cache = Tensor::cat(&[&zeros, &cache], 2)?;
                }
                self.cache = Some(cache);
            }

            out
        };

        let conv_out = (&c * &conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out).map_err(Error::from)
    }

    fn reset_state(&mut self) {
        self.cache = None;
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
        let (cos, sin) = precompute_freqs(
            cfg.embedding_length / cfg.attention_head_count,
            cfg.rope_freq_base as f32,
            cfg.context_length,
            device,
        )?;
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
        let norm = RmsNorm::from_qtensor(
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
        )
        .map_err(Error::from)?;
        let output_head = ProjectionHead::load(loader, device)?;

        let mut layers = Vec::with_capacity(cfg.block_count);
        for layer_idx in 0..cfg.block_count {
            let prefix = format!("blk.{layer_idx}");
            let legacy_prefix = format!("lfm.layers.{layer_idx}");
            let operator_norm = RmsNorm::from_qtensor(
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
            )
            .map_err(Error::from)?;
            let ffn_norm = RmsNorm::from_qtensor(
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
            )
            .map_err(Error::from)?;
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
                    q_norm: RmsNorm::from_qtensor(
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
                    )
                    .map_err(Error::from)?,
                    k_norm: RmsNorm::from_qtensor(
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
                    )
                    .map_err(Error::from)?,
                    n_head: cfg.attention_head_count,
                    n_kv_head,
                    head_dim: cfg.embedding_length / cfg.attention_head_count,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    kv_cache: None,
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
                    conv: load_dense_any(
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
                    l_cache: cfg.shortconv_l_cache,
                    cache: None,
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
            masks: HashMap::new(),
            vocab_size,
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

    pub fn forward_tokens(&mut self, token_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let hidden_states = self.embed_tokens(token_ids)?;
        let hidden_states = self.forward_embeds(&hidden_states, index_pos)?;
        self.project_last_hidden(&hidden_states)
    }

    pub fn forward_embeds(&mut self, input_embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, seq_len, _) = input_embeds.dims3()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.mask(seq_len, input_embeds.device())?)
        };

        let mut hidden_states = input_embeds.clone();
        for layer in self.layers.iter_mut() {
            let residual = hidden_states.clone();
            let hidden = layer.operator_norm.forward(&hidden_states)?;
            let hidden = match &mut layer.kind {
                LayerKind::Attention(attention) => {
                    attention.forward(&hidden, mask.as_ref(), index_pos)?
                }
                LayerKind::ShortConv(shortconv) => shortconv.forward(&hidden)?,
            };
            hidden_states = (&hidden + &residual)?;

            let residual = hidden_states.clone();
            let hidden = layer.ffn_norm.forward(&hidden_states)?;
            let hidden = layer.mlp.forward(&hidden)?;
            hidden_states = (&hidden + &residual)?;
        }
        self.norm.forward(&hidden_states).map_err(Error::from)
    }

    pub fn reset_state(&mut self) {
        self.masks.clear();
        for layer in &mut self.layers {
            match &mut layer.kind {
                LayerKind::Attention(attention) => attention.reset_state(),
                LayerKind::ShortConv(shortconv) => shortconv.reset_state(),
            }
        }
    }

    fn mask(&mut self, seq_len: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&seq_len) {
            return Ok(mask.clone());
        }

        let mask: Vec<u8> = if let Some(sliding_window) = self.cfg.attention_sliding_window {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i || j + sliding_window < i)))
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        self.masks.insert(seq_len, mask.clone());
        Ok(mask)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
        .map_err(Error::from)
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
