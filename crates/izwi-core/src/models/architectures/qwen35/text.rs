use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{ops, Embedding};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::{Error, Result};
use crate::kernels::metal::{try_fused_gated_delta_recurrent, try_fused_l2_norm};
use crate::models::architectures::qwen3::core::repeat_kv;
use crate::models::shared::attention::flash::try_fused_self_attention;
use crate::models::shared::attention::paged::{
    append_to_pages, default_kv_page_size, default_kv_quantization, materialize_pages,
    paged_decode_attention, KvCacheQuantization, KvPage,
};
use crate::models::shared::weights::gguf::GgufLoader;

use super::chat::Qwen35TextConfig;

pub struct Qwen35TextModel {
    device: Device,
    token_embeddings: Embedding,
    layers: Vec<Qwen35Layer>,
    output_norm: RmsNorm,
    output: QMatMul,
}

pub struct Qwen35TextRuntimeState {
    layers: Vec<Qwen35LayerRuntimeState>,
}

enum Qwen35LayerRuntimeState {
    Linear {
        conv_state: Option<Tensor>,
        recurrent_state: Option<Tensor>,
    },
    Full {
        k_pages: Vec<KvPage>,
        v_pages: Vec<KvPage>,
    },
}

struct Qwen35Layer {
    attn_norm: RmsNorm,
    mixer: Qwen35Mixer,
    post_attention_norm: RmsNorm,
    mlp: Qwen35Mlp,
}

enum Qwen35Mixer {
    Linear(Qwen35LinearAttention),
    Full(Qwen35FullAttention),
}

struct Qwen35Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

struct Qwen35FullAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    mrope_sections: Vec<usize>,
    kv_page_size: usize,
    kv_quantization: KvCacheQuantization,
    rope_inv_freqs: Vec<f32>,
    rope_cache: Mutex<HashMap<[usize; 3], (Tensor, Tensor)>>,
}

struct Qwen35LinearAttention {
    qkv_proj: QMatMul,
    gate_proj: QMatMul,
    beta_proj: QMatMul,
    alpha_proj: QMatMul,
    dt_bias: Tensor,
    a: Tensor,
    conv_kernel: Tensor,
    norm: Qwen35GatedRmsNorm,
    out_proj: QMatMul,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_dim: usize,
    kernel_size: usize,
}

struct Qwen35GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen35TextModel {
    pub fn load(loader: &GgufLoader, cfg: &Qwen35TextConfig, device: &Device) -> Result<Self> {
        if cfg.attention_key_length != cfg.attention_value_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 full attention currently requires key/value head dims to match, found {} and {}",
                cfg.attention_key_length, cfg.attention_value_length
            )));
        }
        if cfg.ssm_time_step_rank == 0 || cfg.ssm_inner_size % cfg.ssm_time_step_rank != 0 {
            return Err(Error::ModelLoadError(format!(
                "Invalid Qwen3.5 linear attention dims: inner_size={}, time_step_rank={}",
                cfg.ssm_inner_size, cfg.ssm_time_step_rank
            )));
        }

        let embedding_weights = loader
            .load_qtensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(Error::from)?;
        let (vocab_size, hidden_size) = embedding_weights.dims2()?;
        if hidden_size != cfg.embedding_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 token embedding width mismatch: GGUF has {hidden_size}, metadata says {}",
                cfg.embedding_length
            )));
        }
        let _ = vocab_size;

        let token_embeddings = Embedding::new(embedding_weights, hidden_size);
        let output_norm = load_rms_norm(loader, device, "output_norm.weight", cfg)?;
        let output = if loader.has_tensor("output.weight") {
            load_qmatmul(loader, device, "output.weight")?
        } else {
            load_qmatmul(loader, device, "token_embd.weight")?
        };

        let mut layers = Vec::with_capacity(cfg.block_count);
        for layer_idx in 0..cfg.block_count {
            let prefix = format!("blk.{layer_idx}");
            let attn_norm =
                load_rms_norm(loader, device, &format!("{prefix}.attn_norm.weight"), cfg)?;
            let post_attention_norm = load_rms_norm(
                loader,
                device,
                &format!("{prefix}.post_attention_norm.weight"),
                cfg,
            )?;
            let mlp = Qwen35Mlp::load(loader, device, &prefix)?;
            let mixer = if is_full_attention_layer(layer_idx, cfg.full_attention_interval) {
                Qwen35Mixer::Full(Qwen35FullAttention::load(loader, device, &prefix, cfg)?)
            } else {
                Qwen35Mixer::Linear(Qwen35LinearAttention::load(loader, device, &prefix, cfg)?)
            };

            layers.push(Qwen35Layer {
                attn_norm,
                mixer,
                post_attention_norm,
                mlp,
            });
        }

        Ok(Self {
            device: device.clone(),
            token_embeddings,
            layers,
            output_norm,
            output,
        })
    }

    pub fn new_state(&self) -> Qwen35TextRuntimeState {
        Qwen35TextRuntimeState {
            layers: self.layers.iter().map(Qwen35Layer::new_state).collect(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.token_embeddings.hidden_size()
    }

    pub fn forward_token_id_at(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        self.forward_input_embedding_at(&hidden, position_ids, state)
    }

    pub fn forward_token_id_hidden_at(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        self.forward_input_embedding_hidden_at(&hidden, position_ids, state)
    }

    pub fn forward_input_embedding_at(
        &self,
        input_embedding: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_input_embedding_hidden_at(input_embedding, position_ids, state)?;
        self.forward_hidden_to_logits(&hidden)
    }

    pub fn forward_input_embedding_hidden_at(
        &self,
        input_embedding: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let mut hidden = input_embedding.clone();
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            hidden = layer.forward(&hidden, layer_state, position_ids)?;
        }
        Ok(hidden)
    }

    pub fn prefill_token_ids(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen35TextRuntimeState,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 prefill span mismatch: {} token ids for {} position ids",
                token_ids.len(),
                position_ids.len()
            )));
        }

        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        let embeddings = self.token_embeddings.forward(&input)?;
        let mut last_hidden = None;
        for (idx, &position_id) in position_ids.iter().enumerate() {
            let hidden = embeddings.narrow(1, idx, 1)?;
            last_hidden =
                Some(self.forward_input_embedding_hidden_at(&hidden, position_id, state)?);
        }

        if !compute_logits {
            return Ok(None);
        }

        let last_hidden = last_hidden.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 prefill span produced no hidden state".to_string())
        })?;
        self.forward_hidden_to_logits(&last_hidden).map(Some)
    }

    pub fn forward_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.output_norm.forward(hidden)?;
        let logits = self.output.forward(&hidden)?;
        logits.i((0, 0)).map_err(Error::from)
    }
}

impl Qwen35Layer {
    fn new_state(&self) -> Qwen35LayerRuntimeState {
        match self.mixer {
            Qwen35Mixer::Linear(_) => Qwen35LayerRuntimeState::Linear {
                conv_state: None,
                recurrent_state: None,
            },
            Qwen35Mixer::Full(_) => Qwen35LayerRuntimeState::Full {
                k_pages: Vec::new(),
                v_pages: Vec::new(),
            },
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
        position_ids: [usize; 3],
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.attn_norm.forward(hidden_states)?;
        let hidden_states = match &self.mixer {
            Qwen35Mixer::Linear(mixer) => mixer.forward(&hidden_states, state)?,
            Qwen35Mixer::Full(mixer) => mixer.forward(&hidden_states, state, position_ids)?,
        };
        let hidden_states = (&residual + &hidden_states)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_norm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(Error::from)
    }
}

impl Qwen35Mlp {
    fn load(loader: &GgufLoader, device: &Device, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: load_qmatmul(loader, device, &format!("{prefix}.ffn_gate.weight"))?,
            up: load_qmatmul(loader, device, &format!("{prefix}.ffn_up.weight"))?,
            down: load_qmatmul(loader, device, &format!("{prefix}.ffn_down.weight"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = ops::silu(&self.gate.forward(hidden_states)?)?;
        let up = self.up.forward(hidden_states)?;
        self.down.forward(&(&gate * &up)?).map_err(Error::from)
    }
}

impl Qwen35FullAttention {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        prefix: &str,
        cfg: &Qwen35TextConfig,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_q.weight"))?,
            k_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_k.weight"))?,
            v_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_v.weight"))?,
            o_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_output.weight"))?,
            q_norm: load_rms_norm(loader, device, &format!("{prefix}.attn_q_norm.weight"), cfg)?,
            k_norm: load_rms_norm(loader, device, &format!("{prefix}.attn_k_norm.weight"), cfg)?,
            num_heads: cfg.attention_head_count,
            num_kv_heads: cfg.attention_head_count_kv,
            head_dim: cfg.attention_key_length,
            rope_dim: cfg.rope_dimension_count.min(cfg.attention_key_length),
            rope_theta: cfg.rope_freq_base,
            mrope_sections: cfg
                .rope_dimension_sections
                .iter()
                .copied()
                .filter(|section| *section > 0)
                .take(3)
                .collect(),
            kv_page_size: default_kv_page_size(),
            kv_quantization: default_kv_quantization(),
            rope_inv_freqs: build_rope_inv_freqs(
                cfg.rope_dimension_count.min(cfg.attention_key_length),
                cfg.rope_freq_base,
            )?,
            rope_cache: Mutex::new(HashMap::new()),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
        position_ids: [usize; 3],
    ) -> Result<Tensor> {
        let (k_pages, v_pages) = match state {
            Qwen35LayerRuntimeState::Full { k_pages, v_pages } => (k_pages, v_pages),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match full-attention layer".to_string(),
                ))
            }
        };

        let q_proj = self.q_proj.forward(hidden_states)?.reshape((
            1,
            1,
            self.num_heads,
            self.head_dim * 2,
        ))?;
        let query_states = q_proj.narrow(3, 0, self.head_dim)?;
        let gate = q_proj.narrow(3, self.head_dim, self.head_dim)?.reshape((
            1,
            1,
            self.num_heads * self.head_dim,
        ))?;
        let key_states = self.k_proj.forward(hidden_states)?.reshape((
            1,
            1,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let value_states = self.v_proj.forward(hidden_states)?.reshape((
            1,
            1,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
        let (query_states, key_states) =
            self.apply_rope(&query_states, &key_states, position_ids)?;

        let key_states = repeat_kv(&key_states, self.num_heads, self.num_kv_heads)?;
        let value_states = repeat_kv(&value_states, self.num_heads, self.num_kv_heads)?;
        append_to_pages(
            self.kv_page_size,
            k_pages,
            &key_states,
            self.kv_quantization,
        )?;
        append_to_pages(
            self.kv_page_size,
            v_pages,
            &value_states,
            self.kv_quantization,
        )?;

        let attn_output = if query_states.dim(1)? == 1 && !k_pages.is_empty() && !v_pages.is_empty()
        {
            paged_decode_attention(
                &query_states,
                k_pages,
                v_pages,
                self.num_heads,
                self.head_dim,
            )?
            .reshape((1, 1, self.num_heads * self.head_dim))?
        } else {
            let key_states = materialize_pages(k_pages)?;
            let value_states = materialize_pages(v_pages)?;
            let query_states = query_states.transpose(1, 2)?.contiguous()?;
            let key_states = key_states.transpose(1, 2)?.contiguous()?;
            let value_states = value_states.transpose(1, 2)?.contiguous()?;
            let attn_output = if let Some(out) = try_fused_self_attention(
                &query_states,
                &key_states,
                &value_states,
                None,
                self.head_dim,
                true,
            )? {
                out
            } else {
                let key_states_t = key_states.transpose(2, 3)?.contiguous()?;
                let attn = query_states.matmul(&key_states_t)?;
                let attn = (attn / (self.head_dim as f64).sqrt())?;
                let attn = ops::softmax_last_dim(&attn)?;
                attn.contiguous()?.matmul(&value_states)?
            };
            attn_output
                .transpose(1, 2)?
                .reshape((1, 1, self.num_heads * self.head_dim))?
        };
        let attn_output = (&attn_output * &ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&attn_output).map_err(Error::from)
    }

    fn apply_rope(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        position_ids: [usize; 3],
    ) -> Result<(Tensor, Tensor)> {
        if self.rope_dim == 0 {
            return Ok((query_states.clone(), key_states.clone()));
        }

        let (cos, sin) =
            self.cached_mrope(position_ids, query_states.device(), query_states.dtype())?;

        let query_rot = query_states.narrow(3, 0, self.rope_dim)?;
        let key_rot = key_states.narrow(3, 0, self.rope_dim)?;
        let query_rot = apply_rotary_emb(&query_rot, &cos, &sin)?;
        let key_rot = apply_rotary_emb(&key_rot, &cos, &sin)?;

        if self.rope_dim == self.head_dim {
            return Ok((query_rot, key_rot));
        }

        let query_pass = query_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        let key_pass = key_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        Ok((
            Tensor::cat(&[&query_rot, &query_pass], 3)?,
            Tensor::cat(&[&key_rot, &key_pass], 3)?,
        ))
    }

    fn cached_mrope(
        &self,
        position_ids: [usize; 3],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        if let Ok(cache) = self.rope_cache.lock() {
            if let Some((cos, sin)) = cache.get(&position_ids) {
                return Ok((cos.clone(), sin.clone()));
            }
        }

        let (cos, sin) = build_mrope(
            self.rope_dim,
            position_ids,
            &self.mrope_sections,
            &self.rope_inv_freqs,
            device,
            dtype,
        )?;
        if let Ok(mut cache) = self.rope_cache.lock() {
            cache.insert(position_ids, (cos.clone(), sin.clone()));
        }
        Ok((cos, sin))
    }
}

impl Qwen35LinearAttention {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        prefix: &str,
        cfg: &Qwen35TextConfig,
    ) -> Result<Self> {
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let head_k_dim = cfg.ssm_state_size;
        let head_v_dim = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

        let dt_bias_name = if loader.has_tensor(&format!("{prefix}.ssm_dt.bias")) {
            format!("{prefix}.ssm_dt.bias")
        } else {
            format!("{prefix}.ssm_dt")
        };
        let dt_bias = load_vector(loader, device, &dt_bias_name, num_v_heads)?.reshape((
            1,
            1,
            num_v_heads,
        ))?;
        let a = load_vector(loader, device, &format!("{prefix}.ssm_a"), num_v_heads)?.reshape((
            1,
            1,
            num_v_heads,
        ))?;
        let conv_kernel = normalize_conv_kernel(
            load_dense(
                loader,
                device,
                &format!("{prefix}.ssm_conv1d.weight"),
                Some(DType::F32),
            )?,
            conv_dim,
            cfg.ssm_conv_kernel,
        )?;
        let norm = Qwen35GatedRmsNorm {
            weight: load_vector(
                loader,
                device,
                &format!("{prefix}.ssm_norm.weight"),
                head_v_dim,
            )?,
            eps: cfg.attention_layer_norm_rms_epsilon,
        };

        Ok(Self {
            qkv_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_qkv.weight"))?,
            gate_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_gate.weight"))?,
            beta_proj: load_qmatmul(loader, device, &format!("{prefix}.ssm_beta.weight"))?,
            alpha_proj: load_qmatmul(loader, device, &format!("{prefix}.ssm_alpha.weight"))?,
            dt_bias,
            a,
            conv_kernel,
            norm,
            out_proj: load_qmatmul(loader, device, &format!("{prefix}.ssm_out.weight"))?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            kernel_size: cfg.ssm_conv_kernel,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
    ) -> Result<Tensor> {
        let (conv_state, recurrent_state) = match state {
            Qwen35LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        let mixed_qkv = self.qkv_proj.forward(hidden_states)?;
        let z = self.gate_proj.forward(hidden_states)?;
        let beta = ops::sigmoid(&self.beta_proj.forward(hidden_states)?)?;
        let alpha = self.alpha_proj.forward(hidden_states)?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        let mixed_qkv = self.depthwise_conv_step(&mixed_qkv, conv_state)?;

        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let query =
            mixed_qkv
                .narrow(2, 0, key_width)?
                .reshape((1, self.num_k_heads, self.head_k_dim))?;
        let key = mixed_qkv.narrow(2, key_width, key_width)?.reshape((
            1,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let value = mixed_qkv.narrow(2, key_width * 2, value_width)?.reshape((
            1,
            self.num_v_heads,
            self.head_v_dim,
        ))?;

        let mut query = l2norm(&query, 1e-6)?;
        let mut key = l2norm(&key, 1e-6)?;
        if self.num_v_heads != self.num_k_heads {
            if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
                return Err(Error::InferenceError(format!(
                    "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                    self.num_v_heads, self.num_k_heads
                )));
            }
            let repeats = self.num_v_heads / self.num_k_heads;
            query = repeat_head_states(&query, repeats)?;
            key = repeat_head_states(&key, repeats)?;
        }

        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                value.dtype(),
                value.device(),
            )?
        };

        let beta = beta.reshape((1, self.num_v_heads))?;
        let g = g.reshape((1, self.num_v_heads))?;
        let (output, next_state) =
            recurrent_gated_delta(&query, &key, &value, &g, &beta, current_state)?;
        *recurrent_state = Some(next_state);

        let output = output.reshape((self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output.reshape((1, 1, self.num_v_heads * self.head_v_dim))?;
        self.out_proj.forward(&output).map_err(Error::from)
    }

    fn depthwise_conv_step(
        &self,
        mixed_qkv: &Tensor,
        conv_state: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let current = mixed_qkv.i((0, 0))?;
        let current = if current.dtype() != self.conv_kernel.dtype() {
            current.to_dtype(self.conv_kernel.dtype())?
        } else {
            current
        };
        let current = current.reshape((self.conv_dim, 1))?;

        let window = if self.kernel_size <= 1 {
            current
        } else {
            let previous = if let Some(state) = conv_state.take() {
                state
            } else {
                Tensor::zeros(
                    (self.conv_dim, self.kernel_size - 1),
                    self.conv_kernel.dtype(),
                    self.conv_kernel.device(),
                )?
            };
            let window = Tensor::cat(&[&previous, &current], 1)?;
            *conv_state = Some(window.narrow(1, 1, self.kernel_size - 1)?);
            window
        };

        let convolved = (&window * &self.conv_kernel)?.sum(D::Minus1)?;
        let convolved = ops::silu(&convolved)?;
        convolved
            .reshape((1, 1, self.conv_dim))
            .map_err(Error::from)
    }
}

impl Qwen35GatedRmsNorm {
    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let normalized = candle_nn::ops::rms_norm(hidden_states, &self.weight, self.eps as f32)?;
        (&normalized * &ops::silu(gate)?).map_err(Error::from)
    }
}

fn is_full_attention_layer(layer_idx: usize, full_attention_interval: usize) -> bool {
    full_attention_interval > 0 && (layer_idx + 1).is_multiple_of(full_attention_interval)
}

fn load_qmatmul(loader: &GgufLoader, device: &Device, name: &str) -> Result<QMatMul> {
    let weights = Arc::new(loader.load_qtensor(name, device)?);
    QMatMul::from_weights(weights).map_err(Error::from)
}

fn load_rms_norm(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    cfg: &Qwen35TextConfig,
) -> Result<RmsNorm> {
    RmsNorm::from_qtensor(
        loader.load_qtensor(name, device)?,
        cfg.attention_layer_norm_rms_epsilon,
    )
    .map_err(Error::from)
}

fn load_dense(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    dtype: Option<DType>,
) -> Result<Tensor> {
    let mut tensor = loader
        .load_qtensor(name, device)?
        .dequantize(device)
        .map_err(Error::from)?;
    if let Some(dtype) = dtype {
        if tensor.dtype() != dtype {
            tensor = tensor.to_dtype(dtype)?;
        }
    }
    Ok(tensor)
}

fn load_vector(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    expected_len: usize,
) -> Result<Tensor> {
    let tensor = load_dense(loader, device, name, Some(DType::F32))?;
    let actual_len = tensor.elem_count();
    if actual_len != expected_len {
        return Err(Error::ModelLoadError(format!(
            "Unexpected tensor size for {name}: expected {expected_len} elements, found {actual_len}"
        )));
    }
    tensor.reshape((expected_len,)).map_err(Error::from)
}

fn normalize_conv_kernel(
    tensor: Tensor,
    expected_channels: usize,
    expected_kernel: usize,
) -> Result<Tensor> {
    match tensor.rank() {
        2 => {
            let (d0, d1) = tensor.dims2()?;
            if d0 == expected_channels && d1 == expected_kernel {
                Ok(tensor)
            } else if d0 == expected_kernel && d1 == expected_channels {
                tensor.transpose(0, 1)?.contiguous().map_err(Error::from)
            } else {
                Err(Error::ModelLoadError(format!(
                    "Unexpected Qwen3.5 conv kernel shape: ({d0}, {d1}) for expected ({expected_channels}, {expected_kernel})"
                )))
            }
        }
        3 => {
            let dims = tensor.dims();
            if dims == [expected_channels, 1, expected_kernel] {
                tensor.squeeze(1).map_err(Error::from)
            } else if dims == [expected_kernel, 1, expected_channels] {
                tensor
                    .squeeze(1)?
                    .transpose(0, 1)?
                    .contiguous()
                    .map_err(Error::from)
            } else {
                Err(Error::ModelLoadError(format!(
                    "Unexpected rank-3 Qwen3.5 conv kernel shape: {:?}",
                    dims
                )))
            }
        }
        rank => Err(Error::ModelLoadError(format!(
            "Unexpected Qwen3.5 conv kernel rank {rank}"
        ))),
    }
}

fn build_mrope(
    rope_dim: usize,
    position_ids: [usize; 3],
    mrope_sections: &[usize],
    inv_freqs: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = rope_dim / 2;
    if inv_freqs.len() != half_dim {
        return Err(Error::InferenceError(format!(
            "Invalid Qwen3.5 rotary dimension {rope_dim}"
        )));
    }

    let mut temporal = vec![0f32; half_dim];
    let mut height = vec![0f32; half_dim];
    let mut width = vec![0f32; half_dim];
    for (idx, inv_freq) in inv_freqs.iter().enumerate() {
        temporal[idx] = position_ids[0] as f32 * inv_freq;
        height[idx] = position_ids[1] as f32 * inv_freq;
        width[idx] = position_ids[2] as f32 * inv_freq;
    }

    let mut interleaved = temporal.clone();
    if position_ids[0] != position_ids[1] || position_ids[0] != position_ids[2] {
        if mrope_sections.iter().sum::<usize>() != half_dim || mrope_sections.len() < 3 {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.5 multimodal RoPE sections {:?} for rotary dim {}",
                mrope_sections, rope_dim
            )));
        }

        for (offset, source, section_len) in [
            (1usize, &height, mrope_sections[1]),
            (2usize, &width, mrope_sections[2]),
        ] {
            let stop = section_len * 3;
            for idx in (offset..stop.min(half_dim)).step_by(3) {
                interleaved[idx] = source[idx];
            }
        }
    }

    let mut emb = Vec::with_capacity(rope_dim);
    emb.extend_from_slice(&interleaved);
    emb.extend_from_slice(&interleaved);
    let emb = Tensor::from_vec(emb, (1, 1, 1, rope_dim), device)?.to_dtype(dtype)?;
    Ok((emb.cos()?, emb.sin()?))
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
    x.broadcast_mul(cos)?
        .broadcast_add(&rotated.broadcast_mul(sin)?)
        .map_err(Error::from)
}

fn build_rope_inv_freqs(rope_dim: usize, rope_theta: f64) -> Result<Vec<f32>> {
    let half_dim = rope_dim / 2;
    let inv_freqs: Vec<f32> = (0..rope_dim)
        .step_by(2)
        .map(|idx| (1.0f64 / rope_theta.powf(idx as f64 / rope_dim as f64)) as f32)
        .collect();
    if inv_freqs.len() != half_dim {
        return Err(Error::InferenceError(format!(
            "Invalid Qwen3.5 rotary dimension {rope_dim}"
        )));
    }
    Ok(inv_freqs)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0)?.log().map_err(Error::from)
}

fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    // Try fused Metal kernel first for F32 tensors
    if x.dtype() == DType::F32 {
        if let Some(result) = try_fused_l2_norm(x, eps) {
            return Ok(result);
        }
    }

    // Fallback to standard implementation
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?)
        .map_err(Error::from)
}

fn repeat_head_states(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, heads, dim) = x.dims3()?;
    // Match llama.cpp's tiled repeat layout for Qwen3.5 linear attention:
    // [h0, h1, ...] -> [h0, h1, ..., h0, h1, ...].
    let expanded = x.unsqueeze(1)?.broadcast_as((batch, repeats, heads, dim))?;
    expanded
        .reshape((batch, repeats * heads, dim))
        .map_err(Error::from)
}

fn recurrent_gated_delta(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
) -> Result<(Tensor, Tensor)> {
    // Try fused Metal kernel first (for F32 on Metal devices)
    if query.dtype() == DType::F32 {
        if let Some(result) = try_fused_gated_delta_recurrent(query, key, value, g, beta, &state) {
            return Ok(result);
        }
    }

    // Fallback to original implementation
    let dim = query.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let query = (query * scale)?;
    let g = g.exp()?.reshape((1, g.dim(1)?, 1, 1))?;
    let beta = beta.reshape((1, beta.dim(1)?, 1))?;

    let state = state.broadcast_mul(&g)?;
    let kv_mem = state.broadcast_mul(&key.unsqueeze(3)?)?.sum(2)?;
    let delta = (value - &kv_mem)?.broadcast_mul(&beta)?;
    let state = (&state + &key.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;
    let output = state.broadcast_mul(&query.unsqueeze(3)?)?.sum(2)?;
    Ok((output, state))
}

#[cfg(test)]
mod tests {
    use super::repeat_head_states;
    use candle_core::{Device, Tensor};

    #[test]
    fn repeat_head_states_uses_tiled_order() {
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu)
            .expect("tensor should build");
        let repeated = repeat_head_states(&x, 2).expect("repeat should succeed");
        let values = repeated.to_vec3::<f32>().expect("values");

        assert_eq!(
            values,
            vec![vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0]
            ]]
        );
    }
}
