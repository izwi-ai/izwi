//! Whisper model shim adapted from Candle's implementation so generated
//! positional tensors follow the active Izwi model dtype and decoder attention
//! consumes lifecycle-owned physical state.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, Conv1d, Conv1dConfig, Embedding, LayerNorm, Module, VarBuilder};
use candle_transformers::models::whisper::Config;
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear};

use crate::backends::state::{StaticAttentionLayerValue, StaticAttentionRaggedRow};
use crate::engine::{
    InvocationStaticAttentionLease, RetainedStaticAttentionRuntimeV2,
    RetainedStaticAttentionSequenceId,
};
use crate::error::{Error, Result};
use crate::kv::v2::{DomainStepIntent, StateUpdateKind};
use crate::models::shared::attention::flash::try_fused_self_attention;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::memory::accounting::deep_copy_tensor_storage;

use super::physical::WHISPER_CROSS_STATE_DOMAIN;

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

fn to_add_dtype(tensor: Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        Ok(tensor.to_dtype(dtype)?)
    }
}

fn whisper_metal_sdpa_enabled() -> bool {
    std::env::var("IZWI_WHISPER_METAL_SDPA")
        .ok()
        .map(|value| whisper_metal_sdpa_env_value_enabled(&value))
        .unwrap_or(false)
}

fn whisper_metal_sdpa_env_value_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[derive(Debug, Clone)]
enum AttentionProjections {
    SelfQkv(Linear),
    Cross {
        query: Linear,
        key: Linear,
        value: Linear,
    },
}

#[derive(Debug, Clone)]
struct MultiHeadAttention {
    projections: AttentionProjections,
    out: Linear,
    n_head: usize,
    span: tracing::Span,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
}

impl MultiHeadAttention {
    fn load_self(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let projections = AttentionProjections::SelfQkv(Self::load_self_qkv(n_state, &vb)?);
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            projections,
            out,
            n_head,
            span,
            softmax_span,
            matmul_span,
        })
    }

    fn load_cross(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let projections = AttentionProjections::Cross { query, key, value };
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            projections,
            out,
            n_head,
            span,
            softmax_span,
            matmul_span,
        })
    }

    fn forward_dense_self_attention(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (q, k, v) = self.project_self_attention_qkv_head_major(x)?;
        let wv = self.dense_attention(&q, &k, &v)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn load_self_qkv(n_state: usize, vb: &VarBuilder) -> Result<Linear> {
        let q_weight = vb.pp("q_proj").get((n_state, n_state), "weight")?;
        let k_weight = vb.pp("k_proj").get((n_state, n_state), "weight")?;
        let v_weight = vb.pp("v_proj").get((n_state, n_state), "weight")?;
        let q_bias = vb.pp("q_proj").get(n_state, "bias")?;
        let k_bias = Tensor::zeros(n_state, q_bias.dtype(), q_bias.device())?;
        let v_bias = vb.pp("v_proj").get(n_state, "bias")?;
        let weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
        let bias = Tensor::cat(&[&q_bias, &k_bias, &v_bias], 0)?;
        Ok(Linear::from_weights(weight, Some(bias)))
    }

    fn project_self_attention_qkv_head_major(
        &self,
        x: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let AttentionProjections::SelfQkv(self_qkv) = &self.projections else {
            return Err(Error::InferenceError(
                "Whisper cross-attention projections were used for self attention".into(),
            ));
        };
        let qkv = self_qkv.forward(x)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let q = reshape_head_major(&chunks[0], self.n_head)?;
        let k = reshape_head_major(&chunks[1], self.n_head)?.contiguous()?;
        let v = reshape_head_major(&chunks[2], self.n_head)?.contiguous()?;
        Ok((q, k, v))
    }

    fn project_self_attention_qkv_token_major(
        &self,
        x: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let AttentionProjections::SelfQkv(self_qkv) = &self.projections else {
            return Err(Error::InferenceError(
                "Whisper cross-attention projections were used for self attention".into(),
            ));
        };
        let qkv = self_qkv.forward(x)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        Ok((
            reshape_token_major(&chunks[0], self.n_head)?,
            reshape_token_major(&chunks[1], self.n_head)?,
            reshape_token_major(&chunks[2], self.n_head)?,
        ))
    }

    fn project_cross_attention_query(&self, x: &Tensor) -> Result<Tensor> {
        let AttentionProjections::Cross { query, .. } = &self.projections else {
            return Err(Error::InferenceError(
                "Whisper self-attention projections were used for cross attention".into(),
            ));
        };
        reshape_token_major(&query.forward(x)?, self.n_head)
    }

    fn project_cross_attention_memory(
        &self,
        model_layer: u32,
        audio_features: &Tensor,
    ) -> Result<StaticAttentionLayerValue> {
        let AttentionProjections::Cross { key, value, .. } = &self.projections else {
            return Err(Error::InferenceError(
                "Whisper self-attention projections were used for cross attention".into(),
            ));
        };
        Ok(StaticAttentionLayerValue {
            model_layer,
            keys: reshape_token_major(&key.forward(audio_features)?, self.n_head)?,
            values: reshape_token_major(&value.forward(audio_features)?, self.n_head)?,
        })
    }

    fn project_output_token_major(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.out.forward(&flatten_token_major(x)?)?)
    }

    fn dense_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_, _, _, head_dim) = q.dims4()?;
        let scale = (head_dim as f64).powf(-0.25);

        let can_try_fused =
            q.device().is_cuda() || (q.device().is_metal() && whisper_metal_sdpa_enabled());
        if can_try_fused {
            if let Ok(Some(wv)) = try_fused_self_attention(q, k, v, None, head_dim, false) {
                return Ok(wv.transpose(1, 2)?.flatten_from(2)?);
            }
        }

        let q = (q * scale)?;
        let k = (k.transpose(2, 3)? * scale)?;
        let qk = {
            let _enter = self.matmul_span.enter();
            q.matmul(&k)?
        };
        let w = {
            let _enter = self.softmax_span.enter();
            candle_nn::ops::softmax_last_dim(&qk)?
        };
        let wv = {
            let _enter = self.matmul_span.enter();
            w.matmul(v)?
        }
        .transpose(1, 2)?
        .flatten_from(2)?;
        Ok(wv)
    }
}

fn reshape_head_major(x: &Tensor, n_head: usize) -> Result<Tensor> {
    let (n_batch, n_ctx, n_state) = x.dims3()?;
    if n_head == 0 || n_state % n_head != 0 {
        return Err(Error::InvalidInput(
            "Whisper attention hidden size is not divisible by its head count".into(),
        ));
    }
    x.reshape((n_batch, n_ctx, n_head, n_state / n_head))?
        .transpose(1, 2)
        .map_err(Error::from)
}

fn reshape_token_major(x: &Tensor, n_head: usize) -> Result<Tensor> {
    let (n_batch, n_ctx, n_state) = x.dims3()?;
    if n_batch != 1 {
        return Err(Error::InvalidInput(format!(
            "Whisper physical attention requires batch size 1, got {n_batch}"
        )));
    }
    if n_head == 0 || n_state % n_head != 0 {
        return Err(Error::InvalidInput(
            "Whisper attention hidden size is not divisible by its head count".into(),
        ));
    }
    x.reshape((n_ctx, n_head, n_state / n_head))?
        .contiguous()
        .map_err(Error::from)
}

fn flatten_token_major(x: &Tensor) -> Result<Tensor> {
    let (n_ctx, n_head, head_dim) = x.dims3()?;
    let n_state = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| Error::InvalidInput("Whisper attention width overflow".into()))?;
    x.reshape((1, n_ctx, n_state)).map_err(Error::from)
}

#[derive(Debug, Clone)]
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
    span: tracing::Span,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, ca: bool, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load_self(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if ca {
            let cross_attn =
                MultiHeadAttention::load_cross(n_state, n_head, vb.pp("encoder_attn"))?;
            let cross_attn_ln = layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
            span,
        })
    }

    fn forward_encoder(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attn = self
            .attn
            .forward_dense_self_attention(&self.attn_ln.forward(x)?)?;
        let x = (x + attn)?;
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        Ok((x + mlp)?)
    }

    fn project_cross_attention_memory(
        &self,
        model_layer: u32,
        audio_features: &Tensor,
    ) -> Result<StaticAttentionLayerValue> {
        let (cross_attn, _) = self.cross_attn.as_ref().ok_or_else(|| {
            Error::InvalidInput("Whisper decoder layer has no cross-attention projection".into())
        })?;
        let projected = cross_attn.project_cross_attention_memory(model_layer, audio_features)?;
        Ok(StaticAttentionLayerValue {
            model_layer: projected.model_layer,
            keys: deep_copy_tensor_storage(&projected.keys)?,
            values: deep_copy_tensor_storage(&projected.values)?,
        })
    }

    fn forward_decoder_physical(
        &self,
        model_layer: usize,
        x: &Tensor,
        self_kv: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        cross_kv: &InvocationStaticAttentionLease,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_input = self.attn_ln.forward(x)?;
        let (q, k, v) = self
            .attn
            .project_self_attention_qkv_token_major(&self_input)?;
        let head_dim = q.dim(2)?;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let attended = self_kv.write_and_attend(model_layer, prepared, &q, &k, &v, scale)?;
        let mut x = (x + self.attn.project_output_token_major(&attended)?)?;

        let (cross_attn, cross_ln) = self.cross_attn.as_ref().ok_or_else(|| {
            Error::InvalidInput("Whisper decoder layer has no cross attention".into())
        })?;
        let cross_query = cross_attn.project_cross_attention_query(&cross_ln.forward(&x)?)?;
        let query_len = u32::try_from(cross_query.dim(0)?)
            .map_err(|_| Error::InvalidInput("Whisper query length exceeds u32".into()))?;
        let model_layer = u32::try_from(model_layer)
            .map_err(|_| Error::InvalidInput("Whisper layer index exceeds u32".into()))?;
        let cross_attended = cross_kv.attend(
            model_layer,
            &cross_query,
            &[StaticAttentionRaggedRow {
                query_start: 0,
                query_len,
            }],
            scale,
        )?;
        x = (&x + cross_attn.project_output_token_major(&cross_attended)?)?;

        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        Ok((x + mlp)?)
    }

    fn forward_decoder_retained(
        &self,
        model_layer: usize,
        x: &Tensor,
        self_kv: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        cross_runtime: &RetainedStaticAttentionRuntimeV2,
        cross_sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_input = self.attn_ln.forward(x)?;
        let (q, k, v) = self
            .attn
            .project_self_attention_qkv_token_major(&self_input)?;
        let head_dim = q.dim(2)?;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let attended = self_kv.write_and_attend(model_layer, prepared, &q, &k, &v, scale)?;
        let mut x = (x + self.attn.project_output_token_major(&attended)?)?;

        let (cross_attn, cross_ln) = self.cross_attn.as_ref().ok_or_else(|| {
            Error::InvalidInput("Whisper decoder layer has no cross attention".into())
        })?;
        let cross_query = cross_attn.project_cross_attention_query(&cross_ln.forward(&x)?)?;
        let query_len = u32::try_from(cross_query.dim(0)?)
            .map_err(|_| Error::InvalidInput("Whisper query length exceeds u32".into()))?;
        let model_layer = u32::try_from(model_layer)
            .map_err(|_| Error::InvalidInput("Whisper layer index exceeds u32".into()))?;
        let cross_attended = cross_runtime.attend(
            cross_sequence,
            model_layer,
            &cross_query,
            &[StaticAttentionRaggedRow {
                query_start: 0,
                query_len,
            }],
            scale,
        )?;
        x = (&x + cross_attn.project_output_token_major(&cross_attended)?)?;

        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        Ok((x + mlp)?)
    }
}

fn sinusoids(length: usize, channels: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    to_add_dtype(sincos, dtype)
}

#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    span: tracing::Span,
    conv1_span: tracing::Span,
    conv2_span: tracing::Span,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "audio-encoder");
        let conv1_span = tracing::span!(tracing::Level::TRACE, "conv1");
        let conv2_span = tracing::span!(tracing::Level::TRACE, "conv2");
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(n_ctx, n_state, vb.device(), vb.dtype())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            conv1_span,
            conv2_span,
            span,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = {
            let _enter = self.conv1_span.enter();
            self.conv1.forward(x)?.gelu()?
        };
        let x = {
            let _enter = self.conv2_span.enter();
            self.conv2.forward(&x)?.gelu()?
        };
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.dims3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let positional_embedding = to_add_dtype(positional_embedding, x.dtype())?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in &self.blocks {
            x = block.forward_encoder(&x)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }

    pub(crate) fn forward_batch(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _, frames) = x.dims3()?;
        if batch == 0 || frames == 0 {
            return Err(Error::InvalidInput(
                "Whisper encoder batch requires non-empty rows and frames".into(),
            ));
        }
        self.forward(x)
    }
}

#[derive(Debug, Clone)]
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    n_head: usize,
    span: tracing::Span,
    span_final: tracing::Span,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "text-decoder");
        let span_final = tracing::span!(tracing::Level::TRACE, "text-decoder-final");
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let token_embedding = embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, vb.pp("layer_norm"))?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            n_head,
            span,
            span_final,
        })
    }

    pub(crate) fn install_cross_attention_memory(
        &self,
        audio_features: &Tensor,
        source_identity: [u8; 32],
        cross_kv: &mut InvocationStaticAttentionLease,
    ) -> Result<()> {
        let (batch, memory_tokens, _) = audio_features.dims3()?;
        if batch != 1 || memory_tokens == 0 {
            return Err(Error::InvalidInput(format!(
                "Whisper cross attention requires one non-empty encoder row, got batch {batch} and {memory_tokens} tokens"
            )));
        }
        let target_cursor = u64::try_from(memory_tokens)
            .map_err(|_| Error::InvalidInput("Whisper encoder length exceeds u64".into()))?;
        let intent = DomainStepIntent {
            domain: WHISPER_CROSS_STATE_DOMAIN,
            expected_cursor: 0,
            target_cursor,
            update: StateUpdateKind::StaticInitialize {
                source_identity,
                components: Vec::new(),
            },
        };
        cross_kv.begin_install(&intent)?;
        for layer in self.prepare_cross_attention_memory(audio_features)? {
            cross_kv.install_layer(layer)?;
        }
        cross_kv.commit_install()
    }

    pub(crate) fn prepare_cross_attention_memory(
        &self,
        audio_features: &Tensor,
    ) -> Result<Vec<StaticAttentionLayerValue>> {
        let (batch, memory_tokens, _) = audio_features.dims3()?;
        if batch != 1 || memory_tokens == 0 {
            return Err(Error::InvalidInput(format!(
                "Whisper cross attention requires one non-empty encoder row, got batch {batch} and {memory_tokens} tokens"
            )));
        }
        self.blocks
            .iter()
            .enumerate()
            .map(|(model_layer, block)| {
                let model_layer = u32::try_from(model_layer)
                    .map_err(|_| Error::InvalidInput("Whisper layer index exceeds u32".into()))?;
                block.project_cross_attention_memory(model_layer, audio_features)
            })
            .collect()
    }

    pub(crate) fn forward_physical_at(
        &self,
        tokens: &Tensor,
        position_offset: usize,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &InvocationStaticAttentionLease,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (batch, token_count) = tokens.dims2()?;
        if batch != 1 || token_count == 0 {
            return Err(Error::InvalidInput(format!(
                "Whisper physical decoder requires one non-empty token row, got batch {batch} and {token_count} tokens"
            )));
        }
        if position_offset != self_kv.context_len() {
            return Err(Error::InvalidInput(format!(
                "Whisper decoder position {position_offset} does not match physical cache cursor {}",
                self_kv.context_len()
            )));
        }
        let hidden_size = self.token_embedding.embeddings().dim(1)?;
        if self.n_head == 0 || hidden_size % self.n_head != 0 {
            return Err(Error::InvalidInput(
                "Whisper decoder hidden size is not divisible by its head count".into(),
            ));
        }
        self_kv.validate_model(self.blocks.len(), self.n_head, hidden_size / self.n_head)?;
        if cross_kv.metadata()?.is_none() {
            return Err(Error::InvalidInput(
                "Whisper cross-attention memory is not installed".into(),
            ));
        }

        let position_end = position_offset
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("Whisper decoder position overflow".into()))?;
        if position_end > self.positional_embedding.dim(0)? {
            return Err(Error::InvalidInput(format!(
                "Whisper decoder position {position_end} exceeds its positional capacity"
            )));
        }
        let token_embedding = self.token_embedding.forward(tokens)?;
        let positional_embedding =
            self.positional_embedding
                .narrow(0, position_offset, token_count)?;
        let positional_embedding = to_add_dtype(positional_embedding, token_embedding.dtype())?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        let mut prepared = self_kv.prepare_append(position_offset, token_count)?;
        for (model_layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_decoder_physical(
                model_layer,
                &x,
                self_kv,
                &mut prepared,
                cross_kv,
            )?;
        }
        let x = self.ln.forward(&x)?;
        self_kv.commit_prepared(prepared)?;
        Ok(x)
    }

    pub(crate) fn forward_retained_at(
        &self,
        tokens: &Tensor,
        position_offset: usize,
        self_kv: &mut PhysicalPagedKvCache,
        cross_runtime: &RetainedStaticAttentionRuntimeV2,
        cross_sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (batch, token_count) = tokens.dims2()?;
        if batch != 1 || token_count == 0 {
            return Err(Error::InvalidInput(format!(
                "Whisper retained decoder requires one non-empty token row, got batch {batch} and {token_count} tokens"
            )));
        }
        if position_offset != self_kv.context_len() {
            return Err(Error::InvalidInput(format!(
                "Whisper decoder position {position_offset} does not match physical cache cursor {}",
                self_kv.context_len()
            )));
        }
        let hidden_size = self.token_embedding.embeddings().dim(1)?;
        if self.n_head == 0 || hidden_size % self.n_head != 0 {
            return Err(Error::InvalidInput(
                "Whisper decoder hidden size is not divisible by its head count".into(),
            ));
        }
        self_kv.validate_model(self.blocks.len(), self.n_head, hidden_size / self.n_head)?;
        if cross_runtime.read(cross_sequence)?.is_none() {
            return Err(Error::InvalidInput(
                "Whisper retained cross-attention memory is not installed".into(),
            ));
        }
        let position_end = position_offset
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("Whisper decoder position overflow".into()))?;
        if position_end > self.positional_embedding.dim(0)? {
            return Err(Error::InvalidInput(format!(
                "Whisper decoder position {position_end} exceeds its positional capacity"
            )));
        }
        let token_embedding = self.token_embedding.forward(tokens)?;
        let positional_embedding =
            self.positional_embedding
                .narrow(0, position_offset, token_count)?;
        let positional_embedding = to_add_dtype(positional_embedding, token_embedding.dtype())?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        let mut prepared = self_kv.prepare_append(position_offset, token_count)?;
        for (model_layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_decoder_retained(
                model_layer,
                &x,
                self_kv,
                &mut prepared,
                cross_runtime,
                cross_sequence,
            )?;
        }
        let x = self.ln.forward(&x)?;
        self_kv.commit_prepared(prepared)?;
        Ok(x)
    }

    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let b_size = x.dim(0)?;
        let w = self.token_embedding.embeddings().broadcast_left(b_size)?;
        let logits = {
            let _enter = self.span_final.enter();
            x.matmul(&w.t()?)?
        };
        Ok(logits)
    }
}

#[derive(Debug, Clone)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(vb: &VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        flatten_token_major, reshape_head_major, reshape_token_major, sinusoids,
        whisper_metal_sdpa_env_value_enabled,
    };
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn sinusoidal_embedding_uses_requested_dtype() {
        let device = Device::Cpu;
        let pos = sinusoids(4, 8, &device, DType::F16).expect("sinusoids");
        assert_eq!(pos.dtype(), DType::F16);

        let activations = Tensor::zeros((1, 4, 8), DType::F16, &device).expect("activations");
        activations
            .broadcast_add(&pos)
            .expect("same-dtype positional add");
    }

    #[test]
    fn attention_reshape_helpers_preserve_token_and_head_order() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            (0..24).map(|value| value as f32).collect::<Vec<_>>(),
            (1, 3, 8),
            &device,
        )
        .expect("input");
        let token_major = reshape_token_major(&input, 2).expect("token-major");
        assert_eq!(token_major.dims(), &[3, 2, 4]);
        let flattened = flatten_token_major(&token_major).expect("flattened");
        assert_eq!(flattened.dims(), input.dims());
        assert_eq!(
            flattened.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            input.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        let head_major = reshape_head_major(&input, 2).expect("head-major");
        assert_eq!(head_major.dims(), &[1, 2, 3, 4]);
        let head_major_values = head_major
            .transpose(1, 2)
            .unwrap()
            .reshape((1, 3, 8))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            head_major_values,
            input.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn token_major_reshape_rejects_non_scalar_batch() {
        let input = Tensor::zeros((2, 3, 8), DType::F32, &Device::Cpu).expect("input");
        assert!(reshape_token_major(&input, 2).is_err());
    }

    #[test]
    fn whisper_metal_sdpa_is_opt_in() {
        assert!(whisper_metal_sdpa_env_value_enabled("1"));
        assert!(whisper_metal_sdpa_env_value_enabled("true"));
        assert!(whisper_metal_sdpa_env_value_enabled("yes"));
        assert!(whisper_metal_sdpa_env_value_enabled("on"));
        assert!(!whisper_metal_sdpa_env_value_enabled(""));
        assert!(!whisper_metal_sdpa_env_value_enabled("0"));
        assert!(!whisper_metal_sdpa_env_value_enabled("false"));
    }
}
