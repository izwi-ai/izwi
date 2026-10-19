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
use crate::models::shared::memory::exact::exact_zeros;

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

/// Query rows the dense attention scores at a time. A 30 s encoder window
/// scored all 1,500 rows at once: `[20, 1500, 1500]` scores in the model
/// dtype, their F32 copy, the F32 softmax and the weights back in the model
/// dtype, held in 0.77 GB of Metal pool buffers. Each row's softmax reads
/// only its own scores, so blocks of rows compute the same values.
const DENSE_ATTENTION_QUERY_BLOCK: usize = 256;

fn whisper_metal_sdpa_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_WHISPER_METAL_SDPA", false));
    ENABLED.get()
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
        // Joined at its exact size (`Tensor::cat` rounds the joined weight
        // up to a power of two on Metal), then the parts are dropped and the
        // device synchronized, so Candle frees their buffers before a later
        // weight can inherit one.
        let weight = crate::models::shared::memory::exact::exact_concat_rows(&[
            &q_weight, &k_weight, &v_weight,
        ])?;
        let bias = Tensor::cat(&[&q_bias, &k_bias, &v_bias], 0)?;
        drop((q_weight, k_weight, v_weight));
        weight.device().synchronize()?;
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
        let (batch, heads, rows, _) = q.dims4()?;
        if rows <= DENSE_ATTENTION_QUERY_BLOCK {
            return self.attend_rows(&q, &k, v);
        }
        // The blocks are written into an exact-size output and dropped, so
        // no finished block pins a pooled buffer while the next one runs.
        let output = exact_zeros((batch, rows, heads * v.dim(3)?), v.dtype(), v.device())?;
        for first in (0..rows).step_by(DENSE_ATTENTION_QUERY_BLOCK) {
            let block = DENSE_ATTENTION_QUERY_BLOCK.min(rows - first);
            let wv = self.attend_rows(&q.narrow(2, first, block)?.contiguous()?, &k, v)?;
            output.slice_set(&wv, 1, first)?;
        }
        Ok(output)
    }

    /// The scaled query rows `q` (`[B, H, rows, D]`) attending over every
    /// key, token-major: `[B, rows, H·D]`.
    fn attend_rows(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let qk = {
            let _enter = self.matmul_span.enter();
            q.matmul(k)?
        };
        let w = {
            let _enter = self.softmax_span.enter();
            attention_weights(&qk, v.dtype())?
        };
        let wv = {
            let _enter = self.matmul_span.enter();
            w.matmul(v)?
        }
        .transpose(1, 2)?
        .flatten_from(2)?
        .contiguous()?;
        Ok(wv)
    }

    /// Dense cross attention of `x` over `audio_features`, without
    /// lifecycle state (the load-time numeric probe).
    fn forward_dense_cross_attention(&self, x: &Tensor, audio_features: &Tensor) -> Result<Tensor> {
        let AttentionProjections::Cross { query, key, value } = &self.projections else {
            return Err(Error::InferenceError(
                "Whisper self-attention projections were used for cross attention".into(),
            ));
        };
        let q = reshape_head_major(&query.forward(x)?, self.n_head)?;
        let k = reshape_head_major(&key.forward(audio_features)?, self.n_head)?.contiguous()?;
        let v = reshape_head_major(&value.forward(audio_features)?, self.n_head)?.contiguous()?;
        let wv = self.dense_attention(&q, &k, &v)?;
        Ok(self.out.forward(&wv)?)
    }
}

/// The attention softmax on F32 scores whatever the model dtype, as OpenAI
/// Whisper (`qk.float()`) and mlx-whisper (`precise=True`) do: Candle's
/// Metal softmax keeps its running max in the input dtype and its CPU
/// softmax sums in it. The weights return in `dtype`.
fn attention_weights(qk: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(candle_nn::ops::softmax_last_dim(&qk.to_dtype(DType::F32)?)?.to_dtype(dtype)?)
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
        let mlp = self.mlp_linear2.forward(&gelu(
            &self.mlp_linear1.forward(&self.mlp_ln.forward(&x)?)?,
        )?)?;
        Ok((x + mlp)?)
    }

    /// One decoder token through this block without lifecycle state: with a
    /// single token, causal self attention is attention over itself.
    fn forward_decoder_dense_single(&self, x: &Tensor, audio_features: &Tensor) -> Result<Tensor> {
        if x.dim(1)? != 1 {
            return Err(Error::InvalidInput(
                "Whisper dense decoder probe takes exactly one token".into(),
            ));
        }
        let (cross_attn, cross_ln) = self.cross_attn.as_ref().ok_or_else(|| {
            Error::InvalidInput("Whisper decoder layer has no cross attention".into())
        })?;
        let attn = self
            .attn
            .forward_dense_self_attention(&self.attn_ln.forward(x)?)?;
        let x = (x + attn)?;
        let cross =
            cross_attn.forward_dense_cross_attention(&cross_ln.forward(&x)?, audio_features)?;
        let x = (x + cross)?;
        let mlp = self.mlp_linear2.forward(&gelu(
            &self.mlp_linear1.forward(&self.mlp_ln.forward(&x)?)?,
        )?)?;
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

        let mlp = self.mlp_linear2.forward(&gelu(
            &self.mlp_linear1.forward(&self.mlp_ln.forward(&x)?)?,
        )?)?;
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

        let mlp = self.mlp_linear2.forward(&gelu(
            &self.mlp_linear1.forward(&self.mlp_ln.forward(&x)?)?,
        )?)?;
        Ok((x + mlp)?)
    }
}

/// Exact (erf) GELU, as OpenAI Whisper's `nn.GELU()`/`F.gelu` and HF's
/// `ACT2FN["gelu"]` compute it. Candle's `gelu()` is the tanh approximation.
fn gelu(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
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
        self.forward_observed(x, &mut |_, _| Ok(()))
    }

    /// [`Self::forward`], passing each block's output to `observe` (the
    /// load-time numeric self-check reads the residual stream per layer).
    pub(crate) fn forward_observed(
        &self,
        x: &Tensor,
        observe: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = {
            let _enter = self.conv1_span.enter();
            gelu(&self.conv1.forward(x)?)?
        };
        let x = {
            let _enter = self.conv2_span.enter();
            gelu(&self.conv2.forward(&x)?)?
        };
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.dims3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let positional_embedding = to_add_dtype(positional_embedding, x.dtype())?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward_encoder(&x)?;
            observe(index, &x)?;
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

    /// One decoder step for `token` at position 0 over `audio_features`
    /// (`[1, frames, d_model]`), without lifecycle state, passing each
    /// block's output to `observe`. Returns the F32 logits `[1, 1, vocab]`.
    /// The load-time numeric self-check and the precision parity tests use
    /// it; serving decodes through the physical caches.
    pub(crate) fn forward_dense_single_step(
        &self,
        token: u32,
        audio_features: &Tensor,
        observe: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let tokens = Tensor::new(&[[token]], audio_features.device())?;
        let token_embedding = self.token_embedding.forward(&tokens)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, 1)?;
        let positional_embedding = to_add_dtype(positional_embedding, token_embedding.dtype())?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward_decoder_dense_single(&x, audio_features)?;
            observe(index, &x)?;
        }
        self.final_linear(&self.ln.forward(&x)?)
    }

    /// Logits in F32 whatever the model dtype, as OpenAI Whisper's
    /// `.float()`: the matmul accumulates in F32 on every backend, and the
    /// sampler, the suppression masks and the log-probabilities read F32.
    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let b_size = x.dim(0)?;
        let w = self.token_embedding.embeddings().broadcast_left(b_size)?;
        let logits = {
            let _enter = self.span_final.enter();
            x.matmul(&w.t()?)?
        };
        Ok(logits.to_dtype(DType::F32)?)
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

    /// The load-time numeric probe: `mel` (`[1, mel_bins, frames]`, in the
    /// model dtype) through the encoder, then one decoder step for `token`
    /// over its output. Returns each encoder and decoder block's output, the
    /// encoder output and the F32 logits, copied to the host as they are
    /// produced so the probe holds no device memory per layer.
    pub(crate) fn numeric_probe(&self, mel: &Tensor, token: u32) -> Result<Vec<(String, Tensor)>> {
        let mut outputs = Vec::new();
        let host = |x: &Tensor| x.to_device(&Device::Cpu).map_err(Error::from);
        let features = self.encoder.forward_observed(mel, &mut |index, x| {
            outputs.push((format!("encoder.layers.{index}"), host(x)?));
            Ok(())
        })?;
        outputs.push(("encoder.output".into(), host(&features)?));
        let mut observe = |index: usize, x: &Tensor| {
            outputs.push((format!("decoder.layers.{index}"), host(x)?));
            Ok(())
        };
        let logits = self
            .decoder
            .forward_dense_single_step(token, &features, &mut observe)?;
        outputs.push(("logits".into(), host(&logits)?));
        Ok(outputs)
    }
}

/// A tiny random Whisper for precision tests: checkpoint-named F32
/// tensors with xorshift values, loadable at any dtype through
/// `VarBuilder::from_tensors`.
#[cfg(test)]
pub(crate) mod test_fixture {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};
    use candle_transformers::models::whisper::Config;

    use crate::kernels::numeric_check::xorshift_tensor;

    pub(crate) fn tiny_config() -> Config {
        Config {
            num_mel_bins: 16,
            max_source_positions: 24,
            d_model: 64,
            encoder_attention_heads: 4,
            encoder_layers: 2,
            vocab_size: 96,
            max_target_positions: 8,
            decoder_attention_heads: 4,
            decoder_layers: 2,
            suppress_tokens: Vec::new(),
        }
    }

    /// Projection weights in `[-weight_scale, weight_scale)`; LayerNorm
    /// gains near one and small biases, so activations stay O(1) at a
    /// weight scale of about 0.2.
    pub(crate) fn tiny_tensors(config: &Config, weight_scale: f32) -> HashMap<String, Tensor> {
        let d = config.d_model;
        let mut seed = 0u64;
        let mut tensors = HashMap::new();
        let mut put = |name: String, shape: &[usize], scale: f32, offset: f32| {
            seed += 1;
            let tensor = (xorshift_tensor(shape, seed, scale, &Device::Cpu).unwrap()
                + f64::from(offset))
            .unwrap();
            tensors.insert(name, tensor);
        };
        let linear = |put: &mut dyn FnMut(String, &[usize], f32, f32),
                      name: &str,
                      out: usize,
                      inp: usize,
                      bias: bool| {
            put(format!("{name}.weight"), &[out, inp], weight_scale, 0.0);
            if bias {
                put(format!("{name}.bias"), &[out], 0.05, 0.0);
            }
        };
        let layer_norm = |put: &mut dyn FnMut(String, &[usize], f32, f32), name: &str| {
            put(format!("{name}.weight"), &[d], 0.1, 1.0);
            put(format!("{name}.bias"), &[d], 0.05, 0.0);
        };
        let block = |put: &mut dyn FnMut(String, &[usize], f32, f32), prefix: &str, cross: bool| {
            for (proj, bias) in [("q_proj", true), ("k_proj", false), ("v_proj", true)] {
                linear(put, &format!("{prefix}.self_attn.{proj}"), d, d, bias);
            }
            linear(put, &format!("{prefix}.self_attn.out_proj"), d, d, true);
            layer_norm(put, &format!("{prefix}.self_attn_layer_norm"));
            if cross {
                for (proj, bias) in [
                    ("q_proj", true),
                    ("k_proj", false),
                    ("v_proj", true),
                    ("out_proj", true),
                ] {
                    linear(put, &format!("{prefix}.encoder_attn.{proj}"), d, d, bias);
                }
                layer_norm(put, &format!("{prefix}.encoder_attn_layer_norm"));
            }
            linear(put, &format!("{prefix}.fc1"), 4 * d, d, true);
            linear(put, &format!("{prefix}.fc2"), d, 4 * d, true);
            layer_norm(put, &format!("{prefix}.final_layer_norm"));
        };
        let encoder = "model.encoder";
        put(
            format!("{encoder}.conv1.weight"),
            &[d, config.num_mel_bins, 3],
            weight_scale,
            0.0,
        );
        put(format!("{encoder}.conv1.bias"), &[d], 0.05, 0.0);
        put(
            format!("{encoder}.conv2.weight"),
            &[d, d, 3],
            weight_scale,
            0.0,
        );
        put(format!("{encoder}.conv2.bias"), &[d], 0.05, 0.0);
        for layer in 0..config.encoder_layers {
            block(&mut put, &format!("{encoder}.layers.{layer}"), false);
        }
        layer_norm(&mut put, &format!("{encoder}.layer_norm"));
        let decoder = "model.decoder";
        put(
            format!("{decoder}.embed_tokens.weight"),
            &[config.vocab_size, d],
            1.0,
            0.0,
        );
        put(
            format!("{decoder}.embed_positions.weight"),
            &[config.max_target_positions, d],
            0.5,
            0.0,
        );
        for layer in 0..config.decoder_layers {
            block(&mut put, &format!("{decoder}.layers.{layer}"), true);
        }
        layer_norm(&mut put, &format!("{decoder}.layer_norm"));
        tensors
    }
}

#[cfg(test)]
mod tests {
    use super::{
        flatten_token_major, gelu, reshape_head_major, reshape_token_major, sinusoids,
        whisper_metal_sdpa_enabled,
    };
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn gelu_is_exact_erf_like_openai_whisper() {
        let x = Tensor::new(&[1.0f32, -1.0, 3.0], &Device::Cpu).expect("input");
        let y = gelu(&x).expect("gelu").to_vec1::<f32>().expect("values");
        // torch.nn.functional.gelu(torch.tensor([1., -1., 3.])), approximate="none".
        let expected = [0.841_344_7f32, -0.158_655_26, 2.995_950_2];
        for (got, want) in y.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "gelu {got} != {want}");
        }
    }

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
        let _guard = crate::env_test_lock().lock().expect("env lock");
        for (value, expected) in [
            ("1", true),
            ("true", true),
            ("YES", true),
            (" on ", true),
            ("", false),
            ("0", false),
            ("off", false),
        ] {
            std::env::set_var("IZWI_WHISPER_METAL_SDPA", value);
            assert_eq!(whisper_metal_sdpa_enabled(), expected, "{value:?}");
        }
        std::env::remove_var("IZWI_WHISPER_METAL_SDPA");
        assert!(!whisper_metal_sdpa_enabled());
    }

    /// The tiny model at `dtype` on `device`, and its probe on the shared
    /// xorshift mel: (name, host F32 values) per probed tensor.
    fn probe_at(dtype: DType, device: &Device) -> Vec<(String, Vec<f32>)> {
        use super::test_fixture::{tiny_config, tiny_tensors};
        use crate::kernels::numeric_check::{host_f32, xorshift_tensor};

        let config = tiny_config();
        let vb = candle_nn::VarBuilder::from_tensors(tiny_tensors(&config, 0.2), dtype, device);
        let whisper = super::Whisper::load(&vb, config.clone()).expect("tiny whisper");
        let frames = config.max_source_positions * 2;
        let mel = xorshift_tensor(&[1, config.num_mel_bins, frames], 7, 1.5, device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        whisper
            .numeric_probe(&mel, 3)
            .expect("probe")
            .into_iter()
            .map(|(name, tensor)| (name, host_f32(&tensor).unwrap()))
            .collect()
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap()
    }

    /// F16 weights and activations against F32 on the same weights: every
    /// block output and the encoder output within F16 rounding, the logits
    /// F32 with the same top-1.
    fn assert_f16_tracks_f32(f16: &[(String, Vec<f32>)], f32: &[(String, Vec<f32>)]) {
        use crate::kernels::numeric_check::relative_l2;

        assert_eq!(f16.len(), f32.len());
        // Two blocks each side, the encoder output and the logits.
        assert_eq!(f16.len(), 6);
        for ((name, low), (reference_name, reference)) in f16.iter().zip(f32) {
            assert_eq!(name, reference_name);
            let rel = relative_l2(low, reference);
            assert!(rel < 1e-2, "{name}: F16 vs F32 relative L2 {rel:e}");
        }
        let (_, logits) = f16.last().unwrap();
        let (_, reference) = f32.last().unwrap();
        assert_eq!(logits.len(), super::test_fixture::tiny_config().vocab_size);
        assert_eq!(argmax(logits), argmax(reference), "top-1 logit");
    }

    #[test]
    fn f16_whisper_tracks_f32_encoder_and_decoder_on_cpu() {
        let f32 = probe_at(DType::F32, &Device::Cpu);
        let f16 = probe_at(DType::F16, &Device::Cpu);
        assert_f16_tracks_f32(&f16, &f32);
        // F16 is not silently F32: the outputs differ by rounding.
        assert_ne!(f16[0].1, f32[0].1);
    }

    #[test]
    fn whisper_logits_are_f32_under_f16_weights() {
        use super::test_fixture::{tiny_config, tiny_tensors};
        use crate::kernels::numeric_check::xorshift_tensor;

        let config = tiny_config();
        let device = Device::Cpu;
        let vb =
            candle_nn::VarBuilder::from_tensors(tiny_tensors(&config, 0.2), DType::F16, &device);
        let whisper = super::Whisper::load(&vb, config.clone()).expect("tiny whisper");
        let features = xorshift_tensor(
            &[1, config.max_source_positions, config.d_model],
            9,
            1.0,
            &device,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
        let logits = whisper
            .decoder
            .forward_dense_single_step(3, &features, &mut |_, x| {
                assert_eq!(x.dtype(), DType::F16, "the residual stream stays F16");
                Ok(())
            })
            .unwrap();
        assert_eq!(logits.dtype(), DType::F32);
        assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
    }

    /// Dense attention scored by blocks of query rows returns what one pass
    /// over every row does, including a short last block, within F32
    /// rounding (the CPU matrix product blocks its sums by its row count).
    #[test]
    fn blocked_dense_attention_matches_one_pass() {
        use super::{MultiHeadAttention, DENSE_ATTENTION_QUERY_BLOCK};
        let device = Device::Cpu;
        let (heads, head_dim) = (4, 16);
        let attention = MultiHeadAttention::load_self(
            heads * head_dim,
            heads,
            candle_nn::VarBuilder::zeros(DType::F32, &device),
        )
        .unwrap();
        let rows = 2 * DENSE_ATTENTION_QUERY_BLOCK + 37;
        let hashed = |seed: usize, rows: usize| {
            let values = (0..heads * rows * head_dim)
                .map(|i| (((i * 2_654_435_761 + seed) % 1_000) as f32 / 500.0) - 1.0)
                .collect::<Vec<_>>();
            Tensor::from_vec(values, (1, heads, rows, head_dim), &device).unwrap()
        };
        let (q, k, v) = (hashed(1, rows), hashed(2, rows), hashed(3, rows));
        let blocked = attention.dense_attention(&q, &k, &v).unwrap();
        let scale = (head_dim as f64).powf(-0.25);
        let one_pass = attention
            .attend_rows(
                &(&q * scale).unwrap(),
                &(k.transpose(2, 3).unwrap() * scale).unwrap(),
                &v,
            )
            .unwrap();
        assert_eq!(blocked.dims(), &[1, rows, heads * head_dim]);
        assert_eq!(blocked.dims(), one_pass.dims());
        let difference = (blocked - one_pass)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(difference <= 1e-6, "{difference}");
    }

    /// A softmax over 4096 equal F16 scores: an F16 running sum stops at
    /// 2048 (2048 + 1 rounds back to 2048), which would double every
    /// attention weight. On F32 scores each weight is 1/4096, exact in F16.
    #[test]
    fn whisper_attention_softmax_runs_on_f32_scores() {
        let qk = Tensor::zeros((1, 1, 1, 4096), DType::F16, &Device::Cpu).unwrap();
        let weights = super::attention_weights(&qk, DType::F16).unwrap();
        assert_eq!(weights.dtype(), DType::F16);
        let values = weights
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|value| *value == 1.0 / 4096.0),
            "{:?}",
            &values[..4]
        );
    }

    /// The tiny model in F16 on the GPU: finite throughout, and within F16
    /// rounding of the CPU F32 reference with the same top-1 logit.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_f16_whisper_is_finite_and_tracks_cpu_f32() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_f16_whisper_is_finite_and_tracks_cpu_f32",
        ) else {
            return;
        };
        let f16 = probe_at(DType::F16, &device);
        for (name, values) in &f16 {
            assert!(values.iter().all(|value| value.is_finite()), "{name}");
        }
        assert_f16_tracks_f32(&f16, &probe_at(DType::F32, &Device::Cpu));
    }
}
