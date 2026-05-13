//! Whisper model shim adapted from Candle's implementation so generated
//! positional/mask tensors follow the active Izwi model dtype.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, kv_cache::KvCache, Conv1d, Conv1dConfig, Embedding, LayerNorm, Module, VarBuilder,
};
use candle_transformers::models::whisper::Config;
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear};

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
        tensor.to_dtype(dtype)
    }
}

#[derive(Debug, Clone)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    span: tracing::Span,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
    self_kv_cache: Option<KvCache>,
    cross_kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    fn load(
        n_state: usize,
        n_head: usize,
        self_cache_len: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        let self_kv_cache = self_cache_len.map(|max_seq_len| KvCache::new(1, max_seq_len));
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            span,
            softmax_span,
            matmul_span,
            self_kv_cache,
            cross_kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_cache: bool,
        query_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let q = self.query.forward(x)?;
        let (k, v, query_pos) = match xa {
            None => {
                let k = self.key.forward(x)?;
                let v = self.value.forward(x)?;
                if let Some(cache) = self.self_kv_cache.as_mut() {
                    if flush_cache {
                        cache.reset();
                    }
                    let query_pos = cache.current_seq_len();
                    let (k, v) = cache.append(&k.contiguous()?, &v.contiguous()?)?;
                    (k, v, query_pos)
                } else {
                    (k, v, query_pos)
                }
            }
            Some(x) => {
                if flush_cache {
                    self.cross_kv_cache = None;
                }
                if let Some((k, v)) = &self.cross_kv_cache {
                    (k.clone(), v.clone(), 0)
                } else {
                    let k = self.key.forward(x)?;
                    let v = self.value.forward(x)?;
                    self.cross_kv_cache = Some((k.clone(), v.clone()));
                    (k, v, 0)
                }
            }
        };
        let wv = self.qkv_attention(&q, &k, &v, mask, query_pos)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        query_pos: usize,
    ) -> Result<Tensor> {
        let (_, q_len, n_state) = q.dims3()?;
        let kv_len = k.dim(1)?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = {
            let _enter = self.matmul_span.enter();
            q.matmul(&k)?
        };
        if let Some(mask) = mask {
            let mask = attention_mask_window(mask, query_pos, q_len, kv_len, qk.dtype())?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = {
            let _enter = self.softmax_span.enter();
            candle_nn::ops::softmax_last_dim(&qk)?
        };
        let wv = {
            let _enter = self.matmul_span.enter();
            w.matmul(&v)?
        }
        .transpose(1, 2)?
        .flatten_from(2)?;
        Ok(wv)
    }

    fn reset_kv_cache(&mut self) {
        if let Some(cache) = self.self_kv_cache.as_mut() {
            cache.reset();
        }
        self.cross_kv_cache = None;
    }
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
    fn load(
        n_state: usize,
        n_head: usize,
        ca: bool,
        self_cache_len: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load(n_state, n_head, self_cache_len, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if ca {
            let cross_attn =
                MultiHeadAttention::load(n_state, n_head, None, vb.pp("encoder_attn"))?;
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

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_kv_cache: bool,
        position_offset: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attn = self.attn.forward(
            &self.attn_ln.forward(x)?,
            None,
            mask,
            flush_kv_cache,
            position_offset,
        )?;
        let mut x = (x + attn)?;
        if let Some((attn, ln)) = &mut self.cross_attn {
            x = (&x + attn.forward(&ln.forward(&x)?, xa, None, flush_kv_cache, 0)?)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        x + mlp
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
        if let Some((attn, _)) = &mut self.cross_attn {
            attn.reset_kv_cache();
        }
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
                ResidualAttentionBlock::load(
                    n_state,
                    n_head,
                    false,
                    None,
                    vb.pp(format!("layers.{i}")),
                )
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

    pub fn forward(&mut self, x: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
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
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, None, None, flush_kv_cache, 0)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
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
                ResidualAttentionBlock::load(
                    n_state,
                    n_head,
                    true,
                    Some(n_ctx),
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, vb.pp("layer_norm"))?;
        let mask = causal_attention_mask(n_ctx, vb.dtype(), vb.device())?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            span,
            span_final,
        })
    }

    pub fn forward(&mut self, x: &Tensor, xa: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        self.forward_at(x, xa, 0, flush_kv_cache)
    }

    pub fn forward_at(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        position_offset: usize,
        flush_kv_cache: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let last = x.dim(D::Minus1)?;
        let token_embedding = self.token_embedding.forward(x)?;
        let positional_embedding = self.positional_embedding.narrow(0, position_offset, last)?;
        let positional_embedding = to_add_dtype(positional_embedding, token_embedding.dtype())?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(
                &x,
                Some(xa),
                Some(&self.mask),
                flush_kv_cache,
                position_offset,
            )?;
        }
        self.ln.forward(&x)
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

    pub fn reset_kv_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_kv_cache();
        }
    }
}

fn causal_attention_mask(n_ctx: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..n_ctx)
        .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
        .collect();
    let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), device)?;
    to_add_dtype(mask, dtype)
}

fn attention_mask_window(
    mask: &Tensor,
    query_pos: usize,
    q_len: usize,
    kv_len: usize,
    dtype: DType,
) -> Result<Tensor> {
    let mask = mask.i((query_pos..query_pos + q_len, 0..kv_len))?;
    to_add_dtype(mask, dtype)
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

    pub fn reset_kv_cache(&mut self) {
        self.encoder
            .blocks
            .iter_mut()
            .for_each(|b| b.reset_kv_cache());
        self.decoder.reset_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::{attention_mask_window, causal_attention_mask, sinusoids};
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
    fn decoder_mask_uses_requested_dtype() {
        let device = Device::Cpu;
        let mask = causal_attention_mask(4, DType::F16, &device).expect("mask");
        assert_eq!(mask.dtype(), DType::F16);

        let scores = Tensor::zeros((1, 2, 4, 4), DType::F16, &device).expect("scores");
        scores.broadcast_add(&mask).expect("same-dtype mask add");
    }

    #[test]
    fn decoder_mask_window_uses_absolute_query_position() {
        let device = Device::Cpu;
        let mask = causal_attention_mask(4, DType::F32, &device).expect("mask");
        let window = attention_mask_window(&mask, 2, 1, 4, DType::F32).expect("offset mask window");
        let rows = window.to_vec2::<f32>().expect("mask rows");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], 0.0);
        assert_eq!(rows[0][1], 0.0);
        assert_eq!(rows[0][2], 0.0);
        assert!(rows[0][3].is_infinite() && rows[0][3].is_sign_negative());
    }
}
