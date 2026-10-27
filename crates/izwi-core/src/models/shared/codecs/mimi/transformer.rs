//! Mimi's transformers (8 pre-LayerNorm layers, rotate-half RoPE, LayerScale,
//! exact GELU) with a causal sliding window: the query at position `p` sees
//! keys `p − window < k ≤ p`, as Kyutai's streaming Mimi and upstream Maya2's
//! decoder do.

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use super::config::MimiConfig;
use crate::error::Result;
use crate::models::shared::attention::gqa::{
    banded_sdpa_bhsd, compact_gqa_sdpa_bhsd, AttentionBand, CompactGqaMask,
};

struct MimiLayer {
    input_norm: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    attn_scale: Tensor,
    post_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    mlp_scale: Tensor,
}

pub(crate) struct MimiTransformer {
    layers: Vec<MimiLayer>,
    heads: usize,
    head_dim: usize,
    window: usize,
    /// RoPE inverse frequencies, host F64.
    inv_freq: Vec<f64>,
}

/// One stream's transformer state: the post-RoPE keys and values of the last
/// `window − 1` positions per layer, and the next position.
#[derive(Clone)]
pub(crate) struct MimiTransformerState {
    pub(crate) position: usize,
    kv: Vec<Option<(Tensor, Tensor)>>,
}

impl MimiTransformerState {
    pub(crate) fn new(layers: usize) -> Self {
        Self {
            position: 0,
            kv: vec![None; layers],
        }
    }
}

impl MimiTransformer {
    pub(crate) fn load(cfg: &MimiConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| {
                let vb = vb.pp(format!("layers.{index}"));
                let attn = vb.pp("self_attn");
                Ok(MimiLayer {
                    input_norm: candle_nn::layer_norm(d, cfg.norm_eps, vb.pp("input_layernorm"))?,
                    q: candle_nn::linear_no_bias(d, d, attn.pp("q_proj"))?,
                    k: candle_nn::linear_no_bias(d, d, attn.pp("k_proj"))?,
                    v: candle_nn::linear_no_bias(d, d, attn.pp("v_proj"))?,
                    o: candle_nn::linear_no_bias(d, d, attn.pp("o_proj"))?,
                    attn_scale: vb.pp("self_attn_layer_scale").get(d, "scale")?,
                    post_norm: candle_nn::layer_norm(
                        d,
                        cfg.norm_eps,
                        vb.pp("post_attention_layernorm"),
                    )?,
                    fc1: candle_nn::linear_no_bias(d, cfg.intermediate_size, vb.pp("mlp.fc1"))?,
                    fc2: candle_nn::linear_no_bias(cfg.intermediate_size, d, vb.pp("mlp.fc2"))?,
                    mlp_scale: vb.pp("mlp_layer_scale").get(d, "scale")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let half = cfg.head_dim / 2;
        let inv_freq = (0..half)
            .map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f64 / cfg.head_dim as f64))
            .collect();
        Ok(Self {
            layers,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            window: cfg.sliding_window,
            inv_freq,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// cos and sin `[T, head_dim]` (the half table repeated, transformers'
    /// `cat(freqs, freqs)`) for positions `start..start + count`.
    fn rope(&self, start: usize, count: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let mut cos = Vec::with_capacity(count * 2 * half);
        let mut sin = Vec::with_capacity(count * 2 * half);
        for position in start..start + count {
            for _ in 0..2 {
                for &freq in &self.inv_freq {
                    let angle = position as f64 * freq;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        let shape = (count, 2 * half);
        Ok((
            Tensor::from_vec(cos, shape, device)?,
            Tensor::from_vec(sin, shape, device)?,
        ))
    }

    /// `[1, heads, T, head_dim]` projections of `x` `[T, D]`.
    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        Ok(x.reshape((1, tokens, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    fn feed_forward(layer: &MimiLayer, x: &Tensor) -> Result<Tensor> {
        let h = layer
            .fc1
            .forward(&layer.post_norm.forward(x)?)?
            .gelu_erf()?;
        let h = layer.fc2.forward(&h)?;
        Ok((x + h.broadcast_mul(&layer.mlp_scale)?)?)
    }

    fn merge_heads(&self, a: &Tensor) -> Result<Tensor> {
        let tokens = a.dim(2)?;
        Ok(a.transpose(1, 2)?
            .reshape((tokens, self.heads * self.head_dim))?)
    }

    /// The whole sequence `x` `[T, D]` from position 0.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let (cos, sin) = self.rope(0, tokens, x.device())?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut x = x.clone();
        for layer in &self.layers {
            let h = layer.input_norm.forward(&x)?;
            let q = rotate_half(&self.heads(&layer.q.forward(&h)?)?, &cos, &sin)?;
            let k = rotate_half(&self.heads(&layer.k.forward(&h)?)?, &cos, &sin)?;
            let v = self.heads(&layer.v.forward(&h)?)?;
            let a = banded_sdpa_bhsd(&q, &k, &v, AttentionBand::causal_window(self.window), scale)?;
            let a = layer.o.forward(&self.merge_heads(&a)?)?;
            x = (x + a.broadcast_mul(&layer.attn_scale)?)?;
            x = Self::feed_forward(layer, &x)?;
        }
        Ok(x)
    }

    /// The next `T` positions `x` `[T, D]` of a stream.
    pub(crate) fn forward_stream(
        &self,
        x: &Tensor,
        state: &mut MimiTransformerState,
    ) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let start = state.position;
        let device = x.device();
        let (cos, sin) = self.rope(start, tokens, device)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let past = state.kv[0]
            .as_ref()
            .map(|(k, _)| k.dim(2))
            .transpose()?
            .unwrap_or(0);
        let mask = window_mask(start, tokens, past, self.window, device)?;
        let keep = self.window - 1;
        let mut x = x.clone();
        for (layer, kv) in self.layers.iter().zip(state.kv.iter_mut()) {
            let h = layer.input_norm.forward(&x)?;
            let q = rotate_half(&self.heads(&layer.q.forward(&h)?)?, &cos, &sin)?;
            let k = rotate_half(&self.heads(&layer.k.forward(&h)?)?, &cos, &sin)?;
            let v = self.heads(&layer.v.forward(&h)?)?;
            let (k, v) = match kv.take() {
                Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
                None => (k, v),
            };
            let a =
                compact_gqa_sdpa_bhsd(&q, &k, &v, Some(CompactGqaMask::Additive(&mask)), scale)?;
            let length = k.dim(2)?;
            let kept = keep.min(length);
            *kv = Some((
                k.narrow(2, length - kept, kept)?.contiguous()?,
                v.narrow(2, length - kept, kept)?.contiguous()?,
            ));
            let a = layer.o.forward(&self.merge_heads(&a)?)?;
            x = (x + a.broadcast_mul(&layer.attn_scale)?)?;
            x = Self::feed_forward(layer, &x)?;
        }
        state.position += tokens;
        Ok(x)
    }
}

/// Additive `[T, past + T]` mask: key `j` sits at `start − past + j`.
fn window_mask(
    start: usize,
    tokens: usize,
    past: usize,
    window: usize,
    device: &Device,
) -> Result<Tensor> {
    let keys = past + tokens;
    let mut data = vec![0f32; tokens * keys];
    for (query, row) in data.chunks_exact_mut(keys).enumerate() {
        let query_pos = (start + query) as isize;
        for (key, value) in row.iter_mut().enumerate() {
            let key_pos = start as isize - past as isize + key as isize;
            if key_pos > query_pos || key_pos <= query_pos - window as isize {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(data, (tokens, keys), device)?.to_dtype(DType::F32)?)
}

/// transformers' `apply_rotary_pos_emb` on `x` `[1, H, T, d]`.
fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    Ok((x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_mask_sees_the_last_window_keys() {
        // Two cached keys (positions 3, 4), two queries (5, 6), window 3.
        let mask = window_mask(5, 2, 2, 3, &Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let visible = |row: &[f32]| row.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
        assert_eq!(visible(&mask[0]), vec![true, true, true, false]);
        assert_eq!(visible(&mask[1]), vec![false, true, true, true]);
    }
}
