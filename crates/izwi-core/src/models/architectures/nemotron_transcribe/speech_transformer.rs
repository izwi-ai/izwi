//! NeMo's `TransformerEncoder` as this model runs it: feature stacking, a
//! pre-encode projection, `embed_norm`, pre-norm blocks and `final_norm`.
//! One module serves three configurations: the ASR encoder (FP8 linears,
//! q/k LayerNorm, RoPE on 64 of 128 dims), the diarizer (BF16 linears, no
//! q/k norm, full RoPE) and the CTC trunk (no pre-encode, RoPE on 40 of 80).
//!
//! * Feature stacking zero-pads the end to a multiple of the factor and
//!   reshapes time-major frames, so each encoder frame holds `factor`
//!   consecutive mel frames, frame-major.
//! * Attention splits `w_qkv` into `[q | k | v]`, applies the per-head q/k
//!   LayerNorm (with bias) before RoPE, rotates the first `rotary_dims` dims
//!   NeoX-style (`rotate_half`) with positions from 0 in each call, masks
//!   padded keys only, and scales by `1/sqrt(head_dim)`.
//! * The FFN is `Linear → GELU(erf) → Linear`, both with bias.
//!
//! RoPE tables follow upstream: F32 angles, cosine and sine rounded to BF16.
//! Everything else runs in F32.

use candle_core::{DType, Device, Tensor, D};
use half::bf16;

use super::checkpoint::{is_fp8_linear, NemotronCheckpoint};
use super::config::{NemotronTranscribeConfig, SpeechTransformerConfig};
use super::linear::{NtLinear, Pass};
use super::precision::Fp8Mode;
use crate::error::{Error, Result};

/// A LayerNorm's F32 weight and bias.
#[derive(Debug, Clone)]
pub struct NtLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl NtLayerNorm {
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        name: &str,
        width: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: Tensor::from_vec(
                checkpoint.f32_values(&format!("{name}.weight"), &[width])?,
                width,
                device,
            )?,
            bias: Tensor::from_vec(
                checkpoint.f32_values(&format!("{name}.bias"), &[width])?,
                width,
                device,
            )?,
            eps,
        })
    }

    /// LayerNorm over the last dimension of F32 `x`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centred = x.broadcast_sub(&mean)?;
        let variance = centred.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centred.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

/// Host RoPE tables `[positions, rotary_dims]` (`cat(freqs, freqs)`),
/// rounded to BF16 as upstream stores them.
#[derive(Debug, Clone)]
pub struct RopeTables {
    rotary_dims: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTables {
    pub fn new(rotary_dims: usize, base: f64, positions: usize) -> Self {
        let half = rotary_dims / 2;
        let inv_freq = (0..half)
            .map(|i| (1.0 / base.powf((2 * i) as f64 / rotary_dims as f64)) as f32)
            .collect::<Vec<_>>();
        let mut cos = vec![0f32; positions * rotary_dims];
        let mut sin = vec![0f32; positions * rotary_dims];
        for t in 0..positions {
            for (i, f) in inv_freq.iter().enumerate() {
                // torch: an F32 outer product, then cos/sin, then BF16.
                let angle = f64::from(t as f32 * f);
                let (c, s) = (
                    bf16::from_f32(angle.cos() as f32).to_f32(),
                    bf16::from_f32(angle.sin() as f32).to_f32(),
                );
                for column in [i, i + half] {
                    cos[t * rotary_dims + column] = c;
                    sin[t * rotary_dims + column] = s;
                }
            }
        }
        Self {
            rotary_dims,
            cos,
            sin,
        }
    }

    fn positions(&self) -> usize {
        self.cos.len() / self.rotary_dims
    }

    /// `(cos, sin)` `[len, 1, rotary_dims]` for positions `0..len`.
    fn slice(&self, len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        if len > self.positions() {
            return Err(Error::InferenceError(format!(
                "RoPE table of {} positions cannot cover {len}",
                self.positions()
            )));
        }
        let shape = (len, 1, self.rotary_dims);
        Ok((
            Tensor::from_slice(&self.cos[..len * self.rotary_dims], shape, device)?,
            Tensor::from_slice(&self.sin[..len * self.rotary_dims], shape, device)?,
        ))
    }

    #[cfg(test)]
    pub(crate) fn value(&self, position: usize, column: usize) -> (f32, f32) {
        let i = position * self.rotary_dims + column;
        (self.cos[i], self.sin[i])
    }
}

/// Rotates the first `rotary_dims` dims of `x` `[T, H, hd]` (NeoX
/// `rotate_half`), passing the rest through.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dims: usize) -> Result<Tensor> {
    let head_dim = x.dim(2)?;
    let rot = x.narrow(2, 0, rotary_dims)?;
    let half = rotary_dims / 2;
    let rotated = Tensor::cat(
        &[&rot.narrow(2, half, half)?.neg()?, &rot.narrow(2, 0, half)?],
        2,
    )?;
    let rot = (rot.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?;
    if rotary_dims == head_dim {
        Ok(rot)
    } else {
        Ok(Tensor::cat(
            &[&rot, &x.narrow(2, rotary_dims, head_dim - rotary_dims)?],
            2,
        )?)
    }
}

struct Block {
    norm1: NtLayerNorm,
    norm2: NtLayerNorm,
    w_qkv: NtLinear,
    q_norm: Option<NtLayerNorm>,
    k_norm: Option<NtLayerNorm>,
    out_proj: NtLinear,
    ffn_in: NtLinear,
    ffn_out: NtLinear,
}

/// A loaded encoder.
pub struct SpeechTransformer {
    config: SpeechTransformerConfig,
    pre_encode: Option<NtLinear>,
    embed_norm: NtLayerNorm,
    blocks: Vec<Block>,
    final_norm: NtLayerNorm,
    rope: RopeTables,
    device: Device,
}

/// Linears the family loads FP8 unless the exclude list names them; the
/// CTC head's F32 tensors are always dense.
fn load_linear(
    checkpoint: &NemotronCheckpoint,
    config: &NemotronTranscribeConfig,
    module: &str,
    rows: usize,
    cols: usize,
    bias: bool,
    device: &Device,
    mode: Fp8Mode,
    dense_only: bool,
) -> Result<NtLinear> {
    if !dense_only && is_fp8_linear(config, module) {
        NtLinear::load_fp8(checkpoint, module, rows, cols, bias, device, mode)
    } else {
        NtLinear::load_dense(checkpoint, module, rows, cols, bias, device)
    }
}

/// The most frames one call may attend over: a 40 s window with 4 s of
/// context on each side is 600 frames, and the diarizer adds its cache and
/// FIFO; the CTC trunk runs at twice the frame rate.
pub const MAX_POSITIONS: usize = 5000;

impl SpeechTransformer {
    /// Loads the encoder under `prefix`. `pre_encode` loads the stacking
    /// projection; `dense_only` keeps every linear dense (the CTC head).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        encoder: SpeechTransformerConfig,
        prefix: &str,
        pre_encode: bool,
        dense_only: bool,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let d = encoder.d_model;
        let eps = encoder.layer_norm_eps;
        let ln = |name: String, width| NtLayerNorm::load(checkpoint, &name, width, eps, device);
        let lin = |name: String, rows, cols, bias| {
            load_linear(
                checkpoint, config, &name, rows, cols, bias, device, mode, dense_only,
            )
        };
        let pre_encode = if pre_encode {
            Some(lin(
                format!("{prefix}.pre_encode.proj"),
                d,
                encoder.feat_in * encoder.stacking,
                false,
            )?)
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(encoder.layers);
        for layer in 0..encoder.layers {
            let l = format!("{prefix}.layers.{layer}");
            let head = encoder.head_dim();
            blocks.push(Block {
                norm1: ln(format!("{l}.norm1"), d)?,
                norm2: ln(format!("{l}.norm2"), d)?,
                w_qkv: lin(format!("{l}.attn.w_qkv"), 3 * d, d, false)?,
                q_norm: encoder
                    .qk_norm
                    .then(|| ln(format!("{l}.attn.q_norm"), head))
                    .transpose()?,
                k_norm: encoder
                    .qk_norm
                    .then(|| ln(format!("{l}.attn.k_norm"), head))
                    .transpose()?,
                out_proj: lin(format!("{l}.attn.out_proj"), d, d, true)?,
                ffn_in: lin(format!("{l}.ffn.net.0"), encoder.ff_dim, d, true)?,
                ffn_out: lin(format!("{l}.ffn.net.3"), d, encoder.ff_dim, true)?,
            });
        }
        Ok(Self {
            pre_encode,
            embed_norm: ln(format!("{prefix}.embed_norm"), d)?,
            final_norm: ln(format!("{prefix}.final_norm"), d)?,
            blocks,
            rope: RopeTables::new(encoder.rotary_dims, encoder.rope_base, MAX_POSITIONS),
            config: encoder,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &SpeechTransformerConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resident bytes of the encoder's weights.
    pub fn resident_bytes(&self) -> usize {
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                b.w_qkv.resident_bytes()
                    + b.out_proj.resident_bytes()
                    + b.ffn_in.resident_bytes()
                    + b.ffn_out.resident_bytes()
            })
            .sum::<usize>();
        blocks + self.pre_encode.as_ref().map_or(0, NtLinear::resident_bytes)
    }

    /// Stacks time-major mel frames `[frames, feat_in]` (host F32) and
    /// projects them: `[ceil(frames / factor), d_model]` on the device.
    pub fn pre_encode(&self, mel: &[f32]) -> Result<Tensor> {
        let projection = self.pre_encode.as_ref().ok_or_else(|| {
            Error::InferenceError("this encoder has no pre-encode projection".into())
        })?;
        let feat = self.config.feat_in;
        let factor = self.config.stacking;
        let frames = mel.len() / feat;
        let stacked_frames = frames.div_ceil(factor);
        let mut stacked = mel.to_vec();
        stacked.resize(stacked_frames * factor * feat, 0.0);
        let x = Tensor::from_vec(stacked, (stacked_frames, factor * feat), &self.device)?;
        projection.forward(&x, Pass::Span)
    }

    /// The encoder over pre-encoded frames `x` `[T, d_model]`, with keys at
    /// `valid..T` masked. Positions run `0..T`.
    pub fn forward_pre_encoded(&self, x: &Tensor, valid: usize) -> Result<Tensor> {
        let (frames, width) = x.dims2()?;
        if width != self.config.d_model || valid > frames {
            return Err(Error::InferenceError(format!(
                "speech encoder input [{frames}, {width}] with {valid} valid frames"
            )));
        }
        let heads = self.config.heads;
        let head_dim = self.config.head_dim();
        let (cos, sin) = self.rope.slice(frames, &self.device)?;
        let mask = if valid < frames {
            let mut values = vec![0f32; frames];
            for value in &mut values[valid..] {
                *value = f32::NEG_INFINITY;
            }
            Some(Tensor::from_vec(values, (1, 1, frames), &self.device)?)
        } else {
            None
        };
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut x = self.embed_norm.forward(x)?;
        for block in &self.blocks {
            let h = block.norm1.forward(&x)?;
            let qkv = block.w_qkv.forward(&h, Pass::Span)?;
            let split = |index: usize| -> Result<Tensor> {
                Ok(qkv
                    .narrow(1, index * heads * head_dim, heads * head_dim)?
                    .reshape((frames, heads, head_dim))?)
            };
            let (mut q, mut k, v) = (split(0)?, split(1)?, split(2)?);
            if let (Some(qn), Some(kn)) = (&block.q_norm, &block.k_norm) {
                q = qn.forward(&q)?;
                k = kn.forward(&k)?;
            }
            let q = apply_rope(&q, &cos, &sin, self.config.rotary_dims)?;
            let k = apply_rope(&k, &cos, &sin, self.config.rotary_dims)?;
            // [H, T, hd]
            let q = q.transpose(0, 1)?.contiguous()?;
            let k = k.transpose(0, 1)?.contiguous()?;
            let v = v.transpose(0, 1)?.contiguous()?;
            let mut scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
            if let Some(mask) = &mask {
                scores = scores.broadcast_add(mask)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let attended = probs
                .matmul(&v)?
                .transpose(0, 1)?
                .reshape((frames, heads * head_dim))?;
            x = (x + block.out_proj.forward(&attended, Pass::Span)?)?;
            let h = block.norm2.forward(&x)?;
            let h = block.ffn_in.forward(&h, Pass::Span)?.gelu_erf()?;
            x = (x + block.ffn_out.forward(&h, Pass::Span)?)?;
        }
        self.final_norm.forward(&x)
    }

    /// Pre-encode and encode `mel` frames, `valid_mel` of them valid.
    pub fn forward_mel(&self, mel: &[f32], valid_mel: usize) -> Result<Tensor> {
        let x = self.pre_encode(mel)?;
        let valid = valid_mel.div_ceil(self.config.stacking);
        self.forward_pre_encoded(&x, valid)
    }
}

/// A device tensor's values on the host.
pub(crate) fn host_values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_rotate_half_with_the_base_over_the_rotated_dims() {
        let tables = RopeTables::new(64, 10_000.0, 600);
        // Column i and i + 32 share a frequency; the base is taken over 64.
        let inv = (1.0 / 10_000f64.powf(2.0 * 5.0 / 64.0)) as f32;
        let angle = f64::from(599f32 * inv);
        let (c, s) = tables.value(599, 5);
        assert_eq!(c, bf16::from_f32(angle.cos() as f32).to_f32());
        assert_eq!(s, bf16::from_f32(angle.sin() as f32).to_f32());
        assert_eq!(tables.value(599, 37), (c, s));
        assert_eq!(tables.value(0, 0), (1.0, 0.0));
    }

    #[test]
    fn partial_rope_rotates_only_the_leading_dims_neox_style() {
        let device = Device::Cpu;
        let tables = RopeTables::new(4, 10_000.0, 8);
        let (cos, sin) = tables.slice(2, &device).unwrap();
        let x = Tensor::arange(0f32, 2.0 * 6.0, &device)
            .unwrap()
            .reshape((2, 1, 6))
            .unwrap();
        let y = apply_rope(&x, &cos, &sin, 4).unwrap();
        let y = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Position 0 is the identity.
        assert_eq!(&y[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // Position 1: pairs (x0, x2) and (x1, x3) rotate; x4, x5 pass.
        let (c0, s0) = tables.value(1, 0);
        let (c1, s1) = tables.value(1, 1);
        let x = [6f32, 7.0, 8.0, 9.0, 10.0, 11.0];
        assert!((y[6] - (x[0] * c0 - x[2] * s0)).abs() < 1e-5);
        assert!((y[7] - (x[1] * c1 - x[3] * s1)).abs() < 1e-5);
        assert!((y[8] - (x[2] * c0 + x[0] * s0)).abs() < 1e-5);
        assert!((y[9] - (x[3] * c1 + x[1] * s1)).abs() < 1e-5);
        assert_eq!(&y[10..], &[10.0, 11.0]);
    }

    #[test]
    fn layer_norm_has_bias_and_population_variance() {
        let device = Device::Cpu;
        let norm = NtLayerNorm {
            weight: Tensor::new(&[2f32, 2.0], &device).unwrap(),
            bias: Tensor::new(&[1f32, 1.0], &device).unwrap(),
            eps: 0.0,
        };
        let y = norm
            .forward(&Tensor::new(&[[1f32, 3.0]], &device).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![-1.0, 3.0]);
    }
}
