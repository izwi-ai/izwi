//! The audio tower and the projector.
//!
//! The tower is the Whisper large-v3 encoder: two convolutions (the second
//! of stride 2), the **stored** position table added to each window from
//! row 0, pre-norm blocks (scores scaled by `head_dim^-0.5`, no key bias),
//! a final LayerNorm, no mask. Attention runs on the shared fused kernel
//! where the build has one (CUDA FlashAttention, Metal SDPA) and as
//! composed products elsewhere. Each 30 s window is
//! one batch row; windows never attend to each other, so windows of
//! different requests share a batch.
//!
//! The projector stacks four consecutive frames into one vector and runs
//! `linear_1`, exact GELU, `linear_2` (no biases, no norm). The result
//! enters the decoder as it is: audio embeddings are not scaled by
//! `sqrt(hidden)` the way text embeddings are.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{LayerNorm, Module};

use super::checkpoint::{SovCheckpoint, ENCODER_PREFIX, PROJECTOR_PREFIX};
use super::config::{SovEncoderConfig, SovTranscribeConfig};
use crate::error::{Error, Result};

const LAYER_NORM_EPS: f64 = 1e-5;

/// `x [.., in] -> [.., out]` with `weight [out, in]`.
pub(crate) fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let y = x.broadcast_matmul(&weight.t()?)?;
    Ok(match bias {
        Some(bias) => y.broadcast_add(bias)?,
        None => y,
    })
}

struct EncoderLayer {
    attn_norm: LayerNorm,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    ffn_norm: LayerNorm,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

pub struct SovEncoder {
    config: SovEncoderConfig,
    dtype: DType,
    device: Device,
    /// The convolutions run in F32 whatever the compute dtype.
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    positions: Tensor,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
    resident_bytes: usize,
}

impl SovEncoder {
    pub fn load(
        checkpoint: &SovCheckpoint,
        config: &SovTranscribeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let e = config.encoder;
        let a = ENCODER_PREFIX;
        let mut resident_bytes = 0usize;
        let counted = std::cell::Cell::new(0usize);
        let get = |name: String, dtype: DType| -> Result<Tensor> {
            let tensor = checkpoint.tensor(&name, device, dtype)?;
            counted.set(counted.get() + tensor.elem_count() * dtype.size_in_bytes());
            Ok(tensor)
        };
        let conv1_w = get(format!("{a}.conv1.weight"), DType::F32)?;
        let conv1_b = get(format!("{a}.conv1.bias"), DType::F32)?;
        let conv2_w = get(format!("{a}.conv2.weight"), DType::F32)?;
        let conv2_b = get(format!("{a}.conv2.bias"), DType::F32)?;
        let positions = get(format!("{a}.embed_positions.weight"), dtype)?;
        let mut layers = Vec::with_capacity(e.layers);
        for layer in 0..e.layers {
            let l = format!("{a}.layers.{layer}");
            let norm = |name: &str| -> Result<LayerNorm> {
                Ok(LayerNorm::new(
                    get(format!("{l}.{name}.weight"), dtype)?,
                    get(format!("{l}.{name}.bias"), dtype)?,
                    LAYER_NORM_EPS,
                ))
            };
            let attn_norm = norm("self_attn_layer_norm")?;
            let ffn_norm = norm("final_layer_norm")?;
            layers.push(EncoderLayer {
                attn_norm,
                q_w: get(format!("{l}.self_attn.q_proj.weight"), dtype)?,
                q_b: get(format!("{l}.self_attn.q_proj.bias"), dtype)?,
                k_w: get(format!("{l}.self_attn.k_proj.weight"), dtype)?,
                v_w: get(format!("{l}.self_attn.v_proj.weight"), dtype)?,
                v_b: get(format!("{l}.self_attn.v_proj.bias"), dtype)?,
                out_w: get(format!("{l}.self_attn.out_proj.weight"), dtype)?,
                out_b: get(format!("{l}.self_attn.out_proj.bias"), dtype)?,
                ffn_norm,
                fc1_w: get(format!("{l}.fc1.weight"), dtype)?,
                fc1_b: get(format!("{l}.fc1.bias"), dtype)?,
                fc2_w: get(format!("{l}.fc2.weight"), dtype)?,
                fc2_b: get(format!("{l}.fc2.bias"), dtype)?,
            });
        }
        let final_norm = LayerNorm::new(
            get(format!("{a}.layer_norm.weight"), dtype)?,
            get(format!("{a}.layer_norm.bias"), dtype)?,
            LAYER_NORM_EPS,
        );
        resident_bytes += counted.get();
        Ok(Self {
            config: e,
            dtype,
            device: device.clone(),
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            positions,
            layers,
            final_norm,
            resident_bytes,
        })
    }

    pub fn config(&self) -> &SovEncoderConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Activation bytes of one window in the widest layer (the MLP).
    pub fn workspace_per_window_bytes(&self) -> u64 {
        let e = &self.config;
        let frame = e.ffn + 4 * e.width + e.heads * e.positions;
        (e.positions * frame * self.dtype.size_in_bytes()) as u64
    }

    /// The convolution stem: `mel [windows, n_mels, 2 * positions]` F32 to
    /// `[windows, positions, width]` in the compute dtype, positions added.
    fn stem(&self, mel: &Tensor) -> Result<Tensor> {
        let e = &self.config;
        let (_, bins, frames) = mel.dims3()?;
        if bins != e.n_mels || frames != e.positions * 2 {
            return Err(Error::InvalidInput(format!(
                "the encoder takes [windows, {}, {}] features, got {:?}",
                e.n_mels,
                e.positions * 2,
                mel.dims()
            )));
        }
        let bias = |b: &Tensor| b.reshape((1, e.width, 1));
        let x = mel
            .to_dtype(DType::F32)?
            .conv1d(&self.conv1_w, 1, 1, 1, 1)?
            .broadcast_add(&bias(&self.conv1_b)?)?
            .gelu_erf()?;
        let x = x
            .conv1d(&self.conv2_w, 1, 2, 1, 1)?
            .broadcast_add(&bias(&self.conv2_b)?)?
            .gelu_erf()?;
        let x = x.transpose(1, 2)?.contiguous()?.to_dtype(self.dtype)?;
        Ok(x.broadcast_add(&self.positions)?)
    }

    fn attention(&self, layer: &EncoderLayer, x: &Tensor) -> Result<Tensor> {
        let e = &self.config;
        let (windows, frames, _) = x.dims3()?;
        let head_dim = e.width / e.heads;
        let split = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((windows, frames, e.heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let scale = (head_dim as f64).powf(-0.5);
        let q = split(linear(x, &layer.q_w, Some(&layer.q_b))?)?;
        let k = split(linear(x, &layer.k_w, None)?)?;
        let v = split(linear(x, &layer.v_w, Some(&layer.v_b))?)?;
        // The fused kernel (CUDA FlashAttention, Metal SDPA) scales the
        // scores itself and never holds the 1500 x 1500 matrices; elsewhere
        // the composed form scales q first, as the reference does.
        let fused = crate::models::shared::attention::flash::try_fused_self_attention_scaled(
            &q,
            &k,
            &v,
            None,
            head_dim,
            false,
            scale as f32,
        )?;
        let out = match fused {
            Some(out) => out,
            None => {
                let weights = (q * scale)?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
                let weights = candle_nn::ops::softmax_last_dim(&weights.to_dtype(DType::F32)?)?
                    .to_dtype(self.dtype)?;
                weights.matmul(&v)?
            }
        };
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((windows, frames, e.width))?;
        linear(&out, &layer.out_w, Some(&layer.out_b))
    }

    /// Encodes `mel [windows, n_mels, 2 * positions]` to
    /// `[windows, positions, width]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = self.stem(mel)?;
        for layer in &self.layers {
            let h = layer.attn_norm.forward(&x)?;
            x = (x + self.attention(layer, &h)?)?;
            let h = layer.ffn_norm.forward(&x)?;
            let h = linear(&h, &layer.fc1_w, Some(&layer.fc1_b))?.gelu_erf()?;
            x = (x + linear(&h, &layer.fc2_w, Some(&layer.fc2_b))?)?;
        }
        Ok(self.final_norm.forward(&x)?)
    }
}

pub struct SovProjector {
    downsample: usize,
    linear_1: Tensor,
    linear_2: Tensor,
    resident_bytes: usize,
}

impl SovProjector {
    pub fn load(
        checkpoint: &SovCheckpoint,
        config: &SovTranscribeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let p = PROJECTOR_PREFIX;
        let linear_1 = checkpoint.tensor(&format!("{p}.linear_1.weight"), device, dtype)?;
        let linear_2 = checkpoint.tensor(&format!("{p}.linear_2.weight"), device, dtype)?;
        let resident_bytes =
            (linear_1.elem_count() + linear_2.elem_count()) * dtype.size_in_bytes();
        Ok(Self {
            downsample: config.downsample_factor,
            linear_1,
            linear_2,
            resident_bytes,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// `[windows, positions, width]` to audio embeddings
    /// `[windows, positions / downsample, hidden]`.
    pub fn forward(&self, encoded: &Tensor) -> Result<Tensor> {
        let (windows, frames, width) = encoded.dims3()?;
        if frames % self.downsample != 0 {
            return Err(Error::InferenceError(
                "the encoder output is not a whole number of stacked frames".into(),
            ));
        }
        let stacked = encoded.contiguous()?.reshape((
            windows,
            frames / self.downsample,
            width * self.downsample,
        ))?;
        let hidden = linear(&stacked, &self.linear_1, None)?.gelu_erf()?;
        linear(&hidden, &self.linear_2, None)
    }
}

/// The largest absolute value of a tensor, for finite checks.
pub(crate) fn max_abs(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .abs()?
        .flatten_all()?
        .max(D::Minus1)?
        .to_scalar::<f32>()?)
}
