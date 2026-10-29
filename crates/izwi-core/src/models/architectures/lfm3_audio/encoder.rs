//! LFM3-Audio's causal speech encoder (`Lfm3AudioEncoder`) and projector.
//!
//! Log-mel frames `[128, T]` go through three stride-2 3x3 convolutions over
//! the (mel, time) plane (a full one, then two depthwise ones each followed
//! by a 1x1, all with SiLU), which turn 128 mels and 8 frames into one 4096
//! vector, then a 1x1 to 768: one encoder token per 80 ms. A 17-layer LFM2
//! model (10 width-3 ShortConvs, 7 attention layers) runs over the tokens
//! with causal attention limited to the last 32 tokens, and a projector
//! (LayerNorm, Linear, GELU, Linear) maps them into the language model.
//!
//! The model is causal with bounded reach, so it streams: each call encodes
//! the tokens whose 7-frame lookahead has arrived and keeps only the frames
//! and layer inputs later tokens still see.
//!
//! In the GGUF bundle these are the mmproj's `a.conv1d.*`, `a.pre_encode.*`,
//! `a.blk.*`, `a.post_ln` and `mm.a.mlp.*` tensors; the LFM2 layers load
//! through the LFM2 backbone under its own tensor names.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};

use crate::error::{Error, Result};
use crate::models::architectures::lfm2::backbone::{Lfm2StatelessStream, QuantizedLfm2Backbone};
use crate::models::architectures::lfm2::config::Lfm2BackboneConfig;
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::ENCODER_STRIDE;
use super::preprocessor::N_MELS;

const CHANNELS: usize = 256;
const WIDTH: usize = 768;
const LM_WIDTH: usize = 2_048;
const LAYERS: usize = 17;
const WINDOW: usize = 32;
/// Encoder tokens the RoPE tables cover: about 22 minutes of one stream.
const CONTEXT_TOKENS: usize = 16_384;

/// A 3x3 stride-2 convolution with padding 1 whose input is one channel
/// (`conv_in`) or matches its output channel for channel (depthwise).
struct StridedConv3 {
    /// `[CHANNELS, 3, 3]`.
    weight: Tensor,
    /// `[1, CHANNELS, 1, 1]`.
    bias: Tensor,
}

struct Conv1x1 {
    /// `[in, out]`.
    weight_t: Tensor,
    bias: Tensor,
}

pub(crate) struct Lfm3AudioEncoder {
    conv_in: StridedConv3,
    conv_mid: StridedConv3,
    lin_mid: Conv1x1,
    conv_end: StridedConv3,
    lin_end: Conv1x1,
    lin_out: QMatMul,
    lin_out_bias: Tensor,
    lfm: QuantizedLfm2Backbone,
    norm_weight: Tensor,
    norm_bias: Tensor,
    linear_1: QMatMul,
    linear_1_bias: Tensor,
    linear_2: QMatMul,
    linear_2_bias: Tensor,
    device: Device,
}

/// One input stream's encoder state: the mel frames later tokens still
/// read, and the LFM2 layers' bounded inputs.
#[derive(Debug, Clone, Default)]
pub(crate) struct Lfm3AudioEncoderStream {
    /// `[N_MELS, frames]` mel-major, from frame `offset` on.
    features: Vec<Vec<f32>>,
    offset: usize,
    num_frames: usize,
    num_tokens: usize,
    lfm: Lfm2StatelessStream,
    finished: bool,
}

impl Lfm3AudioEncoderStream {
    pub(crate) fn tokens(&self) -> usize {
        self.num_tokens
    }
}

/// The encoder layers' tensor names under the LFM2 backbone's GGUF names.
fn encoder_tensor_name(stored: &str) -> Option<String> {
    if stored == "a.post_ln.weight" {
        return Some("output_norm.weight".into());
    }
    let (layer, leaf) = stored.strip_prefix("a.blk.")?.split_once('.')?;
    let leaf = match leaf {
        "ln1.weight" => "attn_norm.weight",
        "ln2.weight" => "ffn_norm.weight",
        "conv_dw.weight" => "shortconv.conv.weight",
        "conv_pw1.weight" => "shortconv.in_proj.weight",
        "conv_pw2.weight" => "shortconv.out_proj.weight",
        "attn_out.weight" => "attn_output.weight",
        other => other,
    };
    Some(format!("blk.{layer}.{leaf}"))
}

impl Lfm3AudioEncoder {
    pub(crate) fn load(mmproj: &GgufLoader, device: &Device) -> Result<Self> {
        let dense = |name: &str| mmproj.load_tensor(name, DType::F32, device);
        let qmatmul = |name: &str| -> Result<QMatMul> {
            Ok(QMatMul::from_qtensor(mmproj.load_qtensor(name, device)?)?)
        };
        let strided = |index: usize| -> Result<StridedConv3> {
            Ok(StridedConv3 {
                weight: dense(&format!("a.conv1d.{index}.weight"))?.reshape((CHANNELS, 3, 3))?,
                bias: dense(&format!("a.conv1d.{index}.bias"))?.reshape((1, CHANNELS, 1, 1))?,
            })
        };
        let pointwise = |index: usize| -> Result<Conv1x1> {
            Ok(Conv1x1 {
                weight_t: dense(&format!("a.conv1d.{index}.weight"))?
                    .reshape((CHANNELS, CHANNELS))?
                    .t()?
                    .contiguous()?,
                bias: dense(&format!("a.conv1d.{index}.bias"))?.reshape(CHANNELS)?,
            })
        };
        let attention_head_count_kv = (0..LAYERS)
            .map(|layer| {
                if mmproj.has_tensor(&format!("a.blk.{layer}.attn_q.weight")) {
                    4
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let config = Lfm2BackboneConfig {
            architecture: "lfm2".into(),
            block_count: LAYERS,
            context_length: CONTEXT_TOKENS,
            embedding_length: WIDTH,
            embedding_length_out: None,
            feed_forward_length: Some(1_792),
            attention_head_count: 8,
            attention_head_count_kv,
            attention_layer_norm_rms_epsilon: 1e-5,
            attention_sliding_window: Some(WINDOW),
            rope_freq_base: 1_000_000.0,
            shortconv_l_cache: 3,
            lfm3: None,
        };
        let lfm = QuantizedLfm2Backbone::load_trunk(
            &mmproj.renamed(encoder_tensor_name),
            config,
            device,
        )?;
        Ok(Self {
            conv_in: strided(0)?,
            conv_mid: strided(2)?,
            lin_mid: pointwise(3)?,
            conv_end: strided(5)?,
            lin_end: pointwise(6)?,
            lin_out: qmatmul("a.pre_encode.out.weight")?,
            lin_out_bias: dense("a.pre_encode.out.bias")?,
            lfm,
            norm_weight: dense("mm.a.mlp.0.weight")?,
            norm_bias: dense("mm.a.mlp.0.bias")?,
            linear_1: qmatmul("mm.a.mlp.1.weight")?,
            linear_1_bias: dense("mm.a.mlp.1.bias")?,
            linear_2: qmatmul("mm.a.mlp.3.weight")?,
            linear_2_bias: dense("mm.a.mlp.3.bias")?,
            device: device.clone(),
        })
    }

    /// Language-model embeddings `[1, tokens, 2048]` of a whole clip's
    /// features (`[N_MELS, frames]` mel-major): one per 8 frames, rounded up.
    pub(crate) fn encode(&self, features: &[f32], frames: usize) -> Result<Tensor> {
        if frames == 0 || features.len() != N_MELS * frames {
            return Err(Error::InvalidInput(
                "LFM3-Audio encoder needs whole mel frames".into(),
            ));
        }
        let tokens = self.downsample(features, frames)?;
        let encoded = self.lfm.forward_embeds_stateless(&tokens)?;
        self.project(&encoded)
    }

    /// The embeddings a stream completes with `features` (`[N_MELS,
    /// frames]`): every token whose lookahead has arrived, or, with
    /// `is_final`, the rest. `None` when no token completes.
    pub(crate) fn stream(
        &self,
        stream: &mut Lfm3AudioEncoderStream,
        features: &[f32],
        frames: usize,
        is_final: bool,
    ) -> Result<Option<Tensor>> {
        if stream.finished {
            return Err(Error::InvalidInput(
                "LFM3-Audio encoder stream is finished".into(),
            ));
        }
        if features.len() != N_MELS * frames {
            return Err(Error::InvalidInput(
                "LFM3-Audio encoder needs whole mel frames".into(),
            ));
        }
        if stream.features.is_empty() {
            stream.features = vec![Vec::new(); N_MELS];
        }
        for (mel, row) in stream.features.iter_mut().enumerate() {
            row.extend_from_slice(&features[mel * frames..][..frames]);
        }
        let num_frames = stream.num_frames + frames;
        let num_tokens =
            (num_frames + if is_final { ENCODER_STRIDE - 1 } else { 0 }) / ENCODER_STRIDE;
        let count = num_tokens - stream.num_tokens;
        let kept = stream.features[0].len();
        let output = if count > 0 {
            let window = stream.features.concat();
            let start = stream.num_tokens - stream.offset / ENCODER_STRIDE;
            let tokens = self.downsample(&window, kept)?.narrow(1, start, count)?;
            let encoded = self.lfm.forward_embeds_stream(&mut stream.lfm, &tokens)?;
            Some(self.project(&encoded)?)
        } else {
            None
        };
        // The next token reads frames from the last token's own on.
        let offset = num_tokens.saturating_sub(1) * ENCODER_STRIDE;
        let drop = (offset - stream.offset).min(kept);
        for row in &mut stream.features {
            row.drain(..drop);
        }
        stream.offset = offset;
        stream.num_frames = num_frames;
        stream.num_tokens = num_tokens;
        stream.finished = is_final;
        Ok(output)
    }

    /// `[1, ceil(frames / 8), 768]` encoder inputs of `[N_MELS, frames]`.
    fn downsample(&self, features: &[f32], frames: usize) -> Result<Tensor> {
        let x = Tensor::from_slice(features, (1, 1, N_MELS, frames), &self.device)?;
        let x = candle_nn::ops::silu(&self.conv_in.forward(&x)?)?;
        let x = candle_nn::ops::silu(&self.lin_mid.forward(&self.conv_mid.forward(&x)?)?)?;
        let x = candle_nn::ops::silu(&self.lin_end.forward(&self.conv_end.forward(&x)?)?)?;
        let (batch, channels, mels, tokens) = x.dims4()?;
        // `flatten(1, 2)`: channel-major over (channel, mel).
        let x = x
            .permute((0, 3, 1, 2))?
            .reshape((batch, tokens, channels * mels))?
            .contiguous()?;
        Ok(self
            .lin_out
            .forward(&x)?
            .broadcast_add(&self.lin_out_bias)?)
    }

    fn project(&self, encoded: &Tensor) -> Result<Tensor> {
        let x = candle_nn::ops::layer_norm(encoded, &self.norm_weight, &self.norm_bias, 1e-5)?;
        let x = self
            .linear_1
            .forward(&x)?
            .broadcast_add(&self.linear_1_bias)?
            .gelu_erf()?;
        let x = self
            .linear_2
            .forward(&x)?
            .broadcast_add(&self.linear_2_bias)?;
        if x.dim(2)? != LM_WIDTH {
            return Err(Error::InferenceError(
                "LFM3-Audio projector width mismatch".into(),
            ));
        }
        Ok(x)
    }
}

impl StridedConv3 {
    /// `[B, C or 1, H, W] -> [B, CHANNELS, ceil(H / 2), ceil(W / 2)]`:
    /// nine strided taps, each a broadcast multiply-add (Candle runs grouped
    /// convolutions one group at a time).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, height, width) = x.dims4()?;
        let (out_h, out_w) = (height.div_ceil(2), width.div_ceil(2));
        let padded = x.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
        let indices = |len: usize, tap: usize| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                (0..len)
                    .map(|index| (2 * index + tap) as u32)
                    .collect::<Vec<_>>(),
                len,
                x.device(),
            )?)
        };
        let mut out: Option<Tensor> = None;
        for a in 0..3 {
            let rows = padded.index_select(&indices(out_h, a)?, 2)?;
            for b in 0..3 {
                let taps = rows.index_select(&indices(out_w, b)?, 3)?;
                let weight = self
                    .weight
                    .narrow(1, a, 1)?
                    .narrow(2, b, 1)?
                    .reshape((1, CHANNELS, 1, 1))?;
                let term = taps.broadcast_mul(&weight)?;
                out = Some(match out {
                    Some(sum) => sum.add(&term)?,
                    None => term,
                });
            }
        }
        Ok(out.expect("nine taps").broadcast_add(&self.bias)?)
    }
}

impl Conv1x1 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = x.dims4()?;
        let rows = x
            .permute((0, 2, 3, 1))?
            .reshape((batch * height * width, channels))?
            .contiguous()?;
        Ok(rows
            .matmul(&self.weight_t)?
            .broadcast_add(&self.bias)?
            .reshape((batch, height, width, CHANNELS))?
            .permute((0, 3, 1, 2))?
            .contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strided taps equal Candle's own convolution (full and depthwise).
    #[test]
    fn strided_taps_match_candle_conv2d() {
        use crate::kernels::numeric_check::xorshift_tensor;
        let device = Device::Cpu;
        let weight = xorshift_tensor(&[CHANNELS, 1, 3, 3], 3, 1.0, &device).unwrap();
        let bias = xorshift_tensor(&[CHANNELS], 4, 1.0, &device).unwrap();
        let conv = StridedConv3 {
            weight: weight.reshape((CHANNELS, 3, 3)).unwrap(),
            bias: bias.reshape((1, CHANNELS, 1, 1)).unwrap(),
        };
        for (channels, groups) in [(1, 1), (CHANNELS, CHANNELS)] {
            let x = xorshift_tensor(&[1, channels, 9, 13], 5, 1.0, &device).unwrap();
            let expected = x
                .conv2d(&weight, 1, 2, 1, groups)
                .unwrap()
                .broadcast_add(&bias.reshape((1, CHANNELS, 1, 1)).unwrap())
                .unwrap();
            let actual = conv.forward(&x).unwrap();
            assert_eq!(actual.dims(), expected.dims());
            let diff = (actual - expected)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-5, "channels {channels}: {diff}");
        }
    }

    #[test]
    fn encoder_tensors_take_the_backbone_names() {
        assert_eq!(
            encoder_tensor_name("a.blk.3.ln1.weight").unwrap(),
            "blk.3.attn_norm.weight"
        );
        assert_eq!(
            encoder_tensor_name("a.blk.0.conv_dw.weight").unwrap(),
            "blk.0.shortconv.conv.weight"
        );
        assert_eq!(
            encoder_tensor_name("a.blk.2.attn_q_norm.weight").unwrap(),
            "blk.2.attn_q_norm.weight"
        );
        assert_eq!(
            encoder_tensor_name("a.post_ln.weight").unwrap(),
            "output_norm.weight"
        );
        assert_eq!(encoder_tensor_name("mm.a.mlp.1.weight"), None);
        assert_eq!(encoder_tensor_name("a.gen.code.blk.0.ln1.weight"), None);
    }
}
