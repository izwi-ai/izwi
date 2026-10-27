//! Mimi's decoder: codes → quantized latent → depthwise ×2 upsample →
//! transformer (25 Hz) → SEANet decoder (×960) → 24 kHz audio.
//!
//! Every layer is causal, so a stream decoded a few frames at a time is the
//! whole-sequence decode (up to summation order) and needs no lookahead.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use super::config::MimiConfig;
use super::conv::{CausalConv1d, CausalConvTranspose1d, PadMode, ResnetBlock};
use super::quantizer::MimiQuantizer;
use super::transformer::{MimiTransformer, MimiTransformerState};
use crate::error::{Error, Result};

struct UpBlock {
    conv: CausalConvTranspose1d,
    resnet: ResnetBlock,
}

pub struct MimiDecoder {
    quantizer: MimiQuantizer,
    upsample: CausalConvTranspose1d,
    transformer: MimiTransformer,
    first: CausalConv1d,
    blocks: Vec<UpBlock>,
    last: CausalConv1d,
    samples_per_frame: usize,
    device: Device,
}

/// One stream's decoder state: every causal layer's carry and the
/// transformer's window, all zero/empty for a new stream.
#[derive(Clone)]
pub struct MimiDecoderState {
    upsample_tail: Tensor,
    transformer: MimiTransformerState,
    first_history: Tensor,
    /// Per up block: the transposed conv's spill and the residual conv's
    /// history.
    block_tails: Vec<Tensor>,
    block_histories: Vec<Tensor>,
    last_history: Tensor,
    frames: usize,
}

impl MimiDecoderState {
    /// Frames decoded so far.
    pub fn frames(&self) -> usize {
        self.frames
    }
}

impl MimiDecoder {
    pub(crate) fn load(cfg: &MimiConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let d = cfg.hidden_size;
        let quantizer = MimiQuantizer::load(cfg, vb.pp("quantizer"))?;
        let upsample = CausalConvTranspose1d::load_depthwise(d, 4, 2, vb.pp("upsample.conv"))?;
        let transformer = MimiTransformer::load(cfg, vb.pp("decoder_transformer"))?;
        let layers = vb.pp("decoder.layers");
        let ratios = &cfg.upsampling_ratios;
        let mut scale = 1usize << ratios.len();
        let first = CausalConv1d::load(
            d,
            scale * cfg.num_filters,
            cfg.kernel_size,
            1,
            1,
            true,
            PadMode::Constant,
            layers.pp("0.conv"),
        )?;
        // Layer indices: conv, then (ELU, transposed conv, resnet) per ratio,
        // then ELU and the last conv.
        let mut index = 1;
        let mut blocks = Vec::with_capacity(ratios.len());
        for &ratio in ratios {
            let channels = scale * cfg.num_filters;
            let conv = CausalConvTranspose1d::load(
                channels,
                channels / 2,
                2 * ratio,
                ratio,
                layers.pp(format!("{}.conv", index + 1)),
            )?;
            let resnet = ResnetBlock::load(
                channels / 2,
                cfg.compress,
                cfg.residual_kernel_size,
                1,
                layers.pp(format!("{}", index + 2)),
            )?;
            blocks.push(UpBlock { conv, resnet });
            index += 3;
            scale /= 2;
        }
        let last = CausalConv1d::load(
            cfg.num_filters,
            cfg.audio_channels,
            cfg.last_kernel_size,
            1,
            1,
            true,
            PadMode::Constant,
            layers.pp(format!("{}.conv", index + 1)),
        )?;
        Ok(Self {
            quantizer,
            upsample,
            transformer,
            first,
            blocks,
            last,
            samples_per_frame: cfg.samples_per_frame(),
            device,
        })
    }

    pub fn samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }

    pub fn codebooks(&self) -> usize {
        self.quantizer.codebooks()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_state(&self) -> Result<MimiDecoderState> {
        let zeros = |channels: usize, length: usize| -> Result<Tensor> {
            Ok(Tensor::zeros(
                (1, channels, length),
                candle_core::DType::F32,
                &self.device,
            )?)
        };
        Ok(MimiDecoderState {
            upsample_tail: zeros(self.upsample.out_channels(), self.upsample.overlap())?,
            transformer: MimiTransformerState::new(self.transformer.num_layers()),
            first_history: zeros(self.first.in_channels(), self.first.history())?,
            block_tails: self
                .blocks
                .iter()
                .map(|block| zeros(block.conv.out_channels(), block.conv.overlap()))
                .collect::<Result<_>>()?,
            block_histories: self
                .blocks
                .iter()
                .map(|block| {
                    zeros(
                        block.resnet.conv1.in_channels(),
                        block.resnet.conv1.history(),
                    )
                })
                .collect::<Result<_>>()?,
            last_history: zeros(self.last.in_channels(), self.last.history())?,
            frames: 0,
        })
    }

    fn latent(&self, codes: &[u32]) -> Result<Tensor> {
        // [N, hidden] → [1, hidden, N]
        Ok(self
            .quantizer
            .decode(codes, &self.device)?
            .t()?
            .unsqueeze(0)?
            .contiguous()?)
    }

    /// The next frames of a stream (`codes` frame-major, a whole number of
    /// frames): `frames · samples_per_frame` samples, F32 `[samples]`.
    pub fn decode_stream(&self, state: &mut MimiDecoderState, codes: &[u32]) -> Result<Tensor> {
        let frames = codes.len() / self.codebooks();
        let x = self.latent(codes)?;
        let x = self.upsample.forward_stream(&x, &mut state.upsample_tail)?;
        let x = self
            .transformer
            .forward_stream(&x.squeeze(0)?.t()?.contiguous()?, &mut state.transformer)?
            .t()?
            .unsqueeze(0)?
            .contiguous()?;
        let mut x = self.first.forward_stream(&x, &mut state.first_history)?;
        for ((block, tail), history) in self
            .blocks
            .iter()
            .zip(state.block_tails.iter_mut())
            .zip(state.block_histories.iter_mut())
        {
            x = block.conv.forward_stream(&x.elu(1.0)?, tail)?;
            x = block.resnet.forward_stream(&x, history)?;
        }
        let audio = self
            .last
            .forward_stream(&x.elu(1.0)?, &mut state.last_history)?;
        state.frames += frames;
        self.check_samples(audio.flatten_all()?, frames)
    }

    /// Every frame of `codes` at once (the reference the stream must equal).
    pub fn decode(&self, codes: &[u32]) -> Result<Tensor> {
        let frames = codes.len() / self.codebooks();
        let x = self.upsample.forward(&self.latent(codes)?)?;
        let x = self
            .transformer
            .forward(&x.squeeze(0)?.t()?.contiguous()?)?
            .t()?
            .unsqueeze(0)?
            .contiguous()?;
        let mut x = self.first.forward(&x)?;
        for block in &self.blocks {
            x = block.conv.forward(&x.elu(1.0)?)?;
            x = block.resnet.forward(&x)?;
        }
        let audio = self.last.forward(&x.elu(1.0)?)?;
        self.check_samples(audio.flatten_all()?, frames)
    }

    fn check_samples(&self, audio: Tensor, frames: usize) -> Result<Tensor> {
        let expected = frames * self.samples_per_frame;
        if audio.dim(0)? != expected {
            return Err(Error::InferenceError(format!(
                "Mimi decoded {} samples for {frames} frames (expected {expected})",
                audio.dim(0)?
            )));
        }
        Ok(audio)
    }
}
