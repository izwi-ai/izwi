//! Mimi's decoder: codes → quantized latent → depthwise ×2 upsample →
//! transformer (25 Hz) → SEANet decoder (×960) → 24 kHz audio.
//!
//! Every layer is causal, so a stream decoded a few frames at a time is the
//! whole-sequence decode (up to summation order) and needs no lookahead.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::config::MimiConfig;
use super::conv::{CausalConv1d, CausalConvTranspose1d, PadMode, ResnetBlock};
use super::quantizer::MimiQuantizer;
use super::transformer::{MimiTransformer, MimiTransformerState};
use crate::error::{Error, Result};
use crate::models::shared::codecs::reduced_precision::{self, CodecPrecision};

struct UpBlock {
    conv: CausalConvTranspose1d,
    resnet: ResnetBlock,
}

/// The SEANet decoder after the transformer (the first convolution, the
/// upsampling blocks, the last convolution) and the dtype it runs in.
struct Seanet {
    first: CausalConv1d,
    blocks: Vec<UpBlock>,
    last: CausalConv1d,
    dtype: DType,
}

impl Seanet {
    fn with_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            first: self.first.with_dtype(dtype)?,
            blocks: self
                .blocks
                .iter()
                .map(|block| -> Result<UpBlock> {
                    Ok(UpBlock {
                        conv: block.conv.with_dtype(dtype)?,
                        resnet: block.resnet.with_dtype(dtype)?,
                    })
                })
                .collect::<Result<_>>()?,
            last: self.last.with_dtype(dtype)?,
            dtype,
        })
    }
}

pub struct MimiDecoder {
    quantizer: MimiQuantizer,
    upsample: CausalConvTranspose1d,
    transformer: MimiTransformer,
    seanet: Seanet,
    samples_per_frame: usize,
    codebook_size: usize,
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
            seanet: Seanet {
                first,
                blocks,
                last,
                dtype: DType::F32,
            },
            samples_per_frame: cfg.samples_per_frame(),
            codebook_size: cfg.codebook_size,
            device,
        })
    }

    /// The dtype the SEANet decoder runs in (the quantizer, upsample and
    /// transformer stay F32; samples come back F32).
    pub fn seanet_dtype(&self) -> DType {
        self.seanet.dtype
    }

    /// Runs the SEANet decoder in `dtype` when its samples stay within
    /// `max_rel_l2` relative L2 error of F32's ([`reduced_precision::judge`]
    /// over [`Self::check_samples`]; [`reduced_precision::MAX_REL_L2`] at
    /// load), else keeps it as it is; returns the verdict.
    pub(crate) fn try_seanet_dtype(&mut self, dtype: DType, max_rel_l2: f64) -> CodecPrecision {
        let checked = self.seanet.with_dtype(dtype).and_then(|reduced| {
            let reference = self.check_samples()?;
            let kept = std::mem::replace(&mut self.seanet, reduced);
            match self.check_samples() {
                Ok(samples) => Ok((kept, reference, samples)),
                Err(error) => {
                    self.seanet = kept;
                    Err(error)
                }
            }
        });
        match checked {
            Ok((kept, reference, samples)) => {
                let (precision, keep) =
                    reduced_precision::judge(&reference, &samples, dtype, max_rel_l2);
                if !keep {
                    self.seanet = kept;
                }
                precision
            }
            Err(error) => CodecPrecision::f32(true, format!("{dtype:?} check failed: {error}")),
        }
    }

    /// The precision check's input: twelve seeded frames streamed in three
    /// chunks of four from a new stream.
    fn check_samples(&self) -> Result<Vec<f32>> {
        let books = self.codebooks();
        let mut seed = 0x6d69_6d69u64;
        let codes = (0..12 * books)
            .map(|_| {
                seed = crate::models::shared::noise::mix(seed.wrapping_add(1));
                (seed % self.codebook_size as u64) as u32
            })
            .collect::<Vec<_>>();
        let mut state = self.new_state()?;
        let mut samples = Vec::with_capacity(12 * self.samples_per_frame);
        for chunk in codes.chunks(4 * books) {
            samples.extend(self.decode_stream(&mut state, chunk)?.to_vec1::<f32>()?);
        }
        Ok(samples)
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
        let zeros = |channels: usize, length: usize, dtype: DType| -> Result<Tensor> {
            Ok(Tensor::zeros((1, channels, length), dtype, &self.device)?)
        };
        let seanet = &self.seanet;
        Ok(MimiDecoderState {
            upsample_tail: zeros(
                self.upsample.out_channels(),
                self.upsample.overlap(),
                DType::F32,
            )?,
            transformer: MimiTransformerState::new(self.transformer.num_layers()),
            first_history: zeros(
                seanet.first.in_channels(),
                seanet.first.history(),
                seanet.dtype,
            )?,
            block_tails: seanet
                .blocks
                .iter()
                .map(|block| {
                    zeros(
                        block.conv.out_channels(),
                        block.conv.overlap(),
                        seanet.dtype,
                    )
                })
                .collect::<Result<_>>()?,
            block_histories: seanet
                .blocks
                .iter()
                .map(|block| {
                    zeros(
                        block.resnet.conv1.in_channels(),
                        block.resnet.conv1.history(),
                        seanet.dtype,
                    )
                })
                .collect::<Result<_>>()?,
            last_history: zeros(
                seanet.last.in_channels(),
                seanet.last.history(),
                seanet.dtype,
            )?,
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
            .contiguous()?
            .to_dtype(self.seanet.dtype)?;
        let seanet = &self.seanet;
        let mut x = seanet.first.forward_stream(&x, &mut state.first_history)?;
        for ((block, tail), history) in seanet
            .blocks
            .iter()
            .zip(state.block_tails.iter_mut())
            .zip(state.block_histories.iter_mut())
        {
            x = block.conv.forward_stream(&x.elu(1.0)?, tail)?;
            x = block.resnet.forward_stream(&x, history)?;
        }
        let audio = seanet
            .last
            .forward_stream(&x.elu(1.0)?, &mut state.last_history)?
            .to_dtype(DType::F32)?;
        state.frames += frames;
        self.check_length(audio.flatten_all()?, frames)
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
            .contiguous()?
            .to_dtype(self.seanet.dtype)?;
        let mut x = self.seanet.first.forward(&x)?;
        for block in &self.seanet.blocks {
            x = block.conv.forward(&x.elu(1.0)?)?;
            x = block.resnet.forward(&x)?;
        }
        let audio = self
            .seanet
            .last
            .forward(&x.elu(1.0)?)?
            .to_dtype(DType::F32)?;
        self.check_length(audio.flatten_all()?, frames)
    }

    fn check_length(&self, audio: Tensor, frames: usize) -> Result<Tensor> {
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
