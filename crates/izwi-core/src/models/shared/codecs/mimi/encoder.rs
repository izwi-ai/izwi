//! Mimi's encoder: 24 kHz audio → SEANet encoder (÷960) → transformer
//! (25 Hz) → strided downsample (12.5 Hz) → split-RVQ codes. Used offline,
//! once per reference clip.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use super::config::MimiConfig;
use super::conv::{CausalConv1d, PadMode, ResnetBlock};
use super::quantizer::MimiQuantizer;
use super::transformer::MimiTransformer;
use crate::error::{Error, Result};
use crate::models::shared::memory::exact::exact_zeros;

enum Layer {
    Conv(CausalConv1d),
    Resnet(ResnetBlock),
    Elu,
}

pub struct MimiEncoder {
    layers: Vec<Layer>,
    transformer: MimiTransformer,
    downsample: CausalConv1d,
    quantizer: MimiQuantizer,
    /// Input samples per SEANet output step, and the samples left of a step's
    /// own hop that it reads.
    hop: usize,
    reach: usize,
    device: Device,
}

/// SEANet output steps one pass of the convolution stack covers. With its
/// left context a window reads about 132 steps (~127k samples), which keeps
/// the full-rate 64-channel activations of a 30 s reference out of memory.
const WINDOW_STEPS: usize = 128;

impl MimiEncoder {
    pub(crate) fn load(cfg: &MimiConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let conv = |vb: VarBuilder,
                    input: usize,
                    output: usize,
                    kernel: usize,
                    stride: usize|
         -> Result<CausalConv1d> {
            CausalConv1d::load(
                input,
                output,
                kernel,
                stride,
                1,
                true,
                PadMode::Constant,
                vb.pp("conv"),
            )
        };
        let net = vb.pp("encoder.layers");
        let mut layers = vec![Layer::Conv(conv(
            net.pp("0"),
            cfg.audio_channels,
            cfg.num_filters,
            cfg.kernel_size,
            1,
        )?)];
        let mut index = 1;
        let mut scale = 1usize;
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let channels = scale * cfg.num_filters;
            layers.push(Layer::Resnet(ResnetBlock::load(
                channels,
                cfg.compress,
                cfg.residual_kernel_size,
                1,
                net.pp(format!("{index}")),
            )?));
            layers.push(Layer::Elu);
            layers.push(Layer::Conv(conv(
                net.pp(format!("{}", index + 2)),
                channels,
                channels * 2,
                ratio * 2,
                ratio,
            )?));
            index += 3;
            scale *= 2;
        }
        layers.push(Layer::Elu);
        layers.push(Layer::Conv(conv(
            net.pp(format!("{}", index + 1)),
            scale * cfg.num_filters,
            cfg.hidden_size,
            cfg.last_kernel_size,
            1,
        )?));
        let downsample = CausalConv1d::load(
            cfg.hidden_size,
            cfg.hidden_size,
            4,
            2,
            1,
            false,
            PadMode::Replicate,
            vb.pp("downsample.conv"),
        )?;
        let (hop, reach) = hop_and_reach(cfg);
        Ok(Self {
            layers,
            transformer: MimiTransformer::load(cfg, vb.pp("encoder_transformer"))?,
            downsample,
            quantizer: MimiQuantizer::load(cfg, vb.pp("quantizer"))?,
            hop,
            reach,
            device,
        })
    }

    fn seanet(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = match layer {
                Layer::Conv(conv) => conv.forward(&x)?,
                Layer::Resnet(block) => block.forward(&x)?,
                Layer::Elu => x.elu(1.0)?,
            };
        }
        Ok(x)
    }

    /// [`Self::seanet`] over `x` `[1, 1, samples]` in windows of
    /// [`WINDOW_STEPS`] output steps. Each window starts on a hop boundary
    /// and reads `ceil(reach / hop)` steps of left context whose outputs it
    /// drops (the only steps its zero padding reaches); every window but the
    /// last spans whole hops, and the last ends where the sequence ends, so
    /// the stitched output is the whole-sequence output.
    fn seanet_windowed(&self, x: &Tensor) -> Result<Tensor> {
        let samples = x.dim(2)?;
        let steps = samples.div_ceil(self.hop);
        if steps <= WINDOW_STEPS {
            return self.seanet(x);
        }
        let context = self.reach.div_ceil(self.hop);
        let mut output: Option<Tensor> = None;
        let mut first = 0;
        while first < steps {
            let last = (first + WINDOW_STEPS).min(steps);
            let read_first = first.saturating_sub(context);
            let start = read_first * self.hop;
            let end = (last * self.hop).min(samples);
            let window = self.seanet(&x.narrow(2, start, end - start)?)?;
            let kept = window
                .narrow(2, first - read_first, last - first)?
                .contiguous()?;
            let stitched = match output.take() {
                Some(stitched) => stitched,
                None => exact_zeros((1, kept.dim(1)?, steps), kept.dtype(), kept.device())?,
            };
            stitched.slice_set(&kept, 2, first)?;
            output = Some(stitched);
            first = last;
        }
        output.ok_or_else(|| Error::InferenceError("Mimi encoder produced no windows".into()))
    }

    /// Codes of 24 kHz mono `samples`, frame-major (`frames · 32`).
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<u32>> {
        if samples.is_empty() {
            return Err(Error::InvalidInput("Mimi cannot encode empty audio".into()));
        }
        let x = Tensor::from_slice(samples, (1, 1, samples.len()), &self.device)?;
        let latent = self.seanet_windowed(&x)?; // [1, D, T25]
        let latent = self
            .transformer
            .forward(&latent.squeeze(0)?.t()?.contiguous()?)?
            .t()?
            .unsqueeze(0)?
            .contiguous()?;
        let latent = self.downsample.forward(&latent)?; // [1, D, F]
        self.quantizer
            .encode(&latent.squeeze(0)?.t()?.contiguous()?)
    }
}

/// The convolution stack's hop (input samples per output step) and its
/// causal reach (samples before a step's own hop that it reads): every conv
/// is causal, so the reaches of its layers add up in input samples.
fn hop_and_reach(cfg: &MimiConfig) -> (usize, usize) {
    let mut stride = 1usize;
    let mut reach = cfg.kernel_size - 1;
    for &ratio in cfg.upsampling_ratios.iter().rev() {
        reach += (cfg.residual_kernel_size - 1) * stride;
        // Kernel 2·ratio, stride ratio.
        reach += ratio * stride;
        stride *= ratio;
    }
    reach += (cfg.last_kernel_size - 1) * stride;
    (stride, reach)
}

#[cfg(test)]
impl MimiEncoder {
    pub(crate) fn seanet_for_test(&self, x: &Tensor) -> Result<Tensor> {
        self.seanet(x)
    }

    pub(crate) fn seanet_windowed_for_test(&self, x: &Tensor) -> Result<Tensor> {
        self.seanet_windowed(x)
    }
}
