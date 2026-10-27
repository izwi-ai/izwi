//! SNAC 24 kHz's decoder (<https://github.com/hubertsiuzdak/snac>, MIT),
//! ported natively: codes → residual codebooks → convolutional decoder →
//! 24 kHz waveform. The encoder is never loaded.
//!
//! Three details differ from candle-transformers' port, which izwi does not
//! use:
//! * the decoder ends in `tanh`, as upstream's does (candle's omits it);
//! * each decoder block's noise is supplied by the caller, drawn on the host
//!   from the request's own stream, never from a device RNG;
//! * the depthwise convolutions run as one shift-and-accumulate op
//!   ([`depthwise_conv1d`]), not one convolution per channel.
//!
//! Every activation is F32 on every backend.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::error::{Error, Result};
use crate::kernels::depthwise_conv::{depthwise_conv1d, DepthwiseConv1dSpec};
use crate::models::shared::weights::pytorch::PthTensorMap;

use super::frames::SnacCodes;

/// `hubertsiuzdak/snac_24khz`'s `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnacConfig {
    pub sampling_rate: u32,
    /// The latent width: `encoder_dim · 2^len(encoder_rates)`.
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: [usize; 4],
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vq_strides: [usize; 3],
}

impl SnacConfig {
    pub fn snac_24khz() -> Self {
        Self {
            sampling_rate: 24_000,
            latent_dim: 48 * 16,
            decoder_dim: 1_024,
            decoder_rates: [8, 8, 4, 2],
            codebook_size: 4_096,
            codebook_dim: 8,
            vq_strides: [4, 2, 1],
        }
    }

    /// Latent steps (the finest codes) per frame.
    pub fn latent_steps_per_frame(&self) -> usize {
        self.vq_strides[0]
    }

    /// Time steps per frame at the output of decoder block `block`; the last
    /// block's is the frame's sample count.
    pub fn block_steps_per_frame(&self, block: usize) -> usize {
        self.latent_steps_per_frame() * self.decoder_rates[..=block].iter().product::<usize>()
    }

    pub fn samples_per_frame(&self) -> usize {
        self.block_steps_per_frame(self.decoder_rates.len() - 1)
    }
}

/// Per-block noise for a batch: `noise[block]` holds, row after row, every
/// row's `frames · block_steps_per_frame(block)` standard normals.
pub type SnacNoise = [Vec<f32>; 4];

/// `x + sin²(αx)/(α + 1e-9)`, `α` per channel.
struct Snake {
    alpha: Tensor,
    inv_alpha: Tensor,
}

impl Snake {
    fn load(vb: &VarBuilder, channels: usize) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self {
            inv_alpha: alpha.affine(1.0, 1e-9)?.recip()?,
            alpha,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sin = x.broadcast_mul(&self.alpha)?.sin()?;
        Ok(x.add(&sin.sqr()?.broadcast_mul(&self.inv_alpha)?)?)
    }
}

/// A 1×1 convolution as a matmul: `weight` [out, in], `bias` [1, out, 1].
struct Pointwise {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Pointwise {
    fn load(vb: &VarBuilder, bias: bool) -> Result<Self> {
        let kernel = weight_norm(vb)?;
        let (out, input, k) = kernel.dims3()?;
        if k != 1 {
            return Err(Error::ModelLoadError(format!(
                "SNAC pointwise conv {} has kernel {k}",
                vb.prefix()
            )));
        }
        Ok(Self {
            weight: kernel.reshape((out, input))?,
            bias: bias
                .then(|| -> Result<Tensor> { Ok(vb.get(out, "bias")?.reshape((1, out, 1))?) })
                .transpose()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.weight.broadcast_matmul(x)?;
        Ok(match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        })
    }
}

/// A depthwise convolution: `weight` [C, 1, K].
struct Depthwise {
    weight: Tensor,
    bias: Tensor,
    spec: DepthwiseConv1dSpec,
}

impl Depthwise {
    fn load(vb: &VarBuilder, dilation: usize) -> Result<Self> {
        let weight = weight_norm(vb)?;
        let (channels, one, k) = weight.dims3()?;
        if one != 1 {
            return Err(Error::ModelLoadError(format!(
                "SNAC depthwise conv {} is not depthwise",
                vb.prefix()
            )));
        }
        Ok(Self {
            bias: vb.get(channels, "bias")?,
            weight,
            spec: DepthwiseConv1dSpec::symmetric((k - 1) * dilation / 2).with_dilation(dilation),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(depthwise_conv1d(
            x,
            &self.weight,
            Some(&self.bias),
            self.spec,
        )?)
    }
}

struct ResidualUnit {
    snake1: Snake,
    conv: Depthwise,
    snake2: Snake,
    pointwise: Pointwise,
}

impl ResidualUnit {
    fn load(vb: &VarBuilder, channels: usize, dilation: usize) -> Result<Self> {
        let vb = vb.pp("block");
        Ok(Self {
            snake1: Snake::load(&vb.pp(0), channels)?,
            conv: Depthwise::load(&vb.pp(1), dilation)?,
            snake2: Snake::load(&vb.pp(2), channels)?,
            pointwise: Pointwise::load(&vb.pp(3), true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.snake1.forward(x)?;
        let y = self.conv.forward(&y)?;
        let y = self.pointwise.forward(&self.snake2.forward(&y)?)?;
        Ok(x.add(&y)?)
    }
}

struct DecoderBlock {
    snake: Snake,
    upsample: Tensor,
    upsample_bias: Tensor,
    stride: usize,
    noise: Pointwise,
    residual: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn load(vb: &VarBuilder, input: usize, output: usize, stride: usize) -> Result<Self> {
        let vb = vb.pp("block");
        let upsample = vb.pp(1);
        let kernel = weight_norm(&upsample)?;
        if kernel.dims() != [input, output, 2 * stride] {
            return Err(Error::ModelLoadError(format!(
                "SNAC upsampling conv {} has shape {:?}",
                upsample.prefix(),
                kernel.dims()
            )));
        }
        Ok(Self {
            snake: Snake::load(&vb.pp(0), input)?,
            upsample_bias: upsample.get(output, "bias")?.reshape((1, output, 1))?,
            upsample: kernel,
            stride,
            noise: Pointwise::load(&vb.pp(2).pp("linear"), false)?,
            residual: [
                ResidualUnit::load(&vb.pp(3), output, 1)?,
                ResidualUnit::load(&vb.pp(4), output, 3)?,
                ResidualUnit::load(&vb.pp(5), output, 9)?,
            ],
        })
    }

    /// `noise` is [batch, 1, time] at this block's output rate.
    fn forward(&self, x: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let x = self.snake.forward(x)?;
        let x = x
            .conv_transpose1d(
                &self.upsample,
                self.stride.div_ceil(2),
                self.stride % 2,
                self.stride,
                1,
                1,
            )?
            .broadcast_add(&self.upsample_bias)?;
        let x = x.add(&noise.broadcast_mul(&self.noise.forward(&x)?)?)?;
        let mut x = x;
        for unit in &self.residual {
            x = unit.forward(&x)?;
        }
        Ok(x)
    }
}

struct Quantizer {
    /// [codebook_size, codebook_dim]
    codebook: Tensor,
    out_proj: Pointwise,
    stride: usize,
}

pub struct SnacDecoder {
    cfg: SnacConfig,
    device: Device,
    quantizers: Vec<Quantizer>,
    conv_in: Depthwise,
    expand: Pointwise,
    blocks: Vec<DecoderBlock>,
    snake_out: Snake,
    conv_out: Tensor,
    conv_out_bias: Tensor,
}

impl SnacDecoder {
    /// Load the decoder and codebooks from the checkpoint at `path` (a
    /// `torch.save` zip), fusing weight norm on the host, onto `device` in
    /// F32.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        if !path.is_file() {
            return Err(Error::ModelLoadError(format!(
                "SNAC decoder weights {} are missing",
                path.display()
            )));
        }
        let archive = PthTensorMap::open(path, None)?;
        let tensors = archive.read_where(&Device::Cpu, Some(DType::F32), |name| {
            name.starts_with("decoder.") || name.starts_with("quantizer.")
        })?;
        let vb = VarBuilder::from_tensors(tensors.into_iter().collect(), DType::F32, &Device::Cpu);
        Self::from_var_builder(SnacConfig::snac_24khz(), &vb)?.into_device(device)
    }

    fn from_var_builder(cfg: SnacConfig, vb: &VarBuilder) -> Result<Self> {
        let quantizers = cfg
            .vq_strides
            .iter()
            .enumerate()
            .map(|(index, &stride)| -> Result<Quantizer> {
                let vb = vb.pp("quantizer").pp("quantizers").pp(index);
                Ok(Quantizer {
                    codebook: vb.get((cfg.codebook_size, cfg.codebook_dim), "codebook.weight")?,
                    out_proj: Pointwise::load(&vb.pp("out_proj"), true)?,
                    stride,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let model = vb.pp("decoder").pp("model");
        let mut channels = cfg.decoder_dim;
        let mut blocks = Vec::with_capacity(cfg.decoder_rates.len());
        for (index, &stride) in cfg.decoder_rates.iter().enumerate() {
            blocks.push(DecoderBlock::load(
                &model.pp(2 + index),
                channels,
                channels / 2,
                stride,
            )?);
            channels /= 2;
        }
        let tail = 2 + cfg.decoder_rates.len();
        let conv_out = model.pp(tail + 1);
        Ok(Self {
            conv_in: Depthwise::load(&model.pp(0), 1)?,
            expand: Pointwise::load(&model.pp(1), true)?,
            snake_out: Snake::load(&model.pp(tail), channels)?,
            conv_out_bias: conv_out.get(1, "bias")?,
            conv_out: weight_norm(&conv_out)?,
            quantizers,
            blocks,
            device: Device::Cpu,
            cfg,
        })
    }

    fn into_device(self, device: &Device) -> Result<Self> {
        if device.same_device(&self.device) {
            return Ok(self);
        }
        let move_t = |t: &Tensor| -> Result<Tensor> { Ok(t.to_device(device)?) };
        let move_snake = |s: &Snake| -> Result<Snake> {
            Ok(Snake {
                alpha: move_t(&s.alpha)?,
                inv_alpha: move_t(&s.inv_alpha)?,
            })
        };
        let move_pointwise = |p: &Pointwise| -> Result<Pointwise> {
            Ok(Pointwise {
                weight: move_t(&p.weight)?,
                bias: p.bias.as_ref().map(move_t).transpose()?,
            })
        };
        let move_depthwise = |d: &Depthwise| -> Result<Depthwise> {
            Ok(Depthwise {
                weight: move_t(&d.weight)?,
                bias: move_t(&d.bias)?,
                spec: d.spec,
            })
        };
        let move_unit = |u: &ResidualUnit| -> Result<ResidualUnit> {
            Ok(ResidualUnit {
                snake1: move_snake(&u.snake1)?,
                conv: move_depthwise(&u.conv)?,
                snake2: move_snake(&u.snake2)?,
                pointwise: move_pointwise(&u.pointwise)?,
            })
        };
        Ok(Self {
            quantizers: self
                .quantizers
                .iter()
                .map(|q| -> Result<Quantizer> {
                    Ok(Quantizer {
                        codebook: move_t(&q.codebook)?,
                        out_proj: move_pointwise(&q.out_proj)?,
                        stride: q.stride,
                    })
                })
                .collect::<Result<_>>()?,
            conv_in: move_depthwise(&self.conv_in)?,
            expand: move_pointwise(&self.expand)?,
            blocks: self
                .blocks
                .iter()
                .map(|b| -> Result<DecoderBlock> {
                    Ok(DecoderBlock {
                        snake: move_snake(&b.snake)?,
                        upsample: move_t(&b.upsample)?,
                        upsample_bias: move_t(&b.upsample_bias)?,
                        stride: b.stride,
                        noise: move_pointwise(&b.noise)?,
                        residual: [
                            move_unit(&b.residual[0])?,
                            move_unit(&b.residual[1])?,
                            move_unit(&b.residual[2])?,
                        ],
                    })
                })
                .collect::<Result<_>>()?,
            snake_out: move_snake(&self.snake_out)?,
            conv_out: move_t(&self.conv_out)?,
            conv_out_bias: move_t(&self.conv_out_bias)?,
            device: device.clone(),
            cfg: self.cfg,
        })
    }

    pub fn config(&self) -> &SnacConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Decode a batch of equally long code sequences with the given noise;
    /// each row's waveform, `frames · samples_per_frame` samples.
    pub fn decode(&self, rows: &[&SnacCodes], noise: &SnacNoise) -> Result<Vec<Vec<f32>>> {
        let Some(first) = rows.first() else {
            return Ok(Vec::new());
        };
        let frames = first.frames();
        if frames == 0 || rows.iter().any(|row| row.frames() != frames) {
            return Err(Error::InvalidInput(
                "SNAC decode needs equally long, non-empty rows".into(),
            ));
        }
        let batch = rows.len();
        for (block, values) in noise.iter().enumerate() {
            if values.len() != batch * frames * self.cfg.block_steps_per_frame(block) {
                return Err(Error::InvalidInput(format!(
                    "SNAC block {block} noise has {} values for {batch} rows of {frames} frames",
                    values.len()
                )));
            }
        }
        let levels: [Vec<u32>; 3] = [
            rows.iter()
                .flat_map(|row| row.coarse.iter().copied())
                .collect(),
            rows.iter()
                .flat_map(|row| row.medium.iter().copied())
                .collect(),
            rows.iter()
                .flat_map(|row| row.fine.iter().copied())
                .collect(),
        ];
        let latent_steps = frames * self.cfg.latent_steps_per_frame();
        let mut latent: Option<Tensor> = None;
        for (quantizer, codes) in self.quantizers.iter().zip(&levels) {
            if codes
                .iter()
                .any(|&code| code as usize >= self.cfg.codebook_size)
            {
                return Err(Error::InvalidInput("SNAC code outside the codebook".into()));
            }
            let steps = codes.len() / batch;
            let ids = Tensor::from_slice(codes, codes.len(), &self.device)?;
            let embedded = quantizer
                .codebook
                .index_select(&ids, 0)?
                .reshape((batch, steps, self.cfg.codebook_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let projected = quantizer.out_proj.forward(&embedded)?;
            let projected = repeat_interleave_time(&projected, quantizer.stride)?;
            if projected.dim(2)? != latent_steps {
                return Err(Error::InvalidInput(
                    "SNAC levels disagree on the frame count".into(),
                ));
            }
            latent = Some(match latent {
                Some(sum) => sum.add(&projected)?,
                None => projected,
            });
        }
        let latent = latent.expect("three quantizers");
        let mut x = self.expand.forward(&self.conv_in.forward(&latent)?)?;
        for (block, decoder_block) in self.blocks.iter().enumerate() {
            let steps = frames * self.cfg.block_steps_per_frame(block);
            let noise = Tensor::from_slice(&noise[block], (batch, 1, steps), &self.device)?;
            x = decoder_block.forward(&x, &noise)?;
        }
        let x = self.snake_out.forward(&x)?;
        let x = x
            .conv1d(&self.conv_out, 3, 1, 1, 1)?
            .broadcast_add(&self.conv_out_bias.reshape((1, 1, 1))?)?
            .tanh()?;
        let samples = x
            .to_device(&Device::Cpu)?
            .reshape((batch, ()))?
            .to_vec2::<f32>()?;
        if samples.iter().flatten().any(|sample| !sample.is_finite()) {
            return Err(Error::InferenceError(
                "SNAC decoder produced non-finite samples".into(),
            ));
        }
        Ok(samples)
    }
}

/// `x` [batch, channels, time] with every step repeated `factor` times.
fn repeat_interleave_time(x: &Tensor, factor: usize) -> Result<Tensor> {
    if factor == 1 {
        return Ok(x.clone());
    }
    let (batch, channels, time) = x.dims3()?;
    Ok(x.unsqueeze(3)?
        .broadcast_as((batch, channels, time, factor))?
        .reshape((batch, channels, time * factor))?)
}

/// `g · v / ‖v‖` over every dimension but the first, from PyTorch's
/// `parametrizations.weight.original0` (g) and `original1` (v), in F32.
fn weight_norm(vb: &VarBuilder) -> Result<Tensor> {
    let g = vb.get_unchecked_dtype("parametrizations.weight.original0", DType::F32)?;
    let v = vb.get_unchecked_dtype("parametrizations.weight.original1", DType::F32)?;
    let rank = v.rank();
    if rank < 2 || g.dims()[0] != v.dims()[0] {
        return Err(Error::ModelLoadError(format!(
            "SNAC weight-norm pair {} has shapes {:?} and {:?}",
            vb.prefix(),
            g.dims(),
            v.dims()
        )));
    }
    let reduce = (1..rank).collect::<Vec<_>>();
    let norm = v.sqr()?.sum_keepdim(reduce)?.sqrt()?;
    Ok(v.broadcast_mul(&g.broadcast_div(&norm)?)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_geometry_matches_snac_24khz() {
        let cfg = SnacConfig::snac_24khz();
        let steps = (0..4)
            .map(|block| cfg.block_steps_per_frame(block))
            .collect::<Vec<_>>();
        assert_eq!(steps, vec![32, 256, 1_024, 2_048]);
        assert_eq!(
            cfg.samples_per_frame(),
            super::super::config::SAMPLES_PER_FRAME
        );
    }

    #[test]
    fn repeat_interleave_repeats_each_step() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &Device::Cpu).unwrap();
        let y = repeat_interleave_time(&x, 2).unwrap();
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 1., 2., 2., 3., 3., 4., 4.]
        );
    }
}
