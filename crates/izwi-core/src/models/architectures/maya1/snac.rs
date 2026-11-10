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
//! Every activation is F32 on every backend, unless an operator opts a
//! CUDA load into BF16 (`IZWI_MAYA1_SNAC_BF16`) and its load-time check
//! against F32 keeps it.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::depthwise_conv::{depthwise_conv1d, DepthwiseConv1dSpec};
use crate::models::shared::codecs::reduced_precision::{self, CodecPrecision};
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

/// Rows every SNAC call runs on CUDA when windows batch
/// ([`SnacDecoder::call_rows`]).
pub const SNAC_CUDA_CALL_ROWS: usize = 4;

/// Kill switch for batching windows on CUDA.
pub(crate) const SNAC_BATCHING_ENV: &str = "IZWI_MAYA1_SNAC_BATCHING";

/// Whether CUDA SNAC calls run [`SNAC_CUDA_CALL_ROWS`] rows and audio
/// quanta batch across requests (`IZWI_MAYA1_SNAC_BATCHING`, default on).
pub(crate) fn snac_batching_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::env::flag_or(SNAC_BATCHING_ENV, true))
}

/// Opt-in for a BF16 SNAC on CUDA (a boolean; off when unset).
pub(crate) const SNAC_BF16_ENV: &str = "IZWI_MAYA1_SNAC_BF16";

fn snac_bf16_requested() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(SNAC_BF16_ENV, false));
    ON.get()
}

/// Device bytes a load on `backend` prices for SNAC's BF16 copy of
/// `elements` weights, built beside the F32 decoder for the load check:
/// zero off CUDA or unless an operator asked for it.
pub(crate) fn snac_bf16_device_bytes(backend: BackendKind, elements: u64) -> u64 {
    if backend == BackendKind::Cuda && snac_bf16_requested() {
        elements.saturating_mul(DType::BF16.size_in_bytes() as u64)
    } else {
        0
    }
}

pub struct SnacDecoder {
    cfg: SnacConfig,
    device: Device,
    /// What the weights and activations run in, and why.
    precision: CodecPrecision,
    /// Rows every call runs, padded with copies of its first row, so a
    /// window's shapes (and with them Candle's batched GEMM and im2col
    /// kernels, which cuBLAS may pick by batch count) are the same however
    /// many windows share the call. `None`: each call runs its own rows.
    call_rows: Option<usize>,
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
        let mut decoder =
            Self::from_var_builder(SnacConfig::snac_24khz(), &vb)?.into_device(device)?;
        if device.is_cuda() && snac_batching_enabled() {
            decoder.call_rows = Some(SNAC_CUDA_CALL_ROWS);
        }
        let requested = snac_bf16_requested();
        Ok(if device.is_cuda() {
            decoder.with_bf16_if(requested, reduced_precision::MAX_REL_L2)
        } else {
            decoder.precision = CodecPrecision::f32(
                requested,
                if requested {
                    "a BF16 SNAC runs on CUDA only"
                } else {
                    "F32"
                },
            );
            decoder
        })
    }

    /// This decoder in BF16 when `requested` and the load-time check keeps
    /// it: three seeded six-frame windows decoded through both, the BF16
    /// samples within `max_rel_l2` relative L2 error of the F32 ones
    /// ([`reduced_precision::MAX_REL_L2`] at load). Otherwise the F32
    /// decoder, with why.
    pub(crate) fn with_bf16_if(mut self, requested: bool, max_rel_l2: f64) -> Self {
        if !requested {
            self.precision = CodecPrecision::f32(false, format!("F32 ({SNAC_BF16_ENV} is off)"));
            return self;
        }
        let checked = self.with_dtype(DType::BF16).and_then(|candidate| {
            let reference = self.check_samples()?;
            let samples = candidate.check_samples()?;
            let verdict = reduced_precision::judge(&reference, &samples, DType::BF16, max_rel_l2);
            Ok((candidate, verdict))
        });
        match checked {
            Ok((mut candidate, (precision, true))) => {
                tracing::info!(reason = %precision.reason, "SNAC runs in BF16");
                candidate.precision = precision;
                candidate
            }
            Ok((_, (precision, false))) => {
                tracing::warn!(reason = %precision.reason, "SNAC stays F32");
                self.precision = precision;
                self
            }
            Err(error) => {
                tracing::warn!(%error, "SNAC's BF16 check failed; SNAC stays F32");
                self.precision = CodecPrecision::f32(true, format!("BF16 check failed: {error}"));
                self
            }
        }
    }

    /// What the decoder runs in, and why.
    pub(crate) fn precision(&self) -> &CodecPrecision {
        &self.precision
    }

    /// Every sample of three seeded six-frame windows decoded in one call,
    /// the precision check's input.
    fn check_samples(&self) -> Result<Vec<f32>> {
        let codes = self.check_codes();
        let samples = self.decode(
            &codes.iter().collect::<Vec<_>>(),
            &super::noise::snac_noise(&self.cfg, &check_windows(&codes)),
        )?;
        Ok(samples.into_iter().flatten().collect())
    }

    /// Three seeded six-frame windows of codes, the load checks' input.
    fn check_codes(&self) -> Vec<SnacCodes> {
        use crate::models::shared::noise::SplitMix64;
        (0..3u64)
            .map(|row| {
                let mut rng = SplitMix64::new(0x5eed + row);
                let size = self.cfg.codebook_size as u64;
                let mut codes = |count: usize| {
                    (0..count)
                        .map(|_| (rng.next_u64() % size) as u32)
                        .collect::<Vec<_>>()
                };
                SnacCodes {
                    coarse: codes(6),
                    medium: codes(12),
                    fine: codes(24),
                }
            })
            .collect()
    }

    /// Rows every call runs (padding included), when fixed.
    pub fn call_rows(&self) -> Option<usize> {
        self.call_rows
    }

    /// Whether rows sharing a call decode exactly as each does alone on this
    /// device: three seeded six-frame windows in one call against each in a
    /// call of its own, bit for bit.
    pub(crate) fn rows_are_independent(&self) -> Result<bool> {
        let codes = self.check_codes();
        let windows = check_windows(&codes);
        let together = self.decode(
            &codes.iter().collect::<Vec<_>>(),
            &super::noise::snac_noise(&self.cfg, &windows),
        )?;
        for (window, together) in windows.iter().zip(&together) {
            let alone = self.decode(
                &[window.1],
                &super::noise::snac_noise(&self.cfg, &[*window]),
            )?;
            let bits = |samples: &[f32]| samples.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
            if bits(&alone[0]) != bits(together) {
                return Ok(false);
            }
        }
        Ok(true)
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
            precision: CodecPrecision::f32(false, "F32"),
            call_rows: None,
            cfg,
        })
    }

    fn into_device(self, device: &Device) -> Result<Self> {
        if device.same_device(&self.device) {
            return Ok(self);
        }
        self.map_tensors(device, &|t| Ok(t.to_device(device)?))
    }

    /// Every weight in `dtype` (the activations follow them; samples come
    /// back F32).
    fn with_dtype(&self, dtype: DType) -> Result<Self> {
        let mut decoder = self.map_tensors(&self.device, &|t| Ok(t.to_dtype(dtype)?))?;
        decoder.precision.dtype = dtype;
        Ok(decoder)
    }

    /// This decoder with `move_t` applied to every tensor, on `device`.
    fn map_tensors(
        &self,
        device: &Device,
        move_t: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Self> {
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
            precision: self.precision.clone(),
            call_rows: self.call_rows,
            cfg: self.cfg.clone(),
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
        let Some(call_rows) = self.call_rows else {
            return self.decode_rows(rows, noise, frames);
        };
        // Every call runs `call_rows` rows: the rows in order, then copies
        // of each call's first row.
        let mut samples = Vec::with_capacity(batch);
        for (chunk, start) in rows.chunks(call_rows).zip((0..batch).step_by(call_rows)) {
            let mut padded = chunk.to_vec();
            padded.resize(call_rows, chunk[0]);
            let take = |block: usize, row: usize| {
                let len = frames * self.cfg.block_steps_per_frame(block);
                &noise[block][row * len..(row + 1) * len]
            };
            let padded_noise: SnacNoise = std::array::from_fn(|block| {
                (0..call_rows)
                    .flat_map(|slot| {
                        let row = if slot < chunk.len() {
                            start + slot
                        } else {
                            start
                        };
                        take(block, row).iter().copied()
                    })
                    .collect()
            });
            let mut decoded = self.decode_rows(&padded, &padded_noise, frames)?;
            decoded.truncate(chunk.len());
            samples.extend(decoded);
        }
        Ok(samples)
    }

    /// [`Self::decode`]'s rows (checked) in one call as they are.
    fn decode_rows(
        &self,
        rows: &[&SnacCodes],
        noise: &SnacNoise,
        frames: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let batch = rows.len();
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
            let noise = Tensor::from_slice(&noise[block], (batch, 1, steps), &self.device)?
                .to_dtype(self.precision.dtype)?;
            x = decoder_block.forward(&x, &noise)?;
        }
        let x = self.snake_out.forward(&x)?;
        let x = x
            .conv1d(&self.conv_out, 3, 1, 1, 1)?
            .broadcast_add(&self.conv_out_bias.reshape((1, 1, 1))?)?
            .tanh()?
            .to_dtype(DType::F32)?;
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

/// The load checks' noise windows of `codes`: each at frame 0, seeded by
/// its row.
fn check_windows(codes: &[SnacCodes]) -> Vec<(usize, &SnacCodes, u64)> {
    codes
        .iter()
        .enumerate()
        .map(|(row, codes)| (0, codes, row as u64 + 1))
        .collect()
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
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::models::shared::noise::SplitMix64;

    /// SNAC's layout at toy width (decoder 32 channels, latent 16), random
    /// weights, every `call_rows` rows a call.
    pub(crate) fn tiny_decoder(device: &Device, call_rows: Option<usize>) -> SnacDecoder {
        let cfg = SnacConfig {
            latent_dim: 16,
            decoder_dim: 32,
            ..SnacConfig::snac_24khz()
        };
        let mut rng = SplitMix64::new(23);
        let mut tensors = HashMap::new();
        let mut put = |name: String, shape: &[usize], scale: f32, offset: f32| {
            let count = shape.iter().product();
            let values = rng
                .standard_normals(count)
                .into_iter()
                .map(|v| v * scale + offset)
                .collect::<Vec<_>>();
            tensors.insert(name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap());
        };
        let latent = cfg.latent_dim;
        for index in 0..3 {
            put(
                format!("quantizer.quantizers.{index}.codebook.weight"),
                &[cfg.codebook_size, cfg.codebook_dim],
                1.0,
                0.0,
            );
        }
        let mut conv = |prefix: String, shape: [usize; 3], bias: Option<usize>| {
            put(
                format!("{prefix}.parametrizations.weight.original0"),
                &[shape[0], 1, 1],
                0.1,
                1.0,
            );
            put(
                format!("{prefix}.parametrizations.weight.original1"),
                &shape,
                1.0,
                0.0,
            );
            if let Some(len) = bias {
                put(format!("{prefix}.bias"), &[len], 0.1, 0.0);
            }
        };
        for index in 0..3 {
            conv(
                format!("quantizer.quantizers.{index}.out_proj"),
                [latent, cfg.codebook_dim, 1],
                Some(latent),
            );
        }
        let m = "decoder.model";
        conv(format!("{m}.0"), [latent, 1, 7], Some(latent));
        conv(
            format!("{m}.1"),
            [cfg.decoder_dim, latent, 1],
            Some(cfg.decoder_dim),
        );
        let mut channels = cfg.decoder_dim;
        for (index, &stride) in cfg.decoder_rates.iter().enumerate() {
            let (input, output) = (channels, channels / 2);
            let b = format!("{m}.{}.block", 2 + index);
            conv(format!("{b}.1"), [input, output, 2 * stride], Some(output));
            conv(format!("{b}.2.linear"), [output, output, 1], None);
            for unit in 3..6 {
                let u = format!("{b}.{unit}.block");
                conv(format!("{u}.1"), [output, 1, 7], Some(output));
                conv(format!("{u}.3"), [output, output, 1], Some(output));
            }
            channels = output;
        }
        conv(format!("{m}.7"), [1, channels, 7], Some(1));
        drop(conv);
        let mut channels = cfg.decoder_dim;
        for index in 0..4 {
            let b = format!("{m}.{}.block", 2 + index);
            put(format!("{b}.0.alpha"), &[1, channels, 1], 0.1, 1.0);
            channels /= 2;
            for unit in 3..6 {
                for snake in [0, 2] {
                    put(
                        format!("{b}.{unit}.block.{snake}.alpha"),
                        &[1, channels, 1],
                        0.1,
                        1.0,
                    );
                }
            }
        }
        put(format!("{m}.6.alpha"), &[1, channels, 1], 0.1, 1.0);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let mut decoder = SnacDecoder::from_var_builder(cfg, &vb)
            .unwrap()
            .into_device(device)
            .unwrap();
        decoder.call_rows = call_rows;
        decoder
    }

    /// `frames` random frames of codes.
    pub(crate) fn random_codes(seed: u64, frames: usize) -> SnacCodes {
        let mut rng = SplitMix64::new(seed);
        let mut codes = |count: usize| {
            (0..count)
                .map(|_| (rng.next_u64() % 4_096) as u32)
                .collect::<Vec<_>>()
        };
        SnacCodes {
            coarse: codes(frames),
            medium: codes(2 * frames),
            fine: codes(4 * frames),
        }
    }

    /// Rows decode the same in one call as each alone, with and without the
    /// fixed call rows, and the fixed rows change nothing on the CPU.
    #[test]
    fn cpu_rows_decode_alone_as_together_with_any_call_rows() {
        use super::super::noise::snac_noise;
        let rows = (0..5).map(|seed| random_codes(seed, 6)).collect::<Vec<_>>();
        let windows = rows
            .iter()
            .enumerate()
            .map(|(index, codes)| (index * 3, codes, 40 + index as u64))
            .collect::<Vec<_>>();
        let mut expected = None;
        for call_rows in [None, Some(4), Some(2)] {
            let snac = tiny_decoder(&Device::Cpu, call_rows);
            assert!(snac.rows_are_independent().unwrap());
            let together = snac
                .decode(
                    &rows.iter().collect::<Vec<_>>(),
                    &snac_noise(snac.config(), &windows),
                )
                .unwrap();
            assert_eq!(together.len(), rows.len());
            for (window, together) in windows.iter().zip(&together) {
                let alone = snac
                    .decode(&[window.1], &snac_noise(snac.config(), &[*window]))
                    .unwrap();
                assert_eq!(&alone[0], together);
                assert_eq!(together.len(), 6 * 2_048);
            }
            match &expected {
                None => expected = Some(together),
                Some(expected) => assert_eq!(expected, &together),
            }
        }
    }

    /// The BF16 check keeps a BF16 SNAC within its bound and refuses one
    /// past it. The toy SNAC's random weights are ill-conditioned (rounding
    /// them to BF16 alone moves its samples ~23%), so at the load's 5% bound
    /// it stays F32 with the measured error recorded, and at a loose bound
    /// the BF16 decoder is kept and decodes finite samples that are not
    /// F32's bits (so BF16 ran), rows still decoding together as alone.
    /// Without the request the decoder stays F32.
    #[test]
    fn the_bf16_check_keeps_a_close_snac_and_refuses_a_far_one() {
        use super::super::noise::snac_noise;
        let rows = (0..3)
            .map(|seed| random_codes(seed + 70, 5))
            .collect::<Vec<_>>();
        let windows = rows
            .iter()
            .enumerate()
            .map(|(index, codes)| (index * 2, codes, 90 + index as u64))
            .collect::<Vec<_>>();
        let decode = |snac: &SnacDecoder| {
            let noise = snac_noise(snac.config(), &windows);
            snac.decode(&rows.iter().collect::<Vec<_>>(), &noise)
                .unwrap()
                .concat()
        };
        let reference = decode(&tiny_decoder(&Device::Cpu, Some(4)));

        let refused =
            tiny_decoder(&Device::Cpu, Some(4)).with_bf16_if(true, reduced_precision::MAX_REL_L2);
        let precision = refused.precision();
        assert_eq!(precision.dtype, DType::F32, "{precision:?}");
        assert!(precision.requested);
        assert!(precision.rel_l2.unwrap() > reduced_precision::MAX_REL_L2);
        assert!(precision.reason.contains("past"), "{}", precision.reason);
        assert_eq!(decode(&refused), reference);

        let kept = tiny_decoder(&Device::Cpu, Some(4)).with_bf16_if(true, 1.0);
        assert_eq!(
            kept.precision().dtype,
            DType::BF16,
            "{:?}",
            kept.precision()
        );
        assert!(kept.precision().rel_l2.unwrap() <= 1.0);
        assert!(kept.rows_are_independent().unwrap());
        let reduced = decode(&kept);
        assert_eq!(reduced.len(), reference.len());
        assert!(reduced.iter().all(|sample| sample.is_finite()));
        assert_ne!(reduced, reference, "BF16 ran");

        let off = tiny_decoder(&Device::Cpu, Some(4)).with_bf16_if(false, 1.0);
        assert_eq!(off.precision().dtype, DType::F32);
        assert!(off.precision().reason.contains(SNAC_BF16_ENV));
        assert_eq!(decode(&off), reference);
    }

    /// Only a CUDA load that asks for BF16 prices its copy.
    #[test]
    fn only_a_cuda_bf16_request_prices_the_copy() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os(SNAC_BF16_ENV);
        std::env::remove_var(SNAC_BF16_ENV);
        let off = snac_bf16_device_bytes(BackendKind::Cuda, 1_000);
        std::env::set_var(SNAC_BF16_ENV, "1");
        let cuda = snac_bf16_device_bytes(BackendKind::Cuda, 1_000);
        let metal = snac_bf16_device_bytes(BackendKind::Metal, 1_000);
        match previous {
            Some(value) => std::env::set_var(SNAC_BF16_ENV, value),
            None => std::env::remove_var(SNAC_BF16_ENV),
        }
        assert_eq!((off, cuda, metal), (0, 2_000, 0));
    }

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
