//! Mimi's causal convolutions: whole-sequence forms (transformers'
//! `MimiConv1d` / `MimiConvTranspose1d` with causal padding) and the
//! streaming forms the decoder runs a few frames at a time.
//!
//! Streaming carries, per stateful layer, the input history a causal
//! convolution reads to the left of a chunk, or the overlap a transposed
//! convolution spills past a chunk's end (stored without the bias, which the
//! next chunk's own output already carries). With zero initial state the
//! concatenated chunks equal the whole-sequence output up to summation order.

use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

use crate::error::{Error, Result};
use crate::kernels::depthwise_conv::depthwise_conv_transpose1d;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PadMode {
    Constant,
    Replicate,
}

/// `MimiConv1d` with causal padding: `padding_total = (k − 1)·d + 1 − s`
/// on the left, plus the extra right padding that makes the last window
/// whole.
pub(crate) struct CausalConv1d {
    conv: Conv1d,
    stride: usize,
    effective_kernel: usize,
    padding_total: usize,
    pad_mode: PadMode,
    in_channels: usize,
}

impl CausalConv1d {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        bias: bool,
        pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups: 1,
            ..Default::default()
        };
        let conv = if bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)?
        };
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        Ok(Self {
            conv,
            stride,
            effective_kernel,
            padding_total: effective_kernel - stride,
            pad_mode,
            in_channels,
        })
    }

    /// Input samples left of a chunk a streaming stride-1 convolution reads.
    pub(crate) fn history(&self) -> usize {
        self.padding_total
    }

    pub(crate) fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// transformers' `_get_extra_padding_for_conv1d`.
    fn extra_padding(&self, length: usize) -> usize {
        let n_frames = (length as f64 - self.effective_kernel as f64 + self.padding_total as f64)
            / self.stride as f64
            + 1.0;
        let n_frames = n_frames.ceil() as i64 - 1;
        let ideal = n_frames * self.stride as i64 + self.effective_kernel as i64
            - self.padding_total as i64;
        (ideal - length as i64).max(0) as usize
    }

    /// The whole sequence `x` `[B, C, T]`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let extra = self.extra_padding(x.dim(2)?);
        let x = match self.pad_mode {
            PadMode::Constant => x.pad_with_zeros(2, self.padding_total, extra)?,
            PadMode::Replicate => x.pad_with_same(2, self.padding_total, extra)?,
        };
        Ok(self.conv.forward(&x)?)
    }

    /// The next chunk `x` `[1, C, T]` of a stride-1 convolution, given and
    /// updating its left `history` `[1, C, padding_total]`.
    pub(crate) fn forward_stream(&self, x: &Tensor, history: &mut Tensor) -> Result<Tensor> {
        if self.stride != 1 {
            return Err(Error::InvalidInput(
                "Mimi streams only stride-1 convolutions".into(),
            ));
        }
        if self.padding_total == 0 {
            return Ok(self.conv.forward(x)?);
        }
        let joined = Tensor::cat(&[&*history, x], 2)?;
        let length = joined.dim(2)?;
        *history = joined
            .narrow(2, length - self.padding_total, self.padding_total)?
            .contiguous()?;
        Ok(self.conv.forward(&joined)?)
    }
}

/// `MimiConvTranspose1d` with causal trimming (`trim_right_ratio` 1): the
/// output keeps `T·s` samples and drops the `k − s` that spill past the end.
pub(crate) struct CausalConvTranspose1d {
    kind: TransposeKind,
    stride: usize,
    kernel: usize,
    out_channels: usize,
}

enum TransposeKind {
    Dense {
        conv: ConvTranspose1d,
    },
    /// groups == channels: weight `[C, 1, K]`, no bias.
    Depthwise {
        weight: Tensor,
    },
}

impl CausalConvTranspose1d {
    pub(crate) fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let conv = candle_nn::conv_transpose1d(in_channels, out_channels, kernel_size, cfg, vb)?;
        Ok(Self {
            kind: TransposeKind::Dense { conv },
            stride,
            kernel: kernel_size,
            out_channels,
        })
    }

    /// The depthwise, bias-free form Mimi's upsample uses.
    pub(crate) fn load_depthwise(
        channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((channels, 1, kernel_size), "weight")?;
        Ok(Self {
            kind: TransposeKind::Depthwise { weight },
            stride,
            kernel: kernel_size,
            out_channels: channels,
        })
    }

    /// Output samples a chunk spills past its end.
    pub(crate) fn overlap(&self) -> usize {
        self.kernel - self.stride
    }

    pub(crate) fn out_channels(&self) -> usize {
        self.out_channels
    }

    fn bias(&self) -> Option<&Tensor> {
        match &self.kind {
            TransposeKind::Dense { conv } => conv.bias(),
            TransposeKind::Depthwise { .. } => None,
        }
    }

    /// The untrimmed output, `(T − 1)·s + k` samples.
    fn full(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match &self.kind {
            TransposeKind::Dense { conv } => conv.forward(x)?,
            TransposeKind::Depthwise { weight } => {
                depthwise_conv_transpose1d(x, weight, None, self.stride, 0, 0)?
            }
        })
    }

    /// The whole sequence `x` `[B, C, T]` → `[B, C', T·s]`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let produced = x.dim(2)? * self.stride;
        Ok(self.full(x)?.narrow(2, 0, produced)?)
    }

    /// The next chunk `x` `[1, C, T]`, adding the previous chunk's spill
    /// `tail` `[1, C', k − s]` and replacing it with this chunk's.
    pub(crate) fn forward_stream(&self, x: &Tensor, tail: &mut Tensor) -> Result<Tensor> {
        let overlap = self.overlap();
        let produced = x.dim(2)? * self.stride;
        if overlap > produced {
            return Err(Error::InvalidInput(
                "Mimi transposed-convolution chunk is shorter than its overlap".into(),
            ));
        }
        let y = self.full(x)?;
        if overlap == 0 {
            return Ok(y.narrow(2, 0, produced)?);
        }
        let mut spill = y.narrow(2, produced, overlap)?;
        if let Some(bias) = self.bias() {
            spill = spill.broadcast_sub(&bias.reshape((1, (), 1))?)?;
        }
        let head = (y.narrow(2, 0, overlap)? + &*tail)?;
        let out = if produced > overlap {
            Tensor::cat(&[&head, &y.narrow(2, overlap, produced - overlap)?], 2)?
        } else {
            head
        };
        *tail = spill.contiguous()?;
        Ok(out)
    }
}

/// SEANet's residual block: ELU → conv k (dilation d) → ELU → conv 1, plus
/// the identity.
pub(crate) struct ResnetBlock {
    pub(crate) conv1: CausalConv1d,
    pub(crate) conv2: CausalConv1d,
}

impl ResnetBlock {
    pub(crate) fn load(
        dim: usize,
        compress: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = dim / compress;
        Ok(Self {
            conv1: CausalConv1d::load(
                dim,
                hidden,
                kernel_size,
                1,
                dilation,
                true,
                PadMode::Constant,
                vb.pp("block.1.conv"),
            )?,
            conv2: CausalConv1d::load(
                hidden,
                dim,
                1,
                1,
                1,
                true,
                PadMode::Constant,
                vb.pp("block.3.conv"),
            )?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&x.elu(1.0)?)?;
        let h = self.conv2.forward(&h.elu(1.0)?)?;
        Ok((x + h)?)
    }

    /// The streaming form; only `conv1` keeps history (`conv2` is 1-wide).
    pub(crate) fn forward_stream(&self, x: &Tensor, history: &mut Tensor) -> Result<Tensor> {
        let h = self.conv1.forward_stream(&x.elu(1.0)?, history)?;
        let h = self.conv2.forward(&h.elu(1.0)?)?;
        Ok((x + h)?)
    }
}
