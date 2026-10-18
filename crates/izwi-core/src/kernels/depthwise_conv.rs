//! Depthwise (groups == channels) 1-D convolution as shift-and-accumulate,
//! plus the transposed and strided 2-D variants built the same way.
//!
//! Candle's grouped `conv1d` splits the input and kernel into `groups` chunks
//! and runs one single-group convolution per chunk before concatenating them
//! (candle-core `conv.rs`). For a depthwise layer that is one im2col/matmul
//! pair per channel: 1024-2048 dispatches per layer, 25k-100k per encoder
//! pass. Here the output is the sum over the K taps of a shifted input slice
//! scaled by that tap's per-channel weight,
//!
//! `y[b, c, t] = bias[c] + Σ_k x_pad[b, c, t + k·dilation] · w[c, k]`,
//!
//! which is 2K + 2 element-wise ops whatever the channel count. It uses only
//! ordinary tensor ops, so it is valid on every backend and dtype. Half
//! precision inputs accumulate in F32 and are cast back once.
//!
//! A folded inference BatchNorm (or any per-channel affine applied after the
//! convolution) is expressed through [`fold_channel_affine`] at load time and
//! passed as `weight`/`bias`.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Result, Tensor};

static SHIFT_ACCUMULATE_CALLS: AtomicU64 = AtomicU64::new(0);
static SHIFT_ACCUMULATE_TAPS: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of depthwise convolutions served by the shared
/// shift-and-accumulate op (diagnostics: shows the path engaged).
pub fn depthwise_conv1d_calls() -> u64 {
    SHIFT_ACCUMULATE_CALLS.load(Ordering::Relaxed)
}

/// Process-wide count of taps accumulated by the shared op. Each tap is one
/// multiply plus one add, independent of the channel count.
pub fn depthwise_conv1d_taps() -> u64 {
    SHIFT_ACCUMULATE_TAPS.load(Ordering::Relaxed)
}

/// Memory layout of the activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvLayout {
    /// `(batch, channels, time)`, as Candle's `conv1d`.
    ChannelsFirst,
    /// `(batch, time, channels)`.
    ChannelsLast,
}

/// Geometry of a stride-1 depthwise convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthwiseConv1dSpec {
    pub pad_left: usize,
    pub pad_right: usize,
    pub dilation: usize,
    pub layout: ConvLayout,
}

impl DepthwiseConv1dSpec {
    /// Candle-style symmetric zero padding on both sides.
    pub fn symmetric(padding: usize) -> Self {
        Self::padded(padding, padding)
    }

    /// Explicit left/right zero padding (`0, 0` for pre-padded inputs).
    pub fn padded(pad_left: usize, pad_right: usize) -> Self {
        Self {
            pad_left,
            pad_right,
            dilation: 1,
            layout: ConvLayout::ChannelsFirst,
        }
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn with_layout(mut self, layout: ConvLayout) -> Self {
        self.layout = layout;
        self
    }
}

/// Whether `weight` is a depthwise kernel for a `stride`-strided convolution
/// with `groups` groups over `channels` channels that [`depthwise_conv1d`]
/// can serve.
pub fn is_depthwise_conv1d(weight: &Tensor, channels: usize, groups: usize, stride: usize) -> bool {
    stride == 1
        && groups == channels
        && channels > 0
        && matches!(weight.dims(), [c, 1, k] if *c == channels && *k > 0)
}

/// Depthwise 1-D convolution. `weight` is `(C, 1, K)` or `(C, K)`; `bias` is
/// `(C,)`. Matches `conv1d(pad(x), weight, padding=0, stride=1, dilation,
/// groups=C) + bias` up to summation order.
pub fn depthwise_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv1dSpec,
) -> Result<Tensor> {
    let (channel_dim, time_dim) = match spec.layout {
        ConvLayout::ChannelsFirst => (1, 2),
        ConvLayout::ChannelsLast => (2, 1),
    };
    let (_batch, _, _) = x.dims3()?;
    let channels = x.dim(channel_dim)?;
    let taps = match weight.dims() {
        [c, 1, k] | [c, k] if *c == channels && *k > 0 => *k,
        dims => candle_core::bail!(
            "depthwise conv1d weight {dims:?} does not match {channels} channels"
        ),
    };
    let dilation = spec.dilation.max(1);
    let padded_len = x.dim(time_dim)? + spec.pad_left + spec.pad_right;
    let span = dilation * (taps - 1) + 1;
    if padded_len < span {
        candle_core::bail!(
            "depthwise conv1d input of padded length {padded_len} is shorter than the kernel span {span}"
        );
    }
    let out_len = padded_len - span + 1;

    let out_dtype = x.dtype();
    let acc_dtype = match out_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    let x = x.to_dtype(acc_dtype)?;
    let x = if spec.pad_left > 0 || spec.pad_right > 0 {
        x.pad_with_zeros(time_dim, spec.pad_left, spec.pad_right)?
    } else {
        x
    };
    // Per-tap weight rows shaped to broadcast against the activation.
    let weight = weight.reshape((channels, taps))?.to_dtype(acc_dtype)?;
    let weight = match spec.layout {
        ConvLayout::ChannelsFirst => weight.reshape((1, channels, taps))?,
        ConvLayout::ChannelsLast => weight.t()?.contiguous()?.reshape((1, taps, channels))?,
    };
    let tap_weight = |k: usize| -> Result<Tensor> {
        match spec.layout {
            ConvLayout::ChannelsFirst => weight.narrow(2, k, 1),
            ConvLayout::ChannelsLast => weight.narrow(1, k, 1),
        }
    };

    let mut acc = x
        .narrow(time_dim, 0, out_len)?
        .broadcast_mul(&tap_weight(0)?)?;
    for k in 1..taps {
        let shifted = x.narrow(time_dim, k * dilation, out_len)?;
        acc = (acc + shifted.broadcast_mul(&tap_weight(k)?)?)?;
    }
    if let Some(bias) = bias {
        let bias = bias.to_dtype(acc_dtype)?;
        let bias = match spec.layout {
            ConvLayout::ChannelsFirst => bias.reshape((1, channels, 1))?,
            ConvLayout::ChannelsLast => bias.reshape((1, 1, channels))?,
        };
        acc = acc.broadcast_add(&bias)?;
    }

    SHIFT_ACCUMULATE_CALLS.fetch_add(1, Ordering::Relaxed);
    SHIFT_ACCUMULATE_TAPS.fetch_add(taps as u64, Ordering::Relaxed);
    acc.to_dtype(out_dtype)
}

static TRANSPOSED_CALLS: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of depthwise transposed convolutions served by
/// [`depthwise_conv_transpose1d`].
pub fn depthwise_conv_transpose1d_calls() -> u64 {
    TRANSPOSED_CALLS.load(Ordering::Relaxed)
}

/// Whether `weight` is a depthwise transposed-conv kernel (`groups == C`,
/// one output channel per group, dilation 1) that
/// [`depthwise_conv_transpose1d`] can serve.
pub fn is_depthwise_conv_transpose1d(
    weight: &Tensor,
    channels: usize,
    groups: usize,
    dilation: usize,
) -> bool {
    dilation == 1
        && groups == channels
        && channels > 0
        && matches!(weight.dims(), [c, 1, k] if *c == channels && *k > 0)
}

/// Depthwise (`groups == channels`) transposed 1-D convolution on
/// `(batch, channels, time)`, matching Candle's grouped `conv_transpose1d`
/// (`weight` `(C, 1, K)`, dilation 1) up to summation order.
///
/// Candle splits a grouped transposed conv into one transposed conv per
/// channel plus a concatenation. Here each output phase `r` (of `stride`) is
/// a sum of shifted input slices, `phase_r[m] = Σ_q w[r + stride·q]·x[m − q]`;
/// the phases are interleaved and the padding trimmed. That is about
/// `K + stride + 3` element-wise ops whatever the channel count.
pub fn depthwise_conv_transpose1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<Tensor> {
    let (batch, channels, len) = x.dims3()?;
    let taps = match weight.dims() {
        [c, 1, k] if *c == channels && *k > 0 => *k,
        dims => candle_core::bail!(
            "depthwise conv_transpose1d weight {dims:?} does not match {channels} channels"
        ),
    };
    let stride = stride.max(1);
    let full_len = (len.max(1) - 1) * stride + taps + output_padding;
    if len == 0 || full_len < 2 * padding + 1 {
        candle_core::bail!("depthwise conv_transpose1d input of length {len} is too short");
    }
    let out_len = full_len - 2 * padding;
    let phase_len = (out_len + padding).div_ceil(stride);
    let shifts = taps.div_ceil(stride);

    let out_dtype = x.dtype();
    let acc_dtype = match out_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    // x[m − q] for q < shifts and m < phase_len: shift q starts at
    // `shifts − 1 − q` in the left-padded input.
    let right_pad = phase_len.saturating_sub(len);
    let x = x
        .to_dtype(acc_dtype)?
        .pad_with_zeros(2, shifts - 1, right_pad)?;
    let weight = weight.reshape((1, channels, taps))?.to_dtype(acc_dtype)?;
    let mut phases = Vec::with_capacity(stride);
    for r in 0..stride {
        let mut phase: Option<Tensor> = None;
        for q in 0..shifts {
            let k = r + stride * q;
            if k >= taps {
                break;
            }
            let term = x
                .narrow(2, shifts - 1 - q, phase_len)?
                .broadcast_mul(&weight.narrow(2, k, 1)?)?;
            phase = Some(match phase {
                Some(acc) => (acc + term)?,
                None => term,
            });
        }
        phases.push(match phase {
            Some(phase) => phase,
            None => Tensor::zeros((batch, channels, phase_len), acc_dtype, x.device())?,
        });
    }
    let mut out = Tensor::stack(&phases, 3)?
        .reshape((batch, channels, phase_len * stride))?
        .narrow(2, padding, out_len)?;
    if let Some(bias) = bias {
        out = out.broadcast_add(&bias.to_dtype(acc_dtype)?.reshape((1, channels, 1))?)?;
    }
    TRANSPOSED_CALLS.fetch_add(1, Ordering::Relaxed);
    out.to_dtype(out_dtype)
}

static DEPTHWISE_CONV2D_CALLS: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of depthwise 2-D convolutions served by
/// [`depthwise_conv2d`].
pub fn depthwise_conv2d_calls() -> u64 {
    DEPTHWISE_CONV2D_CALLS.load(Ordering::Relaxed)
}

/// Geometry of a depthwise 2-D convolution: one stride and one zero padding
/// for both spatial axes, dilation 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthwiseConv2dSpec {
    pub padding: usize,
    pub stride: usize,
}

/// Depthwise (`groups == channels`) 2-D convolution on
/// `(batch, channels, height, width)`, matching Candle's grouped `conv2d`
/// (`weight` `(C, 1, KH, KW)`, dilation 1) up to summation order.
///
/// Candle runs one convolution per channel and concatenates them. Here the
/// zero-padded input is viewed as `(…, H/s, s, W/s, s)`, so tap `(i, j)` is
/// the strided view at rows `i + s·q` and columns `j + s·p` without a copy,
/// and the output is the sum of the KH·KW scaled views: about
/// `2·KH·KW + 4` ops whatever the channel count.
pub fn depthwise_conv2d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
) -> Result<Tensor> {
    let (batch, channels, height, width) = x.dims4()?;
    let (kernel_h, kernel_w) = match weight.dims() {
        [c, 1, kh, kw] if *c == channels && *kh > 0 && *kw > 0 => (*kh, *kw),
        dims => candle_core::bail!(
            "depthwise conv2d weight {dims:?} does not match {channels} channels"
        ),
    };
    let stride = spec.stride.max(1);
    let (padded_h, padded_w) = (height + 2 * spec.padding, width + 2 * spec.padding);
    if padded_h < kernel_h || padded_w < kernel_w {
        candle_core::bail!(
            "depthwise conv2d input {padded_h}x{padded_w} (padded) is smaller than the kernel {kernel_h}x{kernel_w}"
        );
    }
    let out_h = (padded_h - kernel_h) / stride + 1;
    let out_w = (padded_w - kernel_w) / stride + 1;
    // Each stride phase must hold the furthest tap offset plus the output.
    let phase_h = (kernel_h - 1) / stride + out_h;
    let phase_w = (kernel_w - 1) / stride + out_w;

    let out_dtype = x.dtype();
    let acc_dtype = match out_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    // Zero-pad (or trim never-read trailing rows) to exactly
    // `stride * phase` along each spatial axis.
    let fit = |x: Tensor, dim: usize, size: usize, len: usize| -> Result<Tensor> {
        let right = len.saturating_sub(size + spec.padding);
        let x = x.pad_with_zeros(dim, spec.padding, right)?;
        if x.dim(dim)? > len {
            x.narrow(dim, 0, len)
        } else {
            Ok(x)
        }
    };
    let x = fit(x.to_dtype(acc_dtype)?, 2, height, stride * phase_h)?;
    let x = fit(x, 3, width, stride * phase_w)?;
    // A free view: row `r = stride·q + a` is `grid[.., q, a, ..]`, and the
    // same for columns.
    let grid = x.reshape((batch, channels, phase_h, stride, phase_w, stride))?;
    // One contiguous `(1, C, 1, 1, 1, 1)` weight row per tap.
    let taps = weight
        .reshape((channels, kernel_h * kernel_w))?
        .to_dtype(acc_dtype)?
        .t()?
        .contiguous()?;
    let mut acc: Option<Tensor> = None;
    for i in 0..kernel_h {
        for j in 0..kernel_w {
            let view = grid
                .narrow(2, i / stride, out_h)?
                .narrow(3, i % stride, 1)?
                .narrow(4, j / stride, out_w)?
                .narrow(5, j % stride, 1)?;
            let tap = taps
                .narrow(0, i * kernel_w + j, 1)?
                .reshape((1, channels, 1, 1, 1, 1))?;
            let term = view.broadcast_mul(&tap)?;
            acc = Some(match acc {
                Some(acc) => (acc + term)?,
                None => term,
            });
        }
    }
    let mut out = acc
        .expect("the kernel has at least one tap")
        .reshape((batch, channels, out_h, out_w))?;
    if let Some(bias) = bias {
        out = out.broadcast_add(&bias.to_dtype(acc_dtype)?.reshape((1, channels, 1, 1))?)?;
    }
    DEPTHWISE_CONV2D_CALLS.fetch_add(1, Ordering::Relaxed);
    out.to_dtype(out_dtype)
}

/// Fold a per-channel affine `y' = y * scale + shift` applied after a
/// depthwise convolution into its weight and bias.
pub fn fold_channel_affine(
    weight: &Tensor,
    bias: Option<&Tensor>,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let channels = scale.dims1()?;
    let mut scale_shape = vec![1usize; weight.rank()];
    scale_shape[0] = channels;
    let folded_weight = weight
        .broadcast_mul(&scale.reshape(scale_shape)?)?
        .contiguous()?;
    let folded_bias = match bias {
        Some(bias) => bias.broadcast_mul(scale)?.broadcast_add(shift)?,
        None => shift.clone(),
    };
    Ok((folded_weight, folded_bias.contiguous()?))
}

/// The per-channel `(scale, shift)` of an inference BatchNorm:
/// `scale = gamma / sqrt(var + eps)`, `shift = beta - mean * scale`.
pub fn batch_norm_affine(
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f64,
    weight_and_bias: Option<(&Tensor, &Tensor)>,
) -> Result<(Tensor, Tensor)> {
    let std = (running_var + eps)?.sqrt()?;
    let (scale, shift) = match weight_and_bias {
        Some((gamma, beta)) => {
            let scale = gamma.broadcast_div(&std)?;
            let shift = beta.broadcast_sub(&scale.broadcast_mul(running_mean)?)?;
            (scale, shift)
        }
        None => {
            let scale = std.recip()?;
            let shift = scale.broadcast_mul(running_mean)?.neg()?;
            (scale, shift)
        }
    };
    Ok((scale, shift))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    struct XorShift(u64);

    impl XorShift {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn tensor(&mut self, shape: &[usize]) -> Tensor {
            let n = shape.iter().product::<usize>();
            let data = (0..n).map(|_| self.next_f32()).collect::<Vec<_>>();
            Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims());
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| {
                let d = (x - y).abs();
                if d.is_finite() {
                    d
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, f32::max)
    }

    fn candle_reference(
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        pad_left: usize,
        pad_right: usize,
        dilation: usize,
    ) -> Tensor {
        let channels = x.dim(1).unwrap();
        let padded = x.pad_with_zeros(2, pad_left, pad_right).unwrap();
        let y = padded.conv1d(weight, 0, 1, dilation, channels).unwrap();
        match bias {
            Some(bias) => y
                .broadcast_add(&bias.reshape((1, channels, 1)).unwrap())
                .unwrap(),
            None => y,
        }
    }

    #[test]
    fn max_abs_diff_saturates_nan() {
        let a = Tensor::new(&[f32::NAN, 0.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[0.0f32, 0.0], &Device::Cpu).unwrap();
        assert_eq!(max_abs_diff(&a, &b), f32::INFINITY);
    }

    #[test]
    fn shift_accumulate_matches_candle_grouped_conv1d() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        // (batch, channels, time, taps, pad_left, pad_right, dilation, bias)
        let cases = [
            (1, 64, 37, 9, 4, 4, 1, false), // Parakeet / LFM symmetric
            (2, 48, 50, 15, 7, 7, 1, true), // Granite odd kernel
            (1, 32, 41, 4, 2, 1, 1, false), // Granite-style even kernel
            (2, 16, 29, 9, 8, 0, 1, true),  // causal (Nemotron / VibeVoice)
            (1, 24, 33, 7, 12, 0, 2, true), // dilated causal
            (1, 8, 9, 9, 0, 0, 1, false),   // pre-padded, single output
            (1, 12, 20, 1, 0, 0, 1, true),  // pointwise depthwise
        ];
        for (batch, channels, time, taps, left, right, dilation, with_bias) in cases {
            let x = rng.tensor(&[batch, channels, time]);
            let weight = rng.tensor(&[channels, 1, taps]);
            let bias = with_bias.then(|| rng.tensor(&[channels]));
            let expected = candle_reference(&x, &weight, bias.as_ref(), left, right, dilation);
            let spec = DepthwiseConv1dSpec::padded(left, right).with_dilation(dilation);
            let actual = depthwise_conv1d(&x, &weight, bias.as_ref(), spec).unwrap();
            let diff = max_abs_diff(&expected, &actual);
            assert!(
                diff <= 1e-5,
                "case {:?}: max |diff| {diff}",
                (batch, channels, time, taps, left, right, dilation)
            );

            // (C, K) weights and the channels-last layout give the same result.
            let last = depthwise_conv1d(
                &x.transpose(1, 2).unwrap().contiguous().unwrap(),
                &weight.reshape((channels, taps)).unwrap(),
                bias.as_ref(),
                spec.with_layout(ConvLayout::ChannelsLast),
            )
            .unwrap()
            .transpose(1, 2)
            .unwrap();
            assert!(max_abs_diff(&expected, &last) <= 1e-5);
        }
    }

    #[test]
    fn shift_accumulate_accepts_strided_inputs_and_half_precision() {
        let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
        let (channels, time, taps) = (40, 64, 9);
        // A transposed (non-contiguous) channels-first view.
        let x = rng.tensor(&[1, time, channels]).transpose(1, 2).unwrap();
        let weight = rng.tensor(&[channels, 1, taps]);
        let bias = rng.tensor(&[channels]);
        let expected = candle_reference(&x.contiguous().unwrap(), &weight, Some(&bias), 4, 4, 1);
        let actual =
            depthwise_conv1d(&x, &weight, Some(&bias), DepthwiseConv1dSpec::symmetric(4)).unwrap();
        assert!(max_abs_diff(&expected, &actual) <= 1e-5);

        let half = depthwise_conv1d(
            &x.to_dtype(DType::F16).unwrap(),
            &weight.to_dtype(DType::F16).unwrap(),
            Some(&bias.to_dtype(DType::F16).unwrap()),
            DepthwiseConv1dSpec::symmetric(4),
        )
        .unwrap();
        assert_eq!(half.dtype(), DType::F16);
        // F16 storage of inputs and the output bounds the error, not the sum.
        let half = half.to_dtype(DType::F32).unwrap();
        assert!(max_abs_diff(&expected, &half) <= 2e-2);
    }

    #[test]
    fn folded_batch_norm_matches_conv_then_batch_norm() {
        let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
        let (channels, time, taps) = (32, 45, 15);
        let x = rng.tensor(&[2, channels, time]);
        let weight = rng.tensor(&[channels, 1, taps]);
        let mean = rng.tensor(&[channels]);
        let var = rng
            .tensor(&[channels])
            .abs()
            .unwrap()
            .affine(1.0, 0.1)
            .unwrap();
        let gamma = rng.tensor(&[channels]);
        let beta = rng.tensor(&[channels]);
        let bn = candle_nn::BatchNorm::new(
            channels,
            mean.clone(),
            var.clone(),
            gamma.clone(),
            beta.clone(),
            1e-5,
        )
        .unwrap();
        let conv = candle_reference(&x, &weight, None, 7, 7, 1);
        let expected = candle_nn::ModuleT::forward_t(&bn, &conv, false).unwrap();

        let (scale, shift) = batch_norm_affine(&mean, &var, 1e-5, Some((&gamma, &beta))).unwrap();
        let (w, b) = fold_channel_affine(&weight, None, &scale, &shift).unwrap();
        let actual = depthwise_conv1d(&x, &w, Some(&b), DepthwiseConv1dSpec::symmetric(7)).unwrap();
        let diff = max_abs_diff(&expected, &actual);
        // Relative to the output scale (|y| up to ~30 here), 1e-5 relative.
        let scale_max = expected
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff <= 1e-5 * scale_max.max(1.0), "max |diff| {diff}");
    }

    #[test]
    fn rejects_mismatched_weights_and_short_inputs() {
        let x = Tensor::zeros((1, 4, 3), DType::F32, &Device::Cpu).unwrap();
        let wrong = Tensor::zeros((5, 1, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(depthwise_conv1d(&x, &wrong, None, DepthwiseConv1dSpec::symmetric(1)).is_err());
        let long = Tensor::zeros((4, 1, 9), DType::F32, &Device::Cpu).unwrap();
        assert!(depthwise_conv1d(&x, &long, None, DepthwiseConv1dSpec::padded(0, 0)).is_err());
        assert!(is_depthwise_conv1d(&long, 4, 4, 1));
        assert!(!is_depthwise_conv1d(&long, 4, 4, 2));
        assert!(!is_depthwise_conv1d(&long, 4, 1, 1));
    }

    #[test]
    fn counts_calls_and_taps() {
        let x = Tensor::zeros((1, 4, 16), DType::F32, &Device::Cpu).unwrap();
        let w = Tensor::zeros((4, 1, 9), DType::F32, &Device::Cpu).unwrap();
        let (calls, taps) = (depthwise_conv1d_calls(), depthwise_conv1d_taps());
        depthwise_conv1d(&x, &w, None, DepthwiseConv1dSpec::symmetric(4)).unwrap();
        // Other tests may run concurrently; the counters only grow.
        assert!(depthwise_conv1d_calls() > calls);
        assert!(depthwise_conv1d_taps() >= taps + 9);
    }

    /// CPU micro-benchmark against Candle's per-group loop (Parakeet geometry:
    /// 1024 channels, kernel 9, ~60 s of encoder frames).
    #[test]
    #[ignore]
    fn bench_shift_accumulate_vs_candle_grouped_conv1d() {
        let mut rng = XorShift(0x0123_4567_89AB_CDEF);
        for (channels, taps, time) in [(1024, 9, 750), (2048, 15, 200), (512, 9, 1500)] {
            let x = rng.tensor(&[1, channels, time]);
            let weight = rng.tensor(&[channels, 1, taps]);
            let pad = (taps - 1) / 2;
            let iters = 5;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                candle_reference(&x, &weight, None, pad, pad, 1);
            }
            let candle = start.elapsed() / iters;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                depthwise_conv1d(&x, &weight, None, DepthwiseConv1dSpec::symmetric(pad)).unwrap();
            }
            let shared = start.elapsed() / iters;
            eprintln!(
                "C={channels} K={taps} T={time}: candle grouped {candle:?}, shift-accumulate {shared:?} ({:.1}x)",
                candle.as_secs_f64() / shared.as_secs_f64()
            );
        }
    }

    #[test]
    #[ignore = "timing only: cargo test --release ... -- --ignored --nocapture"]
    fn bench_depthwise_transposed_vs_candle_grouped_conv_transpose1d() {
        let mut rng = XorShift(0x0FED_CBA9_8765_4321);
        for (channels, time) in [(512, 400), (1090, 400)] {
            let x = rng.tensor(&[1, channels, time]);
            let weight = rng.tensor(&[channels, 1, 3]);
            let iters = 5;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                x.conv_transpose1d(&weight, 1, 1, 2, 1, channels).unwrap();
            }
            let candle = start.elapsed() / iters;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                depthwise_conv_transpose1d(&x, &weight, None, 2, 1, 1).unwrap();
            }
            let shared = start.elapsed() / iters;
            eprintln!(
                "C={channels} T={time} k3 s2: candle grouped {candle:?}, phase-decomposed {shared:?} ({:.1}x)",
                candle.as_secs_f64() / shared.as_secs_f64()
            );
        }
    }

    /// The depthwise transposed conv matches Candle's grouped
    /// `conv_transpose1d` (which runs one transposed conv per channel) across
    /// Kokoro's k3/s2/p1/op1 upsampler and other strides, kernels and
    /// paddings, with and without bias.
    #[test]
    fn depthwise_transposed_conv_matches_candle_grouped_conv_transpose1d() {
        let mut rng = XorShift(0x5DEE_CE66_D1CE_4E5B);
        // (batch, channels, time, taps, stride, padding, output_padding, bias)
        let cases = [
            (1, 512, 37, 3, 2, 1, 1, true), // Kokoro F0.1 / N.1 pool
            (2, 64, 23, 3, 2, 1, 1, false), // batched
            (1, 16, 11, 4, 2, 1, 0, true),
            (1, 8, 9, 5, 3, 2, 1, true),
            (1, 8, 6, 2, 2, 0, 0, false),
            (1, 4, 5, 1, 2, 0, 1, true), // fewer taps than stride
            (1, 4, 7, 3, 1, 1, 0, true), // stride 1
        ];
        for (batch, channels, time, taps, stride, padding, output_padding, with_bias) in cases {
            let x = rng.tensor(&[batch, channels, time]);
            let weight = rng.tensor(&[channels, 1, taps]);
            let bias = with_bias.then(|| rng.tensor(&[channels]));
            let mut expected = x
                .conv_transpose1d(&weight, padding, output_padding, stride, 1, channels)
                .unwrap();
            if let Some(bias) = &bias {
                expected = expected
                    .broadcast_add(&bias.reshape((1, channels, 1)).unwrap())
                    .unwrap();
            }
            let before = depthwise_conv_transpose1d_calls();
            let actual = depthwise_conv_transpose1d(
                &x,
                &weight,
                bias.as_ref(),
                stride,
                padding,
                output_padding,
            )
            .unwrap();
            assert!(depthwise_conv_transpose1d_calls() > before);
            let diff = max_abs_diff(&expected, &actual);
            assert!(
                diff <= 1e-5,
                "case {:?}: max |diff| {diff}",
                (batch, channels, time, taps, stride, padding, output_padding)
            );
        }
        assert!(is_depthwise_conv_transpose1d(
            &rng.tensor(&[512, 1, 3]),
            512,
            512,
            1
        ));
        assert!(!is_depthwise_conv_transpose1d(
            &rng.tensor(&[8, 2, 3]),
            8,
            8,
            1
        ));
    }

    /// The strided-tap depthwise conv2d matches Candle's grouped `conv2d`
    /// (one convolution per channel) across the dw_striding k3/s2/p1 geometry
    /// and other strides, kernels, paddings and odd sizes.
    #[test]
    fn depthwise_conv2d_matches_candle_grouped_conv2d() {
        let mut rng = XorShift(0x7A3C_91E5_0B2D_4F61);
        // (batch, channels, height, width, kh, kw, stride, padding, bias)
        let cases = [
            (1, 32, 41, 64, 3, 3, 2, 1, true), // Sortformer / Parakeet dw_striding
            (2, 8, 37, 17, 3, 3, 2, 1, true),  // batched, odd sizes
            (1, 8, 1, 1, 3, 3, 2, 1, false),   // single pixel
            (1, 4, 9, 6, 3, 3, 4, 0, true),    // stride 4 trims a tail row
            (1, 6, 10, 11, 5, 3, 3, 2, true),  // rectangular kernel
            (1, 5, 7, 8, 3, 3, 1, 1, false),   // stride 1
            (1, 3, 8, 9, 1, 1, 2, 0, true),    // 1x1 strided
            (1, 4, 6, 7, 2, 2, 2, 0, true),    // even kernel
        ];
        for (batch, channels, height, width, kh, kw, stride, padding, with_bias) in cases {
            let x = rng.tensor(&[batch, channels, height, width]);
            let weight = rng.tensor(&[channels, 1, kh, kw]);
            let bias = with_bias.then(|| rng.tensor(&[channels]));
            let mut expected = x.conv2d(&weight, padding, stride, 1, channels).unwrap();
            if let Some(bias) = &bias {
                expected = expected
                    .broadcast_add(&bias.reshape((1, channels, 1, 1)).unwrap())
                    .unwrap();
            }
            let before = depthwise_conv2d_calls();
            let actual = depthwise_conv2d(
                &x,
                &weight,
                bias.as_ref(),
                DepthwiseConv2dSpec { padding, stride },
            )
            .unwrap();
            assert!(depthwise_conv2d_calls() > before);
            let diff = max_abs_diff(&expected, &actual);
            assert!(
                diff <= 1e-5,
                "case {:?}: max |diff| {diff}",
                (batch, channels, height, width, kh, kw, stride, padding)
            );
        }
        // Mismatched weights and inputs smaller than the kernel are rejected.
        let x = rng.tensor(&[1, 4, 2, 2]);
        let spec = DepthwiseConv2dSpec {
            padding: 0,
            stride: 1,
        };
        assert!(depthwise_conv2d(&x, &rng.tensor(&[4, 2, 1, 1]), None, spec).is_err());
        assert!(depthwise_conv2d(&x, &rng.tensor(&[4, 1, 3, 3]), None, spec).is_err());
    }

    /// CPU micro-benchmark against Candle's per-group loop (Sortformer
    /// dw_striding conv.2 geometry for a 15 s chunk: 256 channels, 760x64).
    #[test]
    #[ignore = "timing only: cargo test --release ... -- --ignored --nocapture"]
    fn bench_depthwise_conv2d_vs_candle_grouped_conv2d() {
        let mut rng = XorShift(0x2468_ACE0_1357_9BDF);
        for (channels, height, width) in [(256, 760, 64), (256, 380, 32), (256, 52, 64)] {
            let x = rng.tensor(&[1, channels, height, width]);
            let weight = rng.tensor(&[channels, 1, 3, 3]);
            let spec = DepthwiseConv2dSpec {
                padding: 1,
                stride: 2,
            };
            let iters = 5;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                x.conv2d(&weight, 1, 2, 1, channels).unwrap();
            }
            let candle = start.elapsed() / iters;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                depthwise_conv2d(&x, &weight, None, spec).unwrap();
            }
            let shared = start.elapsed() / iters;
            eprintln!(
                "C={channels} {height}x{width} k3 s2: candle grouped {candle:?}, strided taps {shared:?} ({:.1}x)",
                candle.as_secs_f64() / shared.as_secs_f64()
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_depthwise_conv2d_matches_cpu() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_depthwise_conv2d_matches_cpu")
        else {
            return;
        };
        let mut rng = XorShift(0x0BAD_5EED_1234_5678);
        let x = rng.tensor(&[1, 16, 37, 33]);
        let weight = rng.tensor(&[16, 1, 3, 3]);
        let bias = rng.tensor(&[16]);
        let spec = DepthwiseConv2dSpec {
            padding: 1,
            stride: 2,
        };
        let cpu = depthwise_conv2d(&x, &weight, Some(&bias), spec).unwrap();
        let metal = depthwise_conv2d(
            &x.to_device(&device).unwrap(),
            &weight.to_device(&device).unwrap(),
            Some(&bias.to_device(&device).unwrap()),
            spec,
        )
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
        assert!(max_abs_diff(&cpu, &metal) <= 1e-5);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_depthwise_transposed_conv_matches_cpu() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_depthwise_transposed_conv_matches_cpu",
        ) else {
            return;
        };
        let mut rng = XorShift(0x1234_5678_9ABC_DEF1);
        let x = rng.tensor(&[1, 512, 37]);
        let weight = rng.tensor(&[512, 1, 3]);
        let bias = rng.tensor(&[512]);
        let cpu = depthwise_conv_transpose1d(&x, &weight, Some(&bias), 2, 1, 1).unwrap();
        let metal = depthwise_conv_transpose1d(
            &x.to_device(&device).unwrap(),
            &weight.to_device(&device).unwrap(),
            Some(&bias.to_device(&device).unwrap()),
            2,
            1,
            1,
        )
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
        assert!(max_abs_diff(&cpu, &metal) <= 1e-5);
    }
}
