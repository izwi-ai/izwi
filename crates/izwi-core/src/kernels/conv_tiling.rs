//! Output-time tiling for 1-D convolutions.
//!
//! Candle lowers conv1d to im2col + matmul on the CPU and on Metal, so a
//! kernel-k convolution over C input channels and T output steps
//! materializes a `B·C·k·T` buffer before the matmul. For long audio
//! decoders that buffer dominates the working set (Kokoro's final generator
//! stage: 676 KB per duration frame per k=11 conv, KOK-9). Tiling the
//! output time bounds it: each tile convolves its own input window (the
//! tile plus the receptive-field halo of the zero-padded input), so the
//! result is the same convolution, computed in pieces.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Module};

static TILED_CALLS: AtomicU64 = AtomicU64::new(0);
static TILES: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of conv1d calls served in tiles.
pub fn conv1d_tiled_calls() -> u64 {
    TILED_CALLS.load(Ordering::Relaxed)
}

/// Process-wide count of tiles those calls ran.
pub fn conv1d_tiles() -> u64 {
    TILES.load(Ordering::Relaxed)
}

/// The im2col bytes Candle materializes for one stride-1 conv1d producing
/// `t_out` steps.
pub fn conv1d_im2col_bytes(
    batch: usize,
    in_channels: usize,
    kernel: usize,
    t_out: usize,
    elem_bytes: usize,
) -> u64 {
    [batch, in_channels, kernel, t_out, elem_bytes]
        .into_iter()
        .fold(1u64, |acc, n| acc.saturating_mul(n as u64))
}

/// The output steps per tile that keep one tile's im2col within
/// `max_im2col_bytes` (at least one step).
pub fn conv1d_tile_steps(
    batch: usize,
    in_channels: usize,
    kernel: usize,
    elem_bytes: usize,
    max_im2col_bytes: u64,
) -> usize {
    let per_step = conv1d_im2col_bytes(batch, in_channels, kernel, 1, elem_bytes).max(1);
    usize::try_from(max_im2col_bytes / per_step)
        .unwrap_or(usize::MAX)
        .max(1)
}

/// `conv.forward(x)` with its im2col bounded by `max_im2col_bytes`: a
/// stride-1, ungrouped conv whose im2col would exceed the bound runs over
/// output-time tiles of the zero-padded input and concatenates them. Other
/// convs, and convs within the bound, run as one Candle call.
pub fn conv1d_bounded(x: &Tensor, conv: &Conv1d, max_im2col_bytes: u64) -> Result<Tensor> {
    let cfg = conv.config();
    let weight = conv.weight();
    let (batch, in_channels, len) = x.dims3()?;
    let (out_channels, _, kernel) = weight.dims3()?;
    if cfg.stride != 1 || cfg.groups != 1 {
        return conv.forward(x);
    }
    let halo = cfg.dilation * (kernel.saturating_sub(1));
    let padded = len + 2 * cfg.padding;
    if padded <= halo {
        return conv.forward(x);
    }
    let t_out = padded - halo;
    let elem = x.dtype().size_in_bytes();
    if conv1d_im2col_bytes(batch, in_channels, kernel, t_out, elem) <= max_im2col_bytes {
        return conv.forward(x);
    }
    let steps = conv1d_tile_steps(batch, in_channels, kernel, elem, max_im2col_bytes);
    let x = x.pad_with_zeros(2, cfg.padding, cfg.padding)?;
    let mut tiles = Vec::with_capacity(t_out.div_ceil(steps));
    let mut start = 0;
    while start < t_out {
        let take = steps.min(t_out - start);
        let window = x.narrow(2, start, take + halo)?;
        tiles.push(window.conv1d(weight, 0, 1, cfg.dilation, 1)?);
        start += take;
    }
    TILED_CALLS.fetch_add(1, Ordering::Relaxed);
    TILES.fetch_add(tiles.len() as u64, Ordering::Relaxed);
    let out = Tensor::cat(&tiles, 2)?;
    match conv.bias() {
        Some(bias) => out.broadcast_add(&bias.reshape((1, out_channels, 1))?),
        None => Ok(out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};
    use candle_core::Device;
    use candle_nn::Conv1dConfig;

    fn conv(
        device: &Device,
        c_in: usize,
        c_out: usize,
        k: usize,
        cfg: Conv1dConfig,
        bias: bool,
    ) -> Conv1d {
        let weight = xorshift_tensor(&[c_out, c_in, k], 0xC0F1 + k as u64, 0.3, device).unwrap();
        let bias = bias.then(|| xorshift_tensor(&[c_out], 0xB1A5, 0.1, device).unwrap());
        Conv1d::new(weight, bias, cfg)
    }

    fn cfg(padding: usize, dilation: usize) -> Conv1dConfig {
        Conv1dConfig {
            padding,
            stride: 1,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }

    /// KOK-9: tiling the output time gives the untiled convolution for
    /// Kokoro's resblock shapes (k = 3/7/11, dilations 1/3/5, "same"
    /// padding) and for a conv without padding, including ragged last tiles
    /// and a batch, while each tile's im2col stays within the bound.
    #[test]
    fn tiled_conv1d_matches_the_untiled_convolution() {
        let device = Device::Cpu;
        for (k, dilation, bias, batch) in [
            (3, 1, false, 1),
            (7, 3, true, 1),
            (11, 5, true, 2),
            (5, 1, true, 1),
        ] {
            let padding = if k == 5 { 0 } else { dilation * (k - 1) / 2 };
            let conv = conv(&device, 6, 5, k, cfg(padding, dilation), bias);
            let x = xorshift_tensor(&[batch, 6, 301], 0x7111 + k as u64, 1.0, &device).unwrap();
            let expected = conv.forward(&x).unwrap();
            let bound = conv1d_im2col_bytes(batch, 6, k, 37, 4);
            let calls = conv1d_tiled_calls();
            let tiles = conv1d_tiles();
            let tiled = conv1d_bounded(&x, &conv, bound).unwrap();
            assert!(conv1d_tiled_calls() > calls, "k={k} was not tiled");
            assert!(conv1d_tiles() - tiles >= 301usize.div_ceil(37) as u64 - 1);
            assert_eq!(tiled.dims(), expected.dims());
            let diff = max_abs_diff(&host_f32(&tiled).unwrap(), &host_f32(&expected).unwrap());
            assert!(diff <= 1e-5, "k={k} d={dilation}: max |diff| {diff}");
        }
        // Within the bound: one Candle call, bit-identical.
        let conv = conv(&device, 4, 4, 3, cfg(1, 1), true);
        let x = xorshift_tensor(&[1, 4, 50], 0x51, 1.0, &device).unwrap();
        let direct = conv.forward(&x).unwrap();
        let bounded = conv1d_bounded(&x, &conv, u64::MAX).unwrap();
        assert_eq!(host_f32(&direct).unwrap(), host_f32(&bounded).unwrap());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_tiled_conv1d_matches_cpu() {
        let Some((metal, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_tiled_conv1d_matches_cpu")
        else {
            return;
        };
        let run = |device: &Device| {
            let conv = conv(device, 8, 8, 11, cfg(25, 5), true);
            let x = xorshift_tensor(&[1, 8, 513], 0x3A11, 1.0, device).unwrap();
            host_f32(&conv1d_bounded(&x, &conv, conv1d_im2col_bytes(1, 8, 11, 64, 4)).unwrap())
                .unwrap()
        };
        let diff = max_abs_diff(&run(&metal), &run(&Device::Cpu));
        assert!(diff <= 1e-4, "max |diff| {diff}");
    }
}
