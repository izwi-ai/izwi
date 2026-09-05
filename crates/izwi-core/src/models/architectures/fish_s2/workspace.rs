//! Conservative F32, batch-one codec workspace envelopes.
//!
//! These price the full waveform path, including Candle's portable im2col/col2im
//! buffers. cuDNN uses ImplicitGemm in `dac.rs`, so it cannot select an unpriced
//! convolution workspace. The maxima are over sequential stages; model weights
//! and language-model KV caches are priced separately by the runtime.

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::dac::{FishS2DacConfig, ATTENTION_QUERY_BLOCK};

const SMALL_BUFFER_ALLOWANCE: u64 = 16 * 1024 * 1024;
const FFT_BYTES_PER_FRAME: u64 = 256;

#[derive(Clone, Copy)]
struct AudioShape {
    channels: u64,
    frames: u64,
}

impl AudioShape {
    fn bytes(self) -> Result<u64> {
        product(&[self.channels, self.frames, 4])
    }
}

pub(crate) fn preparation_workspace_bytes(input_samples: usize, sample_rate: u32) -> Result<u64> {
    let config = FishS2DacConfig::current();
    let frames = config.reference_frame_count(input_samples, sample_rate)?;
    preparation_envelope(
        &config,
        frames as u64,
        input_samples as u64,
        fft_workspace(input_samples as u64, sample_rate)?,
    )
}

pub(crate) fn decode_workspace_bytes(frames: usize) -> Result<u64> {
    if frames == 0 || frames > FishS2DacConfig::MAX_QUANTIZER_FRAMES {
        return Err(Error::InvalidInput(format!(
            "Fish S2 codec decode requires 1..={} frames",
            FishS2DacConfig::MAX_QUANTIZER_FRAMES
        )));
    }
    decode_envelope(&FishS2DacConfig::current(), frames as u64)
}

pub(crate) fn maximum_decode_workspace_bytes() -> Result<u64> {
    decode_workspace_bytes(FishS2DacConfig::MAX_QUANTIZER_FRAMES)
}

pub(crate) fn maximum_preparation_workspace_bytes() -> Result<u64> {
    let config = FishS2DacConfig::current();
    let native_samples = config.maximum_reference_samples()? as u64;
    // Rounding to the nearest native sample admits up to half a sample more.
    let maximum_input = sum(&[
        product(&[
            sum(&[native_samples, 1])?,
            u64::from(FishS2DacConfig::MAX_REFERENCE_SAMPLE_RATE),
        ])? / u64::from(config.sample_rate),
        1,
    ])?;
    // For Rubato's exact-ratio FFT chunks: fft_in < chunk + rate/gcd and
    // fft_out < resampled_input + target_rate/gcd. This also covers rates whose
    // greatest common divisor is one, and very low input rates with a large FFT.
    let maximum_fft = product(&[
        FFT_BYTES_PER_FRAME,
        sum(&[
            4096,
            u64::from(FishS2DacConfig::MAX_REFERENCE_SAMPLE_RATE),
            native_samples,
            u64::from(config.sample_rate),
            2,
        ])?,
    ])?;
    preparation_envelope(
        &config,
        FishS2DacConfig::MAX_QUANTIZER_FRAMES as u64,
        maximum_input,
        maximum_fft,
    )
}

fn preparation_envelope(
    config: &FishS2DacConfig,
    frames: u64,
    input_samples: u64,
    fft: u64,
) -> Result<u64> {
    let prepared = product(&[frames, config.samples_per_frame()? as u64])?;
    let mut shape = AudioShape {
        channels: 1,
        frames: prepared,
    };
    let (next, mut peak) = conv1d(shape, config.encoder_dim as u64, 7, 1, 1)?;
    shape = next;
    for (rate, layers) in config
        .encoder_rates
        .iter()
        .zip(&config.encoder_transformer_layers)
    {
        let block_input = shape.bytes()?;
        peak = peak.max(sum(&[block_input, residual_unit(shape)?])?);
        let (next, conv) = conv1d(
            shape,
            product(&[shape.channels, 2])?,
            product(&[*rate as u64, 2])?,
            *rate as u64,
            1,
        )?;
        peak = peak.max(sum(&[conv, product(&[block_input, 2])?])?);
        shape = next;
        if *layers > 0 {
            peak = peak.max(sum(&[
                block_input,
                transformer(
                    shape,
                    config.encoder_window_size as u64,
                    config.transformer_head_dim as u64,
                )?,
            ])?);
        }
    }
    let encoder_tail = shape.bytes()?;
    let (next, conv) = conv1d(shape, config.latent_dim as u64, 3, 1, 1)?;
    peak = peak.max(sum(&[conv, encoder_tail])?);
    shape = next;
    let encoder_latent = shape.bytes()?;
    for factor in &config.downsample_factors {
        let block_input = shape.bytes()?;
        let (next, conv) = conv1d(shape, shape.channels, *factor as u64, *factor as u64, 1)?;
        shape = next;
        peak = peak.max(sum(&[encoder_latent, conv])?);
        peak = peak.max(sum(&[encoder_latent, block_input, convnext(shape)?])?);
    }
    peak = peak.max(sum(&[
        encoder_latent,
        transformer(
            shape,
            config.transformer_window_size as u64,
            config.transformer_head_dim as u64,
        )?,
    ])?);
    let distances = product(&[
        shape.frames,
        config
            .semantic_codebook_size
            .max(config.residual_codebook_size) as u64,
        4,
    ])?;
    // Semantic and residual latents, normalized projected vectors, the current
    // codebook, and up to four overlapping distance/matmul results.
    let quantization = sum(&[
        product(&[shape.bytes()?, 12])?,
        product(&[distances, 4])?,
        product(&[
            config.semantic_codebook_size as u64,
            config.codebook_dim as u64,
            4,
            4,
        ])?,
    ])?;
    peak = peak.max(sum(&[encoder_latent, quantization])?);
    sum(&[
        peak,
        product(&[input_samples, 4])?,
        product(&[prepared, 4, 3])?,
        product(&[frames, config.num_codebooks() as u64, 4, 3])?,
        fft,
        SMALL_BUFFER_ALLOWANCE,
    ])
}

fn decode_envelope(config: &FishS2DacConfig, frames: u64) -> Result<u64> {
    let mut shape = AudioShape {
        channels: config.latent_dim as u64,
        frames,
    };
    let quantizer_latents = product(&[shape.bytes()?, 4])?;
    let mut peak = product(&[shape.bytes()?, 12])?;
    peak = peak.max(sum(&[
        quantizer_latents,
        transformer(
            shape,
            config.transformer_window_size as u64,
            config.transformer_head_dim as u64,
        )?,
    ])?);
    for factor in config.downsample_factors.iter().rev() {
        let block_input = shape.bytes()?;
        let (next, conv) = conv_transpose(shape, shape.channels, *factor as u64, *factor as u64)?;
        shape = next;
        peak = peak.max(sum(&[quantizer_latents, conv])?);
        peak = peak.max(sum(&[quantizer_latents, block_input, convnext(shape)?])?);
    }
    let decoder_latent = shape.bytes()?;
    let (next, conv) = conv1d(shape, config.decoder_dim as u64, 7, 1, 1)?;
    peak = peak.max(conv);
    shape = next;
    for rate in &config.decoder_rates {
        let block_input = shape.bytes()?;
        peak = peak.max(snake(shape)?);
        let (next, conv) = conv_transpose(
            shape,
            shape.channels / 2,
            product(&[*rate as u64, 2])?,
            *rate as u64,
        )?;
        peak = peak.max(sum(&[conv, block_input])?);
        shape = next;
        peak = peak.max(sum(&[block_input, residual_unit(shape)?])?);
    }
    let (_, final_conv) = conv1d(shape, 1, 7, 1, 1)?;
    peak = peak.max(sum(&[final_conv, shape.bytes()?])?);
    sum(&[
        peak,
        decoder_latent,
        product(&[frames, config.samples_per_frame()? as u64, 4, 3])?,
        product(&[frames, config.num_codebooks() as u64, 4, 3])?,
        SMALL_BUFFER_ALLOWANCE,
    ])
}

fn conv1d(
    input: AudioShape,
    output_channels: u64,
    kernel: u64,
    stride: u64,
    dilation: u64,
) -> Result<(AudioShape, u64)> {
    let frames = input.frames.div_ceil(stride);
    let output = AudioShape {
        channels: output_channels,
        frames,
    };
    let padded_frames = sum(&[
        product(&[frames - 1, stride])?,
        product(&[kernel - 1, dilation])?,
        1,
    ])?;
    // Input, both padding concatenations, im2col, matmul output, layout copy and
    // bias output. Pricing all groups' im2col together also bounds depthwise conv.
    let peak = sum(&[
        input.bytes()?,
        product(&[input.channels, padded_frames, 4, 2])?,
        product(&[input.channels, frames, kernel, 4])?,
        product(&[output.bytes()?, 3])?,
    ])?;
    Ok((output, peak))
}

fn conv_transpose(
    input: AudioShape,
    channels: u64,
    kernel: u64,
    stride: u64,
) -> Result<(AudioShape, u64)> {
    let frames = product(&[input.frames, stride])?;
    let raw_frames = sum(&[frames, kernel - stride])?;
    let peak = sum(&[
        product(&[input.bytes()?, 2])?,
        product(&[input.frames, channels, kernel, 4])?,
        product(&[raw_frames, channels, 4, 3])?,
    ])?;
    Ok((AudioShape { channels, frames }, peak))
}

fn snake(shape: AudioShape) -> Result<u64> {
    sum(&[
        product(&[shape.bytes()?, 6])?,
        product(&[shape.channels, 4, 3])?,
    ])
}

fn residual_unit(shape: AudioShape) -> Result<u64> {
    let activation = shape.bytes()?;
    let (_, dilated) = conv1d(shape, shape.channels, 7, 1, 9)?;
    let (_, pointwise) = conv1d(shape, shape.channels, 1, 1, 1)?;
    Ok(sum(&[dilated, product(&[activation, 3])?])?
        .max(sum(&[pointwise, product(&[activation, 4])?])?)
        .max(sum(&[snake(shape)?, product(&[activation, 3])?])?))
}

fn convnext(shape: AudioShape) -> Result<u64> {
    let (_, depthwise) = conv1d(shape, shape.channels, 7, 1, 1)?;
    // Residual, layer norm intermediates and the 4x-wide GELU/projection overlap.
    Ok(sum(&[depthwise, product(&[shape.bytes()?, 2])?])?.max(product(&[shape.bytes()?, 16])?))
}

fn transformer(shape: AudioShape, window: u64, head_dim: u64) -> Result<u64> {
    let queries = shape.frames.min(ATTENTION_QUERY_BLOCK as u64);
    let keys = shape.frames.min(sum(&[window, queries - 1])?);
    let scores = product(&[shape.channels / head_dim, queries, keys, 4])?;
    // QKV, contiguous rotary/layout copies, residual/norm tensors, accumulated
    // output blocks and their concatenation; FFN intermediates are 3x channels.
    sum(&[
        product(&[shape.bytes()?, 24])?,
        product(&[shape.bytes()?, 3, 6])?,
        product(&[scores, 4])?,
        product(&[queries, keys, 4, 2])?,
    ])
}

fn fft_workspace(input_samples: u64, sample_rate: u32) -> Result<u64> {
    let target_rate = FishS2DacConfig::current().sample_rate;
    if sample_rate == target_rate {
        return Ok(0);
    }
    let gcd = gcd(u64::from(sample_rate), u64::from(target_rate));
    let minimum_input = u64::from(sample_rate) / gcd;
    let chunks = input_samples.min(4096).div_ceil(minimum_input);
    let input = product(&[chunks, minimum_input])?;
    let output = product(&[chunks, u64::from(target_rate) / gcd])?;
    // Includes real/complex FFT buffers, plan scratch and tables, filter
    // construction, partial-input padding, delay flushing and output growth.
    product(&[sum(&[input, output, 2])?, FFT_BYTES_PER_FRAME])
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |total, value| total.checked_add(*value))
        .ok_or_else(workspace_overflow)
}

fn product(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(1u64, |total, value| total.checked_mul(*value))
        .ok_or_else(workspace_overflow)
}

fn workspace_overflow() -> Error {
    Error::InvalidInput("Fish S2 codec workspace size overflowed".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_workspace_covers_waveform_convolution_and_grows_with_frames() {
        let short = decode_workspace_bytes(100).unwrap();
        let long = decode_workspace_bytes(200).unwrap();
        // Independently counted final decoder stage: 96 channels, 2048 samples
        // per frame, seven-tap im2col plus at least three live activation buffers.
        let required = 200u64 * 2048 * 96 * 4 * (7 + 3);
        assert!(long >= required);
        assert!(long > short);
        assert!(long < short * 2);
        assert!(maximum_decode_workspace_bytes().unwrap() >= long);
    }

    #[test]
    fn native_and_resampled_preparation_are_covered_by_the_global_maximum() {
        let maximum = maximum_preparation_workspace_bytes().unwrap();
        let config = FishS2DacConfig::current();
        let native_max = config.maximum_reference_samples().unwrap();
        for rate in [
            1u32, 100, 8000, 16000, 24000, 44100, 48000, 192000, 383999, 384000,
        ] {
            let samples =
                ((native_max as u128 * u128::from(rate)) / u128::from(config.sample_rate)) as usize;
            assert!(
                preparation_workspace_bytes(samples, rate).unwrap() <= maximum,
                "rate={rate}"
            );
        }
        let thirty_seconds = preparation_workspace_bytes(30 * 16000, 16000).unwrap();
        // The initial encoder residual stage alone has 64 full-rate channels.
        assert!(thirty_seconds > 30 * 44100 * 64 * 4 * 10);
    }

    #[test]
    fn workspace_rejects_invalid_capacities_and_arithmetic_overflow() {
        assert!(decode_workspace_bytes(0).is_err());
        assert!(decode_workspace_bytes(4097).is_err());
        assert!(decode_workspace_bytes(usize::MAX).is_err());
        assert!(preparation_workspace_bytes(1, 0).is_err());
        assert!(preparation_workspace_bytes(1, 384001).is_err());
        assert!(preparation_workspace_bytes(usize::MAX, 384000).is_err());
        assert!(product(&[u64::MAX, 4]).is_err());
        assert!(sum(&[u64::MAX, 1]).is_err());
    }
}
