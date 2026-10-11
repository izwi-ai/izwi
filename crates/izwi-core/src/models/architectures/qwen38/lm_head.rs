//! Block-FP8 LM head for Qwen3.8 on CUDA.
//!
//! The checkpoint keeps the LM head in BF16 (`[248,320, 5,120]`, 2.54 GB), the
//! largest single read of every decode step and of every MTP draft. When the
//! residency policy runs the projections as native block FP8, the head is
//! quantized at load to the checkpoints' own block-FP8 format (E4M3FN bytes
//! plus one F32 scale per 128x128 block), streamed from the BF16 mapping one
//! 128-row block at a time, and runs on the same compact FP8 kernels.
//!
//! This is a deliberate precision reduction: the head's weights lose about
//! three mantissa bits (BF16 to E4M3 with a per-block scale). Two checks guard
//! it. A host quality gate compares FP8 and BF16 logits for four probe rows
//! over every 31st vocabulary row (8% relative L2, 50% of the largest logit:
//! three mantissa bits give about 3% on dense dot products; layout or scale
//! bugs give errors of order 1). A device check then runs the uploaded head
//! and compares it with the host FP8 reference (1% / 2%). A failure fails the
//! load closed and poisons the head's FP8 form for this checkpoint, so the
//! retry admits and loads the BF16 head; admission charges the FP8 bytes only
//! while that form is selected.
//!
//! The gate helpers are Qwen3.8's own copies (Qwen3.6 keeps its own head).

use candle_core::{DType, Device, Module, Tensor};
use half::bf16;
use safetensors::Dtype as SafeDType;

use super::fast_path::compare_values;
use super::native::{IndexedSafetensors, RawBlockFp8Projection};
use super::text::{Qwen38Fp8Kernel, Qwen38Projection};
use crate::error::{Error, Result};
use crate::kernels::cuda::fp8;

/// Kill switch for the block-FP8 LM head (`legacy`, `off`, `0`, `false`):
/// the head stays in the activation dtype.
pub(crate) const FP8_LM_HEAD_ENV: &str = "IZWI_QWEN38_FP8_LM_HEAD";

const QUALITY_TOLERANCE: f64 = 0.08;
const QUALITY_MAX_FRACTION: f32 = 0.5;
const PROBES: usize = 4;
const ROW_STRIDE: usize = 31;
const BLOCK: usize = 128;

/// Resident bytes the FP8 head saves against a two-byte head: one byte per
/// element, less one F32 scale per 128x128 block.
pub(crate) fn fp8_lm_head_saving_bytes(rows: u64, cols: u64) -> u64 {
    (rows * cols).saturating_sub(rows.div_ceil(128) * cols.div_ceil(128) * 4)
}

/// A quantized head that passed both checks.
pub(crate) struct Qwen38Fp8LmHead {
    pub projection: Qwen38Projection,
    /// Relative L2 error of the FP8 logits against BF16 on the gate probes.
    pub gate_relative_l2: f64,
}

/// Quantize the checkpoint's BF16 head `name` (`[rows, cols]`) to block FP8
/// on `device`, run on `activation` rows. `Ok(Err(reason))` when a check
/// failed (nothing stays resident).
pub(crate) fn quantize_lm_head_from_checkpoint(
    tensors: &IndexedSafetensors,
    name: &str,
    [rows, cols]: [usize; 2],
    device: &Device,
    activation: DType,
    kernel: Qwen38Fp8Kernel,
) -> Result<std::result::Result<Qwen38Fp8LmHead, String>> {
    let probes = probe_rows(cols);
    let sampled: Vec<usize> = (0..rows).step_by(ROW_STRIDE).collect();
    let quantized =
        tensors.with_tensor_view(name, Some(SafeDType::BF16), Some(&[rows, cols]), |view| {
            let data = view.data();
            let (bytes, scales) = quantize_bf16_head(data, rows, cols)?;
            let mut exact = Vec::with_capacity(PROBES * sampled.len());
            let mut approx = Vec::with_capacity(PROBES * sampled.len());
            for probe in &probes {
                for &row in &sampled {
                    exact.push(bf16_dot(&data[row * cols * 2..(row + 1) * cols * 2], probe));
                    approx.push(fp8_dot(&bytes, &scales, row, cols, probe));
                }
            }
            Ok((bytes, scales, exact, approx))
        })?;
    let (bytes, scales, exact, approx) = quantized;
    let gate_relative_l2 = relative_l2(&approx, &exact);
    if let Err(error) = compare_values(
        "FP8 LM head quality gate",
        &approx,
        &exact,
        QUALITY_TOLERANCE,
        QUALITY_MAX_FRACTION,
    ) {
        return Ok(Err(error.to_string()));
    }
    // The device multiplies activation-dtype rows: its reference is the
    // host FP8 head on the same rounded probe.
    let probe: Vec<f32> = match activation {
        DType::BF16 => probes[0]
            .iter()
            .map(|&x| bf16::from_f32(x).to_f32())
            .collect(),
        DType::F16 => probes[0]
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect(),
        _ => probes[0].clone(),
    };
    let reference: Vec<f32> = sampled
        .iter()
        .map(|&row| fp8_dot(&bytes, &scales, row, cols, &probe))
        .collect();
    let scale_shape = [rows.div_ceil(BLOCK), cols.div_ceil(BLOCK)];
    let projection = Qwen38Projection::NativeFp8 {
        raw: RawBlockFp8Projection {
            weights: Tensor::from_vec(bytes, (rows, cols), device)?,
            scales: Tensor::from_vec(scales, (scale_shape[0], scale_shape[1]), device)?,
            shape: [rows, cols],
            scale_shape,
            block_shape: [BLOCK, BLOCK],
        },
        kernel,
    };
    let input = Tensor::from_vec(probe, (1, cols), &Device::Cpu)?
        .to_dtype(activation)?
        .to_device(device)?;
    let logits = projection
        .forward(&input)?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let device_sampled: Vec<f32> = sampled.iter().map(|&row| logits[row]).collect();
    if let Err(error) = compare_values(
        "FP8 LM head device check",
        &device_sampled,
        &reference,
        0.01,
        0.02,
    ) {
        return Ok(Err(format!("device check failed: {error}")));
    }
    Ok(Ok(Qwen38Fp8LmHead {
        projection,
        gate_relative_l2,
    }))
}

/// Block-FP8 quantization of a little-endian BF16 `[rows, cols]` payload,
/// one 128-row block at a time in parallel: each block is decoded into an
/// F32 temporary and quantized by `fp8::quantize_block_fp8`, so the result is
/// bit-identical to quantizing the whole matrix at once (blocks never share
/// a scale) without a whole-matrix F32 or BF16 copy.
pub(crate) fn quantize_bf16_head(
    data: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<f32>)> {
    use rayon::prelude::*;
    if rows == 0 || cols == 0 || data.len() != rows * cols * 2 {
        return Err(Error::ModelLoadError(format!(
            "BF16 LM head payload of {} bytes is not [{rows}, {cols}]",
            data.len()
        )));
    }
    let column_blocks = cols.div_ceil(BLOCK);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(rows * cols)
        .map_err(|e| Error::ModelLoadError(format!("FP8 LM head staging: {e}")))?;
    bytes.resize(rows * cols, 0u8);
    let mut scales = vec![0f32; rows.div_ceil(BLOCK) * column_blocks];
    bytes
        .par_chunks_mut(BLOCK * cols)
        .zip(scales.par_chunks_mut(column_blocks))
        .enumerate()
        .try_for_each(|(block, (bytes, scales))| -> Result<()> {
            let block_rows = bytes.len() / cols;
            let source = &data[block * BLOCK * cols * 2..][..block_rows * cols * 2];
            let values: Vec<f32> = source
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect();
            if let Some(index) = values.iter().position(|v| !v.is_finite()) {
                return Err(Error::ModelLoadError(format!(
                    "BF16 LM head has a non-finite value at element {}",
                    block * BLOCK * cols + index
                )));
            }
            let (quantized, block_scales) =
                fp8::quantize_block_fp8(&values, block_rows, cols, |v| v)?;
            bytes.copy_from_slice(&quantized);
            scales.copy_from_slice(&block_scales);
            Ok(())
        })?;
    Ok((bytes, scales))
}

fn relative_l2(values: &[f32], reference: &[f32]) -> f64 {
    let (error, norm) = values
        .iter()
        .zip(reference)
        .fold((0f64, 0f64), |(e, n), (a, b)| {
            (e + f64::from(a - b).powi(2), n + f64::from(*b).powi(2))
        });
    let relative = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    if relative.is_finite() {
        relative
    } else {
        f64::INFINITY
    }
}

fn bf16_dot(row: &[u8], x: &[f32]) -> f32 {
    row.chunks_exact(2)
        .zip(x)
        .map(|(b, x)| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * x)
        .sum()
}

/// Deterministic pseudo-random values in [-1, 1). Structured probes (sine
/// waves) correlate with structured weights and stop the quantization error
/// from averaging out; trained heads and hidden states look random to it.
fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Deterministic probe activations with unit RMS.
fn probe_rows(cols: usize) -> Vec<Vec<f32>> {
    (0..PROBES)
        .map(|probe| {
            let row = pseudo_random(probe as u64 + 1, cols);
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / cols as f32).sqrt();
            row.into_iter().map(|v| v / rms).collect()
        })
        .collect()
}

fn fp8_dot(bytes: &[u8], scales: &[f32], row: usize, cols: usize, x: &[f32]) -> f32 {
    let column_blocks = cols.div_ceil(BLOCK);
    (0..column_blocks)
        .map(|block| {
            let columns = block * BLOCK..((block + 1) * BLOCK).min(cols);
            let partial: f32 = columns
                .map(|column| fp8::decode_e4m3fn(bytes[row * cols + column]) * x[column])
                .sum();
            partial * scales[(row / BLOCK) * column_blocks + block]
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random head whose rows differ in scale, as trained heads do.
    fn head(rows: usize, cols: usize) -> Vec<f32> {
        pseudo_random(99, rows * cols)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v * 0.05 * (1.0 + ((i / cols) % 7) as f32))
            .collect()
    }

    fn bf16_payload(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn streamed_bf16_quantizer_is_bit_equal_to_whole_matrix_quantization() {
        // 300 rows: two full 128-row blocks and a partial one; 320 columns:
        // a partial column block too.
        let (rows, cols) = (300usize, 320usize);
        let values = head(rows, cols);
        let payload = bf16_payload(&values);
        let (bytes, scales) = quantize_bf16_head(&payload, rows, cols).unwrap();
        let rounded: Vec<f32> = values.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
        let (expected_bytes, expected_scales) =
            fp8::quantize_block_fp8(&rounded, rows, cols, |v| v).unwrap();
        assert_eq!(bytes, expected_bytes);
        assert_eq!(
            scales.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            expected_scales
                .iter()
                .map(|s| s.to_bits())
                .collect::<Vec<_>>()
        );
        let mut poisoned = payload.clone();
        poisoned[2 * 1000..2 * 1001].copy_from_slice(&bf16::NAN.to_bits().to_le_bytes());
        assert!(quantize_bf16_head(&poisoned, rows, cols).is_err());
        assert!(quantize_bf16_head(&payload, rows, cols + 1).is_err());
    }

    #[test]
    fn quality_gate_passes_a_dense_head_and_fails_shuffled_rows() {
        let (rows, cols) = (256usize, 256usize);
        let values = head(rows, cols);
        let payload = bf16_payload(&values);
        let (bytes, scales) = quantize_bf16_head(&payload, rows, cols).unwrap();
        let probe = &probe_rows(cols)[1];
        let exact: Vec<f32> = (0..rows)
            .map(|row| bf16_dot(&payload[row * cols * 2..(row + 1) * cols * 2], probe))
            .collect();
        let approx: Vec<f32> = (0..rows)
            .map(|row| fp8_dot(&bytes, &scales, row, cols, probe))
            .collect();
        // Half the gate's tolerance: real heads keep a margin.
        compare_values("dense head", &approx, &exact, QUALITY_TOLERANCE / 2.0, 0.25).unwrap();
        assert!(
            relative_l2(&approx, &exact) > 1e-3,
            "FP8 is a real precision reduction"
        );
        // Rows read from the wrong place, the layout bug the gate exists
        // for, fail it (RED).
        let shuffled: Vec<f32> = (0..rows)
            .map(|row| fp8_dot(&bytes, &scales, (row * 37) % rows, cols, probe))
            .collect();
        assert!(compare_values(
            "shuffled head",
            &shuffled,
            &exact,
            QUALITY_TOLERANCE,
            QUALITY_MAX_FRACTION
        )
        .is_err());
    }

    #[test]
    fn saving_is_one_byte_per_element_less_the_block_scales() {
        assert_eq!(
            fp8_lm_head_saving_bytes(248_320, 5_120),
            1_271_398_400 - 1_940 * 40 * 4
        );
        assert_eq!(fp8_lm_head_saving_bytes(248_320, 5_120), 1_271_088_000);
    }
}
