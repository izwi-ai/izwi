//! The opt-in block-FP8 language-model head (Metal and CUDA).
//!
//! The checkpoint keeps the head in BF16 (`[131072, 2688]`, 704 MB), the
//! largest single read of a decode step. With [`FP8_LM_HEAD_ENV`] on, a GPU
//! load quantizes it on the host to the checkpoint's own block-FP8 form
//! (E4M3FN bytes plus one F32 scale per 128×128 block, 352 MB) and runs it
//! through the FP8 GEMVs the decoder's projections use. FP8 logits can swap
//! a greedy pick between near-tied tokens, so the head stays off by default
//! (D6) until a quality run on real audio clears it.
//!
//! Two checks guard it, as Qwen3.6's head does: a host quality gate compares
//! FP8 and exact logits for probe rows over a strided sample of the
//! vocabulary, and a device check compares the uploaded head with the host
//! FP8 reference. Either failure keeps the 16-bit head, and the diagnostics
//! say why.
use candle_core::{Device, Tensor};

use super::super::linear::NtLinear;
use crate::error::Result;
use crate::kernels::cuda::fp8;

/// Opt-in switch for the block-FP8 head (`1`, `on`, `true`). Read at load.
pub const FP8_LM_HEAD_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_FP8_LM_HEAD";

/// Relative L2 error the FP8 logits may show against the exact ones on the
/// probes: three mantissa bits give about 3% on dense dot products; layout
/// or scale bugs give errors of order 1.
const QUALITY_TOLERANCE: f64 = 0.08;
/// Largest single-logit error, as a fraction of the largest logit.
const QUALITY_MAX_FRACTION: f32 = 0.5;
/// The device head against the host FP8 head on the same probe.
const DEVICE_TOLERANCE: f64 = 1e-3;
const PROBES: usize = 4;
/// About 4,200 of the 131,072 vocabulary rows enter the host gate.
const ROW_STRIDE: usize = 31;

/// Whether a load on `device` asks for the FP8 head: a GPU, with
/// [`FP8_LM_HEAD_ENV`] on. Admission keeps pricing the 16-bit head, which a
/// failed check falls back to.
pub fn requested(device: &Device) -> bool {
    !device.is_cpu() && crate::env::flag_or(FP8_LM_HEAD_ENV, false)
}

/// Resident bytes of a `[rows, cols]` FP8 head: one byte per element plus
/// one F32 scale per 128×128 block.
pub fn fp8_bytes(rows: u64, cols: u64) -> u64 {
    rows * cols + 4 * rows.div_ceil(128) * cols.div_ceil(128)
}

/// The FP8 head from the exact F32 values `head` (`[rows, cols]`), or why it
/// stays 16-bit.
pub(crate) fn quantize(
    head: &[f32],
    rows: usize,
    cols: usize,
    device: &Device,
) -> Result<std::result::Result<NtLinear, String>> {
    if !rows.is_multiple_of(64) || !cols.is_multiple_of(128) {
        return Ok(Err(format!(
            "a [{rows}, {cols}] head is not whole 64x128 FP8 tiles"
        )));
    }
    let (bytes, scales) = fp8::quantize_block_fp8(head, rows, cols, |v| v)?;
    let sampled: Vec<usize> = (0..rows).step_by(ROW_STRIDE).collect();
    let fp8_logits = |probe: &[f32]| -> Vec<f32> {
        sampled
            .iter()
            .map(|&row| fp8_dot(&bytes, &scales, row, cols, probe))
            .collect()
    };
    let probes: Vec<Vec<f32>> = (0..PROBES as u64)
        .map(|seed| pseudo_random(seed + 1, cols))
        .collect();
    for probe in &probes {
        let exact: Vec<f32> = sampled
            .iter()
            .map(|&row| {
                head[row * cols..(row + 1) * cols]
                    .iter()
                    .zip(probe)
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
            })
            .collect();
        let approx = fp8_logits(probe);
        let error = crate::kernels::numeric_check::relative_l2(&approx, &exact);
        let largest = exact.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = approx
            .iter()
            .zip(&exact)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        // NaN fails both comparisons.
        let within = error <= QUALITY_TOLERANCE && worst <= QUALITY_MAX_FRACTION * largest;
        if !within {
            return Ok(Err(format!(
                "quality gate: relative L2 {error:.3e} (limit {QUALITY_TOLERANCE}), \
                 largest error {worst:.3e} of {largest:.3e}"
            )));
        }
    }
    let reference = fp8_logits(&probes[0]);
    let linear = NtLinear::fp8_from_blocks(bytes, scales, rows, cols, device)?;
    let input = Tensor::from_vec(probes[0].clone(), (1, cols), device)?;
    let logits = linear
        .forward(&input, super::super::linear::Pass::Step)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let device_sampled: Vec<f32> = sampled.iter().map(|&row| logits[row]).collect();
    let error = crate::kernels::numeric_check::relative_l2(&device_sampled, &reference);
    let within = error <= DEVICE_TOLERANCE;
    if !within {
        return Ok(Err(format!(
            "device check: relative L2 {error:.3e} (limit {DEVICE_TOLERANCE})"
        )));
    }
    Ok(Ok(linear))
}

/// Row `row`'s FP8 logit for `x`, in F32.
fn fp8_dot(bytes: &[u8], scales: &[f32], row: usize, cols: usize, x: &[f32]) -> f32 {
    let column_blocks = cols.div_ceil(128);
    bytes[row * cols..(row + 1) * cols]
        .iter()
        .zip(x)
        .enumerate()
        .map(|(c, (&b, &x))| {
            fp8::decode_e4m3fn(b) * scales[(row / 128) * column_blocks + c / 128] * x
        })
        .sum()
}

/// Deterministic pseudo-random values in [-1, 1): trained heads and hidden
/// states look random to the quantization error, so it averages out as it
/// does in decode.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn head(rows: usize, cols: usize) -> Vec<f32> {
        pseudo_random(7, rows * cols)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v * (0.02 + (i / cols % 5) as f32 * 0.01))
            .collect()
    }

    #[test]
    fn the_cpu_never_asks_and_the_bytes_halve() {
        assert!(!requested(&Device::Cpu));
        let bf16 = 131_072u64 * 2_688 * 2;
        let fp8 = fp8_bytes(131_072, 2_688);
        assert!(fp8 < bf16 / 2 + bf16 / 100, "{fp8} vs {bf16}");
    }

    #[test]
    fn a_ragged_head_stays_16_bit() {
        let values = head(100, 128);
        assert!(quantize(&values, 100, 128, &Device::Cpu).unwrap().is_err());
    }

    #[test]
    fn the_gate_passes_a_well_scaled_head_and_refuses_a_broken_one() {
        let (rows, cols) = (256usize, 256usize);
        let values = head(rows, cols);
        let (bytes, scales) = fp8::quantize_block_fp8(&values, rows, cols, |v| v).unwrap();
        // The host FP8 logits track the exact ones.
        let probe = pseudo_random(1, cols);
        let exact: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|c| values[r * cols + c] * probe[c]).sum())
            .collect();
        let approx: Vec<f32> = (0..rows)
            .map(|r| fp8_dot(&bytes, &scales, r, cols, &probe))
            .collect();
        let error = crate::kernels::numeric_check::relative_l2(&approx, &exact);
        assert!(error < QUALITY_TOLERANCE, "{error}");
        // A head with a NaN never passes.
        let mut broken = values.clone();
        broken[3] = f32::NAN;
        let refused = match fp8::quantize_block_fp8(&broken, rows, cols, |v| v) {
            Err(_) => true,
            Ok((bytes, scales)) => {
                let approx: Vec<f32> = (0..rows)
                    .map(|r| fp8_dot(&bytes, &scales, r, cols, &probe))
                    .collect();
                let error = crate::kernels::numeric_check::relative_l2(&approx, &exact);
                error.is_nan() || error > QUALITY_TOLERANCE
            }
        };
        assert!(refused);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_quantizes_and_checks_the_head() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_quantizes_and_checks_the_head")
        else {
            return;
        };
        let (rows, cols) = (512usize, 256usize);
        let values = head(rows, cols);
        let linear = quantize(&values, rows, cols, &device)
            .unwrap()
            .expect("the gate and the device check pass");
        assert!(linear.is_fp8());
        assert_eq!(
            linear.resident_bytes() as u64,
            fp8_bytes(rows as u64, cols as u64)
        );
    }
}
