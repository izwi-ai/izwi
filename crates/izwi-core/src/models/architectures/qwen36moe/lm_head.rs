//! Block-FP8 LM head on CUDA.
//!
//! The checkpoint keeps the LM head in BF16 (`[248,320, 2,048]`, about 1 GB),
//! the largest single read of every decode step. On CUDA it is quantized on the
//! host at load to the checkpoints' own block-FP8 format (E4M3FN bytes plus one
//! F32 scale per 128×128 block) and runs through the compact FP8 GEMV like the
//! trunk projections. Two checks guard it. A host quality gate compares FP8
//! and BF16 logits for probe rows over a strided sample of the vocabulary. A
//! device check then runs the uploaded head and compares it with the host FP8
//! reference. Either failure keeps the BF16 head. Admission charges the FP8
//! bytes whenever [`fp8_lm_head_requested`] holds.
use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
use half::bf16;

use super::fast_path::{compare_values, legacy_requested};
use super::native::{Qwen36MoeNativeCheckpoint, Qwen36MoeProjectionResidency};
use super::text::Qwen36Projection;
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::cuda::fp8;

/// Kill switch for the block-FP8 LM head (`0`, `off`, `false` or `legacy`).
pub const FP8_LM_HEAD_ENV: &str = "IZWI_QWEN36_FP8_LM_HEAD";

/// Relative L2 error the FP8 logits may show against BF16 on the probes. Three
/// mantissa bits give about 3% on dense dot products; layout or scale bugs
/// give errors of order 1.
const QUALITY_TOLERANCE: f64 = 0.08;
/// Largest single-logit error, as a fraction of the largest logit.
const QUALITY_MAX_FRACTION: f32 = 0.5;
const PROBES: usize = 4;
/// About 8,000 of the 248,320 vocabulary rows enter the host gate.
const ROW_STRIDE: usize = 31;

/// Whether the LM head loads as block FP8: CUDA with the raw block-FP8
/// residency (so `cuda.mode` and `projection_backend = "q8"` turn it off),
/// unless [`FP8_LM_HEAD_ENV`] does. Admission and the loader both resolve it
/// here.
pub fn fp8_lm_head_requested(
    backend: BackendKind,
    performance: &crate::performance::CudaPerformanceConfig,
) -> bool {
    backend == BackendKind::Cuda
        && Qwen36MoeNativeCheckpoint::projection_residency_policy_with_performance(
            backend,
            performance,
        ) == Qwen36MoeProjectionResidency::NativeFp8WithQ8Fallback
        && !legacy_requested(FP8_LM_HEAD_ENV)
}

/// Resident bytes the FP8 head saves against BF16: 2 bytes per element
/// become 1, plus one F32 scale per 128×128 block.
pub fn fp8_lm_head_saving_bytes(lm_head_elements: u64) -> u64 {
    lm_head_elements.saturating_sub(lm_head_elements.div_ceil(128 * 128) * 4)
}

/// Quantize a host BF16 head `[rows, cols]` and upload it to `device`.
/// `Ok(Err(reason))` means the head should stay BF16.
pub(crate) fn quantize_lm_head(
    host: &Tensor,
    device: &Device,
) -> Result<std::result::Result<Qwen36Projection, String>> {
    let (rows, cols) = host.dims2()?;
    if !fp8::provider_supported(device, DType::BF16, rows, cols) {
        return Ok(Err(format!(
            "the block-FP8 GEMV does not cover a [{rows}, {cols}] head on this device"
        )));
    }
    let host = host.to_dtype(DType::BF16)?.contiguous()?;
    let (storage, layout) = host.storage_and_layout();
    let (Storage::Cpu(CpuStorage::BF16(data)), Some((start, end))) =
        (&*storage, layout.contiguous_offsets())
    else {
        return Err(Error::ModelLoadError(
            "the FP8 LM head quantizes a contiguous host BF16 tensor".into(),
        ));
    };
    let values = &data[start..end];
    let (bytes, scales) = fp8::quantize_block_fp8(values, rows, cols, bf16::to_f32)?;
    let probes = probe_rows(cols);
    let sampled: Vec<usize> = (0..rows).step_by(ROW_STRIDE).collect();
    let fp8_logits = |probe: &[f32]| -> Vec<f32> {
        sampled
            .iter()
            .map(|&row| fp8_dot(&bytes, &scales, row, cols, probe))
            .collect()
    };
    let mut exact = Vec::with_capacity(PROBES * sampled.len());
    let mut approx = Vec::with_capacity(PROBES * sampled.len());
    for probe in &probes {
        exact.extend(sampled.iter().map(|&row| {
            values[row * cols..][..cols]
                .iter()
                .zip(probe)
                .map(|(w, x)| w.to_f32() * x)
                .sum::<f32>()
        }));
        approx.extend(fp8_logits(probe));
    }
    if let Err(error) = compare_values(
        "FP8 LM head quality gate",
        &approx,
        &exact,
        QUALITY_TOLERANCE,
        QUALITY_MAX_FRACTION,
    ) {
        return Ok(Err(error.to_string()));
    }
    // The device multiplies BF16 activations: its reference is the host FP8
    // head on the same rounded probe.
    let probe: Vec<f32> = probes[0]
        .iter()
        .map(|&x| bf16::from_f32(x).to_f32())
        .collect();
    let reference = fp8_logits(&probe);
    drop(storage);

    let weights = Tensor::from_vec(bytes, (rows, cols), device)?;
    let scale_tensor = Tensor::from_vec(scales, (rows.div_ceil(128), cols.div_ceil(128)), device)?;
    let projection = Qwen36Projection::CompactFp8 {
        weights,
        scales: scale_tensor,
    };
    let input = Tensor::from_vec(probe, (1, cols), &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(device)?;
    let device_logits = projection
        .forward(&input)?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let device_sampled: Vec<f32> = sampled.iter().map(|&row| device_logits[row]).collect();
    if let Err(error) = compare_values(
        "FP8 LM head device check",
        &device_sampled,
        &reference,
        0.01,
        0.02,
    ) {
        return Ok(Err(format!("device check failed: {error}")));
    }
    Ok(Ok(projection))
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
    let column_blocks = cols.div_ceil(128);
    (0..column_blocks)
        .map(|block| {
            let columns = block * 128..((block + 1) * 128).min(cols);
            let partial: f32 = columns
                .map(|column| fp8::decode_e4m3fn(bytes[row * cols + column]) * x[column])
                .sum();
            partial * scales[(row / 128) * column_blocks + block]
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random head whose rows differ in scale, as trained heads do.
    fn head(rows: usize, cols: usize) -> Tensor {
        let values: Vec<f32> = pseudo_random(99, rows * cols)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v * 0.05 * (1.0 + ((i / cols) % 7) as f32))
            .collect();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    #[test]
    fn the_switch_follows_the_cuda_residency_and_the_kill_switch() {
        let default = crate::performance::CudaPerformanceConfig::default();
        assert!(fp8_lm_head_requested(BackendKind::Cuda, &default));
        assert!(!fp8_lm_head_requested(BackendKind::Cpu, &default));
        assert!(!fp8_lm_head_requested(BackendKind::Metal, &default));
        let q8 = crate::performance::CudaPerformanceConfig {
            projection_backend: crate::performance::CudaProjectionBackend::Q8,
            ..default
        };
        assert!(!fp8_lm_head_requested(BackendKind::Cuda, &q8));
    }

    #[test]
    fn saving_is_one_byte_per_element_less_the_block_scales() {
        let elements = 248_320u64 * 2_048;
        assert_eq!(
            fp8_lm_head_saving_bytes(elements),
            elements - 1_940 * 16 * 4
        );
    }

    #[test]
    fn quantization_quality_on_a_dense_head_stays_inside_the_gate() {
        let host = head(256, 256);
        let values = host
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (bytes, scales) = fp8::quantize_block_fp8(&values, 256, 256, |v| v).unwrap();
        let probe = &probe_rows(256)[1];
        let exact: Vec<f32> = (0..256)
            .map(|row| {
                values[row * 256..][..256]
                    .iter()
                    .zip(probe)
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect();
        let approx: Vec<f32> = (0..256)
            .map(|row| fp8_dot(&bytes, &scales, row, 256, probe))
            .collect();
        // Half the gate's tolerance: real heads keep a margin.
        compare_values("dense head", &approx, &exact, QUALITY_TOLERANCE / 2.0, 0.25).unwrap();
        // Rows read from the wrong place, the kind of layout bug the gate
        // exists for, fail it.
        let shuffled: Vec<f32> = (0..256)
            .map(|row| fp8_dot(&bytes, &scales, (row * 37) % 256, 256, probe))
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
    fn heads_the_gemv_cannot_run_stay_bf16() {
        // The CPU has no block-FP8 GEMV provider for BF16 activations.
        let outcome = quantize_lm_head(&head(128, 128), &Device::Cpu).unwrap();
        assert!(outcome.is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fp8_lm_head_passes_its_checks_on_a_synthetic_head() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let projection = quantize_lm_head(&head(512, 256), &device)
            .unwrap()
            .unwrap_or_else(|reason| panic!("FP8 head refused: {reason}"));
        assert!(matches!(projection, Qwen36Projection::CompactFp8 { .. }));
    }
}
