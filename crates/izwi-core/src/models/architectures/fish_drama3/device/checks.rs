//! The load-time table checks every device load runs (§6 "Self-checks"),
//! each with its counter in the backend's report (`FishDrama3MetalChecks`,
//! `FishDrama3CudaChecks`):
//!
//! * **Host RoPE tables.** Both conventions (the trunk's NeoX half-split,
//!   the tower's interleaved pairs) rotate xorshift heads at positions 0,
//!   12,287 and 262,143 on the device against F64. A breach refuses the
//!   load: a wrong rotation has no slower path that fixes it.
//! * **FP8 GEMV vs F64 dequant.** The first and last routed experts of the
//!   first and last MoE layers, every projection kind, through the FP8
//!   GEMV over the stacked resident bytes, against the F64 dequantization
//!   of the original checkpoint tensors on 64 sampled rows: a stacking or
//!   scale fault shows here, which a check against the stacked tensors
//!   themselves could not see.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{expert, FishDrama3Checkpoint, Fp8BlockWeight};
use super::super::config::{FishDrama3AttentionConfig, FishDrama3Config, FishDrama3RopeLayout};
use super::super::precision::{FP8_GEMV_TOLERANCE, ROPE_TOLERANCE};
use super::super::trunk::attention::FishDrama3Rope;
use super::rope;
use super::trunk::DeviceTrunk;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{relative_l2, xorshift_values};

/// The positions the RoPE check rotates at: the first, the last of a
/// worst admitted request, the last a table may be asked for.
pub(crate) const ROPE_POSITIONS: [usize; 3] = [0, 12_287, 262_143];
/// Rows of each FP8 projection the check compares.
const FP8_ROWS: usize = 64;

pub(crate) fn load_error(backend: &str, message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!(
        "Fish Audio Drama 3 {backend} self-check: {}",
        message.into()
    ))
}

/// The F64 rotation of one head-major row `x` at `position`.
pub(crate) fn rope_reference(
    x: &[f32],
    position: usize,
    config: &FishDrama3AttentionConfig,
) -> Vec<f64> {
    let rotary = config.rotary_dims;
    let half = rotary / 2;
    let mut out = x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
    for head in out.chunks_exact_mut(config.head_dim) {
        let original = head.to_vec();
        for i in 0..half {
            let angle = position as f64 / config.rope_theta.powf((2 * i) as f64 / rotary as f64);
            let (a, b) = match config.rope_layout {
                FishDrama3RopeLayout::HalfSplit => (i, i + half),
                FishDrama3RopeLayout::InterleavedPairs => (2 * i, 2 * i + 1),
            };
            head[a] = original[a] * angle.cos() - original[b] * angle.sin();
            head[b] = original[b] * angle.cos() + original[a] * angle.sin();
        }
    }
    out
}

/// Rotates xorshift heads on `device` through the family's host tables at
/// [`ROPE_POSITIONS`] for each of `configs`, recording each worst error in
/// `checks` and every one over [`ROPE_TOLERANCE`] in `breaches`; refuses
/// the load on any breach.
pub(crate) fn check_rope(
    configs: &[(&str, FishDrama3AttentionConfig)],
    device: &Device,
    backend: &str,
    checks: &mut Vec<(String, usize, f64)>,
    breaches: &mut u64,
) -> Result<()> {
    for (name, config) in configs {
        let table = FishDrama3Rope::new(config);
        let heads = config.heads;
        let x = xorshift_values(heads * config.head_dim, 0x0D3A, 1.0);
        for position in ROPE_POSITIONS {
            let input = Tensor::from_slice(&x, (1, heads, config.head_dim), device)?;
            let rotated = rope(&input, &table, &[position])?
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let expected = rope_reference(&x, position, config);
            let error = rotated
                .iter()
                .zip(&expected)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0f64, |worst, e| {
                    if e.is_finite() {
                        worst.max(e)
                    } else {
                        f64::INFINITY
                    }
                });
            checks.push((format!("{name} {:?}", config.rope_layout), position, error));
            if error > ROPE_TOLERANCE {
                *breaches += 1;
            }
        }
    }
    if *breaches > 0 {
        return Err(load_error(
            backend,
            format!(
                "{breaches} RoPE rotations deviate from F64 by more than {ROPE_TOLERANCE:.0e}: \
                 {checks:?}"
            ),
        ));
    }
    Ok(())
}

/// The FP8 GEMV of `x` (`[2, cols]` F32, rounded to `dtype` for the
/// kernel) over `[rows, cols]` resident bytes `weights` and `scales`,
/// against the original tensor's F64 dequantization on [`FP8_ROWS`]
/// sampled rows.
pub(crate) fn fp8_error(
    weights: &Tensor,
    scales: &Tensor,
    original: &Fp8BlockWeight<'_>,
    dtype: DType,
) -> Result<f64> {
    let (rows, cols) = (original.rows, original.cols);
    let device = weights.device();
    let x = xorshift_values(2 * cols, 0xF8, 1.0);
    let input = Tensor::from_slice(&x, (2, cols), device)?.to_dtype(dtype)?;
    let read = input
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    let y = crate::kernels::cuda::fp8::block_fp8_projection(
        &input,
        &weights.contiguous()?,
        &scales.contiguous()?,
    )?
    .to_dtype(DType::F32)?
    .to_device(&Device::Cpu)?
    .to_vec2::<f32>()?;
    let count = rows.min(FP8_ROWS);
    let sampled = (0..count).map(|i| i * rows / count).collect::<Vec<_>>();
    let mut got = Vec::with_capacity(2 * count);
    let mut expected = Vec::with_capacity(2 * count);
    for (m, input) in read.iter().enumerate() {
        for &row in &sampled {
            let w = original
                .dequantized_row(row)
                .ok_or_else(|| Error::ModelLoadError(format!("FP8 check: no row {row}")))?;
            expected.push(
                w.iter()
                    .zip(input)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>() as f32,
            );
            got.push(y[m][row]);
        }
    }
    Ok(relative_l2(&got, &expected))
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "f32",
    }
}

/// Checks the stacked experts of `trunk` against the checkpoint's originals
/// with `dtype` activations (the fused path's), recording each projection
/// in `checks` and each breach in `breaches`. A breach then runs again with
/// `fallback` activations when the backend has one (Metal's F32), and the
/// load is refused when that misses too, or when there is none. Returns
/// whether every projection passed at `dtype`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_fp8(
    checkpoint: &FishDrama3Checkpoint,
    config: &FishDrama3Config,
    trunk: &DeviceTrunk,
    dtype: DType,
    fallback: Option<DType>,
    backend: &str,
    checks: &mut Vec<(String, f64)>,
    breaches: &mut u64,
) -> Result<bool> {
    let t = &config.trunk;
    let (inter, block) = (t.moe.intermediate, config.fp8_block);
    let layers = [0, t.layers.len() - 1];
    let experts = [0, t.moe.experts - 1];
    let mut fused_sound = true;
    for &layer in layers
        .iter()
        .take(if layers[0] == layers[1] { 1 } else { 2 })
    {
        let moe = &trunk.layers[layer].moe;
        for &e in experts
            .iter()
            .take(if experts[0] == experts[1] { 1 } else { 2 })
        {
            let (w13, s13) = (moe.w13.get(e)?, moe.s13.get(e)?);
            let (w2, s2) = (moe.w2.get(e)?, moe.s2.get(e)?);
            let kinds = [
                (
                    "gate_proj",
                    w13.narrow(0, 0, inter)?,
                    s13.narrow(0, 0, inter / block)?,
                ),
                (
                    "up_proj",
                    w13.narrow(0, inter, inter)?,
                    s13.narrow(0, inter / block, inter / block)?,
                ),
                ("down_proj", w2, s2),
            ];
            for (kind, weights, scales) in kinds {
                let module = format!("{}.{kind}", expert(layer, e));
                let (rows, cols) = weights.dims2()?;
                let original = checkpoint.fp8_block_weight(&module, rows, cols, block)?;
                let error = fp8_error(&weights, &scales, &original, dtype)?;
                checks.push((format!("{module} {}", dtype_name(dtype)), error));
                if error <= FP8_GEMV_TOLERANCE {
                    continue;
                }
                *breaches += 1;
                fused_sound = false;
                let Some(fallback) = fallback else {
                    return Err(load_error(
                        backend,
                        format!(
                            "the FP8 GEMV over {module} deviates by {error:.3e} from the \
                             checkpoint (tolerance {FP8_GEMV_TOLERANCE:.0e})"
                        ),
                    ));
                };
                let full = fp8_error(&weights, &scales, &original, fallback)?;
                checks.push((format!("{module} {}", dtype_name(fallback)), full));
                if full > FP8_GEMV_TOLERANCE {
                    return Err(load_error(
                        backend,
                        format!(
                            "the FP8 GEMV over {module} deviates by {full:.3e} with {} \
                             activations (tolerance {FP8_GEMV_TOLERANCE:.0e})",
                            dtype_name(fallback).to_uppercase()
                        ),
                    ));
                }
            }
        }
    }
    Ok(fused_sound)
}
