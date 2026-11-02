//! Dtype policy and the load-time Q8_0 self-check (D4).
//!
//! The CPU holds every projection as Q8_0 to fit about 37 GiB. Rounding
//! keeps each block's error under 1/254 of its largest value, which holds
//! an ordinary class near 1% (weights and the step kernel's Q8_0
//! activations together) and an outlier-heavy one within a few percent.
//! What that bound misses is the block scale itself,
//! stored as F16: weights under about 1e-5 flush to zero and over about 8e6
//! overflow, and a layout or kernel fault breaks everything. So each class
//! is checked before it loads: rows sampled across its first instances run
//! through the Q8_0 step kernel on xorshift input, and their relative L2
//! error against the exact F32 weights (in F64) decides the class. Over
//! [`Q8_0_TOLERANCE`] the class stays F32 and the report counts it, so a
//! degraded load shows in diagnostics rather than in the audio.
//!
//! Metal (D5) runs F16 activations beside an F32 residual. Its load-time
//! checks (`metal::self_check`) report through [`FishDrama3MetalChecks`]:
//! the F16 range probe per layer, the FP8 GEMV against the original
//! tensors, and the host RoPE tables. CUDA runs BF16 throughout; its checks
//! (`cuda::self_check`) report through [`FishDrama3CudaChecks`]: the same
//! table checks, and the device checks that gate its kernels.
//! `IZWI_FISH_DRAMA3_DTYPE` overrides the activation dtype
//! ([`activation_dtype`]) and `IZWI_FISH_DRAMA3_MOE_BACKEND` the routed
//! experts' path ([`moe_backend`]).

use candle_core::{DType, Device};
use serde_json::json;

use super::checkpoint::{
    expert, tower_head, tower_shared_layer, trunk_layer, FishDrama3Checkpoint,
};
use super::config::FishDrama3Config;
use super::device::trunk::FishDrama3MoeBackend;
use super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::residency::{
    FishDrama3ClassForms, FishDrama3CpuPrecision, FishDrama3ProjectionClass, FishDrama3Residency,
    FishDrama3WeightForm,
};
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{relative_l2, xorshift_values};

/// The largest relative L2 error a Q8_0 class may show on the self-check.
/// Weights of ordinary spread measure 0.5–1% (Q8_0 weights and the step
/// kernel's Q8_0 activations together), and a 40× outlier in every block
/// about 2.5%; an underflowing block scale or a layout fault measures
/// 50–140%.
pub const Q8_0_TOLERANCE: f64 = 5e-2;
/// Rows sampled per checked tensor.
pub const SELF_CHECK_ROWS: usize = 64;
/// Input rows per check.
const SELF_CHECK_INPUTS: usize = 2;

/// One checked tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct FishDrama3ProjectionCheck {
    pub class: FishDrama3ProjectionClass,
    pub tensor: String,
    pub relative_l2: f64,
}

/// What the self-check found and decided.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FishDrama3PrecisionReport {
    pub tolerance: f64,
    pub checks: Vec<FishDrama3ProjectionCheck>,
    /// Classes kept F32 because their Q8_0 check failed (the diagnostics
    /// counter; zero on a healthy load).
    pub q8_fallback_classes: u64,
    /// The forms each class loaded in.
    pub forms: Vec<(FishDrama3ProjectionClass, FishDrama3WeightForm)>,
}

impl FishDrama3PrecisionReport {
    pub fn diagnostics(&self) -> serde_json::Value {
        json!({
            "q8_0_tolerance": self.tolerance,
            "q8_0_fallback_classes": self.q8_fallback_classes,
            "projection_checks": self.checks.iter().map(|check| json!({
                "class": check.class.name(),
                "tensor": check.tensor,
                "relative_l2": check.relative_l2,
            })).collect::<Vec<_>>(),
            "forms": self.forms.iter().map(|(class, form)| json!({
                "class": class.name(),
                "form": format!("{form:?}"),
            })).collect::<Vec<_>>(),
        })
    }
}

/// Overrides a load's activation dtype: `f16` or `f32` on Metal, `f32` on
/// the CPU (whose numerics are F32 throughout).
pub const DTYPE_ENV: &str = "IZWI_FISH_DRAMA3_DTYPE";
/// `fused` (the default) or `loop`: how a Metal or CUDA load runs its
/// routed experts.
pub const MOE_BACKEND_ENV: &str = "IZWI_FISH_DRAMA3_MOE_BACKEND";

/// The activation dtype a load on `residency` runs: `raw` (the
/// [`DTYPE_ENV`] value) validated for the backend, else the backend's own
/// (§6: F16 on Metal, F32 on the CPU, BF16 on CUDA). A value the backend
/// cannot run is an error, never silently ignored.
pub fn activation_dtype(residency: FishDrama3Residency, raw: Option<&str>) -> Result<DType> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return check_activations(residency, None);
    };
    let dtype = match raw.to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "float16" => DType::F16,
        "f32" | "fp32" | "float32" => DType::F32,
        "bf16" | "bfloat16" => DType::BF16,
        _ => {
            return Err(Error::ModelLoadError(format!(
                "{DTYPE_ENV}={raw} is not a dtype (f16, bf16, f32)"
            )))
        }
    };
    check_activations(residency, Some(dtype))
}

/// `requested` if `residency` runs it, else an error; the backend's own
/// activation dtype when nothing is requested.
pub fn check_activations(
    residency: FishDrama3Residency,
    requested: Option<DType>,
) -> Result<DType> {
    let default = match residency {
        FishDrama3Residency::Metal => DType::F16,
        FishDrama3Residency::Cpu => DType::F32,
        FishDrama3Residency::Cuda => DType::BF16,
    };
    let Some(dtype) = requested else {
        return Ok(default);
    };
    let allowed = match residency {
        FishDrama3Residency::Metal => matches!(dtype, DType::F16 | DType::F32),
        FishDrama3Residency::Cpu | FishDrama3Residency::Cuda => dtype == default,
    };
    if !allowed {
        return Err(Error::ModelLoadError(format!(
            "{DTYPE_ENV}={dtype:?}: Fish Audio Drama 3 on {residency:?} runs {}",
            match residency {
                FishDrama3Residency::Metal => "f16 or f32 activations",
                FishDrama3Residency::Cpu => "f32 activations",
                FishDrama3Residency::Cuda => "bf16 activations",
            }
        )));
    }
    Ok(dtype)
}

/// The routed experts' path from `raw` (the [`MOE_BACKEND_ENV`] value).
pub fn moe_backend(raw: Option<&str>) -> Result<FishDrama3MoeBackend> {
    match raw.map(str::trim).filter(|raw| !raw.is_empty()) {
        None => Ok(FishDrama3MoeBackend::Fused),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "fused" => Ok(FishDrama3MoeBackend::Fused),
            "loop" => Ok(FishDrama3MoeBackend::Loop),
            _ => Err(Error::ModelLoadError(format!(
                "{MOE_BACKEND_ENV}={raw} is not a MoE backend (fused, loop)"
            ))),
        },
    }
}

/// One probed component of a Metal load: its peak magnitude on the probe and
/// whether it left F16's safe range.
#[derive(Debug, Clone, PartialEq)]
pub struct FishDrama3ProbedComponent {
    pub name: String,
    pub max_abs: f32,
    pub non_finite: usize,
    pub breached: bool,
}

/// What a Metal load's checks found and decided (`metal::self_check`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FishDrama3MetalChecks {
    /// The activation dtype the load ran its layers in by default.
    pub activations: String,
    /// The F16 ceiling the probe held activations to.
    pub max_abs_limit: f32,
    /// Every trunk layer, the final norm, every tower layer, the compact
    /// head and the codebook heads.
    pub components: Vec<FishDrama3ProbedComponent>,
    /// Components moved to F32 activations (the range probe's counter).
    pub f16_range_breaches: u64,
    /// The paged KV dtype both domains store (F32 when a K or V breached).
    pub kv_dtype: String,
    /// Trunk layers whose routed experts run the host loop.
    pub moe_loop_layers: Vec<usize>,
    /// Relative L2 of the FP8 GEMV against the original tensors' F64
    /// dequantization, per checked projection.
    pub fp8_checks: Vec<(String, f64)>,
    /// FP8 checks over [`FP8_GEMV_TOLERANCE`] (the counter).
    pub fp8_gemv_breaches: u64,
    /// The largest error of the device rotation against F64, per layout and
    /// position.
    pub rope_checks: Vec<(String, usize, f64)>,
    /// RoPE checks over [`ROPE_TOLERANCE`] (the counter).
    pub rope_table_breaches: u64,
}

/// The largest relative L2 the FP8 GEMV (16-bit activations in and out) may
/// show against the F64 dequantization of the original tensor: F16 input
/// and output rounding measure about 5e-4 and BF16's about 3e-3; a scale or
/// layout fault, order 1.
pub const FP8_GEMV_TOLERANCE: f64 = 1e-2;
/// The largest absolute error of a rotated unit-scale vector against F64:
/// F32 tables and F32 rotation measure under 1e-6 at any position; an F32
/// angle product at position 262,143 is off by up to 1.6e-2 rad.
pub const ROPE_TOLERANCE: f64 = 1e-5;

impl FishDrama3MetalChecks {
    pub fn diagnostics(&self) -> serde_json::Value {
        json!({
            "activations": self.activations,
            "max_abs_limit": self.max_abs_limit,
            "f16_range_breaches": self.f16_range_breaches,
            "kv_dtype": self.kv_dtype,
            "moe_loop_layers": self.moe_loop_layers,
            "fp8_gemv_breaches": self.fp8_gemv_breaches,
            "rope_table_breaches": self.rope_table_breaches,
            "components": self.components.iter().map(|c| json!({
                "name": c.name,
                "max_abs": c.max_abs,
                "non_finite": c.non_finite,
                "breached": c.breached,
            })).collect::<Vec<_>>(),
            "fp8_checks": self.fp8_checks.iter().map(|(name, error)| json!({
                "projection": name,
                "relative_l2": error,
            })).collect::<Vec<_>>(),
            "rope_checks": self.rope_checks.iter().map(|(layout, position, error)| json!({
                "layout": layout,
                "position": position,
                "max_abs_error": error,
            })).collect::<Vec<_>>(),
        })
    }
}

/// What a CUDA load's checks found and decided (`cuda::self_check`). The
/// table checks refuse a load that misses them; a kernel check that fails
/// leaves its kernel off and counts it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FishDrama3CudaChecks {
    /// The activation dtype every layer runs (BF16).
    pub activations: String,
    /// Relative L2 of the FP8 GEMV against the original tensors' F64
    /// dequantization, per checked projection.
    pub fp8_checks: Vec<(String, f64)>,
    /// FP8 checks over [`FP8_GEMV_TOLERANCE`] (the counter; the load is
    /// refused at the first).
    pub fp8_gemv_breaches: u64,
    /// The largest error of the device rotation against F64, per layout and
    /// position.
    pub rope_checks: Vec<(String, usize, f64)>,
    /// RoPE checks over [`ROPE_TOLERANCE`] (the counter).
    pub rope_table_breaches: u64,
    /// The row-invariant multi-row GEMV's device check: `verified`, or why
    /// steps run one GEMV per row instead.
    pub row_gemv: String,
    /// 1 when that check failed.
    pub row_gemv_breaches: u64,
    /// The chunked DeltaNet prefill's device check: `verified`, or why long
    /// spans run the per-row kernels instead.
    pub chunked_deltanet: String,
    /// 1 when that check failed.
    pub chunked_deltanet_breaches: u64,
    /// The grouped MoE prefill's kernels: `tensor-core` once their device
    /// check passed, else `warp` and why.
    pub grouped_moe: String,
    /// 1 when the tensor-core check failed.
    pub grouped_mma_breaches: u64,
    /// The paged KV dtype both domains store (BF16).
    pub kv_dtype: String,
}

impl FishDrama3CudaChecks {
    pub fn diagnostics(&self) -> serde_json::Value {
        json!({
            "activations": self.activations,
            "kv_dtype": self.kv_dtype,
            "fp8_gemv_breaches": self.fp8_gemv_breaches,
            "rope_table_breaches": self.rope_table_breaches,
            "row_gemv": self.row_gemv,
            "row_gemv_breaches": self.row_gemv_breaches,
            "chunked_deltanet": self.chunked_deltanet,
            "chunked_deltanet_breaches": self.chunked_deltanet_breaches,
            "grouped_moe": self.grouped_moe,
            "grouped_mma_breaches": self.grouped_mma_breaches,
            "fp8_checks": self.fp8_checks.iter().map(|(name, error)| json!({
                "projection": name,
                "relative_l2": error,
            })).collect::<Vec<_>>(),
            "rope_checks": self.rope_checks.iter().map(|(layout, position, error)| json!({
                "layout": layout,
                "position": position,
                "max_abs_error": error,
            })).collect::<Vec<_>>(),
        })
    }
}

/// A tensor the check samples: the BF16 `.weight` name, or the FP8 module.
struct Probe {
    name: String,
    rows: usize,
    cols: usize,
    fp8: bool,
}

/// The first instances of each class: the first layer of the right kind,
/// the first and last routed experts of layer 0, the first tower layer and
/// head.
fn probes(config: &FishDrama3Config, class: FishDrama3ProjectionClass) -> Result<Vec<Probe>> {
    let t = &config.trunk;
    let h = t.hidden;
    let first = |layers: Vec<usize>| {
        layers.first().copied().ok_or_else(|| {
            Error::ModelLoadError(format!(
                "Fish Audio Drama 3 self-check: no layer holds {}",
                class.name()
            ))
        })
    };
    let dense = |name: String, rows, cols| Probe {
        name,
        rows,
        cols,
        fp8: false,
    };
    Ok(match class {
        FishDrama3ProjectionClass::AttentionIn => {
            let x = format!(
                "{}.self_attn",
                trunk_layer(first(t.full_attention_layers())?)
            );
            let a = &t.attention;
            vec![
                dense(format!("{x}.q_proj.weight"), 2 * a.q_width(), h),
                dense(format!("{x}.k_proj.weight"), a.kv_width(), h),
                dense(format!("{x}.v_proj.weight"), a.kv_width(), h),
            ]
        }
        FishDrama3ProjectionClass::AttentionOut => {
            let x = format!(
                "{}.self_attn",
                trunk_layer(first(t.full_attention_layers())?)
            );
            vec![dense(
                format!("{x}.o_proj.weight"),
                h,
                t.attention.q_width(),
            )]
        }
        FishDrama3ProjectionClass::DeltaNetIn => {
            let x = format!(
                "{}.linear_attn",
                trunk_layer(first(t.linear_attention_layers())?)
            );
            vec![
                dense(
                    format!("{x}.in_proj_qkv.weight"),
                    t.linear.conv_channels(),
                    h,
                ),
                dense(format!("{x}.in_proj_z.weight"), t.linear.value_width(), h),
            ]
        }
        FishDrama3ProjectionClass::DeltaNetOut => {
            let x = format!(
                "{}.linear_attn",
                trunk_layer(first(t.linear_attention_layers())?)
            );
            vec![dense(
                format!("{x}.out_proj.weight"),
                h,
                t.linear.value_width(),
            )]
        }
        FishDrama3ProjectionClass::SharedExpert => {
            let x = format!("{}.mlp.shared_expert", trunk_layer(0));
            let si = t.moe.shared_intermediate;
            vec![
                dense(format!("{x}.gate_proj.weight"), si, h),
                dense(format!("{x}.up_proj.weight"), si, h),
                dense(format!("{x}.down_proj.weight"), h, si),
            ]
        }
        FishDrama3ProjectionClass::RoutedExpert => {
            let mi = t.moe.intermediate;
            let mut out = Vec::new();
            for e in [0, t.moe.experts - 1] {
                for (name, rows, cols) in [
                    ("gate_proj", mi, h),
                    ("up_proj", mi, h),
                    ("down_proj", h, mi),
                ] {
                    out.push(Probe {
                        name: format!("{}.{name}", expert(0, e)),
                        rows,
                        cols,
                        fp8: true,
                    });
                }
                if t.moe.experts == 1 {
                    break;
                }
            }
            out
        }
        FishDrama3ProjectionClass::TowerMerge => {
            let d = config.tower.dim;
            vec![dense(
                format!("{}.eh_proj.weight", tower_shared_layer(0)),
                d,
                2 * d,
            )]
        }
        FishDrama3ProjectionClass::TowerAttention => {
            let tower = &config.tower;
            let x = format!("{}.transformer_layer.attention", tower_shared_layer(0));
            vec![
                dense(format!("{x}.wqkv.weight"), tower.wqkv_rows(), tower.dim),
                dense(
                    format!("{x}.wo.weight"),
                    tower.dim,
                    tower.attention.q_width(),
                ),
            ]
        }
        FishDrama3ProjectionClass::TowerFeedForward => {
            let tower = &config.tower;
            let x = format!("{}.transformer_layer.feed_forward", tower_shared_layer(0));
            let (d, i) = (tower.dim, tower.intermediate);
            vec![
                dense(format!("{x}.w1.weight"), i, d),
                dense(format!("{x}.w3.weight"), i, d),
                dense(format!("{x}.w2.weight"), d, i),
            ]
        }
        FishDrama3ProjectionClass::TowerHead => vec![dense(
            tower_head(0),
            config.tower.codebook_vocab,
            config.tower.dim,
        )],
    })
}

/// Up to [`SELF_CHECK_ROWS`] rows spread over `rows`, ascending.
fn sampled_rows(rows: usize) -> Vec<usize> {
    let count = rows.min(SELF_CHECK_ROWS);
    (0..count).map(|i| i * rows / count).collect()
}

/// The relative L2 error of the Q8_0 step kernel over `values` (`rows`
/// sampled rows of `cols`, exact F32) against the same rows in F64.
pub fn q8_0_relative_error(values: &[f32], rows: usize, cols: usize) -> Result<f64> {
    let layer = FishDrama3Linear::from_values(
        values.to_vec(),
        rows,
        cols,
        FishDrama3WeightForm::Q8_0,
        &Device::Cpu,
    )?;
    let x = xorshift_values(SELF_CHECK_INPUTS * cols, 0xD3A3, 1.0);
    let y = layer.forward_host(&x, SELF_CHECK_INPUTS, FishDrama3KernelPhase::Step)?;
    let mut expected = vec![0f32; SELF_CHECK_INPUTS * rows];
    for m in 0..SELF_CHECK_INPUTS {
        let input = &x[m * cols..(m + 1) * cols];
        for (row, weights) in values.chunks_exact(cols).enumerate() {
            expected[m * rows + row] = input
                .iter()
                .zip(weights)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32;
        }
    }
    Ok(relative_l2(&y, &expected))
}

fn probe_values(
    checkpoint: &FishDrama3Checkpoint,
    probe: &Probe,
    block: usize,
) -> Result<Vec<f32>> {
    let rows = sampled_rows(probe.rows);
    if probe.fp8 {
        let weight = checkpoint.fp8_block_weight(&probe.name, probe.rows, probe.cols, block)?;
        let mut out = Vec::with_capacity(rows.len() * probe.cols);
        for row in rows {
            out.extend(weight.dequantized_row(row).ok_or_else(|| {
                Error::ModelLoadError(format!("{} has no row {row}", probe.name))
            })?);
        }
        Ok(out)
    } else {
        checkpoint.bf16_rows_to_f32(&probe.name, &rows)
    }
}

/// The form each of `classes` loads in on the CPU (the trunk's or the
/// tower's, [`FishDrama3ProjectionClass::TRUNK`] / `TOWER`), and the report.
pub fn decide_cpu_forms(
    checkpoint: &FishDrama3Checkpoint,
    config: &FishDrama3Config,
    precision: FishDrama3CpuPrecision,
    tolerance: f64,
    classes: &[FishDrama3ProjectionClass],
) -> Result<(FishDrama3ClassForms, FishDrama3PrecisionReport)> {
    let mut report = FishDrama3PrecisionReport {
        tolerance,
        ..Default::default()
    };
    let mut forms = match precision {
        FishDrama3CpuPrecision::F32 => FishDrama3ClassForms::uniform(FishDrama3WeightForm::F32),
        FishDrama3CpuPrecision::Q8_0 => FishDrama3ClassForms::of(FishDrama3Residency::Cpu),
    };
    if precision == FishDrama3CpuPrecision::Q8_0 {
        for &class in classes {
            let mut worst = 0f64;
            for probe in probes(config, class)? {
                let values = probe_values(checkpoint, &probe, config.fp8_block)?;
                let error = q8_0_relative_error(&values, values.len() / probe.cols, probe.cols)?;
                worst = worst.max(error);
                report.checks.push(FishDrama3ProjectionCheck {
                    class,
                    tensor: probe.name,
                    relative_l2: error,
                });
            }
            // `relative_l2` saturates a non-finite error to infinity.
            if worst > tolerance {
                tracing::warn!(
                    "Fish Audio Drama 3: Q8_0 {} projections deviate by {worst:.3e} \
                     (tolerance {tolerance:.1e}); keeping them F32",
                    class.name()
                );
                forms.set(class, FishDrama3WeightForm::F32);
                report.q8_fallback_classes += 1;
            }
        }
    }
    report.forms = classes
        .iter()
        .map(|&class| (class, forms.form(class)))
        .collect();
    Ok((forms, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dtype_and_expert_overrides_are_validated_per_backend() {
        use FishDrama3Residency::{Cpu, Cuda, Metal};
        assert_eq!(activation_dtype(Metal, None).unwrap(), DType::F16);
        assert_eq!(activation_dtype(Metal, Some(" F32 ")).unwrap(), DType::F32);
        assert_eq!(activation_dtype(Metal, Some("f16")).unwrap(), DType::F16);
        assert!(activation_dtype(Metal, Some("bf16")).is_err());
        assert!(activation_dtype(Metal, Some("f8")).is_err());
        assert_eq!(activation_dtype(Cpu, Some("")).unwrap(), DType::F32);
        assert_eq!(activation_dtype(Cpu, Some("fp32")).unwrap(), DType::F32);
        assert!(activation_dtype(Cpu, Some("f16")).is_err());
        assert_eq!(activation_dtype(Cuda, None).unwrap(), DType::BF16);
        assert_eq!(moe_backend(None).unwrap(), FishDrama3MoeBackend::Fused);
        assert_eq!(
            moe_backend(Some("LOOP")).unwrap(),
            FishDrama3MoeBackend::Loop
        );
        assert_eq!(
            moe_backend(Some("fused")).unwrap(),
            FishDrama3MoeBackend::Fused
        );
        assert!(moe_backend(Some("grouped")).is_err());
    }

    #[test]
    fn ordinary_weights_pass_and_f16_scale_underflow_fails_the_q8_0_check() {
        let (rows, cols) = (64, 256);
        let ordinary = xorshift_values(rows * cols, 11, 0.05);
        let error = q8_0_relative_error(&ordinary, rows, cols).unwrap();
        assert!(error < Q8_0_TOLERANCE / 2.0, "{error}");
        // A 40x outlier in every block flattens the rest of the block to a
        // few steps, and still stays inside the tolerance.
        let mut outliers = ordinary.clone();
        outliers.iter_mut().step_by(32).for_each(|v| *v = 2.0);
        let error = q8_0_relative_error(&outliers, rows, cols).unwrap();
        assert!(error < Q8_0_TOLERANCE, "{error}");
        // Block scales under F16's smallest subnormal store as zero.
        let tiny = ordinary.iter().map(|v| v * 1e-5).collect::<Vec<_>>();
        let error = q8_0_relative_error(&tiny, rows, cols).unwrap();
        assert!(error > 0.5, "{error}");
        assert_eq!(sampled_rows(1000).len(), SELF_CHECK_ROWS);
        assert_eq!(sampled_rows(10), (0..10).collect::<Vec<_>>());
    }
}
