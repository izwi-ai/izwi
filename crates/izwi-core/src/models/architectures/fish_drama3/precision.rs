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

use candle_core::Device;
use serde_json::json;

use super::checkpoint::{
    expert, tower_head, tower_shared_layer, trunk_layer, FishDrama3Checkpoint,
};
use super::config::FishDrama3Config;
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
