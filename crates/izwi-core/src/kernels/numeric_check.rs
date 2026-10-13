//! Load-time numeric self-checks for lower-precision paths.
//!
//! A family that lowers precision on a device (F16 weights, F16 or BF16
//! activations) runs a short forward on deterministic, unstructured input at
//! load and keeps the lower-precision representation only when every probed
//! tensor is finite and stays inside the F16-safe range. llama.cpp (#26223)
//! and izwi's own Gemma4 case turned all-NaN once activations passed 65504,
//! so the default ceiling sits well below it. A failed check records why, in
//! the shape of a `DTypeSelection` reason, so diagnostics can say which
//! representation loaded and why.
//!
//! The comparison helpers saturate a non-finite difference to infinity: a
//! fold with `f32::max` returns the non-NaN operand and reads a NaN output as
//! zero error.

use std::borrow::Cow;

use candle_core::{DType, Device, Result, Tensor};

/// The default max-abs ceiling for a probed tensor: half of F16's 65504, so a
/// check fails before any value can overflow F16 storage or staging.
pub const F16_SAFE_MAX_ABS: f32 = 3.0e4;

/// Deterministic, unstructured values in `[-scale, scale)`. Smooth inputs
/// (sinusoids) can hide a wrong mapping that xorshift values expose.
pub fn xorshift_values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

/// An F32 tensor of [`xorshift_values`] with `shape`, on `device`.
pub fn xorshift_tensor(shape: &[usize], seed: u64, scale: f32, device: &Device) -> Result<Tensor> {
    let len = shape.iter().product();
    Tensor::from_vec(xorshift_values(len, seed, scale), shape, device)
}

/// `||values - reference|| / ||reference||`, infinite when either side is
/// non-finite.
pub fn relative_l2(values: &[f32], reference: &[f32]) -> f64 {
    if values.len() != reference.len() {
        return f64::INFINITY;
    }
    let (error, norm) =
        values
            .iter()
            .zip(reference)
            .fold((0f64, 0f64), |(error, norm), (&a, &b)| {
                let diff = f64::from(a) - f64::from(b);
                (error + diff * diff, norm + f64::from(b) * f64::from(b))
            });
    let relative = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    if relative.is_finite() {
        relative
    } else {
        f64::INFINITY
    }
}

/// The largest `|a - b|`, infinite when any pair differs by a non-finite
/// amount (a NaN or an infinity on either side).
pub fn max_abs_diff(values: &[f32], reference: &[f32]) -> f32 {
    if values.len() != reference.len() {
        return f32::INFINITY;
    }
    values.iter().zip(reference).fold(0f32, |worst, (&a, &b)| {
        let diff = (a - b).abs();
        if diff.is_finite() {
            if diff > worst {
                diff
            } else {
                worst
            }
        } else {
            f32::INFINITY
        }
    })
}

/// A tensor's values as host F32, flattened.
pub fn host_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()
}

/// One probed tensor's statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbedTensor {
    pub name: String,
    /// The largest finite magnitude.
    pub max_abs: f32,
    pub non_finite: usize,
}

impl ProbedTensor {
    pub fn from_values(name: impl Into<String>, values: &[f32]) -> Self {
        let mut max_abs = 0f32;
        let mut non_finite = 0;
        for value in values {
            if value.is_finite() {
                max_abs = max_abs.max(value.abs());
            } else {
                non_finite += 1;
            }
        }
        Self {
            name: name.into(),
            max_abs,
            non_finite,
        }
    }
}

/// The outcome of [`run_numeric_self_check`].
#[derive(Debug, Clone, PartialEq)]
pub struct NumericSelfCheckReport {
    pub label: Cow<'static, str>,
    pub passed: bool,
    /// Why the check passed or failed, for a `DTypeSelection`-style reason.
    pub reason: String,
    pub max_abs_limit: f32,
    pub tensors: Vec<ProbedTensor>,
}

impl NumericSelfCheckReport {
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "label": self.label,
            "passed": self.passed,
            "reason": self.reason,
            "max_abs_limit": self.max_abs_limit,
            "tensors": self
                .tensors
                .iter()
                .map(|tensor| serde_json::json!({
                    "name": tensor.name,
                    "max_abs": tensor.max_abs,
                    "non_finite": tensor.non_finite,
                }))
                .collect::<Vec<_>>(),
        })
    }
}

/// Runs `probe` and checks every tensor it returns: all values finite and
/// each tensor's max-abs below `max_abs_limit`. `probe` builds its own input,
/// normally from [`xorshift_tensor`]. An error from `probe` fails the check
/// with that error as the reason; it is never propagated, so the caller can
/// fall back to its reference representation.
pub fn run_numeric_self_check<F, E>(
    label: impl Into<Cow<'static, str>>,
    max_abs_limit: f32,
    probe: F,
) -> NumericSelfCheckReport
where
    F: FnOnce() -> std::result::Result<Vec<(String, Tensor)>, E>,
    E: std::fmt::Display,
{
    let label = label.into();
    let probed = probe().map_err(|err| err.to_string()).and_then(|outputs| {
        outputs
            .into_iter()
            .map(|(name, tensor)| {
                let values = host_f32(&tensor).map_err(|err| err.to_string())?;
                Ok(ProbedTensor::from_values(name, &values))
            })
            .collect::<std::result::Result<Vec<_>, String>>()
    });
    let tensors = match probed {
        Ok(tensors) => tensors,
        Err(err) => {
            return NumericSelfCheckReport {
                reason: format!("{label} self-check failed to run: {err}"),
                label,
                passed: false,
                max_abs_limit,
                tensors: Vec::new(),
            };
        }
    };
    let failure = if tensors.is_empty() {
        Some(format!("{label} self-check probed no tensors"))
    } else if let Some(tensor) = tensors.iter().find(|tensor| tensor.non_finite > 0) {
        Some(format!(
            "{label} self-check: {} has {} non-finite values",
            tensor.name, tensor.non_finite
        ))
    } else {
        tensors
            .iter()
            .find(|tensor| tensor.max_abs >= max_abs_limit)
            .map(|tensor| {
                format!(
                    "{label} self-check: {} reaches |x| = {:.4e}, above the {:.1e} limit",
                    tensor.name, tensor.max_abs, max_abs_limit
                )
            })
    };
    let passed = failure.is_none();
    let reason = failure.unwrap_or_else(|| {
        let peak = tensors
            .iter()
            .map(|tensor| tensor.max_abs)
            .fold(0f32, f32::max);
        format!(
            "{label} self-check passed: {} tensors finite, max |x| = {peak:.4e} below {max_abs_limit:.1e}",
            tensors.len()
        )
    });
    NumericSelfCheckReport {
        label,
        passed,
        reason,
        max_abs_limit,
        tensors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_values_are_bounded_unstructured_and_repeatable() {
        let values = xorshift_values(4096, 7, 0.5);
        assert_eq!(values, xorshift_values(4096, 7, 0.5));
        assert_ne!(values, xorshift_values(4096, 8, 0.5));
        assert!(values.iter().all(|value| (-0.5..0.5).contains(value)));
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        // Unstructured: neighbouring values are not correlated.
        let lag1 =
            values.windows(2).map(|pair| pair[0] * pair[1]).sum::<f32>() / values.len() as f32;
        assert!(lag1.abs() < 0.01, "lag-1 correlation {lag1}");
    }

    #[test]
    fn comparisons_saturate_non_finite_differences() {
        let reference = [1.0, -2.0, 3.0];
        assert_eq!(max_abs_diff(&reference, &reference), 0.0);
        assert_eq!(relative_l2(&reference, &reference), 0.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let values = [1.0, bad, 3.0];
            assert_eq!(max_abs_diff(&values, &reference), f32::INFINITY);
            assert_eq!(relative_l2(&values, &reference), f64::INFINITY);
            // A NaN in the reference counts too.
            assert_eq!(max_abs_diff(&reference, &values), f32::INFINITY);
        }
        // A NaN after a large finite difference still saturates.
        assert_eq!(max_abs_diff(&[10.0, f32::NAN], &[0.0, 0.0]), f32::INFINITY);
        assert_eq!(max_abs_diff(&[1.0], &[1.0, 2.0]), f32::INFINITY);
        let close = relative_l2(&[1.0, 2.0], &[1.0, 2.002]);
        assert!(close > 0.0 && close < 1e-3, "{close}");
    }

    #[test]
    fn self_check_passes_finite_in_range_outputs() {
        let report = run_numeric_self_check("probe", F16_SAFE_MAX_ABS, || {
            let x = xorshift_tensor(&[4, 8], 1, 2.0, &Device::Cpu)?;
            Ok::<_, candle_core::Error>(vec![("x".into(), x.clone()), ("y".into(), (x * 100.0)?)])
        });
        assert!(report.passed, "{}", report.reason);
        assert_eq!(report.tensors.len(), 2);
        assert!(report.tensors[1].max_abs <= 200.0 && report.tensors[1].max_abs > 100.0);
        assert!(report.reason.contains("passed"));
        assert_eq!(report.diagnostics()["passed"], true);
    }

    #[test]
    fn self_check_fails_on_nan_overflow_errors_and_empty_probes() {
        let tensor = |values: Vec<f32>| Tensor::new(values, &Device::Cpu);

        let nan = run_numeric_self_check("nan", F16_SAFE_MAX_ABS, || {
            Ok::<_, candle_core::Error>(vec![("h".into(), tensor(vec![1.0, f32::NAN])?)])
        });
        assert!(!nan.passed);
        assert!(nan.reason.contains("non-finite"), "{}", nan.reason);
        assert_eq!(nan.tensors[0].non_finite, 1);

        let inf = run_numeric_self_check("inf", F16_SAFE_MAX_ABS, || {
            Ok::<_, candle_core::Error>(vec![("h".into(), tensor(vec![f32::INFINITY])?)])
        });
        assert!(!inf.passed);

        let large = run_numeric_self_check("large", F16_SAFE_MAX_ABS, || {
            Ok::<_, candle_core::Error>(vec![
                ("ok".into(), tensor(vec![1.0])?),
                ("big".into(), tensor(vec![-4.0e4])?),
            ])
        });
        assert!(!large.passed);
        assert!(large.reason.contains("big"), "{}", large.reason);

        let error = run_numeric_self_check("error", F16_SAFE_MAX_ABS, || {
            Err(candle_core::Error::Msg("kernel refused".into()))
        });
        assert!(!error.passed);
        assert!(error.reason.contains("kernel refused"), "{}", error.reason);

        let empty = run_numeric_self_check("empty", F16_SAFE_MAX_ABS, || {
            Ok::<_, candle_core::Error>(Vec::new())
        });
        assert!(!empty.passed);
    }
}
