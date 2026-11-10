//! An operator's opt-in to run a codec decoder in BF16 on CUDA, and the
//! load-time check that keeps it only when its samples stay close to the
//! F32 decoder's.
//!
//! The check decodes fixed, seeded inputs through both and measures the
//! BF16 samples' error against the F32 ones: the relative L2 error
//! `‖bf16 − f32‖ / ‖f32‖` must be at most [`MAX_REL_L2`] and every sample
//! finite. The measured error is kept for the model diagnostics, so a run
//! shows how close the codec came, not only whether it passed.

use candle_core::DType;
use serde_json::json;

/// The load check's bound on the relative L2 error of the reduced-precision
/// samples: 5%, a 26 dB signal-to-error ratio.
pub(crate) const MAX_REL_L2: f64 = 0.05;

/// What a codec decoder runs in, and why.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CodecPrecision {
    /// The dtype the decoder runs in.
    pub(crate) dtype: DType,
    /// Whether the operator asked for BF16.
    pub(crate) requested: bool,
    /// The load-time check's relative L2 error and largest absolute sample
    /// error, when it ran.
    pub(crate) rel_l2: Option<f64>,
    pub(crate) max_abs: Option<f64>,
    pub(crate) reason: String,
}

impl CodecPrecision {
    /// F32, for `reason`.
    pub(crate) fn f32(requested: bool, reason: impl Into<String>) -> Self {
        Self {
            dtype: DType::F32,
            requested,
            rel_l2: None,
            max_abs: None,
            reason: reason.into(),
        }
    }

    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        json!({
            "dtype": format!("{:?}", self.dtype),
            "requested_bf16": self.requested,
            "check_rel_l2": self.rel_l2,
            "check_max_abs": self.max_abs,
            "max_rel_l2": MAX_REL_L2,
            "reason": self.reason,
        })
    }
}

/// The check's verdict on `candidate` samples (in `dtype`) against the F32
/// `reference`, at a relative L2 bound of `max_rel_l2` ([`MAX_REL_L2`] at
/// load): the precision to record, and whether the candidate is kept.
pub(crate) fn judge(
    reference: &[f32],
    candidate: &[f32],
    dtype: DType,
    max_rel_l2: f64,
) -> (CodecPrecision, bool) {
    if reference.len() != candidate.len() || reference.is_empty() {
        return (
            CodecPrecision::f32(
                true,
                format!(
                    "the {dtype:?} check decoded {} samples against {}",
                    candidate.len(),
                    reference.len()
                ),
            ),
            false,
        );
    }
    if candidate.iter().any(|sample| !sample.is_finite()) {
        return (
            CodecPrecision::f32(true, format!("{dtype:?} decoded non-finite samples")),
            false,
        );
    }
    let (mut error, mut energy, mut max_abs) = (0f64, 0f64, 0f64);
    for (&a, &b) in reference.iter().zip(candidate) {
        let difference = f64::from(b) - f64::from(a);
        error += difference * difference;
        energy += f64::from(a) * f64::from(a);
        max_abs = max_abs.max(difference.abs());
    }
    let rel_l2 = if energy > 0.0 {
        (error / energy).sqrt()
    } else if error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let kept = rel_l2 <= max_rel_l2;
    let reason = if kept {
        format!("{dtype:?} samples within {max_rel_l2} relative L2 error of F32 at load")
    } else {
        format!("{dtype:?} samples {rel_l2:.4} relative L2 error from F32, past {max_rel_l2}")
    };
    let precision = CodecPrecision {
        dtype: if kept { dtype } else { DType::F32 },
        requested: true,
        rel_l2: Some(rel_l2),
        max_abs: Some(max_abs),
        reason,
    };
    (precision, kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_check_keeps_close_samples_and_refuses_far_or_broken_ones() {
        let reference = (0..400)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect::<Vec<_>>();
        let close = reference
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 2 == 0 { 0.004 } else { -0.004 })
            .collect::<Vec<_>>();
        let (precision, kept) = judge(&reference, &close, DType::BF16, MAX_REL_L2);
        assert!(kept, "{precision:?}");
        assert_eq!(precision.dtype, DType::BF16);
        assert!(precision.rel_l2.unwrap() < 0.02);
        assert!((precision.max_abs.unwrap() - 0.004).abs() < 1e-6);

        let far = reference.iter().map(|v| v * 1.2).collect::<Vec<_>>();
        let (precision, kept) = judge(&reference, &far, DType::BF16, MAX_REL_L2);
        assert!(!kept);
        assert_eq!(precision.dtype, DType::F32);
        assert!(precision.reason.contains("past"), "{}", precision.reason);
        assert!(judge(&reference, &far, DType::BF16, 0.25).1);

        let mut broken = close.clone();
        broken[7] = f32::NAN;
        assert!(!judge(&reference, &broken, DType::BF16, MAX_REL_L2).1);
        assert!(!judge(&reference, &close[1..], DType::BF16, MAX_REL_L2).1);
    }
}
