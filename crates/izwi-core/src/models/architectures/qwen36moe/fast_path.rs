//! Shared bookkeeping for Qwen3.6 fused fast paths (MoE experts, DeltaNet
//! decode, ...).
//!
//! Each block resolves at load to [`Qwen36FusedPath::Fused`] or to
//! [`Qwen36FusedPath::Legacy`] with a reason. A block only takes the fused path
//! after a self-check on its own device passes, and an environment switch can
//! force the legacy Candle chain without a code change. Diagnostics aggregate
//! the per-layer outcomes with [`summarize`].

use crate::error::{Error, Result};

/// Execution path a block resolved to at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Qwen36FusedPath {
    Fused,
    Legacy { reason: String },
}

impl Qwen36FusedPath {
    pub(crate) fn legacy(reason: impl Into<String>) -> Self {
        Self::Legacy {
            reason: reason.into(),
        }
    }

    pub(crate) fn is_fused(&self) -> bool {
        matches!(self, Self::Fused)
    }
}

/// Whether the environment switch `var` asks for the legacy path
/// (`legacy`, `off`, `0` or `false`, case-insensitive).
pub(crate) fn legacy_requested(var: &str) -> bool {
    legacy_value(std::env::var(var).ok().as_deref())
}

pub(crate) fn legacy_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "legacy" | "off" | "0" | "false"
        )
    })
}

/// Diagnostics summary over per-layer paths: overall backend
/// (`fused`/`legacy`/`mixed`/`none`), layer counts, and distinct reasons.
pub(crate) fn summarize<'a>(
    paths: impl IntoIterator<Item = &'a Qwen36FusedPath>,
) -> serde_json::Value {
    let mut fused = 0usize;
    let mut legacy = 0usize;
    let mut reasons: Vec<&str> = Vec::new();
    for path in paths {
        match path {
            Qwen36FusedPath::Fused => fused += 1,
            Qwen36FusedPath::Legacy { reason } => {
                legacy += 1;
                if !reasons.contains(&reason.as_str()) {
                    reasons.push(reason);
                }
            }
        }
    }
    let backend = match (fused, legacy) {
        (0, 0) => "none",
        (_, 0) => "fused",
        (0, _) => "legacy",
        _ => "mixed",
    };
    serde_json::json!({
        "backend": backend,
        "fused_layers": fused,
        "legacy_layers": legacy,
        "legacy_reasons": reasons,
    })
}

/// Self-check comparison of a fused output against its reference: relative L2
/// error at most `rel_l2`, every element within `max_frac` of the largest
/// reference magnitude, and no non-finite values.
pub(crate) fn compare_values(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    rel_l2: f64,
    max_frac: f32,
) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(Error::InferenceError(format!(
            "{label}: {} values vs {} in the reference",
            actual.len(),
            expected.len()
        )));
    }
    let mut err = 0f64;
    let mut norm = 0f64;
    let mut max_ref = 0f32;
    let mut max_err = 0f32;
    for (a, b) in actual.iter().zip(expected) {
        if !a.is_finite() {
            return Err(Error::InferenceError(format!(
                "{label}: fused path produced a non-finite value"
            )));
        }
        err += f64::from(a - b).powi(2);
        norm += f64::from(*b).powi(2);
        max_ref = max_ref.max(b.abs());
        max_err = max_err.max((a - b).abs());
    }
    let observed = (err / norm.max(f64::MIN_POSITIVE)).sqrt();
    if observed > rel_l2 || max_err > max_frac * max_ref.max(f32::MIN_POSITIVE) {
        return Err(Error::InferenceError(format!(
            "{label}: fused output diverges from the reference: relative L2 {observed:.4}, max error {max_err} vs max |ref| {max_ref}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_switch_values() {
        for value in ["legacy", " OFF ", "0", "False"] {
            assert!(legacy_value(Some(value)), "{value}");
        }
        for value in ["auto", "1", "fused", ""] {
            assert!(!legacy_value(Some(value)), "{value}");
        }
        assert!(!legacy_value(None));
    }

    #[test]
    fn summary_reports_mixed_paths_with_distinct_reasons() {
        let paths = [
            Qwen36FusedPath::Fused,
            Qwen36FusedPath::legacy("a"),
            Qwen36FusedPath::legacy("a"),
            Qwen36FusedPath::legacy("b"),
        ];
        let summary = summarize(&paths);
        assert_eq!(summary["backend"], "mixed");
        assert_eq!(summary["fused_layers"], 1);
        assert_eq!(summary["legacy_layers"], 3);
        assert_eq!(summary["legacy_reasons"], serde_json::json!(["a", "b"]));
        assert_eq!(summarize(&[])["backend"], "none");
    }

    #[test]
    fn compare_rejects_wrong_outputs_and_accepts_rounding_noise() {
        let reference = [1.0f32, -2.0, 0.5, 4.0];
        assert!(compare_values("t", &[1.004, -2.01, 0.498, 4.02], &reference, 0.03, 0.08).is_ok());
        assert!(compare_values("t", &[-1.0, 2.0, 0.5, 4.0], &reference, 0.03, 0.08).is_err());
        assert!(compare_values("t", &[f32::NAN, -2.0, 0.5, 4.0], &reference, 0.03, 0.08).is_err());
        assert!(compare_values("t", &[1.0], &reference, 0.03, 0.08).is_err());
    }
}
