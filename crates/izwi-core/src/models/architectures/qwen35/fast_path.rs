//! Bookkeeping for Qwen3.5 fused fast paths.
//!
//! Each block resolves at load to [`Qwen35FastPath::Fused`] or to
//! [`Qwen35FastPath::Legacy`] with a reason. A block takes the fused path
//! only after a self-check on its own device and weights passes. An
//! `IZWI_QWEN35_*` environment switch (any device) or the global CUDA
//! performance switches (CUDA only) force the Candle op chain without a code
//! change. Diagnostics aggregate the per-layer outcomes with [`summarize`].
//!
//! Copied from the Qwen3.6 fast-path bookkeeping rather than shared, because
//! the families are forked on purpose. The comparison here also rejects a
//! non-finite reference.

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::performance::CudaPerformanceConfig;

/// Execution path a block resolved to at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Qwen35FastPath {
    Fused,
    Legacy { reason: String },
}

impl Qwen35FastPath {
    pub(crate) fn legacy(reason: impl Into<String>) -> Self {
        Self::Legacy {
            reason: reason.into(),
        }
    }
}

/// The global CUDA performance switches as they apply to Qwen3.5: each field
/// is the reason its path is off, if it is. They bind on CUDA only; the
/// `IZWI_QWEN35_*` switches bind on every device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Qwen35CudaSwitches {
    pub(crate) fused_decode_off: Option<&'static str>,
    /// The izwi-owned K-quant MMVQ projections (`cuda.packed_projections`).
    pub(crate) kquant_off: Option<&'static str>,
    /// Whole-step CUDA graph decode (`cuda.decode_graphs`).
    pub(crate) graphs_off: Option<&'static str>,
}

impl Qwen35CudaSwitches {
    pub(crate) fn from_performance(
        backend: BackendKind,
        performance: &CudaPerformanceConfig,
    ) -> Self {
        if backend != BackendKind::Cuda {
            return Self::default();
        }
        let off = |mode: crate::performance::OptimizationMode, reason: &'static str| {
            if !performance.enabled() {
                Some("cuda.mode=off")
            } else {
                (!mode.enabled()).then_some(reason)
            }
        };
        Self {
            fused_decode_off: off(performance.fused_decode, "cuda.fused_decode=off"),
            kquant_off: off(
                performance.packed_projections,
                "cuda.packed_projections=off",
            ),
            graphs_off: off(performance.decode_graphs, "cuda.decode_graphs=off"),
        }
    }
}

/// Whether the environment switch `var` asks for the legacy path: a false
/// value (`false`, `0`, `no`, `off`) or its registered `legacy` word.
pub(crate) fn legacy_requested(var: &str) -> bool {
    !crate::env::flag_or(var, true)
}

/// Diagnostics summary over per-layer paths: overall backend
/// (`fused`/`legacy`/`mixed`/`none`), layer counts, and distinct reasons.
pub(crate) fn summarize<'a>(
    paths: impl IntoIterator<Item = &'a Qwen35FastPath>,
) -> serde_json::Value {
    let mut fused = 0usize;
    let mut legacy = 0usize;
    let mut reasons: Vec<&str> = Vec::new();
    for path in paths {
        match path {
            Qwen35FastPath::Fused => fused += 1,
            Qwen35FastPath::Legacy { reason } => {
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
/// reference magnitude, and every value (output and reference) finite.
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
        if !a.is_finite() || !b.is_finite() {
            return Err(Error::InferenceError(format!(
                "{label}: non-finite value (fused {a}, reference {b})"
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
            "{label}: fused output diverges from the reference: relative L2 {observed:.3e}, max error {max_err} vs max |ref| {max_ref}"
        )));
    }
    Ok(())
}

/// Deterministic, unstructured self-check inputs in `[-scale, scale)`.
/// Smooth inputs (sinusoids) can hide a wrong head mapping; xorshift values
/// do not.
pub(crate) fn xorshift_values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::OptimizationMode;

    #[test]
    fn cuda_switches_bind_on_cuda_only() {
        let mut performance = CudaPerformanceConfig::default();
        for backend in [BackendKind::Cuda, BackendKind::Metal, BackendKind::Cpu] {
            assert_eq!(
                Qwen35CudaSwitches::from_performance(backend, &performance),
                Qwen35CudaSwitches::default()
            );
        }
        performance.fused_decode = OptimizationMode::Off;
        assert_eq!(
            Qwen35CudaSwitches::from_performance(BackendKind::Cuda, &performance).fused_decode_off,
            Some("cuda.fused_decode=off")
        );
        assert_eq!(
            Qwen35CudaSwitches::from_performance(BackendKind::Metal, &performance),
            Qwen35CudaSwitches::default()
        );
        performance.fused_decode = OptimizationMode::Auto;
        performance.packed_projections = OptimizationMode::Off;
        let switches = Qwen35CudaSwitches::from_performance(BackendKind::Cuda, &performance);
        assert_eq!(switches.kquant_off, Some("cuda.packed_projections=off"));
        assert_eq!(switches.fused_decode_off, None);
        let performance = CudaPerformanceConfig {
            mode: OptimizationMode::Off,
            ..CudaPerformanceConfig::default()
        };
        let switches = Qwen35CudaSwitches::from_performance(BackendKind::Cuda, &performance);
        assert_eq!(switches.fused_decode_off, Some("cuda.mode=off"));
        assert_eq!(switches.kquant_off, Some("cuda.mode=off"));
    }

    #[test]
    fn summary_reports_mixed_paths_with_distinct_reasons() {
        let paths = [
            Qwen35FastPath::Fused,
            Qwen35FastPath::legacy("a"),
            Qwen35FastPath::legacy("a"),
            Qwen35FastPath::legacy("b"),
        ];
        let summary = summarize(&paths);
        assert_eq!(summary["backend"], "mixed");
        assert_eq!(summary["fused_layers"], 1);
        assert_eq!(summary["legacy_layers"], 3);
        assert_eq!(summary["legacy_reasons"], serde_json::json!(["a", "b"]));
        assert_eq!(summarize(&[])["backend"], "none");
    }

    #[test]
    fn compare_rejects_wrong_or_non_finite_values_and_accepts_rounding_noise() {
        let reference = [1.0f32, -2.0, 0.5, 4.0];
        assert!(
            compare_values("t", &[1.0001, -2.0002, 0.5, 4.0003], &reference, 1e-3, 1e-2).is_ok()
        );
        assert!(compare_values("t", &[-1.0, 2.0, 0.5, 4.0], &reference, 1e-3, 1e-2).is_err());
        assert!(compare_values("t", &[f32::NAN, -2.0, 0.5, 4.0], &reference, 1e-3, 1e-2).is_err());
        assert!(compare_values("t", &reference, &[f32::NAN, -2.0, 0.5, 4.0], 1e-3, 1e-2).is_err());
        assert!(compare_values("t", &[1.0], &reference, 1e-3, 1e-2).is_err());
    }

    #[test]
    fn xorshift_values_are_bounded_unstructured_and_repeatable() {
        let values = xorshift_values(4096, 7, 1.5);
        assert_eq!(values, xorshift_values(4096, 7, 1.5));
        assert_ne!(values, xorshift_values(4096, 8, 1.5));
        assert!(values.iter().all(|value| (-1.5..1.5).contains(value)));
        // Neighbours are uncorrelated, unlike a sinusoid.
        let lag1 = values.windows(2).map(|pair| pair[0] * pair[1]).sum::<f32>()
            / values.iter().map(|value| value * value).sum::<f32>();
        assert!(lag1.abs() < 0.1, "lag-1 autocorrelation {lag1}");
    }
}
