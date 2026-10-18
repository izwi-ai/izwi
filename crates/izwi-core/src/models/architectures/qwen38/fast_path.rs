//! Bookkeeping for the Qwen3.8 fused decode fast paths.
//!
//! Each fast path resolves once at load to [`Qwen38FastPath::Fused`] or to
//! [`Qwen38FastPath::Legacy`] with a reason. A path is fused only after a
//! self-check on its own device, at the checkpoint's geometry, matches the
//! Candle composition it replaces; a mismatch keeps the composition with a
//! `self_check_failed` reason. A `IZWI_QWEN38_FUSED_*` switch forces one path
//! back to the composition without a code change, and the global
//! `cuda.mode`/`cuda.fused_decode` switches turn all of them off on CUDA.
//!
//! Qwen3.8 owns this module; Qwen3.6 keeps its own bookkeeping. The two
//! families share only the generic kernels under `kernels/`.

use candle_core::Device;

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::performance::{CudaPerformanceConfig, OptimizationMode};

/// Kill switch for the fused residual add + RMSNorm.
pub(crate) const FUSED_NORM_ENV: &str = "IZWI_QWEN38_FUSED_NORM";
/// Kill switch for the fused DeltaNet (GDN) decode.
pub(crate) const FUSED_GDN_ENV: &str = "IZWI_QWEN38_FUSED_GDN";
/// Switch for the grouped DeltaNet prefill kernel (`auto`, or `legacy`/
/// `off`/`0`/`false`): prefill expands q/k to the value heads and runs
/// today's recurrence.
pub(crate) const GDN_PREFILL_ENV: &str = "IZWI_QWEN38_GDN_PREFILL";
/// Kill switch for the fused q/k norm + partial M-RoPE.
pub(crate) const FUSED_QK_ROPE_ENV: &str = "IZWI_QWEN38_FUSED_QK_ROPE";
/// Kill switch for the K/V write from the fused q/k launch.
pub(crate) const FUSED_KV_WRITE_ENV: &str = "IZWI_QWEN38_FUSED_KV_WRITE";
/// Kill switch for the one-launch attention output gate.
pub(crate) const FUSED_ATTN_GATE_ENV: &str = "IZWI_QWEN38_FUSED_ATTN_GATE";
/// Kill switch for the DeltaNet alpha/beta projection through the router GEMV
/// (the dense `in_proj_a`/`in_proj_b` matrices then go through cuBLAS).
pub(crate) const FUSED_AB_GEMV_ENV: &str = "IZWI_QWEN38_FUSED_AB_GEMV";
/// Qwen3.8-only kill switch for CUDA graph decode (`legacy`, `off`, `0`,
/// `false`): decode runs the eager fused path. The global
/// `cuda.decode_graphs=off` (`IZWI_CUDA_DECODE_GRAPHS=off`, alias
/// `IZWI_QWEN38_DECODE_GRAPHS=off`) turns graph decode off for every family.
pub(crate) const DECODE_GRAPHS_ENV: &str = "IZWI_QWEN38_CUDA_GRAPHS";
/// Kill switch for the whole-step CUDA graph (`legacy`, `off`, `0`, `false`):
/// graph decode keeps its piecewise segments.
pub(crate) const STEP_GRAPH_ENV: &str = "IZWI_QWEN38_STEP_GRAPH";

/// Execution path a fast path resolved to at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Qwen38FastPath {
    Fused,
    Legacy { reason: String },
}

impl Default for Qwen38FastPath {
    fn default() -> Self {
        Self::legacy("not resolved")
    }
}

impl Qwen38FastPath {
    pub(crate) fn legacy(reason: impl Into<String>) -> Self {
        Self::Legacy {
            reason: reason.into(),
        }
    }

    /// The composition stays in force because the fused path disagreed with
    /// it on this device. Counted in `fast_path_self_check_failures_total`.
    pub(crate) fn self_check_failed(label: &str, error: &Error) -> Self {
        tracing::warn!(
            %error,
            path = label,
            "Qwen3.8 fused {label} self-check failed; keeping the Candle composition"
        );
        super::telemetry::record_fast_path_self_check_failure();
        Self::legacy(format!("self_check_failed: {error}"))
    }

    pub(crate) fn is_fused(&self) -> bool {
        matches!(self, Self::Fused)
    }
}

/// The global CUDA performance switches as they apply to the Qwen3.8 fast
/// paths. They bind on CUDA only; the `IZWI_QWEN38_FUSED_*` switches bind on
/// every device. `cuda.mode=off` turns everything off,
/// `cuda.fused_decode=off` the fused decode kernels (and with them graph
/// decode, which needs them), and `cuda.decode_graphs=off` graph decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Qwen38CudaSwitches {
    pub(crate) fused_decode_off: Option<&'static str>,
    pub(crate) decode_graphs_off: Option<&'static str>,
}

impl Qwen38CudaSwitches {
    pub(crate) fn from_performance(
        backend: BackendKind,
        performance: &CudaPerformanceConfig,
    ) -> Self {
        if backend != BackendKind::Cuda {
            return Self::default();
        }
        let off = |mode: OptimizationMode, reason: &'static str| {
            if !performance.enabled() {
                Some("cuda.mode=off")
            } else {
                (!mode.enabled()).then_some(reason)
            }
        };
        let fused_decode_off = off(performance.fused_decode, "cuda.fused_decode=off");
        Self {
            fused_decode_off,
            decode_graphs_off: off(performance.decode_graphs, "cuda.decode_graphs=off")
                .or(fused_decode_off),
        }
    }

    /// The switches for a model loaded on `device`.
    pub(crate) fn for_device(device: &Device, performance: &CudaPerformanceConfig) -> Self {
        let backend = if device.is_cuda() {
            BackendKind::Cuda
        } else if device.is_metal() {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        };
        Self::from_performance(backend, performance)
    }
}

/// The reason a fused decode path cannot run, before any kernel-specific
/// check: its kill switch, the global CUDA switches, or the device. The fast
/// paths run on CUDA in production; `allow_cpu` lets tests resolve them
/// against the portable CPU references in `kernels/cuda/*`.
pub(crate) fn decode_gate(
    env: &str,
    switches: &Qwen38CudaSwitches,
    device: &Device,
    allow_cpu: bool,
) -> Option<Qwen38FastPath> {
    if legacy_requested(env) {
        Some(Qwen38FastPath::legacy(format!("{env}=legacy")))
    } else if let Some(reason) = switches.fused_decode_off {
        Some(Qwen38FastPath::legacy(reason))
    } else if !(device.is_cuda() || (allow_cpu && device.is_cpu())) {
        Some(Qwen38FastPath::legacy(
            "Qwen3.8 fast paths run on CUDA only",
        ))
    } else {
        None
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
    paths: impl IntoIterator<Item = &'a Qwen38FastPath>,
) -> serde_json::Value {
    let mut fused = 0usize;
    let mut legacy = 0usize;
    let mut reasons: Vec<&str> = Vec::new();
    for path in paths {
        match path {
            Qwen38FastPath::Fused => fused += 1,
            Qwen38FastPath::Legacy { reason } => {
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
/// error at most `rel_l2`, and every element within `max_frac` of the largest
/// reference magnitude. A non-finite value on either side fails: a NaN
/// difference saturates to infinity instead of reading as agreement.
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
        let diff = (a - b).abs();
        let diff = if diff.is_finite() {
            diff
        } else {
            f32::INFINITY
        };
        err += f64::from(diff).powi(2);
        norm += f64::from(*b).powi(2);
        max_ref = max_ref.max(b.abs());
        max_err = max_err.max(diff);
    }
    // Every term is finite or saturated to infinity, so neither bound sees a NaN.
    let observed = (err / norm.max(f64::MIN_POSITIVE)).sqrt();
    if observed > rel_l2 || max_err > max_frac * max_ref.max(f32::MIN_POSITIVE) {
        return Err(Error::InferenceError(format!(
            "{label}: fused output diverges from the reference: relative L2 {observed:.4}, max error {max_err} vs max |ref| {max_ref}"
        )));
    }
    Ok(())
}

/// Serializes the tests that execute fused paths or read the process-wide
/// fast-path counters, so a counter delta belongs to the test observing it.
/// Every test that runs a model with fast paths resolved holds it, before
/// `env_test_lock`.
#[cfg(test)]
pub(crate) fn counters_lock() -> std::sync::MutexGuard<'static, ()> {
    static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    COUNTERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_switches_bind_on_cuda_only() {
        let mut performance = CudaPerformanceConfig::default();
        for backend in [BackendKind::Cuda, BackendKind::Metal, BackendKind::Cpu] {
            assert_eq!(
                Qwen38CudaSwitches::from_performance(backend, &performance),
                Qwen38CudaSwitches::default()
            );
        }
        performance.fused_decode = OptimizationMode::Off;
        let switches = Qwen38CudaSwitches::from_performance(BackendKind::Cuda, &performance);
        assert_eq!(switches.fused_decode_off, Some("cuda.fused_decode=off"));
        // Graph decode needs the fused decode kernels.
        assert_eq!(switches.decode_graphs_off, Some("cuda.fused_decode=off"));
        let graphs_off = CudaPerformanceConfig {
            decode_graphs: OptimizationMode::Off,
            ..CudaPerformanceConfig::default()
        };
        let switches = Qwen38CudaSwitches::from_performance(BackendKind::Cuda, &graphs_off);
        assert_eq!(switches.decode_graphs_off, Some("cuda.decode_graphs=off"));
        assert_eq!(switches.fused_decode_off, None);
        assert_eq!(
            Qwen38CudaSwitches::from_performance(BackendKind::Metal, &graphs_off),
            Qwen38CudaSwitches::default()
        );
        assert_eq!(
            Qwen38CudaSwitches::from_performance(BackendKind::Metal, &performance),
            Qwen38CudaSwitches::default()
        );
        assert_eq!(
            Qwen38CudaSwitches::for_device(&Device::Cpu, &performance),
            Qwen38CudaSwitches::default()
        );
        let performance = CudaPerformanceConfig {
            mode: OptimizationMode::Off,
            ..CudaPerformanceConfig::default()
        };
        let switches = Qwen38CudaSwitches::from_performance(BackendKind::Cuda, &performance);
        assert_eq!(switches.fused_decode_off, Some("cuda.mode=off"));
        assert_eq!(switches.decode_graphs_off, Some("cuda.mode=off"));
    }

    #[test]
    fn decode_gate_orders_switch_global_and_device_reasons() {
        let off = Qwen38CudaSwitches {
            fused_decode_off: Some("cuda.fused_decode=off"),
            ..Qwen38CudaSwitches::default()
        };
        let unset = "IZWI_QWEN38_FAST_PATH_GATE_TEST_UNSET";
        assert_eq!(
            decode_gate(unset, &off, &Device::Cpu, true),
            Some(Qwen38FastPath::legacy("cuda.fused_decode=off"))
        );
        let on = Qwen38CudaSwitches::default();
        assert_eq!(
            decode_gate(unset, &on, &Device::Cpu, false),
            Some(Qwen38FastPath::legacy(
                "Qwen3.8 fast paths run on CUDA only"
            ))
        );
        assert_eq!(decode_gate(unset, &on, &Device::Cpu, true), None);
    }

    #[test]
    fn summary_reports_mixed_paths_with_distinct_reasons() {
        let paths = [
            Qwen38FastPath::Fused,
            Qwen38FastPath::legacy("a"),
            Qwen38FastPath::legacy("a"),
            Qwen38FastPath::legacy("b"),
        ];
        let summary = summarize(&paths);
        assert_eq!(summary["backend"], "mixed");
        assert_eq!(summary["fused_layers"], 1);
        assert_eq!(summary["legacy_layers"], 3);
        assert_eq!(summary["legacy_reasons"], serde_json::json!(["a", "b"]));
        assert_eq!(summarize(&[])["backend"], "none");
        assert_eq!(summarize(&[Qwen38FastPath::Fused])["backend"], "fused");
    }

    #[test]
    fn compare_rejects_wrong_or_non_finite_outputs_and_accepts_rounding_noise() {
        let reference = [1.0f32, -2.0, 0.5, 4.0];
        assert!(compare_values("t", &[1.004, -2.01, 0.498, 4.02], &reference, 0.03, 0.08).is_ok());
        assert!(compare_values("t", &[-1.0, 2.0, 0.5, 4.0], &reference, 0.03, 0.08).is_err());
        assert!(compare_values("t", &[1.0], &reference, 0.03, 0.08).is_err());
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in 0..reference.len() {
                let mut values = reference;
                values[index] = poison;
                assert!(compare_values("t", &values, &reference, 0.03, 0.08).is_err());
                // A poisoned reference must not pass either.
                assert!(compare_values("t", &reference, &values, 0.03, 0.08).is_err());
            }
        }
    }
}
