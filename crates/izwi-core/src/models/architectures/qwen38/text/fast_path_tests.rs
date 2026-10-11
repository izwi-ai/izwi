//! The fused decode fast paths (`fast_path.rs`) wired into the Qwen3.8 trunk,
//! exercised against the portable CPU references of the generic kernels
//! (`kernels/cuda/{norm,gdn,rope,gate}.rs`). Production resolves these paths
//! on CUDA only; `enable_fast_paths_for_tests` resolves them on the CPU.

use super::golden_tests::{
    fixture_dir, golden_reference, golden_trunk_rows, load_golden_model, relative_l2, GoldenArena,
};
use super::*;
use crate::models::architectures::qwen38::fast_path::counters_lock;
use crate::models::architectures::qwen38::native::Qwen38NativeCheckpoint;
use crate::models::architectures::qwen38::telemetry;

/// The golden fixture under the F16 plan on the CPU, with or without the
/// fast paths resolved against the CPU references.
fn golden_f16(fast_paths: bool) -> (Qwen38NativeCheckpoint, Qwen38TextModel) {
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(&fixture_dir()).unwrap();
    let mut model = load_golden_model(&checkpoint, &Device::Cpu, ProjectionMaterialization::F16);
    if fast_paths {
        model.enable_fast_paths_for_tests();
    }
    (checkpoint, model)
}

/// Every golden-sequence execution path (chunked prefill + scalar decode,
/// full prefill, ragged batched decode) through the fused model must track the
/// legacy composition to F16 rounding, and stay as close to HF as the legacy
/// F16 plan is (the CPU F16 plan itself sits a few percent from HF).
fn assert_fused_tracks_legacy(legacy: &Qwen38TextModel, fused: &Qwen38TextModel, label: &str) {
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(&fixture_dir()).unwrap();
    let (token_ids, prompt, reference) = golden_reference();
    let rows = |model: &Qwen38TextModel| {
        let arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F16);
        golden_trunk_rows(model, &arena, &token_ids, prompt)
    };
    let (legacy_rows, fused_rows) = (rows(legacy), rows(fused));
    assert_eq!(legacy_rows.len(), fused_rows.len());
    let (mut worst_drift, mut worst_error) = (0f32, 0f32);
    for (legacy, fused) in legacy_rows.iter().zip(&fused_rows) {
        assert_eq!((legacy.path, legacy.position), (fused.path, fused.position));
        let drift = relative_l2(&fused.logits, &legacy.logits);
        assert!(
            drift < 0.02,
            "{label}: fused {} logits at position {} drift from the composition by {drift}",
            fused.path,
            fused.position
        );
        let reference = &reference[fused.position];
        let (error, legacy_error) = (
            relative_l2(&fused.logits, reference),
            relative_l2(&legacy.logits, reference),
        );
        assert!(
            error < legacy_error + 0.01,
            "{label}: fused {} logits at position {} diverge from HF by {error} (legacy {legacy_error})",
            fused.path,
            fused.position
        );
        worst_drift = worst_drift.max(drift);
        worst_error = worst_error.max(error);
    }
    eprintln!(
        "qwen38 {label}: {} rows, worst drift from the composition {worst_drift:e}, worst HF error {worst_error:e}",
        fused_rows.len()
    );
}

#[test]
fn production_resolution_keeps_every_fast_path_legacy_off_cuda() {
    let (_, model) = golden_f16(false);
    let diagnostics = model.fast_path_diagnostics();
    for (name, summary) in diagnostics.as_object().unwrap() {
        assert_eq!(summary["backend"], "legacy", "{name}: {summary}");
        assert_eq!(
            summary["legacy_reasons"],
            serde_json::json!(["Qwen3.8 fast paths run on CUDA only"]),
            "{name}"
        );
    }
    assert!(!model.output_norm.fused);
}

#[test]
fn fused_norms_track_the_composition_end_to_end() {
    let _guard = counters_lock();
    let (_, legacy) = golden_f16(false);
    let (_, fused) = golden_f16(true);
    let summary = &fused.fast_path_diagnostics()["rms_norm"];
    assert_eq!(summary["backend"], "fused", "{summary}");
    assert!(fused.output_norm.fused);
    assert!(fused.layers.iter().all(|layer| {
        layer.attn_norm.fused
            && layer.post_attention_norm.fused
            && match &layer.mixer {
                Qwen38Mixer::Full(attention) => attention.q_norm.fused && attention.k_norm.fused,
                Qwen38Mixer::Linear(_) => true,
            }
    }));

    let before = telemetry::snapshot();
    assert_fused_tracks_legacy(&legacy, &legacy, "norm baseline");
    let legacy_done = telemetry::snapshot();
    assert_fused_tracks_legacy(&legacy, &fused, "fused norms");
    let fused_done = telemetry::snapshot();
    // The legacy runs never touch the fused kernels; the fused run does, for
    // both the plain norms and the residual adds folded into them.
    assert_eq!(
        legacy_done.cuda_fused_add_rms_norm_total,
        before.cuda_fused_add_rms_norm_total
    );
    assert_eq!(
        legacy_done.cuda_fused_rms_norm_total,
        before.cuda_fused_rms_norm_total
    );
    assert!(fused_done.cuda_fused_add_rms_norm_total > legacy_done.cuda_fused_add_rms_norm_total);
    assert!(fused_done.cuda_fused_rms_norm_total > legacy_done.cuda_fused_rms_norm_total);
}

#[test]
fn pending_add_materializes_exactly() {
    let _guard = counters_lock();
    let (_, model) = golden_f16(true);
    let norm = &model.layers[1].attn_norm;
    assert!(norm.fused);
    let wave = |seed: f32| {
        Tensor::from_vec(self_check_wave(2 * 64, seed, 2.0), (2, 1, 64), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
    };
    let (residual, delta) = (wave(1.0), wave(2.0));
    let pending = || Qwen38Hidden {
        residual: residual.clone(),
        pending: Some(delta.clone()),
    };
    let expected = (&residual + &delta).unwrap();
    let materialized = pending().materialize().unwrap();
    let (sum, normalized) = pending().normalized(norm).unwrap();
    // The fused add rounds the sum to F16 exactly as the Candle add does.
    assert_eq!(
        host_values(&materialized).unwrap(),
        host_values(&expected).unwrap()
    );
    assert_eq!(host_values(&sum).unwrap(), host_values(&expected).unwrap());
    let (composition, _) = norm.forward_retaining(&expected).unwrap();
    compare_values(
        "pending add + norm",
        &host_values(&normalized).unwrap(),
        &host_values(&composition).unwrap(),
        0.01,
        0.02,
    )
    .unwrap();
    // With nothing pending the residual passes through untouched.
    let (passed, _) = Qwen38Hidden::new(residual.clone())
        .normalized(norm)
        .unwrap();
    assert_eq!(
        host_values(&passed).unwrap(),
        host_values(&residual).unwrap()
    );
}

#[test]
fn norm_self_check_rejects_a_corrupted_gain() {
    let (_, model) = golden_f16(false);
    let norm = &model.layers[0].attn_norm;
    norm_self_check(norm, &Device::Cpu, DType::F16).unwrap();
    // The checkpoint's raw `w` instead of the effective `1 + w`.
    let raw = (&norm.weight - 1.0).unwrap();
    let error = norm_self_check_with_gain(norm, &raw, &Device::Cpu, DType::F16).unwrap_err();
    assert!(error.to_string().contains("diverges"), "{error}");
    // A gain 5% off on every channel is caught as well.
    let skewed = (&norm.weight * 1.05).unwrap();
    assert!(norm_self_check_with_gain(norm, &skewed, &Device::Cpu, DType::F16).is_err());
    // Under the F32 plan there is nothing to fuse.
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(&fixture_dir()).unwrap();
    let mut f32_model =
        load_golden_model(&checkpoint, &Device::Cpu, ProjectionMaterialization::F32);
    f32_model.enable_fast_paths_for_tests();
    let summary = &f32_model.fast_path_diagnostics()["rms_norm"];
    assert_eq!(summary["backend"], "legacy");
    assert!(summary["legacy_reasons"][0]
        .as_str()
        .unwrap()
        .contains("16-bit activations"));
}

#[test]
fn fused_norm_switch_forces_the_composition() {
    let _env = crate::env_test_lock();
    let (_, mut model) = golden_f16(false);
    std::env::set_var(super::super::fast_path::FUSED_NORM_ENV, "legacy");
    model.enable_fast_paths_for_tests();
    std::env::remove_var(super::super::fast_path::FUSED_NORM_ENV);
    let summary = &model.fast_path_diagnostics()["rms_norm"];
    assert_eq!(summary["backend"], "legacy");
    assert_eq!(
        summary["legacy_reasons"],
        serde_json::json!(["IZWI_QWEN38_FUSED_NORM=legacy"])
    );
    assert!(!model.output_norm.fused);
}

/// Every Qwen3.8 fast-path kill switch.
#[cfg(feature = "cuda")]
const FAST_PATH_SWITCHES: &[&str] = &[super::super::fast_path::FUSED_NORM_ENV];

/// Load with every `IZWI_QWEN38_FUSED_*` switch set to `legacy`: today's
/// composition, resolved through production loading.
#[cfg(feature = "cuda")]
fn load_with_fast_paths_legacy(load: impl FnOnce() -> Qwen38TextModel) -> Qwen38TextModel {
    let _env = crate::env_test_lock();
    for switch in FAST_PATH_SWITCHES {
        std::env::set_var(switch, "legacy");
    }
    let model = load();
    for switch in FAST_PATH_SWITCHES {
        std::env::remove_var(switch);
    }
    model
}

/// The fused norm kernels at Qwen3.8-27B widths (5120 hidden, 256 head) for
/// decode, verification-sized and prefill-sized row counts, with F32 gains.
#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_norms_match_the_composition_at_qwen38_geometry() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    for hidden in [5120usize, 256] {
        let gain = Tensor::from_vec(
            self_check_wave(hidden, 3.0, 0.3)
                .iter()
                .map(|w| 1.0 + w)
                .collect::<Vec<_>>(),
            hidden,
            &device,
        )
        .unwrap();
        let norm = Qwen38RmsNorm::new(gain, 1e-6);
        for rows in [1usize, 8, 2048] {
            let tensor = |seed: f32| {
                Tensor::from_vec(
                    self_check_wave(rows * hidden, seed, 3.0),
                    (rows, 1, hidden),
                    &device,
                )
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
            };
            let (residual, delta) = (tensor(1.0), tensor(2.0));
            let (sum, fused) = norm::add_rms_norm(&residual, &delta, &norm.weight, 1e-6).unwrap();
            let expected_sum = (&residual + &delta).unwrap();
            let (expected, _) = norm.forward_retaining(&expected_sum).unwrap();
            let label = format!("hidden {hidden} rows {rows}");
            assert_eq!(
                host_values(&sum).unwrap(),
                host_values(&expected_sum).unwrap(),
                "{label}: residual sum"
            );
            compare_values(
                &label,
                &host_values(&fused).unwrap(),
                &host_values(&expected).unwrap(),
                0.01,
                0.02,
            )
            .unwrap();
            let (plain_expected, _) = norm.forward_retaining(&residual).unwrap();
            compare_values(
                &label,
                &host_values(&norm::rms_norm(&residual, &norm.weight, 1e-6).unwrap()).unwrap(),
                &host_values(&plain_expected).unwrap(),
                0.01,
                0.02,
            )
            .unwrap();
        }
    }
}

/// Production loading on CUDA (BF16 plan) resolves the fast paths that fit
/// the golden fixture's geometry, and every golden execution path then tracks
/// both today's composition and HF.
#[cfg(feature = "cuda")]
#[test]
fn cuda_golden_trunk_with_fast_paths_tracks_the_composition_and_hf() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let _guard = counters_lock();
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(&fixture_dir()).unwrap();
    let load = || load_golden_model(&checkpoint, &device, ProjectionMaterialization::BF16);
    let legacy = load_with_fast_paths_legacy(load);
    let fused = load();
    let diagnostics = fused.fast_path_diagnostics();
    assert_eq!(diagnostics["rms_norm"]["backend"], "fused", "{diagnostics}");
    for (name, summary) in legacy.fast_path_diagnostics().as_object().unwrap() {
        assert_eq!(summary["backend"], "legacy", "{name}: {summary}");
    }
    let (token_ids, prompt, reference) = golden_reference();
    let rows = |model: &Qwen38TextModel| {
        let arena = GoldenArena::new(&checkpoint.config, &device, DType::BF16);
        golden_trunk_rows(model, &arena, &token_ids, prompt)
    };
    let (legacy_rows, fused_rows) = (rows(&legacy), rows(&fused));
    for (legacy, fused) in legacy_rows.iter().zip(&fused_rows) {
        let drift = relative_l2(&fused.logits, &legacy.logits);
        assert!(
            drift < 0.02,
            "fused {} logits at position {} drift from the composition by {drift}",
            fused.path,
            fused.position
        );
        let error = relative_l2(&fused.logits, &reference[fused.position]);
        assert!(
            error < 0.05,
            "fused {} logits at position {} diverge from HF by {error}",
            fused.path,
            fused.position
        );
    }
}
