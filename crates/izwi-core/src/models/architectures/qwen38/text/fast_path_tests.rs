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
fn fused_fast_paths_track_the_composition_end_to_end() {
    let _guard = counters_lock();
    let (_, legacy) = golden_f16(false);
    let (_, mut fused) = golden_f16(true);
    let diagnostics = fused.fast_path_diagnostics();
    for name in [
        "rms_norm",
        "qk_rope",
        "kv_write",
        "attn_gate",
        "gdn_alpha_beta_gemv",
    ] {
        assert_eq!(
            diagnostics[name]["backend"], "fused",
            "{name}: {diagnostics}"
        );
    }
    // The fixture's 32-wide DeltaNet heads do not fit the 128-wide kernel.
    assert_eq!(diagnostics["gdn_decode"]["backend"], "legacy");
    // The CPU's dense F16 matmul accumulates in F16, so the router GEMV (F32
    // accumulation, one rounding) cannot track it to the composition's F16
    // tolerance over a chaotic decode; its own test checks it against an F32
    // reference. Compare the other fast paths here.
    for layer in &mut fused.layers {
        if let Qwen38Mixer::Linear(linear) = &mut layer.mixer {
            assert!(linear.alpha_beta_rows.take().is_some());
        }
    }
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
    assert_fused_tracks_legacy(&legacy, &legacy, "composition baseline");
    let legacy_done = telemetry::snapshot();
    assert_fused_tracks_legacy(&legacy, &fused, "fused fast paths");
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
    assert_eq!(
        legacy_done.cuda_fused_qk_norm_rope_total,
        before.cuda_fused_qk_norm_rope_total
    );
    assert_eq!(
        legacy_done.cuda_fused_attn_gate_total,
        before.cuda_fused_attn_gate_total
    );
    assert!(fused_done.cuda_fused_qk_norm_rope_total > legacy_done.cuda_fused_qk_norm_rope_total);
    assert!(fused_done.cuda_fused_attn_gate_total > legacy_done.cuda_fused_attn_gate_total);
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
const FAST_PATH_SWITCHES: &[&str] = &[
    super::super::fast_path::FUSED_NORM_ENV,
    super::super::fast_path::FUSED_GDN_ENV,
    super::super::fast_path::FUSED_QK_ROPE_ENV,
    super::super::fast_path::FUSED_KV_WRITE_ENV,
    super::super::fast_path::FUSED_ATTN_GATE_ENV,
    super::super::fast_path::FUSED_AB_GEMV_ENV,
];

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
    for name in [
        "rms_norm",
        "qk_rope",
        "kv_write",
        "attn_gate",
        "gdn_alpha_beta_gemv",
    ] {
        assert_eq!(
            diagnostics[name]["backend"], "fused",
            "{name}: {diagnostics}"
        );
    }
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

/// A DeltaNet mixer at the fused kernel's 128-dim heads with deterministic
/// weights: `key_heads`/`value_heads` heads over a `hidden`-wide residual,
/// projections in `dtype`. Today's CUDA decode kernels run where they would
/// in production (`fused_decode` on CUDA).
fn gdn_mixer(
    key_heads: usize,
    value_heads: usize,
    hidden: usize,
    device: &Device,
    dtype: DType,
) -> Qwen38LinearAttention {
    let dim = gdn::HEAD_DIM;
    let conv_dim = (2 * key_heads + value_heads) * dim;
    let value_width = value_heads * dim;
    let scale = 1.0 / (hidden as f32).sqrt();
    let dense = |out: usize, input: usize, seed: f32, scale: f32| {
        Qwen38Projection::Dense(Linear::new(
            Tensor::from_vec(
                self_check_wave(out * input, seed, scale),
                (out, input),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .to_device(device)
            .unwrap(),
            None,
        ))
    };
    let f32_tensor = |values: Vec<f32>, shape: &[usize]| {
        Tensor::from_vec(values, shape, &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap()
    };
    let conv_kernel = f32_tensor(self_check_wave(conv_dim * 4, 5.0, 0.5), &[conv_dim, 4]);
    Qwen38LinearAttention {
        fused_decode: device.is_cuda(),
        qkv_z_proj: Qwen38ProjectionGroup::Separate(vec![
            dense(conv_dim, hidden, 1.0, 2.0 * scale),
            dense(value_width, hidden, 2.0, 2.0 * scale),
        ]),
        alpha_beta_proj: Qwen38ProjectionGroup::Separate(vec![
            dense(value_heads, hidden, 3.0, 2.0 * scale),
            dense(value_heads, hidden, 4.0, 2.0 * scale),
        ]),
        dt_bias: f32_tensor(self_check_wave(value_heads, 6.0, 1.0), &[1, 1, value_heads]),
        a: f32_tensor(
            self_check_wave(value_heads, 7.0, 1.0)
                .iter()
                .map(|v| -v.exp())
                .collect(),
            &[1, 1, value_heads],
        ),
        conv_kernel_slices: pre_slice_conv_kernel(&conv_kernel, 4).unwrap(),
        conv_kernel,
        norm: Qwen38GatedRmsNorm {
            fused_decode: device.is_cuda(),
            weight: f32_tensor(
                self_check_wave(dim, 8.0, 0.2)
                    .iter()
                    .map(|v| 1.0 + v)
                    .collect(),
                &[dim],
            ),
            eps: 1e-6,
        },
        out_proj: dense(
            hidden,
            value_width,
            9.0,
            (value_width as f32).sqrt().recip(),
        ),
        num_k_heads: key_heads,
        num_v_heads: value_heads,
        head_k_dim: dim,
        head_v_dim: dim,
        conv_dim,
        kernel_size: 4,
        tiled_recurrence_enabled: false,
        tiled_recurrence_tile_size_override: None,
        fused_gdn: None,
        fused_gdn_path: Qwen38FastPath::default(),
        alpha_beta_rows: None,
        alpha_beta_gemv_path: Qwen38FastPath::default(),
        gdn_prefill: false,
        gdn_prefill_path: Qwen38FastPath::default(),
    }
}

/// `rows` decode inputs `[1, 1, hidden]` in `dtype`.
fn gdn_inputs(rows: usize, hidden: usize, seed: f32, device: &Device, dtype: DType) -> Vec<Tensor> {
    (0..rows)
        .map(|row| {
            Tensor::from_vec(
                self_check_wave(hidden, seed + 17.0 * row as f32, 1.0),
                (1, 1, hidden),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .to_device(device)
            .unwrap()
        })
        .collect()
}

/// A zero linear-layer state, as `ensure_state_initialized` builds it.
fn initialized_state(mixer: &Qwen38LinearAttention, device: &Device) -> Qwen38LayerRuntimeState {
    Qwen38LayerRuntimeState::Linear {
        conv_state: Some(ConvRingState::from_slots(
            (0..mixer.kernel_size - 1)
                .map(|_| owned_zero_tensor(&[mixer.conv_dim, 1], DType::F32, device).unwrap())
                .collect(),
        )),
        recurrent_state: Some(
            owned_zero_tensor(
                &[1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim],
                DType::F32,
                device,
            )
            .unwrap(),
        ),
    }
}

/// The logical conv history and recurrent state of a linear layer state.
fn linear_state_values(state: &Qwen38LayerRuntimeState) -> (Vec<f32>, Vec<f32>) {
    let Qwen38LayerRuntimeState::Linear {
        conv_state: Some(conv),
        recurrent_state: Some(recurrent),
    } = state
    else {
        panic!("linear state is not initialized")
    };
    (
        host_values(&conv.contiguous_history().unwrap()).unwrap(),
        host_values(recurrent).unwrap(),
    )
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], rel_l2: f64, max_frac: f32) {
    compare_values(label, actual, expected, rel_l2, max_frac).unwrap();
}

#[test]
fn gdn_reference_matches_q38_portable_chain_at_trunk_geometry() {
    let mixer = gdn_mixer(16, 48, 64, &Device::Cpu, DType::F16);
    let spec = mixer
        .gdn_spec(&Device::Cpu)
        .expect("trunk geometry fits the kernel");
    assert_eq!(spec.conv_dim(), 10240);
    assert!(spec.grouped);
    mixer
        .fused_gdn_self_check(&spec, &Device::Cpu, DType::F16)
        .unwrap();
    let mut mixer = mixer;
    mixer.resolve_fused_gdn(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        DType::F16,
        true,
    );
    assert_eq!(mixer.fused_gdn_path, Qwen38FastPath::Fused);
    assert_eq!(mixer.fused_gdn, Some(spec));
}

#[test]
fn tiled_order_fails_q38_contract() {
    let mixer = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    let grouped = mixer.gdn_spec(&Device::Cpu).unwrap();
    let tiled = GdnDecodeSpec {
        grouped: false,
        ..grouped
    };
    let error = mixer
        .fused_gdn_self_check(&tiled, &Device::Cpu, DType::F16)
        .unwrap_err();
    assert!(error.to_string().contains("diverges"), "{error}");
}

#[test]
fn swapped_alpha_beta_fails_self_check() {
    let mixer = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    let spec = mixer.gdn_spec(&Device::Cpu).unwrap();
    // `in_proj_a` handed over as beta and `in_proj_b` as alpha.
    let error = mixer
        .fused_gdn_self_check_with(&Device::Cpu, DType::F16, |row, conv, recurrent| {
            mixer.fused_decode_row(
                &spec,
                &row.mixed_qkv,
                &row.z,
                &row.alpha,
                &row.beta,
                conv,
                recurrent,
            )
        })
        .unwrap_err();
    assert!(error.to_string().contains("diverges"), "{error}");
}

#[test]
fn conv_f16_oracle_matches_f32() {
    let mixer = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    let c = mixer.conv_dim;
    let x = Tensor::from_vec(self_check_wave(c, 1.0, 2.0), (1, 1, c), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let history = Tensor::from_vec(self_check_wave(3 * c, 2.0, 1.0), (c, 3), &Device::Cpu).unwrap();
    let (out16, next16) = qwen38_conv_decode(&x, &mixer.conv_kernel, &history).unwrap();
    let x32 = x.to_dtype(DType::F32).unwrap();
    let (out32, next32) = qwen38_conv_decode(&x32, &mixer.conv_kernel, &history).unwrap();
    assert_eq!(host_values(&out16).unwrap(), host_values(&out32).unwrap());
    assert_eq!(host_values(&next16).unwrap(), host_values(&next32).unwrap());
    assert_eq!(next16.dims(), [c, 3]);
    // Channel-major next history, oldest first: [h1, h2, x].
    let next = next16.to_vec2::<f32>().unwrap();
    let old = history.to_vec2::<f32>().unwrap();
    let x = host_values(&x32).unwrap();
    for channel in [0, 1, c / 2, c - 1] {
        assert_eq!(
            next[channel],
            [old[channel][1], old[channel][2], x[channel]]
        );
    }
    // Today's portable step agrees with the oracle.
    let mut ring = Some(ConvRingState::from_staged(history.clone()).unwrap());
    let legacy = mixer.depthwise_conv_step(&x32, &mut ring).unwrap();
    assert_close(
        "conv output",
        &host_values(&out32).unwrap(),
        &host_values(&legacy).unwrap(),
        1e-6,
        1e-5,
    );
    assert_eq!(
        host_values(&ring.unwrap().contiguous_history().unwrap()).unwrap(),
        host_values(&next32).unwrap()
    );
}

/// The DeltaNet alpha/beta projection through the router GEMV: the F32 rows
/// pass their probe, decode takes them, and alpha and beta match the dense
/// pair to 16-bit rounding. Rows in the wrong order fail the probe (RED), and
/// F32 activations keep the dense projection.
#[test]
fn alpha_beta_gemv_tracks_the_dense_projection() {
    // The CPU has no BF16 matmul for the dense reference; CUDA covers BF16.
    let dtype = DType::F16;
    let dense = gdn_mixer(2, 6, 64, &Device::Cpu, dtype);
    let mut gemv = gdn_mixer(2, 6, 64, &Device::Cpu, dtype);
    gemv.resolve_alpha_beta_gemv(&Qwen38CudaSwitches::default(), &Device::Cpu, dtype, true);
    assert!(
        gemv.alpha_beta_gemv_path.is_fused(),
        "{dtype:?}: {:?}",
        gemv.alpha_beta_gemv_path
    );
    let rows = gemv.alpha_beta_rows.clone().expect("GEMV rows");
    assert_eq!(rows.dims(), [12, 64]);
    assert_eq!(rows.dtype(), DType::F32);
    for input in gdn_inputs(3, 64, 7.0, &Device::Cpu, dtype) {
        let (_, _, alpha, beta) = gemv.decode_projections(&input).unwrap();
        let (_, _, dense_alpha, dense_beta) = dense.decode_projections(&input).unwrap();
        assert_eq!(alpha.dims(), dense_alpha.dims());
        assert_eq!((alpha.dtype(), beta.dtype()), (dtype, dtype));
        let concat = |a: &Tensor, b: &Tensor| {
            let mut values = host_values(a).unwrap();
            values.extend(host_values(b).unwrap());
            values
        };
        let (actual, dense) = (concat(&alpha, &beta), concat(&dense_alpha, &dense_beta));
        assert_close("alpha and beta", &actual, &dense, 0.01, 0.02);
        // Against the exact F32 product, the GEMV is one rounding of the
        // activation dtype away (half an ulp), and never further than
        // the dense projection.
        let exact = host_values(
            &input
                .to_dtype(DType::F32)
                .unwrap()
                .reshape((1, 64))
                .unwrap()
                .matmul(&rows.t().unwrap())
                .unwrap(),
        )
        .unwrap();
        // A non-finite difference saturates instead of reading as zero.
        let worst = |values: &[f32]| {
            values
                .iter()
                .zip(&exact)
                .map(|(value, exact)| {
                    let error = (value - exact).abs() / exact.abs().max(1.0);
                    if error.is_finite() {
                        error
                    } else {
                        f32::INFINITY
                    }
                })
                .fold(0f32, f32::max)
        };
        // Half an F16 ulp, relative.
        let half_ulp = 2f32.powi(-11);
        assert!(worst(&actual) <= half_ulp, "{}", worst(&actual));
        assert!(worst(&actual) <= worst(&dense));
    }
    let swapped = Tensor::cat(
        &[rows.narrow(0, 6, 6).unwrap(), rows.narrow(0, 0, 6).unwrap()],
        0,
    )
    .unwrap();
    assert!(
        gemv.alpha_beta_gemv_self_check(&swapped, &Device::Cpu, dtype)
            .is_err(),
        "{dtype:?}: alpha and beta rows swapped must fail the probe"
    );
    let mut wide = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F32);
    wide.resolve_alpha_beta_gemv(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        DType::F32,
        true,
    );
    assert!(wide.alpha_beta_rows.is_none());
    assert_eq!(
        wide.alpha_beta_gemv_path,
        Qwen38FastPath::legacy("the alpha/beta GEMV needs 16-bit activations")
    );
}

/// Prefill through the unchanged sequence path, then decode: every fused
/// step tracks today's decode, and the fused counters move only for it.
#[test]
fn fused_gdn_decode_tracks_legacy_end_to_end() {
    let _guard = counters_lock();
    let legacy = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    let mut fused = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    fused.resolve_fused_gdn(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        DType::F16,
        true,
    );
    assert!(
        fused.fused_gdn_path.is_fused(),
        "{:?}",
        fused.fused_gdn_path
    );
    let prompt = Tensor::cat(&gdn_inputs(5, 64, 1.0, &Device::Cpu, DType::F16), 1).unwrap();
    let steps = gdn_inputs(6, 64, 50.0, &Device::Cpu, DType::F16);
    let run = |mixer: &Qwen38LinearAttention| {
        let mut state = initialized_state(mixer, &Device::Cpu);
        mixer.forward_sequence(&prompt, &mut state).unwrap();
        let outputs = steps
            .iter()
            .map(|input| host_values(&mixer.forward(input, &mut state).unwrap()).unwrap())
            .collect::<Vec<_>>();
        (outputs, linear_state_values(&state))
    };
    let before = telemetry::snapshot();
    let (expected, expected_state) = run(&legacy);
    let between = telemetry::snapshot();
    let (actual, actual_state) = run(&fused);
    let after = telemetry::snapshot();
    for (step, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_close(&format!("decode step {step}"), actual, expected, 0.01, 0.03);
    }
    assert_close(
        "conv history",
        &actual_state.0,
        &expected_state.0,
        1e-6,
        1e-6,
    );
    assert_close(
        "recurrent state",
        &actual_state.1,
        &expected_state.1,
        1e-3,
        1e-2,
    );
    assert_eq!(
        between.cuda_fused_gdn_decode_attempts_total,
        before.cuda_fused_gdn_decode_attempts_total
    );
    assert_eq!(
        after.cuda_fused_gdn_decode_success_total - between.cuda_fused_gdn_decode_success_total,
        steps.len() as u64
    );
    assert_eq!(
        after.cuda_fused_gdn_decode_fallback_total,
        between.cuda_fused_gdn_decode_fallback_total
    );
}

/// The ragged batch takes the fused rows (two launches each) and matches
/// each row decoded alone.
#[test]
fn fused_gdn_batch_matches_scalar_rows() {
    let _guard = counters_lock();
    let mut mixer = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    mixer.resolve_fused_gdn(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        DType::F16,
        true,
    );
    assert!(mixer.fused_gdn_path.is_fused());
    // Three rows with different histories: prompts of 1, 3 and 4 tokens.
    let prefilled = |row: usize| {
        let mut state = initialized_state(&mixer, &Device::Cpu);
        let prompt = Tensor::cat(
            &gdn_inputs(
                1 + row + row / 2,
                64,
                3.0 * row as f32,
                &Device::Cpu,
                DType::F16,
            ),
            1,
        )
        .unwrap();
        mixer.forward_sequence(&prompt, &mut state).unwrap();
        state
    };
    let mut scalar_states = (0..3).map(prefilled).collect::<Vec<_>>();
    let mut batch_states = (0..3).map(prefilled).collect::<Vec<_>>();
    for step in 0..3 {
        let inputs = gdn_inputs(3, 64, 90.0 + 7.0 * step as f32, &Device::Cpu, DType::F16);
        let scalar = inputs
            .iter()
            .zip(scalar_states.iter_mut())
            .map(|(input, state)| host_values(&mixer.forward(input, state).unwrap()).unwrap())
            .collect::<Vec<_>>();
        let batch_input = Tensor::cat(&inputs, 0).unwrap();
        let mut refs = batch_states.iter_mut().collect::<Vec<_>>();
        let batched = mixer.forward_decode_batch(&batch_input, &mut refs).unwrap();
        for (row, scalar) in scalar.iter().enumerate() {
            let batched = host_values(&batched.i(row).unwrap()).unwrap();
            assert_close(
                &format!("step {step} row {row}"),
                &batched,
                scalar,
                0.01,
                0.03,
            );
        }
    }
    for (scalar, batch) in scalar_states.iter().zip(&batch_states) {
        let (scalar, batch) = (linear_state_values(scalar), linear_state_values(batch));
        // The history holds the projection rows, which a 3-row F16 matmul
        // rounds differently from a 1-row one.
        assert_close("conv history", &batch.0, &scalar.0, 0.005, 0.01);
        assert_close("recurrent state", &batch.1, &scalar.1, 0.01, 0.01);
    }
}

/// MTP verification runs the unchanged sequence path; after a partial reject
/// the recovered portable ring feeds the next fused step exactly as it feeds
/// today's decode.
#[test]
fn fused_gdn_continues_after_verify_and_partial_recover() {
    let _guard = counters_lock();
    let legacy = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    let mut fused = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    fused.resolve_fused_gdn(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        DType::F16,
        true,
    );
    assert!(fused.fused_gdn_path.is_fused());
    let mut state = initialized_state(&fused, &Device::Cpu);
    let prompt = Tensor::cat(&gdn_inputs(4, 64, 1.0, &Device::Cpu, DType::F16), 1).unwrap();
    fused.forward_sequence(&prompt, &mut state).unwrap();
    // A fused decode step leaves the contiguous kernel history.
    fused
        .forward(
            &gdn_inputs(1, 64, 20.0, &Device::Cpu, DType::F16)[0],
            &mut state,
        )
        .unwrap();
    let draft = Tensor::cat(&gdn_inputs(3, 64, 30.0, &Device::Cpu, DType::F16), 1).unwrap();
    let mut verification = None;
    fused
        .forward_sequence_recording(&draft, &mut state, Some(&mut verification))
        .unwrap();
    let recovered = verification.unwrap().recover(1).unwrap();
    let next = &gdn_inputs(1, 64, 40.0, &Device::Cpu, DType::F16)[0];
    let (mut fused_state, mut legacy_state) = (recovered.clone(), recovered);
    let actual = fused.forward(next, &mut fused_state).unwrap();
    let expected = legacy.forward(next, &mut legacy_state).unwrap();
    assert_close(
        "decode after recover",
        &host_values(&actual).unwrap(),
        &host_values(&expected).unwrap(),
        0.01,
        0.03,
    );
    let (fused_values, legacy_values) = (
        linear_state_values(&fused_state),
        linear_state_values(&legacy_state),
    );
    assert_close(
        "conv history",
        &fused_values.0,
        &legacy_values.0,
        1e-6,
        1e-6,
    );
    assert_close(
        "recurrent state",
        &fused_values.1,
        &legacy_values.1,
        1e-4,
        1e-3,
    );
}

/// Staging keeps the canonical `[C, 3]` history and `[1, 48, 128, 128]`-style
/// state; restoring either the canonical or the legacy `[3, C, 1]` form
/// continues the fused decode bit-for-bit.
#[test]
fn fused_gdn_continues_bit_exact_after_stage_and_restore() {
    let _guard = counters_lock();
    let mut mixer = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F16);
    mixer.resolve_fused_gdn(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        DType::F16,
        true,
    );
    assert!(mixer.fused_gdn_path.is_fused());
    let mut state = initialized_state(&mixer, &Device::Cpu);
    for input in gdn_inputs(3, 64, 5.0, &Device::Cpu, DType::F16) {
        mixer.forward(&input, &mut state).unwrap();
    }
    let Qwen38LayerRuntimeState::Linear {
        conv_state: Some(conv),
        recurrent_state: Some(recurrent),
    } = &state
    else {
        panic!("decode initialized the state")
    };
    let staged = conv.staged_tensor().unwrap();
    assert_eq!(staged.dims(), [mixer.conv_dim, 3]);
    assert_eq!(recurrent.dims(), [1, 6, 128, 128]);
    let legacy_form = staged.t().unwrap().unsqueeze(2).unwrap();
    let next = &gdn_inputs(1, 64, 60.0, &Device::Cpu, DType::F16)[0];
    let continue_from = |history: &Tensor| {
        let mut restored = Qwen38LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState::from_staged(history.clone()).unwrap()),
            recurrent_state: Some(recurrent.clone()),
        };
        let output = host_values(&mixer.forward(next, &mut restored).unwrap()).unwrap();
        (output, linear_state_values(&restored))
    };
    let mut live = state.clone();
    let expected = (
        host_values(&mixer.forward(next, &mut live).unwrap()).unwrap(),
        linear_state_values(&live),
    );
    assert_eq!(continue_from(&staged), expected);
    assert_eq!(continue_from(&legacy_form), expected);
}

/// The CUDA conv entries for BF16/F16 projections compute exactly what the
/// F32 entry computes on the cast input.
#[cfg(feature = "cuda")]
#[test]
fn cuda_conv_decode_16_bit_entries_match_f32() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let c = 10240;
    let weight = Tensor::from_vec(self_check_wave(c * 4, 1.0, 0.5), (c, 4), &device).unwrap();
    let history = Tensor::from_vec(self_check_wave(c * 3, 2.0, 1.0), (c, 3), &device).unwrap();
    for dtype in [DType::BF16, DType::F16] {
        let x = Tensor::from_vec(self_check_wave(c, 3.0, 2.0), (1, 1, c), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let (out16, next16) =
            crate::kernels::cuda::try_qwen38_causal_conv_decode(&x, &weight, &history)
                .expect("16-bit conv entry");
        let (out32, next32) = crate::kernels::cuda::try_qwen38_causal_conv_decode(
            &x.to_dtype(DType::F32).unwrap(),
            &weight,
            &history,
        )
        .expect("F32 conv entry");
        assert_eq!(out16.dtype(), DType::F32);
        assert_eq!(
            host_values(&out16).unwrap(),
            host_values(&out32).unwrap(),
            "{dtype:?}"
        );
        assert_eq!(
            host_values(&next16).unwrap(),
            host_values(&next32).unwrap(),
            "{dtype:?}"
        );
    }
}

/// Production resolution on CUDA (no CPU allowance) at the 27B linear-layer
/// geometry turns the fused decode on, and decode steps through it track
/// today's specialized q38 kernels, scalar and batched.
#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_gdn_matches_legacy_q38_kernels_48_heads() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let _guard = counters_lock();
    let legacy = gdn_mixer(16, 48, 256, &device, DType::BF16);
    let mut fused = gdn_mixer(16, 48, 256, &device, DType::BF16);
    fused.resolve_fused_gdn(&Qwen38CudaSwitches::default(), &device, DType::BF16, false);
    assert_eq!(fused.fused_gdn_path, Qwen38FastPath::Fused);
    let steps = gdn_inputs(4, 256, 11.0, &device, DType::BF16);
    let (mut legacy_state, mut fused_state) = (
        initialized_state(&legacy, &device),
        initialized_state(&fused, &device),
    );
    for (step, input) in steps.iter().enumerate() {
        let expected = legacy.forward(input, &mut legacy_state).unwrap();
        let actual = fused.forward(input, &mut fused_state).unwrap();
        assert_close(
            &format!("CUDA decode step {step}"),
            &host_values(&actual).unwrap(),
            &host_values(&expected).unwrap(),
            0.01,
            0.03,
        );
    }
    let (fused_values, legacy_values) = (
        linear_state_values(&fused_state),
        linear_state_values(&legacy_state),
    );
    assert_close(
        "CUDA conv history",
        &fused_values.0,
        &legacy_values.0,
        1e-6,
        1e-6,
    );
    assert_close(
        "CUDA recurrent state",
        &fused_values.1,
        &legacy_values.1,
        1e-3,
        1e-2,
    );

    let rows = gdn_inputs(2, 256, 70.0, &device, DType::BF16);
    let mut legacy_rows = [legacy_state.clone(), legacy_state];
    let mut fused_rows = [fused_state.clone(), fused_state];
    let batch_input = Tensor::cat(&rows, 0).unwrap();
    let expected = legacy
        .forward_decode_batch(
            &batch_input,
            &mut legacy_rows.iter_mut().collect::<Vec<_>>(),
        )
        .unwrap();
    let actual = fused
        .forward_decode_batch(&batch_input, &mut fused_rows.iter_mut().collect::<Vec<_>>())
        .unwrap();
    assert_close(
        "CUDA batched decode",
        &host_values(&actual).unwrap(),
        &host_values(&expected).unwrap(),
        0.01,
        0.03,
    );
}

/// `[1, rows, hidden]` prefill input in `dtype`.
fn gdn_prefill_input(
    rows: usize,
    hidden: usize,
    seed: f32,
    device: &Device,
    dtype: DType,
) -> Tensor {
    Tensor::from_vec(
        self_check_wave(rows * hidden, seed, 1.0),
        (1, rows, hidden),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(dtype)
    .unwrap()
    .to_device(device)
    .unwrap()
}

/// Two consecutive prefill spans (the second from the first's non-zero
/// state) through `legacy` and `grouped`: outputs, conv histories and
/// recurrent states must agree.
fn assert_grouped_prefill_tracks_legacy(
    legacy: &Qwen38LinearAttention,
    grouped: &Qwen38LinearAttention,
    hidden: usize,
    device: &Device,
    dtype: DType,
    tolerance: (f64, f32),
) {
    let (mut legacy_state, mut grouped_state) = (
        initialized_state(legacy, device),
        initialized_state(grouped, device),
    );
    for (span, rows) in [(0, 20usize), (1, 7)] {
        let input = gdn_prefill_input(rows, hidden, 3.0 + span as f32 * 41.0, device, dtype);
        let expected = legacy.forward_sequence(&input, &mut legacy_state).unwrap();
        let actual = grouped
            .forward_sequence(&input, &mut grouped_state)
            .unwrap();
        assert_close(
            &format!("prefill span {span}"),
            &host_values(&actual).unwrap(),
            &host_values(&expected).unwrap(),
            tolerance.0,
            tolerance.1,
        );
    }
    let (grouped_values, legacy_values) = (
        linear_state_values(&grouped_state),
        linear_state_values(&legacy_state),
    );
    assert_close(
        "conv history",
        &grouped_values.0,
        &legacy_values.0,
        1e-6,
        1e-6,
    );
    assert_close(
        "recurrent state",
        &grouped_values.1,
        &legacy_values.1,
        tolerance.0,
        tolerance.1,
    );
}

/// The grouped DeltaNet prefill at the 27B trunk's layout (16 key heads, 48
/// value heads, 128-dim) resolves against its portable reference, tracks
/// today's expanded recurrence over two spans, and leaves verification spans
/// (which retain q/k for recovery) on today's path.
#[test]
fn grouped_gdn_prefill_resolves_and_tracks_the_expanded_recurrence() {
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let legacy = gdn_mixer(16, 48, 64, &Device::Cpu, DType::F32);
    let mut grouped = gdn_mixer(16, 48, 64, &Device::Cpu, DType::F32);
    let mut shared = None;
    grouped.resolve_gdn_prefill(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        true,
        &mut shared,
    );
    assert_eq!(grouped.gdn_prefill_path, Qwen38FastPath::Fused);
    assert!(grouped.gdn_prefill);
    assert_eq!(
        shared.as_ref().map(|(geometry, _)| *geometry),
        Some([16, 48, 128, 128])
    );
    assert_grouped_prefill_tracks_legacy(
        &legacy,
        &grouped,
        64,
        &Device::Cpu,
        DType::F32,
        (1e-5, 1e-4),
    );

    let input = gdn_prefill_input(3, 64, 9.0, &Device::Cpu, DType::F32);
    let mut recorded = None;
    let mut state = initialized_state(&grouped, &Device::Cpu);
    grouped
        .forward_sequence_recording(&input, &mut state, Some(&mut recorded))
        .unwrap();
    let recorded = recorded.expect("verification spans record their q/k");
    assert_eq!(recorded.query.dims(), [1, 3, 16, 128]);

    // Production keeps it off the CPU; the switch and an unsupported layout
    // keep today's recurrence with a reason.
    let mut off_cpu = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F32);
    off_cpu.resolve_gdn_prefill(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        false,
        &mut None,
    );
    assert!(!off_cpu.gdn_prefill);
    std::env::set_var(GDN_PREFILL_ENV, "legacy");
    let mut switched = gdn_mixer(2, 6, 64, &Device::Cpu, DType::F32);
    switched.resolve_gdn_prefill(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        true,
        &mut None,
    );
    std::env::remove_var(GDN_PREFILL_ENV);
    assert_eq!(
        switched.gdn_prefill_path,
        Qwen38FastPath::legacy("IZWI_QWEN38_GDN_PREFILL=legacy")
    );
    let mut ungrouped = gdn_mixer(4, 6, 64, &Device::Cpu, DType::F32);
    ungrouped.resolve_gdn_prefill(
        &Qwen38CudaSwitches::default(),
        &Device::Cpu,
        true,
        &mut None,
    );
    assert!(!ungrouped.gdn_prefill);
    assert!(
        matches!(&ungrouped.gdn_prefill_path, Qwen38FastPath::Legacy { reason } if reason.contains("grouped")),
        "{:?}",
        ungrouped.gdn_prefill_path
    );
}

/// The grouped DeltaNet prefill on CUDA at the 27B trunk's layout resolves
/// through its load-time self-check and tracks today's prefill (expanded q/k
/// through the tiled qwen35 kernel), with no head expansion on its spans.
#[cfg(feature = "cuda")]
#[test]
fn cuda_grouped_gdn_prefill_matches_the_expanded_qwen35_kernel_48_heads() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let _guard = counters_lock();
    let mut legacy = gdn_mixer(16, 48, 256, &device, DType::BF16);
    legacy.tiled_recurrence_enabled = true;
    let mut grouped = gdn_mixer(16, 48, 256, &device, DType::BF16);
    grouped.resolve_gdn_prefill(&Qwen38CudaSwitches::default(), &device, false, &mut None);
    assert_eq!(grouped.gdn_prefill_path, Qwen38FastPath::Fused);
    let mut grouped_state = initialized_state(&grouped, &device);
    let before = telemetry::snapshot();
    grouped
        .forward_sequence(
            &gdn_prefill_input(256, 256, 5.0, &device, DType::BF16),
            &mut grouped_state,
        )
        .unwrap();
    let after = telemetry::snapshot();
    assert_eq!(
        after.cuda_deltanet_grouped_prefill_success_total
            - before.cuda_deltanet_grouped_prefill_success_total,
        1
    );
    assert_eq!(
        after.cuda_head_expansion_materializations_total,
        before.cuda_head_expansion_materializations_total
    );
    assert_grouped_prefill_tracks_legacy(
        &legacy,
        &grouped,
        256,
        &device,
        DType::BF16,
        (1e-3, 1e-2),
    );
}

/// Deterministic pseudo-random values in `[-scale, scale]` (xorshift): test
/// weights without the structured cancellation a sinusoid gives a long dot
/// product.
fn noise(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// A full-attention block at the 27B attention geometry (24 query and 4 KV
/// heads of 256, 64 rotary dims in [11, 11, 10] interleaved sections, theta
/// 1e7) over a `hidden`-wide residual, with deterministic weights in `dtype`
/// and F32 zero-centered head-norm gains.
fn attention_block(hidden: usize, device: &Device, dtype: DType) -> Qwen38FullAttention {
    let (heads, kv_heads, dim, rope_dim) = (24usize, 4usize, 256usize, 64usize);
    let scale = 2.0 / (hidden as f32).sqrt();
    let dense = |out: usize, input: usize, seed: f32, scale: f32| {
        Qwen38Projection::Dense(Linear::new(
            Tensor::from_vec(
                noise(out * input, seed as u64, scale),
                (out, input),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .to_device(device)
            .unwrap(),
            None,
        ))
    };
    let gain = |seed: f32| {
        Qwen38RmsNorm::new(
            Tensor::from_vec(
                self_check_wave(dim, seed, 0.2)
                    .iter()
                    .map(|w| 1.0 + w)
                    .collect::<Vec<_>>(),
                dim,
                &Device::Cpu,
            )
            .unwrap()
            .to_device(device)
            .unwrap(),
            1e-6,
        )
    };
    Qwen38FullAttention {
        qkv_proj: Qwen38ProjectionGroup::Separate(vec![
            dense(heads * dim * 2, hidden, 1.0, scale),
            dense(kv_heads * dim, hidden, 2.0, scale),
            dense(kv_heads * dim, hidden, 3.0, scale),
        ]),
        o_proj: dense(
            hidden,
            heads * dim,
            4.0,
            ((heads * dim) as f32).sqrt().recip(),
        ),
        q_norm: gain(5.0),
        k_norm: gain(6.0),
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim: dim,
        rope_dim,
        rope_theta: 10_000_000.0,
        mrope_sections: vec![11, 11, 10],
        rope_kernel_enabled: qwen38_rope_kernel_enabled(device),
        rope_inv_freqs: build_rope_inv_freqs(rope_dim, 10_000_000.0).unwrap(),
        fused_qk: None,
        fused_qk_path: Qwen38FastPath::default(),
        fused_kv_write: false,
        fused_kv_write_path: Qwen38FastPath::default(),
        fused_gate: false,
        fused_gate_path: Qwen38FastPath::default(),
    }
}

/// `rows` single-layer paged caches over one arena on `device` (CPU arena
/// off the accelerators), sized for the attention block.
fn attention_caches(
    device: &Device,
    dtype: DType,
    rows: u32,
    instance: u64,
) -> Vec<PhysicalPagedKvCache> {
    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    const PAGES_PER_ROW: u32 = 2;
    let backend = if device.is_cuda() {
        BackendKind::Cuda
    } else if device.is_metal() {
        BackendKind::Metal
    } else {
        BackendKind::Cpu
    };
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(instance),
        backend,
        device_ordinal: (!device.is_cpu()).then_some(0),
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let binding = KvLayerBinding {
        model_layer: 3,
        physical_layer: 0,
    };
    let config = KvArenaConfig {
        id,
        group,
        page_tokens: 8,
        capacity_pages: PAGES_PER_ROW * rows,
        growth: None,
        dtype,
        layers: vec![KvLayerConfig {
            binding,
            num_kv_heads: 4,
            key_head_dim: 256,
            value_head_dim: 256,
        }],
    };
    let arena: Arc<dyn KvArena> = if device.is_cpu() {
        Arc::new(CpuKvArena::new(config).unwrap())
    } else {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            Arc::new(
                crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                    config,
                    device.clone(),
                )
                .unwrap(),
            )
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        unreachable!("accelerator devices need an accelerator build")
    };
    (0..rows)
        .map(|row| {
            let blocks = (row * PAGES_PER_ROW..(row + 1) * PAGES_PER_ROW)
                .map(|index| CacheBlockRef {
                    arena: id,
                    group,
                    index,
                    slot_generation: 1,
                })
                .collect();
            PhysicalPagedKvCache::new(arena.clone(), vec![binding], blocks, 0).unwrap()
        })
        .collect()
}

/// One attention forward over `inputs` `[1, tokens, hidden]` at `positions`,
/// committed to `cache`, through production's per-forward rotary inputs.
fn attend(
    attention: &Qwen38FullAttention,
    inputs: &Tensor,
    positions: &[[usize; 3]],
    cache: &mut PhysicalPagedKvCache,
) -> Vec<f32> {
    let rope = Qwen38RopeInputs::new(positions, inputs.device(), inputs.dtype())
        .with_decode_step(cache, std::iter::once(attention))
        .unwrap();
    let mut prepared = cache
        .prepare_append(cache.context_len(), positions.len())
        .unwrap();
    let output = attention
        .forward_physical(inputs, &rope, cache, &mut prepared, 0)
        .unwrap();
    cache.commit_prepared(prepared).unwrap();
    host_values(&output).unwrap()
}

fn attention_inputs(
    tokens: usize,
    hidden: usize,
    seed: f32,
    device: &Device,
    dtype: DType,
) -> Tensor {
    Tensor::from_vec(
        noise(tokens * hidden, 1000 + seed as u64, 1.0),
        (1, tokens, hidden),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(dtype)
    .unwrap()
    .to_device(device)
    .unwrap()
}

/// Prefill (the chain) then decode at long, text and image positions: the
/// fused q/k (eager on the CPU arena, which declines the fused K/V write)
/// and the fused gate track today's chain at the 27B attention geometry,
/// scalar and batched.
#[test]
fn fused_attention_tracks_the_chain_at_qwen38_geometry() {
    let _guard = counters_lock();
    let (hidden, dtype) = (64, DType::F16);
    let legacy = attention_block(hidden, &Device::Cpu, dtype);
    let mut fused = attention_block(hidden, &Device::Cpu, dtype);
    fused.resolve_fused_attention(&Qwen38CudaSwitches::default(), &Device::Cpu, dtype, true);
    for path in fused.fast_paths() {
        assert_eq!(*path, Qwen38FastPath::Fused);
    }
    let prompt: Vec<[usize; 3]> = (0..3).map(|p| [199_990 + p; 3]).collect();
    let steps = [[199_993; 3], [199_994; 3], [7, 3, 11], [199_996; 3]];
    let run = |attention: &Qwen38FullAttention, instance: u64| {
        let mut cache = attention_caches(&Device::Cpu, dtype, 1, instance).remove(0);
        let mut outputs = vec![attend(
            attention,
            &attention_inputs(3, hidden, 1.0, &Device::Cpu, dtype),
            &prompt,
            &mut cache,
        )];
        for (step, position) in steps.iter().enumerate() {
            outputs.push(attend(
                attention,
                &attention_inputs(1, hidden, 10.0 + step as f32, &Device::Cpu, dtype),
                &[*position],
                &mut cache,
            ));
        }
        outputs
    };
    let before = telemetry::snapshot();
    let expected = run(&legacy, 501);
    let between = telemetry::snapshot();
    let actual = run(&fused, 502);
    let after = telemetry::snapshot();
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_close(
            &format!("attention forward {index}"),
            actual,
            expected,
            0.01,
            0.03,
        );
    }
    assert_eq!(
        between.cuda_fused_qk_norm_rope_total,
        before.cuda_fused_qk_norm_rope_total
    );
    assert_eq!(
        between.cuda_fused_attn_gate_total,
        before.cuda_fused_attn_gate_total
    );
    assert_eq!(
        after.cuda_fused_qk_norm_rope_total - between.cuda_fused_qk_norm_rope_total,
        steps.len() as u64
    );
    // The CPU arena declines the fused K/V write.
    assert_eq!(
        after.cuda_fused_qk_norm_rope_kv_total,
        between.cuda_fused_qk_norm_rope_kv_total
    );
    // The gate also serves the prefill forward.
    assert_eq!(
        after.cuda_fused_attn_gate_total - between.cuda_fused_attn_gate_total,
        steps.len() as u64 + 1
    );

    // Two ragged rows through the batched decode.
    let batch = |attention: &Qwen38FullAttention, instance: u64| {
        let mut caches = attention_caches(&Device::Cpu, dtype, 2, instance);
        for (row, cache) in caches.iter_mut().enumerate() {
            let positions: Vec<[usize; 3]> = (0..2 + row).map(|p| [p; 3]).collect();
            attend(
                attention,
                &attention_inputs(2 + row, hidden, 20.0 + row as f32, &Device::Cpu, dtype),
                &positions,
                cache,
            );
        }
        let positions = [[2, 2, 2], [3000, 3000, 3000]];
        let starts = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        let slots = caches
            .iter()
            .zip(&starts)
            .map(|(cache, start)| cache.slots_for_append(*start, 1).unwrap()[0])
            .collect::<Vec<_>>();
        let lowered = caches[0].arena().lower_slots(&slots).unwrap();
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .zip(&starts)
                .map(|(cache, start)| cache.sequence_table(start + 1).unwrap())
                .collect(),
        };
        let mut completions =
            KvWriteCompletionCollector::new(caches[0].arena().config(), lowered.logical_slots())
                .unwrap();
        let refs = caches.iter().collect::<Vec<_>>();
        let output = attention
            .forward_physical_decode_batch(
                &attention_inputs(2, hidden, 40.0, &Device::Cpu, dtype)
                    .reshape((2, 1, hidden))
                    .unwrap(),
                &positions,
                &refs,
                lowered.as_ref(),
                &metadata,
                &mut completions,
                0,
            )
            .unwrap();
        completions.drain().unwrap();
        host_values(&output).unwrap()
    };
    assert_close(
        "batched attention",
        &batch(&fused, 503),
        &batch(&legacy, 504),
        0.01,
        0.03,
    );
}

/// The self-check at long positions sees a 0.1% frequency error, which a
/// short position would hide; an image position needs valid sections.
#[test]
fn fused_qk_self_check_rejects_a_wrong_rotation() {
    let attention = attention_block(64, &Device::Cpu, DType::F16);
    let spec = QkNormRopeSpec {
        num_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        rope_dim: 64,
        eps: 1e-6,
    };
    let inv_freq = |scale: f32| {
        Tensor::from_vec(
            attention
                .rope_inv_freqs
                .iter()
                .map(|f| f * scale)
                .collect::<Vec<_>>(),
            32,
            &Device::Cpu,
        )
        .unwrap()
    };
    attention
        .fused_qk_self_check(&spec, &inv_freq(1.0), &Device::Cpu, DType::F16)
        .unwrap();
    let error = attention
        .fused_qk_self_check(&spec, &inv_freq(1.001), &Device::Cpu, DType::F16)
        .unwrap_err();
    assert!(error.to_string().contains("3000"), "{error}");
    assert_eq!(
        attention.mrope_position([7, 3, 11]).unwrap().height_section,
        11
    );
    let mut text_only = attention_block(64, &Device::Cpu, DType::F16);
    text_only.mrope_sections = vec![11];
    assert!(text_only.mrope_position([7, 3, 11]).is_none());
    assert!(text_only.mrope_position([7, 7, 7]).is_some());
}

/// Production resolution on CUDA at the 27B attention geometry turns all
/// three paths on; on a live BF16 accelerator arena the decode tokens then
/// take the K/V-writing launch through `write_fused_and_attend`, and every
/// output (including later tokens attending to the fused writes) tracks
/// today's chain.
#[cfg(feature = "cuda")]
#[test]
fn cuda_qk_norm_rope_kv_matches_chain_qwen38_geometry() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let _guard = counters_lock();
    let (hidden, dtype) = (256, DType::BF16);
    let legacy = attention_block(hidden, &device, dtype);
    let mut fused = attention_block(hidden, &device, dtype);
    fused.resolve_fused_attention(&Qwen38CudaSwitches::default(), &device, dtype, false);
    for path in fused.fast_paths() {
        assert_eq!(*path, Qwen38FastPath::Fused);
    }
    let run = |attention: &Qwen38FullAttention, instance: u64| {
        let mut cache = attention_caches(&device, dtype, 1, instance).remove(0);
        let prompt: Vec<[usize; 3]> = (0..4).map(|p| [p; 3]).collect();
        let mut outputs = vec![attend(
            attention,
            &attention_inputs(4, hidden, 1.0, &device, dtype),
            &prompt,
            &mut cache,
        )];
        for step in 0..6 {
            outputs.push(attend(
                attention,
                &attention_inputs(1, hidden, 10.0 + step as f32, &device, dtype),
                &[[4 + step; 3]],
                &mut cache,
            ));
        }
        outputs
    };
    let expected = run(&legacy, 601);
    let before = telemetry::snapshot();
    let actual = run(&fused, 602);
    let after = telemetry::snapshot();
    assert_eq!(
        after.cuda_fused_qk_norm_rope_kv_total - before.cuda_fused_qk_norm_rope_kv_total,
        6
    );
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_close(
            &format!("CUDA attention forward {index}"),
            actual,
            expected,
            0.01,
            0.03,
        );
    }
}
