//! Native block-FP8 residency through production loading, on the FP8-geometry
//! fixture (`mtp::tests::write_fp8_geometry_checkpoint`: 128x128 blocks, so
//! the compact FP8 kernels and the packing paths are reachable; the golden
//! fixture's 16x16 blocks are not).
//!
//! On the CPU the portable reference of `kernels/cuda/fp8.rs` stands in for
//! the device kernels: `force_cpu_native_fp8_for_tests` makes the residency
//! policy resolve native FP8 here through the same gates, packing, load-time
//! check and execution path CUDA uses.

use super::golden_tests::{argmax, relative_l2, GoldenArena};
use super::*;
use crate::models::architectures::qwen38::mtp::tests::{write_fp8_geometry_checkpoint, TestDir};
use crate::models::architectures::qwen38::mtp::Qwen38MtpHead;
use crate::models::architectures::qwen38::native::Qwen38NativeCheckpoint;
use crate::models::architectures::qwen38::residency::{
    force_cpu_native_fp8_for_tests, Qwen38ProjectionPlan, FP8_GEMV_ENV, FP8_LM_HEAD_ENV,
    FP8_PROJECTIONS_ENV,
};
use crate::performance::{CudaProjectionBackend, OptimizationMode, PerformanceConfig};

pub(super) const PROMPT: [u32; 8] = [3, 141, 59, 265, 35, 89, 79, 323];
pub(super) const DECODE_STEPS: usize = 8;

/// Logits of every prompt row (one prefill span) and of each decode step,
/// with the decode tokens forced to `tokens` when given (teacher forcing),
/// otherwise greedy. Returns the rows and the decoded tokens.
pub(super) fn fixture_rows(
    model: &Qwen38TextModel,
    arena: &GoldenArena,
    session: u32,
    tokens: Option<&[u32]>,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let positions: Vec<[usize; 3]> = (0..PROMPT.len() + DECODE_STEPS)
        .map(|p| [p, p, p])
        .collect();
    let host = |logits: &Tensor| -> Vec<Vec<f32>> {
        let vocab = logits.dim(D::Minus1).unwrap();
        logits
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .chunks(vocab)
            .map(<[f32]>::to_vec)
            .collect()
    };
    let mut state = model.new_state();
    let mut cache = arena.session(session);
    let output = model
        .prefill_token_ids_with_hidden_physical(
            &PROMPT,
            &positions[..PROMPT.len()],
            &mut state,
            &mut cache,
            false,
        )
        .unwrap()
        .expect("prefill output");
    let mut rows = host(
        &model
            .project_target_hidden_span(&output.hidden_states)
            .unwrap(),
    );
    let mut decoded = Vec::with_capacity(DECODE_STEPS);
    for step in 0..DECODE_STEPS {
        let next = tokens.map_or_else(
            || argmax(rows.last().unwrap()) as u32,
            |tokens| tokens[step],
        );
        decoded.push(next);
        let logits = model
            .forward_token_id_at_physical(
                next,
                positions[PROMPT.len() + step],
                &mut state,
                &mut cache,
            )
            .unwrap();
        rows.extend(host(&logits));
    }
    (rows, decoded)
}

/// The CPU's F32 expansion (`cpu.projection_backend = "expanded"`): the
/// exact reference the other plans are compared with.
pub(super) fn expanded_cpu() -> PerformanceConfig {
    let mut performance = PerformanceConfig::default();
    performance.cpu.projection_backend = crate::performance::CpuProjectionBackend::Expanded;
    performance
}

fn fixture() -> (TestDir, Qwen38NativeCheckpoint) {
    let dir = TestDir::new("fp8-geometry");
    write_fp8_geometry_checkpoint(dir.path());
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(dir.path()).unwrap();
    (dir, checkpoint)
}

fn resolve(
    checkpoint: &Qwen38NativeCheckpoint,
    target: ProjectionMaterialization,
    performance: &PerformanceConfig,
) -> Qwen38ResidencyPolicy {
    Qwen38ResidencyPolicy::resolve(
        &Device::Cpu,
        target,
        performance,
        &checkpoint.tensors,
        &checkpoint.config,
    )
}

fn load(
    checkpoint: &Qwen38NativeCheckpoint,
    target: ProjectionMaterialization,
    policy: &Qwen38ResidencyPolicy,
) -> Result<Qwen38TextModel> {
    Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &Device::Cpu,
        target,
        &PerformanceConfig::default().cuda,
        policy,
    )
}

/// Every trunk projection that came from a block-FP8 checkpoint tensor, as
/// `(native FP8, packed)`.
fn fp8_projection_forms(model: &Qwen38TextModel) -> Vec<(bool, bool)> {
    let mut forms = Vec::new();
    let mut group = |group: &Qwen38ProjectionGroup| match group {
        Qwen38ProjectionGroup::Separate(parts) => forms.extend(
            parts
                .iter()
                .map(|part| (matches!(part, Qwen38Projection::NativeFp8 { .. }), false)),
        ),
        Qwen38ProjectionGroup::Packed { projection, .. } => forms.push((
            matches!(projection, Qwen38Projection::NativeFp8 { .. }),
            true,
        )),
    };
    for layer in &model.layers {
        group(&layer.mlp.gate_up);
        match &layer.mixer {
            Qwen38Mixer::Linear(linear) => group(&linear.qkv_z_proj),
            Qwen38Mixer::Full(attention) => group(&attention.qkv_proj),
        }
    }
    let mut single = |projection: &Qwen38Projection| {
        forms.push((
            matches!(projection, Qwen38Projection::NativeFp8 { .. }),
            false,
        ))
    };
    for layer in &model.layers {
        single(&layer.mlp.down);
        match &layer.mixer {
            Qwen38Mixer::Linear(linear) => single(&linear.out_proj),
            Qwen38Mixer::Full(attention) => single(&attention.o_proj),
        }
    }
    forms
}

/// The production CPU plan (expanded F32) is the reference; native block
/// FP8 resolved through production loading must keep every projection raw
/// FP8, pack every group, pass its load-time checks, and agree with the
/// expanded load to F32 rounding on every prompt row and decode step.
#[test]
fn forced_native_fp8_trunk_tracks_the_expanded_f32_load() {
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // The block-FP8 LM head is a deliberate precision reduction with its own
    // test; keep the head dense so the trunk comparison stays exact.
    std::env::set_var(FP8_LM_HEAD_ENV, "legacy");
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            std::env::remove_var(FP8_LM_HEAD_ENV);
        }
    }
    let _restore = Restore;
    let (_dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::F32;
    let expanded = load(
        &checkpoint,
        target,
        &resolve(&checkpoint, target, &expanded_cpu()),
    )
    .unwrap();
    assert_eq!(
        expanded.projection_representation(),
        Qwen38ProjectionRepresentation::ExpandedF32
    );
    assert!(fp8_projection_forms(&expanded)
        .iter()
        .all(|(native, _)| !native));

    for packing in [true, false] {
        let _forced = force_cpu_native_fp8_for_tests();
        let mut performance = PerformanceConfig::default();
        if !packing {
            performance.cuda.packed_projections = OptimizationMode::Off;
        }
        let policy = resolve(&checkpoint, target, &performance);
        assert_eq!(
            policy.plan(),
            Qwen38ProjectionPlan::NativeFp8 { explicit: false }
        );
        let native = load(&checkpoint, target, &policy).unwrap();
        let forms = fp8_projection_forms(&native);
        assert!(forms.iter().all(|(native, _)| *native), "{forms:?}");
        // Packed: 3 DeltaNet qkvz + 1 attention qkv + 4 gate_up groups.
        let packed = forms.iter().filter(|(_, packed)| *packed).count();
        assert_eq!(packed, if packing { 8 } else { 0 });
        let summary = summarize(native.fp8_packing());
        assert_eq!(
            summary["fused_layers"],
            if packing { 8 } else { 0 },
            "{summary}"
        );
        assert_eq!(summary["legacy_layers"], if packing { 0 } else { 8 });

        let arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F32);
        let (reference, tokens) = fixture_rows(&expanded, &arena, 0, None);
        let (rows, _) = fixture_rows(&native, &arena, 1, Some(&tokens));
        assert_eq!(rows.len(), PROMPT.len() + DECODE_STEPS);
        let mut worst = 0f32;
        for (index, (row, reference)) in rows.iter().zip(&reference).enumerate() {
            let error = relative_l2(row, reference);
            assert!(error < 1e-4, "row {index}: relative L2 {error}");
            assert_eq!(argmax(row), argmax(reference), "row {index}: greedy token");
            worst = worst.max(error);
        }
        eprintln!(
            "qwen38 CPU native FP8 (packing {packing}) vs expanded: worst relative L2 {worst:e}"
        );

        // The MTP head takes the same forms and checks.
        let head = Qwen38MtpHead::load_native_with_policy(
            &checkpoint.tensors,
            &checkpoint.config,
            &checkpoint.mtp,
            &Device::Cpu,
            target,
            &performance.cuda,
            &policy,
        )
        .unwrap();
        assert_eq!(
            summarize(head.fp8_packing())["fused_layers"],
            if packing { 2 } else { 0 }
        );
    }
}

/// Every native block-FP8 trunk projection, as `(kind tag, prefill GEMM)`.
fn fp8_prefill_gemm_flags(model: &Qwen38TextModel) -> Vec<bool> {
    let mut flags = Vec::new();
    let mut single = |projection: &Qwen38Projection| {
        if let Qwen38Projection::NativeFp8 { prefill_gemm, .. } = projection {
            flags.push(*prefill_gemm);
        }
    };
    for layer in &model.layers {
        let groups = [
            &layer.mlp.gate_up,
            match &layer.mixer {
                Qwen38Mixer::Linear(linear) => &linear.qkv_z_proj,
                Qwen38Mixer::Full(attention) => &attention.qkv_proj,
            },
        ];
        for group in groups {
            match group {
                Qwen38ProjectionGroup::Separate(parts) => parts.iter().for_each(&mut single),
                Qwen38ProjectionGroup::Packed { projection, .. } => single(projection),
            }
        }
        single(&layer.mlp.down);
        match &layer.mixer {
            Qwen38Mixer::Linear(linear) => single(&linear.out_proj),
            Qwen38Mixer::Full(attention) => single(&attention.o_proj),
        }
    }
    flags
}

/// The hidden states of one 20-row prefill (above the 16-row switch-over).
fn long_prefill_hidden(model: &Qwen38TextModel, checkpoint: &Qwen38NativeCheckpoint) -> Vec<f32> {
    let arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F32);
    let tokens: Vec<u32> = (0..20u32).map(|i| (i * 37 + 5) % 500).collect();
    let positions: Vec<[usize; 3]> = (0..tokens.len()).map(|p| [p, p, p]).collect();
    let mut state = model.new_state();
    let mut cache = arena.session(0);
    model
        .prefill_token_ids_with_hidden_physical(&tokens, &positions, &mut state, &mut cache, false)
        .unwrap()
        .expect("prefill output")
        .hidden_states
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The dequantized prefill GEMM through production loading on the CPU
/// stand-in: every projection kind passes its load-time check and takes the
/// GEMM, a 20-row prefill tracks the compact-kernel load to F32 rounding,
/// and the prefill really runs the GEMM (a fault injected into the GEMM's
/// output after load moves the result). RED: with the fault injected during
/// load every kind fails its check, keeps the compact kernel with a
/// `self_check_failed` reason, and the load still succeeds; the switch keeps
/// the policy off.
#[test]
fn forced_prefill_gemm_resolves_per_kind_and_falls_back_on_a_failed_check() {
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (_dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::F32;
    let _forced = force_cpu_native_fp8_for_tests();
    let compact_policy = resolve(&checkpoint, target, &PerformanceConfig::default());
    assert!(
        !compact_policy.fp8_prefill_gemm(),
        "the CPU stand-in leaves the CUDA prefill GEMM off unless forced"
    );
    let compact = load(&checkpoint, target, &compact_policy).unwrap();
    assert!(fp8_prefill_gemm_flags(&compact).iter().all(|flag| !flag));
    assert_eq!(
        summarize(compact.fp8_prefill_gemm_checks())["backend"],
        "none"
    );
    let reference = long_prefill_hidden(&compact, &checkpoint);

    let mut policy = compact_policy.clone();
    policy.force_fp8_prefill_gemm_for_tests();
    assert!(policy.fp8_prefill_gemm());
    let gemm = load(&checkpoint, target, &policy).unwrap();
    let flags = fp8_prefill_gemm_flags(&gemm);
    assert!(
        !flags.is_empty() && flags.iter().all(|flag| *flag),
        "{flags:?}"
    );
    let summary = summarize(gemm.fp8_prefill_gemm_checks());
    assert_eq!(summary["backend"], "fused", "{summary}");
    // Packed qkvz, DeltaNet out, packed qkv, attention out, gate_up, down.
    assert_eq!(summary["fused_layers"], 6, "{summary}");
    let hidden = long_prefill_hidden(&gemm, &checkpoint);
    let error = relative_l2(&hidden, &reference);
    assert!(
        error < 1e-5,
        "prefill GEMM vs compact kernel: relative L2 {error}"
    );
    {
        let _fault = prefill_gemm_fault::inject();
        let faulted = long_prefill_hidden(&gemm, &checkpoint);
        assert!(
            relative_l2(&faulted, &reference) > 1e-2,
            "the 20-row prefill must run the prefill GEMM"
        );
    }

    let failures = crate::models::architectures::qwen38::telemetry::snapshot()
        .fast_path_self_check_failures_total;
    let fallback = {
        let _fault = prefill_gemm_fault::inject();
        load(&checkpoint, target, &policy).unwrap()
    };
    assert!(fp8_prefill_gemm_flags(&fallback).iter().all(|flag| !flag));
    let summary = summarize(fallback.fp8_prefill_gemm_checks());
    assert_eq!(summary["backend"], "legacy", "{summary}");
    assert_eq!(summary["legacy_layers"], 6, "{summary}");
    let reasons = summary["legacy_reasons"].to_string();
    assert!(reasons.contains("self_check_failed"), "{reasons}");
    assert!(reasons.contains("diverges"), "{reasons}");
    assert!(
        crate::models::architectures::qwen38::telemetry::snapshot()
            .fast_path_self_check_failures_total
            >= failures + 6
    );
    assert_eq!(long_prefill_hidden(&fallback, &checkpoint), reference);

    std::env::set_var(
        crate::models::architectures::qwen38::residency::FP8_PREFILL_GEMM_ENV,
        "legacy",
    );
    let switched = resolve(&checkpoint, target, &PerformanceConfig::default());
    std::env::remove_var(crate::models::architectures::qwen38::residency::FP8_PREFILL_GEMM_ENV);
    assert!(!switched.fp8_prefill_gemm());
    assert_eq!(
        switched.diagnostics()["fp8_prefill_gemm"]["legacy_reasons"][0],
        "IZWI_QWEN38_FP8_PREFILL_GEMM=legacy"
    );
}

/// The load-time kernel timings run on CUDA only and honour their switch;
/// the CPU stand-in reports why it skipped them.
#[test]
fn load_time_kernel_timings_are_cuda_only_and_switchable() {
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (_dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::F32;
    let _forced = force_cpu_native_fp8_for_tests();
    let model = load(
        &checkpoint,
        target,
        &resolve(&checkpoint, target, &PerformanceConfig::default()),
    )
    .unwrap();
    assert_eq!(model.kernel_timings()["status"], "cuda_only");
    std::env::set_var(super::kernel_timings::LOAD_TIMINGS_ENV, "off");
    let off = model.kernel_timings();
    std::env::remove_var(super::kernel_timings::LOAD_TIMINGS_ENV);
    assert_eq!(off["status"], "off");
    assert_eq!(off["switch"], "IZWI_QWEN38_LOAD_TIMINGS");
}

/// On CUDA the load-time timings measure every kind through production
/// loading: positive GEMV bandwidth per kind, the prefill GEMM per kind at
/// 256 rows, the compact kernel against it, and the grouped DeltaNet chunk.
#[cfg(feature = "cuda")]
#[test]
fn cuda_load_time_kernel_timings_measure_every_path() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let (_dir, checkpoint) = fixture();
    let performance = PerformanceConfig::default();
    let target = ProjectionMaterialization::BF16;
    let policy = Qwen38ResidencyPolicy::resolve(
        &device,
        target,
        &performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    let model = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &device,
        target,
        &performance.cuda,
        &policy,
    )
    .unwrap();
    assert_eq!(
        summarize(model.fp8_prefill_gemm_checks())["backend"],
        "fused",
        "the prefill GEMM resolves through production loading"
    );
    let timings = model.kernel_timings();
    assert_eq!(timings["status"], "measured", "{timings}");
    let kinds = timings["decode_fp8"]["kinds"].as_array().unwrap();
    assert!(kinds.len() >= 6, "{timings}");
    assert!(
        kinds
            .iter()
            .all(|kind| kind["gb_per_s"].as_f64().unwrap() > 0.0),
        "{timings}"
    );
    let prefill = timings["prefill_projections"]["kinds"].as_array().unwrap();
    assert!(
        prefill
            .iter()
            .all(|kind| kind["gemm_tflop_per_s"].as_f64().unwrap() > 0.0),
        "{timings}"
    );
    assert!(
        timings["prefill_projections"]["compact_vs_gemm"]["mm_ms_256_rows"]
            .as_f64()
            .is_some()
    );
    assert_eq!(timings["deltanet_prefill"]["path"], "grouped", "{timings}");
    assert!(timings["wall_ms"].as_f64().unwrap() < 2_000.0, "{timings}");
}

/// The block-FP8 LM head beside native FP8 projections: quantized at load
/// from the BF16 mapping, gated, and a deliberate precision reduction that
/// stays within a few percent of the dense head's logits.
#[test]
fn forced_fp8_lm_head_quantizes_checks_and_tracks_the_dense_head() {
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (_dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::F32;
    let performance = PerformanceConfig::default();
    let expanded = load(
        &checkpoint,
        target,
        &resolve(&checkpoint, target, &expanded_cpu()),
    )
    .unwrap();
    assert_eq!(expanded.fp8_lm_head_diagnostics()["backend"], "legacy");
    let _forced = force_cpu_native_fp8_for_tests();
    let policy = resolve(&checkpoint, target, &performance);
    assert_eq!(policy.lm_head_form([512, 256]), Qwen38WeightForm::RawFp8);
    let native = load(&checkpoint, target, &policy).unwrap();
    let Qwen38Projection::NativeFp8 { raw, .. } = &native.output else {
        panic!("the LM head must be block FP8");
    };
    assert_eq!(raw.shape, [512, 256]);
    assert_eq!(raw.scale_shape, [4, 2]);
    let diagnostics = native.fp8_lm_head_diagnostics();
    assert_eq!(diagnostics["backend"], "fused", "{diagnostics}");
    let gate = diagnostics["gate_relative_l2"].as_f64().unwrap();
    assert!(gate > 1e-3 && gate < 0.08, "gate relative L2 {gate}");
    assert!(diagnostics["precision"]
        .as_str()
        .unwrap()
        .contains("deliberately reduced"));

    let arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F32);
    let (reference, tokens) = fixture_rows(&expanded, &arena, 0, None);
    let (rows, _) = fixture_rows(&native, &arena, 1, Some(&tokens));
    let mut worst = 0f32;
    let mut agree = 0usize;
    for (index, (row, reference)) in rows.iter().zip(&reference).enumerate() {
        let error = relative_l2(row, reference);
        assert!(error < 0.05, "row {index}: relative L2 {error}");
        worst = worst.max(error);
        agree += usize::from(argmax(row) == argmax(reference));
    }
    eprintln!(
        "qwen38 CPU FP8 LM head vs dense: worst relative L2 {worst:e}, greedy agreement {agree}/{}",
        rows.len()
    );
    assert!(
        agree * 10 >= rows.len() * 9,
        "greedy agreement {agree}/{}",
        rows.len()
    );
    assert!(worst > 1e-4, "the FP8 head must actually be in use");

    // The kill switch and a poisoned head keep the dense head.
    std::env::set_var(FP8_LM_HEAD_ENV, "legacy");
    let killed = resolve(&checkpoint, target, &performance);
    std::env::remove_var(FP8_LM_HEAD_ENV);
    assert_eq!(
        killed.lm_head_form([512, 256]),
        Qwen38WeightForm::Dense(DType::F32)
    );
    assert!(matches!(
        killed.plan(),
        Qwen38ProjectionPlan::NativeFp8 { .. }
    ));
    policy.poison_lm_head("forced head mismatch");
    let poisoned = resolve(&checkpoint, target, &performance);
    assert_eq!(
        poisoned.lm_head_form([512, 256]),
        Qwen38WeightForm::Dense(DType::F32)
    );
    assert!(matches!(
        poisoned.plan(),
        Qwen38ProjectionPlan::NativeFp8 { .. }
    ));
    let model = load(&checkpoint, target, &poisoned).unwrap();
    assert!(matches!(model.output, Qwen38Projection::Dense(_)));
    let diagnostics = model.fp8_lm_head_diagnostics();
    assert!(diagnostics["legacy_reasons"][0]
        .as_str()
        .unwrap()
        .contains("forced head mismatch"));
}

/// The load-time check compares a resident FP8 projection with the ORIGINAL
/// per-part checkpoint bytes, so a pack built in the wrong part order, or
/// with one wrong block scale, fails it (RED), while the true pack passes.
#[test]
fn fp8_load_check_rejects_a_swapped_pack_and_a_wrong_scale() {
    let (_dir, checkpoint) = fixture();
    let block = checkpoint.config.block_fp8.block_shape;
    let prefix = "model.language_model.layers.0.linear_attn";
    let (qkv, z) = (
        format!("{prefix}.in_proj_qkv.weight"),
        format!("{prefix}.in_proj_z.weight"),
    );
    let parts: [(&str, [usize; 2]); 2] = [(&qkv, [1280, 256]), (&z, [768, 256])];
    let check = |projection: &Qwen38Projection| {
        fp8_projection_check(
            &checkpoint.tensors,
            &parts,
            block,
            projection,
            DType::F32,
            &Device::Cpu,
        )
    };
    let pack = |order: &[(&str, [usize; 2])]| Qwen38Projection::NativeFp8 {
        raw: checkpoint
            .tensors
            .materialize_block_fp8_raw_group(order, block, &Device::Cpu)
            .unwrap(),
        kernel: Qwen38Fp8Kernel::Reference,
        prefill_gemm: false,
    };
    check(&pack(&parts)).expect("the true pack passes");
    let swapped = check(&pack(&[parts[1], parts[0]])).unwrap_err();
    assert!(swapped.to_string().contains("diverges"), "{swapped}");

    let Qwen38Projection::NativeFp8 {
        mut raw, kernel, ..
    } = pack(&parts)
    else {
        unreachable!()
    };
    let mut scales = raw.scales.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // One block of the z part (scale row 11 of 16): 3x too large.
    scales[11 * 2] *= 3.0;
    raw.scales = Tensor::from_vec(scales, (16, 2), &Device::Cpu).unwrap();
    let wrong_scale = check(&Qwen38Projection::NativeFp8 {
        raw,
        kernel,
        prefill_gemm: false,
    })
    .unwrap_err();
    assert!(
        wrong_scale.to_string().contains("diverges"),
        "{wrong_scale}"
    );
}

/// A pack's bytes and scale rows are exactly the separate parts', in order.
#[test]
fn fp8_group_packs_row_blocks_and_matches_separate_parts_exactly() {
    let (_dir, checkpoint) = fixture();
    let block = checkpoint.config.block_fp8.block_shape;
    let prefix = "model.language_model.layers.3.self_attn";
    let names = [
        format!("{prefix}.q_proj.weight"),
        format!("{prefix}.k_proj.weight"),
        format!("{prefix}.v_proj.weight"),
    ];
    let parts: Vec<(&str, [usize; 2])> = vec![
        (&names[0], [1024, 256]),
        (&names[1], [256, 256]),
        (&names[2], [256, 256]),
    ];
    let packed = checkpoint
        .tensors
        .materialize_block_fp8_raw_group(&parts, block, &Device::Cpu)
        .unwrap();
    assert_eq!(packed.shape, [1536, 256]);
    assert_eq!(packed.scale_shape, [12, 2]);
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    for (name, shape) in &parts {
        let raw = checkpoint
            .tensors
            .materialize_block_fp8_raw(name, *shape, block, &Device::Cpu)
            .unwrap();
        bytes.extend(raw.weights.flatten_all().unwrap().to_vec1::<u8>().unwrap());
        scales.extend(raw.scales.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
    assert!(bytes.iter().any(|b| *b & 0x7f > 0x40) && bytes.iter().any(|b| *b & 0x7f < 0x20));
    assert_eq!(
        packed
            .weights
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        bytes
    );
    assert_eq!(
        packed
            .scales
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        scales
    );
    // Distinct block scales on both axes: a swapped index cannot hide.
    let distinct: std::collections::BTreeSet<u32> = scales.iter().map(|s| s.to_bits()).collect();
    assert!(distinct.len() > scales.len() / 2, "{scales:?}");
}

/// A part that ends inside a 128-row scale block cannot share the pack's
/// scale grid: the materializer refuses it and the loader keeps the parts
/// separate (still raw FP8), recording why.
#[test]
fn fp8_group_with_non_128_rows_stays_separate() {
    use crate::models::architectures::qwen38::mtp::tests::push_random_fp8_projection;
    let dir = TestDir::new("fp8-unaligned-pack");
    let mut tensors = Vec::new();
    push_random_fp8_projection(&mut tensors, "a.weight", [192, 256], [128, 128], 5);
    push_random_fp8_projection(&mut tensors, "b.weight", [128, 256], [128, 128], 6);
    crate::models::architectures::qwen38::mtp::tests::write_fixture(
        dir.path(),
        &tensors,
        "model.safetensors",
    );
    let source = IndexedSafetensors::open(dir.path()).unwrap();
    let parts: [(&str, [usize; 2]); 2] = [("a.weight", [192, 256]), ("b.weight", [128, 256])];
    let error = source
        .materialize_block_fp8_raw_group(&parts, [128, 128], &Device::Cpu)
        .unwrap_err();
    assert!(error.to_string().contains("scale block"), "{error}");
    // Reversed, only the last part is off the block: that packs.
    source
        .materialize_block_fp8_raw_group(&[parts[1], parts[0]], [128, 128], &Device::Cpu)
        .unwrap();

    let policy = Qwen38ResidencyPolicy::forced_native_fp8_for_tests(ProjectionMaterialization::F32);
    let loader = Qwen38WeightLoader::new(
        &source,
        &policy,
        &Device::Cpu,
        [128, 128],
        ProjectionMaterialization::F32,
    );
    let group = load_native_projection_group(&loader, &parts).unwrap();
    let Qwen38ProjectionGroup::Separate(separate) = &group else {
        panic!("an unaligned pack must stay separate");
    };
    assert!(separate
        .iter()
        .all(|part| matches!(part, Qwen38Projection::NativeFp8 { .. })));
    let packing = loader.into_packing();
    assert_eq!(packing.len(), 1);
    assert!(matches!(&packing[0], Qwen38FastPath::Legacy { reason } if reason.contains("128-row")));
}

/// The residency policy's decisions, resolved through the real gates on the
/// CPU reference: every switch that turns native FP8 off, the GEMV and
/// packing switches, poisoning, and the kernel-contract gate.
#[test]
fn projection_plan_policy_matrix() {
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::F32;
    // Without the CPU stand-in, the CPU keeps Q8_0 (or the F32 expansion).
    assert_eq!(
        resolve(&checkpoint, target, &PerformanceConfig::default()).plan(),
        Qwen38ProjectionPlan::Q8
    );
    assert_eq!(
        resolve(&checkpoint, target, &expanded_cpu()).plan(),
        Qwen38ProjectionPlan::Expanded
    );
    let _forced = force_cpu_native_fp8_for_tests();
    let auto = resolve(&checkpoint, target, &PerformanceConfig::default());
    assert_eq!(
        auto.plan(),
        Qwen38ProjectionPlan::NativeFp8 { explicit: false }
    );
    assert!(auto.fp8_vectorized());
    assert!(auto.packing_allowed(&[Qwen38WeightForm::RawFp8; 2]));
    assert_eq!(auto.execution_mode(), "native_block_fp8_auto_self_checked");
    assert_eq!(auto.lm_head_form([512, 256]), Qwen38WeightForm::RawFp8);

    let mut explicit = PerformanceConfig::default();
    explicit.cuda.projection_backend = CudaProjectionBackend::NativeFp8;
    assert_eq!(
        resolve(&checkpoint, target, &explicit).plan(),
        Qwen38ProjectionPlan::NativeFp8 { explicit: true }
    );

    let q8_reason = |performance: &PerformanceConfig| {
        let policy = resolve(&checkpoint, target, performance);
        assert_eq!(policy.plan(), Qwen38ProjectionPlan::Q8);
        policy.execution_mode()
    };
    let mut mode_off = PerformanceConfig::default();
    mode_off.cuda.mode = OptimizationMode::Off;
    assert_eq!(q8_reason(&mode_off), "q8_0_fallback:cuda.mode=off");
    let mut q8 = PerformanceConfig::default();
    q8.cuda.projection_backend = CudaProjectionBackend::Q8;
    assert_eq!(q8_reason(&q8), "q8_0_fallback:cuda.projection_backend=q8");
    // Q8_0 projections on the CPU take the CPU's Q8_0 head.
    assert_eq!(
        resolve(&checkpoint, target, &q8).lm_head_form([512, 256]),
        Qwen38WeightForm::Q8_0
    );
    std::env::set_var(FP8_PROJECTIONS_ENV, "legacy");
    let killed = q8_reason(&PerformanceConfig::default());
    std::env::remove_var(FP8_PROJECTIONS_ENV);
    assert_eq!(killed, "q8_0_fallback:IZWI_QWEN38_FP8_PROJECTIONS=legacy");

    std::env::set_var(FP8_GEMV_ENV, "legacy");
    let reference = resolve(&checkpoint, target, &PerformanceConfig::default());
    std::env::remove_var(FP8_GEMV_ENV);
    assert!(matches!(
        reference.plan(),
        Qwen38ProjectionPlan::NativeFp8 { .. }
    ));
    assert!(!reference.fp8_vectorized());
    let mut fused_off = PerformanceConfig::default();
    fused_off.cuda.fused_decode = OptimizationMode::Off;
    assert!(!resolve(&checkpoint, target, &fused_off).fp8_vectorized());
    let mut unpacked = PerformanceConfig::default();
    unpacked.cuda.packed_projections = OptimizationMode::Off;
    assert!(
        !resolve(&checkpoint, target, &unpacked).packing_allowed(&[Qwen38WeightForm::RawFp8; 2])
    );

    // A load-time failure poisons this device + checkpoint: Q8 from then on.
    auto.poison_projections("test mismatch");
    let poisoned = q8_reason(&PerformanceConfig::default());
    assert!(
        poisoned.contains("poisoned") && poisoned.contains("test mismatch"),
        "{poisoned}"
    );
    drop(dir);

    // The golden fixture's 16x16 blocks are outside the kernel contract:
    // Auto keeps Q8; an explicit request keeps native FP8 where it fits
    // (nowhere here), so every projection falls back per tensor.
    let golden =
        Qwen38NativeCheckpoint::open_synthetic(&super::golden_tests::fixture_dir()).unwrap();
    let policy = resolve(&golden, target, &PerformanceConfig::default());
    assert_eq!(policy.plan(), Qwen38ProjectionPlan::Q8);
    assert!(
        policy.execution_mode().contains("kernel contract"),
        "{}",
        policy.execution_mode()
    );
    let policy = resolve(&golden, target, &explicit);
    assert_eq!(
        policy.plan(),
        Qwen38ProjectionPlan::NativeFp8 { explicit: true }
    );
    assert_ne!(
        policy.projection_form(safetensors::Dtype::F8_E4M3, [128, 64], [16, 16]),
        Qwen38WeightForm::RawFp8
    );
}

/// A single projection whose resident FP8 form fails the load-time check
/// fails the load closed and poisons the policy for this checkpoint; the
/// next resolution keeps Q8_0, and that load succeeds.
#[test]
fn a_failed_single_check_fails_closed_and_the_retry_loads_q8() {
    let (_dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::F32;
    let _forced = force_cpu_native_fp8_for_tests();
    let policy = resolve(&checkpoint, target, &PerformanceConfig::default());
    let name = "model.language_model.layers.0.mlp.down_proj.weight";
    let loader = Qwen38WeightLoader::new(
        &checkpoint.tensors,
        &policy,
        &Device::Cpu,
        [128, 128],
        target,
    );
    // A wrong scale grid on an otherwise valid projection.
    let mut raw = checkpoint
        .tensors
        .materialize_block_fp8_raw(name, [256, 512], [128, 128], &Device::Cpu)
        .unwrap();
    raw.scales = (&raw.scales * 1.5).unwrap();
    let corrupted = Qwen38Projection::NativeFp8 {
        raw,
        kernel: Qwen38Fp8Kernel::Reference,
        prefill_gemm: false,
    };
    assert!(loader
        .fp8_self_check(&[(name, [256, 512])], &corrupted)
        .is_err());
    // Through the production loader: the corrupted projection fails its
    // check, the load fails closed, and the policy is poisoned.
    loader.corrupt_fp8_scales.set(true);
    let error = load_native_projection(&loader, name, [256, 512])
        .err()
        .expect("a corrupted projection must fail the load");
    assert!(error.to_string().contains("self_check_failed"), "{error}");
    assert!(
        error.to_string().contains("IZWI_QWEN38_FP8_PROJECTIONS"),
        "{error}"
    );
    let retry = resolve(&checkpoint, target, &PerformanceConfig::default());
    assert_eq!(retry.plan(), Qwen38ProjectionPlan::Q8);
    let model = load(&checkpoint, target, &retry).unwrap();
    assert!(fp8_projection_forms(&model)
        .iter()
        .all(|(native, _)| !native));
    assert!(model
        .layers
        .iter()
        .all(|layer| matches!(layer.mlp.down, Qwen38Projection::Quantized(_))));
}

/// Production resolution on CUDA (`IZWI_REQUIRE_CUDA_TEST_DEVICE=1` on the
/// H100): Auto picks native block FP8 with the vectorized GEMV, packs every
/// group, passes every load-time check, and the BF16 trunk tracks the CPU
/// expanded F32 load with the same greedy tokens.
#[cfg(feature = "cuda")]
#[test]
fn cuda_resolve_picks_native_fp8_on_auto_and_packed_groups_match_original_parts() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let (_dir, checkpoint) = fixture();
    let performance = PerformanceConfig::default();
    let target = ProjectionMaterialization::BF16;
    let policy = Qwen38ResidencyPolicy::resolve(
        &device,
        target,
        &performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    assert_eq!(
        policy.plan(),
        Qwen38ProjectionPlan::NativeFp8 { explicit: false },
        "{}",
        policy.execution_mode()
    );
    assert!(policy.fp8_vectorized());
    let native = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &device,
        target,
        &performance.cuda,
        &policy,
    )
    .unwrap();
    assert_eq!(summarize(native.fp8_packing())["fused_layers"], 8);
    let cpu = ProjectionMaterialization::F32;
    let expanded = load(
        &checkpoint,
        cpu,
        &resolve(&checkpoint, cpu, &expanded_cpu()),
    )
    .unwrap();
    let cpu_arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F32);
    let (reference, tokens) = fixture_rows(&expanded, &cpu_arena, 0, None);
    let arena = GoldenArena::new(&checkpoint.config, &device, DType::BF16);
    let (rows, _) = fixture_rows(&native, &arena, 0, Some(&tokens));
    // BF16 activations plus the deliberately reduced block-FP8 LM head.
    let mut agree = 0usize;
    for (index, (row, reference)) in rows.iter().zip(&reference).enumerate() {
        let error = relative_l2(row, reference);
        assert!(error < 5e-2, "row {index}: relative L2 {error}");
        agree += usize::from(argmax(row) == argmax(reference));
    }
    assert!(
        agree * 10 >= rows.len() * 9,
        "greedy agreement {agree}/{}",
        rows.len()
    );
    assert_eq!(native.fp8_lm_head_diagnostics()["backend"], "fused");
    let counters = crate::models::architectures::qwen38::telemetry::snapshot();
    assert!(counters.cuda_fp8_lm_head_calls_total > 0);
    assert!(counters.cuda_fp8_gemv_vectorized_calls_total > 0);
    assert!(counters.cuda_fp8_packed_projection_calls_total > 0);
}

/// A synthetic `[1024, 5120]` BF16 head quantizes and passes both checks on
/// CUDA through the production quantizer.
#[cfg(feature = "cuda")]
#[test]
fn cuda_synthetic_lm_head_passes_the_gate_and_the_device_check() {
    use crate::models::architectures::qwen38::mtp::tests::{
        fixture_dense_for_tests, write_fixture,
    };
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let dir = TestDir::new("cuda-fp8-head");
    write_fixture(
        dir.path(),
        &[fixture_dense_for_tests(
            "lm_head.weight",
            [1024, 5120],
            0.05,
        )],
        "model.safetensors",
    );
    let tensors = IndexedSafetensors::open(dir.path()).unwrap();
    let head = super::super::lm_head::quantize_lm_head_from_checkpoint(
        &tensors,
        "lm_head.weight",
        [1024, 5120],
        &device,
        DType::BF16,
        Qwen38Fp8Kernel::Vectorized,
    )
    .unwrap()
    .expect("a dense random head passes both checks");
    assert!(head.gate_relative_l2 < 0.08);
}

/// Production resolution on Metal: raw block FP8 with F16 activations,
/// packed groups, every projection kind checked against the original bytes;
/// decode rows run the FP8 GEMV and prefill spans the F16 expansion. The
/// trunk tracks the CPU expanded F32 load and the Metal expanded F16 load.
#[cfg(feature = "metal")]
#[test]
fn metal_native_fp8_tracks_cpu_f32_and_metal_expanded_loads() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_native_fp8_tracks_cpu_f32_and_metal_expanded_loads",
    ) else {
        return;
    };
    let _guard = super::super::fast_path::counters_lock();
    let (_dir, checkpoint) = fixture();
    let f16 = ProjectionMaterialization::F16;
    let performance = PerformanceConfig::default();
    let policy = Qwen38ResidencyPolicy::resolve(
        &device,
        f16,
        &performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    assert_eq!(
        policy.plan(),
        Qwen38ProjectionPlan::NativeFp8 { explicit: false },
        "{}",
        policy.fallback_reason()
    );
    let before = crate::models::architectures::qwen38::telemetry::snapshot();
    let native = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &device,
        f16,
        &performance.cuda,
        &policy,
    )
    .unwrap();
    assert_eq!(
        native.projection_representation(),
        Qwen38ProjectionRepresentation::NativeFp8WithDenseF16
    );
    assert_eq!(
        native.projection_representation().as_str(),
        "native_block_fp8_with_dense_f16"
    );
    let forms = fp8_projection_forms(&native);
    assert!(forms.iter().all(|(native, _)| *native), "{forms:?}");
    assert_eq!(summarize(native.fp8_packing())["fused_layers"], 8);
    let gemv = summarize(native.fp8_gemv_checks());
    assert_eq!(gemv["backend"], "fused", "{gemv}");
    // The LM head stays dense F16 on Metal.
    assert!(matches!(native.output, Qwen38Projection::Dense(_)));

    let mut expanded_performance = PerformanceConfig::default();
    expanded_performance.metal.projection_backend =
        crate::performance::MetalProjectionBackend::Expanded;
    let expanded_policy = Qwen38ResidencyPolicy::resolve(
        &device,
        f16,
        &expanded_performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    assert_eq!(expanded_policy.plan(), Qwen38ProjectionPlan::Expanded);
    let metal_expanded = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &device,
        f16,
        &performance.cuda,
        &expanded_policy,
    )
    .unwrap();
    assert_eq!(
        metal_expanded.projection_representation(),
        Qwen38ProjectionRepresentation::ExpandedF16
    );
    let cpu = ProjectionMaterialization::F32;
    let cpu_expanded = load(
        &checkpoint,
        cpu,
        &resolve(&checkpoint, cpu, &expanded_cpu()),
    )
    .unwrap();

    let cpu_arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F32);
    let (reference, tokens) = fixture_rows(&cpu_expanded, &cpu_arena, 0, None);
    let arena = GoldenArena::new(&checkpoint.config, &device, DType::F16);
    let (rows, _) = fixture_rows(&native, &arena, 0, Some(&tokens));
    let (expanded_rows, _) = fixture_rows(&metal_expanded, &arena, 1, Some(&tokens));
    let (mut worst_cpu, mut worst_expanded) = (0f32, 0f32);
    for (index, ((row, cpu), expanded)) in
        rows.iter().zip(&reference).zip(&expanded_rows).enumerate()
    {
        let against_cpu = relative_l2(row, cpu);
        let against_expanded = relative_l2(row, expanded);
        assert!(against_cpu <= 1e-2, "row {index}: vs CPU F32 {against_cpu}");
        assert!(
            against_expanded <= 5e-3,
            "row {index}: vs Metal expanded {against_expanded}"
        );
        assert_eq!(argmax(row), argmax(cpu), "row {index}: greedy token");
        worst_cpu = worst_cpu.max(against_cpu);
        worst_expanded = worst_expanded.max(against_expanded);
    }
    eprintln!(
        "qwen38 metal native FP8: worst relative L2 {worst_cpu:e} vs CPU F32, {worst_expanded:e} vs Metal expanded F16"
    );
    let after = crate::models::architectures::qwen38::telemetry::snapshot();
    assert!(after.metal_fp8_gemv_calls_total > before.metal_fp8_gemv_calls_total);
    assert!(
        after.metal_fp8_dequant_matmul_calls_total > before.metal_fp8_dequant_matmul_calls_total
    );
}

/// On Metal the load-time check sees a pack built in the wrong part order
/// and a wrong block scale (RED), and `IZWI_QWEN38_FP8_GEMV=legacy` routes
/// every projection through the F16 expansion with the same results.
#[cfg(feature = "metal")]
#[test]
fn metal_fp8_check_rejects_bad_packs_and_the_gemv_switch_reroutes() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_fp8_check_rejects_bad_packs_and_the_gemv_switch_reroutes",
    ) else {
        return;
    };
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (_dir, checkpoint) = fixture();
    let block = checkpoint.config.block_fp8.block_shape;
    let prefix = "model.language_model.layers.0.mlp";
    let (gate, up) = (
        format!("{prefix}.gate_proj.weight"),
        format!("{prefix}.up_proj.weight"),
    );
    let parts: [(&str, [usize; 2]); 2] = [(&gate, [512, 256]), (&up, [512, 256])];
    let pack = |order: &[(&str, [usize; 2])], kernel| Qwen38Projection::NativeFp8 {
        raw: checkpoint
            .tensors
            .materialize_block_fp8_raw_group(order, block, &device)
            .unwrap(),
        kernel,
        prefill_gemm: false,
    };
    let check = |projection: &Qwen38Projection| {
        fp8_projection_check(
            &checkpoint.tensors,
            &parts,
            block,
            projection,
            DType::F16,
            &device,
        )
    };
    for kernel in [Qwen38Fp8Kernel::Reference, Qwen38Fp8Kernel::DequantOnly] {
        check(&pack(&parts, kernel)).expect("the true pack passes");
        assert!(check(&pack(&[parts[1], parts[0]], kernel)).is_err());
        let Qwen38Projection::NativeFp8 {
            mut raw, kernel, ..
        } = pack(&parts, kernel)
        else {
            unreachable!()
        };
        let mut scales = raw
            .scales
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        scales[5] *= 3.0;
        raw.scales = Tensor::from_vec(scales, &raw.scale_shape, &device).unwrap();
        assert!(check(&Qwen38Projection::NativeFp8 {
            raw,
            kernel,
            prefill_gemm: false,
        })
        .is_err());
    }

    std::env::set_var(FP8_GEMV_ENV, "legacy");
    let policy = Qwen38ResidencyPolicy::resolve(
        &device,
        ProjectionMaterialization::F16,
        &PerformanceConfig::default(),
        &checkpoint.tensors,
        &checkpoint.config,
    );
    std::env::remove_var(FP8_GEMV_ENV);
    let model = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &device,
        ProjectionMaterialization::F16,
        &PerformanceConfig::default().cuda,
        &policy,
    )
    .unwrap();
    assert!(model.layers.iter().all(|layer| matches!(
        layer.mlp.down,
        Qwen38Projection::NativeFp8 {
            kernel: Qwen38Fp8Kernel::DequantOnly,
            ..
        }
    )));
    assert_eq!(summarize(model.fp8_gemv_checks())["backend"], "legacy");
}

/// The generic Metal FP8 GEMV at the 27B projection shapes (decode rows) and
/// the F16 expansion + matmul at prefill row counts, against a host
/// reference from the same bytes on sampled output rows.
#[cfg(feature = "metal")]
#[test]
fn metal_fp8_kernels_match_the_host_reference_at_qwen38_shapes() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_fp8_kernels_match_the_host_reference_at_qwen38_shapes",
    ) else {
        return;
    };
    let pattern: Vec<u8> = (0..4093u32)
        .map(|i| match (i.wrapping_mul(0x9E37_79B1) >> 11) as u8 {
            b if b & 0x7f == 0x7f => b ^ 0x22,
            b => b,
        })
        .collect();
    let run = |[n, k]: [usize; 2], rows: &[usize]| {
        let mut bytes = Vec::with_capacity(n * k);
        while bytes.len() < n * k {
            let take = pattern.len().min(n * k - bytes.len());
            bytes.extend_from_slice(&pattern[..take]);
        }
        let scale_shape = [n.div_ceil(128), k.div_ceil(128)];
        let scales: Vec<f32> = (0..scale_shape[0] * scale_shape[1])
            .map(|i| 2e-4 * (1.0 + (i % 17) as f32 * 0.21))
            .collect();
        let projection = Qwen38Projection::NativeFp8 {
            raw: super::super::native::RawBlockFp8Projection {
                weights: Tensor::from_slice(&bytes, (n, k), &device).unwrap(),
                scales: Tensor::from_slice(&scales, (scale_shape[0], scale_shape[1]), &device)
                    .unwrap(),
                shape: [n, k],
                scale_shape,
                block_shape: [128, 128],
            },
            kernel: Qwen38Fp8Kernel::Reference,
            prefill_gemm: false,
        };
        let sampled: Vec<usize> = (0..n).step_by(61).chain([n - 1]).collect();
        for &m in rows {
            let x = probe_activations(m, k, DType::F16, m as u64);
            let input = Tensor::from_vec(x.clone(), (1, m, k), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let output = projection
                .forward(&input)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            for (row, activations) in x.chunks(k).enumerate() {
                for &channel in &sampled {
                    actual.push(output[row * n + channel]);
                    expected.push(fp8_row_dot(
                        &bytes[channel * k..(channel + 1) * k],
                        &scales,
                        channel / 128,
                        scale_shape[1],
                        128,
                        activations,
                    ));
                }
            }
            compare_values(&format!("[{n}, {k}] m={m}"), &actual, &expected, 0.01, 0.02).unwrap();
        }
    };
    for shape in [
        [17408, 5120],
        [5120, 17408],
        [16384, 5120],
        [14336, 5120],
        [5120, 6144],
        [34816, 5120],
    ] {
        run(shape, &[1, 2, 3, 4]);
    }
    run([1024, 5120], &[9, 64, 256]);
}

/// The CPU's default residency: block-FP8 projections and the LM head as
/// Q8_0 (packed groups through the tiled converter), F32 elsewhere. Decode
/// rows run Candle's Q8_0 matmul (which also quantizes the activations) and
/// longer spans the tiled F32 path; logits stay within 6% of the F32
/// expansion with the same greedy tokens on at least 90% of rows.
#[test]
fn cpu_q8_residency_tracks_the_expanded_f32_load() {
    let _guard = super::super::fast_path::counters_lock();
    let (_dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::F32;
    let expanded = load(
        &checkpoint,
        target,
        &resolve(&checkpoint, target, &expanded_cpu()),
    )
    .unwrap();
    let policy = resolve(&checkpoint, target, &PerformanceConfig::default());
    assert_eq!(policy.plan(), Qwen38ProjectionPlan::Q8);
    assert_eq!(policy.execution_mode(), "q8_0_cpu_requantized");
    let before = crate::models::architectures::qwen38::telemetry::snapshot();
    let q8 = load(&checkpoint, target, &policy).unwrap();
    assert_eq!(
        q8.projection_representation(),
        Qwen38ProjectionRepresentation::PackedQ8WithDenseF32
    );
    assert_eq!(
        q8.projection_representation().as_str(),
        "q8_0_projections_with_dense_f32"
    );
    assert!(matches!(q8.output, Qwen38Projection::Quantized(_)));
    let packed = q8
        .layers
        .iter()
        .filter(|layer| {
            matches!(
                layer.mlp.gate_up,
                Qwen38ProjectionGroup::Packed {
                    projection: Qwen38Projection::Quantized(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(packed, 4);
    assert!(q8
        .layers
        .iter()
        .all(|layer| matches!(layer.mlp.down, Qwen38Projection::Quantized(_))));

    let arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F32);
    let (reference, tokens) = fixture_rows(&expanded, &arena, 0, None);
    let (rows, _) = fixture_rows(&q8, &arena, 1, Some(&tokens));
    let mut worst = 0f32;
    let mut agree = 0usize;
    for (index, (row, reference)) in rows.iter().zip(&reference).enumerate() {
        let error = relative_l2(row, reference);
        // Weights requantized to Q8_0 everywhere, plus Candle's Q8_0
        // activations on rows that take QMatMul (the 8-row prompt and every
        // decode step): measured 4.4% worst here (2.4% with F32 activations).
        assert!(error <= 6e-2, "row {index}: relative L2 {error}");
        worst = worst.max(error);
        agree += usize::from(argmax(row) == argmax(reference));
    }
    eprintln!(
        "qwen38 CPU Q8_0 vs expanded F32: worst relative L2 {worst:e}, greedy agreement {agree}/{}",
        rows.len()
    );
    assert!(
        agree * 10 >= rows.len() * 9,
        "greedy agreement {agree}/{}",
        rows.len()
    );
    let decode = crate::models::architectures::qwen38::telemetry::snapshot();
    assert!(decode.cpu_q8_matmul_calls_total > before.cpu_q8_matmul_calls_total);

    // A 16-row span takes the tiled F32 path (F32 activations): only the
    // weights' Q8_0 error remains (measured 2.4% worst here).
    let sequence: Vec<u32> = PROMPT
        .iter()
        .copied()
        .chain(tokens.iter().copied())
        .collect();
    let positions: Vec<[usize; 3]> = (0..sequence.len()).map(|p| [p, p, p]).collect();
    let span = |model: &Qwen38TextModel, session| {
        let mut state = model.new_state();
        let mut cache = arena.session(session);
        let output = model
            .prefill_token_ids_with_hidden_physical(
                &sequence, &positions, &mut state, &mut cache, false,
            )
            .unwrap()
            .expect("prefill output");
        model
            .project_target_hidden_span(&output.hidden_states)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let (q8_span, expanded_span) = (span(&q8, 2), span(&expanded, 3));
    let vocab = checkpoint.config.vocab_size;
    for (index, (row, reference)) in q8_span
        .chunks(vocab)
        .zip(expanded_span.chunks(vocab))
        .enumerate()
    {
        let error = relative_l2(row, reference);
        assert!(
            error <= 3e-2,
            "16-row span row {index}: relative L2 {error}"
        );
    }
    let after = crate::models::architectures::qwen38::telemetry::snapshot();
    assert!(after.cpu_q8_tiled_prefill_calls_total > decode.cpu_q8_tiled_prefill_calls_total);
}

/// The CPU policy table: Q8_0 needs K divisible by 32 (otherwise F32), the
/// head follows unless `IZWI_QWEN38_CPU_LM_HEAD=dense`, packing covers Q8_0
/// groups only, and `expanded` keeps today's F32 path.
#[test]
fn cpu_policy_table() {
    use crate::models::architectures::qwen38::residency::CPU_LM_HEAD_ENV;
    use safetensors::Dtype as SafeDType;
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let performance = PerformanceConfig::default();
    let q8 = Qwen38ResidencyPolicy::resolve_static(crate::backends::BackendKind::Cpu, &performance);
    let block = [128, 128];
    assert_eq!(
        q8.projection_form(SafeDType::F8_E4M3, [256, 256], block),
        Qwen38WeightForm::Q8_0
    );
    assert_eq!(
        q8.projection_form(SafeDType::F8_E4M3, [256, 48], [2, 2]),
        Qwen38WeightForm::Dense(DType::F32)
    );
    assert_eq!(
        q8.projection_form(SafeDType::BF16, [256, 256], block),
        Qwen38WeightForm::Dense(DType::F32)
    );
    assert_eq!(q8.lm_head_form([512, 256]), Qwen38WeightForm::Q8_0);
    assert_eq!(
        q8.lm_head_form([512, 4]),
        Qwen38WeightForm::Dense(DType::F32)
    );
    assert!(q8.packing_allowed(&[Qwen38WeightForm::Q8_0; 2]));
    assert!(!q8.packing_allowed(&[Qwen38WeightForm::Dense(DType::F32); 2]));
    std::env::set_var(CPU_LM_HEAD_ENV, "dense");
    let dense_head =
        Qwen38ResidencyPolicy::resolve_static(crate::backends::BackendKind::Cpu, &performance);
    std::env::remove_var(CPU_LM_HEAD_ENV);
    assert_eq!(
        dense_head.lm_head_form([512, 256]),
        Qwen38WeightForm::Dense(DType::F32)
    );
    let expanded =
        Qwen38ResidencyPolicy::resolve_static(crate::backends::BackendKind::Cpu, &expanded_cpu());
    assert_eq!(expanded.plan(), Qwen38ProjectionPlan::Expanded);
    assert_eq!(
        expanded.projection_form(SafeDType::F8_E4M3, [256, 256], block),
        Qwen38WeightForm::Dense(DType::F32)
    );
    assert_eq!(
        expanded.lm_head_form([512, 256]),
        Qwen38WeightForm::Dense(DType::F32)
    );
}
