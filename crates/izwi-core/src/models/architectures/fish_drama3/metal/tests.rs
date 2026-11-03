//! The Metal forms against the CPU reference on the synthetic checkpoint
//! (§10.1, with the published 128-dim DeltaNet heads so the fused DeltaNet
//! kernels run), and the decode step's batch invariance (§7.4).
#![cfg(feature = "metal")]

use candle_core::{DType, Device, Tensor};

use super::super::device::test_support::{
    decode_requests, decode_together, fused_against_eager, host, load, ok, options, pair, params,
    prompt_of, rows, trunk_caches, trunk_span_error, DecodeRequest,
};
use super::super::model::tests::write_synthetic_model_in;
use super::super::model::FishDrama3ModelOptions;
use super::super::residency::FishDrama3CpuPrecision;
use super::super::synthetic::SyntheticGeometry;
use super::super::trunk::FishDrama3StepRow;
use super::super::FishDrama3Model;
use crate::kernels::numeric_check::{relative_l2, xorshift_values};
use crate::models::shared::attention::physical::test_physical_caches_for_model_layers_on;

/// The trunk's final hidden after a span or a step, relative L2 against
/// the CPU's F32 weights [measured on an M1 Pro: span 2.0e-2, steps
/// ≤ 6.3e-3, the L0 geometry's span 6.8e-3, the expert loop 2.5e-2]: F16
/// activations round each layer's inputs and outputs to 11 bits (about
/// 5e-4), and the synthetic trunk is untrained and deep for its width (the
/// CPU tests see a 1% perturbation grow to 5-20% by its last layer). With
/// F32 activations the same kernels measure 2.3e-6, so every layout and
/// convention is exact and only F16 rounding remains.
const TRUNK_TOLERANCE: f64 = 5e-2;
/// Each tower layer's output and the codebook heads' logits [6.3e-3 and
/// 1.4e-3]: 13 layers of the same F16 rounding over unit-RMS inputs.
const TOWER_TOLERANCE: f64 = 2e-2;
/// The compact head over F32 hidden rows [2.9e-4]: one F16 projection.
const HEAD_TOLERANCE: f64 = 2e-3;
/// F32 activations throughout (`IZWI_FISH_DRAMA3_DTYPE=f32`) [2.3e-6]: only
/// F32 summation order differs from the CPU.
const F32_TOLERANCE: f64 = 2e-5;

/// The trunk on Metal against the CPU's F32 weights: a 20-row span, then
/// three steps, on the fused DeltaNet kernels.
#[test]
fn metal_trunk_tracks_the_cpu_reference() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("metal_trunk_tracks_the_cpu_reference")
    else {
        return;
    };
    let (_synthetic, cpu, metal) = pair(SyntheticGeometry::KernelHeads, 5, &device);
    let prompt = rows(&cpu, 23, 9);
    let width = cpu.config().tower.frame_width();
    let (span, steps) = prompt.split_at(20 * width);
    let embed_cpu = cpu.embedding().embed_tensor(span, &Device::Cpu).unwrap();
    let embed_metal = metal.embedding().embed_tensor(span, &device).unwrap();
    // The F16 tables hold every BF16 value exactly: text rows are equal,
    // and the semantic rows' `/ sqrt(2)` rounds within an ulp.
    let (a, b) = (host(&embed_cpu), host(&embed_metal));
    let h = cpu.config().trunk.hidden;
    for (t, (a, b)) in a.chunks_exact(h).zip(b.chunks_exact(h)).enumerate() {
        if t % 3 == 0 {
            assert_eq!(a, b, "text row {t}");
        } else {
            assert!(relative_l2(b, a) < 2e-7, "semantic row {t}");
        }
    }
    let mut cpu_state = cpu.trunk().new_state().unwrap();
    let mut metal_state = metal.trunk().new_state().unwrap();
    let mut cpu_cache = trunk_caches(&cpu, &Device::Cpu, DType::F32, 1).remove(0);
    let mut metal_cache = trunk_caches(&metal, &device, metal.kv_dtype(), 1).remove(0);
    let a = cpu
        .trunk()
        .forward_span(&embed_cpu, 0, &mut cpu_state, &mut cpu_cache)
        .unwrap();
    let spans = super::super::device::fish_drama3_recurrent_deltanet_spans();
    let b = metal
        .trunk()
        .forward_span(&embed_metal, 0, &mut metal_state, &mut metal_cache)
        .unwrap();
    // Every DeltaNet layer took the span kernels.
    let linear = cpu.config().trunk.linear_attention_layers().len() as u64;
    assert!(super::super::device::fish_drama3_recurrent_deltanet_spans() - spans >= linear);
    let error = relative_l2(&host(&b), &host(&a));
    eprintln!("trunk span: {error:.3e}");
    assert!(error < TRUNK_TOLERANCE, "span: {error:.3e}");
    for (t, row) in steps.chunks_exact(width).enumerate() {
        let x_cpu = cpu.embedding().embed_tensor(row, &Device::Cpu).unwrap();
        let x_metal = metal.embedding().embed_tensor(row, &device).unwrap();
        let a = cpu
            .trunk()
            .forward_step(
                &x_cpu,
                &mut [FishDrama3StepRow {
                    state: &mut cpu_state,
                    cache: &mut cpu_cache,
                }],
            )
            .unwrap();
        let b = metal
            .trunk()
            .forward_step(
                &x_metal,
                &mut [FishDrama3StepRow {
                    state: &mut metal_state,
                    cache: &mut metal_cache,
                }],
            )
            .unwrap();
        let error = relative_l2(&host(&b), &host(&a));
        eprintln!("trunk step {t}: {error:.3e}");
        assert!(error < TRUNK_TOLERANCE, "step {t}: {error:.3e}");
    }
}

/// The trunk on the L0 geometry, whose 32-dim DeltaNet heads take the
/// Candle recurrence instead of `gdn_decode`.
#[test]
fn metal_runs_the_l0_geometry_through_the_candle_recurrence() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_runs_the_l0_geometry_through_the_candle_recurrence",
    ) else {
        return;
    };
    let (_synthetic, cpu, metal) = pair(SyntheticGeometry::L0, 5, &device);
    let error = trunk_span_error(&cpu, &metal, &device, 20);
    eprintln!("L0 trunk span: {error:.3e}");
    assert!(error < TRUNK_TOLERANCE, "{error:.3e}");
}

/// The tower teacher-forced over a span, then a forced step, and the
/// compact head, on Metal against the CPU's F32 weights.
#[test]
fn metal_tower_and_heads_track_the_cpu_reference() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("metal_tower_and_heads_track_the_cpu_reference")
    else {
        return;
    };
    use super::super::tower::FishDrama3TowerForcedRow;
    let (_synthetic, cpu, metal) = pair(SyntheticGeometry::KernelHeads, 5, &device);
    let config = cpu.config();
    let (d, width) = (config.tower.dim, config.tower.frame_width());
    let n = 10;
    let frames = rows(&cpu, n + 2, 4);
    // Final-norm-like hidden rows: unit RMS.
    let hidden = xorshift_values((n + 1) * d, 8, 1.7);
    let a = &config.tower.attention;
    let caches = |model: &FishDrama3Model, device: &Device, dtype| {
        test_physical_caches_for_model_layers_on(
            device,
            dtype,
            model.tower().attention_layers(),
            a.kv_heads,
            a.head_dim,
            64,
            1,
        )
        .remove(0)
    };
    let run = |model: &FishDrama3Model, device: &Device, dtype| {
        let mut cache = caches(model, device, dtype);
        let span = Tensor::from_slice(&hidden[..n * d], (n, d), device).unwrap();
        let trace = model
            .tower()
            .forward_span_traced(
                model.embedding(),
                &span,
                &frames[..(n + 1) * width],
                0,
                &mut cache,
            )
            .unwrap();
        let step = Tensor::from_slice(&hidden[n * d..], (1, d), device).unwrap();
        let logits = model
            .tower()
            .forward_step_forced(
                model.embedding(),
                &step,
                &mut [FishDrama3TowerForcedRow {
                    cache: &mut cache,
                    input: &frames[n * width..(n + 1) * width],
                    label: &frames[(n + 1) * width..],
                }],
            )
            .unwrap();
        let head = host(&model.head().logits(&span).unwrap());
        (trace, logits, head)
    };
    let (trace_cpu, logits_cpu, head_cpu) = run(&cpu, &Device::Cpu, DType::F32);
    let (trace_metal, logits_metal, head_metal) = run(&metal, &device, metal.kv_dtype());
    let worst = trace_cpu
        .iter()
        .zip(&trace_metal)
        .map(|(a, b)| relative_l2(b, a))
        .fold(0f64, f64::max);
    let heads = relative_l2(&logits_metal, &logits_cpu);
    let head = relative_l2(&head_metal, &head_cpu);
    eprintln!("tower layers: {worst:.3e}, codebook heads: {heads:.3e}, compact head: {head:.3e}");
    assert!(worst < TOWER_TOLERANCE, "tower layers: {worst:.3e}");
    assert!(heads < TOWER_TOLERANCE, "codebook heads: {heads:.3e}");
    assert!(head < HEAD_TOLERANCE, "compact head: {head:.3e}");
}

/// The fused trunk and tower (residual norms, q/k norm + RoPE, packed
/// SwiGLU) against the eager chain on one load: the kernels round where the
/// chain does, so only their F32 reduction order differs.
#[test]
fn metal_fused_forwards_track_the_eager_chain() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("metal_fused_forwards_track_the_eager_chain")
    else {
        return;
    };
    let (trunk, tower) = fused_against_eager(&device, 5);
    eprintln!("fused against eager: trunk {trunk:.3e}, tower {tower:.3e}");
    assert!(trunk < TRUNK_TOLERANCE, "trunk: {trunk:.3e}");
    assert!(tower < TOWER_TOLERANCE, "tower: {tower:.3e}");
}

/// §7.4 at Metal widths 1 and 2: a request's frames and `h_t` are
/// bit-identical alone and beside a longer and a shorter request, at both
/// row positions, through the whole decode (compact head, device main draw,
/// tower with device-sampled codes, trunk step). RED: a step whose
/// reductions split by batch size (a split-K GEMM) changes the row's bits.
#[test]
fn metal_decode_rows_are_bit_identical_alone_and_together() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_decode_rows_are_bit_identical_alone_and_together",
    ) else {
        return;
    };
    let synthetic = write_synthetic_model_in(7, SyntheticGeometry::KernelHeads, |_| {});
    let model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let row = prompt_of(&model, &rows(&model, 9, 1));
    let longer = prompt_of(&model, &rows(&model, 21, 2));
    let shorter = prompt_of(&model, &rows(&model, 4, 3));
    let rounds = 5;
    let alone = decode_together(&model, &[(&row, 11)], rounds, false).remove(0);
    assert!(alone.0[0].len() >= 2, "frames {:?}", alone.0[0]);
    for (other, seed) in [(&longer, 12), (&shorter, 13)] {
        let first = decode_together(&model, &[(&row, 11), (other, seed)], rounds, false);
        assert_eq!(first[0], alone, "beside a {}-row prompt, first", other.rows);
        let second = decode_together(&model, &[(other, seed), (&row, 11)], rounds, false);
        assert_eq!(
            second[1], alone,
            "beside a {}-row prompt, second",
            other.rows
        );
    }
    let red = decode_together(&model, &[(&row, 11), (&longer, 12)], rounds, true);
    assert_ne!(
        red[0].1, alone.1,
        "a batch-split reduction must change the row's bits"
    );
}

/// One readback a frame on the fused path (every row's main and RAS picks,
/// chosen candidate, 9 head codes and flags), two on the two-readback path
/// (the picks, then the codes); every frame counted as sampled on the
/// device.
#[test]
fn metal_decode_reads_back_once_a_frame() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("metal_decode_reads_back_once_a_frame")
    else {
        return;
    };
    use super::super::device::sampling::{
        fish_drama3_resident_frames, fish_drama3_resident_host_fallbacks,
        fish_drama3_resident_readbacks,
    };
    let synthetic = write_synthetic_model_in(7, SyntheticGeometry::KernelHeads, |_| {});
    let mut model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let a = prompt_of(&model, &rows(&model, 9, 1));
    let b = prompt_of(&model, &rows(&model, 12, 2));
    for (two, readbacks_a_frame) in [(false, 1), (true, 2)] {
        model.two_readbacks = two;
        let mut states = [(&a, 21u64), (&b, 22)]
            .iter()
            .map(|(prompt, seed)| {
                let mut state = model
                    .new_private_request((*prompt).clone(), params(*seed, 16))
                    .unwrap();
                model.run_prefill(&mut state, &ok).unwrap();
                state
            })
            .collect::<Vec<_>>();
        // The first frame masks eos, so both rows make it.
        let (readbacks, frames, fallbacks) = (
            fish_drama3_resident_readbacks(),
            fish_drama3_resident_frames(),
            fish_drama3_resident_host_fallbacks(),
        );
        let counters = super::super::device::fast_path_counters();
        let mut refs = states.iter_mut().collect::<Vec<_>>();
        model.decode_round(&mut refs).unwrap();
        // The step's uploads and its three timed sections are counted.
        let after = super::super::device::fast_path_counters();
        for key in [
            "step_uploads_total",
            "step_main_draw_ms_total",
            "step_tower_ms_total",
            "step_trunk_ms_total",
        ] {
            assert!(
                after[key].as_f64().unwrap() > counters[key].as_f64().unwrap(),
                "{key}: {after}"
            );
        }
        assert_eq!(
            fish_drama3_resident_readbacks() - readbacks,
            readbacks_a_frame,
            "two readbacks: {two}"
        );
        assert_eq!(fish_drama3_resident_frames() - frames, 2);
        assert_eq!(fish_drama3_resident_host_fallbacks(), fallbacks);
        assert!(states.iter().all(|state| state.frames() == 1));
    }
    let summary = model.diagnostics_summary();
    assert!(
        summary["resident_sampling"]["frames_total"]
            .as_u64()
            .unwrap()
            >= 4
    );
}

/// A fused trunk step through its pieces (the stretches the CUDA graphs
/// capture, and the attention mixers between them) makes the whole fused
/// step's frames bit for bit.
#[test]
fn metal_trunk_steps_through_pieces_as_the_whole_fused_step() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_trunk_steps_through_pieces_as_the_whole_fused_step",
    ) else {
        return;
    };
    let synthetic = write_synthetic_model_in(7, SyntheticGeometry::KernelHeads, |_| {});
    let mut model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let a = prompt_of(&model, &rows(&model, 9, 1));
    let b = prompt_of(&model, &rows(&model, 12, 2));
    let pieces = decode_together(&model, &[(&a, 11), (&b, 12)], 4, false);
    model.trunk_mut().device_trunk_mut().unwrap().steps_whole = true;
    let whole = decode_together(&model, &[(&a, 11), (&b, 12)], 4, false);
    assert_eq!(pieces, whole);
}

/// A device-drawn tower step through its pieces (each layer's attention and
/// the stretches between them, which the CUDA graphs capture) makes the
/// layer-by-layer step's frames bit for bit.
#[test]
fn metal_tower_steps_through_pieces_as_the_layer_by_layer_step() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_tower_steps_through_pieces_as_the_layer_by_layer_step",
    ) else {
        return;
    };
    let synthetic = write_synthetic_model_in(7, SyntheticGeometry::KernelHeads, |_| {});
    let mut model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let a = prompt_of(&model, &rows(&model, 9, 1));
    let b = prompt_of(&model, &rows(&model, 12, 2));
    let pieces = decode_together(&model, &[(&a, 11), (&b, 12)], 4, false);
    model.tower_mut().device_tower_mut().unwrap().steps_whole = true;
    let whole = decode_together(&model, &[(&a, 11), (&b, 12)], 4, false);
    assert_eq!(pieces, whole);
}

/// The one-readback frame makes the two-readback path's frames bit for bit:
/// a request that keeps going, one that reaches its frame limit, and one
/// that draws eos (whose speculative tower and trunk step is aborted), alone
/// in no round and together in every one.
#[test]
fn metal_one_readback_frames_are_the_two_readback_frames() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_one_readback_frames_are_the_two_readback_frames",
    ) else {
        return;
    };
    use super::super::device::sampling::fish_drama3_resident_readbacks;
    use super::super::model::FishDrama3StopReason;
    let synthetic = write_synthetic_model_in(7, SyntheticGeometry::KernelHeads, |_| {});
    let mut model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let (a, b, c) = (
        prompt_of(&model, &rows(&model, 9, 1)),
        prompt_of(&model, &rows(&model, 12, 2)),
        prompt_of(&model, &rows(&model, 5, 3)),
    );
    let requests = [
        DecodeRequest {
            prompt: &a,
            seed: 11,
            max_frames: 16,
            eos_at: None,
        },
        DecodeRequest {
            prompt: &b,
            seed: 12,
            max_frames: 3,
            eos_at: None,
        },
        DecodeRequest {
            prompt: &c,
            seed: 13,
            max_frames: 16,
            eos_at: Some(2),
        },
    ];
    let rounds = 6;
    let before = fish_drama3_resident_readbacks();
    let one = decode_requests(&model, &requests, rounds);
    let one_readbacks = fish_drama3_resident_readbacks() - before;
    model.two_readbacks = true;
    let before = fish_drama3_resident_readbacks();
    let two = decode_requests(&model, &requests, rounds);
    let two_readbacks = fish_drama3_resident_readbacks() - before;
    assert_eq!(one, two);
    let stops = one.iter().map(|(_, _, stop)| *stop).collect::<Vec<_>>();
    assert_eq!(
        stops,
        [
            None,
            Some(FishDrama3StopReason::FrameLimit),
            Some(FishDrama3StopReason::EndOfSpeech)
        ]
    );
    assert_eq!(one[0].0[0].len(), rounds);
    assert_eq!(one[1].0[0].len(), 3);
    assert_eq!(one[2].0[0].len(), 2);
    // Request a goes on every round: one readback a round, against two.
    assert_eq!(
        (one_readbacks, two_readbacks),
        (rounds as u64, 2 * rounds as u64)
    );
}

/// Every check of a healthy load passes: no component leaves F16's range,
/// the KV stays F16, the FP8 GEMV matches the original tensors and the RoPE
/// tables match F64 at every checked position.
#[test]
fn a_healthy_metal_load_passes_every_check() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("a_healthy_metal_load_passes_every_check")
    else {
        return;
    };
    let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |_| {});
    let model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let checks = model.metal_checks().expect("a Metal load runs its checks");
    let config = model.config();
    // Every trunk layer, the final norm, the head, every tower layer, the
    // codebook heads.
    let tower_layers = config.tower.shared_layers + config.tower.residual_layers();
    assert_eq!(
        checks.components.len(),
        config.trunk.layers.len() + 2 + tower_layers + 1
    );
    assert_eq!(checks.f16_range_breaches, 0, "{:?}", checks.components);
    assert!(checks
        .components
        .iter()
        .all(|c| !c.breached && c.non_finite == 0));
    assert_eq!(model.kv_dtype(), DType::F16);
    assert_eq!(checks.kv_dtype, "F16");
    assert!(checks.moe_loop_layers.is_empty());
    // Two layers, two experts, three projection kinds.
    assert_eq!(checks.fp8_checks.len(), 12);
    assert_eq!(checks.fp8_gemv_breaches, 0);
    for (name, error) in &checks.fp8_checks {
        assert!(*error < 2e-3, "{name}: {error:.3e}");
    }
    let worst_fp8 = checks
        .fp8_checks
        .iter()
        .map(|(_, e)| *e)
        .fold(0f64, f64::max);
    let worst_rope = checks
        .rope_checks
        .iter()
        .map(|(.., e)| *e)
        .fold(0f64, f64::max);
    let peak = checks
        .components
        .iter()
        .map(|c| c.max_abs)
        .fold(0f32, f32::max);
    eprintln!("FP8 GEMV: {worst_fp8:.3e}, RoPE: {worst_rope:.3e}, probe peak |x|: {peak:.3e}");
    assert_eq!(checks.rope_checks.len(), 6);
    assert_eq!(checks.rope_table_breaches, 0);
    for (layout, position, error) in &checks.rope_checks {
        assert!(*error < 1e-6, "{layout} at {position}: {error:.3e}");
    }
    let diagnostics = &model.diagnostics_summary()["metal_checks"];
    for key in [
        "f16_range_breaches",
        "fp8_gemv_breaches",
        "rope_table_breaches",
        "kv_dtype",
    ] {
        assert!(!diagnostics[key].is_null(), "{key}");
    }
}

/// A layer whose shared expert writes past F16's range breaches the probe:
/// it and every later layer (whose residual it lifted) run F32 activations
/// with the host expert loop, the earlier layers stay F16, and the load
/// still tracks the CPU reference.
#[test]
fn a_layer_past_f16_range_runs_in_f32_with_the_host_expert_loop() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "a_layer_past_f16_range_runs_in_f32_with_the_host_expert_loop",
    ) else {
        return;
    };
    let target = format!(
        "{}.mlp.shared_expert.down_proj.weight",
        super::super::checkpoint::trunk_layer(5)
    );
    let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |tensors| {
        let (_, _, _, bytes) = tensors
            .iter_mut()
            .find(|(name, ..)| *name == target)
            .expect("the shared expert's down projection");
        for chunk in bytes.as_chunks_mut::<2>().0 {
            let value = half::bf16::from_le_bytes(*chunk).to_f32() * 1e5;
            *chunk = half::bf16::from_f32(value).to_le_bytes();
        }
    });
    let metal = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let checks = metal.metal_checks().unwrap();
    let breached = checks
        .components
        .iter()
        .filter(|c| c.breached)
        .map(|c| c.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        breached[..3],
        ["trunk.5", "trunk.6", "trunk.7"],
        "{breached:?}"
    );
    assert_eq!(checks.f16_range_breaches as usize, breached.len());
    assert_eq!(checks.moe_loop_layers, [5, 6, 7]);
    let forms = metal.trunk().device_trunk().unwrap().layer_forms();
    assert!(forms[..5].iter().all(|(act, _)| *act == DType::F16));
    assert!(forms[5..].iter().all(|(act, _)| *act == DType::F32));
    let cpu = load(
        &synthetic,
        &Device::Cpu,
        options(FishDrama3CpuPrecision::F32),
    );
    let error = trunk_span_error(&cpu, &metal, &device, 12);
    eprintln!("breached trunk: {error:.3e}");
    assert!(error < TRUNK_TOLERANCE, "{error:.3e}");
}

/// `IZWI_FISH_DRAMA3_DTYPE=f32` runs every component in F32 (the fused
/// experts give way to the host loop, the KV is F32) and probes nothing;
/// `IZWI_FISH_DRAMA3_MOE_BACKEND=loop` moves only the experts.
#[test]
fn the_f32_override_and_the_expert_loop_track_the_cpu_reference() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "the_f32_override_and_the_expert_loop_track_the_cpu_reference",
    ) else {
        return;
    };
    let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |_| {});
    let cpu = load(
        &synthetic,
        &Device::Cpu,
        options(FishDrama3CpuPrecision::F32),
    );
    let f32 = load(
        &synthetic,
        &device,
        FishDrama3ModelOptions {
            activations: Some(DType::F32),
            ..options(FishDrama3CpuPrecision::Q8_0)
        },
    );
    let checks = f32.metal_checks().unwrap();
    assert!(checks.components.is_empty());
    assert_eq!(f32.kv_dtype(), DType::F32);
    assert_eq!(
        checks.moe_loop_layers.len(),
        cpu.config().trunk.layers.len()
    );
    let error = trunk_span_error(&cpu, &f32, &device, 12);
    eprintln!("F32 activations: {error:.3e}");
    assert!(error < F32_TOLERANCE, "{error:.3e}");
    let mut loop_options = options(FishDrama3CpuPrecision::Q8_0);
    loop_options.trunk.moe_backend = super::super::device::trunk::FishDrama3MoeBackend::Loop;
    let looped = load(&synthetic, &device, loop_options);
    assert_eq!(
        looped.metal_checks().unwrap().moe_loop_layers.len(),
        cpu.config().trunk.layers.len()
    );
    assert_eq!(looped.kv_dtype(), DType::F16);
    let error = trunk_span_error(&cpu, &looped, &device, 12);
    eprintln!("expert loop: {error:.3e}");
    assert!(error < TRUNK_TOLERANCE, "{error:.3e}");
}

/// RED for the two table checks: a scale off by 2 misses the FP8 check by
/// far more than its tolerance, and an F32 angle product (device
/// trigonometry's error) misses the RoPE check at position 262,143.
#[test]
fn the_fp8_and_rope_checks_catch_a_wrong_scale_and_f32_angles() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "the_fp8_and_rope_checks_catch_a_wrong_scale_and_f32_angles",
    ) else {
        return;
    };
    use super::super::precision::{FP8_GEMV_TOLERANCE, ROPE_TOLERANCE};
    let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |_| {});
    let checkpoint = synthetic.open();
    let config = &synthetic.config;
    let (h, inter) = (config.trunk.hidden, config.trunk.moe.intermediate);
    let module = format!("{}.gate_proj", super::super::checkpoint::expert(0, 3));
    let original = checkpoint.fp8_block_weight(&module, inter, h, 128).unwrap();
    let weights = Tensor::from_slice(original.codes, (inter, h), &device).unwrap();
    let scales = Tensor::from_slice(&original.scales, (inter / 128, h / 128), &device).unwrap();
    let right =
        super::super::device::checks::fp8_error(&weights, &scales, &original, DType::F16).unwrap();
    assert!(right < FP8_GEMV_TOLERANCE / 5.0, "{right:.3e}");
    let doubled = (&scales * 2.0).unwrap();
    let wrong =
        super::super::device::checks::fp8_error(&weights, &doubled, &original, DType::F16).unwrap();
    assert!(wrong > 50.0 * FP8_GEMV_TOLERANCE, "{wrong:.3e}");
    // The angle `position * inv_freq` formed in F32, as a device would, at
    // the published geometry (64 of 256 dims rotated).
    let attention = super::super::config::tests::published().trunk.attention;
    let x = xorshift_values(attention.heads * attention.head_dim, 1, 1.0);
    let position = 262_143usize;
    let mut f32_angles = x.clone();
    let rotary = attention.rotary_dims;
    for head in f32_angles.chunks_exact_mut(attention.head_dim) {
        for i in 0..rotary / 2 {
            let inv = attention.rope_theta.powf(-((2 * i) as f64) / rotary as f64) as f32;
            let angle = position as f32 * inv;
            let (a, b) = (i, i + rotary / 2);
            let (x1, x2) = (head[a], head[b]);
            head[a] = x1 * angle.cos() - x2 * angle.sin();
            head[b] = x2 * angle.cos() + x1 * angle.sin();
        }
    }
    let expected = super::super::device::checks::rope_reference(&x, position, &attention);
    let error = f32_angles
        .iter()
        .zip(&expected)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0f64, f64::max);
    assert!(error > 100.0 * ROPE_TOLERANCE, "{error:.3e}");
}
