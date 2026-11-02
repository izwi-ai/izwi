//! The CUDA forms against the CPU reference on the synthetic checkpoint
//! (§10.1, with the published 128-dim DeltaNet heads so the fused DeltaNet
//! kernels run), the decode step's batch invariance (§7.4) at widths 1, 2,
//! 4 and 8, resident sampling and the load checks. They skip without a CUDA
//! device (`IZWI_REQUIRE_CUDA_TEST_DEVICE=1` makes a missing one fail) and
//! have only been compiled so far: every bound below is an estimate from the
//! Metal measurements, to be set from the first run on a GPU.

use candle_core::{DType, Device, Tensor};

use super::super::device::test_support::{
    decode_together, host, load, ok, options, pair, params, prompt_of, rows, trunk_caches,
    trunk_span_error,
};
use super::super::model::tests::write_synthetic_model_in;
use super::super::residency::FishDrama3CpuPrecision;
use super::super::synthetic::SyntheticGeometry;
use super::super::trunk::FishDrama3StepRow;
use super::super::FishDrama3Model;
use crate::kernels::numeric_check::{relative_l2, xorshift_values};
use crate::models::shared::attention::physical::test_physical_caches_for_model_layers_on;

/// The trunk's final hidden after a span or a step, relative L2 against
/// the CPU's F32 weights. Metal's F16 activations measure 2.0e-2 on the span
/// and up to 6.3e-3 on steps; BF16 rounds four times coarser and the
/// residual rounds too, so about four times that. Uncalibrated.
const TRUNK_TOLERANCE: f64 = 2e-1;
/// Each tower layer's output and the codebook heads' logits (Metal: 6.3e-3
/// and 1.4e-3), at four times Metal's bound. Uncalibrated.
const TOWER_TOLERANCE: f64 = 8e-2;
/// The compact head over F32 hidden rows (Metal: 2.9e-4): one BF16
/// projection. Uncalibrated.
const HEAD_TOLERANCE: f64 = 1e-2;

fn device_or_skip() -> Option<Device> {
    crate::kernels::cuda::cuda_test_device()
}

/// The trunk on CUDA against the CPU's F32 weights: a 40-row span (the
/// chunked DeltaNet prefill), a 20-row span (the per-row kernels), then
/// three steps through the row-invariant GEMV.
#[test]
fn cuda_trunk_tracks_the_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let (_synthetic, cpu, gpu) = pair(SyntheticGeometry::KernelHeads, 5, &device);
    let prompt = rows(&cpu, 63, 9);
    let width = cpu.config().tower.frame_width();
    let (spans, steps) = prompt.split_at(60 * width);
    let mut cpu_state = cpu.trunk().new_state().unwrap();
    let mut gpu_state = gpu.trunk().new_state().unwrap();
    let mut cpu_cache = trunk_caches(&cpu, &Device::Cpu, DType::F32, 1).remove(0);
    let mut gpu_cache = trunk_caches(&gpu, &device, gpu.kv_dtype(), 1).remove(0);
    let chunked = super::super::device::fish_drama3_chunked_deltanet_spans();
    for (start, end) in [(0usize, 40usize), (40, 60)] {
        let span = &spans[start * width..end * width];
        let a = cpu
            .trunk()
            .forward_span(
                &cpu.embedding().embed_tensor(span, &Device::Cpu).unwrap(),
                start,
                &mut cpu_state,
                &mut cpu_cache,
            )
            .unwrap();
        let b = gpu
            .trunk()
            .forward_span(
                &gpu.embedding().embed_tensor(span, &device).unwrap(),
                start,
                &mut gpu_state,
                &mut gpu_cache,
            )
            .unwrap();
        let error = relative_l2(&host(&b), &host(&a));
        eprintln!("trunk span {start}..{end}: {error:.3e}");
        assert!(error < TRUNK_TOLERANCE, "span {start}..{end}: {error:.3e}");
    }
    // The 40-row span took the chunked prefill in every DeltaNet layer.
    let linear = cpu.config().trunk.linear_attention_layers().len() as u64;
    assert!(super::super::device::fish_drama3_chunked_deltanet_spans() - chunked >= linear);
    for (t, row) in steps.chunks_exact(width).enumerate() {
        let a = cpu
            .trunk()
            .forward_step(
                &cpu.embedding().embed_tensor(row, &Device::Cpu).unwrap(),
                &mut [FishDrama3StepRow {
                    state: &mut cpu_state,
                    cache: &mut cpu_cache,
                }],
            )
            .unwrap();
        let b = gpu
            .trunk()
            .forward_step(
                &gpu.embedding().embed_tensor(row, &device).unwrap(),
                &mut [FishDrama3StepRow {
                    state: &mut gpu_state,
                    cache: &mut gpu_cache,
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
fn cuda_runs_the_l0_geometry() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let (_synthetic, cpu, gpu) = pair(SyntheticGeometry::L0, 5, &device);
    let error = trunk_span_error(&cpu, &gpu, &device, 20);
    eprintln!("L0 trunk span: {error:.3e}");
    assert!(error < TRUNK_TOLERANCE, "{error:.3e}");
}

/// The tower teacher-forced over a span, then a forced step, and the
/// compact head, on CUDA against the CPU's F32 weights.
#[test]
fn cuda_tower_and_heads_track_the_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use super::super::tower::FishDrama3TowerForcedRow;
    let (_synthetic, cpu, gpu) = pair(SyntheticGeometry::KernelHeads, 5, &device);
    let config = cpu.config();
    let (d, width) = (config.tower.dim, config.tower.frame_width());
    let n = 10;
    let frames = rows(&cpu, n + 2, 4);
    let hidden = xorshift_values((n + 1) * d, 8, 1.7);
    let a = &config.tower.attention;
    let run = |model: &FishDrama3Model, device: &Device, dtype| {
        let mut cache = test_physical_caches_for_model_layers_on(
            device,
            dtype,
            model.tower().attention_layers(),
            a.kv_heads,
            a.head_dim,
            64,
            1,
        )
        .remove(0);
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
    let (trace_gpu, logits_gpu, head_gpu) = run(&gpu, &device, gpu.kv_dtype());
    let worst = trace_cpu
        .iter()
        .zip(&trace_gpu)
        .map(|(a, b)| relative_l2(b, a))
        .fold(0f64, f64::max);
    let heads = relative_l2(&logits_gpu, &logits_cpu);
    let head = relative_l2(&head_gpu, &head_cpu);
    eprintln!("tower layers: {worst:.3e}, codebook heads: {heads:.3e}, compact head: {head:.3e}");
    assert!(worst < TOWER_TOLERANCE, "tower layers: {worst:.3e}");
    assert!(heads < TOWER_TOLERANCE, "codebook heads: {heads:.3e}");
    assert!(head < HEAD_TOLERANCE, "compact head: {head:.3e}");
}

/// §7.4 at CUDA widths 1, 2, 4 and 8: a request's frames and `h_t` are
/// bit-identical alone and at every row position of a batch of longer and
/// shorter requests, through the whole decode (the row-invariant GEMV in
/// every projection, router, tower and head, the fused MoE, the per-request
/// paged decode, device sampling). RED: a step whose reductions split by
/// batch size (a split-K GEMM) changes the row's bits.
#[test]
fn cuda_decode_rows_are_bit_identical_alone_and_together() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let synthetic = write_synthetic_model_in(7, SyntheticGeometry::KernelHeads, |_| {});
    let model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    assert_eq!(
        model.cuda_checks().unwrap().row_gemv,
        "verified",
        "the batch-invariance test runs the multi-row GEMV"
    );
    let subject = prompt_of(&model, &rows(&model, 9, 1));
    let others = (0..7u64)
        .map(|i| {
            prompt_of(
                &model,
                &rows(&model, [4, 21, 13, 2, 30, 7, 17][i as usize], 2 + i),
            )
        })
        .collect::<Vec<_>>();
    let rounds = 5;
    let alone = decode_together(&model, &[(&subject, 11)], rounds, false).remove(0);
    assert!(alone.0[0].len() >= 2, "frames {:?}", alone.0[0]);
    let gemv = super::super::device::fish_drama3_step_row_gemv_calls();
    for width in [2usize, 4, 8] {
        for position in 0..width {
            let mut batch = others[..width - 1]
                .iter()
                .enumerate()
                .map(|(i, prompt)| (prompt, 12 + i as u64))
                .collect::<Vec<_>>();
            batch.insert(position, (&subject, 11));
            let together = decode_together(&model, &batch, rounds, false);
            assert_eq!(
                together[position], alone,
                "width {width}, at row {position}"
            );
        }
    }
    assert!(super::super::device::fish_drama3_step_row_gemv_calls() > gemv);
    let red = decode_together(&model, &[(&subject, 11), (&others[1], 12)], rounds, true);
    assert_ne!(
        red[0].1, alone.1,
        "a batch-split reduction must change the row's bits"
    );
}

/// Two readbacks a frame: the main and RAS picks of every row, then their
/// 9 head codes; every frame counted as sampled on the device.
#[test]
fn cuda_decode_reads_back_twice_a_frame() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use super::super::device::sampling::{
        fish_drama3_resident_frames, fish_drama3_resident_host_fallbacks,
        fish_drama3_resident_readbacks,
    };
    let synthetic = write_synthetic_model_in(7, SyntheticGeometry::KernelHeads, |_| {});
    let model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let a = prompt_of(&model, &rows(&model, 9, 1));
    let b = prompt_of(&model, &rows(&model, 12, 2));
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
    let (readbacks, frames, fallbacks) = (
        fish_drama3_resident_readbacks(),
        fish_drama3_resident_frames(),
        fish_drama3_resident_host_fallbacks(),
    );
    let mut refs = states.iter_mut().collect::<Vec<_>>();
    model.decode_round(&mut refs).unwrap();
    assert_eq!(fish_drama3_resident_readbacks() - readbacks, 2);
    assert_eq!(fish_drama3_resident_frames() - frames, 2);
    assert_eq!(fish_drama3_resident_host_fallbacks(), fallbacks);
    assert!(states.iter().all(|state| state.frames() == 1));
    let summary = model.diagnostics_summary();
    assert!(
        summary["resident_sampling"]["frames_total"]
            .as_u64()
            .unwrap()
            >= 2
    );
    assert!(!summary["device_fast_paths"]["step_row_gemv_calls_total"].is_null());
}

/// Every check of a healthy CUDA load passes: the FP8 GEMV matches the
/// original tensors, the RoPE tables match F64 at every checked position,
/// the row-invariant GEMV and the chunked DeltaNet are verified and on, and
/// the KV is BF16. The synthetic geometry (hidden 128) is outside the
/// grouped MoE kernels, which the report says.
#[test]
fn a_healthy_cuda_load_passes_every_check() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |_| {});
    let model = load(&synthetic, &device, options(FishDrama3CpuPrecision::Q8_0));
    let checks = model.cuda_checks().expect("a CUDA load runs its checks");
    assert!(model.metal_checks().is_none());
    assert_eq!(model.kv_dtype(), DType::BF16);
    assert_eq!(checks.kv_dtype, "BF16");
    assert_eq!(checks.activations, "BF16");
    // Two layers, two experts, three projection kinds.
    assert_eq!(checks.fp8_checks.len(), 12);
    assert_eq!(checks.fp8_gemv_breaches, 0);
    for (name, error) in &checks.fp8_checks {
        // BF16 input and output rounding.
        assert!(*error < 5e-3, "{name}: {error:.3e}");
    }
    assert_eq!(checks.rope_checks.len(), 6);
    assert_eq!(checks.rope_table_breaches, 0);
    for (layout, position, error) in &checks.rope_checks {
        assert!(*error < 1e-6, "{layout} at {position}: {error:.3e}");
    }
    assert_eq!(checks.row_gemv, "verified");
    assert_eq!(checks.row_gemv_breaches, 0);
    assert_eq!(checks.chunked_deltanet, "verified");
    assert_eq!(checks.chunked_deltanet_breaches, 0);
    assert!(
        checks.grouped_moe.starts_with("none"),
        "{}",
        checks.grouped_moe
    );
    let layers = model.trunk().device_trunk().unwrap();
    for layer in &layers.layers {
        if let super::super::device::trunk::DeviceMixer::DeltaNet { net, .. } = &layer.mixer {
            assert!(net.chunked);
        }
    }
    let diagnostics = &model.diagnostics_summary()["cuda_checks"];
    for key in [
        "fp8_gemv_breaches",
        "rope_table_breaches",
        "row_gemv",
        "chunked_deltanet",
        "grouped_moe",
        "kv_dtype",
    ] {
        assert!(!diagnostics[key].is_null(), "{key}");
    }
}

/// RED for the FP8 table check on CUDA: a scale off by 2 misses by far more
/// than its tolerance with BF16 activations, and the right scale passes.
#[test]
fn the_cuda_fp8_check_catches_a_wrong_scale() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use super::super::precision::FP8_GEMV_TOLERANCE;
    let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |_| {});
    let checkpoint = synthetic.open();
    let config = &synthetic.config;
    let (h, inter) = (config.trunk.hidden, config.trunk.moe.intermediate);
    let module = format!("{}.gate_proj", super::super::checkpoint::expert(0, 3));
    let original = checkpoint.fp8_block_weight(&module, inter, h, 128).unwrap();
    let weights = Tensor::from_slice(original.codes, (inter, h), &device).unwrap();
    let scales = Tensor::from_slice(&original.scales, (inter / 128, h / 128), &device).unwrap();
    let check = super::super::device::checks::fp8_error;
    let right = check(&weights, &scales, &original, DType::BF16).unwrap();
    assert!(right < FP8_GEMV_TOLERANCE / 2.0, "{right:.3e}");
    let doubled = (&scales * 2.0).unwrap();
    let wrong = check(&weights, &doubled, &original, DType::BF16).unwrap();
    assert!(wrong > 50.0 * FP8_GEMV_TOLERANCE, "{wrong:.3e}");
}
