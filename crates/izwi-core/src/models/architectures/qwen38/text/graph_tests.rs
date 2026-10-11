//! Graph decode (`decode_graph.rs`) off CUDA, where its segments run eagerly
//! with explicit state tensors: the segmentation, the state IO and the state
//! publication against the standard decode, bit for bit; the load-time
//! resolution and its refusal reasons; and the residency plan admission
//! prices. On CUDA the same segments run through the table-addressed kernels
//! and are captured; the `cuda_` legs run there.
//!
//! The tests never set environment variables: `setenv` races with other
//! threads' `getenv` (a crash on macOS), and the kill switches go through the
//! shared, separately tested `legacy_requested`.

use super::golden_tests::GoldenArena;
use super::*;
use crate::models::architectures::qwen38::mtp::tests::{write_fp8_geometry_checkpoint, TestDir};
use crate::models::architectures::qwen38::native::{pinned_27b_config, Qwen38NativeCheckpoint};
use crate::models::architectures::qwen38::residency::{
    force_cpu_native_fp8_for_tests, graph_decode_reserve_bytes, graph_decode_slab_bytes,
    GRAPH_DECODE_FIXED_BYTES, TENSOR_ISLAND_CACHE_BYTES,
};
use crate::performance::{OptimizationMode, PerformanceConfig};

const PROMPT: [u32; 6] = [3, 141, 59, 265, 35, 89];
const DECODE: [u32; 8] = [79, 323, 7, 211, 5, 498, 64, 300];

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fixture() -> (TestDir, Qwen38NativeCheckpoint) {
    let dir = TestDir::new("graph-decode");
    write_fp8_geometry_checkpoint(dir.path());
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(dir.path()).unwrap();
    (dir, checkpoint)
}

/// The FP8-geometry trunk (3 DeltaNet layers, then attention) under native
/// block FP8 with F16 activations, every fused fast path resolved against
/// the CPU references.
fn fused_model(checkpoint: &Qwen38NativeCheckpoint) -> Qwen38TextModel {
    let _forced = force_cpu_native_fp8_for_tests();
    let performance = PerformanceConfig::default();
    let target = ProjectionMaterialization::F16;
    let policy = Qwen38ResidencyPolicy::resolve(
        &Device::Cpu,
        target,
        &performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    let mut model = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &Device::Cpu,
        target,
        &performance.cuda,
        &policy,
    )
    .unwrap();
    model.enable_fast_paths_for_tests();
    model
}

/// Every DeltaNet layer's conv history and recurrent state, in layer order.
fn state_values(state: &Qwen38TextRuntimeState) -> Vec<f32> {
    let mut values = Vec::new();
    for layer in &state.layers {
        if let Qwen38LayerRuntimeState::Linear {
            conv_state: Some(ring),
            recurrent_state: Some(recurrent),
        } = layer
        {
            values.extend(host_values(&ring.contiguous_history().unwrap()).unwrap());
            values.extend(host_values(recurrent).unwrap());
        }
    }
    values
}

/// The last prompt row's logits, then each decode step's through the
/// batched single-row path (where graph decode lives), and the final
/// DeltaNet state.
fn decode_rows(
    model: &Qwen38TextModel,
    arena: &GoldenArena,
    session: u32,
    tokens: &[u32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut state = model.new_state();
    let mut cache = arena.session(session);
    let positions = (0..PROMPT.len() + tokens.len())
        .map(|position| [position; 3])
        .collect::<Vec<_>>();
    let logits = model
        .prefill_token_ids_physical(
            &PROMPT,
            &positions[..PROMPT.len()],
            &mut state,
            &mut cache,
            true,
        )
        .unwrap()
        .expect("prefill logits");
    let mut rows = vec![host_values(&logits).unwrap()];
    for (step, &token) in tokens.iter().enumerate() {
        let output = model
            .forward_token_ids_batch_at_physical(
                &[token],
                &[positions[PROMPT.len() + step]],
                &mut [&mut state],
                &mut [&mut cache],
            )
            .unwrap();
        assert!(output.hidden_states.dims3().is_ok());
        rows.push(host_values(&output.logits).unwrap());
    }
    assert_eq!(cache.context_len(), PROMPT.len() + tokens.len());
    (rows, state_values(&state))
}

/// Off CUDA the segments run eagerly, so the segmentation itself is
/// testable: the attention split, the state IO and the state publication
/// must reproduce the standard batched decode exactly, step after step.
#[test]
fn segmented_decode_matches_the_standard_batched_path() {
    let _env = env_lock();
    let (_dir, checkpoint) = fixture();
    let standard = fused_model(&checkpoint);
    assert_eq!(standard.cuda_graphs_summary()["backend"], "legacy");
    let mut segmented = fused_model(&checkpoint);
    segmented.enable_segmented_decode_for_tests();
    let paths = segmented.fast_path_diagnostics();
    for name in [
        "rms_norm",
        "gdn_decode",
        "qk_rope",
        "attn_gate",
        "gdn_alpha_beta_gemv",
    ] {
        assert_eq!(paths[name]["backend"], "fused", "{name}: {paths}");
    }
    let graphs = segmented.cuda_graphs_summary();
    assert_eq!(graphs["backend"], "fused", "{graphs}");
    // 3 DeltaNet layers + 1 attention layer: one segment closes before the
    // attention layer, one opens after it.
    assert_eq!(graphs["segments"], 2);
    assert_eq!(graphs["capture"], false);

    let arena = GoldenArena::new(&checkpoint.config, &Device::Cpu, DType::F16);
    let (expected, expected_state) = decode_rows(&standard, &arena, 0, &DECODE);
    let (actual, actual_state) = decode_rows(&segmented, &arena, 1, &DECODE);
    assert_eq!(actual, expected, "segmented decode must be bit-identical");
    assert_eq!(actual_state, expected_state, "published state");
    let graphs = segmented.cuda_graphs_summary();
    assert_eq!(graphs["phase"], "warm", "no capture off CUDA");
    assert_eq!(graphs["warmups"].as_u64(), Some(2 * 8), "{graphs}");
    assert_eq!(graphs["warm_checks"], 0, "the warm check runs on CUDA");
    assert_eq!(graphs["eager_steps"], 0);

    // The comparison sees the decode itself: one different token at step 3
    // leaves the rows before it alone and changes every row after it.
    let mut wrong = DECODE;
    wrong[3] ^= 1;
    let (diverged, _) = decode_rows(&segmented, &arena, 2, &wrong);
    assert_eq!(diverged[..4], expected[..4]);
    for (index, (row, expected)) in diverged.iter().zip(&expected).enumerate().skip(4) {
        assert_ne!(row, expected, "row {index}");
    }
}

/// Production resolution off CUDA, the switches, and every prerequisite the
/// segments need: each refusal names its reason.
#[test]
fn graph_decode_resolution_reports_why_it_stays_off() {
    let _env = env_lock();
    let (_dir, checkpoint) = fixture();
    let mut model = fused_model(&checkpoint);
    let reason = |model: &Qwen38TextModel| {
        let summary = model.cuda_graphs_summary();
        assert_eq!(summary["backend"], "legacy", "{summary}");
        assert_eq!(summary["phase"], "off");
        summary["legacy_reasons"][0].as_str().unwrap().to_string()
    };
    assert_eq!(reason(&model), "CUDA graph decode runs on CUDA only");

    model.resolve_decode_graphs(
        &Qwen38CudaSwitches {
            decode_graphs_off: Some("cuda.decode_graphs=off"),
            ..Qwen38CudaSwitches::default()
        },
        true,
    );
    assert_eq!(reason(&model), "cuda.decode_graphs=off");

    model.finite_diagnostics_enabled = true;
    model.enable_segmented_decode_for_tests();
    assert!(reason(&model).contains("finite diagnostics"));
    model.finite_diagnostics_enabled = false;

    // Without the fused DeltaNet decode (production resolution off CUDA).
    let _forced = force_cpu_native_fp8_for_tests();
    let performance = PerformanceConfig::default();
    let target = ProjectionMaterialization::F16;
    let policy = Qwen38ResidencyPolicy::resolve(
        &Device::Cpu,
        target,
        &performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    let load = |target, policy: &Qwen38ResidencyPolicy| {
        Qwen38TextModel::load_native_with_policy(
            &checkpoint.tensors,
            &checkpoint.config,
            &Device::Cpu,
            target,
            &performance.cuda,
            policy,
        )
        .unwrap()
    };
    let mut unfused = load(target, &policy);
    unfused.enable_segmented_decode_for_tests();
    assert!(
        reason(&unfused).contains("fused DeltaNet decode"),
        "{}",
        reason(&unfused)
    );
    drop(_forced);

    // Candle's Q8_0 QMatMul (the CPU's default plan) cannot be captured.
    let target = ProjectionMaterialization::F32;
    let q8 = Qwen38ResidencyPolicy::resolve(
        &Device::Cpu,
        target,
        &performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    let mut quantized = load(target, &q8);
    quantized.enable_fast_paths_for_tests();
    quantized.enable_segmented_decode_for_tests();
    assert!(
        reason(&quantized).contains("QMatMul"),
        "{}",
        reason(&quantized)
    );
}

/// A resolved graph decode retires the `TensorIsland` regions: a capture
/// cannot nest inside another.
#[test]
fn resolved_graph_decode_retires_the_tensor_islands() {
    let _env = env_lock();
    let (_dir, checkpoint) = fixture();
    let mut model = fused_model(&checkpoint);
    for layer in &mut model.layers {
        layer.graphs_enabled = true;
        layer.mlp.graphs_enabled = true;
    }
    model.enable_segmented_decode_for_tests();
    assert!(model.graph_decode());
    assert!(model
        .layers
        .iter()
        .all(|layer| !layer.graphs_enabled && !layer.mlp.graphs_enabled));
}

/// The residency plan that admission and the loader share: graph decode is
/// planned only on CUDA with native block FP8, its switches on and no MTP
/// head, and then admission reserves three state slabs plus the fixed graph
/// pools instead of the region caches.
#[test]
fn graph_decode_plan_and_reservation_follow_the_switches() {
    let _env = env_lock();
    let text = pinned_27b_config().text;
    assert_eq!(graph_decode_slab_bytes(&text).unwrap(), 156_893_184);
    let reserve = 3 * 156_893_184 + 64 * 1024 * 1024;
    assert_eq!(GRAPH_DECODE_FIXED_BYTES, 64 * 1024 * 1024);
    assert_eq!(graph_decode_reserve_bytes(&text).unwrap(), reserve);

    let mut no_mtp = PerformanceConfig::default();
    no_mtp.cuda.mtp = OptimizationMode::Off;
    let planned = Qwen38ResidencyPolicy::cuda_native_fp8_for_tests(&no_mtp);
    assert!(planned.graph_decode_planned(), "{}", planned.graph_decode_refusal());
    assert_eq!(
        planned.graph_capacity_bytes(&text, &no_mtp).unwrap(),
        reserve
    );

    let refusal = |performance: &PerformanceConfig| {
        let policy = Qwen38ResidencyPolicy::cuda_native_fp8_for_tests(performance);
        assert!(!policy.graph_decode_planned());
        (
            policy.graph_decode_refusal(),
            policy.graph_capacity_bytes(&text, performance).unwrap(),
        )
    };
    let default = PerformanceConfig::default();
    let (reason, bytes) = refusal(&default);
    assert!(reason.contains("MTP head is loaded"), "{reason}");
    assert_eq!(bytes, 2 * TENSOR_ISLAND_CACHE_BYTES, "trunk and MTP region caches");
    let mut graphs_off = no_mtp.clone();
    graphs_off.cuda.decode_graphs = OptimizationMode::Off;
    assert_eq!(
        refusal(&graphs_off),
        ("cuda.decode_graphs=off".to_string(), 0)
    );
    let mut fused_off = no_mtp.clone();
    fused_off.cuda.fused_decode = OptimizationMode::Off;
    let (reason, bytes) = refusal(&fused_off);
    assert!(reason.contains("cuda.fused_decode=off"), "{reason}");
    assert_eq!(bytes, TENSOR_ISLAND_CACHE_BYTES);
    let mut cuda_off = no_mtp.clone();
    cuda_off.cuda.mode = OptimizationMode::Off;
    assert_eq!(refusal(&cuda_off), ("cuda.mode=off".to_string(), 0));

    // The static CUDA estimate keeps Q8_0, which graph decode cannot capture.
    let q8 = Qwen38ResidencyPolicy::resolve_static(crate::backends::BackendKind::Cuda, &no_mtp);
    assert!(q8.graph_decode_refusal().contains("native block-FP8"));
    assert_eq!(q8.graph_capacity_bytes(&text, &no_mtp).unwrap(), TENSOR_ISLAND_CACHE_BYTES);
    for backend in [crate::backends::BackendKind::Cpu, crate::backends::BackendKind::Metal] {
        let policy = Qwen38ResidencyPolicy::resolve_static(backend, &no_mtp);
        assert_eq!(policy.graph_decode_refusal(), "CUDA graph decode runs on CUDA only");
        assert_eq!(policy.graph_capacity_bytes(&text, &no_mtp).unwrap(), 0);
    }
}

/// Production resolution on the GPU: graph decode resolves (no MTP head),
/// passes its warm check against the standard decode, captures and verifies
/// every segment twice, then replays, and tracks the eager decode.
#[cfg(feature = "cuda")]
#[test]
fn cuda_graph_decode_reaches_replay_and_tracks_the_eager_decode() {
    use super::golden_tests::{argmax, relative_l2};
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let _env = env_lock();
    let (_dir, checkpoint) = fixture();
    let target = ProjectionMaterialization::BF16;
    let load = |performance: &PerformanceConfig| {
        let policy = Qwen38ResidencyPolicy::resolve(
            &device,
            target,
            performance,
            &checkpoint.tensors,
            &checkpoint.config,
        );
        Qwen38TextModel::load_native_with_policy(
            &checkpoint.tensors,
            &checkpoint.config,
            &device,
            target,
            &performance.cuda,
            &policy,
        )
        .unwrap()
    };
    let mut performance = PerformanceConfig::default();
    performance.cuda.mtp = OptimizationMode::Off;
    let graphs = load(&performance);
    let summary = graphs.cuda_graphs_summary();
    assert_eq!(summary["backend"], "fused", "{summary}");
    assert_eq!(summary["capture"], true);
    let segments = summary["segments"].as_u64().unwrap();
    performance.cuda.decode_graphs = OptimizationMode::Off;
    let eager = load(&performance);
    assert_eq!(eager.cuda_graphs_summary()["backend"], "legacy");

    let arena = GoldenArena::new(&checkpoint.config, &device, DType::BF16);
    let (expected, _) = decode_rows(&eager, &arena, 0, &DECODE);
    let (actual, _) = decode_rows(&graphs, &arena, 1, &DECODE);
    for (index, (row, expected)) in actual.iter().zip(&expected).enumerate() {
        let error = relative_l2(row, expected);
        assert!(error < 2e-2, "row {index}: relative L2 {error}");
        assert_eq!(argmax(row), argmax(expected), "row {index}");
    }
    // Eight steps: warm (with its check), capture + verify, verify, then
    // five replays.
    let summary = graphs.cuda_graphs_summary();
    assert_eq!(summary["phase"], "replay", "{summary}");
    assert_eq!(summary["warm_checks"].as_u64(), Some(1), "{summary}");
    assert_eq!(summary["captures"].as_u64(), Some(segments), "{summary}");
    assert_eq!(
        summary["verified_segments"].as_u64(),
        Some(2 * segments),
        "{summary}"
    );
    assert_eq!(summary["replays"].as_u64(), Some(5 * segments), "{summary}");
    assert_eq!(summary["eager_steps"], 0);
}
