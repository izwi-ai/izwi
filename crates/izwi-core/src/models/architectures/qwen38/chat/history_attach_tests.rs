//! Cross-turn prefix reuse through production loading: a session that
//! attaches a published history prefix (shared KV pages below the fork plus
//! the forked hybrid-state snapshot) must continue exactly where a fresh run
//! of the same prompt is, and decode what it decodes, on every device lane.
//!
//! The trunk is the FP8-geometry fixture (three 128-dim DeltaNet layers, then
//! gated attention) loaded through `Qwen38ResidencyPolicy::resolve` for the
//! device: native block FP8 with every fused fast path. The CPU leg also runs
//! decode through the whole-step graph body with pipelined quanta (the CPU
//! references stand in for the CUDA kernels), the Metal leg runs production
//! Metal residency, and the CUDA leg (skipped without a CUDA test device;
//! `IZWI_REQUIRE_CUDA_TEST_DEVICE=1` makes it mandatory) runs the captured
//! whole-step graph, the fused transactional GDN and the FP8 prefill GEMM on
//! BF16 KV pages.
//!
//! The executor's transactional state flow is mirrored step by step: every
//! prefill span and decode quantum restores the committed tensor state,
//! runs, stages its state and commits it. The attached session begins at the
//! fork cursor and restores the forked snapshot before its first span.
//!
//! Checks, each exact where it can be exact:
//! - right after the restore, the attached state equals the published
//!   snapshot bit for bit (every device);
//! - after the residual span, the state equals the fresh run's, which took
//!   identical spans (bit for bit on CPU and Metal; within a tight bound on
//!   CUDA, whose determinism this lane cannot check);
//! - both decode the same tokens (through graph decode with pipelined quanta
//!   on CPU and CUDA);
//! - decoding never mutates the published snapshot (aliasing).

use super::*;
use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::state::{
    negotiate_state_plan, PhysicalStateSequenceId, PhysicalStateTransactionId,
    StateBackendPlanRequest, StateComponentValue, TensorStateArena, TensorStateCapacity,
};
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::architectures::qwen38::cache::{
    CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN,
};
use crate::models::architectures::qwen38::fast_path::counters_lock;
use crate::models::architectures::qwen38::mtp::tests::{write_fp8_geometry_checkpoint, TestDir};
use crate::models::architectures::qwen38::native::Qwen38NativeCheckpoint;
use crate::models::architectures::qwen38::residency::force_cpu_native_fp8_for_tests;
use crate::performance::{OptimizationMode, PerformanceConfig};
use candle_core::{DType, Device, DeviceLocation};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Prompt length and the published history floor. The fork is a multiple of
/// every leg's page size (8 and 16 tokens), and both spans are at least 16
/// rows, so CUDA runs them through the FP8 prefill GEMM.
const PROMPT_TOKENS: usize = 57;
const FORK: usize = 32;
const MAX_NEW_TOKENS: usize = 7;
/// Blocks each session owns in the shared arena.
const SESSION_PAGES: u32 = 10;

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Unstructured prompt ids (xorshift), below the fixture's 512-token vocab.
fn prompt_ids() -> Vec<u32> {
    let mut state = 0x9e37_79b9_u32;
    (0..PROMPT_TOKENS)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % 500 + 3
        })
        .collect()
}

/// The FP8-geometry trunk wrapped as a chat model without an MTP head, with
/// prefix sharing on, loaded the way production loads it on `device`. On the
/// CPU the fused fast paths and the whole-step graph body are forced onto
/// their CPU references.
fn chat_model(
    checkpoint: &Qwen38NativeCheckpoint,
    device: &Device,
    kv_storage_provider: Qwen38KvStorageProvider,
) -> Qwen38ChatModel {
    let backend = crate::models::architectures::qwen38::residency::backend_of(device);
    let _forced = (backend == BackendKind::Cpu).then(force_cpu_native_fp8_for_tests);
    let mut performance = PerformanceConfig::default();
    // Shared prefixes and the MTP head exclude each other (the contract keeps
    // every domain private with a head); production CUDA leaves it out.
    performance.cuda.mtp = OptimizationMode::Off;
    let target = match backend {
        BackendKind::Cuda => ProjectionMaterialization::BF16,
        _ => ProjectionMaterialization::F16,
    };
    let policy = Qwen38ResidencyPolicy::resolve(
        device,
        target,
        &performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    let mut text_model = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        device,
        target,
        &performance.cuda,
        &policy,
    )
    .unwrap();
    if backend == BackendKind::Cpu {
        text_model.enable_fast_paths_for_tests();
        text_model.enable_step_graph_for_tests();
    }
    let vocab = checkpoint.config.vocab_size;
    let words = (0..vocab)
        .map(|token| format!("\"t{token} \":{token}"))
        .collect::<Vec<_>>()
        .join(",");
    let inner = Tokenizer::from_hf_json_bytes(
        format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{{"type":"WordLevel","vocab":{{{words}}},"unk_token":"t0 "}}}}"#
        )
        .as_bytes(),
    )
    .unwrap();
    Qwen38ChatModel {
        device_kind: backend,
        prefill_chunk_size: qwen38_prefill_chunk_tokens(backend, text_model.prefill_gemm_active()),
        performance,
        load_timing: serde_json::json!({}),
        cuda_compute_capability: None,
        kv_storage_provider,
        variant: ModelVariant::Qwen3827BFp8,
        tokenizer: Qwen38Tokenizer {
            inner,
            vocab_size: vocab,
            specials: SpecialTokenIds {
                im_start: u32::MAX,
                im_end: 100_000,
                eos: 100_001,
                eos_alt: None,
            },
            literal_special_tokens: Vec::new(),
            chat_template: String::new(),
            default_enable_thinking: false,
        },
        text_config: checkpoint.config.text.clone(),
        text_model,
        mtp_policy: Qwen38MtpPolicy::Disabled,
        mtp_head: None,
        prefix_caching: true,
        prefix_reuse: Qwen38PrefixReuse::default(),
        graph_quantum: GraphQuantumCounters::default(),
        kernel_timings: serde_json::Value::Null,
    }
}

/// The runtime's device tag for an arena (Metal arenas carry the exact
/// device identity Candle reports, not an ordinal).
fn arena_device_ordinal(device: &Device) -> Option<u32> {
    match device.location() {
        DeviceLocation::Cpu => None,
        DeviceLocation::Cuda { gpu_id } => u32::try_from(gpu_id).ok(),
        DeviceLocation::Metal { gpu_id } => {
            let id = gpu_id as u64;
            Some((id ^ (id >> 32)) as u32)
        }
    }
}

/// One paged arena for the fixture's attention layer (model layer 3), with
/// independently addressable blocks so sessions can share prefix pages.
struct Pages {
    arena: Arc<dyn KvArena>,
    id: KvArenaId,
    group: KvGroupId,
    binding: KvLayerBinding,
    page: usize,
}

impl Pages {
    fn new(model: &Qwen38ChatModel, device: &Device, page: usize) -> Self {
        let text = &model.text_config;
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(3_838),
            backend: model.device_kind,
            device_ordinal: arena_device_ordinal(device),
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
            page_tokens: page as u32,
            capacity_pages: 3 * SESSION_PAGES,
            growth: None,
            dtype: model.kv_storage_provider.dtype(),
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: text.attention_head_count_kv as u32,
                key_head_dim: text.attention_key_length as u32,
                value_head_dim: text.attention_value_length as u32,
            }],
        };
        let arena: Arc<dyn KvArena> = if model.device_kind == BackendKind::Cpu {
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
            panic!("accelerator KV arenas need the cuda or metal feature")
        };
        Self {
            arena,
            id,
            group,
            binding,
            page,
        }
    }

    fn view(&self, blocks: &[u32], context_len: usize) -> PhysicalPagedKvCache {
        PhysicalPagedKvCache::new(
            self.arena.clone(),
            vec![self.binding],
            blocks
                .iter()
                .map(|&index| CacheBlockRef {
                    arena: self.id,
                    group: self.group,
                    index,
                    slot_generation: 1,
                })
                .collect(),
            context_len,
        )
        .unwrap()
    }

    /// Session `index`'s own blocks.
    fn blocks(&self, index: u32) -> Vec<u32> {
        (index * SESSION_PAGES..(index + 1) * SESSION_PAGES).collect()
    }
}

/// The executor's transactional tensor state: one arena, fresh transaction
/// ids, and the stage -> commit -> restore cycle every engine step runs.
struct StateFlow {
    arena: TensorStateArena,
    next_transaction: AtomicU64,
}

impl StateFlow {
    fn new(model: &Qwen38ChatModel, device: &Device, page: usize) -> Self {
        let contract = model
            .managed_composite_cache_contract(model.kv_storage_provider.dtype(), page)
            .unwrap();
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: model.device_kind,
                device_ordinal: arena_device_ordinal(device),
                page_tokens_hint: None,
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let capacity = TensorStateCapacity::for_plan(&plan, 8, 8).unwrap();
        Self {
            arena: TensorStateArena::new(Arc::new(plan), capacity, device.clone()).unwrap(),
            next_transaction: AtomicU64::new(1),
        }
    }

    fn transaction(&self) -> PhysicalStateTransactionId {
        PhysicalStateTransactionId::new(self.next_transaction.fetch_add(1, Ordering::Relaxed))
            .unwrap()
    }

    /// Register `sequence` and bind it to `state` (a fresh session).
    fn bind(&self, state: &mut ChatDecodeState, sequence: u64) {
        self.arena
            .register(PhysicalStateSequenceId::new(sequence).unwrap())
            .unwrap();
        state.bind_tensor_sequence(sequence).unwrap();
    }

    /// End an engine step: stage the state and commit it at the cache cursor.
    fn commit(&self, state: &mut ChatDecodeState, sequence: u64) {
        let transaction = self.transaction();
        self.arena
            .begin(transaction, PhysicalStateSequenceId::new(sequence).unwrap())
            .unwrap();
        state
            .stage_tensor_state(&self.arena, transaction.get())
            .unwrap();
        drop(
            self.arena
                .commit(transaction, state.physical_kv.context_len() as u64)
                .unwrap(),
        );
    }

    /// Fork the committed state of `publisher` into a fresh `attached`
    /// sequence at `cursor` (the manager's attach-by-fork).
    fn fork(&self, publisher: u64, attached: u64, cursor: u64) {
        let attached = PhysicalStateSequenceId::new(attached).unwrap();
        self.arena.register(attached).unwrap();
        let transaction = self.transaction();
        self.arena.begin(transaction, attached).unwrap();
        for domain in [RECURRENT_STATE_DOMAIN, CONVOLUTION_STATE_DOMAIN] {
            let snapshot = self
                .arena
                .read(PhysicalStateSequenceId::new(publisher).unwrap(), domain)
                .unwrap()
                .expect("published domain state");
            assert_eq!(snapshot.cursor, cursor);
            let components: Vec<StateComponentValue> = snapshot.components.to_vec();
            self.arena
                .stage_replace(transaction, domain, 0, cursor, components)
                .unwrap();
        }
        drop(self.arena.commit(transaction, cursor).unwrap());
    }

    /// Host copy of a sequence's committed recurrent and conv state.
    fn committed_values(&self, sequence: u64) -> Vec<f32> {
        let mut values = Vec::new();
        for domain in [RECURRENT_STATE_DOMAIN, CONVOLUTION_STATE_DOMAIN] {
            let snapshot = self
                .arena
                .read(PhysicalStateSequenceId::new(sequence).unwrap(), domain)
                .unwrap()
                .expect("committed domain state");
            for component in snapshot.components.iter() {
                values.extend(
                    component
                        .tensor
                        .as_ref()
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
            }
        }
        values
    }
}

fn greedy() -> ChatGenerationConfig {
    ChatGenerationConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        seed: 7,
        ..Default::default()
    }
}

/// Decode to completion, one engine step per quantum: restore, decode the
/// model's preferred quantum, stage and commit.
fn decode(
    model: &Qwen38ChatModel,
    flow: &StateFlow,
    state: &mut ChatDecodeState,
    sequence: u64,
) -> Vec<u32> {
    let quantum = model.preferred_decode_tokens();
    let mut steps = 0;
    while !state.finished {
        state.restore_tensor_state(&flow.arena).unwrap();
        model.decode_quantum(state, quantum).unwrap();
        flow.commit(state, sequence);
        steps += 1;
        assert!(steps <= MAX_NEW_TOKENS + 1, "decode did not finish");
    }
    state.generated_ids.clone()
}

/// The largest absolute difference, with any non-finite one saturating.
fn max_abs_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter().zip(rhs).fold(0f32, |worst, (a, b)| {
        let difference = (a - b).abs();
        if difference.is_finite() {
            worst.max(difference)
        } else {
            f32::INFINITY
        }
    })
}

/// Publish `[0, FORK)`, run the prompt fresh in the same two spans, and run
/// it attached at the fork; see the module docs for the checks. `exact`
/// demands a bit-identical post-prefill state.
fn assert_attach_matches_fresh(
    device: Device,
    kv_storage_provider: Qwen38KvStorageProvider,
    page: usize,
    exact: bool,
) {
    // The CPU leg's fused paths count in the process-wide fast-path
    // counters, and other tests change process-wide switches; hold both.
    let _counters = counters_lock();
    let _env = env_lock();
    // CUDA samples on the device; the CPU leg runs the device sampler's
    // portable references so its graph quanta pipeline too.
    let _sampling = device.is_cpu().then(force_device_sampling_for_tests);
    let dir = TestDir::new("history-attach");
    write_fp8_geometry_checkpoint(dir.path());
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(dir.path()).unwrap();
    let model = chat_model(&checkpoint, &device, kv_storage_provider);
    let backend = model.device_kind;
    // Metal decodes eagerly; the CPU references and CUDA run graph decode.
    let graphs = backend != BackendKind::Metal;
    assert_eq!(
        model.text_model.graph_decode(),
        graphs,
        "{backend:?}: {}",
        model.text_model.cuda_graphs_summary()
    );
    let pages = Pages::new(&model, &device, page);
    let flow = StateFlow::new(&model, &device, page);
    let ids = prompt_ids();
    let total = ids.len();
    let prepared = Qwen38PreparedPrompt {
        prompt_positions: (0..total).map(|position| [position; 3]).collect(),
        next_text_position: total,
        history_boundary: Some(FORK + 3),
        prompt_ids: ids,
    };
    let config = greedy();
    let begin = |cache: PhysicalPagedKvCache| {
        model
            .begin_chunked_prefill_state_physical(
                &[],
                MAX_NEW_TOKENS,
                &config,
                Some(&prepared),
                cache,
                None,
            )
            .unwrap()
    };
    let span = |state: &mut ChatDecodeState, start: usize, end: usize| {
        model
            .continue_chunked_prefill_physical(
                state,
                &[],
                &config,
                Some(&prepared),
                start,
                end,
                total,
            )
            .unwrap()
    };

    // Publisher (sequence 1): the history span, committed at the fork.
    let mut publisher = begin(pages.view(&pages.blocks(0), 0));
    flow.bind(&mut publisher, 1);
    assert!(!span(&mut publisher, 0, FORK));
    flow.commit(&mut publisher, 1);
    let published = flow.committed_values(1);
    assert!(published.iter().any(|value| *value != 0.0));

    // Fresh run (sequence 2): the same two spans, one engine step each.
    let mut fresh = begin(pages.view(&pages.blocks(1), 0));
    flow.bind(&mut fresh, 2);
    assert!(!span(&mut fresh, 0, FORK));
    flow.commit(&mut fresh, 2);
    let fresh_history = flow.committed_values(2);
    if exact {
        assert!(
            fresh_history == published,
            "{backend:?}: deterministic history"
        );
    }
    fresh.restore_tensor_state(&flow.arena).unwrap();
    assert!(span(&mut fresh, FORK, total));
    let fresh_state = fresh.text_state.linear_state_values().unwrap();
    flow.commit(&mut fresh, 2);
    let fresh_tokens = decode(&model, &flow, &mut fresh, 2);
    assert_eq!(fresh_tokens.len(), MAX_NEW_TOKENS);

    // Attached (sequence 3): the publisher's pages below the fork, private
    // pages above it, and the forked snapshot.
    let shared = FORK / page;
    let mut blocks = pages.blocks(0)[..shared].to_vec();
    blocks.extend(&pages.blocks(2)[shared..]);
    let mut attached = begin(pages.view(&blocks, FORK));
    assert_eq!(attached.prefill_progress(), FORK);
    flow.fork(1, 3, FORK as u64);
    attached.bind_tensor_sequence(3).unwrap();
    attached.restore_tensor_state(&flow.arena).unwrap();
    let restored = attached.text_state.linear_state_values().unwrap();
    assert!(
        restored == published,
        "{backend:?}: the attached session must start from the published snapshot, bit for bit (max error {})",
        max_abs_error(&restored, &published)
    );
    assert!(span(&mut attached, FORK, total));
    let attached_state = attached.text_state.linear_state_values().unwrap();
    if exact {
        assert!(
            attached_state == fresh_state,
            "{backend:?}: the attached state must equal the fresh run's, bit for bit (max error {})",
            max_abs_error(&attached_state, &fresh_state)
        );
    } else {
        let scale = fresh_state.iter().fold(0f32, |m, v| m.max(v.abs()));
        let error = max_abs_error(&attached_state, &fresh_state);
        assert!(
            error <= 1e-3 * scale.max(1e-3),
            "{backend:?}: attached state differs from the fresh run by {error} (max |fresh| {scale})"
        );
    }
    flow.commit(&mut attached, 3);
    let pipelined_before = model.graph_quantum.summary()["pipelined_quanta"].as_u64();
    assert_eq!(
        decode(&model, &flow, &mut attached, 3),
        fresh_tokens,
        "{backend:?}: the attached session must decode the fresh run's tokens"
    );
    if graphs {
        // Graph decode with pipelined quanta ran on the attached state.
        assert!(
            model.graph_quantum.summary()["pipelined_quanta"].as_u64() > pipelined_before,
            "{}",
            model.graph_quantum.summary()
        );
    }

    // Decoding both sessions left the published snapshot untouched.
    let after = flow.committed_values(1);
    assert!(
        after == published,
        "{backend:?}: decode mutated the published snapshot (max change {})",
        max_abs_error(&after, &published)
    );

    // An attach that skips the restore starts from empty state and fails
    // closed instead of prefilling from zero state above the fork.
    let mut unrestored = begin(pages.view(&blocks, FORK));
    let error = model
        .continue_chunked_prefill_physical(
            &mut unrestored,
            &[],
            &config,
            Some(&prepared),
            FORK,
            total,
            total,
        )
        .expect_err("an attached span needs its restored state");
    assert!(
        error.to_string().contains("without hybrid state"),
        "{error}"
    );

    // The diagnostics count the prefills: two from zero, two attached.
    let reuse = model.prefix_reuse_diagnostics();
    assert_eq!(reuse["engaged"], true, "{reuse}");
    assert_eq!(reuse["fresh_prefills"], 2, "{reuse}");
    assert_eq!(reuse["attached_prefills"], 2, "{reuse}");
    assert_eq!(reuse["reused_prompt_tokens"], 2 * FORK as u64, "{reuse}");
    assert_eq!(
        reuse["last_prefill"]["reused_tokens"], FORK as u64,
        "{reuse}"
    );
}

#[test]
fn attached_history_matches_the_fresh_run_on_cpu_graph_decode() {
    assert_attach_matches_fresh(Device::Cpu, Qwen38KvStorageProvider::CpuF32, 8, true);
}

#[cfg(feature = "metal")]
#[test]
fn metal_attached_history_matches_the_fresh_run() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("metal_attached_history_matches_the_fresh_run")
    else {
        return;
    };
    assert_attach_matches_fresh(device, Qwen38KvStorageProvider::MetalF16, 8, true);
}

/// The CUDA lane's prefix-reuse evidence: production BF16 KV in 16-token
/// pages, the captured whole-step graph, fused GDN and the FP8 prefill GEMM.
#[cfg(feature = "cuda")]
#[test]
fn cuda_attached_history_matches_the_fresh_run() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    assert_attach_matches_fresh(device, Qwen38KvStorageProvider::CudaBf16, 16, false);
}
