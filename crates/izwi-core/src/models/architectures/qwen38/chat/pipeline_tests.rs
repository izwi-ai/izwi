//! The graph-decode quantum (`decode_graph_quantum`) and its pipelined steps
//! (`decode_quantum_pipelined`) on the CPU, around the FP8-geometry trunk
//! with whole-step graph decode run eagerly (the CPU references of every
//! fused kernel and of the device sampler stand in for CUDA).
use super::*;
use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::architectures::qwen38::fast_path::counters_lock;
use crate::models::architectures::qwen38::mtp::tests::{write_fp8_geometry_checkpoint, TestDir};
use crate::models::architectures::qwen38::residency::force_cpu_native_fp8_for_tests;
use crate::models::registry::NativeChatModel;
use crate::performance::PerformanceConfig;
use candle_core::Device;
use std::sync::Arc;

const PROMPT: [u32; 6] = [3, 141, 59, 265, 35, 89];

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fixture() -> (TestDir, Qwen38NativeCheckpoint) {
    let dir = TestDir::new("graph-quantum");
    write_fp8_geometry_checkpoint(dir.path());
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(dir.path()).unwrap();
    (dir, checkpoint)
}

/// A chat model around the FP8-geometry trunk (native block FP8, F16, every
/// fused fast path on the CPU references) without an MTP head, with the
/// whole-step graph decode resolved for tests when `graphs`.
///
/// Its fused paths count in the process-wide fast-path counters, so tests
/// hold `counters_lock` while they run it.
fn chat_model(checkpoint: &Qwen38NativeCheckpoint, graphs: bool) -> Qwen38ChatModel {
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
    let mut text_model = Qwen38TextModel::load_native_with_policy(
        &checkpoint.tensors,
        &checkpoint.config,
        &Device::Cpu,
        target,
        &performance.cuda,
        &policy,
    )
    .unwrap();
    text_model.enable_fast_paths_for_tests();
    if graphs {
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
        device_kind: BackendKind::Cpu,
        performance,
        load_timing: serde_json::json!({}),
        prefill_chunk_size: 4,
        cuda_compute_capability: None,
        kv_storage_provider: Qwen38KvStorageProvider::CpuF32,
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
        prefix_caching: false,
        prefix_reuse: Qwen38PrefixReuse::default(),
        graph_quantum: GraphQuantumCounters::default(),
        kernel_timings: serde_json::Value::Null,
    }
}

/// A fresh paged cache for the fixture's one attention layer (F16).
fn cache(checkpoint: &Qwen38NativeCheckpoint) -> PhysicalPagedKvCache {
    caches(checkpoint, 1).remove(0)
}

/// `rows` paged caches of four pages each, carved from one arena so a batch
/// can lower their slots into one slot map.
fn caches(checkpoint: &Qwen38NativeCheckpoint, rows: usize) -> Vec<PhysicalPagedKvCache> {
    let text = &checkpoint.config.text;
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(3813),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let binding = KvLayerBinding {
        model_layer: 3,
        physical_layer: 0,
    };
    let arena = Arc::new(
        CpuKvArena::new(KvArenaConfig {
            id,
            group,
            page_tokens: 8,
            capacity_pages: 4 * rows as u32,
            growth: None,
            dtype: DType::F16,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: text.attention_head_count_kv as u32,
                key_head_dim: text.attention_key_length as u32,
                value_head_dim: text.attention_value_length as u32,
            }],
        })
        .unwrap(),
    );
    (0..rows as u32)
        .map(|row| {
            let blocks = (4 * row..4 * row + 4)
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

struct Decoded {
    ids: Vec<u32>,
    text: String,
    calls: usize,
    /// Logprob entries per call, against the tokens each call emitted.
    logprobs: Vec<(usize, usize)>,
    context: usize,
}

/// Decode a fresh request to completion, `quantum` inputs per call.
fn decode(
    model: &Qwen38ChatModel,
    checkpoint: &Qwen38NativeCheckpoint,
    config: &ChatGenerationConfig,
    quantum: usize,
    max_new_tokens: usize,
) -> Decoded {
    let prepared = Qwen38PreparedPrompt {
        prompt_ids: PROMPT.to_vec(),
        prompt_positions: (0..PROMPT.len()).map(|position| [position; 3]).collect(),
        next_text_position: PROMPT.len(),
        history_boundary: None,
    };
    let mut state = model
        .start_decode_state_physical(
            &[],
            max_new_tokens,
            config,
            Some(&prepared),
            cache(checkpoint),
            None,
        )
        .unwrap();
    let (mut text, mut calls, mut logprobs) = (String::new(), 0usize, Vec::new());
    while !state.finished {
        let before = state.tokens_generated;
        let step = model.decode_quantum(&mut state, quantum).unwrap();
        text.push_str(&step.delta);
        calls += 1;
        logprobs.push((
            state.tokens_generated - before,
            std::mem::take(&mut state.pending_logprobs).len(),
        ));
        assert!(step.input_tokens_committed <= quantum);
    }
    Decoded {
        ids: state.generated_ids.clone(),
        text,
        calls,
        logprobs,
        context: state.physical_kv.context_len(),
    }
}

fn pipelined(model: &Qwen38ChatModel) -> u64 {
    model.text_model.cuda_graphs_summary()["whole_step"]["pipelined"]
        .as_u64()
        .unwrap_or(0)
}

/// A graph-decode quantum pipelines its steps once the whole-step graph
/// takes device tokens: device token selection, with step k+1 queued before
/// token k is read back. It must emit exactly what step-by-step decode
/// through the same graph does (greedy and seeded sampling, the device
/// sampler's references on both sides), stop where it stops (discarding the
/// step queued past a stop token, which commits nothing), and keep one
/// logprob per token when logprobs force step-by-step decode.
#[test]
fn a_pipelined_quantum_matches_step_by_step_decode() {
    let _counters = counters_lock();
    let _env = env_lock();
    let _sampling = force_device_sampling_for_tests();
    let (_dir, checkpoint) = fixture();
    let model = chat_model(&checkpoint, true);
    assert!(model.text_model.graph_step_ready());
    let greedy = ChatGenerationConfig::default();

    let expected = decode(&model, &checkpoint, &greedy, 1, 10);
    assert_eq!(pipelined(&model), 0, "a one-token quantum never pipelines");
    assert_eq!(
        expected.ids.len(),
        10,
        "the fixture runs to the cap: {:?}",
        expected.ids
    );
    let actual = decode(&model, &checkpoint, &greedy, 4, 10);
    assert_eq!(actual.ids, expected.ids);
    assert_eq!(actual.text, expected.text);
    assert_eq!(actual.context, expected.context);
    assert!(actual.calls < expected.calls, "{} quanta", actual.calls);
    let after_greedy = pipelined(&model);
    // The first quantum samples the prefill and forwards one token from the
    // host; every later step in a quantum takes the device's token.
    assert!(after_greedy > 0, "no step took a device-selected token");

    // A stop token mid-quantum finishes the stream with the next step already
    // queued: that step is discarded and commits nothing.
    let ids = &expected.ids;
    let stop = ids
        .iter()
        .enumerate()
        .skip(2)
        .find(|(index, token)| !ids[..*index].contains(token))
        .map(|(_, &token)| token)
        .expect("a token first emitted mid-stream");
    let stopping = ChatGenerationConfig {
        stop_token_ids: vec![stop],
        ..ChatGenerationConfig::default()
    };
    let expected_stop = decode(&model, &checkpoint, &stopping, 1, 10);
    let actual_stop = decode(&model, &checkpoint, &stopping, 8, 10);
    assert!(
        expected_stop.ids.len() < ids.len(),
        "the stop token ends early"
    );
    assert_eq!(expected_stop.ids.last(), Some(&stop));
    assert_eq!(
        (actual_stop.ids, actual_stop.text),
        (expected_stop.ids, expected_stop.text)
    );
    assert_eq!(
        actual_stop.context, expected_stop.context,
        "the discarded step committed nothing"
    );
    assert!(pipelined(&model) > after_greedy);

    // Greedy with a repetition penalty: the device history counts every
    // token the quantum chose, as the step-by-step history does (the
    // fixture's plain greedy stream repeats tokens, so the penalty matters).
    assert!(
        (1..ids.len()).any(|index| ids[..index].contains(&ids[index])),
        "plain greedy repeats a token: {ids:?}"
    );
    let penalized = ChatGenerationConfig {
        repetition_penalty: 1.5,
        ..ChatGenerationConfig::default()
    };
    let before = pipelined(&model);
    let expected_penalized = decode(&model, &checkpoint, &penalized, 1, 20);
    // One quantum pipelines the whole stream after the first token, so the
    // penalties inside it come from the device counts alone (RED: without
    // the device-side count update the streams split at the first repeat).
    let actual_penalized = decode(&model, &checkpoint, &penalized, 24, 20);
    assert_ne!(expected_penalized.ids[..10], expected.ids[..]);
    assert_eq!(actual_penalized.ids, expected_penalized.ids);
    assert_eq!(
        actual_penalized.calls, 1,
        "one quantum: the prefill token, then the rest pipelined"
    );
    assert!(pipelined(&model) >= before + 18);

    // Seeded sampling with penalties: the device selector draws what the
    // step-by-step device sampler draws.
    let sampled = ChatGenerationConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        presence_penalty: 0.2,
        seed: 7,
        ..ChatGenerationConfig::default()
    };
    let before = pipelined(&model);
    let expected_sampled = decode(&model, &checkpoint, &sampled, 1, 10);
    let actual_sampled = decode(&model, &checkpoint, &sampled, 4, 10);
    assert_eq!(actual_sampled.ids, expected_sampled.ids);
    assert_eq!(actual_sampled.text, expected_sampled.text);
    assert!(pipelined(&model) > before);
    // A different seed draws a different stream: the comparison sees the RNG.
    let reseeded = decode(
        &model,
        &checkpoint,
        &ChatGenerationConfig {
            seed: 8,
            ..sampled.clone()
        },
        4,
        10,
    );
    assert_ne!(reseeded.ids, expected_sampled.ids);

    // Logprobs keep the quantum step by step, with one entry per token.
    let with_logprobs = ChatGenerationConfig {
        logprobs: true,
        top_logprobs: 2,
        ..ChatGenerationConfig::default()
    };
    let before = pipelined(&model);
    let logged = decode(&model, &checkpoint, &with_logprobs, 4, 10);
    assert_eq!(pipelined(&model), before, "logprobs never pipeline");
    assert_eq!(logged.ids, expected.ids);
    for (emitted, entries) in &logged.logprobs {
        assert_eq!(emitted, entries, "one logprob per emitted token");
    }
    let quantum = model.cuda_graphs_diagnostics()["decode_quantum"].clone();
    assert!(
        quantum["pipelined_quanta"].as_u64().unwrap() > 0,
        "{quantum}"
    );
    assert_eq!(quantum["granted_max"].as_u64(), Some(24), "{quantum}");
}

/// What the model asks the scheduler for: one token per step without graph
/// decode; the configurable graph-decode quantum, sustained past the soft
/// SLA, with it; and the registry tuple the executor reads. (No test sets the
/// switches' environment variables: `setenv` races with other threads'
/// `getenv`.)
#[test]
fn graph_decode_prefers_a_sustained_quantum() {
    let _counters = counters_lock();
    let _env = env_lock();
    assert_eq!(decode_quantum_from(None), 8);
    assert_eq!(decode_quantum_from(Some(" 3 ")), 3);
    assert_eq!(decode_quantum_from(Some("1")), 1);
    assert_eq!(decode_quantum_from(Some("64")), 64);
    for invalid in ["0", "65", "many", ""] {
        assert_eq!(decode_quantum_from(Some(invalid)), 8, "{invalid}");
    }
    let (_dir, checkpoint) = fixture();
    let eager = chat_model(&checkpoint, false);
    assert_eq!(eager.preferred_decode_tokens(), 1);
    assert!(!eager.sustained_decode_quantum());
    let graphs = chat_model(&checkpoint, true);
    assert_eq!(graphs.preferred_decode_tokens(), 8);
    assert!(graphs.sustained_decode_quantum());
    let quantum = graphs.cuda_graphs_diagnostics()["decode_quantum"].clone();
    assert_eq!(quantum["preferred"], 8, "{quantum}");
    assert_eq!(quantum["sustained"], true, "{quantum}");
    assert_eq!(quantum["switch"], DECODE_QUANTUM_ENV);
    // Off CUDA the device sampler is off, so pipelining is too.
    assert_eq!(quantum["pipelining"]["backend"], "legacy", "{quantum}");
    let _sampling = force_device_sampling_for_tests();
    assert_eq!(graphs.pipelining_refusal(), None);

    let native = NativeChatModel::Qwen38(graphs);
    assert_eq!(native.speculative_decode_profile(), (8, true, false));
}

/// An all-greedy batch reads every row's token back from the device sampler
/// at once, which has no logprobs: a row that asked for logprobs must keep
/// the batch on the per-row sampler, or its entries are silently dropped.
#[test]
fn a_greedy_batch_keeps_a_rows_logprobs() {
    let _counters = counters_lock();
    let _env = env_lock();
    let _sampling = force_device_sampling_for_tests();
    let (_dir, checkpoint) = fixture();
    let model = chat_model(&checkpoint, false);
    let prepared = Qwen38PreparedPrompt {
        prompt_ids: PROMPT.to_vec(),
        prompt_positions: (0..PROMPT.len()).map(|position| [position; 3]).collect(),
        next_text_position: PROMPT.len(),
        history_boundary: None,
    };
    let mut caches = caches(&checkpoint, 2);
    let mut start = |config: &ChatGenerationConfig| {
        let mut state = model
            .start_decode_state_physical(&[], 8, config, Some(&prepared), caches.remove(0), None)
            .unwrap();
        // The first token samples the stored prefill logits on its own.
        model.decode_quantum(&mut state, 1).unwrap();
        state.pending_logprobs.clear();
        state
    };
    let greedy = ChatGenerationConfig::default();
    let with_logprobs = ChatGenerationConfig {
        logprobs: true,
        top_logprobs: 2,
        ..ChatGenerationConfig::default()
    };
    let mut plain = start(&greedy);
    let mut logged = start(&with_logprobs);
    for _ in 0..3 {
        let steps = model
            .decode_step_batch(&mut [&mut plain, &mut logged])
            .unwrap();
        assert!(steps.iter().all(|step| !step.finished));
        assert!(plain.pending_logprobs.is_empty());
        assert_eq!(
            std::mem::take(&mut logged.pending_logprobs).len(),
            1,
            "one logprob entry per batched token"
        );
    }
}
