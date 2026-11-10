//! End-to-end tests on the synthetic checkpoint: load, census, the retained
//! row API, span and batch invariance, and device agreement.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use super::checkpoint::NemotronCheckpoint;
use super::config::NemotronTranscribeConfig;
use super::model::{NemotronStepRow, NemotronTranscribeModel, NemotronTranscribeRequest};
use super::prompt::NemotronPromptOptions;
use super::synthetic::{temp_checkpoint, test_audio};
use super::transcript::NemotronFinishReason;
use crate::models::shared::attention::physical::{
    test_physical_caches_for_model_layers_on, PhysicalPagedKvCache,
};

fn caches(
    device: &Device,
    model: &NemotronTranscribeModel,
    rows: usize,
    capacity: usize,
) -> Vec<PhysicalPagedKvCache> {
    let a = model.decoder().config().attention;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };
    test_physical_caches_for_model_layers_on(
        device,
        dtype,
        model.decoder().attention_layers(),
        a.kv_heads,
        a.head_dim,
        capacity,
        rows,
    )
}

fn request(max_new_tokens: usize) -> NemotronTranscribeRequest {
    NemotronTranscribeRequest {
        prompt: NemotronPromptOptions::default(),
        max_new_tokens: Some(max_new_tokens),
        word_timestamps: false,
    }
}

#[test]
fn the_synthetic_checkpoint_passes_the_census_and_loads_on_cpu() {
    let dir = temp_checkpoint();
    let config = NemotronTranscribeConfig::load(dir.path()).unwrap();
    let checkpoint = NemotronCheckpoint::open(dir.path()).unwrap();
    checkpoint.validate_census(&config).unwrap();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let summary = model.diagnostics_summary();
    // Every sampled projection kind was checked against the checkpoint:
    // Mamba in/out, expert and shared up/down, attention q/k/v/o.
    assert_eq!(summary["projection_checks"].as_array().unwrap().len(), 10);
    assert!(summary["mamba2_scan_check"].is_null());
    // The CPU routes on the host over Q8_0 experts.
    assert_eq!(summary["moe_path"], "per_expert");
    assert!(summary["moe_stacked_check"].is_null());
}

#[test]
fn the_census_fails_closed_on_a_missing_or_extra_tensor() {
    let dir = temp_checkpoint();
    let index_path = dir.path().join(super::checkpoint::INDEX_FILE);
    let original = std::fs::read_to_string(&index_path).unwrap();
    let mut index: serde_json::Value = serde_json::from_str(&original).unwrap();
    // A tensor the index names but no file holds.
    index["weight_map"]["backbone.layers.0.mixer.extra"] =
        "model-00002-of-00004.safetensors".into();
    std::fs::write(&index_path, index.to_string()).unwrap();
    assert!(NemotronCheckpoint::open(dir.path()).is_err());
    // A config whose geometry no longer matches the tensors.
    std::fs::write(&index_path, &original).unwrap();
    let config_path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    config["llm_config"]["moe_intermediate_size"] = 64.into();
    std::fs::write(&config_path, config.to_string()).unwrap();
    let config = NemotronTranscribeConfig::load(dir.path()).unwrap();
    let error = NemotronCheckpoint::open(dir.path())
        .unwrap()
        .validate_census(&config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("experts.0.up_proj"), "{error}");
}

#[test]
fn a_request_transcribes_deterministically_through_retained_spans() {
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let audio = test_audio(2.5, 1);
    let mut cache = caches(&Device::Cpu, &model, 1, 1024).remove(0);
    let first = model
        .transcribe_with_cache(&audio, request(12), &mut cache, 64)
        .unwrap();
    assert_eq!(first.audio_tokens, (2.5f64 * 100.0) as usize / 8 + 1);
    assert!(first.generated_tokens <= 12);
    assert!(matches!(
        first.finish_reason,
        NemotronFinishReason::Stop
            | NemotronFinishReason::Length
            | NemotronFinishReason::Repetition
    ));
    let mut cache = caches(&Device::Cpu, &model, 1, 1024).remove(0);
    let again = model
        .transcribe_with_cache(&audio, request(12), &mut cache, 64)
        .unwrap();
    assert_eq!(first.transcript, again.transcript);
    // The timing counts the spans and steps the row ran: the first token
    // comes from the last span, and an eos ends a step without a token.
    let timing = &first.timing;
    assert_eq!(
        timing.prefill_spans as usize,
        first.prompt_tokens.div_ceil(64)
    );
    let accepted =
        first.generated_tokens + usize::from(first.finish_reason == NemotronFinishReason::Stop);
    assert_eq!(timing.decode_steps as usize, accepted - 1);
    assert!(timing.prefill_ms > 0.0 && timing.perception_ms > 0.0);
    assert!(timing.perception_ms <= timing.prefill_ms);
    // No timestamps were asked for, so the CTC stage never ran.
    assert!(timing.ctc_ms < 50.0);
    let json = timing.to_json();
    assert_eq!(json["decode_steps"], timing.decode_steps);
    assert!(model.diagnostics_summary()["decode_ms"].as_f64().unwrap() >= timing.decode_ms);
}

/// The CPU's opt-in F16 KV (`IZWI_NEMOTRON_TRANSCRIBE_CPU_F16_KV`): a
/// prompt prefilled in 17-token spans (each span after the first attends
/// over F16 pages) leaves the first attention layer's residual within F16
/// rounding of the F32 pages' and not equal to it, ends with logits as
/// close, and a row over F16 pages transcribes.
///
/// The rounding is checked after the attention layer: the logits come from
/// a one-row LM head, which Candle's Q8_0 kernel multiplies with its
/// activations quantized to int8, so whether an F16 rounding survives to
/// them depends on the host's kernels (on CI's Linux runner the logits came
/// out identical). Greedy tokens are not compared: on random weights one
/// rounding flips a near-tie, and every token after it differs.
#[test]
fn a_row_transcribes_over_f16_pages_on_the_cpu() {
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let audio = test_audio(2.5, 1);
    let a = model.decoder().config().attention;
    let f16_cache = || {
        test_physical_caches_for_model_layers_on(
            &Device::Cpu,
            DType::F16,
            model.decoder().attention_layers(),
            a.kv_heads,
            a.head_dim,
            1024,
            1,
        )
        .remove(0)
    };
    let layer = model.decoder().attention_layers()[0] as usize;
    // Every span's residual after the first attention layer, and the
    // prompt's last logits.
    let prefill = |mut cache: PhysicalPagedKvCache| {
        let artifact = Arc::new(model.prepare(&audio, request(12)).unwrap());
        let mut state = model.new_row(artifact, 1024).unwrap();
        let mut trace = super::decoder::DecoderTrace::default();
        let request = super::decoder::TraceRequest {
            routes: false,
            hidden_after_layers: vec![layer],
        };
        while state.prefill_cursor() < state.prompt_tokens() {
            let n = 17.min(state.prompt_tokens() - state.prefill_cursor());
            model
                .retained_prefill_traced(&mut state, &mut cache, n, Some((&request, &mut trace)))
                .unwrap();
        }
        let hidden = trace
            .hidden
            .into_iter()
            .flat_map(|(_, values)| values)
            .collect::<Vec<f32>>();
        (hidden, trace.final_logits)
    };
    let (half_hidden, half_logits) = prefill(f16_cache());
    let (full_hidden, full_logits) = prefill(caches(&Device::Cpu, &model, 1, 1024).remove(0));
    let relative_l2 = crate::kernels::numeric_check::relative_l2;
    let error = relative_l2(&half_hidden, &full_hidden);
    assert!(error > 0.0 && error < 1e-2, "{error}");
    let error = relative_l2(&half_logits, &full_logits);
    assert!(!full_logits.is_empty() && error < 1e-2, "{error}");
    let transcript = model
        .transcribe_with_cache(&audio, request(12), &mut f16_cache(), 64)
        .unwrap();
    assert!(!transcript.transcript.raw.is_empty());
}

/// The first decode logits after prefilling in spans of different sizes
/// stay within float tolerance of each other (the span boundary moves the
/// SSM chunking and the attention prefix).
#[test]
fn span_size_does_not_change_the_prefill() {
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let audio = test_audio(3.0, 2);
    let artifact = Arc::new(model.prepare(&audio, request(4)).unwrap());
    let run = |span: usize| {
        let mut cache = caches(&Device::Cpu, &model, 1, 1024).remove(0);
        let mut state = model.new_row(artifact.clone(), 1024).unwrap();
        while state.prefill_cursor() < state.prompt_tokens() {
            let n = span.min(state.prompt_tokens() - state.prefill_cursor());
            model.retained_prefill(&mut state, &mut cache, n).unwrap();
        }
        state.decode_loop().unwrap().generated().to_vec()
    };
    let whole = run(10_000);
    assert_eq!(run(17), whole);
    assert_eq!(run(64), whole);
}

/// The layer-kind prefill split appears in a row's timing only once a
/// profiled span added to it; an unprofiled prefill leaves no times behind.
#[test]
fn prefill_layer_times_report_only_when_profiled() {
    use super::decoder::{prefill_profile_enabled, take_prefill_layer_times, PrefillLayerTimes};
    use super::model::NemotronRowTiming;
    use std::time::Duration;

    let mut timing = NemotronRowTiming::default();
    assert!(timing.to_json().get("prefill_by_layer_ms").is_none());
    timing.add_prefill_layers(PrefillLayerTimes {
        mamba: Duration::from_millis(3),
        moe: Duration::from_millis(5),
        attention: Duration::from_millis(1),
    });
    let json = timing.to_json();
    assert_eq!(json["prefill_by_layer_ms"]["mamba"], 3.0);
    assert_eq!(json["prefill_by_layer_ms"]["moe"], 5.0);
    assert_eq!(json["prefill_by_layer_ms"]["attention"], 1.0);

    // With IZWI_NEMOTRON_TRANSCRIBE_PREFILL_PROFILE=1 the same prefill
    // fills the split instead.
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let artifact = Arc::new(model.prepare(&test_audio(1.0, 3), request(4)).unwrap());
    let mut cache = caches(&Device::Cpu, &model, 1, 1024).remove(0);
    let mut state = model.new_row(artifact, 1024).unwrap();
    let span = state.prompt_tokens();
    model
        .retained_prefill(&mut state, &mut cache, span)
        .unwrap();
    assert_eq!(take_prefill_layer_times(), PrefillLayerTimes::default());
    let timing = state.timing();
    let split = timing.prefill_mamba_ms + timing.prefill_moe_ms + timing.prefill_attention_ms;
    if prefill_profile_enabled() {
        assert!(timing.prefill_mamba_ms > 0.0 && timing.prefill_moe_ms > 0.0);
        assert!(split <= timing.prefill_ms);
    } else {
        assert_eq!(split, 0.0);
    }
}

/// A row decoded beside other rows is bit-identical to the row decoded
/// alone, at every step.
#[test]
fn solo_and_concurrent_rows_decode_identically() {
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let clips = [test_audio(2.0, 3), test_audio(3.3, 4), test_audio(1.2, 5)];
    let solo = clips
        .iter()
        .map(|clip| {
            let mut cache = caches(&Device::Cpu, &model, 1, 1024).remove(0);
            model
                .transcribe_with_cache(clip, request(10), &mut cache, 48)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut rows_caches = caches(&Device::Cpu, &model, 3, 1024);
    let mut states = clips
        .iter()
        .map(|clip| {
            let artifact = Arc::new(model.prepare(clip, request(10)).unwrap());
            model.new_row(artifact, 1024).unwrap()
        })
        .collect::<Vec<_>>();
    // Interleave prefill spans across rows, then decode them together.
    loop {
        let mut progressed = false;
        for (state, cache) in states.iter_mut().zip(rows_caches.iter_mut()) {
            if state.prefill_cursor() < state.prompt_tokens() {
                let n = 31.min(state.prompt_tokens() - state.prefill_cursor());
                model.retained_prefill(state, cache, n).unwrap();
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    while states.iter().any(|s| !s.finished()) {
        let mut rows = states
            .iter_mut()
            .zip(rows_caches.iter_mut())
            .map(|(state, cache)| NemotronStepRow { state, cache })
            .collect::<Vec<_>>();
        model.retained_decode_batch(&mut rows).unwrap();
    }
    for (state, solo) in states.iter().zip(&solo) {
        let batched = model.finish_row(state).unwrap();
        assert_eq!(batched.transcript, solo.transcript);
        assert_eq!(batched.generated_tokens, solo.generated_tokens);
        assert_eq!(batched.finish_reason, solo.finish_reason);
    }
}

#[test]
fn a_rolled_back_quantum_replays_identically() {
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let audio = test_audio(2.0, 6);
    let artifact = Arc::new(model.prepare(&audio, request(6)).unwrap());
    let mut cache = caches(&Device::Cpu, &model, 1, 1024).remove(0);
    let mut state = model.new_row(artifact, 1024).unwrap();
    let span = 20;
    model
        .retained_prefill(&mut state, &mut cache, span)
        .unwrap();
    let checkpoint = state.begin_quantum(&cache);
    model
        .retained_prefill(&mut state, &mut cache, span)
        .unwrap();
    let mamba_after = state.prefill_cursor();
    state.rollback_quantum(&mut cache, &checkpoint).unwrap();
    assert_eq!(state.prefill_cursor(), span);
    assert_eq!(cache.context_len(), span);
    model
        .retained_prefill(&mut state, &mut cache, span)
        .unwrap();
    assert_eq!(state.prefill_cursor(), mamba_after);
    state.commit_quantum(&cache).unwrap();
}

#[test]
fn prompts_too_long_for_the_row_are_refused() {
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let artifact = Arc::new(model.prepare(&test_audio(4.0, 7), request(4)).unwrap());
    assert!(model
        .new_row(artifact.clone(), artifact.prompt_tokens())
        .is_err());
    assert!(model.prepare(&test_audio(0.01, 7), request(4)).is_err());
}

/// Rows of a packed prefill: each row's prefix (already in its cache) and
/// its packed span. Unequal spans, one of a single token, at different
/// cursors.
const PACKED_SHAPES: [(usize, usize); 4] = [(0, 23), (7, 1), (40, 9), (3, 31)];

/// Token ids for row `row`'s prefix (`seed` 0) or span (`seed` 1).
fn packed_ids(vocab: usize, row: usize, seed: usize, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| ((row * 7_919 + seed * 104_729 + i * 6_007) % (vocab - 1)) as u32)
        .collect()
}

/// Fresh rows of `shapes` on the model's device with their prefixes
/// prefilled one row at a time.
fn packed_rows(
    model: &NemotronTranscribeModel,
    shapes: &[(usize, usize)],
) -> (
    Vec<super::decoder::NemotronMambaState>,
    Vec<PhysicalPagedKvCache>,
) {
    let (decoder, device) = (model.decoder(), model.device());
    let mut caches = caches(device, model, shapes.len(), 256);
    let mut states = shapes
        .iter()
        .map(|_| super::decoder::NemotronMambaState::zeros(decoder.config(), device).unwrap())
        .collect::<Vec<_>>();
    for (row, &(prefix, _)) in shapes.iter().enumerate() {
        if prefix > 0 {
            let ids = packed_ids(decoder.config().vocab, row, 0, prefix);
            let embeds = decoder.embeddings().gather(&ids, device).unwrap();
            decoder
                .forward_span(&embeds, &mut states[row], &mut caches[row], None)
                .unwrap();
        }
    }
    (states, caches)
}

/// Row `row`'s span embeddings of `tokens` tokens.
fn span_embeds(model: &NemotronTranscribeModel, row: usize, tokens: usize) -> Tensor {
    let decoder = model.decoder();
    decoder
        .embeddings()
        .gather(
            &packed_ids(decoder.config().vocab, row, 1, tokens),
            model.device(),
        )
        .unwrap()
}

/// Runs every row's span of `shapes` as one packed prefill, every row
/// picking.
fn run_packed(
    model: &NemotronTranscribeModel,
    shapes: &[(usize, usize)],
    states: &mut [super::decoder::NemotronMambaState],
    caches: &mut [PhysicalPagedKvCache],
) -> crate::error::Result<super::decoder::PackedSpans> {
    let mut rows = states
        .iter_mut()
        .zip(caches.iter_mut())
        .enumerate()
        .map(|(row, (state, cache))| super::decoder::SpanRow {
            embeds: span_embeds(model, row, shapes[row].1),
            state,
            cache,
            pick: true,
        })
        .collect::<Vec<_>>();
    model.decoder().forward_spans(&mut rows)
}

/// A Mamba state's values, layer by layer.
fn mamba_values(state: &super::decoder::NemotronMambaState) -> Vec<f32> {
    let host = super::speech_transformer::host_values;
    state
        .conv
        .iter()
        .chain(&state.ssm)
        .flat_map(|t| host(t).unwrap())
        .collect()
}

/// The rows of `shapes` packed against the same spans run one row at a
/// time: the same greedy tokens and clocks, and hidden states, logits and
/// Mamba states within `tolerance` (relative L2). One row packed is its
/// sequential span bit for bit. Returns the worst error.
fn check_packed(model: &NemotronTranscribeModel, shapes: &[(usize, usize)], tolerance: f64) -> f64 {
    use crate::kernels::numeric_check::relative_l2;
    let decoder = model.decoder();
    let (mut states, mut caches) = packed_rows(model, shapes);
    let (mut alone_states, mut alone_caches) = packed_rows(model, shapes);
    let packed = run_packed(model, shapes, &mut states, &mut caches).unwrap();
    let mut alone_hidden = Vec::new();
    let mut alone_picks = Vec::new();
    for (row, &(_, span)) in shapes.iter().enumerate() {
        let hidden = decoder
            .forward_span(
                &span_embeds(model, row, span),
                &mut alone_states[row],
                &mut alone_caches[row],
                None,
            )
            .unwrap();
        alone_picks.push(Some(decoder.greedy(&hidden).unwrap()[0]));
        alone_hidden.push(hidden);
    }
    let alone_hidden = Tensor::cat(&alone_hidden, 0).unwrap();
    let host = super::speech_transformer::host_values;
    let tokens =
        |picks: &[Option<(u32, f32)>]| picks.iter().map(|p| p.unwrap().0).collect::<Vec<_>>();
    assert_eq!(tokens(&packed.picks), tokens(&alone_picks), "{shapes:?}");
    for (row, &(prefix, span)) in shapes.iter().enumerate() {
        assert_eq!(caches[row].context_len(), prefix + span);
        assert_eq!(states[row].cursor(), (prefix + span) as u64);
    }
    let hidden = relative_l2(
        &host(&packed.hidden).unwrap(),
        &host(&alone_hidden).unwrap(),
    );
    let logits = relative_l2(
        &host(&decoder.logits(&packed.hidden).unwrap()).unwrap(),
        &host(&decoder.logits(&alone_hidden).unwrap()).unwrap(),
    );
    // A NaN error is the worst error, never a zero one.
    let worst_of = |errors: &mut dyn Iterator<Item = f64>| {
        errors.fold(0f64, |worst, e| {
            if e.is_nan() {
                f64::INFINITY
            } else {
                worst.max(e)
            }
        })
    };
    let mamba = worst_of(
        &mut states
            .iter()
            .zip(&alone_states)
            .map(|(a, b)| relative_l2(&mamba_values(a), &mamba_values(b))),
    );
    eprintln!(
        "packed vs sequential on {:?} {shapes:?}: hidden {hidden:.3e}, logits {logits:.3e}, mamba {mamba:.3e}",
        model.device()
    );
    if shapes.len() == 1 {
        assert_eq!(host(&packed.hidden).unwrap(), host(&alone_hidden).unwrap());
        assert_eq!(mamba_values(&states[0]), mamba_values(&alone_states[0]));
        assert_eq!(packed.picks, alone_picks);
    }
    let worst = worst_of(&mut [hidden, logits, mamba].into_iter());
    assert!(worst <= tolerance, "{shapes:?}: {worst:.3e}");
    worst
}

/// Packed spans of rows at different cursors, with unequal lengths (one of
/// a single token) or equal ones (the stacked Mamba call), track the same
/// spans run one row at a time within GEMM rounding: each row keeps its
/// span's own Q8_0 kernel, and only the shared tile GEMMs and the router
/// see more rows. Run on the fused and the eager residual norms. On an
/// M-series CPU the worst relative L2 was 3e-8 (the hidden states of the
/// ragged rows; logits and Mamba states equal); a row moved from the
/// row-wise Q8_0 kernel to the tile kernel was 7e-3.
#[test]
fn packed_spans_track_sequential_spans() {
    use super::decoder::DecodePath;
    let dir = temp_checkpoint();
    let mut model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    for fused in [true, false] {
        if !fused {
            model.decoder_mut().force_decode_path(DecodePath::Eager {
                reason: "test".into(),
            });
        }
        check_packed(&model, &PACKED_SHAPES, 1e-5);
        check_packed(&model, &[(5, 12), (0, 12), (30, 12)], 1e-5);
        check_packed(&model, &[(9, 17)], 0.0);
    }
}

/// On Metal, packed spans with rows on both MoE paths (two spans long
/// enough for the GEMMs, two on the warp kernels) track the same spans run
/// one row at a time.
#[cfg(feature = "metal")]
#[test]
fn metal_packed_spans_track_sequential_spans() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("metal_packed_spans_track_sequential_spans")
    else {
        return;
    };
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &device).unwrap();
    let top_k = model.config().decoder.moe.top_k;
    let shapes = [(0, 70), (5, 1), (10, 20), (3, 64)];
    let gemm = shapes
        .iter()
        .map(|&(_, span)| {
            crate::kernels::moe_relu2::uses_gemm(&device, span, top_k, super::synthetic::EXPERTS)
        })
        .collect::<Vec<_>>();
    assert_eq!(gemm, [true, false, false, true]);
    check_packed(&model, &shapes, 1e-4);
    check_packed(&model, &[(5, 12), (0, 12), (30, 12)], 1e-4);
    check_packed(&model, &[(9, 17)], 0.0);
}

/// A packed prefill that fails moves no row: a failure after the first
/// attention layer (every row's KV written) or a row whose cache cannot
/// take its span leaves every row's Mamba state and KV clock where they
/// were, and the same quantum then runs as if it had never failed.
#[test]
fn a_failed_packed_prefill_moves_no_row() {
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let shapes = &PACKED_SHAPES;
    let (mut clean_states, mut clean_caches) = packed_rows(&model, shapes);
    let clean = run_packed(&model, shapes, &mut clean_states, &mut clean_caches).unwrap();

    let (mut states, mut caches) = packed_rows(&model, shapes);
    let before = states.iter().map(mamba_values).collect::<Vec<_>>();
    let unmoved = |states: &[super::decoder::NemotronMambaState],
                   caches: &[PhysicalPagedKvCache]| {
        for (row, &(prefix, _)) in shapes.iter().enumerate() {
            assert_eq!(caches[row].context_len(), prefix);
            assert_eq!(states[row].cursor(), prefix as u64);
            assert_eq!(mamba_values(&states[row]), before[row]);
        }
    };
    let attention = model.decoder().attention_layers()[0] as usize;
    model
        .decoder()
        .fail_packed_prefill_after(Some(attention + 1));
    let error = run_packed(&model, shapes, &mut states, &mut caches).unwrap_err();
    model.decoder().fail_packed_prefill_after(None);
    assert!(error.to_string().contains("a test failed"), "{error}");
    unmoved(&states, &caches);

    // The last row's cache holds one page: its span cannot be prepared.
    let a = model.decoder().config().attention;
    let mut small = test_physical_caches_for_model_layers_on(
        &Device::Cpu,
        DType::F32,
        model.decoder().attention_layers(),
        a.kv_heads,
        a.head_dim,
        16,
        1,
    )
    .remove(0);
    let last = shapes.len() - 1;
    std::mem::swap(&mut caches[last], &mut small);
    let mut fresh_last =
        super::decoder::NemotronMambaState::zeros(model.decoder().config(), &Device::Cpu).unwrap();
    std::mem::swap(&mut states[last], &mut fresh_last);
    assert!(run_packed(&model, shapes, &mut states, &mut caches).is_err());
    std::mem::swap(&mut caches[last], &mut small);
    std::mem::swap(&mut states[last], &mut fresh_last);
    unmoved(&states, &caches);

    let again = run_packed(&model, shapes, &mut states, &mut caches).unwrap();
    let host = super::speech_transformer::host_values;
    assert_eq!(host(&again.hidden).unwrap(), host(&clean.hidden).unwrap());
    assert_eq!(again.picks, clean.picks);
    for (state, clean) in states.iter().zip(&clean_states) {
        assert_eq!(mamba_values(state), mamba_values(clean));
    }
}

/// Rows prefilled in packed quanta (each row's spans on its own grid,
/// prompts ending in different quanta, one row joining late) transcribe as
/// they do alone, and the diagnostics count the packed quanta. With the
/// switch off the rows run one after another, and a quantum of one row is
/// never packed.
#[test]
fn packed_prefill_quanta_transcribe_as_rows_alone() {
    let dir = temp_checkpoint();
    let mut model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    assert!(model.batched_prefill());
    let clips = [
        test_audio(2.0, 3),
        test_audio(3.3, 4),
        test_audio(1.2, 5),
        test_audio(2.7, 9),
    ];
    let span = 16;
    let solo = clips
        .iter()
        .map(|clip| {
            let mut cache = caches(&Device::Cpu, &model, 1, 1024).remove(0);
            model
                .transcribe_with_cache(clip, request(10), &mut cache, span)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let packed_quanta = |model: &NemotronTranscribeModel| {
        model.diagnostics_summary()["batched_prefill"]["packed_quanta"]
            .as_u64()
            .unwrap()
    };
    let run = |model: &NemotronTranscribeModel| {
        let mut rows_caches = caches(&Device::Cpu, model, clips.len(), 1024);
        let mut states = clips
            .iter()
            .map(|clip| {
                let artifact = Arc::new(model.prepare(clip, request(10)).unwrap());
                model.new_row(artifact, 1024).unwrap()
            })
            .collect::<Vec<_>>();
        let mut quantum = 0;
        loop {
            // The last row joins from the third quantum.
            let mut rows = states
                .iter_mut()
                .zip(rows_caches.iter_mut())
                .enumerate()
                .filter(|(row, (state, _))| {
                    (*row + 1 < clips.len() || quantum >= 2)
                        && state.prefill_cursor() < state.prompt_tokens()
                })
                .map(|(_, (state, cache))| {
                    let span = span.min(state.prompt_tokens() - state.prefill_cursor());
                    super::model::NemotronPrefillRow { state, cache, span }
                })
                .collect::<Vec<_>>();
            if rows.is_empty() && quantum >= 2 {
                break;
            }
            model.retained_prefill_batch(&mut rows).unwrap();
            quantum += 1;
        }
        while states.iter().any(|s| !s.finished()) {
            let mut rows = states
                .iter_mut()
                .zip(rows_caches.iter_mut())
                .map(|(state, cache)| NemotronStepRow { state, cache })
                .collect::<Vec<_>>();
            model.retained_decode_batch(&mut rows).unwrap();
        }
        for (state, solo) in states.iter().zip(&solo) {
            let batched = model.finish_row(state).unwrap();
            assert_eq!(batched.transcript, solo.transcript);
            assert_eq!(batched.generated_tokens, solo.generated_tokens);
            assert_eq!(batched.finish_reason, solo.finish_reason);
            assert_eq!(
                batched.timing.prefill_spans as usize,
                state.prompt_tokens().div_ceil(span)
            );
        }
    };
    // Every solo span was a quantum of one row.
    assert_eq!(packed_quanta(&model), 0);
    run(&model);
    let packed = packed_quanta(&model);
    assert!(packed > 0);
    let rows = model.diagnostics_summary()["batched_prefill"]["packed_rows"]
        .as_u64()
        .unwrap();
    assert!(rows >= 2 * packed, "{rows} rows in {packed} quanta");
    model.set_batched_prefill(false);
    run(&model);
    assert_eq!(packed_quanta(&model), packed);
}

/// The adapter leases each prefill row a whole span's activations plus
/// 160 MiB (`runtime/adapters/loaded/asr.rs`); in the published geometry
/// that covers what a packed quantum of rows holds, at every span and row
/// count the engine runs.
#[test]
fn the_adapter_lease_covers_a_packed_prefill_quantum() {
    let config = super::config::tests::published();
    for span in [16, 192, 512, 2_048] {
        let lease = span as u64 * 128 * 1024 + 160 * 1024 * 1024;
        for rows in [1usize, 2, 8, 32] {
            let packed = super::model::prefill_workspace_bytes(&config, &vec![span; rows]);
            assert!(packed <= rows as u64 * lease, "{span} x {rows}: {packed}");
        }
        // Shorter last spans hold less than whole ones.
        let ragged = super::model::prefill_workspace_bytes(&config, &[span, 1, span / 2]);
        assert!(ragged <= 3 * lease);
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_loads_the_synthetic_checkpoint_and_tracks_the_cpu_prefill() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_loads_the_synthetic_checkpoint_and_tracks_the_cpu_prefill",
    ) else {
        return;
    };
    let dir = temp_checkpoint();
    let cpu = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let metal = NemotronTranscribeModel::load_on(dir.path(), None, &device).unwrap();
    let summary = metal.diagnostics_summary();
    assert!(summary["mamba2_scan_check"].as_f64().unwrap() < 1e-4);
    // Metal runs the device-routed MoE kernels over stacked experts, checked
    // at load against the checkpoint's expert bytes; the shared expert's
    // projections are still checked one by one.
    assert_eq!(summary["moe_path"], "device_routed");
    assert!(summary["moe_stacked_check"].as_f64().unwrap() < 1e-4);
    assert_eq!(summary["projection_checks"].as_array().unwrap().len(), 8);
    let audio = test_audio(2.0, 8);
    let hidden = |model: &NemotronTranscribeModel, device: &Device| {
        let artifact = Arc::new(model.prepare(&audio, request(2)).unwrap());
        let mut cache = caches(device, model, 1, 1024).remove(0);
        let mut state = model.new_row(artifact, 1024).unwrap();
        let mut trace = super::decoder::DecoderTrace::default();
        let request = super::decoder::TraceRequest {
            routes: false,
            hidden_after_layers: vec![4],
        };
        let n = state.prompt_tokens();
        model
            .retained_prefill_traced(&mut state, &mut cache, n, Some((&request, &mut trace)))
            .unwrap();
        trace.hidden.remove(0).1
    };
    let a = hidden(&cpu, &Device::Cpu);
    let b = hidden(&metal, &device);
    // CPU runs Q8_0 weights, Metal the exact FP8 values.
    let error = crate::kernels::numeric_check::relative_l2(&b, &a);
    assert!(error < 5e-2, "{error}");
}

/// A request with word timestamps runs the CTC stage on the synthetic
/// artifact (on CPU and Metal): every word is timed inside the recording,
/// in order, and the speaker turns take their times from their own words.
#[test]
fn word_timestamps_run_the_ctc_stage_and_time_the_turns() {
    let dir = temp_checkpoint();
    #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
    let mut devices = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    let metal = crate::backends::metal_test_device_or_skip(
        "word_timestamps_run_the_ctc_stage_and_time_the_turns",
    );
    #[cfg(feature = "metal")]
    if let Some((device, _)) = &metal {
        devices.push(device.clone());
    }
    for device in devices {
        let model = NemotronTranscribeModel::load_on(dir.path(), None, &device).unwrap();
        assert!(model.supports_word_timestamps());
        assert_eq!(model.diagnostics_summary()["ctc_timestamp_head"], true);
        let audio = test_audio(3.0, 1);
        let mut cache = caches(&device, &model, 1, 1024).remove(0);
        let mut timed = request(16);
        timed.word_timestamps = true;
        let result = model
            .transcribe_with_cache(&audio, timed, &mut cache, 64)
            .unwrap();
        let timestamps = result
            .timestamps
            .as_ref()
            .expect("timestamps were asked for");
        eprintln!("{device:?}: {:?} → {timestamps:?}", result.transcript.raw);
        assert!(timestamps.error.is_none(), "{:?}", timestamps.error);
        assert!(!timestamps.words.is_empty());
        let mut previous = 0.0;
        for word in &timestamps.words {
            assert!(
                word.start >= previous - 1e-9 && word.end >= word.start,
                "{word:?}"
            );
            assert!(word.end <= 3.0 + 1e-6, "{word:?}");
            previous = word.start;
        }
        let turns = result.timed_turns();
        assert_eq!(turns.len(), result.transcript.turns.len());
        assert!(turns.iter().any(|turn| turn.start.is_some()), "{turns:?}");
        // Without the request flag the stage does not run.
        let mut cache = caches(&device, &model, 1, 1024).remove(0);
        let plain = model
            .transcribe_with_cache(&audio, request(16), &mut cache, 64)
            .unwrap();
        assert!(plain.timestamps.is_none());
        assert_eq!(plain.transcript, result.transcript);
    }
}

/// The fused decode step (fused norms, Mamba step kernels, device greedy)
/// transcribes as the eager composition does, and the summary names the
/// path.
#[test]
fn the_fused_decode_transcribes_as_the_eager_one() {
    use super::decoder::DecodePath;
    let dir = temp_checkpoint();
    let mut model = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    assert_eq!(model.diagnostics_summary()["decode_path"]["path"], "fused");
    let audio = test_audio(2.5, 3);
    let run = |model: &NemotronTranscribeModel| {
        let mut cache = caches(&Device::Cpu, model, 1, 1024).remove(0);
        model
            .transcribe_with_cache(&audio, request(16), &mut cache, 64)
            .unwrap()
    };
    let fused = run(&model);
    model.decoder_mut().force_decode_path(DecodePath::Eager {
        reason: "test".into(),
    });
    let eager = run(&model);
    assert_eq!(fused.transcript, eager.transcript);
    assert_eq!(fused.generated_tokens, eager.generated_tokens);
}

/// On CUDA the fused step's stretches are captured, verified and replayed,
/// and the transcript is the eager stretches' own.
#[cfg(feature = "cuda")]
#[test]
fn cuda_decode_graphs_transcribe_as_the_eager_stretches() {
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    let dir = temp_checkpoint();
    let mut model = NemotronTranscribeModel::load_on(dir.path(), None, &device).unwrap();
    let audio = test_audio(2.5, 3);
    let run = |model: &NemotronTranscribeModel| {
        let mut cache = caches(&device, model, 1, 1024).remove(0);
        model
            .transcribe_with_cache(&audio, request(16), &mut cache, 64)
            .unwrap()
    };
    let graphed = run(&model);
    let graphs = model.diagnostics_summary()["decode_graphs"].clone();
    assert!(graphs["off"].is_null(), "{graphs}");
    assert!(graphs["verified"].as_u64().unwrap() >= 4, "{graphs}");
    assert!(graphs["replays"].as_u64().unwrap() > 0, "{graphs}");
    model.decoder_mut().disable_graphs();
    let eager = run(&model);
    assert_eq!(graphed.transcript, eager.transcript);
    assert_eq!(graphed.generated_tokens, eager.generated_tokens);
}

/// On Metal the load-time kernel check passes, so decode runs fused, and a
/// row decodes the same alone and beside another.
#[cfg(feature = "metal")]
#[test]
fn metal_decodes_fused_and_rows_alone_and_together_alike() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_decodes_fused_and_rows_alone_and_together_alike",
    ) else {
        return;
    };
    let dir = temp_checkpoint();
    let model = NemotronTranscribeModel::load_on(dir.path(), None, &device).unwrap();
    let summary = model.diagnostics_summary();
    assert_eq!(summary["decode_path"]["path"], "fused", "{summary}");
    let audio = [test_audio(2.0, 4), test_audio(2.6, 5)];
    let artifacts = audio
        .iter()
        .map(|a| Arc::new(model.prepare(a, request(10)).unwrap()))
        .collect::<Vec<_>>();
    let decode = |rows: &[usize]| {
        let mut caches = caches(&device, &model, rows.len(), 1024);
        let mut states = rows
            .iter()
            .map(|&i| {
                let context = artifacts[i].max_context();
                model.new_row(artifacts[i].clone(), context).unwrap()
            })
            .collect::<Vec<_>>();
        for (state, cache) in states.iter_mut().zip(caches.iter_mut()) {
            while state.prefill_cursor() < state.prompt_tokens() {
                let n = 64.min(state.prompt_tokens() - state.prefill_cursor());
                model.retained_prefill(state, cache, n).unwrap();
            }
        }
        while states.iter().any(|s| !s.finished()) {
            let mut step = states
                .iter_mut()
                .zip(caches.iter_mut())
                .map(|(state, cache)| NemotronStepRow { state, cache })
                .collect::<Vec<_>>();
            model.retained_decode_batch(&mut step).unwrap();
        }
        states
            .iter()
            .map(|s| model.finish_row(s).unwrap().transcript)
            .collect::<Vec<_>>()
    };
    let together = decode(&[0, 1]);
    assert_eq!(decode(&[0])[0], together[0]);
    assert_eq!(decode(&[1])[0], together[1]);
}
