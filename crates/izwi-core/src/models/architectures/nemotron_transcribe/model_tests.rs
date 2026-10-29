//! End-to-end tests on the synthetic checkpoint: load, census, the retained
//! row API, span and batch invariance, and device agreement.

use std::sync::Arc;

use candle_core::{DType, Device};

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

#[cfg(feature = "metal")]
#[test]
fn metal_loads_the_synthetic_checkpoint_and_tracks_the_cpu_prefill() {
    let Ok(device) = Device::new_metal(0) else {
        return;
    };
    let dir = temp_checkpoint();
    let cpu = NemotronTranscribeModel::load_on(dir.path(), None, &Device::Cpu).unwrap();
    let metal = NemotronTranscribeModel::load_on(dir.path(), None, &device).unwrap();
    assert!(
        metal.diagnostics_summary()["mamba2_scan_check"]
            .as_f64()
            .unwrap()
            < 1e-4
    );
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
    if let Ok(device) = Device::new_metal(0) {
        devices.push(device);
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
