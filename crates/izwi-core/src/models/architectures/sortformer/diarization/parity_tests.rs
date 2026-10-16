//! Real-checkpoint probability parity against the NeMo references in
//! `tests/fixtures/sortformer_parity/` (see the README there).
//!
//! Each test needs the local checkpoint and the 16 kHz inputs rebuilt by
//! `scripts/fixtures/prepare_sortformer_parity_audio.py`; it skips when
//! either is absent. Every streaming profile is set explicitly from the
//! fixture, so the tests pin the graph and the streaming arithmetic, not the
//! default profile. Segments come from NeMo's default post-processing
//! (onset = offset = 0.5, no padding or minimum durations) on raw channels,
//! which is what the references record.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

/// Largest per-frame |p_izwi - p_nemo| the parity tests accept.
const PROB_TOLERANCE: f32 = 2e-3;
/// Segment boundaries sit on the 10 ms grid; half a step absorbs f32 slop.
const SEGMENT_TOLERANCE_SECS: f32 = 0.005;

fn models_root() -> PathBuf {
    std::env::var("IZWI_MODELS_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models")
        })
}

fn load_local_model(variant: ModelVariant, archive: &str) -> Option<SortformerDiarizerModel> {
    let model_dir = models_root().join(variant.dir_name());
    if !model_dir.join(archive).exists() {
        eprintln!(
            "Skipping parity test, model not found at {}",
            model_dir.display()
        );
        return None;
    }
    // Loading resolves the streaming profile from the environment.
    let _env = crate::env_test_lock().lock().unwrap();
    Some(
        SortformerDiarizerModel::load(&model_dir, variant, DeviceProfile::cpu())
            .expect("local Sortformer checkpoint should load"),
    )
}

fn load_v21() -> Option<SortformerDiarizerModel> {
    load_local_model(
        ModelVariant::DiarStreamingSortformer4SpkV21,
        "diar_streaming_sortformer_4spk-v2.1.nemo",
    )
}

fn load_nemotron3() -> Option<SortformerDiarizerModel> {
    load_local_model(
        ModelVariant::Nemotron3Diarization,
        "Nemotron-3-Diarization.nemo",
    )
}

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sortformer_parity")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

/// The exact 16 kHz samples the reference consumed, as raw f32le.
fn parity_audio(name: &str) -> Option<Vec<f32>> {
    let dir = std::env::var("IZWI_SORTFORMER_PARITY_AUDIO_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/sortformer-parity-audio")
        });
    let path = dir.join(name);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!(
            "Skipping parity test, {} is missing; run \
             scripts/fixtures/prepare_sortformer_parity_audio.py",
            path.display()
        );
        return None;
    };
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not f32le",
        path.display()
    );
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    )
}

fn usize_field(value: &Value, key: &str) -> usize {
    value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("fixture field {key} missing")) as usize
}

fn f32_rows(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .expect("probability rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("probability row")
                .iter()
                .map(|p| p.as_f64().expect("probability") as f32)
                .collect()
        })
        .collect()
}

fn reference_segments(value: &Value) -> Vec<(f32, f32, usize)> {
    let mut segments = value
        .as_array()
        .expect("segments")
        .iter()
        .map(|segment| {
            (
                segment[0].as_f64().expect("start") as f32,
                segment[1].as_f64().expect("end") as f32,
                segment[2].as_u64().expect("speaker") as usize,
            )
        })
        .collect::<Vec<_>>();
    sort_segments(&mut segments);
    segments
}

/// The references ran NeMo's legacy `ts_vad_post_processing` loop over 10 ms
/// repeats, which ends a segment still active at the last of `frames` rows
/// at `frames x frame_secs - 10 ms`. Current NeMo (`binarization_vectorized`),
/// which izwi follows (SRT-11), ends it at `frames x frame_secs`; nothing
/// else differs at onset = offset without padding.
fn as_vectorized(
    mut segments: Vec<(f32, f32, usize)>,
    frames: usize,
    frame_secs: f32,
) -> Vec<(f32, f32, usize)> {
    let end = frames as f32 * frame_secs;
    for segment in &mut segments {
        if (segment.1 - (end - TS_VAD_FRAME_LENGTH_SECS)).abs() <= SEGMENT_TOLERANCE_SECS {
            segment.1 = end;
        }
    }
    segments
}

fn sort_segments(segments: &mut [(f32, f32, usize)]) {
    segments.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
}

/// The loaded model's streaming config with the fixture's profile applied.
fn fixture_streaming_config(
    model: &SortformerDiarizerModel,
    streaming: &Value,
) -> SortformerStreamingConfig {
    let mut cfg = model.model.streaming.expect("bounded streaming profile");
    cfg.chunk_len = usize_field(streaming, "chunk_len");
    cfg.chunk_right_context = usize_field(streaming, "chunk_right_context");
    cfg.fifo_len = usize_field(streaming, "fifo_len");
    cfg.spkcache_update_period = usize_field(streaming, "spkcache_update_period");
    cfg.spkcache_len = usize_field(streaming, "spkcache_len");
    if streaming.get("chunk_left_context").is_some() {
        cfg.chunk_left_context = usize_field(streaming, "chunk_left_context");
    }
    cfg.validate()
        .expect("fixture streaming profile should validate")
}

fn probabilities(
    model: &SortformerDiarizerModel,
    samples: &[f32],
    cfg: SortformerStreamingConfig,
) -> Vec<Vec<f32>> {
    let feature_frames = model.model.preprocessor.feature_frame_count(samples.len());
    model
        .model
        .infer_speaker_probabilities_streaming(samples, feature_frames, cfg, None)
        .expect("streaming inference should run")
}

/// NeMo's default post-processing on every raw channel.
fn nemo_default_segments(
    probs: &[Vec<f32>],
    num_speakers: usize,
    frame_repeat: usize,
) -> Vec<(f32, f32, usize)> {
    let params = PostProcessingParams {
        onset: 0.5,
        offset: 0.5,
        pad_onset: 0.0,
        pad_offset: 0.0,
        min_duration_on: 0.0,
        min_duration_off: 0.0,
        filter_speech_first: true,
    };
    let mut segments = (0..num_speakers)
        .flat_map(|speaker| {
            predlist_to_timestamps(probs, speaker, &params, frame_repeat)
                .into_iter()
                .map(move |(start, end)| (start, end, speaker))
        })
        .collect::<Vec<_>>();
    sort_segments(&mut segments);
    segments
}

/// NaN-safe: a NaN difference counts as infinitely far.
fn max_abs_diff(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "probability row count");
    let mut max = 0.0f32;
    for (a_row, e_row) in actual.iter().zip(expected) {
        assert_eq!(a_row.len(), e_row.len(), "probability row width");
        for (a, e) in a_row.iter().zip(e_row) {
            let diff = (a - e).abs();
            if diff.is_nan() {
                return f32::INFINITY;
            }
            max = max.max(diff);
        }
    }
    max
}

fn segments_equal(actual: &[(f32, f32, usize)], expected: &[(f32, f32, usize)]) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(a, e)| {
            a.2 == e.2
                && (a.0 - e.0).abs() <= SEGMENT_TOLERANCE_SECS
                && (a.1 - e.1).abs() <= SEGMENT_TOLERANCE_SECS
        })
}

struct ParityOutcome {
    label: String,
    max_abs: f32,
    segments_equal: bool,
    detail: String,
}

impl ParityOutcome {
    fn report(outcomes: &[ParityOutcome]) {
        for outcome in outcomes {
            println!(
                "parity {}: max|dp| = {:.3e}, segments equal = {}",
                outcome.label, outcome.max_abs, outcome.segments_equal
            );
        }
        let failures = outcomes
            .iter()
            .filter(|outcome| {
                // NaN-safe: a NaN maximum fails as surely as a large one.
                outcome.max_abs.is_nan()
                    || outcome.max_abs > PROB_TOLERANCE
                    || !outcome.segments_equal
            })
            .map(|outcome| format!("{}: {}", outcome.label, outcome.detail))
            .collect::<Vec<_>>();
        assert!(
            failures.is_empty(),
            "parity failures (tolerance {PROB_TOLERANCE:e}):\n{}",
            failures.join("\n")
        );
    }
}

fn v21_outcome(
    model: &SortformerDiarizerModel,
    samples: &[f32],
    label: &str,
    run: &Value,
) -> ParityOutcome {
    let cfg = fixture_streaming_config(model, &run["streaming"]);
    let probs = probabilities(model, samples, cfg);
    let expected_probs = f32_rows(&run["probs_80ms"]);
    let max_abs = max_abs_diff(&probs, &expected_probs);
    let actual = nemo_default_segments(&probs, model.model.num_speakers, TS_VAD_UNIT_FRAME_COUNT);
    let expected = as_vectorized(reference_segments(&run["segments"]), probs.len(), 0.08);
    ParityOutcome {
        label: label.to_string(),
        max_abs,
        segments_equal: segments_equal(&actual, &expected),
        detail: format!("izwi {actual:?}\nnemo {expected:?}"),
    }
}

fn v21_fixture_parity(fixture_name: &str, profiles: &[&str]) {
    let reference = fixture(fixture_name);
    let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
        return;
    };
    let Some(model) = load_v21() else {
        return;
    };
    let outcomes = profiles
        .iter()
        .map(|profile| {
            v21_outcome(
                &model,
                &samples,
                &format!("v2.1 {fixture_name} {profile}"),
                &reference["runs"][*profile],
            )
        })
        .collect::<Vec<_>>();
    ParityOutcome::report(&outcomes);
}

#[test]
#[ignore = "requires the local Sortformer v2.1 checkpoint and prepared parity audio"]
fn sortformer_v21_matches_nemo_probabilities_on_diarization() {
    v21_fixture_parity("v21-diarization.json", &["model", "card_high"]);
}

#[test]
#[ignore = "requires the local Sortformer v2.1 checkpoint and prepared parity audio"]
fn sortformer_v21_matches_nemo_probabilities_on_diarization_2() {
    v21_fixture_parity("v21-diarization-2.json", &["model", "card_high"]);
}

#[test]
#[ignore = "requires the local Nemotron-3-Diarization checkpoint and prepared parity audio"]
fn nemotron3_matches_nemo_probabilities_on_the_10s_fixture() {
    let reference = fixture("nemotron3-diarization-10s.json");
    let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
        return;
    };
    let Some(model) = load_nemotron3() else {
        return;
    };
    let outcomes = ["test_small", "test_small_rc"]
        .iter()
        .map(|profile| {
            let run = &reference["runs"][*profile];
            let cfg = fixture_streaming_config(&model, &run["streaming"]);
            let probs = probabilities(&model, &samples, cfg);
            let max_abs = max_abs_diff(&probs, &f32_rows(&run["probs_10ms"]));
            let actual = nemo_default_segments(&probs, model.model.num_speakers, 1);
            let expected = as_vectorized(
                reference_segments(&run["segments"]),
                probs.len(),
                TS_VAD_FRAME_LENGTH_SECS,
            );
            ParityOutcome {
                label: format!("nemotron3 diarization-10s {profile}"),
                max_abs,
                segments_equal: segments_equal(&actual, &expected),
                detail: format!("izwi {actual:?}\nnemo {expected:?}"),
            }
        })
        .collect::<Vec<_>>();
    ParityOutcome::report(&outcomes);
}

/// The diarization-2 reference keeps 100 ms means of the active channels
/// plus each channel's maximum; compare izwi's 10 ms rows on that basis.
///
/// The `yaml` profile is compared only up to its first speaker-cache
/// compression (two 264-frame chunks, 42.24 s). That compression meets an
/// exact score tie across the top-k cut (4 of 9 tied candidates kept), and
/// which tied frames `torch.topk` keeps is implementation-defined; izwi
/// keeps the lowest indices. The full-file deviation is still printed.
#[test]
#[ignore = "requires the local Nemotron-3-Diarization checkpoint and prepared parity audio"]
fn nemotron3_matches_nemo_probabilities_on_diarization_2() {
    let reference = fixture("nemotron3-diarization-2.json");
    let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
        return;
    };
    let Some(model) = load_nemotron3() else {
        return;
    };
    let yaml_pre_compression_rows = 2 * 264 * TS_VAD_UNIT_FRAME_COUNT / 10;
    let outcomes = [
        ("yaml", Some(yaml_pre_compression_rows)),
        ("card_offline", None),
    ]
    .iter()
    .map(|(profile, compared_rows)| {
        nemotron3_pooled_outcome(
            &model,
            &samples,
            profile,
            &reference["runs"][*profile],
            *compared_rows,
        )
    })
    .collect::<Vec<_>>();
    ParityOutcome::report(&outcomes);
}

fn nemotron3_pooled_outcome(
    model: &SortformerDiarizerModel,
    samples: &[f32],
    profile: &str,
    run: &Value,
    compared_rows: Option<usize>,
) -> ParityOutcome {
    let cfg = fixture_streaming_config(model, &run["streaming"]);
    let probs = probabilities(model, samples, cfg);
    let pooled = run["probs_100ms_active_speakers"]
        .as_object()
        .expect("pooled probabilities");
    let mut full_max_abs = 0.0f32;
    let mut compared_max_abs = 0.0f32;
    for (speaker, expected) in pooled {
        let speaker = speaker.parse::<usize>().expect("speaker id");
        let actual = probs
            .as_chunks::<10>()
            .0
            .iter()
            .map(|rows| vec![rows.iter().map(|row| row[speaker]).sum::<f32>() / 10.0])
            .collect::<Vec<_>>();
        let expected = expected
            .as_array()
            .expect("pooled rows")
            .iter()
            .map(|p| vec![p.as_f64().expect("probability") as f32])
            .collect::<Vec<_>>();
        full_max_abs = full_max_abs.max(max_abs_diff(&actual, &expected));
        let rows = compared_rows.unwrap_or(expected.len());
        compared_max_abs = compared_max_abs.max(max_abs_diff(&actual[..rows], &expected[..rows]));
    }
    let actual = nemo_default_segments(&probs, model.model.num_speakers, 1);
    let expected = as_vectorized(
        reference_segments(&run["segments"]),
        probs.len(),
        TS_VAD_FRAME_LENGTH_SECS,
    );
    let mut segments_match = segments_equal(&actual, &expected);
    if compared_rows.is_none() {
        let channel_max = (0..model.model.num_speakers)
            .map(|speaker| vec![probs.iter().map(|row| row[speaker]).fold(0.0f32, f32::max)])
            .collect::<Vec<_>>();
        let expected_max = run["summary"]["per_speaker_max"]
            .as_array()
            .expect("per-speaker max")
            .iter()
            .map(|p| vec![p.as_f64().expect("probability") as f32])
            .collect::<Vec<_>>();
        compared_max_abs = compared_max_abs.max(max_abs_diff(&channel_max, &expected_max));
    } else {
        println!(
            "parity nemotron3 diarization-2 {profile} (full file, not asserted): \
             max|dp| = {full_max_abs:.3e}, segments equal = {segments_match}"
        );
        segments_match = true;
    }
    ParityOutcome {
        label: format!("nemotron3 diarization-2 {profile}"),
        max_abs: compared_max_abs,
        segments_equal: segments_match,
        detail: format!("izwi {actual:?}\nnemo {expected:?}"),
    }
}

/// Runs without per-frame reference rows record each channel's mean, max
/// and count of frames above 0.5; compare those, plus the segments.
fn v21_summary_outcome(
    model: &SortformerDiarizerModel,
    samples: &[f32],
    label: &str,
    run: &Value,
) -> ParityOutcome {
    let cfg = fixture_streaming_config(model, &run["streaming"]);
    let probs = probabilities(model, samples, cfg);
    assert_eq!(probs.len(), usize_field(run, "output_frames"));
    let mut max_abs = 0.0f32;
    let mut counts_equal = true;
    let mut counts = Vec::new();
    for speaker in 0..model.model.num_speakers {
        let summary = &run["probs_summary"][format!("spk{speaker}")];
        let channel = probs.iter().map(|row| row[speaker]).collect::<Vec<_>>();
        let mean = channel.iter().sum::<f32>() / channel.len() as f32;
        let max = channel.iter().copied().fold(0.0f32, f32::max);
        let above = channel.iter().filter(|p| **p > 0.5).count();
        let expected = |key: &str| summary[key].as_f64().expect("summary value") as f32;
        max_abs = max_abs.max(max_abs_diff(
            &[vec![mean, max]],
            &[vec![expected("mean"), expected("max")]],
        ));
        counts_equal &= above == usize_field(summary, "frames_gt_0_5");
        counts.push(above);
    }
    let actual = nemo_default_segments(&probs, model.model.num_speakers, TS_VAD_UNIT_FRAME_COUNT);
    let expected = as_vectorized(reference_segments(&run["segments"]), probs.len(), 0.08);
    ParityOutcome {
        label: label.to_string(),
        max_abs,
        segments_equal: counts_equal && segments_equal(&actual, &expected),
        detail: format!(
            "frames > 0.5 {counts:?} vs {}\nizwi {actual:?}\nnemo {expected:?}",
            run["probs_summary"]
        ),
    }
}

/// SRT-1: on sparse speech the cache compression must pick disabled
/// (-inf) slots; NeMo places them after every kept frame.
#[test]
#[ignore = "requires the local Sortformer v2.1 checkpoint and prepared parity audio"]
fn sortformer_v21_matches_nemo_cache_order_on_sparse_speech() {
    let reference = fixture("v21-sparse2.json");
    let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
        return;
    };
    let Some(model) = load_v21() else {
        return;
    };
    let outcomes = ["model", "card_high"]
        .iter()
        .map(|profile| {
            v21_summary_outcome(
                &model,
                &samples,
                &format!("v2.1 sparse2 {profile}"),
                &reference["runs"][*profile],
            )
        })
        .collect::<Vec<_>>();
    ParityOutcome::report(&outcomes);
}

/// SRT-2: the card low-latency profile (chunk 6 < FIFO 188) pops the
/// update period, not the whole FIFO.
#[test]
#[ignore = "requires the local Sortformer v2.1 checkpoint and prepared parity audio"]
fn sortformer_v21_matches_nemo_low_latency_fifo_pops_on_diarization_2() {
    let reference = fixture("v21-diarization-2.json");
    let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
        return;
    };
    let Some(model) = load_v21() else {
        return;
    };
    ParityOutcome::report(&[v21_summary_outcome(
        &model,
        &samples,
        "v2.1 diarization-2 card_low",
        &reference["runs"]["card_low"],
    )]);
}

/// `diarize` segments as `(start, end, label index)`, sorted.
fn diarize_segments(
    model: &SortformerDiarizerModel,
    samples: &[f32],
    config: &DiarizationConfig,
) -> Vec<(f32, f32, usize)> {
    let result = model
        .diarize(samples, TARGET_SAMPLE_RATE, config)
        .expect("diarize should run");
    let mut segments = result
        .segments
        .iter()
        .map(|segment| {
            let label = segment
                .speaker
                .strip_prefix("SPEAKER_")
                .and_then(|id| id.parse::<usize>().ok())
                .expect("SPEAKER_nn label");
            (segment.start_secs, segment.end_secs, label)
        })
        .collect::<Vec<_>>();
    sort_segments(&mut segments);
    segments
}

/// NeMo's raw channels relabelled in first-appearance order, as `diarize`
/// labels speakers, with ends clamped to the audio duration.
fn arrival_labelled(mut segments: Vec<(f32, f32, usize)>, duration: f32) -> Vec<(f32, f32, usize)> {
    let mut order = Vec::<usize>::new();
    for segment in &segments {
        if !order.contains(&segment.2) {
            order.push(segment.2);
        }
    }
    for segment in &mut segments {
        segment.1 = segment.1.min(duration);
        segment.2 = order
            .iter()
            .position(|channel| *channel == segment.2)
            .unwrap();
    }
    segments
}

fn without_vad_gate<T>(f: impl FnOnce() -> T) -> T {
    let _env = crate::env_test_lock().lock().unwrap();
    let key = "IZWI_SORTFORMER_VAD_GATE";
    let previous = std::env::var(key).ok();
    std::env::remove_var(key);
    let result = f();
    if let Some(value) = previous {
        std::env::set_var(key, value);
    }
    result
}

/// D3/SRT-4: with the VAD gate off (the default), the served `diarize`
/// path reproduces NeMo's segments under each card offline profile. The
/// gate used to fragment Nemotron-3 on diarization-2 into ~30 segments
/// against the reference's 13.
#[test]
#[ignore = "requires the local Sortformer checkpoints and prepared parity audio"]
fn diarize_without_the_vad_gate_matches_nemo_segments() {
    let mut outcomes = Vec::new();
    for (fixture_name, profile, load) in [
        (
            "v21-diarization.json",
            "card_high",
            load_v21 as fn() -> Option<SortformerDiarizerModel>,
        ),
        ("v21-diarization-2.json", "card_high", load_v21),
        (
            "nemotron3-diarization-2.json",
            "card_offline",
            load_nemotron3,
        ),
    ] {
        let reference = fixture(fixture_name);
        let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
            return;
        };
        let Some(mut model) = load() else {
            continue;
        };
        let run = &reference["runs"][profile];
        model.model.streaming = Some(fixture_streaming_config(&model, &run["streaming"]));
        let actual =
            without_vad_gate(|| diarize_segments(&model, &samples, &DiarizationConfig::default()));
        let duration = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
        // Rows at the reference's frame period (80 ms v2.1, 10 ms Nemotron-3).
        let frame_secs = reference["frame_secs"].as_f64().expect("frame_secs") as f32;
        let feature_frames = model.model.preprocessor.feature_frame_count(samples.len());
        let frames =
            feature_frames.div_ceil((frame_secs / TS_VAD_FRAME_LENGTH_SECS).round() as usize);
        let expected = arrival_labelled(
            as_vectorized(reference_segments(&run["segments"]), frames, frame_secs),
            duration,
        );
        println!(
            "diarize {fixture_name} {profile}: {} segments (reference {})",
            actual.len(),
            expected.len()
        );
        outcomes.push(ParityOutcome {
            label: format!("diarize {fixture_name} {profile} without the VAD gate"),
            max_abs: 0.0,
            segments_equal: segments_equal(&actual, &expected),
            detail: format!("izwi {actual:?}\nnemo {expected:?}"),
        });
    }
    ParityOutcome::report(&outcomes);
}

/// SRT-4: on quiet speech (diarization.wav at -35 dBFS peak) the opt-in
/// gate keeps every gate-off segment to within one encoded frame at each
/// end: it may only trim a padded edge, never drop speech.
#[test]
#[ignore = "requires the local Sortformer v2.1 checkpoint and prepared parity audio"]
fn opt_in_vad_gate_keeps_quiet_speech_within_one_frame() {
    let reference = fixture("v21-diarization.json");
    let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
        return;
    };
    let Some(mut model) = load_v21() else {
        return;
    };
    model.model.streaming = Some(fixture_streaming_config(
        &model,
        &reference["runs"]["card_high"]["streaming"],
    ));
    let peak = samples.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
    let gain = 10f32.powf(-35.0 / 20.0) / peak;
    let quiet = samples.iter().map(|s| s * gain).collect::<Vec<_>>();
    let config = |vad_gate| DiarizationConfig {
        vad_gate: Some(vad_gate),
        ..DiarizationConfig::default()
    };
    let ungated = diarize_segments(&model, &quiet, &config(false));
    let gated = diarize_segments(&model, &quiet, &config(true));
    println!(
        "quiet speech: {} segments ungated, {} gated",
        ungated.len(),
        gated.len()
    );

    let frame = 0.08 + SEGMENT_TOLERANCE_SECS;
    let uncovered = ungated
        .iter()
        .filter(|(start, end, speaker)| {
            // The gated pieces of this speaker inside the segment must span
            // it, less at most one frame at each end and no inner gap.
            let mut pieces = gated
                .iter()
                .filter(|piece| piece.2 == *speaker && piece.1 > *start && piece.0 < *end)
                .map(|piece| (piece.0.max(*start), piece.1.min(*end)))
                .collect::<Vec<_>>();
            pieces.sort_by(|a, b| a.0.total_cmp(&b.0));
            let Some(first) = pieces.first() else {
                return true;
            };
            let mut covered_to = first.1;
            for piece in &pieces[1..] {
                if piece.0 > covered_to + SEGMENT_TOLERANCE_SECS {
                    return true;
                }
                covered_to = covered_to.max(piece.1);
            }
            first.0 > start + frame || covered_to < end - frame
        })
        .collect::<Vec<_>>();
    assert!(
        uncovered.is_empty(),
        "the gate dropped quiet speech: {uncovered:?}\nungated {ungated:?}\ngated {gated:?}"
    );
}

/// Per-chunk CPU time and RTF of the streaming forward on diarization-2
/// (116 s) for v2.1 `model` and `card_high` and Nemotron-3 `card_offline`;
/// the best of two runs. `IZWI_SORTFORMER_BENCH_DUMP=<tag>` writes each
/// run's probabilities to `target/sortformer-bench/<run>-<tag>.f32`, and
/// `IZWI_SORTFORMER_BENCH_BASELINE=<tag>` prints the max |Δp| against
/// that dump, so a change can show it is bit-exact or bounded.
#[test]
#[ignore = "requires the local Sortformer checkpoints and prepared parity audio"]
fn sortformer_streaming_timings_on_diarization_2() {
    let dump_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/sortformer-bench");
    let dump_tag = std::env::var("IZWI_SORTFORMER_BENCH_DUMP").ok();
    let baseline_tag = std::env::var("IZWI_SORTFORMER_BENCH_BASELINE").ok();
    for (fixture_name, profile, load) in [
        (
            "v21-diarization-2.json",
            "model",
            load_v21 as fn() -> Option<SortformerDiarizerModel>,
        ),
        ("v21-diarization-2.json", "card_high", load_v21),
        (
            "nemotron3-diarization-2.json",
            "card_offline",
            load_nemotron3,
        ),
    ] {
        let reference = fixture(fixture_name);
        let Some(samples) = parity_audio(reference["audio"].as_str().expect("audio")) else {
            return;
        };
        let Some(model) = load() else {
            continue;
        };
        let cfg = fixture_streaming_config(&model, &reference["runs"][profile]["streaming"]);
        let feature_frames = model.model.preprocessor.feature_frame_count(samples.len());
        let chunks = plan_streaming_feature_chunks(feature_frames, cfg).len();
        let mut best = f64::INFINITY;
        let mut probs = Vec::new();
        for _ in 0..2 {
            let start = std::time::Instant::now();
            probs = probabilities(&model, &samples, cfg);
            best = best.min(start.elapsed().as_secs_f64());
        }
        let duration = samples.len() as f64 / f64::from(TARGET_SAMPLE_RATE);
        let run = format!("{}-{profile}", fixture_name.trim_end_matches(".json"));
        println!(
            "bench {run}: {chunks} chunks, {:.1} ms/chunk, total {best:.2} s, RTF {:.4}",
            best * 1e3 / chunks as f64,
            best / duration
        );
        // Production loading engages the fast paths: every chunk runs each
        // Conformer layer's depthwise conv through the shared op.
        let kernels = model.family_diagnostics()["kernels"].clone();
        println!("bench {run} kernels: {kernels}");
        if let Some(calls) = kernels.get("conformer_depthwise_conv1d_calls") {
            let expected = 2 * chunks * PRODUCTION_CONFORMER_LAYERS;
            assert_eq!(
                calls.as_u64(),
                Some(expected as u64),
                "{run} depthwise calls"
            );
        }
        if let Some(calls) = kernels.get("subsampling_depthwise_conv2d_calls") {
            let expected = 2 * chunks * 2;
            assert_eq!(calls.as_u64(), Some(expected as u64), "{run} conv2d calls");
        }
        // SRT-7: at most two host reads per chunk (the probabilities, and the
        // silent rows for v2.1's running silence mean).
        let path = model.family_diagnostics()["streaming_path"].clone();
        println!("bench {run} streaming path: {path}");
        if let Some(reads) = path.get("host_reads").and_then(serde_json::Value::as_u64) {
            assert!(reads <= 2 * 2 * chunks as u64, "{run}: {reads} host reads");
        }
        let flat = probs.iter().flatten().copied().collect::<Vec<f32>>();
        if let Some(tag) = &dump_tag {
            std::fs::create_dir_all(&dump_dir).expect("create bench dump dir");
            let bytes = flat
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<u8>>();
            std::fs::write(dump_dir.join(format!("{run}-{tag}.f32")), bytes)
                .expect("write bench dump");
        }
        if let Some(tag) = &baseline_tag {
            let path = dump_dir.join(format!("{run}-{tag}.f32"));
            let baseline = std::fs::read(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<f32>>();
            assert_eq!(baseline.len(), flat.len(), "{run} baseline length");
            let bit_exact = baseline
                .iter()
                .zip(&flat)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            let max_abs = max_abs_diff(std::slice::from_ref(&flat), &[baseline]);
            println!("bench {run} vs {tag}: max|dp| = {max_abs:.3e}, bit-exact = {bit_exact}");
        }
    }
}
