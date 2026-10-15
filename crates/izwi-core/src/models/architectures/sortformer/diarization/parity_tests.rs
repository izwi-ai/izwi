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
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
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
            ts_vad_post_processing(probs, speaker, &params, frame_repeat)
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
    let expected = reference_segments(&run["segments"]);
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
            let expected = reference_segments(&run["segments"]);
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
            .chunks_exact(10)
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
    let expected = reference_segments(&run["segments"]);
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
    let expected = reference_segments(&run["segments"]);
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
