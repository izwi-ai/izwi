//! Parity with upstream: transformers 4.57.1 and `snac` 1.2.1 through
//! `scripts/fixtures/generate_maya1_reference.py` (fixtures in
//! `tests/fixtures/maya1`).
//!
//! The sampler test runs everywhere. The others need the checkpoint
//! (`IZWI_MAYA1_MODEL_DIR`, with `snac_24khz.pth` beside it) and are
//! ignored by default:
//!
//! ```text
//! IZWI_MAYA1_MODEL_DIR=… cargo test -p izwi-core --features metal --lib \
//!     maya1::reference_tests -- --ignored --test-threads=1
//! ```

use std::collections::BTreeMap;

use candle_core::{DType, Device};

use super::config::{Maya1LmConfig, HEAD_ROWS};
use super::fixture_support::{fixture, sample_error, Sample};
use super::frames::unpack_frames;
use super::noise::snac_noise;
use super::prompt::Maya1Tokenizer;
use super::sampling::{process_logits, HeadHistory, Maya1SamplingParams};
use super::snac::SnacDecoder;
use super::stream::{Maya1AudioStream, MAYA1_WINDOW_PLAN};
use super::trunk::Maya1ForwardRow;
use super::Maya1Runtime;
use crate::models::shared::noise::SplitMix64;

/// The checkpoint directory for the ignored weight tests.
fn model_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("IZWI_MAYA1_MODEL_DIR")
            .expect("set IZWI_MAYA1_MODEL_DIR to a Maya1-3B download (with snac_24khz.pth)"),
    )
}

#[derive(serde::Deserialize)]
struct SamplerCase {
    logits_state: u64,
    history: Vec<usize>,
    generated: usize,
    live: Vec<(usize, f32)>,
}

#[derive(serde::Deserialize)]
struct SamplerFixture {
    cases: Vec<SamplerCase>,
}

/// Transformers' processors (repetition penalty 1.1, minimum 28 tokens,
/// temperature 0.4, top-k 50, top-p 0.9) on the same head-only logits.
#[test]
fn sampler_matches_transformers_processors() {
    let fixture: SamplerFixture = fixture("sampler.json");
    let params = Maya1SamplingParams::default();
    for case in &fixture.cases {
        let mut logits = SplitMix64::new(case.logits_state)
            .standard_normals(HEAD_ROWS)
            .into_iter()
            .map(|value| value * 0.5)
            .collect::<Vec<_>>();
        logits[0] = 2.5;
        let mut history = HeadHistory::default();
        for &row in &case.history {
            history.record(row);
        }
        let scores = process_logits(&logits, &history, case.generated, &params).unwrap();
        let live = scores
            .iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .map(|(row, &score)| (row, score))
            .collect::<Vec<_>>();
        assert_eq!(
            live.iter().map(|&(row, _)| row).collect::<Vec<_>>(),
            case.live.iter().map(|&(row, _)| row).collect::<Vec<_>>(),
            "live rows after {} tokens",
            case.generated
        );
        for (&(row, actual), &(_, expected)) in live.iter().zip(&case.live) {
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "row {row}: {actual} vs {expected}"
            );
        }
    }
}

#[derive(serde::Deserialize)]
struct PromptCase {
    description: String,
    text: String,
    ids: Vec<u32>,
    text_tokens: usize,
}

#[derive(serde::Deserialize)]
struct PromptFixture {
    cases: Vec<PromptCase>,
}

#[test]
#[ignore = "needs the Maya1 checkpoint (IZWI_MAYA1_MODEL_DIR)"]
fn prompt_ids_match_upstream() {
    let tokenizer = Maya1Tokenizer::load(&model_dir().join("tokenizer.json")).unwrap();
    let fixture: PromptFixture = fixture("prompts.json");
    for case in &fixture.cases {
        let description = super::prompt::normalize_description(&case.description).unwrap();
        let text = super::prompt::normalize_text(&case.text).unwrap();
        assert_eq!(
            description, case.description,
            "fixture descriptions are normalised"
        );
        assert_eq!(
            tokenizer.prompt_ids(&description, &text).unwrap(),
            case.ids,
            "{}",
            case.text
        );
        assert_eq!(tokenizer.text_tokens(&text).unwrap(), case.text_tokens);
    }
}

#[derive(serde::Deserialize)]
struct SnacFixture {
    seed: u64,
    codes: Vec<u32>,
    whole: Sample,
    windows: Vec<[usize; 4]>,
    windowed: Sample,
    windowed_vs_whole_error: f32,
    noise_block0_first4: Vec<f32>,
}

/// SNAC with izwi's noise: the whole decode, and izwi's incremental windows
/// as frames arrive one by one, against upstream's module run on the same
/// windows.
#[test]
#[ignore = "needs the SNAC checkpoint (IZWI_MAYA1_MODEL_DIR)"]
fn snac_decoder_matches_upstream_whole_and_windowed() {
    let fixture: SnacFixture = fixture("snac.json");
    let snac = SnacDecoder::load(&model_dir().join("snac_24khz.pth"), &Device::Cpu).unwrap();
    let tokens = fixture
        .codes
        .iter()
        .enumerate()
        .map(|(index, &code)| super::config::AUDIO_CODE_START + (index % 7) as u32 * 4096 + code)
        .collect::<Vec<_>>();
    let codes = unpack_frames(&tokens);
    let frames = codes.frames();
    let noise = snac_noise(snac.config(), &[(0, &codes, fixture.seed)]);
    assert_eq!(&noise[0][..4], fixture.noise_block0_first4.as_slice());
    let started = std::time::Instant::now();
    let whole = snac.decode(&[&codes], &noise).unwrap().remove(0);
    eprintln!(
        "SNAC whole decode of {frames} frames ({:.2} s of audio): {:.0} ms on the CPU",
        frames as f32 * 2048.0 / 24_000.0,
        started.elapsed().as_secs_f32() * 1000.0
    );
    let error = sample_error("whole", &whole, &[frames * 2048], &fixture.whole);
    assert!(error < 5e-4, "whole decode error {error}");

    let mut stream = Maya1AudioStream::new(MAYA1_WINDOW_PLAN, fixture.seed);
    let mut windowed = Vec::new();
    let mut spans = Vec::new();
    for available in (1..=frames).map(|n| (n, false)).chain([(frames, true)]) {
        while let Some(window) = stream.next_window(available.0, available.1) {
            spans.push([window.first, window.start, window.end, window.decoded_end]);
            let started = std::time::Instant::now();
            windowed.extend(stream.decode(&snac, &codes, window).unwrap());
            eprintln!(
                "window {:?}: {} frames decoded in {:.0} ms",
                (window.start, window.end),
                window.decoded_frames(),
                started.elapsed().as_secs_f32() * 1000.0
            );
            if window.last {
                break;
            }
        }
    }
    assert!(stream.done());
    assert_eq!(spans, fixture.windows);
    let error = sample_error(
        "windowed",
        &windowed,
        &[(frames - 1) * 2048],
        &fixture.windowed,
    );
    assert!(error < 5e-4, "windowed decode error {error}");
    assert!(fixture.windowed_vs_whole_error < 1e-4);
}

#[derive(serde::Deserialize)]
struct HeadRecord {
    top: Vec<(usize, f32)>,
    sample: Sample,
}

#[derive(serde::Deserialize)]
struct LmFixture {
    tensors: BTreeMap<String, (String, Vec<usize>)>,
    prompt: Vec<u32>,
    sampled: Vec<u32>,
    steps: Vec<usize>,
    truncated_layers: usize,
    truncated_f32: BTreeMap<String, HeadRecord>,
    full_bf16: BTreeMap<String, HeadRecord>,
}

/// Teacher-forced head logits at the fixture's steps: step 0 is the
/// prefill, step `s` follows `s` sampled tokens.
fn forced_head_logits(runtime: &Maya1Runtime, fixture: &LmFixture) -> BTreeMap<usize, Vec<f32>> {
    let mut cache = runtime
        .scratch_cache(fixture.prompt.len() + fixture.sampled.len())
        .unwrap();
    let last = *fixture.steps.iter().max().unwrap();
    let mut logits = runtime
        .trunk()
        .forward(&mut [Maya1ForwardRow {
            tokens: &fixture.prompt,
            cache: &mut cache,
        }])
        .unwrap();
    let mut out = BTreeMap::new();
    for step in 0..=last {
        if fixture.steps.contains(&step) {
            out.insert(step, logits.to_vec2::<f32>().unwrap().remove(0));
        }
        if step < last {
            logits = runtime
                .trunk()
                .forward(&mut [Maya1ForwardRow {
                    tokens: &fixture.sampled[step..=step],
                    cache: &mut cache,
                }])
                .unwrap();
        }
    }
    out
}

fn top8(row: &[f32]) -> Vec<(usize, f32)> {
    let mut indexed = row.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(8);
    indexed
}

#[test]
#[ignore = "needs the Maya1 checkpoint (IZWI_MAYA1_MODEL_DIR)"]
fn published_safetensors_match_the_census() {
    let fixture: LmFixture = fixture("lm.json");
    let expected = super::published_tensors(&Maya1LmConfig::pinned());
    assert_eq!(fixture.tensors.len(), expected.len());
    for (name, shape) in expected {
        let (dtype, published) = fixture
            .tensors
            .get(&name)
            .unwrap_or_else(|| panic!("{name} missing from the census"));
        assert_eq!(dtype, "BF16", "{name}");
        assert_eq!(published, &shape, "{name}");
    }
    let dir = model_dir();
    let config = Maya1LmConfig::load(&dir).unwrap();
    assert_eq!(config, Maya1LmConfig::pinned());
}

/// The first four layers in F32 on the CPU against transformers' own
/// four-layer truncation, under teacher forcing.
#[test]
#[ignore = "needs the Maya1 checkpoint (IZWI_MAYA1_MODEL_DIR)"]
fn truncated_decoder_matches_upstream_f32() {
    let fixture: LmFixture = fixture("lm.json");
    let runtime = Maya1Runtime::load_truncated(
        &model_dir(),
        &Device::Cpu,
        DType::F32,
        fixture.truncated_layers,
    )
    .unwrap();
    let logits = forced_head_logits(&runtime, &fixture);
    for (step, record) in &fixture.truncated_f32 {
        let row = &logits[&step.parse::<usize>().unwrap()];
        let ours = top8(row);
        for (&(row_ours, value), &(row_ref, expected)) in ours.iter().zip(&record.top) {
            assert_eq!(row_ours, row_ref, "step {step} top-8 order");
            assert!(
                (value - expected).abs() < 2e-3,
                "step {step}: {value} vs {expected}"
            );
        }
        let error = sample_error(&format!("step {step}"), row, &[HEAD_ROWS], &record.sample);
        assert!(error < 1e-3, "step {step} head error {error}");
    }
}

/// The whole decoder in F16 on Metal (the 16 GB machines this runs on
/// cannot hold it in F32) against transformers in BF16: the same likeliest
/// code at every forced step.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Maya1 checkpoint (IZWI_MAYA1_MODEL_DIR)"]
fn full_decoder_tracks_upstream_bf16_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease =
        crate::backends::metal_test_device_or_skip("full_decoder_tracks_upstream_bf16_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture: LmFixture = fixture("lm.json");
    let runtime = Maya1Runtime::load(&model_dir(), &profile.device, DType::F16).unwrap();
    let check = runtime.numeric_self_check();
    assert!(check.passed, "{:?}", check.reason);
    let logits = forced_head_logits(&runtime, &fixture);
    for (step, record) in &fixture.full_bf16 {
        let row = &logits[&step.parse::<usize>().unwrap()];
        let ours = top8(row);
        assert_eq!(ours[0].0, record.top[0].0, "step {step} likeliest code");
        let shared = ours
            .iter()
            .filter(|(row, _)| record.top.iter().any(|(theirs, _)| theirs == row))
            .count();
        assert!(shared >= 6, "step {step}: {shared}/8 of the top-8 shared");
        let error = sample_error(&format!("step {step}"), row, &[HEAD_ROWS], &record.sample);
        assert!(error < 0.1, "step {step} head error {error}");
    }
}
