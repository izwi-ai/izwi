//! Parity with upstream: transformers 4.57.1 and `snac` 1.2.1 through
//! `scripts/fixtures/generate_maya1_reference.py` (fixtures in
//! `tests/fixtures/maya1`).
//!
//! The sampler test runs everywhere. The others need the checkpoint
//! (`IZWI_MAYA1_MODEL_DIR`, with `snac_24khz.pth` beside it), and the GGUF
//! ones the GGUF release too (`IZWI_MAYA1_GGUF_DIR`), and are ignored by
//! default:
//!
//! ```text
//! IZWI_MAYA1_MODEL_DIR=… IZWI_MAYA1_GGUF_DIR=… cargo test -p izwi-core \
//!     --features metal --lib maya1::reference_tests -- --ignored --test-threads=1
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

/// The GGUF release's directory for the ignored GGUF tests.
fn gguf_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("IZWI_MAYA1_GGUF_DIR")
            .expect("set IZWI_MAYA1_GGUF_DIR to a Maya1-3B-GGUF download"),
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

/// Every projection the GGUF loader builds is the shards' within Q8_0's
/// error, its Q and K rows back in half-split order, and the head's rows
/// are the table's from the stop token on.
#[test]
#[ignore = "needs both Maya1 releases (IZWI_MAYA1_MODEL_DIR, IZWI_MAYA1_GGUF_DIR)"]
fn gguf_projections_match_the_shards() {
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::Tensor;

    use super::config::{HEAD_FIRST_TOKEN, WEIGHT_SHARDS};
    use crate::backends::BackendKind;
    use crate::models::shared::weights::gguf::GgufLoader;
    use crate::models::shared::weights::gguf_projection::{
        joined_qtensor, row_slice_qtensor, QuantizedPart,
    };

    let cfg = Maya1LmConfig::pinned();
    let paths = WEIGHT_SHARDS
        .iter()
        .map(|shard| model_dir().join(shard))
        .collect::<Vec<_>>();
    // The shards are not written while the test maps them.
    let shards = unsafe { MmapedSafetensors::multi(&paths) }.unwrap();
    let gguf = GgufLoader::from_path_with_backend(
        &gguf_dir().join(super::MAYA1_GGUF_FILE),
        BackendKind::Cpu,
    )
    .unwrap();
    let cpu = Device::Cpu;
    let dense = |names: &[String]| -> Tensor {
        let parts = names
            .iter()
            .map(|name| {
                shards
                    .load(name, &cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        Tensor::cat(&parts, 0).unwrap()
    };
    // Relative RMS error; Q8_0 keeps it near 0.5%.
    let relative = |ours: &Tensor, theirs: &Tensor| -> f32 {
        let diff = (ours - theirs).unwrap().sqr().unwrap().mean_all().unwrap();
        let scale = theirs.sqr().unwrap().mean_all().unwrap();
        (diff.to_scalar::<f32>().unwrap() / scale.to_scalar::<f32>().unwrap()).sqrt()
    };
    for layer in [0, cfg.num_hidden_layers / 2, cfg.num_hidden_layers - 1] {
        let hf = |name: &str| format!("model.layers.{layer}.{name}.weight");
        let blk = |name: &str| format!("blk.{layer}.{name}.weight");
        let (q, k, v) = (blk("attn_q"), blk("attn_k"), blk("attn_v"));
        let ours = joined_qtensor(
            &gguf,
            &[
                QuantizedPart {
                    name: &q,
                    rope_heads: Some(cfg.num_attention_heads),
                },
                QuantizedPart {
                    name: &k,
                    rope_heads: Some(cfg.num_key_value_heads),
                },
                QuantizedPart {
                    name: &v,
                    rope_heads: None,
                },
            ],
            &cpu,
        )
        .unwrap()
        .dequantize(&cpu)
        .unwrap();
        let theirs = dense(&[
            hf("self_attn.q_proj"),
            hf("self_attn.k_proj"),
            hf("self_attn.v_proj"),
        ]);
        let error = relative(&ours, &theirs);
        assert!(error < 0.02, "layer {layer} qkv: {error}");
        let (gate, up) = (blk("ffn_gate"), blk("ffn_up"));
        let ours = joined_qtensor(
            &gguf,
            &[
                QuantizedPart {
                    name: &gate,
                    rope_heads: None,
                },
                QuantizedPart {
                    name: &up,
                    rope_heads: None,
                },
            ],
            &cpu,
        )
        .unwrap()
        .dequantize(&cpu)
        .unwrap();
        let theirs = dense(&[hf("mlp.gate_proj"), hf("mlp.up_proj")]);
        let error = relative(&ours, &theirs);
        assert!(error < 0.02, "layer {layer} gate/up: {error}");
    }
    let head = HEAD_FIRST_TOKEN as usize;
    let ours = row_slice_qtensor(&gguf, "token_embd.weight", head..head + HEAD_ROWS, &cpu)
        .unwrap()
        .dequantize(&cpu)
        .unwrap();
    let theirs = shards
        .load("model.embed_tokens.weight", &cpu)
        .unwrap()
        .narrow(0, head, HEAD_ROWS)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let error = relative(&ours, &theirs);
    assert!(error < 0.02, "head rows: {error}");
}

/// The GGUF release's whole decoder (Q8_0 weights, F32 activations) on the
/// best local device against transformers in BF16: quantization may swap
/// near-tied codes, so the likeliest code is upstream's top three and most
/// of the top eight are shared.
#[test]
#[ignore = "needs the GGUF release (IZWI_MAYA1_GGUF_DIR)"]
fn gguf_decoder_tracks_upstream_bf16() {
    use crate::backends::{BackendPreference, DeviceSelector};
    #[cfg(feature = "metal")]
    let _lease = crate::backends::metal_test_device_or_skip("gguf_decoder_tracks_upstream_bf16");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Auto).unwrap();
    let fixture: LmFixture = fixture("lm.json");
    let runtime = Maya1Runtime::load_gguf(&gguf_dir(), &profile.device).unwrap();
    assert_eq!(runtime.lm_dtype(), DType::F32);
    let logits = forced_head_logits(&runtime, &fixture);
    for (step, record) in &fixture.full_bf16 {
        let row = &logits[&step.parse::<usize>().unwrap()];
        let ours = top8(row);
        assert!(
            record.top[..3]
                .iter()
                .any(|(theirs, _)| *theirs == ours[0].0),
            "step {step}: likeliest code {} is not upstream's top three",
            ours[0].0
        );
        let shared = ours
            .iter()
            .filter(|(row, _)| record.top.iter().any(|(theirs, _)| theirs == row))
            .count();
        assert!(shared >= 5, "step {step}: {shared}/8 of the top-8 shared");
        let error = sample_error(&format!("step {step}"), row, &[HEAD_ROWS], &record.sample);
        eprintln!(
            "step {step}: top-1 {} shared {shared}/8 error {error:.4}",
            ours[0].0
        );
        assert!(error < 0.25, "step {step} head error {error}");
    }
}
