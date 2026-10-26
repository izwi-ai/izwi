//! Real-checkpoint parity for Chatterbox against upstream's PyTorch code
//! (F32 CPU) in `tests/fixtures/chatterbox/reference.json`
//! (`scripts/fixtures/generate_chatterbox_reference.py`).
//!
//! The non-ignored tests check the fixture against izwi's own constants (the
//! published tensor census against the geometry, the noise streams against
//! the script's numpy mirror) and need no weights.
//!
//! The ignored legs need the English checkpoint from
//! `IZWI_CHATTERBOX_MODEL_DIR` (else the default models directory). Per
//! backend they check, at the fixture's sample points:
//! * text normalisation and token ids;
//! * the conditioning, both prefill rows and their top-8 logits, then
//!   teacher-forced decode on upstream's sampled tokens through izwi's
//!   paged KV (two arenas, one per guidance branch);
//! * the sampler's logit processing on the first step;
//! * S3Gen on upstream's tokens with the shared flow noise: `mu`, the
//!   sample after Euler steps 1 and 5, and the mel;
//! * HiFT with the shared phases and noise: F0, source and waveform.
//!
//! They write the vocoded fixture tokens and a free-running izwi utterance
//! to `target/chatterbox-e2e/` for listening.
//!
//! Run: `cargo test -p izwi-core --release --features metal --lib
//! chatterbox::reference_tests -- --ignored --nocapture`
use std::collections::BTreeMap;
use std::path::PathBuf;

use candle_core::{DType, Tensor};

use super::config::{HiftConfig, S3GenConfig, T3Config};
use super::noise::{NoiseStream, SplitMix64};
use super::s3gen::S3GenFlow;
use super::t3::T3ForwardRow;
use super::text::normalize_punctuation;
use super::*;
use crate::backends::DeviceProfile;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

#[derive(serde::Deserialize)]
struct Fixture {
    seed: u64,
    noise_check: BTreeMap<String, Vec<f32>>,
    text_cases: Vec<TextCase>,
    t3: T3Fixture,
    s3gen: S3GenFixture,
    hift: HiftFixture,
}

#[derive(serde::Deserialize)]
struct TextCase {
    text: String,
    normalized: String,
    ids: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct Sample {
    shape: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f32>,
    rms: f32,
}

#[derive(serde::Deserialize)]
struct Step {
    next: u32,
    cond_top: Vec<(u32, f32)>,
    uncond_top: Vec<(u32, f32)>,
}

#[derive(serde::Deserialize)]
struct Processed {
    kept: Vec<u32>,
    top: Vec<(u32, f32)>,
}

#[derive(serde::Deserialize)]
struct T3Fixture {
    text: String,
    text_ids: Vec<u32>,
    cfg_weight: f32,
    exaggeration: f32,
    conditioning: Sample,
    prefill_cond: Sample,
    prefill_uncond: Sample,
    prefill_cond_top: Vec<(u32, f32)>,
    prefill_uncond_top: Vec<(u32, f32)>,
    sampled: Vec<u32>,
    steps: Vec<Step>,
    processed_first_step: Processed,
}

#[derive(serde::Deserialize)]
struct S3GenFixture {
    speech_tokens: Vec<u32>,
    frames: usize,
    mu: Sample,
    x_step1: Sample,
    x_step5: Sample,
    mel: Sample,
}

#[derive(serde::Deserialize)]
struct HiftFixture {
    f0: Sample,
    source: Sample,
    wav: Sample,
    wav_faded: Sample,
}

#[derive(serde::Deserialize)]
struct Census {
    files: BTreeMap<String, CensusFile>,
}

#[derive(serde::Deserialize)]
struct CensusFile {
    count: usize,
    tensors: BTreeMap<String, (String, Vec<usize>)>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chatterbox")
}

fn fixture() -> Fixture {
    serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("reference.json")).unwrap())
        .unwrap()
}

fn census() -> Census {
    serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("tensor_census.json")).unwrap(),
    )
    .unwrap()
}

fn model_dir() -> PathBuf {
    std::env::var_os("IZWI_CHATTERBOX_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi/models/Chatterbox-500M")
        })
}

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/chatterbox-e2e/out");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Max |izwi − reference| at the sample points over the reference RMS.
fn sample_error(name: &str, actual: &[f32], shape: &[usize], sample: &Sample) -> f32 {
    assert_eq!(shape, sample.shape.as_slice(), "{name} shape");
    let error = sample
        .indices
        .iter()
        .zip(&sample.values)
        .map(|(&index, &expected)| {
            let diff = (actual[index] - expected).abs();
            if diff.is_finite() {
                diff
            } else {
                f32::INFINITY
            }
        })
        .fold(0f32, f32::max);
    error / sample.rms.max(f32::EPSILON)
}

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Max |izwi − reference| over the reference's top logits, and whether
/// izwi's argmax agrees unless the reference's top two are within `tie`.
fn top_error(logits: &[f32], top: &[(u32, f32)], tie: f32) -> (f32, bool) {
    let error = top
        .iter()
        .map(|&(id, expected)| (logits[id as usize] - expected).abs())
        .fold(0f32, f32::max);
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index as u32)
        .unwrap();
    let near_tie = top.len() > 1 && (top[0].1 - top[1].1) < tie;
    (error, near_tie || argmax == top[0].0)
}

fn write_wav(name: &str, samples: &[f32]) -> PathBuf {
    let path = output_dir().join(name);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: S3GEN_SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    path
}

#[test]
fn noise_streams_match_the_fixture_mirror() {
    let fixture = fixture();
    let seed = fixture.seed;
    let flow = SplitMix64::for_stream(seed, NoiseStream::FlowNoise).standard_normals(4);
    let phase = SplitMix64::for_stream(seed, NoiseStream::HarmonicPhase).uniforms(4);
    let source = SplitMix64::for_stream(seed, NoiseStream::SourceNoise).standard_normals(4);
    assert_eq!(flow, fixture.noise_check["flow_noise"]);
    assert_eq!(phase, fixture.noise_check["harmonic_phase"]);
    assert_eq!(source, fixture.noise_check["source_noise"]);
}

#[test]
fn text_normalization_matches_upstream_cases() {
    for case in fixture().text_cases {
        assert_eq!(
            normalize_punctuation(&case.text),
            case.normalized,
            "{:?}",
            case.text
        );
    }
}

/// The published tensors agree with the geometry izwi hard-codes.
#[test]
fn published_census_matches_the_geometry() {
    let census = census();
    let t3 = &census.files["t3_cfg.safetensors"];
    let cfg = T3Config::english();
    assert_eq!(t3.count, 292);
    let shape = |file: &CensusFile, name: &str| file.tensors[name].1.clone();
    assert_eq!(
        shape(t3, "text_emb.weight"),
        vec![cfg.text_vocab_size, cfg.hidden_size]
    );
    assert_eq!(
        shape(t3, "speech_emb.weight"),
        vec![cfg.speech_vocab_size, cfg.hidden_size]
    );
    assert_eq!(
        shape(t3, "text_pos_emb.emb.weight"),
        vec![cfg.text_positions, cfg.hidden_size]
    );
    assert_eq!(
        shape(t3, "speech_pos_emb.emb.weight"),
        vec![cfg.speech_positions, cfg.hidden_size]
    );
    assert_eq!(
        shape(t3, "cond_enc.perceiver.pre_attention_query"),
        vec![1, cfg.perceiver_queries, cfg.hidden_size]
    );
    assert_eq!(
        shape(t3, "tfmr.layers.0.mlp.gate_proj.weight"),
        vec![cfg.intermediate_size, cfg.hidden_size]
    );
    let layers = t3
        .tensors
        .keys()
        .filter(|name| name.ends_with(".input_layernorm.weight"))
        .count();
    assert_eq!(layers, cfg.num_hidden_layers);
    assert!(t3.tensors.values().all(|(dtype, _)| dtype == "F32"));

    let s3gen = &census.files["s3gen.safetensors"];
    let s3 = S3GenConfig::current();
    assert_eq!(
        shape(s3gen, "flow.input_embedding.weight"),
        vec![s3.vocab_size, s3.encoder_dim]
    );
    assert_eq!(
        shape(s3gen, "flow.decoder.estimator.time_mlp.linear_1.weight"),
        vec![s3.time_embed_dim, s3.estimator_in_channels]
    );
    let mids = (0..)
        .take_while(|i| {
            s3gen.tensors.contains_key(&format!(
                "flow.decoder.estimator.mid_blocks.{i}.0.res_conv.weight"
            ))
        })
        .count();
    assert_eq!(mids, s3.estimator_mid_blocks);
    let hift = HiftConfig::current();
    assert_eq!(
        shape(s3gen, "mel2wav.m_source.l_linear.weight"),
        vec![1, hift.harmonics + 1]
    );
    assert_eq!(
        shape(s3gen, "mel2wav.conv_post.parametrizations.weight.original1"),
        vec![hift.istft_n_fft + 2, hift.base_channels >> 3, 7]
    );
}

struct Tolerance {
    /// Max error at the sample points over the reference RMS.
    embeddings: f32,
    /// Max |izwi − reference| over a step's reference top-8 logits.
    logit: f32,
    /// Top-2 margin under which the argmax may differ.
    near_tie: f32,
    /// S3Gen and HiFT stages, over the reference RMS.
    flow: f32,
    mel: f32,
    wav: f32,
}

fn run_leg(device: DeviceProfile, tolerance: Tolerance, label: &str) {
    let fixture = fixture();
    let dir = model_dir();
    if !dir.join(config::T3_WEIGHTS_FILE).is_file() {
        panic!("Chatterbox checkpoint not found at {}", dir.display());
    }
    let runtime = ChatterboxRuntime::load(&dir, &device.device, DType::F32).unwrap();

    for case in &fixture.text_cases {
        let ids = runtime
            .tokenizer()
            .encode_normalized(&normalize_punctuation(&case.text))
            .unwrap();
        assert_eq!(ids, case.ids, "token ids of {:?}", case.text);
    }

    // --- T3
    let t3 = runtime.t3();
    let cfg = t3.config().clone();
    let fx = &fixture.t3;
    let text_ids = runtime.tokenizer().text_sequence(&fx.text).unwrap();
    assert_eq!(text_ids, fx.text_ids);
    let conditioning = t3
        .conditioning(&runtime.default_voice().t3, fx.exaggeration)
        .unwrap();
    let mut errors = BTreeMap::new();
    errors.insert(
        "conditioning",
        sample_error(
            "conditioning",
            &host(&conditioning),
            conditioning.dims(),
            &fx.conditioning,
        ),
    );
    let (cond_row, uncond_row) = t3.prefill_embeds(&conditioning, &text_ids, true).unwrap();
    let uncond_row = uncond_row.unwrap();
    errors.insert(
        "prefill_cond",
        sample_error(
            "prefill_cond",
            &host(&cond_row),
            cond_row.dims(),
            &fx.prefill_cond,
        ),
    );
    errors.insert(
        "prefill_uncond",
        sample_error(
            "prefill_uncond",
            &host(&uncond_row),
            uncond_row.dims(),
            &fx.prefill_uncond,
        ),
    );

    let capacity = runtime.sequence_tokens(text_ids.len(), fx.steps.len() + 2);
    let scratch = || {
        PhysicalPagedKvCache::scratch_for_probe(
            &device.device,
            DType::F32,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            capacity,
        )
        .unwrap()
    };
    let (mut cond_cache, mut uncond_cache) = (scratch(), scratch());
    let mut logits = t3
        .forward(&mut [
            T3ForwardRow {
                embeds: cond_row,
                cache: &mut cond_cache,
            },
            T3ForwardRow {
                embeds: uncond_row,
                cache: &mut uncond_cache,
            },
        ])
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let prefill_logits = logits.clone();
    let (cond_err, cond_ok) = top_error(&logits[0], &fx.prefill_cond_top, tolerance.near_tie);
    let (uncond_err, uncond_ok) = top_error(&logits[1], &fx.prefill_uncond_top, tolerance.near_tie);
    assert!(
        cond_ok && uncond_ok,
        "prefill argmax differs from the reference"
    );
    let mut logit_error = cond_err.max(uncond_err);
    for (index, step) in fx.steps.iter().enumerate() {
        let (cond_err, cond_ok) = top_error(&logits[0], &step.cond_top, tolerance.near_tie);
        let (uncond_err, uncond_ok) = top_error(&logits[1], &step.uncond_top, tolerance.near_tie);
        assert!(
            cond_ok && uncond_ok,
            "step {index}: argmax differs from the reference"
        );
        logit_error = logit_error.max(cond_err).max(uncond_err);
        let embed = t3.speech_step_embed(step.next, index + 1).unwrap();
        logits = t3
            .forward(&mut [
                T3ForwardRow {
                    embeds: embed.clone(),
                    cache: &mut cond_cache,
                },
                T3ForwardRow {
                    embeds: embed,
                    cache: &mut uncond_cache,
                },
            ])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
    }

    // Sampler processing on the first step.
    let params = ChatterboxGenerationParams::defaults(1).sampling;
    let processed = sampling::process_logits(
        &prefill_logits[0],
        Some(&prefill_logits[1]),
        &[cfg.start_speech_token],
        &T3SamplingParams {
            cfg_weight: fx.cfg_weight,
            ..params
        },
    )
    .unwrap();
    let kept = processed
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, _)| index as u32)
        .collect::<Vec<_>>();
    let (processed_err, _) = top_error(&processed, &fx.processed_first_step.top, f32::INFINITY);
    assert_eq!(
        kept, fx.processed_first_step.kept,
        "min-p kept a different set"
    );

    // --- S3Gen on upstream's tokens, with the shared noise.
    let s3 = &fixture.s3gen;
    let voice = runtime.default_voice();
    assert_eq!(
        S3GenFlow::total_frames(voice.s3gen.prompt_tokens(), s3.speech_tokens.len()),
        s3.frames
    );
    let flow = runtime.flow();
    let noise = SplitMix64::for_stream(fixture.seed, NoiseStream::FlowNoise)
        .standard_normals(80 * s3.frames);
    let mut state = flow
        .begin(&voice.s3gen, &s3.speech_tokens, &noise, true)
        .unwrap();
    let mu = flow_state_tensor(&state, "mu");
    errors.insert("mu", sample_error("mu", &host(&mu), mu.dims(), &s3.mu));
    flow.step(&mut state, 1).unwrap();
    let x1 = flow_state_tensor(&state, "x");
    errors.insert(
        "x_step1",
        sample_error("x_step1", &host(&x1), x1.dims(), &s3.x_step1),
    );
    flow.step(&mut state, 4).unwrap();
    let x5 = flow_state_tensor(&state, "x");
    errors.insert(
        "x_step5",
        sample_error("x_step5", &host(&x5), x5.dims(), &s3.x_step5),
    );
    flow.step(&mut state, 5).unwrap();
    let mel = flow.finish(&state).unwrap();
    errors.insert("mel", sample_error("mel", &host(&mel), mel.dims(), &s3.mel));

    // --- HiFT on izwi's mel, with the shared phases and noise.
    let hf = &fixture.hift;
    let vocoder = runtime.vocoder();
    let f0 = vocoder.predict_f0(&mel).unwrap();
    errors.insert("f0", sample_error("f0", &f0, &[f0.len()], &hf.f0));
    let source = vocoder.harmonic_source(&f0, fixture.seed);
    errors.insert(
        "source",
        sample_error("source", &source, &[source.len()], &hf.source),
    );
    let mut wav = vocoder.decode(&mel, &source).unwrap();
    errors.insert("wav", sample_error("wav", &wav, &[wav.len()], &hf.wav));
    hift::apply_trim_fade(&mut wav);
    errors.insert(
        "wav_faded",
        sample_error("wav_faded", &wav, &[wav.len()], &hf.wav_faded),
    );
    let fixture_wav = write_wav(&format!("fixture-tokens-{label}.wav"), &wav);

    // A free-running izwi utterance with izwi's own sampler.
    let gen_params = ChatterboxGenerationParams::defaults(fixture.seed);
    let budget = gen_params.max_speech_tokens;
    let capacity = runtime.sequence_tokens(text_ids.len(), budget);
    let scratch = || {
        PhysicalPagedKvCache::scratch_for_probe(
            &device.device,
            DType::F32,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            capacity,
        )
        .unwrap()
    };
    let (mut cond_cache, mut uncond_cache) = (scratch(), scratch());
    let output = runtime
        .generate_physical(
            &fx.text,
            voice,
            &gen_params,
            &mut cond_cache,
            Some(&mut uncond_cache),
        )
        .unwrap();
    let izwi_wav = write_wav(&format!("izwi-free-run-{label}.wav"), &output.samples);
    let seconds = output.samples.len() as f32 / S3GEN_SAMPLE_RATE as f32;
    let rms =
        (output.samples.iter().map(|s| s * s).sum::<f32>() / output.samples.len() as f32).sqrt();

    eprintln!(
        "[{label}] errors (of RMS): {errors:?}; top-8 logit error {logit_error:.2e}, processed {processed_err:.2e}"
    );
    eprintln!(
        "[{label}] free run: {} tokens ({:?}), {seconds:.2} s, RMS {rms:.4}; T3 {:.0} ms, flow {:.0} ms, vocoder {:.0} ms",
        output.speech_tokens.len(),
        output.stop_reason,
        output.t3_ms,
        output.flow_ms,
        output.vocoder_ms
    );
    eprintln!(
        "[{label}] wrote {} and {}",
        fixture_wav.display(),
        izwi_wav.display()
    );

    assert!(
        logit_error <= tolerance.logit,
        "top-8 logit error {logit_error}"
    );
    assert!(
        processed_err <= tolerance.logit * 2.0,
        "processed logit error {processed_err}"
    );
    for name in ["conditioning", "prefill_cond", "prefill_uncond"] {
        assert!(
            errors[name] <= tolerance.embeddings,
            "{name}: {}",
            errors[name]
        );
    }
    for name in ["mu", "x_step1", "x_step5"] {
        assert!(errors[name] <= tolerance.flow, "{name}: {}", errors[name]);
    }
    assert!(errors["mel"] <= tolerance.mel, "mel: {}", errors["mel"]);
    for name in ["f0", "source", "wav", "wav_faded"] {
        assert!(errors[name] <= tolerance.wav, "{name}: {}", errors[name]);
    }
    assert_eq!(output.stop_reason, ChatterboxStopReason::EndOfSpeech);
    assert!(seconds > 2.0 && seconds < 15.0, "{seconds} s");
    assert!(rms > 0.01, "free run is near silent: RMS {rms}");
}

/// A flow state's tensors, for the parity probes.
fn flow_state_tensor(state: &s3gen::S3GenFlowState, which: &str) -> Tensor {
    state.probe(which).unwrap()
}

#[test]
#[ignore = "needs the Chatterbox checkpoint (IZWI_CHATTERBOX_MODEL_DIR)"]
fn chatterbox_matches_the_f32_reference_on_cpu() {
    run_leg(
        DeviceProfile::cpu(),
        Tolerance {
            embeddings: 1e-4,
            logit: 0.01,
            near_tie: 0.05,
            flow: 1e-3,
            mel: 1e-3,
            wav: 1e-2,
        },
        "cpu",
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Chatterbox checkpoint (IZWI_CHATTERBOX_MODEL_DIR)"]
fn chatterbox_matches_the_f32_reference_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    use candle_core::Device;
    let _lease =
        crate::backends::metal_test_device_or_skip("chatterbox_matches_the_f32_reference_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    run_leg(
        profile,
        Tolerance {
            embeddings: 1e-3,
            logit: 0.05,
            near_tie: 0.25,
            flow: 5e-3,
            mel: 5e-3,
            wav: 5e-2,
        },
        "metal",
    );
}

/// How one Euler step's estimator call scales with rows and frames on
/// Metal: whether batching acoustic steps across requests would pay.
/// `batch` counts both guidance rows of each request.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Chatterbox checkpoint (IZWI_CHATTERBOX_MODEL_DIR)"]
fn chatterbox_estimator_cost_by_rows_and_frames_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    use candle_core::Device;
    let _lease = crate::backends::metal_test_device_or_skip(
        "chatterbox_estimator_cost_by_rows_and_frames_on_metal",
    );
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    let runtime = ChatterboxRuntime::load(&model_dir(), &profile.device, DType::F32).unwrap();
    for (batch, frames) in [(2, 250), (2, 500), (2, 1000), (4, 500), (8, 500), (16, 500)] {
        runtime.flow().probe_estimator(batch, frames).unwrap();
        let runs = 5;
        let started = std::time::Instant::now();
        for _ in 0..runs {
            runtime.flow().probe_estimator(batch, frames).unwrap();
        }
        let ms = started.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        eprintln!(
            "estimator rows {batch:2} frames {frames:4}: {ms:7.1} ms per step, {:6.1} ms per request",
            ms * 2.0 / batch as f64
        );
    }
    for frames in [500, 1000] {
        for (part, ms) in runtime.flow().probe_estimator_parts(2, frames, 10).unwrap() {
            eprintln!("rows 2 frames {frames:4} {part:24} {ms:7.2} ms");
        }
    }
}
