//! Tests on the real checkpoint (ignored: they need the 3 GB download).
//!
//! `IZWI_MING_TTS_MODEL_DIR` names a Ming-3.0-tts-0.5B folder holding
//! `model.safetensors`, the tokenizer, `config.json` and
//! `campplus_cn_common.bin`. `IZWI_MING_TTS_OUT_DIR`, when set, receives
//! the generated audio as WAV files for listening and for a
//! speech-recognition round trip.
//!
//! Parity with the PyTorch reference is checked against the fixtures
//! `scripts/fixtures/generate_ming_tts_reference.py` writes; those tests
//! skip while the fixtures are absent.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device};

use super::config::{MingTtsConfig, LATENT_DIM, LATENT_SAMPLES, PINNED_TOKENS, TOKENIZER_FILE};
use super::prompt::MingTokenizer;
use super::speaker::{CampPlus, SPEAKER_EMBEDDING_DIM};
use super::{
    MingControl, MingEmotion, MingGenerationParams, MingPrecision, MingReferenceVoice, MingRequest,
    MingRuntime, MingStopReason, MingVoice, DEFAULT_GUIDANCE, SAMPLE_RATE, SPEAKER_WEIGHTS_FILE,
};

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("IZWI_MING_TTS_MODEL_DIR")
            .expect("set IZWI_MING_TTS_MODEL_DIR to a Ming-3.0-tts-0.5B download"),
    )
}

/// The test device and its dtype. A Metal device comes with the shared
/// device's test lease, held while this lives.
struct TestDevice {
    device: Device,
    dtype: DType,
    _lease: Option<Box<dyn std::any::Any>>,
}

fn device() -> TestDevice {
    let (metal, dtype) = match std::env::var("IZWI_MING_TTS_TEST_DEVICE").as_deref() {
        Ok("metal") => (true, DType::F16),
        Ok("metal-f32") => (true, DType::F32),
        _ => (false, DType::F32),
    };
    if !metal {
        return TestDevice {
            device: Device::Cpu,
            dtype,
            _lease: None,
        };
    }
    #[cfg(feature = "metal")]
    {
        let (device, lease) = crate::backends::metal_test_device_or_skip("ming_tts reference")
            .expect("IZWI_MING_TTS_TEST_DEVICE=metal on a machine without Metal");
        TestDevice {
            device,
            dtype,
            _lease: Some(Box::new(lease)),
        }
    }
    #[cfg(not(feature = "metal"))]
    panic!("IZWI_MING_TTS_TEST_DEVICE=metal needs a build with the metal feature")
}

/// The runtime at the first precision whose probe passes, and the device
/// it runs on (keep it alive with the runtime).
fn runtime() -> (MingRuntime, TestDevice) {
    let test_device = device();
    let (device, dtype) = (&test_device.device, test_device.dtype);
    let precision = MingPrecision::ladder(dtype)
        .into_iter()
        .find(|precision| {
            let runtime = MingRuntime::load(&model_dir(), device, *precision).unwrap();
            let report = runtime.numeric_self_check();
            eprintln!("{}", report.reason);
            report.passed
        })
        .expect("a precision that passes its probe");
    eprintln!("precision: {precision:?}");
    let runtime = MingRuntime::load(&model_dir(), device, precision).unwrap();
    (runtime, test_device)
}

fn write_wav(name: &str, samples: &[f32]) {
    let Ok(dir) = std::env::var("IZWI_MING_TTS_OUT_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    let data = samples
        .iter()
        .flat_map(|sample| ((sample.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes())
        .collect::<Vec<_>>();
    let mut wav = Vec::with_capacity(44 + data.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);
    std::fs::write(Path::new(&dir).join(format!("{name}.wav")), wav).unwrap();
}

fn params(seed: u64) -> MingGenerationParams {
    MingGenerationParams {
        guidance: DEFAULT_GUIDANCE,
        max_new_tokens: 300,
        seed,
    }
}

fn speak(runtime: &MingRuntime, name: &str, request: &MingRequest, seed: u64) -> Vec<f32> {
    let mut cache = runtime.scratch_cache(super::CONTEXT_POSITIONS).unwrap();
    let started = std::time::Instant::now();
    let output = runtime
        .generate_physical(request, &params(seed), &mut cache)
        .unwrap();
    let seconds = output.samples.len() as f32 / SAMPLE_RATE as f32;
    eprintln!(
        "{name}: {} prompt positions, {} latents, {:?}, {seconds:.2} s of audio in {:.2} s \
         (lm {:.0} ms, codec {:.0} ms)",
        output.prompt_positions,
        output.latents,
        output.stop_reason,
        started.elapsed().as_secs_f32(),
        output.lm_ms,
        output.codec_ms,
    );
    assert_eq!(output.stop_reason, MingStopReason::EndOfSpeech, "{name}");
    assert!(output.samples.iter().all(|sample| sample.is_finite()));
    let peak = output
        .samples
        .iter()
        .fold(0f32, |peak, s| peak.max(s.abs()));
    assert!(peak > 0.02 && peak < 1.5, "{name} peak {peak}");
    write_wav(name, &output.samples);
    output.samples
}

fn built_in(text: &str, voice: &str, language: Option<&str>) -> MingRequest {
    MingRequest {
        text: text.to_string(),
        language: language.map(str::to_string),
        voice: MingVoice::Captioned,
        control: MingControl::BuiltInVoice(voice.to_string()),
    }
}

#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn published_config_and_tokenizer_match_the_port() {
    let dir = model_dir();
    assert_eq!(MingTtsConfig::load(&dir).unwrap(), MingTtsConfig::pinned());
    let tokenizer = MingTokenizer::load(&dir.join(TOKENIZER_FILE)).unwrap();
    let prompt = tokenizer
        .prompt_ids("Hello there.", Some("English"), true, Some("{}"), None)
        .unwrap();
    // <|im_start|> system \n ... and the speaker slot in order.
    assert_eq!(prompt.ids[0], PINNED_TOKENS[1].1);
    let row = prompt.speaker_row.unwrap();
    assert_eq!(prompt.ids[row - 1], super::config::TOKEN_SPEAKER_START);
    assert_eq!(prompt.ids[row], super::config::TOKEN_SPEAKER_PAD);
    assert_eq!(prompt.ids[row + 1], super::config::TOKEN_SPEAKER_END);
    assert_eq!(
        *prompt.ids.last().unwrap(),
        super::config::TOKEN_AUDIO_START
    );
    assert!(prompt.text_tokens >= 3);
    let with_reference = tokenizer
        .prompt_ids("Hello there.", None, true, None, Some(("A reference.", 7)))
        .unwrap();
    let latent_row = with_reference.latent_row.unwrap();
    assert_eq!(with_reference.ids.len(), latent_row + 7);
    assert!(with_reference.ids[latent_row..]
        .iter()
        .all(|&id| id == super::config::TOKEN_AUDIO_PATCH));
}

#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn built_in_voices_speak_english_and_chinese() {
    let (runtime, _device) = runtime();
    speak(
        &runtime,
        "builtin_en_emma",
        &built_in(
            "The old radio crackled softly as the storm moved across the valley.",
            "US_Emma",
            Some("en"),
        ),
        11,
    );
    speak(
        &runtime,
        "builtin_en_george",
        &built_in(
            "Please keep your tickets ready, the train leaves in four minutes.",
            "UK_George",
            Some("en"),
        ),
        12,
    );
    speak(
        &runtime,
        "builtin_zh_xiaorou",
        &built_in(
            "傍晚的风从山口吹过来，远处的云慢慢染成了金色。",
            "灵小柔",
            Some("zh"),
        ),
        13,
    );
}

#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn a_seed_reproduces_and_another_seed_differs() {
    let (runtime, _device) = runtime();
    let request = built_in("Same words, same seed.", "US_Emma", None);
    let a = speak(&runtime, "seed_a", &request, 5);
    let b = speak(&runtime, "seed_a_again", &request, 5);
    assert_eq!(a, b);
    let c = speak(&runtime, "seed_b", &request, 6);
    assert_ne!(a, c);
}

#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn voice_design_and_controls_speak() {
    let (runtime, _device) = runtime();
    let request = |text: &str, control: MingControl| MingRequest {
        text: text.to_string(),
        language: None,
        voice: MingVoice::Captioned,
        control,
    };
    speak(
        &runtime,
        "design_warm_storyteller",
        &request(
            "Once upon a time, a small boat drifted quietly down the river.",
            MingControl::VoiceDesign(
                "A warm, intimate storytelling voice with a gentle pace and a hint of nostalgia."
                    .into(),
            ),
        ),
        21,
    );
    speak(
        &runtime,
        "freeform_excited",
        &request(
            "We did it, every single test passed on the first try!",
            MingControl::Freeform(
                "Speak with bright, excited energy, a little out of breath.".into(),
            ),
        ),
        22,
    );
    speak(
        &runtime,
        "tags_sigh_cough",
        &built_in(
            "[Sigh] It is still raining, [Cough] so we will have to move the party indoors.",
            "US_Emma",
            None,
        ),
        23,
    );
}

#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn a_reference_voice_clones_with_and_without_its_transcript() {
    let (runtime, _device) = runtime();
    // The reference is a built-in voice's own audio, so the test needs no
    // recording: clone what the model just said.
    let source_text = "This is the voice I would like you to copy, please.";
    let source = speak(
        &runtime,
        "clone_source",
        &built_in(source_text, "UK_George", None),
        31,
    );
    let with_text = runtime
        .encode_reference(&source, SAMPLE_RATE, Some(source_text))
        .unwrap();
    assert_eq!(with_text.embedding.len(), SPEAKER_EMBEDDING_DIM);
    assert_eq!(
        with_text.latent_count(),
        source.len().div_ceil(LATENT_SAMPLES)
    );
    assert_eq!(with_text.latents.len() % LATENT_DIM, 0);
    // The same clip encodes the same way every time.
    assert_eq!(
        runtime
            .encode_reference(&source, SAMPLE_RATE, Some(source_text))
            .unwrap(),
        with_text
    );
    let target = "And now the same voice says something entirely new.";
    let clone = |voice: MingReferenceVoice, control: MingControl| MingRequest {
        text: target.to_string(),
        language: None,
        voice: MingVoice::Reference(Arc::new(voice)),
        control,
    };
    speak(
        &runtime,
        "clone_with_transcript",
        &clone(with_text.clone(), MingControl::None),
        32,
    );
    let audio_only = runtime
        .encode_reference(&source, SAMPLE_RATE, None)
        .unwrap();
    assert!(audio_only.latents.is_empty());
    speak(
        &runtime,
        "clone_audio_only",
        &clone(audio_only.clone(), MingControl::None),
        33,
    );
    speak(
        &runtime,
        "clone_emotion_sad",
        &clone(audio_only, MingControl::Emotion(MingEmotion::Sad)),
        34,
    );
    // Reference encode → decode keeps the clip recognisable in level.
    let decoded = runtime.decoder().decode(&with_text.latents).unwrap();
    assert_eq!(decoded.len(), with_text.latent_count() * LATENT_SAMPLES);
    write_wav("clone_source_roundtrip", &decoded);
    let rms = |samples: &[f32]| {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    };
    let ratio = rms(&decoded) / rms(&source);
    assert!(
        (0.5..2.0).contains(&ratio),
        "round-trip level ratio {ratio}"
    );
}

#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn speaker_encoder_separates_voices() {
    let (runtime, _device) = runtime();
    let speaker = CampPlus::open(&model_dir().join(SPEAKER_WEIGHTS_FILE)).unwrap();
    let text = "A short sentence for comparing two different speakers.";
    let emma_a = speak(&runtime, "spk_emma_a", &built_in(text, "US_Emma", None), 41);
    let emma_b = speak(
        &runtime,
        "spk_emma_b",
        &built_in(
            "Another line, spoken by the very same voice.",
            "US_Emma",
            None,
        ),
        42,
    );
    let george = speak(
        &runtime,
        "spk_george",
        &built_in(text, "UK_George", None),
        43,
    );
    let embed = |samples: &[f32]| speaker.embed_clip(samples, SAMPLE_RATE).unwrap();
    let cosine = |a: &[f32], b: &[f32]| {
        let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b))
    };
    let (a, b, g) = (embed(&emma_a), embed(&emma_b), embed(&george));
    let same = cosine(&a, &b);
    let other = cosine(&a, &g);
    eprintln!("speaker cosine: same voice {same:.3}, other voice {other:.3}");
    assert!(same > other + 0.1, "same {same} vs other {other}");
}

#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn half_precision_passes_its_range_probe() {
    // Some step of the ladder passes; the language model keeps the device's
    // dtype whatever the flow path needs.
    let (runtime, device) = runtime();
    assert_eq!(runtime.lm_dtype(), device.dtype);
    let report = runtime.numeric_self_check();
    eprintln!("{}", report.diagnostics());
    assert!(report.passed, "{}", report.reason);
}

/// Where the language model's values outgrow F16: each layer's largest
/// magnitudes, run densely in F16 on the test device (a diagnostic; it
/// asserts nothing).
#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn trace_language_model_magnitudes() {
    let test_device = device();
    let (device, dtype) = (&test_device.device, test_device.dtype);
    let host = MingRuntime::load(&model_dir(), &Device::Cpu, MingPrecision::F32).unwrap();
    let prompt = host
        .prompt(&built_in(
            "The quick brown fox jumps over the lazy dog, twice!",
            "US_Emma",
            None,
        ))
        .unwrap();
    let cfg = MingTtsConfig::load(&model_dir()).unwrap();
    let vb = super::weights::mmap_checkpoint(&model_dir(), dtype, device).unwrap();
    let stack =
        super::stack::MingWindowStack::load(cfg.llm.clone(), vb.pp("model").pp("model"), 4_096)
            .unwrap();
    let x = candle_core::Tensor::from_vec(
        prompt.embeds.clone(),
        (prompt.positions, cfg.llm.hidden_size),
        device,
    )
    .unwrap();
    eprintln!("layer | q | attn out | gate | up | gated | mlp out | residual ({dtype:?})");
    for (layer, row) in stack.trace_layers(&x).unwrap().iter().enumerate() {
        eprintln!("{layer:2} | {row:?}");
    }
}

/// Milliseconds per call of `step` on the test device, waiting for the
/// device after each call (two warm-up calls first).
fn time_ms(runtime: &MingRuntime, calls: usize, mut step: impl FnMut()) -> f64 {
    for _ in 0..2 {
        step();
    }
    runtime.device().synchronize().unwrap();
    let started = std::time::Instant::now();
    for _ in 0..calls {
        step();
        runtime.device().synchronize().unwrap();
    }
    started.elapsed().as_secs_f64() * 1000.0 / calls as f64
}

/// What each stage costs per latent on the test device, and the real-time
/// factor end to end (a benchmark; it asserts nothing).
/// `IZWI_MING_TTS_BENCH_CALLS` sets the timed calls per stage (default 20).
#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn bench_components() {
    use super::stack::MingForwardRow;
    use super::stream::MingAudioStream;

    let calls = std::env::var("IZWI_MING_TTS_BENCH_CALLS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20usize);
    let (runtime, _device) = runtime();
    let request = built_in(
        "The quick brown fox jumps over the lazy dog, and then it runs into the woods to rest.",
        "US_Emma",
        None,
    );
    let prompt = runtime.prompt(&request).unwrap();
    let embeds = runtime.prompt_rows(&prompt.embeds).unwrap();
    let hidden_size = runtime.config().llm.hidden_size;
    let prefill_ms = time_ms(&runtime, calls.min(5), || {
        let mut cache = runtime.scratch_cache(super::CONTEXT_POSITIONS).unwrap();
        runtime
            .trunk()
            .forward(&mut [MingForwardRow {
                embeds: &embeds,
                cache: &mut cache,
            }])
            .unwrap();
    });
    eprintln!(
        "prefill ({} positions): {prefill_ms:.1} ms",
        prompt.positions
    );

    // A sentence end to end; its latents feed the VAE timing below.
    let mut cache = runtime.scratch_cache(super::CONTEXT_POSITIONS).unwrap();
    let started = std::time::Instant::now();
    let output = runtime
        .generate_physical(&request, &params(11), &mut cache)
        .unwrap();
    let wall = started.elapsed().as_secs_f64();
    let audio = output.samples.len() as f64 / f64::from(SAMPLE_RATE);
    write_wav("bench_end_to_end", &output.samples);
    eprintln!(
        "end to end: {} latents, {audio:.2} s of audio in {wall:.2} s, RTF {:.2} \
         (lm+flow {:.0} ms = {:.1} ms/latent, codec {:.0} ms)",
        output.latents,
        wall / audio,
        output.lm_ms,
        f64::from(output.lm_ms) / output.latents as f64,
        output.codec_ms,
    );

    let hidden: Vec<f32> = (0..hidden_size)
        .map(|index| ((index as f32) * 0.37).sin())
        .collect();
    let history = vec![0.1f32; LATENT_DIM];
    let noise = vec![0.5f32; LATENT_DIM];
    let flow_ms = time_ms(&runtime, calls, || {
        runtime
            .sample_latent(&hidden, &history, &noise, DEFAULT_GUIDANCE)
            .unwrap();
    });
    let aggregator_ms = time_ms(&runtime, calls, || {
        runtime.latent_embeddings(&history).unwrap();
    });
    let stop_ms = time_ms(&runtime, calls, || {
        runtime.stops_at(&hidden, 10);
    });
    eprintln!("flow head (10 steps, CFG batch 2): {flow_ms:.1} ms");
    for requests in [2usize, 4, 8] {
        let inputs = (0..requests)
            .map(|_| super::MingLatentInput {
                hidden: &hidden,
                history: &history,
                noise: &noise,
                guidance: DEFAULT_GUIDANCE,
            })
            .collect::<Vec<_>>();
        let ms = time_ms(&runtime, calls, || {
            runtime.sample_latents(&inputs).unwrap();
        });
        eprintln!(
            "flow head, {requests} requests together: {ms:.1} ms ({:.1} ms a request)",
            ms / requests as f64
        );
    }
    eprintln!("aggregator: {aggregator_ms:.1} ms; stop head: {stop_ms:.2} ms");

    for width in [1usize, 2, 4] {
        let mut caches = (0..width)
            .map(|_| {
                let mut cache = runtime.scratch_cache(super::CONTEXT_POSITIONS).unwrap();
                runtime
                    .trunk()
                    .forward(&mut [MingForwardRow {
                        embeds: &embeds,
                        cache: &mut cache,
                    }])
                    .unwrap();
                cache
            })
            .collect::<Vec<_>>();
        let next = runtime.latent_embeddings(&history).unwrap();
        // Each call appends a position; stay inside the context.
        let room = super::CONTEXT_POSITIONS - prompt.positions - 1;
        let width_calls = calls.min(room / 3);
        let step_ms = time_ms(&runtime, width_calls, || {
            let mut rows = caches
                .iter_mut()
                .map(|cache| MingForwardRow {
                    embeds: &next,
                    cache,
                })
                .collect::<Vec<_>>();
            runtime.trunk().forward_batch(&mut rows).unwrap();
        });
        eprintln!(
            "language-model step, {width} row(s): {step_ms:.1} ms ({:.1} ms/row)",
            step_ms / width as f64
        );
    }

    // The VAE's cost does not depend on the values: the sentence's latent
    // count of stand-ins.
    let latents = (0..output.latents * LATENT_DIM)
        .map(|index| ((index as f32) * 0.11).sin())
        .collect::<Vec<_>>();
    let latents = &latents;
    let window_ms = {
        let count = output.latents;
        let started = std::time::Instant::now();
        let mut stream = MingAudioStream::default();
        let mut windows = 0;
        while let Some(window) = stream.next_window(count, true) {
            stream.decode(runtime.decoder(), latents, window).unwrap();
            windows += 1;
        }
        runtime.device().synchronize().unwrap();
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "VAE: {count} latents in {windows} window(s), {ms:.1} ms ({:.1} ms/latent)",
            ms / count as f64
        );
        ms / count as f64
    };
    let per_latent = flow_ms + aggregator_ms + stop_ms + window_ms;
    eprintln!(
        "per latent excluding the language model: {per_latent:.1} ms; 80 ms of audio per latent"
    );
}

/// Requests sampled together get the latents and embeddings they get
/// alone, on the published checkpoint at the test device's precision, for
/// every count from two to eight (a shared call's rounding must not depend
/// on how many requests share it).
#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn requests_together_match_each_alone_on_the_checkpoint() {
    let (runtime, _device) = runtime();
    let hidden_size = runtime.config().llm.hidden_size;
    let mut rng = crate::models::shared::noise::SplitMix64::new(21);
    let mut normals = |count: usize, scale: f32| {
        rng.standard_normals(count)[..count]
            .iter()
            .map(|value| value * scale)
            .collect::<Vec<f32>>()
    };
    let rows = (0..8)
        .map(|_| {
            (
                normals(hidden_size, 3.0),
                normals(LATENT_DIM, 1.0),
                normals(LATENT_DIM, 1.0),
            )
        })
        .collect::<Vec<_>>();
    let alone = rows
        .iter()
        .map(|(hidden, history, noise)| {
            runtime
                .sample_latent(hidden, history, noise, DEFAULT_GUIDANCE)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let host = |tensor: candle_core::Tensor| {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    };
    let embedded_alone = alone
        .iter()
        .map(|latent| host(runtime.latent_embeddings(latent).unwrap()).remove(0))
        .collect::<Vec<_>>();
    for count in 2..=rows.len() {
        let inputs = rows[..count]
            .iter()
            .map(|(hidden, history, noise)| super::MingLatentInput {
                hidden,
                history,
                noise,
                guidance: DEFAULT_GUIDANCE,
            })
            .collect::<Vec<_>>();
        let together = runtime.sample_latents(&inputs).unwrap();
        assert_eq!(together, alone[..count], "sampler, {count} requests");
        let embedded = host(runtime.latent_embeddings(&alone[..count].concat()).unwrap());
        assert_eq!(
            embedded,
            embedded_alone[..count],
            "aggregator, {count} requests"
        );
    }
}

/// Speaker similarity between stretches of WAV files (a diagnostic; it
/// asserts nothing). `IZWI_MING_TTS_VOICE_WAVS` lists `path:start-end,...`
/// (seconds) entries separated by `;`; each stretch is embedded with CAM++
/// and compared with every other stretch of its file.
#[test]
#[ignore = "needs the Ming-3.0-tts-0.5B checkpoint"]
fn voice_identity_across_stretches() {
    let Ok(spec) = std::env::var("IZWI_MING_TTS_VOICE_WAVS") else {
        eprintln!("set IZWI_MING_TTS_VOICE_WAVS");
        return;
    };
    let speaker = CampPlus::open(&model_dir().join(SPEAKER_WEIGHTS_FILE)).unwrap();
    let cosine = |a: &[f32], b: &[f32]| {
        let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b))
    };
    for entry in spec.split(';').filter(|entry| !entry.trim().is_empty()) {
        let (path, windows) = entry.rsplit_once(':').unwrap();
        let bytes = std::fs::read(path).unwrap();
        // 16-bit mono PCM: find the `data` chunk.
        let rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let data = bytes.windows(4).position(|w| w == b"data").unwrap() + 8;
        let samples = bytes[data..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32768.0)
            .collect::<Vec<_>>();
        let stretches = windows
            .split(',')
            .map(|window| {
                let (start, end) = window.split_once('-').unwrap();
                let at = |s: &str| (s.parse::<f32>().unwrap() * rate as f32) as usize;
                let (start, end) = (at(start), at(end).min(samples.len()));
                (
                    window.to_string(),
                    speaker.embed_clip(&samples[start..end], rate).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        eprintln!("{path}");
        for (name, a) in &stretches {
            let row = stretches
                .iter()
                .map(|(_, b)| format!("{:5.2}", cosine(a, b)))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("  {name:>12}: {row}");
        }
    }
}
