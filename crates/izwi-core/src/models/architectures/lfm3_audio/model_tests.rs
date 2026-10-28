//! Real-checkpoint checks for LFM3-Audio-2B (Q4_K_M language model, Q8_0
//! mmproj).
//!
//! The tests are ignored because they need the published GGUFs. They look in
//! `IZWI_LFM25_MODELS_DIR`, else the default models directory, and skip when
//! `LFM3-Audio-2B-GGUF` is missing. Each backend leg checks:
//! * ASR of `data/fox.wav` returns the spoken sentence;
//! * TTS of a sentence, transcribed back by the same model, returns it, for
//!   both voices;
//! * the speech streamed through the codec equals one offline decode of the
//!   same frames;
//! * a fixed seed reproduces the same speech.
//!
//! Run: `cargo test -p izwi-core --features metal --release --lib lfm3_audio::model::tests -- --ignored --nocapture`
use super::*;
use crate::backends::DeviceProfile;
use crate::models::architectures::lfm2::physical::Lfm2StateLayout;
use crate::models::shared::attention::physical::test_physical_caches_for_model_layers_on;
use std::time::Instant;

use super::super::prompts::{asr_prompt, tts_prompt, Lfm3AudioVoice};

const FOLDER: &str = "LFM3-Audio-2B-GGUF";
const SENTENCE: &str = "The quick brown fox jumps over the lazy dog.";
/// Context each test sequence reserves.
const CAPACITY_TOKENS: usize = 4_096;

fn models_dir() -> PathBuf {
    std::env::var_os("IZWI_LFM25_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .expect("data dir")
                .join("izwi")
                .join("models")
        })
}

fn read_wav(name: &str) -> (Vec<f32>, u32) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data")
        .join(name);
    let mut reader = hound::WavReader::open(path).expect("fixture wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    let samples = reader
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32_768.0)
        .collect();
    (samples, spec.sample_rate)
}

fn write_wav(name: &str, samples: &[f32]) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-e2e/out");
    std::fs::create_dir_all(&dir).unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: OUTPUT_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(dir.join(name), spec).unwrap();
    for sample in samples {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * 32_767.0) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word error rate of `hypothesis` against `reference`.
fn wer(reference: &str, hypothesis: &str) -> f32 {
    let (reference, hypothesis) = (words(reference), words(hypothesis));
    let mut row = (0..=hypothesis.len()).collect::<Vec<_>>();
    for (i, word) in reference.iter().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, other) in hypothesis.iter().enumerate() {
            let substitution = previous + usize::from(word != other);
            previous = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
        }
    }
    row[hypothesis.len()] as f32 / reference.len().max(1) as f32
}

fn fresh_state(model: &Lfm3AudioModel) -> RetainedLmState {
    let config = model.lm_config();
    let layout = Lfm2StateLayout::from_config(config).unwrap();
    let kv_heads = config
        .attention_head_count_kv
        .iter()
        .copied()
        .max()
        .unwrap();
    let head_dim = config.embedding_length / config.attention_head_count;
    let cache = test_physical_caches_for_model_layers_on(
        &model.device().device,
        DType::F32,
        &layout.attention_model_layers,
        kv_heads,
        head_dim,
        CAPACITY_TOKENS,
        1,
    )
    .pop()
    .unwrap();
    RetainedLmState {
        cache,
        shortconv: model.lm.new_shortconv_state(),
    }
}

fn run(model: &Lfm3AudioModel, request: Lfm3AudioRequest) -> (Lfm3AudioOutput, f64) {
    let mut state = fresh_state(model);
    let started = Instant::now();
    let mut streamed = Vec::new();
    let output = model
        .generate(
            request,
            &mut state,
            &mut || false,
            &mut |_| {},
            &mut |samples| streamed.extend_from_slice(samples),
        )
        .unwrap();
    assert_eq!(streamed, output.samples, "streamed audio is the output");
    (output, started.elapsed().as_secs_f64())
}

fn transcribe(model: &Lfm3AudioModel, samples: &[f32], sample_rate: u32) -> (String, f64) {
    let started = Instant::now();
    let audio = model.encode_audio(samples, sample_rate).unwrap();
    let (output, decode) = run(
        model,
        Lfm3AudioRequest {
            prompt: asr_prompt(Some("en")).unwrap(),
            audio: Some(audio),
            mode: Lfm3AudioMode::Text,
            text_only: true,
            max_slots: 256,
            text_sampling: Lfm3TextSampling::default(),
            audio_sampling: Lfm3AudioSampling::default(),
            seed: 0,
        },
    );
    assert_eq!(output.stop_reason, "eos", "ASR ends its turn");
    let _ = decode;
    (output.text, started.elapsed().as_secs_f64())
}

fn speak(model: &Lfm3AudioModel, voice: &str, seed: u64) -> (Lfm3AudioOutput, f64) {
    let voice = Lfm3AudioVoice::resolve(Some(voice), Some("en-US")).unwrap();
    run(
        model,
        Lfm3AudioRequest {
            prompt: tts_prompt(SENTENCE, voice),
            audio: None,
            mode: Lfm3AudioMode::Audio,
            text_only: false,
            max_slots: 400,
            text_sampling: Lfm3TextSampling::default(),
            audio_sampling: Lfm3AudioSampling::default(),
            seed,
        },
    )
}

fn run_leg(profile: DeviceProfile) {
    let dir = models_dir().join(FOLDER);
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let backend = format!("{:?}", profile.kind);
    let started = Instant::now();
    let model = Lfm3AudioModel::load(&dir, ModelVariant::Lfm3Audio2BGguf, profile).unwrap();
    eprintln!(
        "{backend}: loaded in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(model.max_context_tokens(), LFM3_AUDIO_CONTEXT_TOKENS);

    let (fox, rate) = read_wav("fox.wav");
    let (text, seconds) = transcribe(&model, &fox, rate);
    let clip = fox.len() as f64 / f64::from(rate);
    eprintln!(
        "{backend}: ASR fox.wav -> {text:?} in {seconds:.2}s (RTF {:.3})",
        seconds / clip
    );
    assert!(wer(SENTENCE, &text) <= 0.1, "{backend}: ASR {text:?}");

    for voice in ["female", "male"] {
        let (output, seconds) = speak(&model, voice, 7);
        let audio = output.samples.len() as f64 / f64::from(OUTPUT_SAMPLE_RATE);
        eprintln!(
            "{backend}: TTS {voice}: {} frames, {audio:.2}s audio in {seconds:.2}s (RTF {:.3}), stop {}",
            output.audio_frames,
            seconds / audio.max(1e-6),
            output.stop_reason
        );
        write_wav(&format!("{backend}-{voice}.wav"), &output.samples);
        assert_eq!(
            output.stop_reason, "eos",
            "{backend}/{voice}: TTS ends its turn"
        );
        assert!(
            (1.5..8.0).contains(&audio),
            "{backend}/{voice}: {audio:.2}s of speech for one sentence"
        );
        assert_eq!(
            output.samples.len(),
            output.audio_frames * SAMPLES_PER_FRAME,
            "{backend}/{voice}: one frame is 80 ms"
        );
        let offline = model.codec.decode(&output.frames).unwrap();
        let drift = offline
            .iter()
            .zip(&output.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{backend}: TTS {voice} streamed vs offline codec drift {drift:e}");
        assert_eq!(offline.len(), output.samples.len());
        assert!(
            drift <= 1e-4,
            "{backend}/{voice}: streamed codec drift {drift}"
        );
        let (heard, _) = transcribe(&model, &output.samples, OUTPUT_SAMPLE_RATE);
        let error = wer(SENTENCE, &heard);
        eprintln!("{backend}: TTS {voice} round trip -> {heard:?} (WER {error:.2})");
        assert!(error <= 0.2, "{backend}/{voice}: round trip {heard:?}");
    }

    let (first, _) = speak(&model, "female", 11);
    let (second, _) = speak(&model, "female", 11);
    assert_eq!(
        first.samples, second.samples,
        "{backend}: a seed fixes the speech"
    );
}

#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_transcribes_and_speaks_on_cpu() {
    run_leg(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_transcribes_and_speaks_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease =
        crate::backends::metal_test_device_or_skip("lfm3_audio_transcribes_and_speaks_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    run_leg(profile);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_transcribes_and_speaks_on_cuda() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Cuda).unwrap();
    if !matches!(profile.device, Device::Cuda(_)) {
        eprintln!("skipping: no CUDA device");
        return;
    }
    run_leg(profile);
}

/// Per-frame cost of each stage, warm, on one backend.
fn time_components(profile: DeviceProfile) {
    let dir = models_dir().join(FOLDER);
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let backend = format!("{:?}", profile.kind);
    let model = Lfm3AudioModel::load(&dir, ModelVariant::Lfm3Audio2BGguf, profile).unwrap();
    let device = model.device().device.clone();
    let sync = || device.synchronize().unwrap();
    let mut state = fresh_state(&model);
    let voice = Lfm3AudioVoice::resolve(Some("female"), Some("en-US")).unwrap();
    let request = Lfm3AudioRequest {
        prompt: tts_prompt(SENTENCE, voice),
        audio: None,
        mode: Lfm3AudioMode::Audio,
        text_only: false,
        max_slots: 1,
        text_sampling: Lfm3TextSampling::default(),
        audio_sampling: Lfm3AudioSampling::default(),
        seed: 3,
    };
    let (embeds, tokens) = model.prompt_embeds(&request).unwrap();
    let hidden = state.forward(&model.lm, &embeds, 0).unwrap();
    let mut last = hidden
        .i((.., tokens - 1..tokens, ..))
        .unwrap()
        .squeeze(1)
        .unwrap();
    let mut rng = SplitMix64::new(1);
    let history = FrameHistory::default();
    let sampling = Lfm3AudioSampling::default();
    let mut frames = Vec::new();
    let time = |label: &str, iterations: usize, run: &mut dyn FnMut()| {
        run();
        sync();
        let started = Instant::now();
        for _ in 0..iterations {
            run();
        }
        sync();
        eprintln!(
            "{backend}: {label}: {:.2} ms",
            started.elapsed().as_secs_f64() * 1000.0 / iterations as f64
        );
    };
    let mut position = tokens;
    time("frame (depth + feedback + LM step)", 24, &mut || {
        let frame = model
            .depth
            .sample_frames(&last, &sampling, &[&history], &mut [&mut rng])
            .unwrap()[0];
        frames.push(frame);
        let next = model.feedback_embed(&frame).unwrap();
        last = state
            .forward(&model.lm, &next, position)
            .unwrap()
            .squeeze(1)
            .unwrap();
        position += 1;
    });
    let frame = frames[0];
    time("depth sample_frames (3 passes)", 24, &mut || {
        model
            .depth
            .sample_frames(&last, &sampling, &[&history], &mut [&mut rng])
            .unwrap();
    });
    let inputs = model.depth.frame_inputs(&last).unwrap();
    time("depth one pass (logits on device)", 24, &mut || {
        model
            .depth
            .pass_logits(
                &inputs,
                &[frame],
                &[[false; CODEBOOKS]],
                &[(0, 0), (0, 3), (0, 6), (0, 9)],
            )
            .unwrap();
    });
    time("feedback embed", 24, &mut || {
        model.feedback_embed(&frame).unwrap();
    });
    let next = model.feedback_embed(&frame).unwrap();
    let mut lm_state = fresh_state(&model);
    lm_state.forward(&model.lm, &embeds, 0).unwrap();
    let mut lm_position = tokens;
    time("LM step", 24, &mut || {
        lm_state.forward(&model.lm, &next, lm_position).unwrap();
        lm_position += 1;
    });
    let speech = frames
        .iter()
        .copied()
        .filter(|frame| frame[0] != AUDIO_EOS_CODE)
        .take(6)
        .collect::<Vec<_>>();
    let mut codec = super::super::codec::LiquidCodecStream::default();
    time("codec 6 frames", 8, &mut || {
        model
            .codec
            .decode_stream(&mut codec, &speech, false)
            .unwrap();
    });
}

#[test]
#[ignore = "timing; needs the LFM3-Audio-2B GGUF"]
fn lfm3_audio_component_timing_on_cpu() {
    time_components(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "timing; needs the LFM3-Audio-2B GGUF"]
fn lfm3_audio_component_timing_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("lfm3_audio_component_timing_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        return;
    }
    time_components(profile);
}
