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
    // The KV storage production leases on this backend.
    let kv_dtype = match super::super::physical::lfm3_audio_kv_dtype(model.device().kind.into()) {
        crate::kv::v2::StateDType::F16 => DType::F16,
        _ => DType::F32,
    };
    let cache = test_physical_caches_for_model_layers_on(
        &model.device().device,
        kv_dtype,
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
            .sample_frames(&last, &[&sampling], &[&history], &mut [&mut rng])
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
            .sample_frames(&last, &[&sampling], &[&history], &mut [&mut rng])
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

/// Each language-model decode row's hidden state, step by step, decoded
/// alone (`width` 1) or beside the other rows, after a solo prefill of a
/// prompt of its own length.
fn lm_decode_rows(model: &Lfm3AudioModel, together: bool) -> Vec<Vec<Vec<f32>>> {
    use crate::kernels::numeric_check::xorshift_tensor;
    let config = model.lm_config();
    let layout = Lfm2StateLayout::from_config(config).unwrap();
    let kv_heads = config
        .attention_head_count_kv
        .iter()
        .copied()
        .max()
        .unwrap();
    let head_dim = config.embedding_length / config.attention_head_count;
    let kv_dtype = match super::super::physical::lfm3_audio_kv_dtype(model.device().kind.into()) {
        crate::kv::v2::StateDType::F16 => DType::F16,
        _ => DType::F32,
    };
    let device = &model.device().device;
    let hidden = config.embedding_length;
    let prompts = [5usize, 9, 3, 12];
    let mut caches = test_physical_caches_for_model_layers_on(
        device,
        kv_dtype,
        &layout.attention_model_layers,
        kv_heads,
        head_dim,
        256,
        prompts.len(),
    );
    let mut states = (0..prompts.len())
        .map(|_| model.lm.new_shortconv_state())
        .collect::<Vec<_>>();
    for (row, len) in prompts.iter().enumerate() {
        let prompt = xorshift_tensor(&[1, *len, hidden], 100 + row as u64, 0.5, device).unwrap();
        model
            .lm
            .forward_embeds_retained(&prompt, 0, &mut caches[row], &mut states[row])
            .unwrap();
    }
    let mut out = vec![Vec::new(); prompts.len()];
    for step in 0..6u64 {
        let embeds = (0..prompts.len())
            .map(|row| xorshift_tensor(&[1, 1, hidden], 1000 * step + row as u64, 0.5, device))
            .collect::<candle_core::Result<Vec<_>>>()
            .unwrap();
        let positions = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        let groups: Vec<Vec<usize>> = if together {
            vec![(0..prompts.len()).collect()]
        } else {
            (0..prompts.len()).map(|row| vec![row]).collect()
        };
        for group in groups {
            let mut group_states = states
                .iter_mut()
                .enumerate()
                .filter(|(row, _)| group.contains(row))
                .map(|(_, state)| state)
                .collect::<Vec<_>>();
            let mut group_caches = caches
                .iter_mut()
                .enumerate()
                .filter(|(row, _)| group.contains(row))
                .map(|(_, cache)| cache)
                .collect::<Vec<_>>();
            let output = model
                .lm
                .forward_embeds_retained_batch(
                    &group
                        .iter()
                        .map(|row| embeds[*row].clone())
                        .collect::<Vec<_>>(),
                    &group.iter().map(|row| positions[*row]).collect::<Vec<_>>(),
                    &mut group_states,
                    &mut group_caches,
                )
                .unwrap();
            for (row, last) in group.iter().zip(output.last_hidden) {
                out[*row].push(last.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            }
        }
    }
    out
}

/// Each row's speech frames over a few draws, from its own generator and
/// history, sampled alone or beside the other rows.
fn depth_frames(model: &Lfm3AudioModel, hidden: &Tensor, together: bool) -> Vec<[u32; CODEBOOKS]> {
    let rows = hidden.dim(0).unwrap();
    let sampling = Lfm3AudioSampling::default();
    let mut rngs = (0..rows)
        .map(|row| SplitMix64::new(7 + row as u64))
        .collect::<Vec<_>>();
    let history = FrameHistory::default();
    let mut frames = Vec::new();
    for _ in 0..3 {
        if together {
            frames.extend(
                model
                    .depth
                    .sample_frames(
                        hidden,
                        &vec![&sampling; rows],
                        &vec![&history; rows],
                        &mut rngs.iter_mut().collect::<Vec<_>>(),
                    )
                    .unwrap(),
            );
        } else {
            for (row, rng) in rngs.iter_mut().enumerate() {
                frames.extend(
                    model
                        .depth
                        .sample_frames(
                            &hidden.narrow(0, row, 1).unwrap(),
                            &[&sampling],
                            &[&history],
                            &mut [rng],
                        )
                        .unwrap(),
                );
            }
        }
    }
    frames
}

/// With rows made invariant, a decode row has the same bits alone and four
/// wide, in the language model and the depth decoder: batched audio then
/// samples what a request served alone samples.
fn rows_are_batch_invariant(profile: DeviceProfile) {
    let dir = models_dir().join(FOLDER);
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let backend = format!("{:?}", profile.kind);
    // LFM3-Audio loads with `make_decode_rows_invariant` on.
    let model = Lfm3AudioModel::load(&dir, ModelVariant::Lfm3Audio2BGguf, profile).unwrap();
    assert_eq!(
        lm_decode_rows(&model, false),
        lm_decode_rows(&model, true),
        "{backend}: invariant language-model rows"
    );

    let hidden = crate::kernels::numeric_check::xorshift_tensor(
        &[4, model.lm_config().embedding_length],
        77,
        2.0,
        &model.device().device,
    )
    .unwrap();
    assert_eq!(
        depth_frames(&model, &hidden, true),
        depth_frames(&model, &hidden, false),
        "{backend}: depth rows differ alone and four wide"
    );
}

#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_rows_are_batch_invariant_on_cpu() {
    rows_are_batch_invariant(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_rows_are_batch_invariant_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease =
        crate::backends::metal_test_device_or_skip("lfm3_audio_rows_are_batch_invariant_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        return;
    }
    rows_are_batch_invariant(profile);
}

/// One test arena's paged caches for `rows` language-model rows, in the KV
/// dtype production leases on this backend.
fn row_caches(model: &Lfm3AudioModel, rows: usize, capacity: usize) -> Vec<PhysicalPagedKvCache> {
    let config = model.lm_config();
    let layout = Lfm2StateLayout::from_config(config).unwrap();
    let kv_heads = config
        .attention_head_count_kv
        .iter()
        .copied()
        .max()
        .unwrap();
    let head_dim = config.embedding_length / config.attention_head_count;
    let kv_dtype = match super::super::physical::lfm3_audio_kv_dtype(model.device().kind.into()) {
        crate::kv::v2::StateDType::F16 => DType::F16,
        _ => DType::F32,
    };
    test_physical_caches_for_model_layers_on(
        &model.device().device,
        kv_dtype,
        &layout.attention_model_layers,
        kv_heads,
        head_dim,
        capacity,
        rows,
    )
}

/// Each seeded row's frames, samples and text: one row per request that
/// `request(i)` builds. Row `i` prefills and joins the batch at decode step
/// `joins[i]`; alone, every row runs by itself.
fn run_rows(
    model: &Lfm3AudioModel,
    rows: usize,
    request: impl Fn(usize) -> Lfm3AudioRequest,
    joins: &[usize],
    together: bool,
) -> Vec<(Vec<[u32; CODEBOOKS]>, Vec<f32>, String)> {
    use super::super::retained::{Lfm3AudioRetainedState, Lfm3AudioStepRow};
    let capacity = 512;
    let mut caches = row_caches(model, rows, capacity);
    let mut states = (0..rows)
        .map(|row| {
            Lfm3AudioRetainedState::new(model, model.prepare_row(request(row)).unwrap(), capacity)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let rows = vec![(); rows];
    let mut samples = vec![Vec::new(); rows.len()];
    let schedules: Vec<Vec<usize>> = if together {
        vec![(0..rows.len()).collect()]
    } else {
        (0..rows.len()).map(|row| vec![row]).collect()
    };
    for group in schedules {
        let mut joined = Vec::new();
        for step in 0.. {
            for &row in &group {
                if !joined.contains(&row) && (!together || joins[row] == step) {
                    let span = states[row].prompt_tokens();
                    model
                        .retained_prefill(&mut states[row], &mut caches[row], span)
                        .unwrap();
                    joined.push(row);
                }
            }
            let live = joined
                .iter()
                .copied()
                .filter(|&row| !states[row].finished())
                .collect::<Vec<_>>();
            if live.is_empty() && joined.len() == group.len() {
                break;
            }
            let mut step_rows = states
                .iter_mut()
                .zip(caches.iter_mut())
                .enumerate()
                .filter(|(row, _)| live.contains(row))
                .map(|(_, (state, cache))| Lfm3AudioStepRow { state, cache })
                .collect::<Vec<_>>();
            if !step_rows.is_empty() {
                model.retained_decode_batch(&mut step_rows).unwrap();
            }
            drop(step_rows);
            for &row in &live {
                samples[row].extend(states[row].take_samples());
            }
        }
    }
    states
        .iter()
        .zip(samples)
        .map(|(state, samples)| {
            let output = state.output().unwrap();
            (output.frames.clone(), samples, output.text.clone())
        })
        .collect()
}

/// Speech rows joining a running batch at different steps each speak
/// exactly what they speak alone: the same frames and the same samples.
fn tts_rows_are_batch_invariant(profile: DeviceProfile) {
    let dir = models_dir().join(FOLDER);
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let backend = format!("{:?}", profile.kind);
    let model = Lfm3AudioModel::load(&dir, ModelVariant::Lfm3Audio2BGguf, profile).unwrap();
    let female = Lfm3AudioVoice::resolve(Some("female"), None).unwrap();
    let male = Lfm3AudioVoice::resolve(Some("male"), None).unwrap();
    let rows = [
        (SENTENCE, female, 11),
        ("Batching keeps every voice its own.", female, 12),
        (SENTENCE, male, 13),
        ("A late row joins mid-sentence.", female, 14),
    ];
    let joins = [0, 0, 5, 11];
    let request = |row: usize| {
        let (text, voice, seed) = rows[row];
        tts_request(text, voice, 60, Lfm3AudioSampling::default(), seed).unwrap()
    };
    let alone = run_rows(&model, rows.len(), request, &joins, false);
    let together = run_rows(&model, rows.len(), request, &joins, true);
    for (index, (alone, together)) in alone.iter().zip(&together).enumerate() {
        eprintln!(
            "{backend}: row {index}: {} frames, {} samples alone",
            alone.0.len(),
            alone.1.len()
        );
        assert!(!alone.0.is_empty(), "{backend}: row {index} spoke");
        assert_eq!(alone.0, together.0, "{backend}: row {index} frames");
        assert!(alone.1 == together.1, "{backend}: row {index} samples");
    }
}

/// LiveASR fed 100 ms pieces writes the speech it hears as it hears it, in
/// segments of at most 29 s as the runtime runs them: each new segment hears
/// the previous one's last 2.88 s again and the transcript drops the words
/// that repeat. A 29.7 s clip and the same clip then the fox sentence (34 s)
/// both lose no word where the second segment starts.
fn live_asr_transcribes_while_streaming(profile: DeviceProfile) {
    use super::live_asr::{
        Lfm3LiveTranscript, LFM3_LIVE_ASR_OVERLAP_SECONDS, LFM3_LIVE_ASR_SEGMENT_SECONDS,
    };
    let dir = models_dir().join(FOLDER);
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let backend = format!("{:?}", profile.kind);
    let model = Lfm3AudioModel::load(&dir, ModelVariant::Lfm3Audio2BGguf, profile).unwrap();
    let to_16k = |(samples, rate): (Vec<f32>, u32)| {
        crate::audio::resample_mono_high_quality(&samples, rate, INPUT_SAMPLE_RATE).unwrap()
    };
    let speech = to_16k(read_wav("test.wav"));
    let fox = to_16k(read_wav("fox.wav"));
    let reference = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/test.md"),
    )
    .unwrap();
    let long = [speech.as_slice(), &[0.0; 16_000], &fox].concat();
    let segment = (LFM3_LIVE_ASR_SEGMENT_SECONDS * 16_000.0) as usize;
    let overlap = (LFM3_LIVE_ASR_OVERLAP_SECONDS * 16_000.0) as usize;
    for (name, audio, reference) in [
        ("29.7 s", &speech, reference.clone()),
        ("34 s", &long, format!("{reference} {SENTENCE}")),
    ] {
        let mut transcript = Lfm3LiveTranscript::default();
        let mut live = model.start_live_asr();
        let mut state = fresh_state(&model);
        let mut text = String::new();
        let (mut heard, mut segments, mut slowest) = (0, 1, 0f64);
        let started = Instant::now();
        let pieces = audio.chunks(1_600).collect::<Vec<_>>();
        for (index, piece) in pieces.iter().enumerate() {
            let push = Instant::now();
            if heard + piece.len() > segment {
                let tail = model
                    .push_live_asr(&mut live, &[], true, &mut state)
                    .unwrap();
                text.push_str(&transcript.push(&tail));
                live = model.start_live_asr();
                state = fresh_state(&model);
                segments += 1;
                text.push_str(&transcript.start_segment());
                let offset = index * 1_600;
                let again = &audio[offset - overlap..offset];
                let repeat = model
                    .push_live_asr(&mut live, again, false, &mut state)
                    .unwrap();
                text.push_str(&transcript.push(&repeat));
                heard = again.len();
            }
            let is_final = index + 1 == pieces.len();
            let written = model
                .push_live_asr(&mut live, piece, is_final, &mut state)
                .unwrap();
            text.push_str(&transcript.push(&written));
            heard += piece.len();
            slowest = slowest.max(push.elapsed().as_secs_f64());
        }
        text.push_str(&transcript.flush());
        assert!(live.finished());
        let seconds = audio.len() as f64 / 16_000.0;
        let words_heard = words(&text);
        let recall = words(&reference)
            .iter()
            .filter(|word| words_heard.contains(word))
            .count() as f64
            / words(&reference).len() as f64;
        eprintln!(
            "{backend}: LiveASR {name} -> recall {recall:.2}, WER {:.2}, {segments} segments, \
             RTF {:.2}, slowest push {:.0} ms: {text:?}",
            wer(&reference, &text),
            started.elapsed().as_secs_f64() / seconds,
            slowest * 1000.0
        );
        assert_eq!(segments, 2, "{backend}: {name} segments");
        assert!(recall >= 0.9, "{backend}: {name} recall {recall:.2}");
        assert!(
            words_heard.contains(&"impossible".to_string()),
            "{backend}: {name} kept the last word before the new segment"
        );
        assert_eq!(
            words_heard
                .iter()
                .filter(|word| *word == "amplifying")
                .count(),
            1,
            "{backend}: {name} repeated the overlap"
        );
    }
}

#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_live_asr_on_cpu() {
    live_asr_transcribes_while_streaming(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_live_asr_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("lfm3_audio_live_asr_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        return;
    }
    live_asr_transcribes_while_streaming(profile);
}

/// Voice-chat replies (text and speech interleaved) joining a running batch
/// at different steps each reply exactly as they reply alone, past the 256
/// tokens where CPU attention starts slicing a row's context.
fn reply_rows_are_batch_invariant(profile: DeviceProfile) {
    let dir = models_dir().join(FOLDER);
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let backend = format!("{:?}", profile.kind);
    let model = Lfm3AudioModel::load(&dir, ModelVariant::Lfm3Audio2BGguf, profile).unwrap();
    let (fox, rate) = read_wav("fox.wav");
    let padded = [vec![0.0; rate as usize], fox.clone()].concat();
    let turns = [(&fox, 31), (&fox, 32), (&padded, 33), (&fox, 34)];
    let joins = [0, 0, 5, 11];
    let request = |row: usize| {
        let (audio, seed) = turns[row];
        reply_request(
            &[],
            model.encode_audio(audio, rate).unwrap(),
            None,
            220,
            Lfm3TextSampling::default(),
            Lfm3AudioSampling::default(),
            seed,
        )
    };
    let alone = run_rows(&model, turns.len(), request, &joins, false);
    let together = run_rows(&model, turns.len(), request, &joins, true);
    for (index, (alone, together)) in alone.iter().zip(&together).enumerate() {
        eprintln!(
            "{backend}: reply {index}: {} frames, {} samples, {:?} alone",
            alone.0.len(),
            alone.1.len(),
            alone.2
        );
        assert!(!alone.0.is_empty(), "{backend}: reply {index} spoke");
        assert_eq!(alone.2, together.2, "{backend}: reply {index} text");
        assert_eq!(alone.0, together.0, "{backend}: reply {index} frames");
        assert!(alone.1 == together.1, "{backend}: reply {index} samples");
    }
}

#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_reply_rows_are_batch_invariant_on_cpu() {
    reply_rows_are_batch_invariant(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_reply_rows_are_batch_invariant_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip(
        "lfm3_audio_reply_rows_are_batch_invariant_on_metal",
    );
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        return;
    }
    reply_rows_are_batch_invariant(profile);
}

#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_tts_rows_are_batch_invariant_on_cpu() {
    tts_rows_are_batch_invariant(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM3-Audio-2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_audio_tts_rows_are_batch_invariant_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip(
        "lfm3_audio_tts_rows_are_batch_invariant_on_metal",
    );
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        return;
    }
    tts_rows_are_batch_invariant(profile);
}
