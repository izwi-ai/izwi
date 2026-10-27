//! Streamed against final-only synthesis on real weights, for choosing the
//! window plan and gating seams.
//!
//! T3 does not depend on streaming, so a stream renders exactly the tokens
//! the final-only utterance does. Each sentence is generated final-only,
//! then those tokens are fed to a stream as T3 would sample them, under
//! each candidate [`ChatterboxStreamPlan`]. Reported per plan:
//! * cost: the acoustic time of each window and of the final-only render,
//!   and when the first audio would be ready (T3's time to the first
//!   window's tokens plus that window);
//! * seams: the streamed and final-only log spectra around each window's
//!   first sample against their distance elsewhere, and the waveform's
//!   largest step there against its largest step elsewhere;
//! * the voice encoder's cosine between streamed and final-only audio, and
//!   PerTh's detector score on each.
//!
//! WAVs land in `target/chatterbox-e2e/out/stream/` for transcription.
//! With `IZWI_STREAM_REFERENCE` (a 16-bit mono WAV) the voice is a clone
//! of its first 10 s, and each stream's voice-encoder cosine to it is
//! reported too.
//!
//! Run: `IZWI_CHATTERBOX_MODEL_DIR=… cargo test -p izwi-core --release
//! --features metal --lib chatterbox::stream_tests -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use candle_core::DType;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use super::perth::{PerthDetector, PERTH_CHECKPOINT_FILE};
use super::reference_dsp::sinc_resample;
use super::stream::{ChatterboxStreamPlan, ChatterboxStreamUnit, CHATTERBOX_STREAM_PLAN};
use super::*;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

const SENTENCES: [&str; 3] = [
    "The museum opens at nine, and the guided tour of the old lighthouse begins every hour.",
    "Please read the instructions twice before you begin, then write your name at the top.",
    "When the storm finally passed, the villagers walked down to the shore and found the old \
     lighthouse still standing, its lamp dark but its walls unbroken, as if it had simply \
     waited for them to return.",
];

fn model_dir() -> PathBuf {
    std::env::var_os("IZWI_CHATTERBOX_MODEL_DIR")
        .map(PathBuf::from)
        .expect("IZWI_CHATTERBOX_MODEL_DIR")
}

fn out_dir() -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/chatterbox-e2e/out/stream");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_wav(name: &str, samples: &[f32]) {
    let path = out_dir().join(name);
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    let data = (samples.len() * 2) as u32;
    let rate = S3GEN_SAMPLE_RATE;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

/// Log-magnitude frames (1024-point, 10 ms hop) of `samples`.
fn log_spectra(samples: &[f32]) -> Vec<Vec<f32>> {
    const N: usize = 1024;
    const HOP: usize = 240;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N);
    let window = (0..N)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos())
        .collect::<Vec<_>>();
    let mut frames = Vec::new();
    let mut start = 0;
    while start + N <= samples.len() {
        let mut buffer = (0..N)
            .map(|i| Complex::new(samples[start + i] * window[i], 0.0))
            .collect::<Vec<_>>();
        fft.process(&mut buffer);
        frames.push(
            buffer[..N / 2 + 1]
                .iter()
                .map(|value| (value.norm() + 1e-5).ln())
                .collect(),
        );
        start += HOP;
    }
    frames
}

fn percentile(values: &mut [f32], share: f32) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[((values.len() - 1) as f32 * share) as usize]
}

struct SeamReport {
    spectral_seam: f32,
    spectral_elsewhere_p95: f32,
    step_seam: f32,
    step_elsewhere_p99: f32,
}

/// Spectral distance to the final-only render and waveform steps around
/// each seam (±50 ms) against the rest.
fn seams(streamed: &[f32], whole: &[f32], seams: &[usize]) -> SeamReport {
    let shared = streamed.len().min(whole.len());
    let (a, b) = (
        log_spectra(&streamed[..shared]),
        log_spectra(&whole[..shared]),
    );
    let distance = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum::<f32>() / x.len() as f32)
        .collect::<Vec<_>>();
    let near = |sample: usize| seams.iter().any(|&seam| sample.abs_diff(seam) < 1200);
    let mut spectral_seam = 0f32;
    let mut elsewhere = Vec::new();
    for (frame, d) in distance.iter().enumerate() {
        if near(frame * 240 + 512) {
            spectral_seam = spectral_seam.max(*d);
        } else {
            elsewhere.push(*d);
        }
    }
    let mut step_seam = 0f32;
    let mut steps = Vec::new();
    for n in 1..streamed.len() {
        let step = (streamed[n] - streamed[n - 1]).abs();
        if near(n) {
            step_seam = step_seam.max(step);
        } else {
            steps.push(step);
        }
    }
    SeamReport {
        spectral_seam,
        spectral_elsewhere_p95: percentile(&mut elsewhere, 0.95),
        step_seam,
        step_elsewhere_p99: percentile(&mut steps, 0.999),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "needs the Chatterbox checkpoint (IZWI_CHATTERBOX_MODEL_DIR)"]
fn streamed_windows_against_final_only_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("streamed_windows_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, candle_core::Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    let dir = model_dir();
    let runtime = ChatterboxRuntime::load(&dir, &profile.device, DType::F16, true).unwrap();
    let detector =
        PerthDetector::load(&dir.join(PERTH_CHECKPOINT_FILE), &candle_core::Device::Cpu).unwrap();
    let embed = |samples: &[f32]| {
        let wav16 = sinc_resample(samples, S3GEN_SAMPLE_RATE, 16_000);
        runtime
            .reference
            .voice_encoder()
            .embed(&wav16, &profile.device)
            .unwrap()
    };
    let plans = std::env::var("IZWI_STREAM_PLANS")
        .ok()
        .map(|spec| {
            spec.split(';')
                .map(|plan| {
                    let v = plan
                        .split(',')
                        .map(|n| n.trim().parse::<usize>().unwrap())
                        .collect::<Vec<_>>();
                    ChatterboxStreamPlan {
                        first_tokens: v[0],
                        max_tokens: v[1],
                        left_context_tokens: v[2],
                        prompt_tokens: v[3],
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![CHATTERBOX_STREAM_PLAN]);
    // A clone from a 16-bit mono WAV's first 10 s, or the bundled voice.
    let reference = std::env::var_os("IZWI_STREAM_REFERENCE").map(|path| {
        let mut reader = hound::WavReader::open(PathBuf::from(path)).unwrap();
        let rate = reader.spec().sample_rate;
        let samples = reader
            .samples::<i16>()
            .take(10 * rate as usize)
            .map(|sample| sample.unwrap() as f32 / 32768.0)
            .collect::<Vec<_>>();
        (samples, rate)
    });
    let voice = match &reference {
        Some((samples, rate)) => runtime.prepare_voice_from_audio(samples, *rate).unwrap(),
        None => runtime.default_voice().clone(),
    };
    let reference_embedding = reference
        .as_ref()
        .map(|(samples, rate)| embed(&sinc_resample(samples, *rate, S3GEN_SAMPLE_RATE)));
    let cfg = runtime.t3().config().clone();
    for (index, text) in SENTENCES.iter().enumerate() {
        let seed = 41 + index as u64;
        let mut params = ChatterboxGenerationParams::defaults(seed);
        params.max_speech_tokens = chatterbox_auto_max_speech_tokens(text);
        let text_ids = runtime.tokenizer().text_sequence(text).unwrap();
        let capacity = runtime.sequence_tokens(text_ids.len(), params.max_speech_tokens);
        let scratch = || {
            PhysicalPagedKvCache::scratch_for_probe(
                &profile.device,
                runtime.t3_dtype(),
                cfg.num_hidden_layers,
                cfg.num_attention_heads,
                cfg.head_dim,
                capacity,
            )
            .unwrap()
        };
        let (mut cond, mut uncond) = (scratch(), scratch());
        let whole = runtime
            .generate_physical(text, &voice, &params, &mut cond, Some(&mut uncond))
            .unwrap();
        let tokens = whole.speech_tokens.clone();
        let t3_per_token = whole.t3_ms / tokens.len().max(1) as f32;
        let whole_acoustic = whole.flow_ms + whole.vocoder_ms;
        write_wav(&format!("s{index}-final.wav"), &whole.samples);
        let whole_embedding = embed(&whole.samples);
        let whole_score = detector.detect(&whole.samples, S3GEN_SAMPLE_RATE).unwrap();
        eprintln!(
            "s{index}: {} tokens, {:.2} s audio, T3 {:.1} ms/token, final-only acoustic {:.0} ms, watermark {:.3}",
            tokens.len(),
            whole.samples.len() as f32 / S3GEN_SAMPLE_RATE as f32,
            t3_per_token,
            whole_acoustic,
            whole_score
        );
        for plan in &plans {
            let mut stream = runtime.stream(*plan, seed);
            let mut audio = Vec::new();
            let mut seam_at = Vec::new();
            let mut window_ms = Vec::new();
            let mut first_audio_ms = None;
            let mut acoustic_ms = 0f32;
            for available in 1..=tokens.len() + 1 {
                let stopped = available > tokens.len();
                let visible = &tokens[..available.min(tokens.len())];
                while runtime.stream_window_due(&stream, visible.len(), stopped) {
                    let started = Instant::now();
                    loop {
                        match runtime
                            .stream_step(&voice, visible, &mut stream, stopped)
                            .unwrap()
                        {
                            ChatterboxStreamUnit::Rendering => {}
                            ChatterboxStreamUnit::Rendered(samples) => {
                                let ms = started.elapsed().as_secs_f32() * 1000.0;
                                acoustic_ms += ms;
                                window_ms.push(ms);
                                if first_audio_ms.is_none() && !samples.is_empty() {
                                    first_audio_ms =
                                        Some(visible.len() as f32 * t3_per_token + acoustic_ms);
                                }
                                if !audio.is_empty() {
                                    seam_at.push(audio.len());
                                }
                                audio.extend(samples);
                                break;
                            }
                        }
                    }
                    if stream.done() {
                        break;
                    }
                }
            }
            let name = format!(
                "s{index}-stream-{}-{}-{}-{}.wav",
                plan.first_tokens, plan.max_tokens, plan.left_context_tokens, plan.prompt_tokens
            );
            write_wav(&name, &audio);
            let report = seams(&audio, &whole.samples, &seam_at);
            let embedding = embed(&audio);
            let similarity = cosine(&embedding, &whole_embedding);
            if let Some(reference) = &reference_embedding {
                eprintln!(
                    "    VE cosine to the reference: streamed {:.3}, final-only {:.3}",
                    cosine(&embedding, reference),
                    cosine(&whole_embedding, reference)
                );
            }
            let score = detector.detect(&audio, S3GEN_SAMPLE_RATE).unwrap();
            eprintln!(
                "  plan {:?}: {} windows {:?} ms, acoustic {:.0} ms ({:.2}× final-only), first audio {:.0} ms",
                (plan.first_tokens, plan.max_tokens, plan.left_context_tokens, plan.prompt_tokens),
                window_ms.len(),
                window_ms.iter().map(|ms| ms.round() as i32).collect::<Vec<_>>(),
                acoustic_ms,
                acoustic_ms / whole_acoustic,
                first_audio_ms.unwrap_or(f32::NAN),
            );
            eprintln!(
                "    samples {} vs {}, seams: spectral {:.3} (elsewhere p95 {:.3}), step {:.4} (elsewhere p99.9 {:.4}), VE cosine {:.3}, watermark {:.3}",
                audio.len(),
                whole.samples.len(),
                report.spectral_seam,
                report.spectral_elsewhere_p95,
                report.step_seam,
                report.step_elsewhere_p99,
                similarity,
                score
            );
            assert!(audio.len().abs_diff(whole.samples.len()) <= S3GEN_SAMPLE_RATE as usize / 25);
            assert!(audio.iter().all(|sample| sample.is_finite()));
        }
    }
}
