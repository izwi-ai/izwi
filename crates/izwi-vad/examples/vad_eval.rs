//! VAD noise evaluation: scores a synthetic noise corpus and the repository
//! speech fixtures through the no-speech gate rule and realtime endpointing,
//! comparing the legacy rules (any frame >= threshold, no SNR) with the
//! current ones (speech evidence + SNR margin).
//!
//! ```text
//! cargo run --release -p izwi-vad --features test-signals --example vad_eval [-- <data dir>]
//! ```
//!
//! The data dir defaults to the repository `data/` (fox.wav, test.wav).

use std::path::{Path, PathBuf};
use std::time::Instant;

use izwi_vad::test_signals::{
    attenuated, brown_noise, fan_noise, keyboard_clicks, mains_hum, mix_at_snr, pink_noise,
    rms_dbfs, tonal_music, white_noise,
};
use izwi_vad::{
    resample_linear_f32, EndpointConfig, EndpointDetector, EndpointEvent, SpeechEvidence,
    SpeechEvidenceConfig, VadFrame, VadScorer, DEFAULT_SPEECH_THRESHOLD, VAD_SAMPLE_RATE,
};

const SR: usize = VAD_SAMPLE_RATE as usize;

#[derive(Clone, Copy, PartialEq)]
enum Expect {
    Noise,
    /// Noise the model cannot tell from speech; reported, not counted.
    KnownLeak,
    Speech,
}

struct Clip {
    name: String,
    samples: Vec<f32>,
    expect: Expect,
}

fn main() {
    let data_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"));
    let fox = load_fixture(&data_dir.join("fox.wav"));
    let test = load_fixture(&data_dir.join("test.wav"));

    let started = Instant::now();
    let mut scored_ms = 0.0f32;
    gate_report(&corpus(&fox, &test), &mut scored_ms);
    realtime_report(&fox, &mut scored_ms);
    let elapsed = started.elapsed().as_secs_f32();
    println!(
        "\nscored {:.0} s of audio in {:.2} s (RTF {:.5}, includes reporting)",
        scored_ms / 1000.0,
        elapsed,
        elapsed / (scored_ms / 1000.0)
    );
}

fn corpus(fox: &[f32], test: &[f32]) -> Vec<Clip> {
    let len = SR * 10;
    let mut clips = Vec::new();
    let mut noise = |name: String, samples: Vec<f32>, expect| {
        clips.push(Clip {
            name,
            samples,
            expect,
        })
    };
    noise("silence".into(), vec![0.0; len], Expect::Noise);
    for peak in [0.005, 0.05, 0.2] {
        noise(
            format!("white p{peak}"),
            white_noise(len, peak, 4),
            Expect::Noise,
        );
    }
    for peak in [0.02, 0.05, 0.1, 0.3, 0.8] {
        noise(
            format!("pink p{peak}"),
            pink_noise(len, peak, 4),
            Expect::Noise,
        );
    }
    for peak in [0.05, 0.3] {
        noise(
            format!("brown p{peak}"),
            brown_noise(len, peak, 4),
            Expect::Noise,
        );
    }
    noise("hum60 p0.3".into(), mains_hum(len, 0.3), Expect::Noise);
    noise("fan p0.3".into(), fan_noise(len, 0.3, 4), Expect::Noise);
    noise(
        "keyboard p0.5".into(),
        keyboard_clicks(len, 0.5, 4),
        Expect::Noise,
    );
    noise(
        "music p0.3".into(),
        tonal_music(len, 0.3, 4),
        Expect::KnownLeak,
    );

    for (name, speech) in [("fox", fox), ("test", test)] {
        for db in [0.0, -20.0, -40.0] {
            clips.push(Clip {
                name: format!("{name} {db} dB"),
                samples: attenuated(speech, db),
                expect: Expect::Speech,
            });
        }
        let pink = pink_noise(speech.len(), 0.5, 9);
        for snr in [20.0, 10.0, 5.0, 0.0, -5.0] {
            clips.push(Clip {
                name: format!("{name} + pink {snr} dB"),
                samples: mix_at_snr(speech, &pink, snr),
                expect: Expect::Speech,
            });
        }
        for (noise_name, noise) in [
            ("white", white_noise(speech.len(), 0.5, 9)),
            ("fan", fan_noise(speech.len(), 0.5, 9)),
            ("hum", mains_hum(speech.len(), 0.5)),
        ] {
            for snr in [10.0, 0.0] {
                clips.push(Clip {
                    name: format!("{name} + {noise_name} {snr} dB"),
                    samples: mix_at_snr(speech, &noise, snr),
                    expect: Expect::Speech,
                });
            }
        }
    }
    clips
}

fn gate_report(clips: &[Clip], scored_ms: &mut f32) {
    println!("## No-speech gate (pass = reaches the model)\n");
    println!(
        "{:<24} {:>7} {:>6} | {:>6} | {:>9} {:>6} | {:>6}",
        "clip", "dBFS", "peak", "legacy", "evidence", "need", "new"
    );
    let (mut legacy_fa, mut new_fa, mut legacy_fr, mut new_fr, mut noise_n, mut speech_n) =
        (0, 0, 0, 0, 0, 0);
    for clip in clips {
        let frames = score(&clip.samples);
        *scored_ms += frames.len() as f32 * 16.0;
        let legacy = frames
            .iter()
            .any(|frame| frame.score >= DEFAULT_SPEECH_THRESHOLD);
        let mut evidence = SpeechEvidence::new(SpeechEvidenceConfig::default());
        for frame in &frames {
            evidence.push(frame);
        }
        let new = evidence.is_sufficient();
        match clip.expect {
            Expect::Noise => {
                noise_n += 1;
                legacy_fa += legacy as usize;
                new_fa += new as usize;
            }
            Expect::Speech => {
                speech_n += 1;
                legacy_fr += (!legacy) as usize;
                new_fr += (!new) as usize;
            }
            Expect::KnownLeak => {}
        }
        let mark = |pass: bool| match (clip.expect, pass) {
            (Expect::Noise, true) => "PASS!",
            (Expect::Speech, false) => "gate!",
            (_, true) => "pass",
            (_, false) => "gate",
        };
        println!(
            "{:<24} {:>7.1} {:>6.2} | {:>6} | {:>7.0}ms {:>4.0}ms | {:>6}",
            clip.name,
            rms_dbfs(&clip.samples),
            evidence.peak_score(),
            mark(legacy),
            evidence.best_window_ms(),
            evidence.required_ms(),
            mark(new),
        );
    }
    println!(
        "\nnoise reaching the model: legacy {legacy_fa}/{noise_n}, new {new_fa}/{noise_n}; \
         speech gated: legacy {legacy_fr}/{speech_n}, new {new_fr}/{speech_n} \
         ('!' marks a wrong decision; music is a known leak and not counted)"
    );
}

/// Fox spoken 3 s into 12 s of noise, replayed through endpointing with and
/// without the SNR margin.
fn realtime_report(fox: &[f32], scored_ms: &mut f32) {
    println!("\n## Realtime endpointing (fox inside noise)\n");
    let (onset_ms, offset_ms) = speech_span_ms(fox);
    println!(
        "reference speech span in fox: {onset_ms:.0}-{offset_ms:.0} ms; each case places fox at 3000 ms \
         (1 lead-in-free case at 0 ms)\n"
    );
    let len = SR * 12;
    let cases: Vec<(String, Vec<f32>, usize)> = {
        let mut cases = Vec::new();
        for (name, noise) in [
            ("hum", mains_hum(len, 0.5)),
            ("fan", fan_noise(len, 0.5, 7)),
            ("pink", pink_noise(len, 0.5, 7)),
            ("white", white_noise(len, 0.5, 7)),
            ("brown", brown_noise(len, 0.5, 7)),
        ] {
            for snr in [10.0, 0.0] {
                cases.push((
                    format!("{name} {snr} dB"),
                    place(fox, &noise, snr, SR * 3),
                    SR * 3,
                ));
            }
        }
        cases.push((
            "silence".into(),
            place(fox, &vec![0.0; len], 0.0, SR * 3),
            SR * 3,
        ));
        let pink = pink_noise(len, 0.5, 7);
        cases.push((
            "pink 10 dB, speech at 0".into(),
            place(fox, &pink, 10.0, 0),
            0,
        ));
        cases
    };

    println!(
        "{:<24} | {:<44} | {:<44}",
        "case", "legacy (no SNR)", "SNR margin"
    );
    for (name, samples, at) in cases {
        let frames = score(&samples);
        *scored_ms += frames.len() as f32 * 16.0;
        let at_ms = at as f32 * 1000.0 / SR as f32;
        let (onset, offset) = (at_ms + onset_ms, at_ms + offset_ms);
        let legacy = replay(
            &frames,
            EndpointConfig {
                snr_margin_db: 0.0,
                continue_snr_margin_db: 0.0,
                ..EndpointConfig::default()
            },
        );
        let current = replay(&frames, sweep_config());
        println!(
            "{:<24} | {:<44} | {:<44}",
            name,
            describe(&legacy, onset, offset),
            describe(&current, onset, offset)
        );
    }
}

#[derive(Debug)]
struct Turn {
    start_ms: f32,
    end_ms: f32,
    /// Last speech frame before the end, so end clipping can be measured.
    last_speech_ms: f32,
    outcome: &'static str,
}

/// `VAD_EVAL_START_SNR` / `VAD_EVAL_CONTINUE_SNR` / `VAD_EVAL_END_THRESHOLD`
/// override the current endpoint settings for tuning sweeps.
fn sweep_config() -> EndpointConfig {
    let env = |name: &str| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
    };
    let defaults = EndpointConfig::default();
    EndpointConfig {
        snr_margin_db: env("VAD_EVAL_START_SNR").unwrap_or(defaults.snr_margin_db),
        continue_snr_margin_db: env("VAD_EVAL_CONTINUE_SNR")
            .unwrap_or(defaults.continue_snr_margin_db),
        end_threshold: env("VAD_EVAL_END_THRESHOLD").unwrap_or(defaults.end_threshold),
        ..defaults
    }
}

fn replay(frames: &[VadFrame], config: EndpointConfig) -> Vec<Turn> {
    let mut detector = EndpointDetector::new(config);
    let mut turns = Vec::new();
    let (mut start_ms, mut last_speech_ms) = (0.0f32, 0.0f32);
    for frame in frames {
        let decision = detector.process_frame(frame);
        if decision.is_speech {
            last_speech_ms = frame.end_ms;
        }
        for event in decision.events {
            match event {
                EndpointEvent::SpeechStart => start_ms = frame.start_ms,
                EndpointEvent::SpeechEnd(reason) => turns.push(Turn {
                    start_ms,
                    end_ms: frame.end_ms,
                    last_speech_ms,
                    outcome: reason.as_str(),
                }),
                EndpointEvent::NoiseRejected => turns.push(Turn {
                    start_ms,
                    end_ms: frame.end_ms,
                    last_speech_ms,
                    outcome: "noise",
                }),
            }
        }
    }
    if detector.is_active() {
        turns.push(Turn {
            start_ms,
            end_ms: frames.last().map_or(0.0, |frame| frame.end_ms),
            last_speech_ms,
            outcome: "open",
        });
    }
    turns
}

/// One line per case: the turn count, then onset delay and end clipping of
/// the turn covering the speech, and any wrong turns.
fn describe(turns: &[Turn], onset_ms: f32, offset_ms: f32) -> String {
    let speech_turn = turns.iter().find(|turn| {
        turn.outcome != "noise" && turn.start_ms < offset_ms && turn.end_ms > onset_ms
    });
    let outside = |turn: &&Turn| turn.start_ms < onset_ms - 200.0 || turn.start_ms > offset_ms;
    // A false turn reaches ASR; a rejected blip only flashes `user_speech_start`.
    let wrong = turns
        .iter()
        .filter(outside)
        .filter(|turn| turn.outcome != "noise")
        .count();
    let blips = turns
        .iter()
        .filter(outside)
        .filter(|turn| turn.outcome == "noise")
        .count();
    match speech_turn {
        Some(turn) => {
            format!(
                "{} turn(s) start{:+.0} clip {:.0} {}{}",
                turns.len(),
                turn.start_ms - onset_ms,
                (offset_ms - turn.last_speech_ms).max(0.0),
                turn.outcome,
                if wrong > 0 {
                    format!(" FALSE x{wrong}")
                } else {
                    String::new()
                },
            ) + &if blips > 0 {
                format!(" blips x{blips}")
            } else {
                String::new()
            }
        }
        None => format!("{} turn(s), speech MISSED", turns.len()),
    }
}

/// `speech` mixed into `noise` at `at`, with the noise level set by the SNR
/// against the speech.
fn place(speech: &[f32], noise: &[f32], snr_db: f32, at: usize) -> Vec<f32> {
    let noise_rms = rms(noise);
    let gain = if noise_rms > 0.0 {
        rms(speech) / noise_rms * 10f32.powf(-snr_db / 20.0)
    } else {
        0.0
    };
    let mut out: Vec<f32> = noise.iter().map(|sample| sample * gain).collect();
    for (idx, sample) in speech.iter().enumerate() {
        if let Some(slot) = out.get_mut(at + idx) {
            *slot = (*slot + sample).clamp(-1.0, 1.0);
        }
    }
    out
}

/// First and last frame of clean speech the model scores as speech.
fn speech_span_ms(speech: &[f32]) -> (f32, f32) {
    let frames = score(speech);
    let mut speech_frames = frames
        .iter()
        .filter(|frame| frame.score >= DEFAULT_SPEECH_THRESHOLD);
    let onset = speech_frames
        .clone()
        .next()
        .map_or(0.0, |frame| frame.start_ms);
    let offset = speech_frames.next_back().map_or(0.0, |frame| frame.end_ms);
    (onset, offset)
}

fn score(samples: &[f32]) -> Vec<VadFrame> {
    VadScorer::score_complete_f32(samples, VAD_SAMPLE_RATE).expect("16 kHz scoring")
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

/// A 16-bit speech fixture as 16 kHz mono f32.
fn load_fixture(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = reader
        .samples::<i16>()
        .map(|sample| sample.expect("16-bit sample") as f32 / 32768.0)
        .collect();
    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    resample_linear_f32(&mono, spec.sample_rate, VAD_SAMPLE_RATE).expect("resample fixture")
}
