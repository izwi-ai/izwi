//! Deterministic synthetic noise for VAD tests and evaluation.
//!
//! Every generator is seeded and allocation-only, so a test that fails on a
//! noise clip fails the same way on every machine. Levels are set by peak
//! amplitude, as the ASR matrix fixtures (`sox ... vol`) are.

use std::f32::consts::TAU;

/// Deterministic xorshift samples, uniform in `[-1, 1)`.
pub fn uniform_noise(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// White noise scaled to `peak`.
pub fn white_noise(len: usize, peak: f32, seed: u64) -> Vec<f32> {
    scaled_to_peak(uniform_noise(len, seed), peak)
}

/// Pink noise (Paul Kellet's filter over white noise), scaled to `peak`.
pub fn pink_noise(len: usize, peak: f32, seed: u64) -> Vec<f32> {
    let mut b = [0.0f32; 7];
    let samples = uniform_noise(len, seed)
        .into_iter()
        .map(|white| {
            b[0] = 0.99886 * b[0] + white * 0.055_517_9;
            b[1] = 0.99332 * b[1] + white * 0.075_075_9;
            b[2] = 0.96900 * b[2] + white * 0.153_852;
            b[3] = 0.86650 * b[3] + white * 0.310_485_6;
            b[4] = 0.55000 * b[4] + white * 0.532_952_2;
            b[5] = -0.7616 * b[5] - white * 0.016_898;
            let pink = b.iter().sum::<f32>() + white * 0.5362;
            b[6] = white * 0.115_926;
            pink
        })
        .collect();
    scaled_to_peak(samples, peak)
}

/// Brown (red) noise: leaky-integrated white noise with its DC drift removed,
/// scaled to `peak`.
pub fn brown_noise(len: usize, peak: f32, seed: u64) -> Vec<f32> {
    let mut level = 0.0f32;
    let integrated = uniform_noise(len, seed)
        .into_iter()
        .map(|white| {
            level = 0.998 * level + white;
            level
        })
        .collect::<Vec<_>>();
    scaled_to_peak(high_pass(&integrated, 20.0, SAMPLE_RATE), peak)
}

/// Mains hum: 60 Hz and six harmonics at falling level, scaled to `peak`.
pub fn mains_hum(len: usize, peak: f32) -> Vec<f32> {
    let samples = (0..len)
        .map(|idx| {
            let t = idx as f32 / SAMPLE_RATE;
            (1..=7)
                .map(|harmonic| (TAU * 60.0 * harmonic as f32 * t).sin() / harmonic as f32)
                .sum()
        })
        .collect();
    scaled_to_peak(samples, peak)
}

/// Fan noise: low-passed white noise (800 Hz) with a 120 Hz motor tone,
/// scaled to `peak`.
pub fn fan_noise(len: usize, peak: f32, seed: u64) -> Vec<f32> {
    let rumble = low_pass(&uniform_noise(len, seed), 800.0, SAMPLE_RATE);
    let rumble_peak = peak_abs(&rumble).max(f32::EPSILON);
    let samples = rumble
        .iter()
        .enumerate()
        .map(|(idx, sample)| {
            let t = idx as f32 / SAMPLE_RATE;
            sample / rumble_peak + 0.3 * (TAU * 120.0 * t).sin()
        })
        .collect();
    scaled_to_peak(samples, peak)
}

/// Keyboard-like clicks: six decaying noise bursts per second, scaled to
/// `peak`.
pub fn keyboard_clicks(len: usize, peak: f32, seed: u64) -> Vec<f32> {
    const BURST: usize = 400;
    let mut samples = vec![0.0f32; len];
    if len <= BURST {
        return samples;
    }
    let positions = uniform_noise(len / (SAMPLE_RATE as usize / 6) + 1, seed ^ 0x9E37);
    let bursts = uniform_noise(BURST * positions.len(), seed);
    for (burst_idx, position) in positions.iter().enumerate() {
        let start = (((position + 1.0) * 0.5) * (len - BURST) as f32) as usize;
        for offset in 0..BURST {
            let decay = (-(offset as f32) / 40.0).exp();
            samples[start + offset] += bursts[burst_idx * BURST + offset] * decay;
        }
    }
    scaled_to_peak(samples, peak)
}

/// Tonal "music": one decaying harmonic note per second on a chromatic scale,
/// scaled to `peak`.
pub fn tonal_music(len: usize, peak: f32, seed: u64) -> Vec<f32> {
    let notes = uniform_noise(len / SAMPLE_RATE as usize + 1, seed);
    let samples = (0..len)
        .map(|idx| {
            let second = idx / SAMPLE_RATE as usize;
            let t = (idx % SAMPLE_RATE as usize) as f32 / SAMPLE_RATE;
            let semitone = ((notes[second] + 1.0) * 6.0).floor();
            let f0 = 220.0 * 2f32.powf(semitone / 12.0);
            let envelope = (-t / 0.4).exp();
            envelope
                * (1..=6)
                    .map(|harmonic| (TAU * f0 * harmonic as f32 * t).sin() / harmonic as f32)
                    .sum::<f32>()
        })
        .collect();
    scaled_to_peak(samples, peak)
}

/// `speech + noise`, with the noise scaled so the speech-to-noise RMS ratio
/// is `snr_db`. The noise repeats if it is shorter than the speech.
pub fn mix_at_snr(speech: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    if speech.is_empty() || noise.is_empty() {
        return speech.to_vec();
    }
    let noise_rms = rms(noise).max(f32::EPSILON);
    let gain = rms(speech) / noise_rms * 10f32.powf(-snr_db / 20.0);
    speech
        .iter()
        .zip(noise.iter().cycle())
        .map(|(speech, noise)| (speech + noise * gain).clamp(-1.0, 1.0))
        .collect()
}

/// `samples` scaled by `db` decibels.
pub fn attenuated(samples: &[f32], db: f32) -> Vec<f32> {
    let gain = 10f32.powf(db / 20.0);
    samples.iter().map(|sample| sample * gain).collect()
}

/// RMS level of `samples` in dBFS.
pub fn rms_dbfs(samples: &[f32]) -> f32 {
    20.0 * rms(samples).max(1e-9).log10()
}

const SAMPLE_RATE: f32 = crate::VAD_SAMPLE_RATE as f32;

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

fn peak_abs(samples: &[f32]) -> f32 {
    samples
        .iter()
        .fold(0.0f32, |max, sample| max.max(sample.abs()))
}

fn scaled_to_peak(mut samples: Vec<f32>, peak: f32) -> Vec<f32> {
    let max = peak_abs(&samples);
    if max > 0.0 {
        for sample in &mut samples {
            *sample *= peak / max;
        }
    }
    samples
}

/// Second-order Butterworth low-pass (RBJ biquad).
fn low_pass(samples: &[f32], cutoff_hz: f32, sample_rate: f32) -> Vec<f32> {
    let w0 = TAU * cutoff_hz / sample_rate;
    let alpha = w0.sin() / std::f32::consts::SQRT_2;
    let cos = w0.cos();
    let b = [(1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0];
    biquad(samples, b, [1.0 + alpha, -2.0 * cos, 1.0 - alpha])
}

/// Second-order Butterworth high-pass (RBJ biquad).
fn high_pass(samples: &[f32], cutoff_hz: f32, sample_rate: f32) -> Vec<f32> {
    let w0 = TAU * cutoff_hz / sample_rate;
    let alpha = w0.sin() / std::f32::consts::SQRT_2;
    let cos = w0.cos();
    let b = [(1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0];
    biquad(samples, b, [1.0 + alpha, -2.0 * cos, 1.0 - alpha])
}

fn biquad(samples: &[f32], b: [f32; 3], a: [f32; 3]) -> Vec<f32> {
    let (b0, b1, b2) = (b[0] / a[0], b[1] / a[0], b[2] / a[0]);
    let (a1, a2) = (a[1] / a[0], a[2] / a[0]);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    samples
        .iter()
        .map(|&x| {
            let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            (x2, x1, y2, y1) = (x1, x, y1, y);
            y
        })
        .collect()
}
