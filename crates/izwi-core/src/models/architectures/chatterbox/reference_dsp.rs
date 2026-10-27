//! Host signal processing for preparing a voice from reference audio, each
//! routine matching the upstream function it replaces:
//!
//! * [`sinc_resample`]: torchaudio's `Resample` (`sinc_interp_hann`, width
//!   6, rolloff 0.99), exactly, kernel built in F64. Upstream loads the
//!   reference with librosa's SoXR resampler, which has no exact Rust
//!   counterpart; izwi uses this one for every rate change.
//! * [`trim_silence`]: librosa's `effects.trim` (top 20 dB, 2048/512 RMS
//!   frames) before the voice encoder.
//! * [`log_mel_s3tokenizer`]: the S3 tokenizer's 128-bin log10 mel.
//! * [`mel_s3gen_reference`]: S3Gen's 80-bin natural-log magnitude mel at
//!   24 kHz.
//! * [`mel_voice_encoder`]: the voice encoder's 40-bin power mel.
//! * [`kaldi_fbank`]: torchaudio's `compliance.kaldi.fbank` (80 bins, Povey
//!   window, pre-emphasis 0.97), mean-normalised as CAMPPlus expects.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// torchaudio `Resample(orig, new)` of one mono signal.
pub fn sinc_resample(input: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new || input.is_empty() {
        return input.to_vec();
    }
    let g = gcd(u64::from(orig), u64::from(new));
    let orig = (u64::from(orig) / g) as usize;
    let new = (u64::from(new) / g) as usize;
    const WIDTH_ZEROS: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let base = orig.min(new) as f64 * ROLLOFF;
    let width = (WIDTH_ZEROS * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    // kernel[phase][tap], in F64 then F32 as torchaudio rounds it.
    let mut kernel = vec![0f32; new * taps];
    for phase in 0..new {
        for tap in 0..taps {
            let idx = (tap as f64 - width as f64) / orig as f64;
            let t = (-(phase as f64) / new as f64 + idx) * base;
            let t = t.clamp(-WIDTH_ZEROS, WIDTH_ZEROS);
            let window = (t * std::f64::consts::PI / WIDTH_ZEROS / 2.0).cos().powi(2);
            let t = t * std::f64::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[phase * taps + tap] = (sinc * window * base / orig as f64) as f32;
        }
    }
    // Pad `width` before and `width + orig` after, then a strided
    // convolution: output block `b` reads padded[b * orig ..][..taps].
    let length = input.len();
    let mut padded = vec![0f32; width + length + width + orig];
    padded[width..width + length].copy_from_slice(input);
    let blocks = (padded.len() - taps) / orig + 1;
    let target = (new as u128 * length as u128).div_ceil(orig as u128) as usize;
    let mut output = Vec::with_capacity(blocks * new);
    for block in 0..blocks {
        let window = &padded[block * orig..block * orig + taps];
        for phase in 0..new {
            let row = &kernel[phase * taps..(phase + 1) * taps];
            let mut acc = 0f32;
            for (x, k) in window.iter().zip(row) {
                acc += x * k;
            }
            output.push(acc);
        }
    }
    output.truncate(target);
    output
}

/// librosa `effects.trim(top_db=20)`: drop leading and trailing frames whose
/// RMS is more than 20 dB below the loudest frame.
pub fn trim_silence(input: &[f32], top_db: f32) -> &[f32] {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    if input.is_empty() {
        return input;
    }
    // librosa.feature.rms(center=True, pad_mode="constant").
    let pad = FRAME / 2;
    let frames = 1 + input.len() / HOP;
    let sample = |i: isize| -> f64 {
        if i < 0 || i as usize >= input.len() {
            0.0
        } else {
            f64::from(input[i as usize])
        }
    };
    let rms = (0..frames)
        .map(|frame| {
            let start = (frame * HOP) as isize - pad as isize;
            let power = (0..FRAME as isize)
                .map(|offset| sample(start + offset).powi(2))
                .sum::<f64>()
                / FRAME as f64;
            power.sqrt()
        })
        .collect::<Vec<_>>();
    let peak = rms.iter().copied().fold(0f64, f64::max);
    if peak <= 0.0 {
        return &input[..0];
    }
    // amplitude_to_db(rms, ref=max): 20 log10(max(rms, 1e-5) / peak), clipped
    // at top_db 80 below the peak; frames above −top_db are not silent.
    let threshold = -f64::from(top_db);
    let loud = |value: f64| 20.0 * (value.max(1e-5) / peak).log10() > threshold;
    let Some(first) = rms.iter().position(|&value| loud(value)) else {
        return &input[..0];
    };
    let last = rms.iter().rposition(|&value| loud(value)).unwrap_or(first);
    let start = first * HOP;
    let end = ((last + 1) * HOP).min(input.len());
    &input[start..end]
}

/// Periodic Hann (`torch.hann_window`, `scipy.signal.get_window("hann")`).
pub(crate) fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// librosa's Slaney mel filterbank (`htk=False`, `norm="slaney"`), in F64,
/// row-major [mels, bins].
pub fn slaney_mel_filters(
    sample_rate: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f64> {
    fn hz_to_mel(f: f64) -> f64 {
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = (6.4f64).ln() / 27.0;
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    }
    fn mel_to_hz(m: f64) -> f64 {
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = (6.4f64).ln() / 27.0;
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    }
    let bins = n_fft / 2 + 1;
    let fft_freqs = (0..bins)
        .map(|i| i as f64 * sample_rate / n_fft as f64)
        .collect::<Vec<_>>();
    let (mel_min, mel_max) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let mel_f = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect::<Vec<_>>();
    let mut weights = vec![0f64; n_mels * bins];
    for m in 0..n_mels {
        let fdiff_lo = mel_f[m + 1] - mel_f[m];
        let fdiff_hi = mel_f[m + 2] - mel_f[m + 1];
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (b, &freq) in fft_freqs.iter().enumerate() {
            let lower = -(mel_f[m] - freq) / fdiff_lo;
            let upper = (mel_f[m + 2] - freq) / fdiff_hi;
            weights[m * bins + b] = lower.min(upper).max(0.0) * enorm;
        }
    }
    weights
}

/// |STFT|^power frames of `signal` (already padded as the caller needs),
/// Hann window of `n_fft`, hop `hop`, no centring: [frames, bins].
fn power_frames(signal: &[f32], n_fft: usize, hop: usize, power: i32) -> Vec<Vec<f64>> {
    if signal.len() < n_fft {
        return Vec::new();
    }
    let window = hann_periodic(n_fft);
    let fft = FftPlanner::<f64>::new().plan_fft_forward(n_fft);
    let frames = 1 + (signal.len() - n_fft) / hop;
    let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
    (0..frames)
        .map(|frame| {
            for (i, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex::new(f64::from(signal[frame * hop + i]) * window[i], 0.0);
            }
            fft.process(&mut buffer);
            buffer[..n_fft / 2 + 1]
                .iter()
                .map(|c| match power {
                    1 => (c.norm_sqr() + 1e-9).sqrt(),
                    _ => c.norm_sqr(),
                })
                .collect()
        })
        .collect()
}

/// Reflect padding of `pad` samples each side (numpy/torch "reflect").
pub(crate) fn reflect_pad(input: &[f32], pad: usize) -> Vec<f32> {
    let n = input.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(input[i.min(n - 1)]);
    }
    out.extend_from_slice(input);
    for i in 0..pad {
        out.push(input[n.saturating_sub(2 + i)]);
    }
    out
}

fn apply_filters(frames: &[Vec<f64>], filters: &[f64], n_mels: usize) -> Vec<Vec<f64>> {
    let bins = filters.len() / n_mels;
    frames
        .iter()
        .map(|spectrum| {
            (0..n_mels)
                .map(|m| {
                    filters[m * bins..(m + 1) * bins]
                        .iter()
                        .zip(spectrum)
                        .map(|(w, p)| w * p)
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// The S3 tokenizer's log-mel at 16 kHz: centred STFT (n_fft 400, hop 160,
/// reflect), last frame dropped, power, the checkpoint's 128 mel filters
/// ([128, 201] row-major), log10 with a 1e-10 floor, floored 8 below the
/// max, then `(x + 4) / 4`. Returns [128, frames] row-major.
pub fn log_mel_s3tokenizer(samples: &[f32], filters: &[f32]) -> (Vec<f32>, usize) {
    const N_FFT: usize = 400;
    const HOP: usize = 160;
    const MELS: usize = 128;
    if samples.len() < 2 {
        return (Vec::new(), 0);
    }
    let padded = reflect_pad(samples, N_FFT / 2);
    let mut frames = power_frames(&padded, N_FFT, HOP, 2);
    frames.pop();
    let filters = filters.iter().map(|&w| f64::from(w)).collect::<Vec<_>>();
    let mel = apply_filters(&frames, &filters, MELS);
    let count = mel.len();
    let log = mel
        .iter()
        .flat_map(|row| row.iter().map(|&v| (v.max(1e-10) as f32).log10()))
        .collect::<Vec<f32>>();
    let max = log.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0f32; MELS * count];
    for (t, chunk) in log.chunks(MELS).enumerate() {
        for (m, &v) in chunk.iter().enumerate() {
            out[m * count + t] = (v.max(max - 8.0) + 4.0) / 4.0;
        }
    }
    (out, count)
}

/// S3Gen's reference mel at 24 kHz: reflect-pad 720 each side, n_fft 1920,
/// hop 480, no centring, magnitude `sqrt(re² + im² + 1e-9)`, 80 Slaney mels
/// up to 8 kHz, natural log with a 1e-5 floor. Returns [frames, 80].
pub fn mel_s3gen_reference(samples: &[f32]) -> (Vec<f32>, usize) {
    const N_FFT: usize = 1920;
    const HOP: usize = 480;
    const MELS: usize = 80;
    let padded = reflect_pad(samples, (N_FFT - HOP) / 2);
    let frames = power_frames(&padded, N_FFT, HOP, 1);
    let filters = slaney_mel_filters(24_000.0, N_FFT, MELS, 0.0, 8_000.0)
        .into_iter()
        .map(|w| f64::from(w as f32))
        .collect::<Vec<_>>();
    let mel = apply_filters(&frames, &filters, MELS);
    let count = mel.len();
    let out = mel
        .into_iter()
        .flat_map(|row| row.into_iter().map(|v| (v as f32).max(1e-5).ln()))
        .collect();
    (out, count)
}

/// The voice encoder's mel at 16 kHz: centred librosa STFT (n_fft 400, hop
/// 160, reflect), power, 40 Slaney mels up to 8 kHz. Returns [frames, 40].
pub fn mel_voice_encoder(samples: &[f32]) -> (Vec<f32>, usize) {
    const N_FFT: usize = 400;
    const HOP: usize = 160;
    const MELS: usize = 40;
    if samples.len() < 2 {
        return (Vec::new(), 0);
    }
    let padded = reflect_pad(samples, N_FFT / 2);
    let frames = power_frames(&padded, N_FFT, HOP, 2);
    let filters = slaney_mel_filters(16_000.0, N_FFT, MELS, 0.0, 8_000.0)
        .into_iter()
        .map(|w| f64::from(w as f32))
        .collect::<Vec<_>>();
    let mel = apply_filters(&frames, &filters, MELS);
    let count = mel.len();
    (
        mel.into_iter()
            .flat_map(|row| row.into_iter().map(|v| v as f32))
            .collect(),
        count,
    )
}

/// torchaudio `compliance.kaldi.fbank(num_mel_bins=80)` at 16 kHz with its
/// defaults (25 ms frames, 10 ms shift, snip edges, DC removal,
/// pre-emphasis 0.97, Povey window, 512-point FFT, power, Kaldi mel bins
/// from 20 Hz to Nyquist, natural log floored at F32 epsilon), then each
/// bin's mean over time subtracted (CAMPPlus' `extract_feature`).
/// Returns [frames, 80].
pub fn kaldi_fbank(samples: &[f32]) -> (Vec<f32>, usize) {
    const FRAME: usize = 400;
    const SHIFT: usize = 160;
    const PADDED: usize = 512;
    const BINS: usize = 80;
    const PREEMPH: f32 = 0.97;
    if samples.len() < FRAME {
        return (Vec::new(), 0);
    }
    let frames = 1 + (samples.len() - FRAME) / SHIFT;
    // Povey window: symmetric Hann to the power 0.85.
    let window = (0..FRAME)
        .map(|i| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (FRAME - 1) as f64).cos();
            hann.powf(0.85) as f32
        })
        .collect::<Vec<_>>();
    // Kaldi mel banks on the 257 positive bins (the Nyquist column is zero).
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let nyquist = 8_000.0;
    let (low, high) = (mel(20.0), mel(nyquist));
    let delta = (high - low) / (BINS + 1) as f64;
    let fft_bin_width = 16_000.0 / PADDED as f64;
    let mut banks = vec![0f32; BINS * (PADDED / 2 + 1)];
    for b in 0..BINS {
        let left = low + b as f64 * delta;
        let center = left + delta;
        let right = center + delta;
        for k in 0..PADDED / 2 {
            let m = mel(fft_bin_width * k as f64);
            let up = (m - left) / (center - left);
            let down = (right - m) / (right - center);
            banks[b * (PADDED / 2 + 1) + k] = up.min(down).max(0.0) as f32;
        }
    }
    let fft = FftPlanner::<f32>::new().plan_fft_forward(PADDED);
    let mut buffer = vec![Complex::new(0f32, 0f32); PADDED];
    let mut features = vec![0f32; frames * BINS];
    for frame in 0..frames {
        let raw = &samples[frame * SHIFT..frame * SHIFT + FRAME];
        let mean = raw.iter().sum::<f32>() / FRAME as f32;
        let centred = raw.iter().map(|&x| x - mean).collect::<Vec<_>>();
        for i in 0..PADDED {
            buffer[i] = if i < FRAME {
                let previous = centred[i.saturating_sub(1)];
                Complex::new((centred[i] - PREEMPH * previous) * window[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.process(&mut buffer);
        for b in 0..BINS {
            let row = &banks[b * (PADDED / 2 + 1)..(b + 1) * (PADDED / 2 + 1)];
            let energy = row
                .iter()
                .zip(&buffer)
                .map(|(w, c)| w * c.norm_sqr())
                .sum::<f32>();
            features[frame * BINS + b] = energy.max(f32::EPSILON).ln();
        }
    }
    for b in 0..BINS {
        let mean = (0..frames).map(|t| features[t * BINS + b]).sum::<f32>() / frames as f32;
        for t in 0..frames {
            features[t * BINS + b] -= mean;
        }
    }
    (features, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_length_ratio_and_a_low_tone() {
        let input = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24_000.0).sin())
            .collect::<Vec<_>>();
        let out = sinc_resample(&input, 24_000, 16_000);
        assert_eq!(out.len(), 16_000);
        // Away from the edges the tone survives at 16 kHz.
        for (i, value) in out.iter().enumerate().skip(100).take(15_800) {
            let expected = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
            assert!(
                (value - expected).abs() < 2e-3,
                "{i}: {value} vs {expected}"
            );
        }
        assert_eq!(sinc_resample(&input, 24_000, 24_000), input);
    }

    #[test]
    fn trim_drops_leading_and_trailing_silence() {
        let mut signal = vec![0f32; 8_192];
        signal.extend((0..8_192).map(|i| (i as f32 * 0.05).sin() * 0.5));
        signal.extend(vec![0f32; 8_192]);
        let trimmed = trim_silence(&signal, 20.0);
        assert!(trimmed.len() < signal.len());
        assert!(trimmed.len() >= 8_192);
    }

    #[test]
    fn slaney_filters_match_librosa_shape_and_normalisation() {
        let filters = slaney_mel_filters(16_000.0, 400, 40, 0.0, 8_000.0);
        assert_eq!(filters.len(), 40 * 201);
        // Every filter has weight, none below zero.
        for m in 0..40 {
            let row = &filters[m * 201..(m + 1) * 201];
            assert!(row.iter().all(|&w| w >= 0.0));
            assert!(row.iter().any(|&w| w > 0.0), "filter {m} is empty");
        }
    }

    #[test]
    fn kaldi_fbank_frames_and_mean_normalises() {
        let samples = (0..16_000)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect::<Vec<_>>();
        let (features, frames) = kaldi_fbank(&samples);
        assert_eq!(frames, 1 + (16_000 - 400) / 160);
        for b in 0..80 {
            let mean = (0..frames).map(|t| features[t * 80 + b]).sum::<f32>() / frames as f32;
            assert!(mean.abs() < 1e-3, "bin {b} mean {mean}");
        }
    }
}
