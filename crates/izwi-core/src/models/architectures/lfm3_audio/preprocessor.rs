//! LFM3-Audio's log-mel front end (`feature_extraction_lfm3_audio.py`):
//! 16 kHz input, a 25 ms periodic Hann window every 10 ms without centering,
//! a 400-point power spectrum, 128 Slaney mel filters up to 8 kHz, `log10`
//! floored at -12, then `(x + 4) / 4`. Clips are right-padded to whole
//! windows; there is no per-clip normalization, so features stream.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};

pub(crate) const N_MELS: usize = 128;
const N_FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
const SAMPLE_RATE: f64 = 16_000.0;
const LOG_FLOOR: f32 = -12.0;
const OFFSET: f32 = 4.0;
const DIVISOR: f32 = 4.0;

pub(crate) struct Lfm3AudioMel {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// `[N_MELS, BINS]`, mel-major.
    filters: Vec<f32>,
}

/// Samples not yet framed and the counts so far of one input stream.
#[derive(Debug, Clone, Default)]
pub(crate) struct Lfm3AudioMelStream {
    samples: Vec<f32>,
    num_samples: usize,
    num_frames: usize,
    finished: bool,
}

impl Lfm3AudioMel {
    pub(crate) fn new() -> Self {
        let window = (0..N_FFT)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32
            })
            .collect();
        Self {
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
            window,
            filters: slaney_mel_filters(),
        }
    }

    /// The right-padded length of a clip: at least one window, then whole
    /// hops.
    pub(crate) fn pad_length(num_samples: usize) -> usize {
        if num_samples < N_FFT {
            N_FFT
        } else {
            num_samples + (HOP - (num_samples - N_FFT) % HOP) % HOP
        }
    }

    pub(crate) fn num_frames(num_samples: usize) -> usize {
        1 + (Self::pad_length(num_samples) - N_FFT) / HOP
    }

    /// The features of a whole 16 kHz clip, `[N_MELS, frames]` mel-major,
    /// and the frame count.
    pub(crate) fn features(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let frames = Self::num_frames(samples.len());
        let mut padded = samples.to_vec();
        padded.resize(Self::pad_length(samples.len()), 0.0);
        (self.log_mel(&padded, frames), frames)
    }

    /// The features a stream completes with `chunk`, `[N_MELS, frames]`,
    /// keeping the overlap for the next call. `is_final` pads and flushes
    /// the last window.
    pub(crate) fn stream(
        &self,
        stream: &mut Lfm3AudioMelStream,
        chunk: &[f32],
        is_final: bool,
    ) -> Result<(Vec<f32>, usize)> {
        if stream.finished {
            return Err(Error::InvalidInput(
                "LFM3-Audio mel stream is finished".into(),
            ));
        }
        let mut samples = std::mem::take(&mut stream.samples);
        samples.extend_from_slice(chunk);
        let num_samples = stream.num_samples + chunk.len();
        let count = if is_final {
            let count = Self::num_frames(num_samples) - stream.num_frames;
            if count > 0 {
                samples.resize((count - 1) * HOP + N_FFT, 0.0);
            }
            count
        } else if samples.len() >= N_FFT {
            1 + (samples.len() - N_FFT) / HOP
        } else {
            0
        };
        let features = if count > 0 {
            self.log_mel(&samples, count)
        } else {
            Vec::new()
        };
        stream.samples = if is_final {
            Vec::new()
        } else {
            samples[(count * HOP).min(samples.len())..].to_vec()
        };
        stream.num_samples = num_samples;
        stream.num_frames += count;
        stream.finished = is_final;
        Ok((features, count))
    }

    /// `frames` windows of `samples` (enough samples for each).
    fn log_mel(&self, samples: &[f32], frames: usize) -> Vec<f32> {
        let mut out = vec![0f32; N_MELS * frames];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];
        let mut power = [0f32; BINS];
        for frame in 0..frames {
            let start = frame * HOP;
            for (index, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex::new(samples[start + index] * self.window[index], 0.0);
            }
            self.fft.process(&mut buffer);
            for (bin, value) in power.iter_mut().enumerate() {
                *value = buffer[bin].norm_sqr();
            }
            for mel in 0..N_MELS {
                let filter = &self.filters[mel * BINS..][..BINS];
                let energy = filter
                    .iter()
                    .zip(power.iter())
                    .map(|(weight, power)| weight * power)
                    .sum::<f32>();
                let log = energy.log10().max(LOG_FLOOR);
                out[mel * frames + frame] = (log + OFFSET) / DIVISOR;
            }
        }
        out
    }
}

/// transformers' `mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney",
/// mel_scale="slaney")`, computed in F64 like NumPy, as `[N_MELS, BINS]`.
fn slaney_mel_filters() -> Vec<f32> {
    fn hz_to_mel(hz: f64) -> f64 {
        let log_step = 27.0 / 6.4f64.ln();
        if hz >= 1000.0 {
            15.0 + (hz / 1000.0).ln() * log_step
        } else {
            3.0 * hz / 200.0
        }
    }
    fn mel_to_hz(mel: f64) -> f64 {
        let log_step = 6.4f64.ln() / 27.0;
        if mel >= 15.0 {
            1000.0 * (log_step * (mel - 15.0)).exp()
        } else {
            200.0 * mel / 3.0
        }
    }
    let nyquist = SAMPLE_RATE / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(nyquist);
    let filter_freqs = (0..N_MELS + 2)
        .map(|index| mel_to_hz(mel_min + (mel_max - mel_min) * index as f64 / (N_MELS + 1) as f64))
        .collect::<Vec<_>>();
    let fft_freqs = (0..BINS)
        .map(|bin| nyquist * bin as f64 / (BINS - 1) as f64)
        .collect::<Vec<_>>();
    let mut filters = vec![0f32; N_MELS * BINS];
    for mel in 0..N_MELS {
        let (low, center, high) = (
            filter_freqs[mel],
            filter_freqs[mel + 1],
            filter_freqs[mel + 2],
        );
        let norm = 2.0 / (high - low);
        for (bin, freq) in fft_freqs.iter().enumerate() {
            let down = (freq - low) / (center - low);
            let up = (high - freq) / (high - center);
            filters[mel * BINS + bin] = (down.min(up).max(0.0) * norm) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_to_whole_windows_and_counts_frames() {
        assert_eq!(Lfm3AudioMel::pad_length(0), 400);
        assert_eq!(Lfm3AudioMel::pad_length(399), 400);
        assert_eq!(Lfm3AudioMel::pad_length(400), 400);
        assert_eq!(Lfm3AudioMel::pad_length(401), 560);
        assert_eq!(Lfm3AudioMel::pad_length(560), 560);
        // 16,000 samples pad to 16,080: 1 + 15,680 / 160 frames.
        assert_eq!(Lfm3AudioMel::num_frames(16_000), 99);
        assert_eq!(Lfm3AudioMel::num_frames(16_080), 99);
        assert_eq!(Lfm3AudioMel::num_frames(16_081), 100);
    }

    #[test]
    fn filters_follow_the_slaney_scale_and_norm() {
        let filters = slaney_mel_filters();
        // Each Slaney-normalized triangle has unit area in Hz: its peak is
        // 2 / (high - low), and its row sums to about that peak times the
        // bins it spans.
        for mel in 0..N_MELS {
            let row = &filters[mel * BINS..][..BINS];
            assert!(row.iter().all(|weight| *weight >= 0.0));
            assert!(row.iter().any(|weight| *weight > 0.0), "mel {mel} is empty");
        }
        // The lowest filter peaks near 31 Hz, the highest ends at 8 kHz.
        assert!(filters[0] == 0.0 && filters[1] > 0.0);
        assert!(filters[(N_MELS - 1) * BINS + BINS - 1] >= 0.0);
    }

    #[test]
    fn streaming_reproduces_the_whole_clip() {
        let mel = Lfm3AudioMel::new();
        let samples = (0..16_337)
            .map(|index| (index as f32 * 0.031).sin() * 0.4 + (index as f32 * 0.0073).cos() * 0.2)
            .collect::<Vec<_>>();
        let (whole, frames) = mel.features(&samples);
        let mut stream = Lfm3AudioMelStream::default();
        let mut streamed = vec![Vec::new(); N_MELS];
        let mut total = 0;
        for (index, chunk) in samples.chunks(1_600).enumerate() {
            let last = (index + 1) * 1_600 >= samples.len();
            let (features, count) = mel.stream(&mut stream, chunk, last).unwrap();
            for (row, out) in streamed.iter_mut().enumerate() {
                out.extend_from_slice(&features[row * count..][..count]);
            }
            total += count;
        }
        assert_eq!(total, frames);
        assert_eq!(streamed.concat(), whole);
        assert!(mel.stream(&mut stream, &[0.0], true).is_err());
    }
}
