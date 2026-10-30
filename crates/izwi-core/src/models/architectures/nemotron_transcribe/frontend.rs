//! The NeMo log-mel front end and per-feature normalization, on the host.
//!
//! This is the family's own copy of NeMo's `FilterbankFeatures` as the
//! perception module runs it at inference (no dither):
//!
//! 1. `Tv = L // 160` valid frames; the STFT yields `Tv + 1`.
//! 2. Preemphasis `y[0] = x[0]`, `y[n] = x[n] - 0.97 x[n-1]`.
//! 3. Centre zero padding of `n_fft / 2` (constant, not reflect), a
//!    symmetric Hann(400) centred in 512, `|X|` taken as a square root and
//!    squared again, in F32.
//! 4. `ln(fb . power + 2^-24)`; frames `t >= Tv` are 0.0 (not silence).
//! 5. The result is BF16, the perception module's dtype.
//!
//! The ASR branch normalizes per feature over the whole recording with
//! torch's BF16 rounding points (`normalize_batch`): BF16 reference-centred
//! mean, BF16 squared deviations, an F32 unbiased variance, `+ 1e-5` on the
//! standard deviation and an F32 quotient, cast to BF16 at the encoder. The
//! diarizer reads the raw BF16 log-mel.
//!
//! The checkpoint's stored filterbank and window are used, after checking
//! them against a recomputation (librosa's Slaney filterbank, symmetric
//! Hann), so a checkpoint whose front end changed fails to load.

use std::sync::Arc;

use half::bf16;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::config::NemotronTranscribeFrontendConfig;
use crate::error::{Error, Result};

/// Recomputed entries of the stored filterbank or window may differ from it
/// by one BF16 ulp (F64 against NumPy's rounding); any larger difference
/// fails the load.
const MAX_RECOMPUTE_ULPS: u16 = 1;

/// The front end with its checked filterbank and window.
#[derive(Clone)]
pub struct NemotronMelFrontend {
    config: NemotronTranscribeFrontendConfig,
    /// The symmetric Hann window, `win_length` values.
    window: Vec<f32>,
    /// `[n_mels, n_fft / 2 + 1]`, row-major.
    filterbank: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl std::fmt::Debug for NemotronMelFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NemotronMelFrontend")
            .field("config", &self.config)
            .finish()
    }
}

/// One recording's log-mel, kept BF16 and time-major, with the ASR branch's
/// normalization statistics.
#[derive(Debug, Clone)]
pub struct NemotronMel {
    n_mels: usize,
    /// `Tv = samples // hop`.
    valid_frames: usize,
    /// `[Tv + 1, n_mels]`; the last frame is 0.0.
    raw: Vec<bf16>,
    mean: Vec<bf16>,
    std: Vec<f32>,
}

impl NemotronMelFrontend {
    /// Builds the front end from the checkpoint's stored `window` and
    /// `filterbank` (`[n_mels, n_fft / 2 + 1]`), after checking both against
    /// their recomputation.
    pub fn new(
        config: NemotronTranscribeFrontendConfig,
        window: Vec<f32>,
        filterbank: Vec<f32>,
    ) -> Result<Self> {
        let bins = config.n_fft / 2 + 1;
        if window.len() != config.win_length || filterbank.len() != config.n_mels * bins {
            return Err(Error::ModelLoadError(format!(
                "Nemotron 3.5 Transcribe front end: stored window {} and filterbank {} do not match [{}] and [{}, {bins}]",
                window.len(),
                filterbank.len(),
                config.win_length,
                config.n_mels
            )));
        }
        let expected_window = symmetric_hann(config.win_length);
        let expected_fb = slaney_filterbank(
            config.sample_rate as f64,
            config.n_fft,
            config.n_mels,
            0.0,
            config.sample_rate as f64 / 2.0,
        );
        check_stored("window", &window, &expected_window)?;
        check_stored("filterbank", &filterbank, &expected_fb)?;
        let fft = FftPlanner::<f32>::new().plan_fft_forward(config.n_fft);
        Ok(Self {
            config,
            window,
            filterbank,
            fft,
        })
    }

    pub fn config(&self) -> &NemotronTranscribeFrontendConfig {
        &self.config
    }

    /// The log-mel of `samples` (mono, 16 kHz) and its normalization
    /// statistics.
    pub fn compute(&self, samples: &[f32]) -> Result<NemotronMel> {
        let c = &self.config;
        let valid_frames = samples.len() / c.hop_length;
        if valid_frames < 2 {
            return Err(Error::InvalidInput(format!(
                "Nemotron 3.5 Transcribe needs at least two mel frames ({} samples); got {}",
                2 * c.hop_length,
                samples.len()
            )));
        }
        let frames = valid_frames + 1;
        let pad = c.n_fft / 2;
        let bins = c.n_fft / 2 + 1;
        // Preemphasis, then the centre zero padding.
        let mut padded = vec![0f32; samples.len() + 2 * pad];
        padded[pad] = samples[0];
        for n in 1..samples.len() {
            padded[pad + n] = samples[n] - c.preemphasis * samples[n - 1];
        }
        let window_offset = (c.n_fft - c.win_length) / 2;
        let mut raw = vec![bf16::ZERO; frames * c.n_mels];
        use rayon::prelude::*;
        raw.par_chunks_mut(c.n_mels * 64)
            .enumerate()
            .for_each(|(block, out)| {
                let mut buffer = vec![Complex::new(0f32, 0f32); c.n_fft];
                let mut power = vec![0f32; bins];
                for (local, row) in out.chunks_mut(c.n_mels).enumerate() {
                    let t = block * 64 + local;
                    if t >= valid_frames {
                        // Frames past `Tv` stay exactly 0.0.
                        continue;
                    }
                    let start = t * c.hop_length;
                    for (i, slot) in buffer.iter_mut().enumerate() {
                        let w = if i >= window_offset && i < window_offset + c.win_length {
                            self.window[i - window_offset]
                        } else {
                            0.0
                        };
                        *slot = Complex::new(padded[start + i] * w, 0.0);
                    }
                    self.fft.process(&mut buffer);
                    for (p, value) in power.iter_mut().zip(&buffer[..bins]) {
                        let magnitude = (value.re * value.re + value.im * value.im).sqrt();
                        *p = magnitude * magnitude;
                    }
                    for (m, out) in row.iter_mut().enumerate() {
                        let weights = &self.filterbank[m * bins..(m + 1) * bins];
                        let mel = weights
                            .iter()
                            .zip(&power)
                            .fold(0f32, |acc, (w, p)| w.mul_add(*p, acc));
                        *out = bf16::from_f32((mel + c.log_guard).ln());
                    }
                }
            });
        let (mean, std) = per_feature_statistics(&raw, c.n_mels, valid_frames);
        Ok(NemotronMel {
            n_mels: c.n_mels,
            valid_frames,
            raw,
            mean,
            std,
        })
    }
}

impl NemotronMel {
    /// `Tv`, the valid mel frames.
    pub fn valid_frames(&self) -> usize {
        self.valid_frames
    }

    /// `Tv + 1`, the frames the encoders window over.
    pub fn frames(&self) -> usize {
        self.valid_frames + 1
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Bytes held on the host.
    pub fn host_bytes(&self) -> usize {
        self.raw.len() * 2 + self.mean.len() * 2 + self.std.len() * 4
    }

    /// Raw BF16 log-mel frames `start..end`, time-major, as F32 (the
    /// diarizer's input).
    pub fn raw_window(&self, start: usize, end: usize) -> Vec<f32> {
        self.raw[start * self.n_mels..end * self.n_mels]
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }

    /// The ASR branch's normalized frames `start..end`, time-major, rounded
    /// to BF16 as they enter the encoder; frames `t >= Tv` are 0.
    pub fn normalized_window(&self, start: usize, end: usize) -> Vec<f32> {
        let mut out = vec![0f32; (end - start) * self.n_mels];
        for t in start..end.min(self.valid_frames) {
            for m in 0..self.n_mels {
                let centred =
                    bf16::from_f32(self.raw[t * self.n_mels + m].to_f32() - self.mean[m].to_f32());
                out[(t - start) * self.n_mels + m] =
                    bf16::from_f32(centred.to_f32() / self.std[m]).to_f32();
            }
        }
        out
    }

    #[cfg(test)]
    pub(crate) fn statistics(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.mean.iter().map(|v| v.to_f32()).collect(),
            self.std.clone(),
        )
    }
}

/// NeMo's `normalize_batch(per_feature)` statistics over the first `valid`
/// frames of time-major BF16 `x`, with torch's BF16 rounding points.
fn per_feature_statistics(x: &[bf16], n_mels: usize, valid: usize) -> (Vec<bf16>, Vec<f32>) {
    let mut mean = vec![bf16::ZERO; n_mels];
    let mut std = vec![0f32; n_mels];
    for m in 0..n_mels {
        let reference = x[m].to_f32();
        let centred_sum = (0..valid)
            .map(|t| bf16::from_f32(x[t * n_mels + m].to_f32() - reference).to_f32())
            .sum::<f32>();
        let quotient = bf16::from_f32(bf16::from_f32(centred_sum).to_f32() / valid as f32);
        mean[m] = bf16::from_f32(reference + quotient.to_f32());
        let squares = (0..valid)
            .map(|t| {
                let d = bf16::from_f32(x[t * n_mels + m].to_f32() - mean[m].to_f32()).to_f32();
                bf16::from_f32(d * d).to_f32()
            })
            .sum::<f32>();
        let variance = bf16::from_f32(squares).to_f32() / (valid as f32 - 1.0);
        let deviation = variance.sqrt();
        std[m] = if deviation.is_nan() { 0.0 } else { deviation } + 1e-5;
    }
    (mean, std)
}

/// The symmetric Hann window (`torch.hann_window(n, periodic=False)`).
pub fn symmetric_hann(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()) as f32
        })
        .collect()
}

/// librosa's Slaney mel filterbank (`htk=False`, `norm="slaney"`), computed
/// in F64 and returned as F32 `[n_mels, n_fft / 2 + 1]`.
pub fn slaney_filterbank(
    sample_rate: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    fn hz_to_mel(f: f64) -> f64 {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let logstep = 6.4f64.ln() / 27.0;
        if f >= min_log_hz {
            min_log_hz / f_sp + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    }
    fn mel_to_hz(m: f64) -> f64 {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4f64.ln() / 27.0;
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
    let (lo, hi) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let mel_f = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect::<Vec<_>>();
    let mut weights = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let (below, above) = (mel_f[m + 1] - mel_f[m], mel_f[m + 2] - mel_f[m + 1]);
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (b, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_f[m]) / below;
            let upper = (mel_f[m + 2] - freq) / above;
            weights[m * bins + b] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    weights
}

/// Fails when a stored front-end table is not the BF16 rounding of its
/// recomputation (beyond [`MAX_RECOMPUTE_ULPS`]).
fn check_stored(what: &str, stored: &[f32], expected: &[f32]) -> Result<()> {
    let mut worst = 0u16;
    for (index, (&s, &e)) in stored.iter().zip(expected).enumerate() {
        let (s, e) = (bf16::from_f32(s), bf16::from_f32(e));
        let ulps = if s == e {
            0
        } else if s.is_sign_negative() != e.is_sign_negative()
            && s.to_f32() != 0.0
            && e.to_f32() != 0.0
        {
            u16::MAX
        } else {
            s.to_bits().abs_diff(e.to_bits())
        };
        if ulps > MAX_RECOMPUTE_ULPS {
            return Err(Error::ModelLoadError(format!(
                "Nemotron 3.5 Transcribe front end: stored {what}[{index}] = {} is not the expected {} (a symmetric Hann window and a Slaney filterbank)",
                s.to_f32(),
                e.to_f32()
            )));
        }
        worst = worst.max(ulps);
    }
    let _ = worst;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn published_tables() -> (Vec<f32>, Vec<f32>) {
        let read = |bytes: &[u8]| {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b).to_f32())
                .collect::<Vec<_>>()
        };
        (
            read(include_bytes!(
                "../../../../tests/fixtures/nemotron_transcribe/featurizer_window.bf16"
            )),
            read(include_bytes!(
                "../../../../tests/fixtures/nemotron_transcribe/featurizer_fb.bf16"
            )),
        )
    }

    fn frontend() -> NemotronMelFrontend {
        let (window, fb) = published_tables();
        NemotronMelFrontend::new(
            super::super::config::tests::published().frontend,
            window,
            fb,
        )
        .unwrap()
    }

    #[test]
    fn the_published_window_and_filterbank_are_symmetric_hann_and_slaney() {
        let (window, fb) = published_tables();
        let exact = |stored: &[f32], expected: &[f32]| {
            stored
                .iter()
                .zip(expected)
                .filter(|(s, e)| bf16::from_f32(**s) == bf16::from_f32(**e))
                .count()
        };
        assert_eq!(exact(&window, &symmetric_hann(400)), 400);
        let expected = slaney_filterbank(16_000.0, 512, 128, 0.0, 8_000.0);
        assert_eq!(exact(&fb, &expected), 128 * 257);
        // RED: a periodic window is refused.
        let periodic = (0..400)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 400.0).cos()) as f32)
            .collect::<Vec<_>>();
        assert!(NemotronMelFrontend::new(
            super::super::config::tests::published().frontend,
            periodic,
            fb.clone()
        )
        .is_err());
    }

    #[test]
    fn frames_padding_and_the_zero_frame_follow_nemo() {
        let frontend = frontend();
        let samples = crate::kernels::numeric_check::xorshift_values(16_000 + 37, 3, 0.3);
        let mel = frontend.compute(&samples).unwrap();
        assert_eq!(mel.valid_frames(), 100);
        assert_eq!(mel.frames(), 101);
        // The frame past Tv is exactly 0.0, not log(guard).
        assert!(mel.raw_window(100, 101).iter().all(|v| *v == 0.0));
        assert!(mel.raw_window(99, 100).iter().all(|v| *v != 0.0));
        // Normalized valid frames have mean ~0 and unbiased std ~1 per bin.
        let norm = mel.normalized_window(0, 101);
        for m in [0, 64, 127] {
            let column = (0..100)
                .map(|t| f64::from(norm[t * 128 + m]))
                .collect::<Vec<_>>();
            let mean = column.iter().sum::<f64>() / 100.0;
            let var = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 99.0;
            assert!(mean.abs() < 0.02, "bin {m} mean {mean}");
            assert!(
                (var.sqrt() - 1.0).abs() < 0.02,
                "bin {m} std {}",
                var.sqrt()
            );
        }
        assert!(norm[100 * 128..].iter().all(|v| *v == 0.0));
        assert!(frontend.compute(&samples[..319]).is_err());
    }

    #[test]
    fn the_spectrum_uses_zero_padding_and_preemphasis() {
        // A DC signal: preemphasis leaves 0.03 * DC after the first sample, and
        // zero padding (not reflect) makes the first frame differ from the
        // middle ones.
        let frontend = frontend();
        let samples = vec![0.5f32; 16_000];
        let mel = frontend.compute(&samples).unwrap();
        let first = mel.raw_window(0, 1);
        let middle = mel.raw_window(50, 51);
        assert!(first.iter().zip(&middle).any(|(a, b)| (a - b).abs() > 0.1));
        // A direct F64 computation of one middle frame's lowest bin.
        let t = 50;
        let pad = 256;
        let mut padded = vec![0f64; samples.len() + 512];
        padded[pad] = 0.5;
        for n in 1..samples.len() {
            padded[pad + n] = 0.5 - 0.97 * 0.5;
        }
        let window = symmetric_hann(400);
        let mut power0 = 0f64;
        let mut re = 0f64;
        for i in 0..512 {
            let w = if (56..456).contains(&i) {
                f64::from(window[i - 56])
            } else {
                0.0
            };
            re += padded[t * 160 + i] * w;
        }
        power0 += re * re;
        let fb = slaney_filterbank(16_000.0, 512, 128, 0.0, 8_000.0);
        // Mel bin 0 weights only FFT bins 0 and 1; bin 1 of a DC frame is the
        // window's first harmonic, small next to bin 0.
        let mut re1 = 0f64;
        let mut im1 = 0f64;
        for i in 0..512 {
            let w = if (56..456).contains(&i) {
                f64::from(window[i - 56])
            } else {
                0.0
            };
            let angle = -2.0 * std::f64::consts::PI * i as f64 / 512.0;
            re1 += padded[t * 160 + i] * w * angle.cos();
            im1 += padded[t * 160 + i] * w * angle.sin();
        }
        let mel0 = f64::from(fb[0]) * power0 + f64::from(fb[1]) * (re1 * re1 + im1 * im1);
        let expected = (mel0 + 2f64.powi(-24)).ln();
        assert!((f64::from(middle[0]) - expected).abs() < 0.02 * expected.abs().max(1.0));
    }
}
