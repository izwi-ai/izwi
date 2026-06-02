//! Lightweight decoded-audio inspection for request boundary validation.

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioInspection {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub duration_secs: f32,
    pub peak: f32,
    pub rms: f32,
    pub clipped_samples: usize,
}

pub fn inspect_audio_bytes(audio_bytes: &[u8]) -> Result<AudioInspection> {
    let (samples, sample_rate) = crate::runtime::audio_io::decode_audio_bytes(audio_bytes)?;
    Ok(AudioInspection::from_mono_samples(&samples, sample_rate))
}

impl AudioInspection {
    pub fn from_mono_samples(samples: &[f32], sample_rate: u32) -> Self {
        let sample_count = samples.len();
        let duration_secs = if sample_rate == 0 {
            0.0
        } else {
            sample_count as f32 / sample_rate as f32
        };
        let mut peak = 0.0f32;
        let mut sum_squares = 0.0f64;
        let mut clipped_samples = 0usize;

        for sample in samples {
            let abs = sample.abs();
            peak = peak.max(abs);
            sum_squares += (*sample as f64) * (*sample as f64);
            if abs >= 1.0 {
                clipped_samples += 1;
            }
        }

        let rms = if sample_count == 0 {
            0.0
        } else {
            (sum_squares / sample_count as f64).sqrt() as f32
        };

        Self {
            sample_rate,
            sample_count,
            duration_secs,
            peak,
            rms,
            clipped_samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioEncoder, AudioFormat};

    #[test]
    fn inspection_calculates_basic_signal_metadata() {
        let inspection = AudioInspection::from_mono_samples(&[0.0, 0.5, -1.0, 1.0], 4);

        assert_eq!(inspection.sample_rate, 4);
        assert_eq!(inspection.sample_count, 4);
        assert_eq!(inspection.duration_secs, 1.0);
        assert_eq!(inspection.peak, 1.0);
        assert_eq!(inspection.clipped_samples, 2);
        assert!((inspection.rms - 0.75).abs() < 1e-6);
    }

    #[test]
    fn inspection_decodes_wav_bytes() {
        let samples = [0.0, 0.25, -0.25, 0.0];
        let wav = AudioEncoder::new(16_000, 1)
            .encode(&samples, AudioFormat::Wav)
            .expect("wav should encode");

        let inspection = inspect_audio_bytes(&wav).expect("wav should inspect");

        assert_eq!(inspection.sample_rate, 16_000);
        assert_eq!(inspection.sample_count, samples.len());
        assert!((inspection.duration_secs - samples.len() as f32 / 16_000.0).abs() < 1e-6);
        assert!(inspection.peak > 0.24);
    }
}
