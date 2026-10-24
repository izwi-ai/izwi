//! Audio preprocessing for Granite Speech 5.0: 80-bin HTK log-mel at 16 kHz,
//! adjacent frames stacked in pairs (160 features at 50 frames per second),
//! as transformers' `GraniteSpeechFeatureExtractor` computes them.

use crate::audio::{resample_mono_high_quality, MelConfig, MelNorm, MelScale, MelSpectrogram};
use crate::error::{Error, Result};

use super::config::GraniteSpeech5ProcessorConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeech5AudioFeatures {
    pub sample_rate: u32,
    pub audio_seconds: f32,
    pub mel_frames: usize,
    pub mel_bins: usize,
    pub encoder_frames: usize,
    pub encoder_dim: usize,
    /// `[encoder_frames][encoder_dim]` stacked log-mel features.
    pub input_features: Vec<Vec<f32>>,
}

pub struct GraniteSpeech5Preprocessor {
    config: GraniteSpeech5ProcessorConfig,
    mel: MelSpectrogram,
}

impl GraniteSpeech5Preprocessor {
    pub fn new(config: GraniteSpeech5ProcessorConfig) -> Result<Self> {
        let melspec = &config.audio_processor.melspec_kwargs;
        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: config.sample_rate() as usize,
            n_fft: melspec.n_fft,
            win_length: Some(melspec.win_length),
            hop_length: melspec.hop_length,
            n_mels: melspec.n_mels,
            f_min: 0.0,
            f_max: (config.sample_rate() / 2) as f32,
            normalize: true,
            mel_scale: MelScale::Htk,
            mel_norm: MelNorm::None,
        })?;
        Ok(Self { config, mel })
    }

    pub fn config(&self) -> &GraniteSpeech5ProcessorConfig {
        &self.config
    }

    pub fn prepare(&self, audio: &[f32], sample_rate: u32) -> Result<GraniteSpeech5AudioFeatures> {
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 audio sample rate must be greater than zero".to_string(),
            ));
        }
        if audio.is_empty() {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 audio input cannot be empty".to_string(),
            ));
        }
        let target_rate = self.config.sample_rate();
        let mut samples = resample_mono_high_quality(audio, sample_rate, target_rate)?;
        for sample in &mut samples {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        let mut log_mel = self.mel.compute(&samples)?;
        let mel_frames = log_mel.len();
        let mel_bins = log_mel.first().map(Vec::len).unwrap_or(0);
        // The feature extractor drops a trailing odd frame before stacking.
        if !log_mel.len().is_multiple_of(2) {
            log_mel.pop();
        }
        let input_features = stack_adjacent_mel_frames(&log_mel);
        Ok(GraniteSpeech5AudioFeatures {
            sample_rate: target_rate,
            audio_seconds: samples.len() as f32 / target_rate as f32,
            mel_frames,
            mel_bins,
            encoder_frames: input_features.len(),
            encoder_dim: input_features.first().map(Vec::len).unwrap_or(0),
            input_features,
        })
    }
}

fn stack_adjacent_mel_frames(log_mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
    log_mel
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[first, second]| {
            let mut frame = Vec::with_capacity(first.len() + second.len());
            frame.extend_from_slice(first);
            frame.extend_from_slice(second);
            frame
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::granite_speech5::asr::config::tests::fixture_dir;

    #[test]
    fn prepares_stacked_16khz_log_mel_from_the_published_settings() {
        let config = GraniteSpeech5ProcessorConfig::load(&fixture_dir()).unwrap();
        let preprocessor = GraniteSpeech5Preprocessor::new(config).unwrap();
        assert_eq!(preprocessor.mel.config().mel_scale, MelScale::Htk);
        assert_eq!(preprocessor.mel.config().mel_norm, MelNorm::None);

        // One second at 8 kHz resamples to 16 kHz first.
        let audio = (0..8_000)
            .map(|index| (index as f32 * 0.05).sin() * 0.3)
            .collect::<Vec<_>>();
        let features = preprocessor.prepare(&audio, 8_000).unwrap();
        assert_eq!(features.sample_rate, 16_000);
        assert!((features.audio_seconds - 1.0).abs() < 1e-3);
        assert_eq!(features.mel_bins, 80);
        assert_eq!(features.encoder_dim, 160);
        assert_eq!(features.encoder_frames, features.mel_frames / 2);
        assert!(features
            .input_features
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn stacking_pairs_frames_and_drops_an_odd_tail() {
        let frames = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert_eq!(
            stack_adjacent_mel_frames(&frames),
            vec![vec![1.0, 2.0, 3.0, 4.0]]
        );
    }

    #[test]
    fn rejects_empty_audio_and_a_zero_sample_rate() {
        let config = GraniteSpeech5ProcessorConfig::load(&fixture_dir()).unwrap();
        let preprocessor = GraniteSpeech5Preprocessor::new(config).unwrap();
        assert!(preprocessor.prepare(&[], 16_000).is_err());
        assert!(preprocessor.prepare(&[0.0; 160], 0).is_err());
    }
}
