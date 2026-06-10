//! Audio preprocessing for Granite Speech ASR.

use crate::audio::{resample_mono_high_quality, MelConfig, MelSpectrogram};
use crate::error::{Error, Result};
use crate::models::architectures::granite_speech::asr::config::GraniteSpeechProcessorConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechAudioFeatures {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub audio_seconds: f32,
    pub mel_frames: usize,
    pub mel_bins: usize,
    pub projected_frames_hint: usize,
    pub log_mel: Vec<Vec<f32>>,
}

pub struct GraniteSpeechPreprocessor {
    config: GraniteSpeechProcessorConfig,
    mel: MelSpectrogram,
}

impl GraniteSpeechPreprocessor {
    pub fn new(config: GraniteSpeechProcessorConfig) -> Result<Self> {
        let mel_cfg = MelConfig {
            sample_rate: config.sample_rate() as usize,
            n_fft: config.audio_processor.melspec_kwargs.n_fft,
            hop_length: config.audio_processor.melspec_kwargs.hop_length,
            n_mels: config.audio_processor.melspec_kwargs.n_mels,
            f_min: 0.0,
            f_max: (config.sample_rate() / 2) as f32,
            normalize: true,
        };
        let mel = MelSpectrogram::new(mel_cfg)?;
        Ok(Self { config, mel })
    }

    pub fn config(&self) -> &GraniteSpeechProcessorConfig {
        &self.config
    }

    pub fn prepare(&self, audio: &[f32], sample_rate: u32) -> Result<GraniteSpeechAudioFeatures> {
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Granite Speech audio sample rate must be greater than zero".to_string(),
            ));
        }
        if audio.is_empty() {
            return Err(Error::InvalidInput(
                "Granite Speech audio input cannot be empty".to_string(),
            ));
        }

        let target_rate = self.config.sample_rate();
        let mut samples = resample_mono_high_quality(audio, sample_rate, target_rate)?;
        for sample in &mut samples {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }

        let log_mel = self.mel.compute(&samples)?;
        let mel_frames = log_mel.len();
        let mel_bins = log_mel.first().map(|frame| frame.len()).unwrap_or(0);
        let downsample = self.config.audio_processor.projector_downsample_rate.max(1);
        let projected_frames_hint = mel_frames.saturating_add(downsample - 1) / downsample;

        Ok(GraniteSpeechAudioFeatures {
            audio_seconds: samples.len() as f32 / target_rate as f32,
            samples,
            sample_rate: target_rate,
            mel_frames,
            mel_bins,
            projected_frames_hint,
            log_mel,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::granite_speech::asr::config::{
        GraniteSpeechAudioProcessorConfig, GraniteSpeechMelSpecConfig,
    };

    fn test_processor() -> GraniteSpeechProcessorConfig {
        GraniteSpeechProcessorConfig {
            audio_processor: GraniteSpeechAudioProcessorConfig {
                feature_extractor_type: Some("GraniteSpeechFeatureExtractor".to_string()),
                melspec_kwargs: GraniteSpeechMelSpecConfig {
                    hop_length: 160,
                    n_fft: 512,
                    n_mels: 80,
                    sample_rate: 16_000,
                    win_length: 400,
                },
                projector_downsample_rate: 5,
                projector_window_size: 15,
                sampling_rate: 16_000,
            },
            audio_token: "<|audio|>".to_string(),
            processor_class: Some("GraniteSpeechProcessor".to_string()),
        }
    }

    #[test]
    fn resample_linear_preserves_identity_rate() {
        let audio = vec![0.0, 0.25, -0.25, 0.5];
        assert_eq!(
            resample_mono_high_quality(&audio, 16_000, 16_000).unwrap(),
            audio
        );
    }

    #[test]
    fn preprocessor_prepares_16khz_log_mel_features() {
        let preprocessor = GraniteSpeechPreprocessor::new(test_processor()).unwrap();
        let audio = vec![0.0f32; 16_000];
        let features = preprocessor.prepare(&audio, 16_000).unwrap();
        assert_eq!(features.sample_rate, 16_000);
        assert_eq!(features.mel_bins, 80);
        assert!(features.mel_frames > 0);
        assert_eq!(
            features.projected_frames_hint,
            features.mel_frames.saturating_add(4) / 5
        );
    }
}
