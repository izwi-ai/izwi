//! Rumble's audio front end on the host: resample to 16 kHz, pad very short
//! clips, and compute Whisper-style log-mel features.

use std::time::Instant;

use crate::audio::{MelConfig, MelNorm, MelScale, MelSpectrogram};
use crate::error::{Error, Result};

use super::audio_tower::get_cnn_output_lengths;
use super::config::RumblePreprocessorConfig;

pub const RUMBLE_SAMPLE_RATE: u32 = 16_000;
/// Most audio one invocation covers: the feature extractor's 30 s window.
/// Longer audio is chunked by the generic ASR route.
pub const RUMBLE_MAX_AUDIO_SECONDS: f32 = 30.0;
/// Audio tokens per second: 13 per 100 mel frames at 100 frames per second.
pub const RUMBLE_AUDIO_TOKENS_PER_SECOND: f32 = 13.0;

/// One request's log-mel features, frame-major (`[frames, n_mels]`).
#[derive(Debug, Clone)]
pub struct RumbleAudioFeatures {
    pub flat: Vec<f32>,
    pub mel_frames: usize,
    pub n_mels: usize,
    /// Audio tokens the tower emits for these frames.
    pub audio_tokens: usize,
    pub input_sample_rate: u32,
    pub input_samples: usize,
    pub resampled_samples: usize,
    /// Zero samples appended to reach the minimum clip length.
    pub padded_samples: usize,
    pub audio_seconds: f32,
    pub resample_ms: f64,
    pub mel_ms: f64,
}

pub struct RumbleFrontend {
    mel: MelSpectrogram,
    min_samples: usize,
    max_frames: usize,
}

impl RumbleFrontend {
    pub fn new(config: &RumblePreprocessorConfig) -> Result<Self> {
        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: RUMBLE_SAMPLE_RATE as usize,
            n_fft: config.n_fft,
            win_length: None,
            hop_length: config.hop_length,
            n_mels: config.feature_size,
            f_min: 0.0,
            f_max: 8_000.0,
            // Normalized in `log_mel`, after the last frame is dropped.
            normalize: false,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        })?;
        let max_frames = (RUMBLE_MAX_AUDIO_SECONDS * RUMBLE_SAMPLE_RATE as f32) as usize
            / config.hop_length.max(1);
        Ok(Self {
            mel,
            min_samples: config.min_length,
            max_frames,
        })
    }

    pub fn n_mels(&self) -> usize {
        self.mel.config().n_mels
    }

    pub fn hop_length(&self) -> usize {
        self.mel.config().hop_length
    }

    /// Mel frames one invocation may hold.
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }

    /// Mel frames for `resampled_samples` of 16 kHz audio: one per hop, after
    /// the minimum-length pad.
    pub fn mel_frames_for(&self, resampled_samples: usize) -> usize {
        resampled_samples.max(self.min_samples) / self.hop_length().max(1)
    }

    /// Features for one invocation of audio.
    pub fn prepare(&self, audio: &[f32], sample_rate: u32) -> Result<RumbleAudioFeatures> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".into()));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput("Audio sample rate is zero".into()));
        }
        let resample_started = Instant::now();
        let mut samples = if sample_rate == RUMBLE_SAMPLE_RATE {
            audio.to_vec()
        } else {
            crate::audio::resample_mono_high_quality(audio, sample_rate, RUMBLE_SAMPLE_RATE)?
        };
        let resample_ms = resample_started.elapsed().as_secs_f64() * 1000.0;
        let resampled_samples = samples.len();
        let audio_seconds = resampled_samples as f32 / RUMBLE_SAMPLE_RATE as f32;
        // Upstream zero-pads clips under 0.5 s and counts the padding as
        // audio (it does not shorten the attention mask).
        let padded_samples = self.min_samples.saturating_sub(resampled_samples);
        if padded_samples > 0 {
            samples.resize(self.min_samples, 0.0);
        }

        let mel_started = Instant::now();
        let (flat, mel_frames) = log_mel(&self.mel, &samples)?;
        let mel_ms = mel_started.elapsed().as_secs_f64() * 1000.0;
        if mel_frames == 0 {
            return Err(Error::InvalidInput("Empty audio input".into()));
        }
        if mel_frames > self.max_frames {
            return Err(Error::InvalidInput(format!(
                "Rumble transcribes up to {RUMBLE_MAX_AUDIO_SECONDS:.0}s per invocation, got {audio_seconds:.1}s"
            )));
        }
        let audio_tokens = get_cnn_output_lengths(&[mel_frames])[0];
        Ok(RumbleAudioFeatures {
            flat,
            mel_frames,
            n_mels: self.n_mels(),
            audio_tokens,
            input_sample_rate: sample_rate,
            input_samples: audio.len(),
            resampled_samples,
            padded_samples,
            audio_seconds,
            resample_ms,
            mel_ms,
        })
    }
}

/// Whisper log-mel features as the published feature extractor computes
/// them: the centre-padded STFT without its last frame, clamped to
/// `max - 8` and scaled, the max taken over the kept frames.
fn log_mel(mel: &MelSpectrogram, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    debug_assert!(!mel.config().normalize);
    let (mut flat, frames) = mel.compute_flat(samples)?;
    let kept = frames.saturating_sub(1);
    flat.truncate(kept * mel.config().n_mels);
    MelSpectrogram::whisper_normalize_flat(&mut flat);
    Ok((flat, kept))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontend() -> RumbleFrontend {
        RumbleFrontend::new(
            &RumblePreprocessorConfig::from_json(include_str!(
                "../../../../tests/fixtures/rumble/preprocessor_config.json"
            ))
            .unwrap(),
        )
        .unwrap()
    }

    fn tone(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| (index as f32 * 0.05).sin() * 0.2)
            .collect()
    }

    #[test]
    fn one_frame_per_hop_and_thirteen_tokens_per_second() {
        let frontend = frontend();
        let features = frontend.prepare(&tone(48_000), 16_000).unwrap();
        assert_eq!(features.mel_frames, 300);
        assert_eq!(features.audio_tokens, 39);
        assert_eq!(features.flat.len(), 300 * 128);
        assert_eq!(features.padded_samples, 0);
        assert_eq!(frontend.mel_frames_for(48_000), 300);
        // A partial chunk: 250 frames -> 2 * 13 + ceil-halved 50 frames.
        let features = frontend.prepare(&tone(40_000), 16_000).unwrap();
        assert_eq!(features.mel_frames, 250);
        assert_eq!(features.audio_tokens, 26 + 7);
    }

    #[test]
    fn a_clip_under_half_a_second_is_padded_to_it() {
        let frontend = frontend();
        let features = frontend.prepare(&tone(1_600), 16_000).unwrap();
        assert_eq!(features.padded_samples, 6_400);
        assert_eq!(features.mel_frames, 50);
        assert_eq!(features.audio_tokens, 7);
        assert_eq!(frontend.mel_frames_for(1_600), 50);
    }

    #[test]
    fn audio_past_the_window_is_refused_and_other_rates_resample() {
        let frontend = frontend();
        let error = frontend
            .prepare(&tone(16_000 * 31), 16_000)
            .unwrap_err()
            .to_string();
        assert!(error.contains("up to 30s"), "{error}");
        let features = frontend.prepare(&tone(48_000), 48_000).unwrap();
        assert_eq!(features.resampled_samples, 16_000);
        assert_eq!(features.mel_frames, 100);
        assert_eq!(features.audio_tokens, 13);
        assert!(frontend.prepare(&[], 16_000).is_err());
    }
}
