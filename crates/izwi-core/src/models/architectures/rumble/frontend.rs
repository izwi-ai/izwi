//! Rumble's audio front end on the host: resample to 16 kHz, pad very short
//! clips, and compute Whisper-style log-mel features as the published
//! feature extractor does.
//!
//! The reference pads every clip with silence to its 30 s window before the
//! STFT and marks `ceil(samples / hop)` frames valid. So the last frames of
//! a clip see silence after it (not the reflected signal), and the audio
//! tower's last 100-frame chunk is filled with the mel of that silence. This
//! front end pads the waveform to the next whole chunk the same way and
//! hands the tower those frames beside the valid count.

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

/// One request's log-mel features, frame-major (`[padded_frames, n_mels]`).
#[derive(Debug, Clone)]
pub struct RumbleAudioFeatures {
    pub flat: Vec<f32>,
    /// Frames of audio: the tower's valid length.
    pub mel_frames: usize,
    /// Frames in `flat`: `mel_frames` rounded up to a whole conv chunk, the
    /// extra ones the mel of the silence the waveform was padded with.
    pub padded_frames: usize,
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
    /// Mel frames per conv chunk of the audio tower (`2 * n_window`).
    chunk_frames: usize,
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
            chunk_frames: (config.n_window * 2).max(1),
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

    /// Valid mel frames for `resampled_samples` of 16 kHz audio: one per
    /// started hop, after the minimum-length pad.
    pub fn mel_frames_for(&self, resampled_samples: usize) -> usize {
        resampled_samples
            .max(self.min_samples)
            .div_ceil(self.hop_length().max(1))
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

        let hop = self.hop_length().max(1);
        let mel_frames = samples.len().div_ceil(hop);
        if mel_frames > self.max_frames {
            return Err(Error::InvalidInput(format!(
                "Rumble transcribes up to {RUMBLE_MAX_AUDIO_SECONDS:.0}s per invocation, got {audio_seconds:.1}s"
            )));
        }
        // Silence to the next whole chunk, and one FFT window more so the
        // STFT's reflection at the end reflects silence; never past the
        // reference's 30 s window, whose own end is reflected.
        let padded_frames = mel_frames.div_ceil(self.chunk_frames) * self.chunk_frames;
        let window_samples = self.max_frames * hop;
        let padded_len = (padded_frames * hop + self.mel.config().n_fft).min(window_samples);
        samples.resize(padded_len.max(samples.len()), 0.0);

        let mel_started = Instant::now();
        let (flat, computed_frames) = log_mel(&self.mel, &samples, padded_frames)?;
        let mel_ms = mel_started.elapsed().as_secs_f64() * 1000.0;
        if computed_frames != padded_frames {
            return Err(Error::InferenceError(format!(
                "Rumble mel geometry drifted: expected {padded_frames} frames, computed {computed_frames}"
            )));
        }
        let audio_tokens = get_cnn_output_lengths(&[mel_frames])[0];
        Ok(RumbleAudioFeatures {
            flat,
            mel_frames,
            padded_frames,
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
/// `max - 8` and scaled. The max is taken over every computed frame (the
/// reference takes it over its whole padded window, whose frames past these
/// are silence and never the max); the first `keep` frames are returned.
fn log_mel(mel: &MelSpectrogram, samples: &[f32], keep: usize) -> Result<(Vec<f32>, usize)> {
    debug_assert!(!mel.config().normalize);
    let n_mels = mel.config().n_mels;
    let (mut flat, frames) = mel.compute_flat(samples)?;
    let computed = frames.saturating_sub(1);
    flat.truncate(computed * n_mels);
    MelSpectrogram::whisper_normalize_flat(&mut flat);
    let kept = keep.min(computed);
    flat.truncate(kept * n_mels);
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
        assert_eq!(features.padded_frames, 300);
        assert_eq!(features.audio_tokens, 39);
        assert_eq!(features.flat.len(), 300 * 128);
        assert_eq!(features.padded_samples, 0);
        assert_eq!(frontend.mel_frames_for(48_000), 300);
        // A partial chunk: 250 frames -> 2 * 13 + ceil-halved 50 frames. The
        // features run to the whole chunk; the valid count does not.
        let features = frontend.prepare(&tone(40_000), 16_000).unwrap();
        assert_eq!(features.mel_frames, 250);
        assert_eq!(features.padded_frames, 300);
        assert_eq!(features.flat.len(), 300 * 128);
        assert_eq!(features.audio_tokens, 26 + 7);
        // A started hop is a valid frame, as the reference's mask counts it.
        let features = frontend.prepare(&tone(40_001), 16_000).unwrap();
        assert_eq!(features.mel_frames, 251);
        assert_eq!(frontend.mel_frames_for(40_001), 251);
        // A full 30 s window is not padded past itself.
        let features = frontend.prepare(&tone(480_000), 16_000).unwrap();
        assert_eq!(
            (features.mel_frames, features.padded_frames),
            (3_000, 3_000)
        );
        assert_eq!(features.audio_tokens, 390);
    }

    /// The frames past the audio are the mel of silence: constant across
    /// frames and mel bins once the audio's last window has passed, and not
    /// the zeros a tensor pad would give.
    #[test]
    fn the_chunk_padding_is_the_mel_of_silence() {
        let frontend = frontend();
        let features = frontend.prepare(&tone(40_000), 16_000).unwrap();
        let row = |frame: usize| &features.flat[frame * 128..(frame + 1) * 128];
        let silence = row(299);
        assert!(silence.iter().all(|value| *value == silence[0]));
        assert_ne!(silence[0], 0.0);
        for frame in 254..300 {
            assert_eq!(row(frame), silence, "frame {frame}");
        }
        // The clamp floor: 8 below the clip's maximum, then scaled.
        let max = features.flat.iter().copied().fold(f32::MIN, f32::max);
        assert!((silence[0] - (max - 2.0)).abs() < 1e-4, "{}", silence[0]);
        assert_ne!(row(100), silence);
    }

    #[test]
    fn a_clip_under_half_a_second_is_padded_to_it() {
        let frontend = frontend();
        let features = frontend.prepare(&tone(1_600), 16_000).unwrap();
        assert_eq!(features.padded_samples, 6_400);
        assert_eq!(features.mel_frames, 50);
        assert_eq!(features.padded_frames, 100);
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
