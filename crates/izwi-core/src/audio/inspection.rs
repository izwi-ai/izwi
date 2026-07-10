//! Lightweight decoded-audio inspection for request boundary validation.

use crate::error::Result;

/// Source-level metadata preserved before audio is downmixed to mono.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSourceMetadata {
    /// Detected source container, such as `wav`, `mp3`, `ogg`, or `mp4`.
    pub container: String,
    /// Detected source codec, such as `pcm_s16le`, `mp3`, `vorbis`, or `aac`.
    pub codec: String,
    /// Sample rate declared by the source stream.
    pub sample_rate: u32,
    /// Channel count declared by the source stream before mono downmixing.
    pub channel_count: u16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioInspection {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub duration_secs: f32,
    pub peak: f32,
    pub rms: f32,
    pub clipped_samples: usize,
}

/// A single strict decode result with both source metadata and mono signal data.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    pub source: AudioSourceMetadata,
    pub mono_samples: Vec<f32>,
    pub inspection: AudioInspection,
}

/// Strictly decode audio once while preserving source metadata and signal inspection.
///
/// This boundary API reports corrupt packets instead of silently skipping them. Runtime model
/// execution continues to use its existing permissive decoder for compatibility.
pub fn decode_and_inspect_audio_bytes(audio_bytes: &[u8]) -> Result<DecodedAudio> {
    let (mono_samples, source) =
        crate::runtime::audio_io::decode_audio_bytes_with_metadata(audio_bytes)?;
    let inspection = AudioInspection::from_mono_samples(&mono_samples, source.sample_rate);
    Ok(DecodedAudio {
        source,
        mono_samples,
        inspection,
    })
}

pub fn inspect_audio_bytes(audio_bytes: &[u8]) -> Result<AudioInspection> {
    decode_and_inspect_audio_bytes(audio_bytes).map(|decoded| decoded.inspection)
}

pub fn decode_audio_bytes_to_mono(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_and_inspect_audio_bytes(audio_bytes)
        .map(|decoded| (decoded.mono_samples, decoded.source.sample_rate))
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
    use std::path::PathBuf;

    fn wav_bytes(channels: u16, sample_rate: u32, interleaved_samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut bytes = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(std::io::Cursor::new(&mut bytes), spec).expect("WAV writer");
            for sample in interleaved_samples {
                writer.write_sample(*sample).expect("WAV sample");
            }
            writer.finalize().expect("WAV finalize");
        }
        bytes
    }

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

    #[test]
    fn decode_audio_bytes_to_mono_returns_samples_and_rate() {
        let samples = [0.0, 0.5, -0.5, 0.0];
        let wav = AudioEncoder::new(22_050, 1)
            .encode(&samples, AudioFormat::Wav)
            .expect("wav should encode");

        let (decoded, sample_rate) = decode_audio_bytes_to_mono(&wav).expect("wav should decode");

        assert_eq!(sample_rate, 22_050);
        assert_eq!(decoded.len(), samples.len());
    }

    #[test]
    fn decoded_audio_preserves_mono_wav_source_metadata() {
        let wav = wav_bytes(1, 22_050, &[0, 8_192, -8_192]);

        let decoded = decode_and_inspect_audio_bytes(&wav).expect("mono WAV should decode");

        assert_eq!(decoded.source.container, "wav");
        assert_eq!(decoded.source.codec, "pcm_s16le");
        assert_eq!(decoded.source.sample_rate, 22_050);
        assert_eq!(decoded.source.channel_count, 1);
        assert_eq!(decoded.mono_samples.len(), 3);
        assert_eq!(decoded.inspection.sample_rate, 22_050);
    }

    #[test]
    fn decoded_audio_preserves_stereo_metadata_while_downmixing() {
        let wav = wav_bytes(2, 16_000, &[8_192, 24_576, 16_384, -16_384]);

        let decoded = decode_and_inspect_audio_bytes(&wav).expect("stereo WAV should decode");

        assert_eq!(decoded.source.container, "wav");
        assert_eq!(decoded.source.codec, "pcm_s16le");
        assert_eq!(decoded.source.sample_rate, 16_000);
        assert_eq!(decoded.source.channel_count, 2);
        assert_eq!(decoded.mono_samples.len(), 2);
        assert!((decoded.mono_samples[0] - 0.5).abs() < 0.02);
        assert!(decoded.mono_samples[1].abs() < 0.02);
    }

    #[test]
    fn decoded_audio_preserves_mp3_fixture_metadata() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/diarization-2.mp3");
        let bytes = std::fs::read(fixture).expect("MP3 fixture");

        let decoded = decode_and_inspect_audio_bytes(&bytes).expect("MP3 should decode");

        assert_eq!(decoded.source.container, "mp3");
        assert_eq!(decoded.source.codec, "mp3");
        assert!(decoded.source.sample_rate > 0);
        assert!(decoded.source.channel_count > 0);
        assert!(!decoded.mono_samples.is_empty());
    }

    #[test]
    fn strict_decode_rejects_invalid_non_audio_bytes() {
        let error = decode_and_inspect_audio_bytes(b"not an audio stream")
            .expect_err("invalid audio must fail strict decoding");

        assert!(error.to_string().contains("Failed to decode audio"));
    }
}
