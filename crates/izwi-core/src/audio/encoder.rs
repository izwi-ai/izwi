//! Audio encoding to various output formats

use hound::{WavSpec, WavWriter};
use std::io::Cursor;
use tracing::debug;

use crate::error::{Error, Result};

/// Supported audio output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// WAV format (PCM)
    Wav,
    /// Raw PCM samples (f32)
    RawF32,
    /// Raw PCM samples (i16)
    RawI16,
}

/// Audio encoder for converting f32 samples to various formats
pub struct AudioEncoder {
    sample_rate: u32,
    channels: u16,
}

impl AudioEncoder {
    /// Create a new encoder
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    /// Encode samples to the specified format
    pub fn encode(&self, samples: &[f32], format: AudioFormat) -> Result<Vec<u8>> {
        match format {
            AudioFormat::Wav => self.encode_wav(samples),
            AudioFormat::RawF32 => self.encode_raw_f32(samples),
            AudioFormat::RawI16 => self.encode_raw_i16(samples),
        }
    }

    /// Encode to WAV format
    fn encode_wav(&self, samples: &[f32]) -> Result<Vec<u8>> {
        let spec = WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut buffer = Cursor::new(Vec::new());
        {
            let mut writer =
                WavWriter::new(&mut buffer, spec).map_err(|e| Error::AudioError(e.to_string()))?;

            for &sample in samples {
                // Convert f32 [-1.0, 1.0] to i16
                let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer
                    .write_sample(sample_i16)
                    .map_err(|e| Error::AudioError(e.to_string()))?;
            }

            writer
                .finalize()
                .map_err(|e| Error::AudioError(e.to_string()))?;
        }

        debug!(
            "Encoded {} samples to WAV ({} bytes)",
            samples.len(),
            buffer.get_ref().len()
        );
        Ok(buffer.into_inner())
    }

    /// Encode to raw f32 samples
    fn encode_raw_f32(&self, samples: &[f32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for &sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Encode to raw i16 samples
    fn encode_raw_i16(&self, samples: &[f32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &sample in samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Get content type for format
    pub fn content_type(format: AudioFormat) -> &'static str {
        match format {
            AudioFormat::Wav => "audio/wav",
            AudioFormat::RawF32 => "application/octet-stream",
            AudioFormat::RawI16 => "application/octet-stream",
        }
    }
}
