//! Uncompressed outputs: WAV, headerless PCM and G.711.

use crate::{g711, to_i16, AudioFormat, CodecError, Framing, StreamEncoder};

/// A RIFF size field that says "unknown": streamed WAV cannot know its
/// length up front, and players read to the end of the stream.
const UNKNOWN_SIZE: u32 = u32::MAX;

/// The 44-byte header of a mono 16-bit PCM WAV file.
fn wav_header(sample_rate: u32, data_bytes: u32) -> [u8; 44] {
    let riff_size = if data_bytes == UNKNOWN_SIZE {
        UNKNOWN_SIZE
    } else {
        data_bytes.saturating_add(36)
    };
    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&riff_size.to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    header[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    header[28..32].copy_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    header[32..34].copy_from_slice(&2u16.to_le_bytes()); // block align
    header[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits a sample
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    header
}

fn push_i16(out: &mut Vec<u8>, pcm: &[f32]) {
    out.reserve(pcm.len() * 2);
    for &sample in pcm {
        out.extend_from_slice(&to_i16(sample).to_le_bytes());
    }
}

/// 16-bit mono WAV. A stream starts with a header whose sizes say
/// "unknown"; a file gets exact sizes, byte for byte what hound writes.
pub(crate) struct WavEncoder {
    rate: u32,
    framing: Framing,
    header_sent: bool,
    /// File framing: the PCM so far.
    data: Vec<u8>,
    finished: bool,
}

impl WavEncoder {
    pub(crate) fn new(rate: u32, framing: Framing) -> Self {
        Self {
            rate,
            framing,
            header_sent: false,
            data: Vec::new(),
            finished: false,
        }
    }
}

impl StreamEncoder for WavEncoder {
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        if self.finished {
            return Err(CodecError::Encode("WAV encoder already finished".into()));
        }
        if self.framing == Framing::File {
            push_i16(&mut self.data, pcm);
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(44 + pcm.len() * 2);
        if !self.header_sent {
            out.extend_from_slice(&wav_header(self.rate, UNKNOWN_SIZE));
            self.header_sent = true;
        }
        push_i16(&mut out, pcm);
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if std::mem::replace(&mut self.finished, true) {
            return Err(CodecError::Encode("WAV encoder already finished".into()));
        }
        match self.framing {
            Framing::Stream if self.header_sent => Ok(Vec::new()),
            Framing::Stream => Ok(wav_header(self.rate, UNKNOWN_SIZE).to_vec()),
            Framing::File => {
                let data_bytes = u32::try_from(self.data.len())
                    .ok()
                    .filter(|bytes| *bytes <= u32::MAX - 36)
                    .ok_or_else(|| {
                        CodecError::Encode("Audio is too long for a WAV file (4 GiB)".into())
                    })?;
                let mut out = Vec::with_capacity(44 + self.data.len());
                out.extend_from_slice(&wav_header(self.rate, data_bytes));
                out.append(&mut self.data);
                Ok(out)
            }
        }
    }

    fn format(&self) -> AudioFormat {
        AudioFormat::Wav
    }

    fn output_rate(&self) -> u32 {
        self.rate
    }
}

/// Headerless sample formats: PCM16, float PCM and G.711.
pub(crate) struct RawEncoder {
    format: AudioFormat,
    rate: u32,
    finished: bool,
}

impl RawEncoder {
    pub(crate) fn new(format: AudioFormat, rate: u32) -> Self {
        Self {
            format,
            rate,
            finished: false,
        }
    }
}

impl StreamEncoder for RawEncoder {
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        if self.finished {
            return Err(CodecError::Encode(format!(
                "{} encoder already finished",
                self.format
            )));
        }
        let mut out = Vec::new();
        match self.format {
            AudioFormat::PcmI16 => push_i16(&mut out, pcm),
            AudioFormat::PcmF32 => {
                out.reserve(pcm.len() * 4);
                for &sample in pcm {
                    out.extend_from_slice(&sample.to_le_bytes());
                }
            }
            AudioFormat::G711Ulaw => {
                out.extend(pcm.iter().map(|&s| g711::linear_to_ulaw(to_i16(s))));
            }
            AudioFormat::G711Alaw => {
                out.extend(pcm.iter().map(|&s| g711::linear_to_alaw(to_i16(s))));
            }
            other => {
                return Err(CodecError::Encode(format!(
                    "{other} is not a headerless format"
                )))
            }
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if std::mem::replace(&mut self.finished, true) {
            return Err(CodecError::Encode(format!(
                "{} encoder already finished",
                self.format
            )));
        }
        Ok(Vec::new())
    }

    fn format(&self) -> AudioFormat {
        self.format
    }

    fn output_rate(&self) -> u32 {
        self.rate
    }
}
