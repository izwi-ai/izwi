//! MP3 through LAME (`mp3lame-encoder`), constant bitrate, mono.
//!
//! LAME takes any number of samples and returns whatever whole frames are
//! ready (576 samples a frame at 24 kHz, 1152 at 44.1/48 kHz), so a stream
//! forwards each call's bytes as they come. `finish` pads the last frame
//! with silence (`lame_encode_flush`); the no-gap flush would drop it. In-order concatenation is always
//! decodable even though the bit reservoir lets a frame borrow bytes from
//! earlier ones. A stream writes no LAME tag (it would need a seek back to
//! the first frame); a file gets one, so players trim the encoder delay.

use std::num::NonZeroU32;

use mp3lame_encoder::{Bitrate, Builder, FlushGap, Mode, MonoPcm, Quality, VbrMode};

use crate::{sanitize, AudioFormat, CodecError, Framing, StreamEncoder};

/// LAME's bitrates, in kbit/s.
const BITRATES: [(u32, Bitrate); 16] = [
    (8, Bitrate::Kbps8),
    (16, Bitrate::Kbps16),
    (24, Bitrate::Kbps24),
    (32, Bitrate::Kbps32),
    (40, Bitrate::Kbps40),
    (48, Bitrate::Kbps48),
    (64, Bitrate::Kbps64),
    (80, Bitrate::Kbps80),
    (96, Bitrate::Kbps96),
    (112, Bitrate::Kbps112),
    (128, Bitrate::Kbps128),
    (160, Bitrate::Kbps160),
    (192, Bitrate::Kbps192),
    (224, Bitrate::Kbps224),
    (256, Bitrate::Kbps256),
    (320, Bitrate::Kbps320),
];

/// The bitrate range (kbit/s) the MPEG version for `rate` allows.
fn bitrate_range(rate: u32) -> (u32, u32) {
    match rate {
        32_000.. => (32, 320),      // MPEG-1
        16_000..32_000 => (8, 160), // MPEG-2 LSF
        _ => (8, 64),               // MPEG-2.5
    }
}

/// The LAME bitrate for audio at `rate`: the requested one (bits a second,
/// rounded down to a step LAME has), or a speech default.
pub(crate) fn bitrate_for(rate: u32, requested: Option<u32>) -> Result<Bitrate, CodecError> {
    let (low, high) = bitrate_range(rate);
    let kbps = match requested {
        Some(bits) => {
            let kbps = bits / 1_000;
            if !(low..=high).contains(&kbps) {
                return Err(CodecError::InvalidOption(format!(
                    "bitrate {bits} is out of range for mp3 at {rate} Hz ({} to {} bit/s)",
                    low * 1_000,
                    high * 1_000
                )));
            }
            kbps
        }
        None => match rate {
            32_000.. => 128,
            16_000..32_000 => 64,
            _ => 32,
        },
    };
    Ok(BITRATES
        .iter()
        .rev()
        .find(|(step, _)| *step <= kbps)
        .map(|(_, bitrate)| *bitrate)
        .expect("8 kbit/s is the floor"))
}

pub(crate) struct Mp3Encoder {
    encoder: mp3lame_encoder::Encoder,
    rate: u32,
    framing: Framing,
    /// File framing: the encoded frames so far.
    file: Vec<u8>,
    scratch: Vec<f32>,
    finished: bool,
}

fn build_error(err: impl std::fmt::Display) -> CodecError {
    CodecError::Encode(format!("Failed to set up the MP3 encoder: {err}"))
}

impl Mp3Encoder {
    pub(crate) fn new(
        rate: u32,
        bitrate: Option<u32>,
        framing: Framing,
    ) -> Result<Self, CodecError> {
        let bitrate = bitrate_for(rate, bitrate)?;
        let mut builder = Builder::new().ok_or_else(|| build_error("out of memory"))?;
        builder.set_num_channels(1).map_err(build_error)?;
        builder.set_mode(Mode::Mono).map_err(build_error)?;
        builder.set_sample_rate(rate).map_err(build_error)?;
        builder
            .set_output_sample_rate(NonZeroU32::new(rate))
            .map_err(build_error)?;
        builder.set_vbr_mode(VbrMode::Off).map_err(build_error)?;
        builder.set_brate(bitrate).map_err(build_error)?;
        builder
            .set_quality(Quality::NearBest)
            .map_err(build_error)?;
        builder
            .set_to_write_vbr_tag(framing == Framing::File)
            .map_err(build_error)?;
        let encoder = builder.build().map_err(build_error)?;
        Ok(Self {
            encoder,
            rate,
            framing,
            file: Vec::new(),
            scratch: Vec::new(),
            finished: false,
        })
    }

    fn encode_into(&mut self, pcm: &[f32], out: &mut Vec<u8>) -> Result<(), CodecError> {
        self.scratch.clear();
        self.scratch.extend(pcm.iter().copied().map(sanitize));
        out.reserve(mp3lame_encoder::max_required_buffer_size(pcm.len()));
        self.encoder
            .encode_to_vec(MonoPcm(self.scratch.as_slice()), out)
            .map_err(|err| CodecError::Encode(format!("MP3 encoding failed: {err}")))?;
        Ok(())
    }
}

impl StreamEncoder for Mp3Encoder {
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        if self.finished {
            return Err(CodecError::Encode("MP3 encoder already finished".into()));
        }
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        match self.framing {
            Framing::Stream => {
                let mut out = Vec::new();
                self.encode_into(pcm, &mut out)?;
                Ok(out)
            }
            Framing::File => {
                let mut file = std::mem::take(&mut self.file);
                let result = self.encode_into(pcm, &mut file);
                self.file = file;
                result.map(|()| Vec::new())
            }
        }
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if std::mem::replace(&mut self.finished, true) {
            return Err(CodecError::Encode("MP3 encoder already finished".into()));
        }
        let mut out = std::mem::take(&mut self.file);
        // "you need at least 7200 bytes" for the final flush.
        out.reserve(7_200);
        self.encoder
            .flush_to_vec::<FlushGap>(&mut out)
            .map_err(|err| CodecError::Encode(format!("MP3 flush failed: {err}")))?;
        if self.framing == Framing::File && self.encoder.is_lame_tag_written() {
            // The first frame is a placeholder for the LAME tag, which
            // carries the encoder delay and padding for gapless players.
            let mut tag = Vec::with_capacity(self.encoder.lame_tag_size());
            if let Some(written) = self.encoder.lame_tag_encode_to_vec(&mut tag) {
                let start = self.encoder.id3v2_tag_size();
                let end = start + written.get();
                if end <= out.len() {
                    out[start..end].copy_from_slice(&tag[..written.get()]);
                }
            }
        }
        Ok(out)
    }

    fn format(&self) -> AudioFormat {
        AudioFormat::Mp3
    }

    fn output_rate(&self) -> u32 {
        self.rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitrates_follow_the_mpeg_version() {
        assert!(matches!(bitrate_for(24_000, None), Ok(Bitrate::Kbps64)));
        assert!(matches!(bitrate_for(44_100, None), Ok(Bitrate::Kbps128)));
        assert!(matches!(
            bitrate_for(24_000, Some(100_000)),
            Ok(Bitrate::Kbps96)
        ));
        assert!(bitrate_for(24_000, Some(192_000)).is_err());
        assert!(bitrate_for(48_000, Some(16_000)).is_err());
    }
}
