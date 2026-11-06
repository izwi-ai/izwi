//! Assistant audio encodings a realtime voice session may ask for (SPK-F11):
//! 16-bit PCM at the model's rate (the default, 384 kb/s at 24 kHz), or
//! G.711 μ-law or A-law at 8 kHz (64 kb/s, telephony). The companding and
//! the resampler are `izwi-audio-codec`'s, shared with `/v1/audio/speech`.

use izwi_audio_codec::StreamResampler;

pub(crate) use izwi_audio_codec::g711::{
    alaw_to_linear, linear_to_alaw, linear_to_ulaw, ulaw_to_linear,
};

/// G.711's sample rate.
pub(crate) const G711_SAMPLE_RATE: u32 = izwi_audio_codec::g711::SAMPLE_RATE;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum OutputAudioFormat {
    #[default]
    PcmI16,
    G711Ulaw,
    G711Alaw,
}

impl OutputAudioFormat {
    /// The format a client named in `output_audio_format`; PCM16 when unset.
    pub(crate) fn parse(name: Option<&str>) -> Result<Self, String> {
        match name.map(str::trim) {
            None | Some("" | "pcm_i16" | "pcm16") => Ok(Self::PcmI16),
            Some("g711_ulaw") => Ok(Self::G711Ulaw),
            Some("g711_alaw") => Ok(Self::G711Alaw),
            Some(other) => Err(format!(
                "Unsupported output_audio_format `{other}` (expected pcm_i16, g711_ulaw or g711_alaw)"
            )),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::PcmI16 => "pcm_i16",
            Self::G711Ulaw => "g711_ulaw",
            Self::G711Alaw => "g711_alaw",
        }
    }

    /// The rate audio from a `model_rate` model goes out at.
    pub(crate) const fn wire_rate(self, model_rate: u32) -> u32 {
        match self {
            Self::PcmI16 => model_rate,
            Self::G711Ulaw | Self::G711Alaw => G711_SAMPLE_RATE,
        }
    }
}

/// Encodes one reply's audio into its session's format, resampling to 8 kHz
/// for G.711 across chunk boundaries.
pub(crate) struct AudioOutputEncoder {
    format: OutputAudioFormat,
    resampler: Option<StreamResampler>,
}

impl AudioOutputEncoder {
    pub(crate) fn new(format: OutputAudioFormat, model_rate: u32) -> Result<Self, String> {
        let resampler = match format {
            OutputAudioFormat::PcmI16 => None,
            OutputAudioFormat::G711Ulaw | OutputAudioFormat::G711Alaw
                if model_rate == G711_SAMPLE_RATE =>
            {
                None
            }
            _ => Some(
                StreamResampler::new(model_rate, G711_SAMPLE_RATE)
                    .map_err(|err| err.to_string())?,
            ),
        };
        Ok(Self { format, resampler })
    }

    /// 16-bit PCM at `target_rate` whatever the model's rate: OpenAI
    /// Realtime's `audio/pcm` is 24 kHz only, so audio from a model at
    /// another rate is resampled rather than mislabelled.
    pub(crate) fn pcm_at(model_rate: u32, target_rate: u32) -> Result<Self, String> {
        let resampler = (model_rate != target_rate)
            .then(|| StreamResampler::new(model_rate, target_rate))
            .transpose()
            .map_err(|err| err.to_string())?;
        Ok(Self {
            format: OutputAudioFormat::PcmI16,
            resampler,
        })
    }

    /// The payload for `samples` (some audio may be held for the next call).
    pub(crate) fn encode(&mut self, samples: &[f32]) -> Result<Vec<u8>, String> {
        let samples = match self.resampler.as_mut() {
            Some(resampler) => resampler.push(samples).map_err(|err| err.to_string())?,
            None => samples.to_vec(),
        };
        Ok(self.compand(&samples))
    }

    /// The payload for the audio still held at the end of the reply.
    pub(crate) fn finish(&mut self) -> Result<Vec<u8>, String> {
        let samples = match self.resampler.as_mut() {
            Some(resampler) => resampler.finish().map_err(|err| err.to_string())?,
            None => Vec::new(),
        };
        Ok(self.compand(&samples))
    }

    fn compand(&self, samples: &[f32]) -> Vec<u8> {
        let linear = samples
            .iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * 32_767.0) as i16);
        match self.format {
            OutputAudioFormat::PcmI16 => linear.flat_map(i16::to_le_bytes).collect(),
            OutputAudioFormat::G711Ulaw => linear.map(linear_to_ulaw).collect(),
            OutputAudioFormat::G711Alaw => linear.map(linear_to_alaw).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g711_codes_match_the_standard_anchors() {
        assert_eq!(linear_to_ulaw(0), 0xFF);
        assert_eq!(linear_to_ulaw(i16::MAX), 0x80);
        assert_eq!(linear_to_ulaw(i16::MIN), 0x00);
        assert_eq!(linear_to_alaw(0), 0xD5);
        assert_eq!(linear_to_alaw(i16::MAX), 0xAA);
        assert_eq!(linear_to_alaw(i16::MIN), 0x2A);
    }

    /// Companding keeps every sample within its segment's quantization step
    /// (relative error under 1/16 above the smallest steps).
    #[test]
    fn g711_round_trips_within_one_quantization_step() {
        for sample in (i16::MIN..=i16::MAX).step_by(7) {
            for (name, decoded) in [
                ("ulaw", ulaw_to_linear(linear_to_ulaw(sample))),
                ("alaw", alaw_to_linear(linear_to_alaw(sample))),
            ] {
                let error = (i32::from(decoded) - i32::from(sample)).abs();
                let allowed = (i32::from(sample).abs() / 16).max(140);
                assert!(error <= allowed, "{name} {sample} -> {decoded}");
            }
        }
    }

    /// G.711 output is 8 kHz, one byte a sample, with the same duration as
    /// the model audio however it is chunked; PCM16 is two bytes a sample at
    /// the model rate.
    #[test]
    fn encoders_keep_duration_across_chunks() {
        let tone: Vec<f32> = (0..24_000)
            .map(|index| (index as f32 * 440.0 * std::f32::consts::TAU / 24_000.0).sin() * 0.5)
            .collect();
        for format in [OutputAudioFormat::G711Ulaw, OutputAudioFormat::G711Alaw] {
            let mut encoder = AudioOutputEncoder::new(format, 24_000).expect("encoder");
            let mut bytes = Vec::new();
            for chunk in tone.chunks(1_337) {
                bytes.extend(encoder.encode(chunk).expect("encode"));
            }
            bytes.extend(encoder.finish().expect("finish"));
            let expected = 8_000;
            assert!(
                (bytes.len() as i64 - expected).abs() <= 16,
                "{format:?}: {} samples for one second",
                bytes.len()
            );
        }
        let mut pcm = AudioOutputEncoder::new(OutputAudioFormat::PcmI16, 24_000).expect("pcm");
        assert_eq!(pcm.encode(&tone[..100]).expect("pcm").len(), 200);
        assert!(pcm.finish().expect("finish").is_empty());
    }

    #[test]
    fn formats_parse_by_their_wire_names() {
        assert_eq!(
            OutputAudioFormat::parse(None),
            Ok(OutputAudioFormat::PcmI16)
        );
        assert_eq!(
            OutputAudioFormat::parse(Some("g711_ulaw")),
            Ok(OutputAudioFormat::G711Ulaw)
        );
        assert_eq!(OutputAudioFormat::G711Alaw.wire_rate(24_000), 8_000);
        assert_eq!(OutputAudioFormat::PcmI16.wire_rate(24_000), 24_000);
        assert!(OutputAudioFormat::parse(Some("opus"))
            .unwrap_err()
            .contains("opus"));
    }
}
