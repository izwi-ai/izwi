//! Assistant audio encodings a realtime voice session may ask for (SPK-F11):
//! 16-bit PCM at the model's rate (the default, 384 kb/s at 24 kHz), or
//! G.711 μ-law or A-law at 8 kHz (64 kb/s, telephony). Opus would need a C
//! encoder; there is none in the dependency tree.

use rubato::{FftFixedIn, Resampler};

/// G.711's sample rate.
pub(crate) const G711_SAMPLE_RATE: u32 = 8_000;
/// Input frames the G.711 resampler takes at a time (20 ms at 24 kHz).
const RESAMPLE_CHUNK_FRAMES: usize = 480;

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
            _ => Some(StreamResampler::new(model_rate, G711_SAMPLE_RATE)?),
        };
        Ok(Self { format, resampler })
    }

    /// The payload for `samples` (some audio may be held for the next call).
    pub(crate) fn encode(&mut self, samples: &[f32]) -> Result<Vec<u8>, String> {
        let samples = match self.resampler.as_mut() {
            Some(resampler) => resampler.push(samples)?,
            None => samples.to_vec(),
        };
        Ok(self.compand(&samples))
    }

    /// The payload for the audio still held at the end of the reply.
    pub(crate) fn finish(&mut self) -> Result<Vec<u8>, String> {
        let samples = match self.resampler.as_mut() {
            Some(resampler) => resampler.finish()?,
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

/// A band-limited resampler fed in arbitrary pieces.
struct StreamResampler {
    resampler: FftFixedIn<f32>,
    from: u32,
    to: u32,
    pending: Vec<f32>,
    /// Leading output samples that are the resampler's delay, not audio.
    delay: usize,
    input_total: usize,
    output_total: usize,
}

impl StreamResampler {
    fn new(from: u32, to: u32) -> Result<Self, String> {
        let resampler =
            FftFixedIn::<f32>::new(from as usize, to as usize, RESAMPLE_CHUNK_FRAMES, 1, 1)
                .map_err(|err| format!("Failed to build the output resampler: {err}"))?;
        Ok(Self {
            delay: resampler.output_delay(),
            resampler,
            from,
            to,
            pending: Vec::new(),
            input_total: 0,
            output_total: 0,
        })
    }

    fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        self.input_total += samples.len();
        self.pending.extend_from_slice(samples);
        let mut out = Vec::new();
        loop {
            let needed = self.resampler.input_frames_next();
            if self.pending.len() < needed {
                break;
            }
            let chunk: Vec<f32> = self.pending.drain(..needed).collect();
            let produced = self
                .resampler
                .process(&[chunk], None)
                .map_err(|err| format!("Output resampling failed: {err}"))?;
            out.extend_from_slice(&produced[0]);
        }
        let out = self.skip_delay(out);
        self.output_total += out.len();
        Ok(out)
    }

    /// The rest of the output: the held input and the resampler's delay
    /// line flushed with silence, cut to the input's duration.
    fn finish(&mut self) -> Result<Vec<f32>, String> {
        let expected =
            (self.input_total as u64 * u64::from(self.to)).div_ceil(u64::from(self.from)) as usize;
        let mut out = Vec::new();
        let mut pending = std::mem::take(&mut self.pending);
        while self.output_total + out.len() < expected + self.delay {
            pending.resize(self.resampler.input_frames_next(), 0.0);
            let produced = self
                .resampler
                .process(&[std::mem::take(&mut pending)], None)
                .map_err(|err| format!("Output resampling failed: {err}"))?;
            out.extend_from_slice(&produced[0]);
        }
        let mut out = self.skip_delay(out);
        out.truncate(expected.saturating_sub(self.output_total));
        self.output_total += out.len();
        Ok(out)
    }

    fn skip_delay(&mut self, mut out: Vec<f32>) -> Vec<f32> {
        let skip = self.delay.min(out.len());
        self.delay -= skip;
        out.drain(..skip);
        out
    }
}

/// ITU-T G.711 μ-law of one 16-bit linear sample.
pub(crate) fn linear_to_ulaw(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32_635;
    let mut pcm = i32::from(sample);
    let sign = if pcm < 0 {
        pcm = -pcm;
        0x80
    } else {
        0
    };
    let pcm = pcm.min(CLIP) + BIAS;
    let exponent = (15 - (pcm as u16).leading_zeros() as i32 - 7).max(0);
    let mantissa = (pcm >> (exponent + 3)) & 0x0F;
    !((sign | (exponent << 4) | mantissa) as u8)
}

/// ITU-T G.711 A-law of one 16-bit linear sample.
pub(crate) fn linear_to_alaw(sample: i16) -> u8 {
    const SEGMENT_ENDS: [i32; 8] = [0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF];
    let mut pcm = i32::from(sample) >> 3;
    let mask = if pcm >= 0 {
        0xD5
    } else {
        pcm = -pcm - 1;
        0x55
    };
    let Some(segment) = SEGMENT_ENDS.iter().position(|end| pcm <= *end) else {
        return 0x7F ^ mask;
    };
    let shift = if segment < 2 { 1 } else { segment };
    let value = ((segment as i32) << 4) | ((pcm >> shift) & 0x0F);
    (value as u8) ^ mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference μ-law decoder (ITU-T G.711), for round trips.
    fn ulaw_to_linear(code: u8) -> i16 {
        let code = !code;
        let exponent = (code >> 4) & 0x07;
        let magnitude = ((((code & 0x0F) as i32) << 3) + 0x84) << exponent;
        let value = magnitude - 0x84;
        (if code & 0x80 != 0 { -value } else { value }) as i16
    }

    /// Reference A-law decoder (ITU-T G.711), for round trips.
    fn alaw_to_linear(code: u8) -> i16 {
        let code = code ^ 0x55;
        let segment = (code & 0x70) >> 4;
        let mut value = ((code & 0x0F) as i32) << 4;
        value = match segment {
            0 => value + 8,
            1 => value + 0x108,
            _ => (value + 0x108) << (segment - 1),
        };
        (if code & 0x80 != 0 { value } else { -value }) as i16
    }

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
