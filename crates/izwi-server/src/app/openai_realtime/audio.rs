//! Audio on the OpenAI Realtime wire: base64 input in the session's format
//! decoded to 16-bit samples, and model speech encoded to the session's
//! output format (24 kHz PCM16 resampled from the model's rate, or G.711 at
//! 8 kHz) and cut into deltas of at most 100 ms.

use base64::Engine;

use super::events::RealtimeError;
use super::session_config::AudioFormat;
use crate::app::realtime_codec::{
    alaw_to_linear, ulaw_to_linear, AudioOutputEncoder, OutputAudioFormat,
};

/// The longest audio one `response.output_audio.delta` carries. Small
/// deltas keep each event far below Pipecat's 1 MiB message limit and match
/// the 100 ms frames LiveKit sends.
pub(crate) const MAX_DELTA_MS: u32 = 100;
/// OpenAI's largest `input_audio_buffer.append`.
pub(crate) const MAX_APPEND_BYTES: usize = 15 * 1024 * 1024;

/// Decode one `input_audio_buffer.append`'s audio to samples at
/// `format.sample_rate()`.
pub(crate) fn decode_input(
    audio_base64: &str,
    format: AudioFormat,
) -> Result<Vec<i16>, RealtimeError> {
    if audio_base64.len() > MAX_APPEND_BYTES / 3 * 4 + 4 {
        return Err(RealtimeError::invalid(
            "audio_too_large",
            "An append carries at most 15 MiB of audio.",
        )
        .with_param("audio"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_base64.trim())
        .map_err(|err| {
            RealtimeError::invalid(
                "invalid_value",
                format!("The audio is not valid base64: {err}"),
            )
            .with_param("audio")
        })?;
    Ok(match format {
        AudioFormat::Pcm24k => {
            if bytes.len() % 2 != 0 {
                return Err(RealtimeError::invalid(
                    "invalid_value",
                    "PCM16 audio must be a whole number of 16-bit samples.",
                )
                .with_param("audio"));
            }
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect()
        }
        AudioFormat::Pcmu => bytes.into_iter().map(ulaw_to_linear).collect(),
        AudioFormat::Pcma => bytes.into_iter().map(alaw_to_linear).collect(),
    })
}

/// Samples as little-endian PCM16 bytes.
pub(crate) fn pcm16_bytes(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

pub(crate) fn base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Encodes one reply's speech into the session's output format.
pub(crate) struct OutputEncoder {
    format: AudioFormat,
    encoder: AudioOutputEncoder,
    /// Encoded bytes not yet sent: deltas go out in whole 100 ms pieces.
    pending: Vec<u8>,
}

impl OutputEncoder {
    pub(crate) fn new(format: AudioFormat, model_rate: u32) -> Result<Self, String> {
        let encoder = match format {
            AudioFormat::Pcm24k => AudioOutputEncoder::pcm_at(model_rate, AudioFormat::PCM_RATE)?,
            AudioFormat::Pcmu => AudioOutputEncoder::new(OutputAudioFormat::G711Ulaw, model_rate)?,
            AudioFormat::Pcma => AudioOutputEncoder::new(OutputAudioFormat::G711Alaw, model_rate)?,
        };
        Ok(Self {
            format,
            encoder,
            pending: Vec::new(),
        })
    }

    fn bytes_per_ms(&self) -> usize {
        match self.format {
            AudioFormat::Pcm24k => 48,
            AudioFormat::Pcmu | AudioFormat::Pcma => 8,
        }
    }

    /// The deltas `samples` complete; `finish` flushes the rest.
    pub(crate) fn push(&mut self, samples: &[f32], finish: bool) -> Result<Vec<Vec<u8>>, String> {
        let encoded = self.encoder.encode(samples)?;
        self.pending.extend_from_slice(&encoded);
        if finish {
            let tail = self.encoder.finish()?;
            self.pending.extend_from_slice(&tail);
        }
        let piece = self.bytes_per_ms() * MAX_DELTA_MS as usize;
        let mut deltas = Vec::new();
        while self.pending.len() >= piece {
            deltas.push(self.pending.drain(..piece).collect());
        }
        if finish && !self.pending.is_empty() {
            deltas.push(std::mem::take(&mut self.pending));
        }
        Ok(deltas)
    }

    /// Milliseconds of audio `bytes` hold in this format.
    pub(crate) fn duration_ms(&self, bytes: usize) -> u64 {
        (bytes / self.bytes_per_ms()) as u64
    }
}

/// `samples` as a WAV file, for the ASR and speech-to-speech models.
pub(crate) fn wav(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>, String> {
    crate::app::realtime_transport::wav_bytes_from_pcm16_mono(samples, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_decodes_pcm16_and_g711() {
        let pcm = base64(&pcm16_bytes(&[1, -2, 300]));
        assert_eq!(
            decode_input(&pcm, AudioFormat::Pcm24k).expect("pcm"),
            [1, -2, 300]
        );
        let ulaw = base64(&[0xFF, 0x7F]);
        assert_eq!(
            decode_input(&ulaw, AudioFormat::Pcmu).expect("ulaw"),
            [0, 0]
        );
        assert!(
            decode_input("AAE", AudioFormat::Pcm24k).is_err(),
            "odd byte count"
        );
        assert!(
            decode_input("!!", AudioFormat::Pcm24k).is_err(),
            "not base64"
        );
    }

    #[test]
    fn speech_goes_out_in_100_ms_deltas_at_24_khz() {
        let mut encoder = OutputEncoder::new(AudioFormat::Pcm24k, 24_000).expect("encoder");
        let deltas = encoder.push(&vec![0.1; 6_000], false).expect("push");
        assert_eq!(deltas.len(), 2, "250 ms makes two whole 100 ms deltas");
        assert!(deltas.iter().all(|delta| delta.len() == 4_800));
        let rest = encoder.push(&[], true).expect("finish");
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].len(), 2_400);
        assert_eq!(encoder.duration_ms(4_800), 100);
    }

    #[test]
    fn other_model_rates_are_resampled_to_24_khz() {
        let mut encoder = OutputEncoder::new(AudioFormat::Pcm24k, 16_000).expect("encoder");
        let mut bytes = 0;
        for delta in encoder.push(&vec![0.2; 16_000], true).expect("push") {
            bytes += delta.len();
        }
        assert!(
            (bytes as i64 - 48_000).abs() <= 96,
            "one second at 24 kHz, got {bytes} bytes"
        );
    }

    #[test]
    fn g711_output_is_8_khz_one_byte_a_sample() {
        let mut encoder = OutputEncoder::new(AudioFormat::Pcmu, 24_000).expect("encoder");
        let bytes: usize = encoder
            .push(&vec![0.3; 24_000], true)
            .expect("push")
            .iter()
            .map(Vec::len)
            .sum();
        assert!((bytes as i64 - 8_000).abs() <= 16, "{bytes}");
    }
}
