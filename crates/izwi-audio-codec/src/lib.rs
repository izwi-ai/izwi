//! Audio encoders shared by every Izwi surface that hands audio to a client:
//! `/v1/audio/speech` (whole files, chunked bodies and SSE deltas), the
//! text-in speech socket, Studio exports, speech-history downloads and the
//! realtime sockets.
//!
//! Every encoder takes mono `f32` PCM in pieces of any size and returns the
//! bytes that are ready, so one byte stream serves a file and a stream alike:
//! the concatenation of everything [`StreamEncoder::push`] and
//! [`StreamEncoder::finish`] returned is always a valid file of the format.
//! [`Framing::File`] only adds what needs the whole audio first (exact WAV
//! sizes, FLAC totals and MD5, the MP3 LAME tag).
//!
//! This crate links no inference code, so izwi-gateway and the CLI can use it
//! too.

#![forbid(unsafe_code)]

mod flac;
pub mod g711;
#[cfg(feature = "mp3")]
mod mp3;
#[cfg(feature = "opus")]
mod opus;
mod pcm;
mod resample;

pub use resample::StreamResampler;

/// An audio format Izwi can hand out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    /// 16-bit PCM in a RIFF/WAVE container.
    Wav,
    /// Headerless 16-bit little-endian PCM.
    PcmI16,
    /// Headerless 32-bit little-endian float PCM.
    PcmF32,
    /// MPEG-1/2 Layer III, constant bitrate.
    Mp3,
    /// Opus in an Ogg container (RFC 7845).
    Opus,
    /// AAC-LC in ADTS frames. Never built in: AAC encoders need a patent
    /// licence, so this build refuses it.
    Aac,
    /// FLAC, 16 bits a sample.
    Flac,
    /// G.711 μ-law at 8 kHz.
    G711Ulaw,
    /// G.711 A-law at 8 kHz.
    G711Alaw,
}

/// MP3's sample rates (MPEG-1, MPEG-2 LSF and MPEG-2.5).
const MP3_RATES: [u32; 9] = [
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
];
/// The rates libopus encodes at.
const OPUS_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];
/// The output rates a caller may ask the container formats for.
const PCM_RATE_RANGE: std::ops::RangeInclusive<u32> = 8_000..=192_000;

impl AudioFormat {
    pub const ALL: [AudioFormat; 9] = [
        Self::Wav,
        Self::PcmI16,
        Self::PcmF32,
        Self::Mp3,
        Self::Opus,
        Self::Aac,
        Self::Flac,
        Self::G711Ulaw,
        Self::G711Alaw,
    ];

    /// The format a request names, in Izwi's or OpenAI's spelling (any case).
    /// OpenAI's bare `pcm` is not here: it means 24 kHz PCM16, which is a rate
    /// as well as a format, so the caller maps it.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "wav" | "wave" => Self::Wav,
            "pcm16" | "pcm_i16" | "raw_i16" | "pcm_s16le" => Self::PcmI16,
            "pcm_f32" | "raw_f32" | "f32" => Self::PcmF32,
            "mp3" | "mpeg" => Self::Mp3,
            "opus" | "ogg" => Self::Opus,
            "aac" => Self::Aac,
            "flac" => Self::Flac,
            "g711_ulaw" | "ulaw" | "mulaw" | "pcmu" => Self::G711Ulaw,
            "g711_alaw" | "alaw" | "pcma" => Self::G711Alaw,
            _ => return None,
        })
    }

    /// The canonical name, as responses report it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::PcmI16 => "pcm_i16",
            Self::PcmF32 => "pcm_f32",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::G711Ulaw => "g711_ulaw",
            Self::G711Alaw => "g711_alaw",
        }
    }

    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::PcmI16 | Self::PcmF32 => "application/octet-stream",
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/ogg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::G711Ulaw => "audio/basic",
            Self::G711Alaw => "audio/x-alaw-basic",
        }
    }

    /// The usual file extension.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::PcmI16 => "pcm",
            Self::PcmF32 => "f32",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::G711Ulaw => "ulaw",
            Self::G711Alaw => "alaw",
        }
    }

    /// Whether this build encodes the format.
    pub const fn is_available(self) -> bool {
        match self {
            Self::Mp3 => cfg!(feature = "mp3"),
            Self::Opus => cfg!(feature = "opus"),
            Self::Aac => false,
            Self::Wav
            | Self::PcmI16
            | Self::PcmF32
            | Self::Flac
            | Self::G711Ulaw
            | Self::G711Alaw => true,
        }
    }

    /// The formats this build encodes.
    pub fn available() -> Vec<AudioFormat> {
        Self::ALL
            .into_iter()
            .filter(|format| format.is_available())
            .collect()
    }

    /// Whether the format takes a bitrate (the lossy codecs).
    pub const fn takes_bitrate(self) -> bool {
        matches!(self, Self::Mp3 | Self::Opus | Self::Aac)
    }

    /// The rate this format's audio goes out at, for audio at `input_rate`
    /// and an optional `requested` rate. A requested rate the format cannot
    /// carry is an error; without one, the input's rate is kept when the
    /// format carries it, else the nearest rate at or above it.
    pub fn output_rate(self, input_rate: u32, requested: Option<u32>) -> Result<u32, CodecError> {
        if input_rate == 0 {
            return Err(CodecError::InvalidOption(
                "The audio's sample rate is unknown (0 Hz)".to_string(),
            ));
        }
        let pick = |rates: &[u32]| -> Result<u32, CodecError> {
            if let Some(rate) = requested {
                return if rates.contains(&rate) {
                    Ok(rate)
                } else {
                    Err(CodecError::InvalidOption(format!(
                        "sample_rate {rate} is not available for {}; use one of {}",
                        self.as_str(),
                        join_rates(rates)
                    )))
                };
            }
            Ok(rates
                .iter()
                .copied()
                .find(|rate| *rate >= input_rate)
                .unwrap_or(*rates.last().expect("rates")))
        };
        match self {
            Self::Aac => Err(CodecError::unsupported(self)),
            Self::Mp3 => pick(&MP3_RATES),
            Self::Opus => pick(&OPUS_RATES),
            Self::G711Ulaw | Self::G711Alaw => pick(&[g711::SAMPLE_RATE]),
            Self::Wav | Self::PcmI16 | Self::PcmF32 | Self::Flac => match requested {
                Some(rate) if PCM_RATE_RANGE.contains(&rate) => Ok(rate),
                Some(rate) => Err(CodecError::InvalidOption(format!(
                    "sample_rate {rate} is out of range ({} to {} Hz)",
                    PCM_RATE_RANGE.start(),
                    PCM_RATE_RANGE.end()
                ))),
                None => Ok(input_rate),
            },
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn join_rates(rates: &[u32]) -> String {
    rates
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether the whole audio is known before the encoder writes (a file) or it
/// arrives while bytes go out (a stream).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Framing {
    /// Bytes go out as audio arrives. Headers that need totals say
    /// "unknown": WAV sizes are `0xFFFFFFFF`, FLAC's STREAMINFO has no total
    /// or MD5, and MP3 has no LAME tag.
    #[default]
    Stream,
    /// Everything goes out at [`StreamEncoder::finish`], with exact headers.
    File,
}

/// What to encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderOptions {
    pub format: AudioFormat,
    /// The rate of the PCM the encoder is given.
    pub input_rate: u32,
    /// The rate to encode at; resampled when it differs from `input_rate`.
    /// `None` picks per [`AudioFormat::output_rate`].
    pub output_rate: Option<u32>,
    /// Target bitrate in bits a second for the lossy codecs; `None` picks a
    /// speech default for the rate.
    pub bitrate: Option<u32>,
    pub framing: Framing,
}

impl EncoderOptions {
    pub fn new(format: AudioFormat, input_rate: u32) -> Self {
        Self {
            format,
            input_rate,
            output_rate: None,
            bitrate: None,
            framing: Framing::Stream,
        }
    }

    pub fn with_output_rate(mut self, rate: Option<u32>) -> Self {
        self.output_rate = rate;
        self
    }

    pub fn with_bitrate(mut self, bitrate: Option<u32>) -> Self {
        self.bitrate = bitrate;
        self
    }

    pub fn with_framing(mut self, framing: Framing) -> Self {
        self.framing = framing;
        self
    }

    /// Checks the options without building an encoder, so a request can be
    /// refused before any work starts. Returns the output rate.
    pub fn validate(&self) -> Result<u32, CodecError> {
        if !self.format.is_available() {
            return Err(CodecError::unsupported(self.format));
        }
        let rate = self.format.output_rate(self.input_rate, self.output_rate)?;
        if let Some(bitrate) = self.bitrate {
            if !self.format.takes_bitrate() {
                return Err(CodecError::InvalidOption(format!(
                    "bitrate applies to mp3 and opus only, not {}",
                    self.format
                )));
            }
            match self.format {
                #[cfg(feature = "mp3")]
                AudioFormat::Mp3 => {
                    mp3::bitrate_for(rate, Some(bitrate))?;
                }
                #[cfg(feature = "opus")]
                AudioFormat::Opus => {
                    opus::check_bitrate(bitrate)?;
                }
                _ => {}
            }
        }
        Ok(rate)
    }
}

/// An encoder fed mono PCM in pieces.
pub trait StreamEncoder: Send {
    /// The bytes ready after `pcm` (possibly none: codecs work in frames).
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError>;
    /// The remaining bytes: held audio, padding and trailers. Call once; the
    /// encoder takes no more audio after it.
    fn finish(&mut self) -> Result<Vec<u8>, CodecError>;
    fn format(&self) -> AudioFormat;
    /// The rate the encoded audio plays at.
    fn output_rate(&self) -> u32;
    fn content_type(&self) -> &'static str {
        self.format().content_type()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CodecError {
    /// The format is not built into this binary.
    #[error("{0}")]
    Unsupported(String),
    /// A rate, bitrate or other option the format cannot take.
    #[error("{0}")]
    InvalidOption(String),
    /// The encoder failed.
    #[error("{0}")]
    Encode(String),
}

impl CodecError {
    fn unsupported(format: AudioFormat) -> Self {
        let reason = match format {
            AudioFormat::Aac => {
                "aac is not available: AAC encoders need a patent licence. Use mp3 or opus."
                    .to_string()
            }
            other => format!("{other} encoding is not built into this server"),
        };
        Self::Unsupported(reason)
    }

    pub fn is_client_error(&self) -> bool {
        matches!(self, Self::Unsupported(_) | Self::InvalidOption(_))
    }
}

/// An encoder for `options`, resampling first when the output rate differs
/// from the input's.
pub fn stream_encoder(options: &EncoderOptions) -> Result<Box<dyn StreamEncoder>, CodecError> {
    let rate = options.validate()?;
    let encoder: Box<dyn StreamEncoder> = match options.format {
        AudioFormat::Wav => Box::new(pcm::WavEncoder::new(rate, options.framing)),
        AudioFormat::PcmI16
        | AudioFormat::PcmF32
        | AudioFormat::G711Ulaw
        | AudioFormat::G711Alaw => Box::new(pcm::RawEncoder::new(options.format, rate)),
        AudioFormat::Flac => Box::new(flac::FlacEncoder::new(rate, options.framing)?),
        #[cfg(feature = "mp3")]
        AudioFormat::Mp3 => Box::new(mp3::Mp3Encoder::new(
            rate,
            options.bitrate,
            options.framing,
        )?),
        #[cfg(feature = "opus")]
        AudioFormat::Opus => Box::new(opus::OggOpusEncoder::new(
            rate,
            options.input_rate,
            options.bitrate,
        )?),
        other => return Err(CodecError::unsupported(other)),
    };
    if rate == options.input_rate {
        return Ok(encoder);
    }
    Ok(Box::new(Resampled {
        resampler: StreamResampler::new(options.input_rate, rate)?,
        inner: encoder,
    }))
}

/// The whole of `pcm` as one file of `options.format` (its framing is
/// ignored: files always get exact headers).
pub fn encode_file(options: &EncoderOptions, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
    let mut encoder = stream_encoder(&options.with_framing(Framing::File))?;
    let mut bytes = encoder.push(pcm)?;
    bytes.extend(encoder.finish()?);
    Ok(bytes)
}

struct Resampled {
    resampler: StreamResampler,
    inner: Box<dyn StreamEncoder>,
}

impl StreamEncoder for Resampled {
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        let resampled = self.resampler.push(pcm)?;
        self.inner.push(&resampled)
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        let tail = self.resampler.finish()?;
        let mut bytes = self.inner.push(&tail)?;
        bytes.extend(self.inner.finish()?);
        Ok(bytes)
    }

    fn format(&self) -> AudioFormat {
        self.inner.format()
    }

    fn output_rate(&self) -> u32 {
        self.inner.output_rate()
    }
}

/// `pcm` as headerless 16-bit little-endian PCM, converted as every Izwi
/// encoder converts it.
pub fn pcm16_le(pcm: &[f32]) -> Vec<u8> {
    pcm.iter()
        .flat_map(|&sample| to_i16(sample).to_le_bytes())
        .collect()
}

/// `sample` as 16-bit PCM, as every Izwi encoder has always converted it
/// (clamped, scaled by 32767, truncated; NaN becomes 0).
#[inline]
pub(crate) fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 32_767.0) as i16
}

/// `sample` clamped to [-1, 1] with non-finite values silenced, for codecs
/// that take float input.
#[inline]
pub(crate) fn sanitize(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests;
