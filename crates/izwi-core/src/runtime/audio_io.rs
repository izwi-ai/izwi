//! Audio decode and preprocessing helpers used by runtime task handlers.

use std::io::Cursor;

use tracing::debug;

use crate::audio::AudioSourceMetadata;
use crate::error::{Error, Result};

const MIB: usize = 1024 * 1024;
pub(crate) const MAX_AUDIO_SOURCE_BYTES: usize = 128 * MIB;
const MAX_DECODED_AUDIO_BYTES: usize = 256 * MIB;
const MAX_AUDIO_DURATION_SECONDS: u64 = 60 * 60;
const MAX_AUDIO_SAMPLE_RATE: u32 = 384_000;
const MAX_AUDIO_CHANNELS: u16 = 32;
pub(crate) const MAX_REFERENCE_SOURCE_BYTES: usize = 32 * MIB;
const MAX_REFERENCE_DECODED_BYTES: usize = 32 * MIB;
const MAX_REFERENCE_DURATION_SECONDS: u64 = 30;
const MAX_BASE64_AUDIO_METADATA_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy)]
struct AudioDecodeLimits {
    max_source_bytes: usize,
    max_decoded_bytes: usize,
    max_duration_seconds: u64,
    max_sample_rate: u32,
    max_channels: u16,
}

impl AudioDecodeLimits {
    const fn inference() -> Self {
        Self {
            max_source_bytes: MAX_AUDIO_SOURCE_BYTES,
            max_decoded_bytes: MAX_DECODED_AUDIO_BYTES,
            max_duration_seconds: MAX_AUDIO_DURATION_SECONDS,
            max_sample_rate: MAX_AUDIO_SAMPLE_RATE,
            max_channels: MAX_AUDIO_CHANNELS,
        }
    }

    const fn reference() -> Self {
        Self {
            max_source_bytes: MAX_REFERENCE_SOURCE_BYTES,
            max_decoded_bytes: MAX_REFERENCE_DECODED_BYTES,
            max_duration_seconds: MAX_REFERENCE_DURATION_SECONDS,
            max_sample_rate: MAX_AUDIO_SAMPLE_RATE,
            max_channels: MAX_AUDIO_CHANNELS,
        }
    }

    fn validate_source_len(self, source_len: usize) -> Result<()> {
        if source_len > self.max_source_bytes {
            return Err(Error::InvalidInput(format!(
                "Encoded audio is {source_len} bytes, exceeding the {}-byte limit",
                self.max_source_bytes
            )));
        }
        Ok(())
    }

    fn validate_format(self, sample_rate: u32, channels: u16) -> Result<()> {
        if sample_rate == 0 || sample_rate > self.max_sample_rate {
            return Err(Error::InvalidInput(format!(
                "Audio sample rate {sample_rate} Hz is outside the supported 1..={} Hz range",
                self.max_sample_rate
            )));
        }
        if channels == 0 || channels > self.max_channels {
            return Err(Error::InvalidInput(format!(
                "Audio channel count {channels} is outside the supported 1..={} range",
                self.max_channels
            )));
        }
        Ok(())
    }

    fn max_mono_samples(self, sample_rate: u32) -> Result<usize> {
        self.validate_format(sample_rate, 1)?;
        let duration_samples = u64::from(sample_rate)
            .checked_mul(self.max_duration_seconds)
            .ok_or_else(|| Error::InvalidInput("Audio duration limit overflowed".to_string()))?;
        let byte_samples = self.max_decoded_bytes / std::mem::size_of::<f32>();
        Ok(usize::try_from(duration_samples)
            .unwrap_or(usize::MAX)
            .min(byte_samples))
    }

    fn validate_mono_samples(self, samples: usize, sample_rate: u32) -> Result<()> {
        let limit = self.max_mono_samples(sample_rate)?;
        if samples > limit {
            return Err(Error::InvalidInput(format!(
                "Decoded audio would contain {samples} mono samples at {sample_rate} Hz, exceeding the {limit}-sample production limit"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeErrorMode {
    Permissive,
    Strict,
}

impl DecodeErrorMode {
    fn is_strict(self) -> bool {
        self == Self::Strict
    }
}

pub(crate) fn base64_decode(data: &str) -> Result<Vec<u8>> {
    base64_decode_with_limit(data, MAX_AUDIO_SOURCE_BYTES)
}

fn base64_decode_with_limit(data: &str, max_decoded_bytes: usize) -> Result<Vec<u8>> {
    use base64::Engine;

    let (payload, encoded_len) = validated_base64_audio_payload(data, max_decoded_bytes)?;

    let decoded = if !payload.as_bytes().iter().any(u8::is_ascii_whitespace) {
        base64::engine::general_purpose::STANDARD.decode(payload.as_bytes())
    } else {
        // Reserve for the actual base64 alphabet, not the attacker-controlled
        // raw string length. In particular, a payload containing a small amount
        // of base64 surrounded by arbitrary whitespace must not cause a second
        // allocation proportional to the whitespace.
        let mut normalized = Vec::new();
        normalized.try_reserve_exact(encoded_len).map_err(|_| {
            Error::Overloaded("Unable to reserve bounded base64 audio input".to_string())
        })?;
        normalized.extend(
            payload
                .as_bytes()
                .iter()
                .copied()
                .filter(|byte| !byte.is_ascii_whitespace()),
        );
        base64::engine::general_purpose::STANDARD.decode(&normalized)
    }
    .map_err(|e| Error::InvalidInput(format!("Base64 audio decode error: {e}")))?;
    if decoded.len() > max_decoded_bytes {
        return Err(Error::InvalidInput(format!(
            "Decoded base64 audio is {} bytes, exceeding the {max_decoded_bytes}-byte limit",
            decoded.len()
        )));
    }
    Ok(decoded)
}

/// Validate the decoded upper bound of a base64 audio payload without
/// allocating or decoding it. Direct engine admission uses the same source
/// contract as the eventual decoder before it moves or clones request data.
pub(crate) fn validate_base64_audio_source_size(
    data: &str,
    max_decoded_bytes: usize,
) -> Result<()> {
    validated_base64_audio_payload(data, max_decoded_bytes).map(|_| ())
}

/// Bound both the retained string and its decoded upper size before a direct
/// request copies or moves base64 audio. The fixed allowance covers ordinary
/// data-URI metadata and modest formatting whitespace without permitting those
/// non-payload bytes to amplify retained memory without limit.
pub(crate) fn validate_base64_audio_source_input(
    data: &str,
    max_decoded_bytes: usize,
) -> Result<()> {
    validate_base64_audio_retained_size(data.len(), max_decoded_bytes)?;
    validate_base64_audio_source_size(data, max_decoded_bytes)
}

/// O(1) retained-input guard suitable for async admission paths. Callers can
/// apply it before admission, then run the decoded-upper-bound scan only after
/// copying work is covered by a lease and moved to a blocking boundary.
pub(crate) fn validate_base64_audio_retained_size(
    retained_bytes: usize,
    max_decoded_bytes: usize,
) -> Result<()> {
    if retained_bytes == 0 {
        return Err(Error::InvalidInput(
            "Base64 audio input cannot be empty".to_string(),
        ));
    }
    let max_encoded_bytes = max_decoded_bytes
        .checked_add(2)
        .and_then(|value| value.checked_div(3))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| Error::InvalidInput("Base64 audio input limit overflow".to_string()))?;
    let retained_limit = max_encoded_bytes
        .checked_add(MAX_BASE64_AUDIO_METADATA_BYTES)
        .ok_or_else(|| Error::InvalidInput("Base64 audio input limit overflow".to_string()))?;
    if retained_bytes > retained_limit {
        return Err(Error::InvalidInput(format!(
            "Base64 audio input retains {} bytes, exceeding the {retained_limit}-byte encoded input limit",
            retained_bytes
        )));
    }
    Ok(())
}

fn validated_base64_audio_payload(data: &str, max_decoded_bytes: usize) -> Result<(&str, usize)> {
    let payload = if data.starts_with("data:") {
        data.split_once(',').map(|(_, b64)| b64).unwrap_or(data)
    } else {
        data
    };

    let encoded_len = payload
        .as_bytes()
        .iter()
        .filter(|byte| !byte.is_ascii_whitespace())
        .count();
    let decoded_upper_bound = encoded_len
        .checked_add(3)
        .and_then(|value| value.checked_div(4))
        .and_then(|groups| groups.checked_mul(3))
        .ok_or_else(|| Error::InvalidInput("Base64 audio size overflowed".to_string()))?;
    if decoded_upper_bound > max_decoded_bytes.saturating_add(2) {
        return Err(Error::InvalidInput(format!(
            "Base64 audio may decode to {decoded_upper_bound} bytes, exceeding the {max_decoded_bytes}-byte limit"
        )));
    }
    Ok((payload, encoded_len))
}

pub(crate) fn decode_reference_audio_base64(data: &str) -> Result<(Vec<f32>, u32)> {
    let bytes = base64_decode_with_limit(data, MAX_REFERENCE_SOURCE_BYTES)?;
    decode_reference_audio_bytes(&bytes)
}

pub(crate) fn decode_audio_bytes_with_metadata(
    audio_bytes: &[u8],
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    decode_audio_bytes_with_metadata_and_limits(
        audio_bytes,
        DecodeErrorMode::Strict,
        AudioDecodeLimits::inference(),
    )
}

fn decode_audio_bytes_with_metadata_and_limits(
    audio_bytes: &[u8],
    error_mode: DecodeErrorMode,
    limits: AudioDecodeLimits,
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    if audio_bytes.is_empty() {
        return Err(Error::InvalidInput("Empty audio input".to_string()));
    }
    limits.validate_source_len(audio_bytes.len())?;

    if is_riff_wave(audio_bytes) {
        if error_mode.is_strict() {
            validate_riff_wave_structure(audio_bytes).map_err(|err| {
                Error::InferenceError(format!("Failed to decode WAV strictly: {err}"))
            })?;
        }
        match decode_wav_bytes_with_metadata(audio_bytes, error_mode, limits) {
            Ok((samples, source)) => {
                return finalize_decoded_audio_with_metadata(samples, source, limits);
            }
            Err(wav_err) => {
                return match decode_audio_bytes_symphonia_with_metadata(
                    audio_bytes,
                    error_mode,
                    limits,
                ) {
                    Ok((samples, source)) => {
                        finalize_decoded_audio_with_metadata(samples, source, limits)
                    }
                    Err(symphonia_err) => Err(Error::InferenceError(format!(
                        "Failed to decode WAV strictly. WAV path: {wav_err}; Symphonia: {symphonia_err}"
                    ))),
                };
            }
        }
    }

    match decode_audio_bytes_symphonia_with_metadata(audio_bytes, error_mode, limits) {
        Ok((samples, source)) => finalize_decoded_audio_with_metadata(samples, source, limits),
        Err(symphonia_err) => {
            let (samples, source) = decode_wav_bytes_hound_with_metadata(
                audio_bytes,
                error_mode,
                limits,
            )
            .map_err(|wav_err| {
                Error::InferenceError(format!(
                            "Failed to decode audio strictly. Symphonia: {symphonia_err}; WAV fallback: {wav_err}"
                        ))
            })?;
            finalize_decoded_audio_with_metadata(samples, source, limits)
        }
    }
}

pub(crate) fn decode_audio_bytes(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_audio_bytes_with_metadata_and_limits(
        audio_bytes,
        DecodeErrorMode::Permissive,
        AudioDecodeLimits::inference(),
    )
    .map(|(samples, source)| (samples, source.sample_rate))
}

pub(crate) fn decode_reference_audio_bytes(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_audio_bytes_with_metadata_and_limits(
        audio_bytes,
        DecodeErrorMode::Permissive,
        AudioDecodeLimits::reference(),
    )
    .map(|(samples, source)| (samples, source.sample_rate))
}

pub(crate) fn decode_wav_bytes(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_audio_bytes(wav_bytes)
}

fn is_riff_wave(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

fn validate_riff_wave_structure(wav_bytes: &[u8]) -> Result<()> {
    let mut offset = 12usize;
    while offset < wav_bytes.len() {
        if offset.saturating_add(8) > wav_bytes.len() {
            return Err(Error::InferenceError(format!(
                "truncated WAV chunk header at byte {offset}"
            )));
        }
        let chunk_size = u32::from_le_bytes([
            wav_bytes[offset + 4],
            wav_bytes[offset + 5],
            wav_bytes[offset + 6],
            wav_bytes[offset + 7],
        ]) as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start.checked_add(chunk_size).ok_or_else(|| {
            Error::InferenceError(format!(
                "WAV chunk at byte {offset} exceeds addressable size"
            ))
        })?;
        if chunk_end > wav_bytes.len() {
            return Err(Error::InferenceError(format!(
                "truncated WAV chunk at byte {offset}: declared {chunk_size} bytes, only {} remain",
                wav_bytes.len().saturating_sub(chunk_start)
            )));
        }
        let padded_end = chunk_end.checked_add(chunk_size & 1).ok_or_else(|| {
            Error::InferenceError(format!("WAV chunk padding at byte {offset} overflowed"))
        })?;
        if padded_end > wav_bytes.len() {
            return Err(Error::InferenceError(format!(
                "truncated WAV padding after chunk at byte {offset}"
            )));
        }
        offset = padded_end;
    }
    Ok(())
}

pub(crate) fn wav_duration_seconds_fast(wav_bytes: &[u8]) -> Option<f32> {
    if !is_riff_wave(wav_bytes) {
        return None;
    }

    let mut offset = 12usize;
    let mut sample_rate = None;
    let mut block_align = None;
    let mut data_len = None;

    while offset.saturating_add(8) <= wav_bytes.len() {
        let chunk_id = &wav_bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            wav_bytes[offset + 4],
            wav_bytes[offset + 5],
            wav_bytes[offset + 6],
            wav_bytes[offset + 7],
        ]) as usize;
        let chunk_start = offset + 8;
        let Some(chunk_end) = chunk_start.checked_add(chunk_size) else {
            break;
        };
        if chunk_end > wav_bytes.len() {
            break;
        }

        match chunk_id {
            b"fmt " if chunk_size >= 16 => {
                sample_rate = Some(u32::from_le_bytes([
                    wav_bytes[chunk_start + 4],
                    wav_bytes[chunk_start + 5],
                    wav_bytes[chunk_start + 6],
                    wav_bytes[chunk_start + 7],
                ]));
                block_align = Some(u16::from_le_bytes([
                    wav_bytes[chunk_start + 12],
                    wav_bytes[chunk_start + 13],
                ]));
            }
            b"data" => data_len = Some(chunk_size),
            _ => {}
        }

        let padded = chunk_end + (chunk_size & 1);
        if padded <= offset {
            break;
        }
        offset = padded;
    }

    let sample_rate = sample_rate?;
    let block_align = usize::from(block_align?);
    let data_len = data_len?;
    if sample_rate == 0 || block_align == 0 {
        return None;
    }
    let frames = data_len / block_align;
    Some(frames as f32 / sample_rate as f32)
}

fn decode_wav_bytes_fast(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_wav_bytes_with_metadata(
        wav_bytes,
        DecodeErrorMode::Permissive,
        AudioDecodeLimits::inference(),
    )
    .map(|(samples, source)| (samples, source.sample_rate))
}

fn decode_wav_bytes_with_metadata(
    wav_bytes: &[u8],
    error_mode: DecodeErrorMode,
    limits: AudioDecodeLimits,
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    decode_wav_pcm16_mono_with_metadata(wav_bytes, error_mode, limits)
        .or_else(|_| decode_wav_bytes_hound_with_metadata(wav_bytes, error_mode, limits))
}

fn decode_wav_pcm16_mono_with_metadata(
    wav_bytes: &[u8],
    error_mode: DecodeErrorMode,
    limits: AudioDecodeLimits,
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    let mut offset = 12usize;
    let mut audio_format = None;
    let mut channels = None;
    let mut sample_rate = None;
    let mut block_align = None;
    let mut bits_per_sample = None;
    let mut data_range = None;

    while offset.saturating_add(8) <= wav_bytes.len() {
        let chunk_id = &wav_bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            wav_bytes[offset + 4],
            wav_bytes[offset + 5],
            wav_bytes[offset + 6],
            wav_bytes[offset + 7],
        ]) as usize;
        let chunk_start = offset + 8;
        let Some(chunk_end) = chunk_start.checked_add(chunk_size) else {
            break;
        };
        if chunk_end > wav_bytes.len() {
            break;
        }

        match chunk_id {
            b"fmt " if chunk_size >= 16 => {
                audio_format = Some(u16::from_le_bytes([
                    wav_bytes[chunk_start],
                    wav_bytes[chunk_start + 1],
                ]));
                channels = Some(u16::from_le_bytes([
                    wav_bytes[chunk_start + 2],
                    wav_bytes[chunk_start + 3],
                ]));
                sample_rate = Some(u32::from_le_bytes([
                    wav_bytes[chunk_start + 4],
                    wav_bytes[chunk_start + 5],
                    wav_bytes[chunk_start + 6],
                    wav_bytes[chunk_start + 7],
                ]));
                block_align = Some(u16::from_le_bytes([
                    wav_bytes[chunk_start + 12],
                    wav_bytes[chunk_start + 13],
                ]));
                bits_per_sample = Some(u16::from_le_bytes([
                    wav_bytes[chunk_start + 14],
                    wav_bytes[chunk_start + 15],
                ]));
            }
            b"data" => data_range = Some(chunk_start..chunk_end),
            _ => {}
        }

        let padded = chunk_end + (chunk_size & 1);
        if padded <= offset {
            break;
        }
        offset = padded;
    }

    let audio_format =
        audio_format.ok_or_else(|| Error::InferenceError("WAV missing fmt chunk".to_string()))?;
    let channels =
        channels.ok_or_else(|| Error::InferenceError("WAV missing channel count".to_string()))?;
    let sample_rate =
        sample_rate.ok_or_else(|| Error::InferenceError("WAV missing sample rate".to_string()))?;
    let block_align =
        block_align.ok_or_else(|| Error::InferenceError("WAV missing block align".to_string()))?;
    let bits_per_sample = bits_per_sample
        .ok_or_else(|| Error::InferenceError("WAV missing bits per sample".to_string()))?;
    let data_range =
        data_range.ok_or_else(|| Error::InferenceError("WAV missing data chunk".to_string()))?;

    if audio_format != 1 || channels == 0 || sample_rate == 0 || bits_per_sample != 16 {
        return Err(Error::InferenceError(
            "WAV fast path only supports PCM16 audio".to_string(),
        ));
    }
    limits.validate_format(sample_rate, channels)?;
    let source_channel_count = channels;
    let channels = channels as usize;
    let block_align = block_align as usize;
    if block_align != channels * 2 {
        return Err(Error::InferenceError(format!(
            "Unsupported PCM16 WAV block alignment: {block_align}"
        )));
    }

    let data = &wav_bytes[data_range];
    if error_mode.is_strict() && data.len() % block_align != 0 {
        return Err(Error::InferenceError(format!(
            "WAV PCM16 data length {} is not aligned to {block_align}-byte frames",
            data.len()
        )));
    }
    let frame_count = data.len() / block_align;
    if frame_count == 0 {
        return Err(Error::InferenceError(
            "Decoded audio produced zero samples".to_string(),
        ));
    }
    limits.validate_mono_samples(frame_count, sample_rate)?;

    let mut samples = Vec::with_capacity(frame_count);
    if channels == 1 {
        for bytes in data[..frame_count * block_align].chunks_exact(2) {
            let sample = i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32767.0;
            samples.push(sample.clamp(-1.0, 1.0));
        }
    } else {
        for frame in data[..frame_count * block_align].chunks_exact(block_align) {
            let mut sum = 0.0f32;
            for channel in 0..channels {
                let idx = channel * 2;
                sum += i16::from_le_bytes([frame[idx], frame[idx + 1]]) as f32;
            }
            samples.push((sum / channels as f32 / 32767.0).clamp(-1.0, 1.0));
        }
    }

    Ok((
        samples,
        AudioSourceMetadata {
            container: "wav".to_string(),
            codec: "pcm_s16le".to_string(),
            sample_rate,
            channel_count: source_channel_count,
        },
    ))
}

fn decode_audio_bytes_symphonia(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_audio_bytes_symphonia_with_metadata(
        audio_bytes,
        DecodeErrorMode::Permissive,
        AudioDecodeLimits::inference(),
    )
    .map(|(samples, source)| (samples, source.sample_rate))
}

fn decode_audio_bytes_symphonia_with_metadata(
    audio_bytes: &[u8],
    error_mode: DecodeErrorMode,
    limits: AudioDecodeLimits,
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use symphonia::default::{get_codecs, get_probe};

    let container = detect_audio_container(audio_bytes).to_string();
    let media_source = MediaSourceStream::new(
        Box::new(Cursor::new(audio_bytes.to_vec())),
        Default::default(),
    );
    let hint = Hint::new();
    let probed = get_probe()
        .format(
            &hint,
            media_source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| {
            Error::InferenceError(format!(
                "Symphonia probe failed for container={container}: {e}"
            ))
        })?;

    let mut format = probed.format;
    let (track_id, codec_params) = {
        let track = format
            .default_track()
            .ok_or_else(|| Error::InferenceError("No default audio track found".to_string()))?;
        (track.id, track.codec_params.clone())
    };
    let codec = get_codecs()
        .get_codec(codec_params.codec)
        .map(|descriptor| descriptor.short_name.to_string())
        .unwrap_or_else(|| format!("{:?}", codec_params.codec));
    let mut sample_rate = codec_params.sample_rate.unwrap_or(0);
    let mut channel_count = codec_params
        .channels
        .map(|channels| u16::try_from(channels.count()).unwrap_or(u16::MAX))
        .unwrap_or(0);
    if sample_rate > 0 && channel_count > 0 {
        limits.validate_format(sample_rate, channel_count)?;
    }
    let mut decoder = get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| Error::InferenceError(format!("Failed to create audio decoder: {e}")))?;

    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(Error::InferenceError(
                    "Audio stream format reset is not supported".to_string(),
                ));
            }
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::IoError(_)) if !error_mode.is_strict() => break,
            Err(SymphoniaError::IoError(err)) => {
                return Err(Error::InferenceError(format!(
                    "Failed reading {container}/{codec} audio packets: {err}"
                )));
            }
            Err(err) => {
                return Err(Error::InferenceError(format!(
                    "Failed reading audio packets: {err}"
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) if !error_mode.is_strict() => continue,
            Err(SymphoniaError::DecodeError(err)) => {
                return Err(Error::InferenceError(format!(
                    "Failed decoding {container}/{codec} audio packet ({channel_count} channels at {sample_rate} Hz): {err}"
                )));
            }
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(Error::InferenceError(
                    "Audio decoder reset is not supported".to_string(),
                ));
            }
            Err(err) => {
                return Err(Error::InferenceError(format!(
                    "Failed decoding audio packet: {err}"
                )));
            }
        };

        if sample_rate == 0 {
            sample_rate = decoded.spec().rate;
        } else if error_mode.is_strict() && sample_rate != decoded.spec().rate {
            return Err(Error::InferenceError(format!(
                "Audio sample rate changed while decoding {container}/{codec} ({sample_rate} -> {})",
                decoded.spec().rate
            )));
        }
        let channels = decoded.spec().channels.count().max(1);
        let decoded_channel_count = u16::try_from(channels).unwrap_or(u16::MAX);
        if channel_count == 0 {
            channel_count = decoded_channel_count;
        } else if error_mode.is_strict() && channel_count != decoded_channel_count {
            return Err(Error::InferenceError(format!(
                "Audio channel count changed while decoding {container}/{codec} ({channel_count} -> {decoded_channel_count})"
            )));
        }
        limits.validate_format(sample_rate, decoded_channel_count)?;
        append_decoded_packet(
            decoded,
            channels,
            &mut samples,
            limits.max_mono_samples(sample_rate)?,
            limits.max_decoded_bytes,
        )?;
    }

    if sample_rate == 0 {
        return Err(Error::InferenceError(
            "Decoded audio is missing sample rate metadata".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(Error::InferenceError(
            "Decoded audio produced zero samples".to_string(),
        ));
    }
    if channel_count == 0 {
        return Err(Error::InferenceError(
            "Decoded audio is missing channel metadata".to_string(),
        ));
    }

    Ok((
        samples,
        AudioSourceMetadata {
            container,
            codec,
            sample_rate,
            channel_count,
        },
    ))
}

fn append_decoded_packet(
    decoded: symphonia::core::audio::AudioBufferRef<'_>,
    channels: usize,
    out: &mut Vec<f32>,
    max_mono_samples: usize,
    max_decoded_bytes: usize,
) -> Result<()> {
    let next_len = out
        .len()
        .checked_add(decoded.frames())
        .ok_or_else(|| Error::InvalidInput("Decoded audio sample count overflowed".to_string()))?;
    if next_len > max_mono_samples {
        return Err(Error::InvalidInput(format!(
            "Decoded audio would exceed the {max_mono_samples}-sample production limit"
        )));
    }
    use symphonia::core::audio::AudioBufferRef;
    match decoded {
        AudioBufferRef::U8(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::U16(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::U24(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::U32(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::S8(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::S16(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::S24(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::S32(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::F32(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
        AudioBufferRef::F64(buffer) => {
            append_planar_packet(buffer.as_ref(), channels, out, max_decoded_bytes)
        }
    }
}

fn append_planar_packet<S>(
    decoded: &symphonia::core::audio::AudioBuffer<S>,
    channels: usize,
    out: &mut Vec<f32>,
    max_decoded_bytes: usize,
) -> Result<()>
where
    S: symphonia::core::sample::Sample + symphonia::core::conv::IntoSample<f32>,
{
    use symphonia::core::audio::Signal;
    use symphonia::core::conv::IntoSample;

    if channels == 0 || channels > decoded.spec().channels.count() {
        return Err(Error::InvalidInput(
            "Decoded audio packet has an invalid channel count".to_string(),
        ));
    }

    let packet_bytes = decoded
        .capacity()
        .checked_mul(channels)
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<S>()))
        .ok_or_else(|| Error::InvalidInput("Decoded audio packet size overflowed".to_string()))?;
    let output_bytes = out
        .len()
        .checked_add(decoded.frames())
        .ok_or_else(|| Error::InvalidInput("Decoded audio output size overflowed".to_string()))?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::InvalidInput("Decoded audio output size overflowed".to_string()))?;
    if packet_bytes
        .checked_add(output_bytes)
        .is_none_or(|peak_bytes| peak_bytes > max_decoded_bytes)
    {
        return Err(Error::InvalidInput(format!(
            "Decoded audio packet and output would exceed the {max_decoded_bytes}-byte production limit"
        )));
    }
    out.try_reserve(decoded.frames()).map_err(|_| {
        Error::Overloaded("Unable to reserve bounded decoded audio output".to_string())
    })?;

    if channels == 1 {
        out.extend(
            decoded
                .chan(0)
                .iter()
                .copied()
                .map(IntoSample::<f32>::into_sample),
        );
        return Ok(());
    }

    for frame in 0..decoded.frames() {
        let mut sum = 0.0f32;
        for channel in 0..channels {
            sum += decoded.chan(channel)[frame].into_sample();
        }
        out.push(sum / channels as f32);
    }
    Ok(())
}

fn decode_wav_bytes_hound(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_wav_bytes_hound_with_metadata(
        wav_bytes,
        DecodeErrorMode::Permissive,
        AudioDecodeLimits::inference(),
    )
    .map(|(samples, source)| (samples, source.sample_rate))
}

fn decode_wav_bytes_hound_with_metadata(
    wav_bytes: &[u8],
    error_mode: DecodeErrorMode,
    limits: AudioDecodeLimits,
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    let cursor = Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| Error::InferenceError(format!("Failed to parse WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    if error_mode.is_strict() && spec.channels == 0 {
        return Err(Error::InferenceError(
            "WAV source has zero channels".to_string(),
        ));
    }
    let source_channel_count = spec.channels.max(1);
    let channels = source_channel_count as usize;
    limits.validate_format(sample_rate, source_channel_count)?;
    let max_mono_samples = limits.max_mono_samples(sample_rate)?;
    let declared_frames = usize::try_from(reader.duration()).unwrap_or(usize::MAX);
    if declared_frames > max_mono_samples {
        return Err(Error::InvalidInput(format!(
            "Decoded WAV would exceed the {max_mono_samples}-sample production limit"
        )));
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.max(1) as u32;
            let max_val = if bits > 1 {
                ((1i64 << (bits - 1)) - 1) as f32
            } else {
                1.0
            };
            decode_hound_frames::<_, i32, _>(
                &mut reader,
                channels,
                declared_frames,
                error_mode,
                |sample| (sample as f32 / max_val).clamp(-1.0, 1.0),
            )?
        }
        hound::SampleFormat::Float => decode_hound_frames::<_, f32, _>(
            &mut reader,
            channels,
            declared_frames,
            error_mode,
            |sample| sample,
        )?,
    };

    Ok((
        samples,
        AudioSourceMetadata {
            container: "wav".to_string(),
            codec: hound_codec_name(spec.sample_format, spec.bits_per_sample),
            sample_rate,
            channel_count: source_channel_count,
        },
    ))
}

fn decode_hound_frames<R, S, F>(
    reader: &mut hound::WavReader<R>,
    channels: usize,
    declared_frames: usize,
    error_mode: DecodeErrorMode,
    mut convert: F,
) -> Result<Vec<f32>>
where
    R: std::io::Read,
    S: hound::Sample,
    F: FnMut(S) -> f32,
{
    // A WAV header may declare a large data chunk even when the underlying
    // source is truncated. Start with a small bounded allocation and let the
    // vector grow only as samples are successfully read.
    const INITIAL_OUTPUT_FRAMES: usize = 16 * 1024;
    let mut output = Vec::new();
    output
        .try_reserve_exact(declared_frames.min(INITIAL_OUTPUT_FRAMES))
        .map_err(|_| {
            Error::Overloaded("Unable to reserve bounded decoded WAV output".to_string())
        })?;
    let mut input = reader.samples::<S>();
    'frames: for _ in 0..declared_frames {
        let mut sum = 0.0f32;
        for _ in 0..channels {
            match input.next() {
                Some(Ok(sample)) => sum += convert(sample),
                Some(Err(err)) if error_mode.is_strict() => {
                    return Err(Error::InferenceError(format!(
                        "Failed decoding WAV sample: {err}"
                    )));
                }
                Some(Err(_)) => break 'frames,
                None if error_mode.is_strict() => {
                    return Err(Error::InferenceError(
                        "WAV ended before its declared sample count".to_string(),
                    ));
                }
                None => break 'frames,
            }
        }
        output.push((sum / channels as f32).clamp(-1.0, 1.0));
    }
    Ok(output)
}

fn hound_codec_name(sample_format: hound::SampleFormat, bits_per_sample: u16) -> String {
    match sample_format {
        hound::SampleFormat::Int => format!("pcm_s{bits_per_sample}le"),
        hound::SampleFormat::Float => format!("pcm_f{bits_per_sample}le"),
    }
}

fn detect_audio_container(audio_bytes: &[u8]) -> &'static str {
    if is_riff_wave(audio_bytes) {
        "wav"
    } else if audio_bytes.starts_with(b"fLaC") {
        "flac"
    } else if audio_bytes.starts_with(b"OggS") {
        "ogg"
    } else if audio_bytes.starts_with(b"ID3") {
        "mp3"
    } else if audio_bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        "matroska"
    } else if audio_bytes.len() >= 12 && &audio_bytes[4..8] == b"ftyp" {
        "mp4"
    } else if audio_bytes.starts_with(b"FORM")
        && audio_bytes
            .get(8..12)
            .is_some_and(|kind| kind == b"AIFF" || kind == b"AIFC")
    {
        "aiff"
    } else if audio_bytes.starts_with(b"caff") {
        "caf"
    } else if looks_like_adts_frame(audio_bytes) {
        "aac"
    } else if looks_like_mpeg_audio_frame(audio_bytes) {
        "mp3"
    } else {
        "unknown"
    }
}

fn looks_like_mpeg_audio_frame(audio_bytes: &[u8]) -> bool {
    audio_bytes.len() >= 2 && audio_bytes[0] == 0xff && audio_bytes[1] & 0xe0 == 0xe0
}

fn looks_like_adts_frame(audio_bytes: &[u8]) -> bool {
    audio_bytes.len() >= 2 && audio_bytes[0] == 0xff && audio_bytes[1] & 0xf6 == 0xf0
}

fn finalize_decoded_audio(
    mut samples: Vec<f32>,
    sample_rate: u32,
    limits: AudioDecodeLimits,
) -> Result<(Vec<f32>, u32)> {
    if sample_rate == 0 {
        return Err(Error::InferenceError(
            "Decoded audio has invalid sample rate 0".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(Error::InferenceError(
            "Decoded audio contains no samples".to_string(),
        ));
    }
    limits.validate_mono_samples(samples.len(), sample_rate)?;

    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        } else {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    Ok((samples, sample_rate))
}

fn finalize_decoded_audio_with_metadata(
    samples: Vec<f32>,
    source: AudioSourceMetadata,
    limits: AudioDecodeLimits,
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    if source.channel_count == 0 {
        return Err(Error::InferenceError(
            "Decoded audio has invalid source channel count 0".to_string(),
        ));
    }
    limits.validate_format(source.sample_rate, source.channel_count)?;
    let (samples, sample_rate) = finalize_decoded_audio(samples, source.sample_rate, limits)?;
    debug_assert_eq!(sample_rate, source.sample_rate);
    Ok((samples, source))
}

pub(crate) fn preprocess_reference_audio(mut samples: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    let original_len = samples.len();

    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }

    // Remove DC bias.
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    for sample in &mut samples {
        *sample -= mean;
    }

    let initial_peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if initial_peak < 1e-5 {
        return Vec::new();
    }

    // Trim leading/trailing silence while keeping short context margins.
    let silence_threshold = (initial_peak * 0.04).max(0.0025);
    let first_idx = samples.iter().position(|s| s.abs() >= silence_threshold);
    let last_idx = samples.iter().rposition(|s| s.abs() >= silence_threshold);
    if let (Some(first), Some(last)) = (first_idx, last_idx) {
        let margin = ((sample_rate as f32) * 0.12) as usize;
        let start = first.saturating_sub(margin);
        let end = (last + margin + 1).min(samples.len());
        samples = samples[start..end].to_vec();
    }

    // Bound reference length to avoid conditioning on long silence/noise tails.
    let max_seconds = 12usize;
    let max_len = sample_rate as usize * max_seconds;
    if samples.len() > max_len && max_len > 0 {
        let window = (sample_rate as usize * 6).clamp(sample_rate as usize, samples.len());
        let best_start = highest_energy_window_start(&samples, window);
        let start = best_start.min(samples.len() - max_len);
        samples = samples[start..start + max_len].to_vec();
    }

    // Normalize into a practical loudness band so encoder sees stable dynamics.
    let mut peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    let rms = (samples
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32;
    let min_rms = 0.035f32;
    if rms > 1e-6 && rms < min_rms {
        let gain = (min_rms / rms).min(6.0);
        for sample in &mut samples {
            *sample *= gain;
        }
    }

    // Final hard limit.
    peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    debug!(
        "Reference preprocessing: {} -> {} samples @ {} Hz",
        original_len,
        samples.len(),
        sample_rate
    );

    samples
}

fn highest_energy_window_start(samples: &[f32], window: usize) -> usize {
    if samples.is_empty() || window == 0 || samples.len() <= window {
        return 0;
    }

    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0f64);
    for &sample in samples {
        let e = (sample as f64) * (sample as f64);
        let next = prefix.last().copied().unwrap_or(0.0) + e;
        prefix.push(next);
    }

    let mut best_start = 0usize;
    let mut best_energy = f64::NEG_INFINITY;
    for start in 0..=samples.len() - window {
        let end = start + window;
        let energy = prefix[end] - prefix[start];
        if energy > best_energy {
            best_energy = energy;
            best_start = start;
        }
    }

    best_start
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[test]
    fn base64_decode_accepts_data_uri_payload() {
        let payload = [1u8, 2, 3, 4, 5];
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let uri = format!("data:audio/mpeg;base64,{b64}");
        let decoded = base64_decode(&uri).expect("data URI decode should succeed");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn base64_decode_rejects_payload_before_oversized_allocation() {
        let error = base64_decode_with_limit("AAAAAAAA", 3)
            .expect_err("six decoded bytes must exceed a three-byte limit");
        assert!(matches!(error, Error::InvalidInput(_)));
        assert!(error.to_string().contains("exceeding the 3-byte limit"));
    }

    #[test]
    fn base64_retained_input_guard_rejects_unbounded_non_payload_bytes_in_constant_time() {
        let error = validate_base64_audio_retained_size(4 + 1024 + 1, 3)
            .expect_err("retained whitespace/data-URI metadata must stay bounded");
        assert!(error.to_string().contains("encoded input limit"));
    }

    #[test]
    fn base64_decode_ignores_whitespace_without_changing_the_bounded_payload() {
        let decoded = base64_decode_with_limit(" \n\t A Q I D \r\n", 3)
            .expect("bounded whitespace normalization should decode");
        assert_eq!(decoded, vec![1, 2, 3]);
    }

    #[test]
    fn decoder_rejects_declared_audio_duration_before_sample_allocation() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            for _ in 0..8_001 {
                writer.write_sample(0_i16).expect("sample");
            }
            writer.finalize().expect("finalize");
        }
        let limits = AudioDecodeLimits {
            max_source_bytes: MIB,
            max_decoded_bytes: MIB,
            max_duration_seconds: 1,
            max_sample_rate: MAX_AUDIO_SAMPLE_RATE,
            max_channels: MAX_AUDIO_CHANNELS,
        };

        let error = decode_audio_bytes_with_metadata_and_limits(
            &wav_bytes,
            DecodeErrorMode::Permissive,
            limits,
        )
        .expect_err("duration above the configured bound must fail");
        assert!(matches!(
            error,
            Error::InferenceError(_) | Error::InvalidInput(_)
        ));
        assert!(error.to_string().contains("production limit"));
    }

    #[test]
    fn hound_downmixes_many_channels_without_materializing_interleaved_output() {
        let channels = 32;
        let frames = 64;
        let spec = hound::WavSpec {
            channels,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            for _ in 0..frames {
                for _ in 0..channels {
                    writer.write_sample(0.25f32).expect("sample");
                }
            }
            writer.finalize().expect("finalize");
        }
        let limits = AudioDecodeLimits {
            max_source_bytes: MIB,
            max_decoded_bytes: frames * std::mem::size_of::<f32>(),
            max_duration_seconds: 1,
            max_sample_rate: MAX_AUDIO_SAMPLE_RATE,
            max_channels: MAX_AUDIO_CHANNELS,
        };

        let (samples, source) =
            decode_wav_bytes_hound_with_metadata(&wav_bytes, DecodeErrorMode::Strict, limits)
                .expect("bounded planar downmix");
        assert_eq!(source.channel_count, channels);
        assert_eq!(samples.len(), frames);
        assert!(samples.iter().all(|sample| (*sample - 0.25).abs() < 1e-6));
    }

    #[test]
    fn permissive_hound_decode_stops_at_a_truncated_declared_payload() {
        let mut wav_bytes = Vec::new();
        wav_bytes.extend_from_slice(b"RIFF");
        wav_bytes.extend_from_slice(&4_000_036u32.to_le_bytes());
        wav_bytes.extend_from_slice(b"WAVEfmt ");
        wav_bytes.extend_from_slice(&16u32.to_le_bytes());
        wav_bytes.extend_from_slice(&1u16.to_le_bytes());
        wav_bytes.extend_from_slice(&1u16.to_le_bytes());
        wav_bytes.extend_from_slice(&16_000u32.to_le_bytes());
        wav_bytes.extend_from_slice(&32_000u32.to_le_bytes());
        wav_bytes.extend_from_slice(&2u16.to_le_bytes());
        wav_bytes.extend_from_slice(&16u16.to_le_bytes());
        wav_bytes.extend_from_slice(b"data");
        wav_bytes.extend_from_slice(&4_000_000u32.to_le_bytes());

        let (samples, _) = decode_wav_bytes_hound_with_metadata(
            &wav_bytes,
            DecodeErrorMode::Permissive,
            AudioDecodeLimits::inference(),
        )
        .expect("permissive decode should stop at the first truncated sample");
        assert!(samples.is_empty());
        assert!(samples.capacity() <= 16 * 1024);
    }

    #[test]
    fn decode_audio_bytes_downmixes_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.75f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.5f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.5f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let (samples, sample_rate) =
            decode_audio_bytes(&wav_bytes).expect("decode should succeed for WAV bytes");
        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!(
            (samples[0] - 0.5).abs() < 0.02,
            "first sample {}",
            samples[0]
        );
        assert!(samples[1].abs() < 0.02, "second sample {}", samples[1]);
    }

    #[test]
    fn strict_metadata_decode_reports_truncated_wav_samples() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            for sample in [1_i16, 2, 3, 4] {
                writer.write_sample(sample).expect("sample");
            }
            writer.finalize().expect("finalize");
        }
        wav_bytes.pop();

        let error = decode_audio_bytes_with_metadata(&wav_bytes)
            .expect_err("strict decode must reject a truncated WAV frame");

        assert!(error.to_string().contains("Failed to decode WAV strictly"));
    }

    #[test]
    fn wav_duration_seconds_fast_reads_data_duration_without_decoding() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            for _ in 0..32_000 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let duration = wav_duration_seconds_fast(&wav_bytes).expect("wav duration");

        assert!((duration - 1.0).abs() < 1e-6, "duration {duration}");
    }
}
