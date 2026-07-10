//! Audio decode and preprocessing helpers used by runtime task handlers.

use std::io::Cursor;

use tracing::debug;

use crate::audio::AudioSourceMetadata;
use crate::error::{Error, Result};

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
    use base64::Engine;

    let payload = if data.starts_with("data:") {
        data.split_once(',').map(|(_, b64)| b64).unwrap_or(data)
    } else {
        data
    };

    if !payload.as_bytes().iter().any(u8::is_ascii_whitespace) {
        return base64::engine::general_purpose::STANDARD
            .decode(payload.as_bytes())
            .map_err(|e| Error::InferenceError(format!("Base64 decode error: {}", e)));
    }

    let normalized: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(normalized.as_bytes())
        .map_err(|e| Error::InferenceError(format!("Base64 decode error: {}", e)))
}

pub(crate) fn decode_audio_bytes_with_metadata(
    audio_bytes: &[u8],
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    if audio_bytes.is_empty() {
        return Err(Error::InvalidInput("Empty audio input".to_string()));
    }

    if is_riff_wave(audio_bytes) {
        validate_riff_wave_structure(audio_bytes).map_err(|err| {
            Error::InferenceError(format!("Failed to decode WAV strictly: {err}"))
        })?;
        match decode_wav_bytes_with_metadata(audio_bytes, DecodeErrorMode::Strict) {
            Ok((samples, source)) => {
                return finalize_decoded_audio_with_metadata(samples, source);
            }
            Err(wav_err) => {
                return match decode_audio_bytes_symphonia_with_metadata(
                    audio_bytes,
                    DecodeErrorMode::Strict,
                ) {
                    Ok((samples, source)) => {
                        finalize_decoded_audio_with_metadata(samples, source)
                    }
                    Err(symphonia_err) => Err(Error::InferenceError(format!(
                        "Failed to decode WAV strictly. WAV path: {wav_err}; Symphonia: {symphonia_err}"
                    ))),
                };
            }
        }
    }

    match decode_audio_bytes_symphonia_with_metadata(audio_bytes, DecodeErrorMode::Strict) {
        Ok((samples, source)) => finalize_decoded_audio_with_metadata(samples, source),
        Err(symphonia_err) => {
            let (samples, source) =
                decode_wav_bytes_hound_with_metadata(audio_bytes, DecodeErrorMode::Strict).map_err(
                    |wav_err| {
                        Error::InferenceError(format!(
                            "Failed to decode audio strictly. Symphonia: {symphonia_err}; WAV fallback: {wav_err}"
                        ))
                    },
                )?;
            finalize_decoded_audio_with_metadata(samples, source)
        }
    }
}

pub(crate) fn decode_audio_bytes(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    if audio_bytes.is_empty() {
        return Err(Error::InvalidInput("Empty audio input".to_string()));
    }

    if is_riff_wave(audio_bytes) {
        match decode_wav_bytes_fast(audio_bytes) {
            Ok((samples, sample_rate)) => return finalize_decoded_audio(samples, sample_rate),
            Err(wav_err) => {
                return match decode_audio_bytes_symphonia(audio_bytes) {
                    Ok((samples, sample_rate)) => finalize_decoded_audio(samples, sample_rate),
                    Err(symphonia_err) => Err(Error::InferenceError(format!(
                        "Failed to decode WAV. WAV fast path: {wav_err}; Symphonia: {symphonia_err}"
                    ))),
                };
            }
        }
    }

    match decode_audio_bytes_symphonia(audio_bytes) {
        Ok((samples, sample_rate)) => finalize_decoded_audio(samples, sample_rate),
        Err(symphonia_err) => {
            let (samples, sample_rate) =
                decode_wav_bytes_hound(audio_bytes).map_err(|wav_err| {
                    Error::InferenceError(format!(
                    "Failed to decode audio. Symphonia: {symphonia_err}; WAV fallback: {wav_err}"
                ))
                })?;
            finalize_decoded_audio(samples, sample_rate)
        }
    }
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
    decode_wav_bytes_with_metadata(wav_bytes, DecodeErrorMode::Permissive)
        .map(|(samples, source)| (samples, source.sample_rate))
}

fn decode_wav_bytes_with_metadata(
    wav_bytes: &[u8],
    error_mode: DecodeErrorMode,
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    decode_wav_pcm16_mono_with_metadata(wav_bytes, error_mode)
        .or_else(|_| decode_wav_bytes_hound_with_metadata(wav_bytes, error_mode))
}

fn decode_wav_pcm16_mono_with_metadata(
    wav_bytes: &[u8],
    error_mode: DecodeErrorMode,
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
    decode_audio_bytes_symphonia_with_metadata(audio_bytes, DecodeErrorMode::Permissive)
        .map(|(samples, source)| (samples, source.sample_rate))
}

fn decode_audio_bytes_symphonia_with_metadata(
    audio_bytes: &[u8],
    error_mode: DecodeErrorMode,
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
        append_decoded_packet(decoded, channels, &mut samples);
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
) {
    use symphonia::core::audio::SampleBuffer;

    let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
    sample_buffer.copy_interleaved_ref(decoded);
    let data = sample_buffer.samples();

    if channels <= 1 {
        out.extend_from_slice(data);
        return;
    }

    for frame in data.chunks(channels) {
        if frame.is_empty() {
            continue;
        }
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum / frame.len() as f32);
    }
}

fn decode_wav_bytes_hound(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_wav_bytes_hound_with_metadata(wav_bytes, DecodeErrorMode::Permissive)
        .map(|(samples, source)| (samples, source.sample_rate))
}

fn decode_wav_bytes_hound_with_metadata(
    wav_bytes: &[u8],
    error_mode: DecodeErrorMode,
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

    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.max(1) as u32;
            let max_val = if bits > 1 {
                ((1i64 << (bits - 1)) - 1) as f32
            } else {
                1.0
            };
            if error_mode.is_strict() {
                reader
                    .samples::<i32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|err| {
                        Error::InferenceError(format!("Failed decoding WAV sample: {err}"))
                    })?
                    .into_iter()
                    .map(|sample| (sample as f32 / max_val).clamp(-1.0, 1.0))
                    .collect()
            } else {
                reader
                    .samples::<i32>()
                    .filter_map(|sample| sample.ok())
                    .map(|sample| (sample as f32 / max_val).clamp(-1.0, 1.0))
                    .collect()
            }
        }
        hound::SampleFormat::Float => {
            if error_mode.is_strict() {
                reader
                    .samples::<f32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|err| {
                        Error::InferenceError(format!("Failed decoding WAV sample: {err}"))
                    })?
            } else {
                reader
                    .samples::<f32>()
                    .filter_map(|sample| sample.ok())
                    .collect()
            }
        }
    };

    if error_mode.is_strict() && samples.len() % channels != 0 {
        return Err(Error::InferenceError(format!(
            "WAV sample count {} is not aligned to {channels} channels",
            samples.len()
        )));
    }

    if channels > 1 {
        let mut mono = Vec::with_capacity(samples.len() / channels + 1);
        for frame in samples.chunks(channels) {
            if frame.is_empty() {
                continue;
            }
            let sum: f32 = frame.iter().copied().sum();
            mono.push(sum / frame.len() as f32);
        }
        samples = mono;
    }

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

fn finalize_decoded_audio(mut samples: Vec<f32>, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
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
) -> Result<(Vec<f32>, AudioSourceMetadata)> {
    if source.channel_count == 0 {
        return Err(Error::InferenceError(
            "Decoded audio has invalid source channel count 0".to_string(),
        ));
    }
    let (samples, sample_rate) = finalize_decoded_audio(samples, source.sample_rate)?;
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
