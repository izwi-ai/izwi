//! Round trips, chunking invariance and framing checks for every encoder.
//! Decoding goes through independent decoders (symphonia; libopus with the
//! `ogg` reader), never this crate's own code.

use super::*;
use std::io::Cursor;

/// Unstructured test audio: a xorshift noise burst over a few partials with
/// an envelope, so a broken encoder cannot pass on a lucky tone.
fn speech_like(rate: u32, seconds: f32) -> Vec<f32> {
    let len = (rate as f32 * seconds) as usize;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            let t = index as f32 / rate as f32;
            let envelope = 0.5 + 0.5 * (t * 3.0 * std::f32::consts::TAU).sin();
            let voiced = (t * 180.0 * std::f32::consts::TAU).sin() * 0.4
                + (t * 720.0 * std::f32::consts::TAU).sin() * 0.2
                + (t * 1_450.0 * std::f32::consts::TAU).sin() * 0.1;
            (voiced + noise * 0.05) * envelope * 0.8
        })
        .collect()
}

fn encode_chunked(options: &EncoderOptions, pcm: &[f32], chunk: usize) -> Vec<u8> {
    let mut encoder = stream_encoder(options).expect("encoder");
    let mut bytes = Vec::new();
    for piece in pcm.chunks(chunk.max(1)) {
        bytes.extend(encoder.push(piece).expect("push"));
    }
    bytes.extend(encoder.finish().expect("finish"));
    bytes
}

/// Mono samples and rate through symphonia (gapless: LAME delay and padding
/// are trimmed when the file says how much).
fn decode_symphonia(bytes: &[u8]) -> (Vec<f32>, u32) {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::default::{get_codecs, get_probe};

    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let mut format = get_probe()
        .probe(
            &Hint::new(),
            source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .expect("probe");
    let track = format.default_track(TrackType::Audio).expect("track");
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .cloned()
        .expect("audio params");
    let mut decoder = get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .expect("decoder");
    let mut rate = params.sample_rate.unwrap_or(0);
    let mut samples = Vec::new();
    let mut packet_samples = Vec::<f32>::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(err) => panic!("packet: {err}"),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).expect("decode");
        rate = decoded.spec().rate();
        let channels = decoded.spec().channels().count();
        assert_eq!(channels, 1, "mono in, mono out");
        decoded.copy_to_vec_interleaved(&mut packet_samples);
        samples.extend_from_slice(&packet_samples);
    }
    (samples, rate)
}

/// Normalized cross-correlation at the best lag within `max_lag` samples
/// (decoded audio may carry encoder delay).
fn best_correlation(reference: &[f32], decoded: &[f32], max_lag: usize) -> (f32, usize) {
    let mut best = (f32::MIN, 0);
    for lag in 0..=max_lag.min(decoded.len()) {
        let pairs = reference.iter().zip(&decoded[lag..]);
        let (mut dot, mut a, mut b) = (0f64, 0f64, 0f64);
        for (x, y) in pairs {
            dot += f64::from(*x) * f64::from(*y);
            a += f64::from(*x) * f64::from(*x);
            b += f64::from(*y) * f64::from(*y);
        }
        let correlation = if a > 0.0 && b > 0.0 {
            (dot / (a.sqrt() * b.sqrt())) as f32
        } else {
            0.0
        };
        if correlation.is_finite() && correlation > best.0 {
            best = (correlation, lag);
        }
    }
    best
}

fn quantized(pcm: &[f32]) -> Vec<f32> {
    pcm.iter()
        .map(|&sample| f32::from(to_i16(sample)) / 32_768.0)
        .collect()
}

#[test]
fn wav_files_match_hound_byte_for_byte() {
    let pcm = speech_like(24_000, 0.3);
    let ours = encode_file(&EncoderOptions::new(AudioFormat::Wav, 24_000), &pcm).expect("wav");
    let mut theirs = Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut theirs, spec).expect("writer");
        for &sample in &pcm {
            writer.write_sample(to_i16(sample)).expect("sample");
        }
        writer.finalize().expect("finalize");
    }
    assert_eq!(ours, theirs.into_inner());
}

#[test]
fn streamed_wav_has_one_header_with_unknown_sizes() {
    let pcm = speech_like(22_050, 0.2);
    let options = EncoderOptions::new(AudioFormat::Wav, 22_050);
    let whole = encode_chunked(&options, &pcm, pcm.len());
    for chunk in [1, 333, 4_096] {
        assert_eq!(
            encode_chunked(&options, &pcm, chunk),
            whole,
            "chunk {chunk}"
        );
    }
    assert_eq!(&whole[0..4], b"RIFF");
    assert_eq!(&whole[4..8], &u32::MAX.to_le_bytes());
    assert_eq!(&whole[40..44], &u32::MAX.to_le_bytes());
    assert_eq!(whole.len(), 44 + pcm.len() * 2);
    let (decoded, rate) = decode_symphonia(&whole);
    assert_eq!(rate, 22_050);
    assert_eq!(decoded.len(), pcm.len());
    assert_eq!(decoded, quantized(&pcm));
    // No audio at all is still a valid (empty) stream.
    let empty = encode_chunked(&options, &[], 1);
    assert_eq!(empty.len(), 44);
}

#[test]
fn headerless_formats_are_byte_exact_and_resample_to_g711() {
    let pcm = speech_like(24_000, 0.1);
    let i16_bytes = encode_file(&EncoderOptions::new(AudioFormat::PcmI16, 24_000), &pcm).unwrap();
    let expected: Vec<u8> = pcm
        .iter()
        .flat_map(|&sample| to_i16(sample).to_le_bytes())
        .collect();
    assert_eq!(i16_bytes, expected);
    let f32_bytes = encode_file(&EncoderOptions::new(AudioFormat::PcmF32, 24_000), &pcm).unwrap();
    assert_eq!(f32_bytes.len(), pcm.len() * 4);
    for format in [AudioFormat::G711Ulaw, AudioFormat::G711Alaw] {
        let options = EncoderOptions::new(format, 24_000);
        assert_eq!(options.validate(), Ok(8_000));
        let bytes = encode_chunked(&options, &pcm, 1_337);
        assert_eq!(
            bytes.len(),
            pcm.len() / 3,
            "{format}: one byte a sample at 8 kHz"
        );
    }
}

#[test]
fn pcm_resamples_to_a_requested_rate_with_the_same_duration() {
    let pcm = speech_like(44_100, 0.5);
    let options = EncoderOptions::new(AudioFormat::PcmI16, 44_100).with_output_rate(Some(24_000));
    for chunk in [1, 441, 7_000, pcm.len()] {
        let bytes = encode_chunked(&options, &pcm, chunk);
        assert_eq!(bytes.len(), 12_000 * 2, "chunk {chunk}");
    }
}

#[test]
fn flac_files_round_trip_bit_exact_with_totals() {
    for rate in [16_000, 24_000, 44_100] {
        let pcm = speech_like(rate, 0.37);
        let bytes = encode_file(&EncoderOptions::new(AudioFormat::Flac, rate), &pcm).unwrap();
        assert_eq!(&bytes[0..4], b"fLaC");
        let (decoded, decoded_rate) = decode_symphonia(&bytes);
        assert_eq!(decoded_rate, rate);
        assert_eq!(decoded, quantized(&pcm), "{rate} Hz");
    }
}

#[test]
fn streamed_flac_is_the_same_audio_however_it_is_chunked() {
    let pcm = speech_like(24_000, 0.41);
    let options = EncoderOptions::new(AudioFormat::Flac, 24_000);
    let whole = encode_chunked(&options, &pcm, pcm.len());
    for chunk in [1, 1_023, 1_025, 4_801] {
        assert_eq!(
            encode_chunked(&options, &pcm, chunk),
            whole,
            "chunk {chunk}"
        );
    }
    // STREAMINFO (bytes 8..42 after the 4-byte magic and 4-byte block
    // header): total samples are "unknown" (0) in a stream.
    let streaminfo = &whole[8..42];
    let total_samples = u64::from(streaminfo[13] & 0x0F) << 32
        | u64::from(u32::from_be_bytes(streaminfo[14..18].try_into().unwrap()));
    assert_eq!(total_samples, 0);
    let (decoded, rate) = decode_symphonia(&whole);
    assert_eq!(rate, 24_000);
    assert_eq!(decoded, quantized(&pcm));
}

#[cfg(feature = "mp3")]
#[test]
fn mp3_streams_and_files_decode_to_the_audio() {
    for rate in [24_000, 22_050, 44_100] {
        let pcm = speech_like(rate, 1.0);
        let stream_options = EncoderOptions::new(AudioFormat::Mp3, rate);
        assert_eq!(
            stream_options.validate(),
            Ok(rate),
            "{rate} Hz is native to MP3"
        );
        let stream = encode_chunked(&stream_options, &pcm, pcm.len());
        for chunk in [1, 577, 4_800] {
            assert_eq!(
                encode_chunked(&stream_options, &pcm, chunk),
                stream,
                "{rate} Hz chunk {chunk}: LAME is chunk-invariant"
            );
        }
        let (decoded, decoded_rate) = decode_symphonia(&stream);
        assert_eq!(decoded_rate, rate);
        let frame = if rate >= 32_000 { 1_152 } else { 576 };
        assert!(
            decoded.len() >= pcm.len() && decoded.len() <= pcm.len() + 1_105 + 3 * frame,
            "{rate} Hz stream: {} samples for {}",
            decoded.len(),
            pcm.len()
        );
        let (correlation, _) = best_correlation(&pcm, &decoded, 2_400);
        assert!(
            correlation > 0.95,
            "{rate} Hz stream correlation {correlation}"
        );

        // A file carries the LAME tag, so a gapless decoder trims the delay.
        let file = encode_file(&stream_options, &pcm).unwrap();
        let (decoded, _) = decode_symphonia(&file);
        assert!(
            decoded.len().abs_diff(pcm.len()) <= frame,
            "{rate} Hz file: {} samples for {}",
            decoded.len(),
            pcm.len()
        );
        let (correlation, lag) = best_correlation(&pcm, &decoded, 64);
        assert!(
            correlation > 0.95,
            "{rate} Hz file correlation {correlation}"
        );
        assert!(lag <= 2, "{rate} Hz file is aligned (lag {lag})");
    }
}

#[cfg(feature = "mp3")]
#[test]
fn mp3_bitrates_set_the_size() {
    let pcm = speech_like(24_000, 2.0);
    let size = |bitrate: u32| {
        encode_file(
            &EncoderOptions::new(AudioFormat::Mp3, 24_000).with_bitrate(Some(bitrate)),
            &pcm,
        )
        .unwrap()
        .len()
    };
    let (low, high) = (size(32_000), size(128_000));
    // CBR: two seconds at 32 and 128 kbit/s, within a few frames.
    assert!(low.abs_diff(8_000) < 1_000, "32 kbit/s gave {low} bytes");
    assert!(
        high.abs_diff(32_000) < 2_000,
        "128 kbit/s gave {high} bytes"
    );
    assert!(EncoderOptions::new(AudioFormat::Mp3, 24_000)
        .with_bitrate(Some(320_000))
        .validate()
        .is_err());
}

#[cfg(feature = "opus")]
mod opus_streams {
    use super::*;
    use ogg::reading::PacketReader;

    pub(super) struct DecodedOpus {
        pub samples: Vec<f32>,
        pub pre_skip: usize,
        pub input_rate: u32,
        pub granules: Vec<u64>,
        pub ended: bool,
    }

    /// Decodes Ogg Opus at 48 kHz with libopus, trimming pre-skip and the
    /// end as RFC 7845 says.
    pub(super) fn decode(bytes: &[u8]) -> DecodedOpus {
        let mut reader = PacketReader::new(Cursor::new(bytes.to_vec()));
        let head = reader.read_packet_expected().expect("OpusHead");
        assert!(
            head.first_in_stream() && head.last_in_page(),
            "OpusHead alone on page 1"
        );
        assert_eq!(&head.data[0..8], b"OpusHead");
        assert_eq!(head.absgp_page(), 0);
        let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]) as usize;
        let input_rate = u32::from_le_bytes(head.data[12..16].try_into().unwrap());
        let tags = reader.read_packet_expected().expect("OpusTags");
        assert_eq!(&tags.data[0..8], b"OpusTags");
        assert!(tags.last_in_page(), "OpusTags finishes page 2");
        let mut decoder = ::opus::Decoder::new(48_000, ::opus::Channels::Mono).expect("decoder");
        let mut samples = Vec::new();
        let mut granules = Vec::new();
        let mut ended = false;
        let mut last_granule = 0;
        let mut buffer = vec![0f32; 5_760];
        while let Some(packet) = reader.read_packet().expect("packet") {
            assert!(!ended, "no packet after end of stream");
            let count = decoder
                .decode_float(&packet.data, &mut buffer, false)
                .expect("decode");
            samples.extend_from_slice(&buffer[..count]);
            if packet.last_in_page() {
                granules.push(packet.absgp_page());
            }
            last_granule = packet.absgp_page();
            ended = packet.last_in_stream();
        }
        let end = (last_granule as usize).min(samples.len());
        let samples = samples[pre_skip.min(end)..end].to_vec();
        DecodedOpus {
            samples,
            pre_skip,
            input_rate,
            granules,
            ended,
        }
    }
}

#[cfg(feature = "opus")]
#[test]
fn ogg_opus_streams_follow_rfc_7845() {
    for (rate, encoder_rate) in [
        (24_000, 24_000),
        (16_000, 16_000),
        (44_100, 48_000),
        (22_050, 24_000),
    ] {
        let pcm = speech_like(rate, 1.0);
        let options = EncoderOptions::new(AudioFormat::Opus, rate);
        assert_eq!(options.validate(), Ok(encoder_rate));
        for chunk in [1, 480, 2_000, pcm.len()] {
            let bytes = encode_chunked(&options, &pcm, chunk);
            let decoded = opus_streams::decode(&bytes);
            assert!(decoded.ended, "{rate} Hz chunk {chunk}: last page has EOS");
            assert_eq!(decoded.input_rate, rate, "OpusHead keeps the original rate");
            assert!(decoded.pre_skip > 0);
            assert!(
                decoded.granules.windows(2).all(|pair| pair[0] <= pair[1]),
                "granules only grow: {:?}",
                decoded.granules
            );
            // 48 kHz output, trimmed to exactly the input's duration.
            let expected = (pcm.len() as u64 * 48_000 / u64::from(rate)) as usize;
            assert!(
                decoded.samples.len().abs_diff(expected) <= 2,
                "{rate} Hz chunk {chunk}: {} samples, expected {expected}",
                decoded.samples.len()
            );
            // Compare at 48 kHz against the input resampled the same way.
            let reference = {
                let mut resampler = StreamResampler::new(rate, 48_000).unwrap();
                let mut out = resampler.push(&pcm).unwrap();
                out.extend(resampler.finish().unwrap());
                out
            };
            let (correlation, lag) = best_correlation(&reference, &decoded.samples, 64);
            assert!(
                correlation > 0.9,
                "{rate} Hz chunk {chunk}: correlation {correlation}"
            );
            assert!(
                lag <= 8,
                "{rate} Hz chunk {chunk}: pre-skip aligns it (lag {lag})"
            );
        }
    }
}

#[cfg(feature = "opus")]
#[test]
fn every_tts_chunk_ends_an_ogg_page() {
    let pcm = speech_like(24_000, 1.0);
    let mut encoder = stream_encoder(&EncoderOptions::new(AudioFormat::Opus, 24_000)).unwrap();
    let first = encoder.push(&pcm[..4_800]).unwrap();
    // Headers plus 200 ms of audio, flushed as whole pages.
    let decoded_prefix = {
        let mut reader = ogg::reading::PacketReader::new(Cursor::new(first.clone()));
        let mut packets = 0;
        while let Ok(Some(_)) = reader.read_packet() {
            packets += 1;
        }
        packets
    };
    assert_eq!(
        decoded_prefix,
        2 + 10,
        "OpusHead, OpusTags and ten 20 ms packets"
    );
    let rest = encoder.push(&pcm[4_800..]).unwrap();
    let tail = encoder.finish().unwrap();
    let mut all = first;
    all.extend(rest);
    all.extend(tail);
    assert!(opus_streams::decode(&all).ended);
}

#[test]
fn empty_audio_finishes_into_a_valid_stream_for_every_format() {
    for format in AudioFormat::available() {
        let options = EncoderOptions::new(format, 24_000);
        let mut encoder = stream_encoder(&options).expect("encoder");
        let bytes = encoder.finish().expect("finish");
        match format {
            AudioFormat::Wav => assert_eq!(bytes.len(), 44),
            AudioFormat::Flac => assert_eq!(&bytes[0..4], b"fLaC"),
            AudioFormat::Opus => assert!(bytes.starts_with(b"OggS")),
            _ => {}
        }
        assert!(
            encoder.push(&[0.0]).is_err(),
            "{format}: no audio after finish"
        );
    }
}

#[test]
fn formats_parse_and_refuse_what_this_build_lacks() {
    assert_eq!(AudioFormat::parse("MP3"), Some(AudioFormat::Mp3));
    assert_eq!(AudioFormat::parse("ogg"), Some(AudioFormat::Opus));
    assert_eq!(AudioFormat::parse("raw_i16"), Some(AudioFormat::PcmI16));
    assert_eq!(AudioFormat::parse("pcm"), None, "callers map OpenAI's pcm");
    let aac = EncoderOptions::new(AudioFormat::Aac, 24_000)
        .validate()
        .unwrap_err();
    assert!(aac.to_string().contains("patent"), "{aac}");
    assert!(aac.is_client_error());
    assert!(!AudioFormat::available().contains(&AudioFormat::Aac));
    let bad_rate = EncoderOptions::new(AudioFormat::Opus, 24_000)
        .with_output_rate(Some(44_100))
        .validate()
        .unwrap_err();
    assert!(bad_rate.to_string().contains("48000"), "{bad_rate}");
    assert!(EncoderOptions::new(AudioFormat::Wav, 24_000)
        .with_bitrate(Some(64_000))
        .validate()
        .is_err());
    assert_eq!(AudioFormat::Opus.content_type(), "audio/ogg");
    assert_eq!(AudioFormat::Mp3.content_type(), "audio/mpeg");
}

/// The encoder CPU budget (plan §9.5): each format's speed in multiples of
/// real time, streaming 24 kHz speech in 40 ms chunks as TTS emits it. Run
/// in release: `cargo test -p izwi-audio-codec --release -- --ignored
/// --nocapture encoders_run`.
#[test]
#[ignore = "a measurement; run in release"]
fn encoders_run_many_times_faster_than_real_time() {
    let rate = 24_000;
    let seconds = 60.0;
    let pcm = speech_like(rate, seconds);
    for format in [
        AudioFormat::Wav,
        AudioFormat::PcmI16,
        AudioFormat::G711Ulaw,
        AudioFormat::Mp3,
        AudioFormat::Opus,
        AudioFormat::Flac,
    ] {
        let options = EncoderOptions::new(format, rate);
        let started = std::time::Instant::now();
        let bytes = encode_chunked(&options, &pcm, 960);
        let elapsed = started.elapsed().as_secs_f32();
        let speed = seconds / elapsed;
        println!(
            "{:>5}: {speed:>8.0}x real time, {:>7.1} ms for {seconds} s, {} bytes",
            format.as_str(),
            elapsed * 1_000.0,
            bytes.len()
        );
        assert!(speed > 100.0, "{} encodes at {speed:.0}x", format.as_str());
    }
}
