//! Opus in Ogg (RFC 7845) through libopus (`opus` crate) and the `ogg` page
//! writer.
//!
//! The stream starts with the OpusHead page and the OpusTags page, then
//! carries 20 ms packets. A page ends at every `push` that produced audio
//! (and after ten packets), so a client gets audio as each TTS chunk is
//! encoded. Granule positions are in 48 kHz samples and include the
//! pre-skip; the last page's granule trims the zero padding that flushes the
//! encoder's lookahead.

use std::sync::atomic::{AtomicU32, Ordering};

use ogg::writing::{PacketWriteEndInfo, PacketWriter};
use opus::{Application, Bitrate, Channels, Encoder};

use crate::{sanitize, AudioFormat, CodecError, StreamEncoder};

/// Packets on one page at most, so a long push still pages every 200 ms.
const PACKETS_PER_PAGE: usize = 10;
/// libopus's recommended packet buffer.
const MAX_PACKET_BYTES: usize = 4_000;
const VENDOR: &str = concat!("izwi-audio-codec ", env!("CARGO_PKG_VERSION"));

pub(crate) fn check_bitrate(bits: u32) -> Result<(), CodecError> {
    if (6_000..=510_000).contains(&bits) {
        Ok(())
    } else {
        Err(CodecError::InvalidOption(format!(
            "bitrate {bits} is out of range for opus (6000 to 510000 bit/s)"
        )))
    }
}

/// A speech bitrate for mono Opus at `rate`.
fn default_bitrate(rate: u32) -> u32 {
    match rate {
        48_000 => 64_000,
        24_000 => 32_000,
        16_000 => 24_000,
        12_000 => 20_000,
        _ => 16_000,
    }
}

/// A serial number per logical stream: chained Ogg streams (one per spoken
/// segment) need distinct serials.
fn next_serial() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.subsec_nanos())
        .unwrap_or(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(seed)
}

pub(crate) struct OggOpusEncoder {
    encoder: Encoder,
    writer: PacketWriter<'static, Vec<u8>>,
    serial: u32,
    rate: u32,
    /// The rate the audio had before any resampling, for OpusHead.
    original_rate: u32,
    /// Samples a packet, at `rate` (20 ms).
    frame: usize,
    /// Encoder delay, at `rate`.
    lookahead: usize,
    /// Pre-skip in 48 kHz samples.
    pre_skip: u64,
    pending: Vec<f32>,
    /// Real input samples at `rate`.
    input_total: u64,
    /// Samples encoded into packets at `rate`, padding included.
    encoded_total: u64,
    headers_written: bool,
    finished: bool,
}

fn opus_error(context: &str, err: impl std::fmt::Display) -> CodecError {
    CodecError::Encode(format!("{context}: {err}"))
}

impl OggOpusEncoder {
    pub(crate) fn new(
        rate: u32,
        original_rate: u32,
        bitrate: Option<u32>,
    ) -> Result<Self, CodecError> {
        let bitrate = bitrate.unwrap_or_else(|| default_bitrate(rate));
        check_bitrate(bitrate)?;
        let mut encoder = Encoder::new(rate, Channels::Mono, Application::Audio)
            .map_err(|err| opus_error("Failed to set up the Opus encoder", err))?;
        encoder
            .set_bitrate(Bitrate::Bits(bitrate as i32))
            .map_err(|err| opus_error("Failed to set the Opus bitrate", err))?;
        let lookahead = encoder
            .get_lookahead()
            .map_err(|err| opus_error("Failed to read the Opus lookahead", err))?
            .max(0) as usize;
        let pre_skip = (lookahead as u64 * 48_000).div_ceil(u64::from(rate));
        Ok(Self {
            encoder,
            writer: PacketWriter::new(Vec::new()),
            serial: next_serial(),
            rate,
            original_rate,
            frame: (rate / 50) as usize,
            lookahead,
            pre_skip,
            pending: Vec::new(),
            input_total: 0,
            encoded_total: 0,
            headers_written: false,
            finished: false,
        })
    }

    /// 48 kHz samples for `samples` at the encoder's rate.
    fn to_48k(&self, samples: u64) -> u64 {
        samples * 48_000 / u64::from(self.rate)
    }

    fn write_headers(&mut self) -> Result<(), CodecError> {
        if std::mem::replace(&mut self.headers_written, true) {
            return Ok(());
        }
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(1); // channels
        head.extend_from_slice(&(self.pre_skip as u16).to_le_bytes());
        head.extend_from_slice(&self.original_rate.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // mapping family: mono/stereo
        let mut tags = Vec::with_capacity(16 + VENDOR.len());
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&(VENDOR.len() as u32).to_le_bytes());
        tags.extend_from_slice(VENDOR.as_bytes());
        tags.extend_from_slice(&0u32.to_le_bytes()); // no user comments
                                                     // OpusHead alone on the first page, OpusTags finishing the second.
        self.write(head, PacketWriteEndInfo::EndPage, 0)?;
        self.write(tags, PacketWriteEndInfo::EndPage, 0)
    }

    fn write(
        &mut self,
        packet: Vec<u8>,
        end: PacketWriteEndInfo,
        granule: u64,
    ) -> Result<(), CodecError> {
        self.writer
            .write_packet(packet, self.serial, end, granule)
            .map_err(|err| opus_error("Ogg page writing failed", err))
    }

    fn encode_frame(&mut self, frame: &[f32]) -> Result<Vec<u8>, CodecError> {
        let mut packet = vec![0u8; MAX_PACKET_BYTES];
        let written = self
            .encoder
            .encode_float(frame, &mut packet)
            .map_err(|err| opus_error("Opus encoding failed", err))?;
        packet.truncate(written);
        self.encoded_total += frame.len() as u64;
        Ok(packet)
    }

    fn drain(&mut self) -> Vec<u8> {
        std::mem::take(self.writer.inner_mut())
    }
}

impl StreamEncoder for OggOpusEncoder {
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        if self.finished {
            return Err(CodecError::Encode("Opus encoder already finished".into()));
        }
        self.write_headers()?;
        self.input_total += pcm.len() as u64;
        self.pending.extend(pcm.iter().copied().map(sanitize));
        let mut packets = Vec::new();
        while self.pending.len() >= self.frame {
            let frame: Vec<f32> = self.pending.drain(..self.frame).collect();
            let packet = self.encode_frame(&frame)?;
            packets.push((packet, self.pre_skip + self.to_48k(self.encoded_total)));
        }
        let count = packets.len();
        for (index, (packet, granule)) in packets.into_iter().enumerate() {
            let end = if index + 1 == count || (index + 1) % PACKETS_PER_PAGE == 0 {
                PacketWriteEndInfo::EndPage
            } else {
                PacketWriteEndInfo::NormalPacket
            };
            self.write(packet, end, granule)?;
        }
        Ok(self.drain())
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if std::mem::replace(&mut self.finished, true) {
            return Err(CodecError::Encode("Opus encoder already finished".into()));
        }
        self.write_headers()?;
        // Encode the held samples plus enough silence to flush the
        // lookahead; the final granule trims what is not audio.
        let target = self.input_total + self.lookahead as u64;
        let mut packets = Vec::new();
        loop {
            let mut frame: Vec<f32> = self
                .pending
                .drain(..self.pending.len().min(self.frame))
                .collect();
            frame.resize(self.frame, 0.0);
            packets.push(self.encode_frame(&frame)?);
            if self.encoded_total >= target {
                break;
            }
        }
        // Every packet here lands on the last page, whose granule is the end
        // of the real audio: earlier pages never pass it, so it only grows.
        let final_granule = self.pre_skip + self.to_48k(self.input_total);
        let last = packets.len() - 1;
        for (index, packet) in packets.into_iter().enumerate() {
            let end = if index == last {
                PacketWriteEndInfo::EndStream
            } else {
                PacketWriteEndInfo::NormalPacket
            };
            self.write(packet, end, final_granule)?;
        }
        Ok(self.drain())
    }

    fn format(&self) -> AudioFormat {
        AudioFormat::Opus
    }

    fn output_rate(&self) -> u32 {
        self.rate
    }
}
