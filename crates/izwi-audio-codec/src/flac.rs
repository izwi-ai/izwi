//! FLAC through `flacenc` (pure Rust), 16 bits a sample, mono.
//!
//! A stream writes `fLaC` and a STREAMINFO block first with the total
//! samples, MD5 and frame sizes left at 0, which RFC 9639 §8.2 defines as
//! "unknown", then one frame per 1024 samples (43 ms at 24 kHz) as audio
//! arrives, and the short remainder at the end. A file is encoded whole, so
//! its STREAMINFO carries the totals and the MD5.

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, Stream};
use flacenc::config;
use flacenc::error::{Verified, Verify};
use flacenc::source::{Fill, FrameBuf, MemSource};

use crate::{to_i16, AudioFormat, CodecError, Framing, StreamEncoder};

/// Samples a frame. Every frame but the last has this many.
const BLOCK_SIZE: usize = 1_024;

fn flac_error(context: &str, err: impl std::fmt::Debug) -> CodecError {
    CodecError::Encode(format!("{context}: {err:?}"))
}

pub(crate) struct FlacEncoder {
    rate: u32,
    framing: Framing,
    config: Verified<config::Encoder>,
    /// The header: STREAMINFO for a stream (the frames refer to it).
    stream: Stream,
    header_sent: bool,
    pending: Vec<i32>,
    next_frame: usize,
    /// File framing: every sample.
    samples: Vec<i32>,
    finished: bool,
}

impl FlacEncoder {
    pub(crate) fn new(rate: u32, framing: Framing) -> Result<Self, CodecError> {
        let config = config::Encoder::default()
            .into_verified()
            .map_err(|(_, err)| flac_error("Invalid FLAC configuration", err))?;
        let mut stream = Stream::new(rate as usize, 1, 16)
            .map_err(|err| flac_error("Invalid FLAC stream parameters", err))?;
        stream
            .stream_info_mut()
            .set_block_sizes(BLOCK_SIZE, BLOCK_SIZE)
            .map_err(|err| flac_error("Invalid FLAC block size", err))?;
        Ok(Self {
            rate,
            framing,
            config,
            stream,
            header_sent: false,
            pending: Vec::new(),
            next_frame: 0,
            samples: Vec::new(),
            finished: false,
        })
    }

    fn header(&mut self, out: &mut Vec<u8>) -> Result<(), CodecError> {
        if std::mem::replace(&mut self.header_sent, true) {
            return Ok(());
        }
        let mut sink = ByteSink::new();
        self.stream
            .write(&mut sink)
            .map_err(|err| flac_error("FLAC header writing failed", err))?;
        out.extend_from_slice(sink.as_slice());
        Ok(())
    }

    fn frame(&mut self, block: &[i32], out: &mut Vec<u8>) -> Result<(), CodecError> {
        let mut buffer = FrameBuf::with_size(1, BLOCK_SIZE)
            .map_err(|err| flac_error("FLAC frame buffer failed", err))?;
        buffer
            .fill_interleaved(block)
            .map_err(|err| flac_error("FLAC frame filling failed", err))?;
        let frame = flacenc::encode_fixed_size_frame(
            &self.config,
            &buffer,
            self.next_frame,
            self.stream.stream_info(),
        )
        .map_err(|err| flac_error("FLAC encoding failed", err))?;
        self.next_frame += 1;
        let mut sink = ByteSink::new();
        frame
            .write(&mut sink)
            .map_err(|err| flac_error("FLAC frame writing failed", err))?;
        out.extend_from_slice(sink.as_slice());
        Ok(())
    }
}

impl StreamEncoder for FlacEncoder {
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        if self.finished {
            return Err(CodecError::Encode("FLAC encoder already finished".into()));
        }
        let samples = pcm.iter().map(|&sample| i32::from(to_i16(sample)));
        if self.framing == Framing::File {
            self.samples.extend(samples);
            return Ok(Vec::new());
        }
        self.pending.extend(samples);
        let mut out = Vec::new();
        self.header(&mut out)?;
        while self.pending.len() >= BLOCK_SIZE {
            let block: Vec<i32> = self.pending.drain(..BLOCK_SIZE).collect();
            self.frame(&block, &mut out)?;
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if std::mem::replace(&mut self.finished, true) {
            return Err(CodecError::Encode("FLAC encoder already finished".into()));
        }
        let mut out = Vec::new();
        match self.framing {
            Framing::Stream => {
                self.header(&mut out)?;
                if !self.pending.is_empty() {
                    let block = std::mem::take(&mut self.pending);
                    self.frame(&block, &mut out)?;
                }
            }
            Framing::File => {
                let source = MemSource::from_samples(&self.samples, 1, 16, self.rate as usize);
                let mut stream =
                    flacenc::encode_with_fixed_block_size(&self.config, source, BLOCK_SIZE)
                        .map_err(|err| flac_error("FLAC encoding failed", err))?;
                // flacenc counts the short last frame in the minimum block
                // size; RFC 9639 §8.2 excludes it, and decoders (symphonia)
                // read min != max as a variable-blocksize stream.
                stream
                    .stream_info_mut()
                    .set_block_sizes(BLOCK_SIZE, BLOCK_SIZE)
                    .map_err(|err| flac_error("Invalid FLAC block size", err))?;
                let mut sink = ByteSink::new();
                stream
                    .write(&mut sink)
                    .map_err(|err| flac_error("FLAC writing failed", err))?;
                out.extend_from_slice(sink.as_slice());
            }
        }
        Ok(out)
    }

    fn format(&self) -> AudioFormat {
        AudioFormat::Flac
    }

    fn output_rate(&self) -> u32 {
        self.rate
    }
}
