//! A band-limited resampler fed in arbitrary pieces, for encoders whose
//! output rate differs from the model's (Opus at 24/48 kHz, G.711 at 8 kHz,
//! OpenAI `pcm` at 24 kHz).

use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler, WindowFunction};

use crate::CodecError;

/// Input frames the resampler takes at a time (20 ms at 24 kHz).
const RESAMPLE_CHUNK_FRAMES: usize = 480;

/// Resamples mono audio pushed in pieces of any size. The output has the
/// input's duration: the resampler's delay is skipped at the start and its
/// delay line is flushed at [`StreamResampler::finish`].
pub struct StreamResampler {
    resampler: Fft<f32>,
    from: u32,
    to: u32,
    pending: Vec<f32>,
    /// Leading output samples that are the resampler's delay, not audio.
    delay: usize,
    input_total: usize,
    output_total: usize,
}

impl std::fmt::Debug for StreamResampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamResampler")
            .field("from", &self.from)
            .field("to", &self.to)
            .finish_non_exhaustive()
    }
}

impl StreamResampler {
    pub fn new(from: u32, to: u32) -> Result<Self, CodecError> {
        if from == 0 || to == 0 {
            return Err(CodecError::InvalidOption(format!(
                "Cannot resample from {from} Hz to {to} Hz"
            )));
        }
        // Fixed input chunks with one FFT block each and the BlackmanHarris2
        // window: rubato 0.15's FftFixedIn with one sub-chunk.
        let resampler = Fft::<f32>::new_custom(
            from as usize,
            to as usize,
            RESAMPLE_CHUNK_FRAMES,
            1,
            1,
            WindowFunction::BlackmanHarris2,
            FixedSync::Input,
        )
        .map_err(|err| {
            CodecError::Encode(format!("Failed to build the output resampler: {err}"))
        })?;
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

    pub fn from_rate(&self) -> u32 {
        self.from
    }

    pub fn to_rate(&self) -> u32 {
        self.to
    }

    /// The output for `samples`; some audio may be held for the next call.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>, CodecError> {
        self.input_total += samples.len();
        self.pending.extend_from_slice(samples);
        let mut out = Vec::new();
        loop {
            let needed = self.resampler.input_frames_next();
            if self.pending.len() < needed {
                break;
            }
            let chunk: Vec<f32> = self.pending.drain(..needed).collect();
            let produced = self.process_chunk(&chunk)?;
            out.extend_from_slice(&produced);
        }
        let out = self.skip_delay(out);
        self.output_total += out.len();
        Ok(out)
    }

    /// The rest of the output: the held input and the resampler's delay
    /// line flushed with silence, cut to the input's duration.
    pub fn finish(&mut self) -> Result<Vec<f32>, CodecError> {
        let expected =
            (self.input_total as u64 * u64::from(self.to)).div_ceil(u64::from(self.from)) as usize;
        let mut out = Vec::new();
        let mut pending = std::mem::take(&mut self.pending);
        while self.output_total + out.len() < expected + self.delay {
            pending.resize(self.resampler.input_frames_next(), 0.0);
            let produced = self.process_chunk(&std::mem::take(&mut pending))?;
            out.extend_from_slice(&produced);
        }
        let mut out = self.skip_delay(out);
        out.truncate(expected.saturating_sub(self.output_total));
        self.output_total += out.len();
        Ok(out)
    }

    /// Resample one `input_frames_next()`-frame chunk.
    fn process_chunk(&mut self, chunk: &[f32]) -> Result<Vec<f32>, CodecError> {
        let input = SequentialSlice::new(chunk, 1, chunk.len())
            .map_err(|err| CodecError::Encode(format!("Output resampling failed: {err}")))?;
        self.resampler
            .process(&input, None)
            .map(|produced| produced.take_data())
            .map_err(|err| CodecError::Encode(format!("Output resampling failed: {err}")))
    }

    fn skip_delay(&mut self, mut out: Vec<f32>) -> Vec<f32> {
        let skip = self.delay.min(out.len());
        self.delay -= skip;
        out.drain(..skip);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chunked resampling has the input's duration whatever the chunking.
    #[test]
    fn chunked_output_keeps_duration() {
        let tone: Vec<f32> = (0..44_100)
            .map(|index| (index as f32 * 440.0 * std::f32::consts::TAU / 44_100.0).sin() * 0.5)
            .collect();
        for chunk in [1usize, 479, 1_337, 44_100] {
            let mut resampler = StreamResampler::new(44_100, 24_000).expect("resampler");
            let mut out = Vec::new();
            for piece in tone.chunks(chunk) {
                out.extend(resampler.push(piece).expect("push"));
            }
            out.extend(resampler.finish().expect("finish"));
            assert_eq!(out.len(), 24_000, "chunk {chunk}");
        }
    }
}
