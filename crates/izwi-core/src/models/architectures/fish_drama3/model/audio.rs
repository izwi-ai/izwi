//! Committed frames through the codec stream (§7.1 stages 3 and 4).
//!
//! The codec only ever sees frames whose decode quantum committed, so an
//! aborted quantum never has to unsay audio. A codec quantum is staged into
//! a [`FishDrama3AudioCandidate`] holding the stream's next history beside
//! the committed one (the two copies S2 prices), and becomes visible only
//! when [`FishDrama3AudioStream::commit`] installs it; dropping the
//! candidate is the rollback. The cadence is S2's: the first audio after
//! 4 frames, then 16-frame chunks, then whatever remains once the request
//! stops. Finalising flushes the stream and holds the PCM to exactly
//! `frames × samples_per_frame`.

use super::super::codec::FishDrama3DacStreamState;
use super::{FishDrama3CancelCheck, FishDrama3Model, FishDrama3RequestState};
use crate::error::{Error, Result};

pub const FISH_DRAMA3_FIRST_AUDIO_FRAMES: usize = 4;
pub const FISH_DRAMA3_AUDIO_CHUNK_FRAMES: usize = 16;

/// One request's committed codec stream.
#[derive(Debug, Clone, Default)]
pub struct FishDrama3AudioStream {
    codec: FishDrama3DacStreamState,
    /// Frames decoded so far.
    frames: usize,
    /// Samples emitted so far.
    samples: usize,
    finished: bool,
}

/// A staged codec quantum: the stream's next history and its PCM, visible
/// once committed.
#[derive(Debug)]
pub struct FishDrama3AudioCandidate {
    codec: FishDrama3DacStreamState,
    /// The stream's frame count it was staged from.
    base: usize,
    frames: usize,
    samples: Vec<f32>,
}

impl FishDrama3AudioCandidate {
    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

impl FishDrama3AudioStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decoded_frames(&self) -> usize {
        self.frames
    }

    pub fn emitted_samples(&self) -> usize {
        self.samples
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Installs `candidate` and returns its PCM. A candidate staged from
    /// another position (a stale or replayed quantum) is refused.
    pub fn commit(&mut self, candidate: FishDrama3AudioCandidate) -> Result<Vec<f32>> {
        if self.finished || candidate.base != self.frames {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 codec quantum staged at frame {} cannot commit at {}",
                candidate.base, self.frames
            )));
        }
        self.codec = candidate.codec;
        self.frames += candidate.frames;
        self.samples += candidate.samples.len();
        Ok(candidate.samples)
    }
}

impl FishDrama3Model {
    /// The frames the next codec quantum of `state` should decode, if one
    /// is due: 4 for the first audio, then 16 at a time, and the remainder
    /// once the request has stopped.
    pub fn audio_quantum_frames(
        &self,
        state: &FishDrama3RequestState,
        stream: &FishDrama3AudioStream,
    ) -> Option<usize> {
        if state.in_quantum() || stream.finished {
            return None;
        }
        let pending = state.frames().saturating_sub(stream.frames);
        let threshold = if stream.frames == 0 {
            FISH_DRAMA3_FIRST_AUDIO_FRAMES
        } else {
            FISH_DRAMA3_AUDIO_CHUNK_FRAMES
        };
        (pending >= threshold || (state.is_finished() && pending > 0))
            .then(|| pending.min(FISH_DRAMA3_AUDIO_CHUNK_FRAMES))
    }

    /// The committed frames `[decoded, decoded + max_frames)` of `state`, or
    /// fewer when fewer are committed.
    fn pending_codes(
        &self,
        state: &FishDrama3RequestState,
        stream: &FishDrama3AudioStream,
        max_frames: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.check_state(state)?;
        if state.in_quantum()
            || stream.finished
            || max_frames == 0
            || max_frames > FISH_DRAMA3_AUDIO_CHUNK_FRAMES
        {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 codec quanta take 1 to 16 committed frames of an open stream"
                    .into(),
            ));
        }
        let start = stream.frames;
        let end = (start + max_frames).min(state.frames());
        if start >= end {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 codec has no committed frames to decode".into(),
            ));
        }
        Ok(state
            .codes()
            .iter()
            .map(|codebook| codebook[start..end].to_vec())
            .collect())
    }

    /// Checks a quantum's PCM and the history it leaves.
    fn check_audio(
        &self,
        frames: usize,
        samples: &[f32],
        codec: &FishDrama3DacStreamState,
    ) -> Result<()> {
        if samples.len() != frames * self.codec.samples_per_frame()?
            || samples.iter().any(|sample| !sample.is_finite())
        {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 codec made {} samples for {frames} frames, or non-finite ones",
                samples.len()
            )));
        }
        if codec.retained_tensor_bytes() > self.codec.stream_history_bound_bytes()? {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 codec stream exceeded its history bound".into(),
            ));
        }
        Ok(())
    }

    /// Stages a codec quantum of up to `max_frames` committed frames; the
    /// stream is unchanged until the candidate commits.
    pub fn stage_audio(
        &self,
        state: &FishDrama3RequestState,
        stream: &FishDrama3AudioStream,
        max_frames: usize,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3AudioCandidate> {
        let codes = self.pending_codes(state, stream, max_frames)?;
        let frames = codes[0].len();
        let mut codec = stream.codec.clone();
        let samples = self.codec.stream_frames(&mut codec, &codes, check)?;
        self.check_audio(frames, &samples, &codec)?;
        Ok(FishDrama3AudioCandidate {
            codec,
            base: stream.frames,
            frames,
            samples,
        })
    }

    /// [`Self::stage_audio`] for several requests in one codec call; each
    /// candidate equals its scalar one.
    pub fn stage_audio_batch(
        &self,
        rows: &[(&FishDrama3RequestState, &FishDrama3AudioStream, usize)],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<FishDrama3AudioCandidate>> {
        if rows.is_empty() {
            return Err(Error::InvalidInput("a codec batch needs a row".into()));
        }
        let mut codes = Vec::with_capacity(rows.len());
        for &(state, stream, max_frames) in rows {
            check()?;
            codes.push(self.pending_codes(state, stream, max_frames)?);
        }
        let mut codecs = rows
            .iter()
            .map(|(_, stream, _)| stream.codec.clone())
            .collect::<Vec<_>>();
        let outputs = self.codec.stream_frames_batch(&mut codecs, &codes, check)?;
        if outputs.len() != rows.len() {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 codec batch lost a row".into(),
            ));
        }
        rows.iter()
            .zip(codes)
            .zip(codecs.into_iter().zip(outputs))
            .map(|(((_, stream, _), codes), (codec, samples))| {
                let frames = codes[0].len();
                self.check_audio(frames, &samples, &codec)?;
                Ok(FishDrama3AudioCandidate {
                    codec,
                    base: stream.frames,
                    frames,
                    samples,
                })
            })
            .collect()
    }

    /// Decodes whatever committed frames remain after eos and flushes the
    /// stream: its PCM is then exactly `frames × samples_per_frame`. A
    /// frame-limit stop is an error.
    pub fn finalize_audio(
        &self,
        state: &FishDrama3RequestState,
        stream: &mut FishDrama3AudioStream,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        check()?;
        self.check_state(state)?;
        let reason = state.stop_reason().filter(|_| !state.in_quantum());
        let Some(reason) = reason else {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 finalises only a stopped, committed request".into(),
            ));
        };
        reason.require_complete()?;
        let mut samples = Vec::new();
        while stream.frames < state.frames() {
            let frames = (state.frames() - stream.frames).min(FISH_DRAMA3_AUDIO_CHUNK_FRAMES);
            let candidate = self.stage_audio(state, stream, frames, check)?;
            samples.extend(stream.commit(candidate)?);
        }
        let tail = stream.codec.flush();
        stream.samples += tail.len();
        samples.extend(tail);
        stream.finished = true;
        let expected = state.frames() * self.codec.samples_per_frame()?;
        if stream.samples != expected {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 stream emitted {} samples for {} frames; expected {expected}",
                stream.samples,
                state.frames()
            )));
        }
        Ok(samples)
    }
}
