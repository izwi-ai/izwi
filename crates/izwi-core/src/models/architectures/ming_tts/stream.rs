//! A request's audio as it is generated: latents decoded in windows, and
//! the reference's trailing-silence trim applied as the samples arrive.
//!
//! The VAE decodes incrementally ([`MingVaeDecodeState`]), so a window's
//! samples are the ones a single decode of the whole utterance gives. The
//! reference trims after decoding everything (`trim_trailing_silence`): it
//! scans 0.1 s frames backwards for the last one whose mean magnitude is
//! above 1e-3 and keeps 0.3 s beyond it. [`TrailingSilenceTrim`] reaches
//! the same cut going forwards: audio is released only up to 0.3 s past the
//! last loud frame seen so far, and what is still held when the utterance
//! ends is dropped.

use super::config::{LATENT_DIM, SAMPLE_RATE};
use super::vae::{MingVaeDecodeState, MingVaeDecoder};
use crate::error::Result;

/// Latents before the first window (0.32 s): early first audio.
pub(crate) const FIRST_WINDOW_LATENTS: usize = 4;
/// Latents per later window (0.64 s).
pub(crate) const WINDOW_LATENTS: usize = 8;

const TRIM_FRAME: usize = SAMPLE_RATE as usize / 10;
const TRIM_TAIL: usize = SAMPLE_RATE as usize * 3 / 10;
const TRIM_THRESHOLD: f32 = 1e-3;

#[derive(Debug, Clone, Default)]
pub(crate) struct TrailingSilenceTrim {
    /// Samples received and not yet released, from sample `released`.
    held: Vec<f32>,
    released: usize,
    /// Whole 0.1 s frames scored so far.
    frames: usize,
    /// The sample the release may run to: 0.3 s past the last loud frame.
    release_until: usize,
}

impl TrailingSilenceTrim {
    fn received(&self) -> usize {
        self.released + self.held.len()
    }

    /// Take `samples` and return what is now safe to play.
    pub(crate) fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.held.extend_from_slice(samples);
        while (self.frames + 1) * TRIM_FRAME <= self.received() {
            let start = self.frames * TRIM_FRAME - self.released;
            let frame = &self.held[start..start + TRIM_FRAME];
            let level = frame.iter().map(|sample| sample.abs()).sum::<f32>() / TRIM_FRAME as f32;
            self.frames += 1;
            if level > TRIM_THRESHOLD {
                self.release_until = self.frames * TRIM_FRAME + TRIM_TAIL;
            }
        }
        // Only whole frames are ever kept by the reference's cut.
        self.release(self.release_until.min(self.frames * TRIM_FRAME))
    }

    /// The utterance is over: release the rest of what the cut keeps.
    pub(crate) fn finish(&mut self) -> Vec<f32> {
        let received = self.received();
        let until = if received < TRIM_FRAME {
            // Shorter than one frame: the reference keeps up to the tail.
            received.min(TRIM_TAIL)
        } else if self.release_until == 0 {
            // Silent throughout: the fixed tail rather than nothing.
            TRIM_TAIL.min(self.frames * TRIM_FRAME)
        } else {
            self.release_until.min(self.frames * TRIM_FRAME)
        };
        let out = self.release(until);
        self.held.clear();
        out
    }

    fn release(&mut self, until: usize) -> Vec<f32> {
        let count = until.saturating_sub(self.released).min(self.held.len());
        self.released += count;
        self.held.drain(..count).collect()
    }
}

/// The latents a window decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MingWindow {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) last: bool,
}

#[derive(Default)]
pub(crate) struct MingAudioStream {
    vae: MingVaeDecodeState,
    trim: TrailingSilenceTrim,
    /// Samples released to the listener.
    samples: usize,
}

impl MingAudioStream {
    /// The window that can run now: `latents` are generated so far and
    /// `stopped` says the model is done.
    pub(crate) fn next_window(&self, latents: usize, stopped: bool) -> Option<MingWindow> {
        if self.vae.finished() {
            return None;
        }
        let from = self.vae.latents();
        if stopped {
            return Some(MingWindow {
                from,
                to: latents,
                last: true,
            });
        }
        let due = if from == 0 {
            FIRST_WINDOW_LATENTS
        } else {
            WINDOW_LATENTS
        };
        (latents >= from + due).then_some(MingWindow {
            from,
            to: latents,
            last: false,
        })
    }

    /// Decode `window` of `latents` (64 values each) and return the samples
    /// it releases.
    pub(crate) fn decode(
        &mut self,
        decoder: &MingVaeDecoder,
        latents: &[f32],
        window: MingWindow,
    ) -> Result<Vec<f32>> {
        let slice = &latents[window.from * LATENT_DIM..window.to * LATENT_DIM];
        let decoded = decoder.push(&mut self.vae, slice, window.last)?;
        let mut out = self.trim.push(&decoded);
        if window.last {
            out.extend(self.trim.finish());
        }
        self.samples += out.len();
        Ok(out)
    }

    /// Whether the last window has run.
    pub(crate) fn done(&self) -> bool {
        self.vae.finished()
    }

    /// Samples released so far.
    pub(crate) fn samples(&self) -> usize {
        self.samples
    }
}

/// The reference's cut applied to a whole utterance.
pub fn trim_trailing_silence(samples: &[f32]) -> Vec<f32> {
    let mut trim = TrailingSilenceTrim::default();
    let mut out = trim.push(samples);
    out.extend(trim.finish());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `frontend/audio_processing.py::trim_trailing_silence`, transcribed.
    fn reference_trim(speech: &[f32]) -> Vec<f32> {
        if speech.is_empty() {
            return Vec::new();
        }
        if speech.len() < TRIM_FRAME {
            return speech[..speech.len().min(TRIM_TAIL)].to_vec();
        }
        let frames = (speech.len() - TRIM_FRAME) / TRIM_FRAME + 1;
        let speech = &speech[..frames * TRIM_FRAME];
        let mut index = frames as isize - 1;
        while index >= 0 {
            let frame = &speech[index as usize * TRIM_FRAME..(index as usize + 1) * TRIM_FRAME];
            if frame.iter().map(|s| s.abs()).sum::<f32>() / TRIM_FRAME as f32 > TRIM_THRESHOLD {
                break;
            }
            index -= 1;
        }
        if index < 0 {
            return speech[..TRIM_TAIL.min(speech.len())].to_vec();
        }
        let end = (index as usize * TRIM_FRAME + TRIM_FRAME + TRIM_TAIL).min(speech.len());
        speech[..end].to_vec()
    }

    fn signal(loud: &[(usize, usize)], len: usize) -> Vec<f32> {
        let mut samples = vec![0f32; len];
        for &(from, to) in loud {
            for (index, sample) in samples[from..to].iter_mut().enumerate() {
                *sample = if index % 2 == 0 { 0.2 } else { -0.2 };
            }
        }
        samples
    }

    #[test]
    fn forward_trim_equals_the_reference_cut() {
        let cases = [
            signal(&[(0, 30_000)], 120_000),
            signal(&[(0, 30_000), (70_000, 75_000)], 120_000),
            signal(&[(0, 119_000)], 120_000),
            signal(&[], 50_000),
            signal(&[(100, 900)], 3_000),
            signal(&[(0, 4_000)], 4_500),
            signal(&[(40_000, 44_100)], 44_100 + 17),
        ];
        for (case, samples) in cases.iter().enumerate() {
            let expected = reference_trim(samples);
            assert_eq!(trim_trailing_silence(samples), expected, "case {case}");
            // In pieces of awkward sizes, the released audio is the same.
            let mut trim = TrailingSilenceTrim::default();
            let mut out = Vec::new();
            for piece in samples.chunks(3_527) {
                out.extend(trim.push(piece));
            }
            out.extend(trim.finish());
            assert_eq!(out, expected, "case {case} in pieces");
        }
    }

    #[test]
    fn nothing_is_released_past_the_tail_of_the_last_loud_frame() {
        let mut trim = TrailingSilenceTrim::default();
        let samples = signal(&[(0, TRIM_FRAME)], 10 * TRIM_FRAME);
        let out = trim.push(&samples);
        assert_eq!(out.len(), TRIM_FRAME + TRIM_TAIL);
        assert!(trim.finish().is_empty());
    }

    #[test]
    fn windows_wait_for_latents_and_the_stop_flushes() {
        let stream = MingAudioStream::default();
        assert_eq!(stream.next_window(FIRST_WINDOW_LATENTS - 1, false), None);
        assert_eq!(
            stream.next_window(FIRST_WINDOW_LATENTS, false),
            Some(MingWindow {
                from: 0,
                to: FIRST_WINDOW_LATENTS,
                last: false
            })
        );
        assert_eq!(
            stream.next_window(2, true),
            Some(MingWindow {
                from: 0,
                to: 2,
                last: true
            })
        );
    }
}
