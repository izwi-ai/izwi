//! Windowed SNAC decoding, for streamed and final-only requests alike.
//!
//! The reference decodes a finished utterance in one SNAC call and drops its
//! first frame (2,048 samples); its streamer instead decodes the last four
//! frames after each new one and keeps the second, which loses the final
//! two frames. Here every request decodes in windows as frames arrive:
//!
//! * a window hands over frames `start..end`, decoding them with up to
//!   `left_context` frames before and `right_context` after;
//! * the decoder's noise is drawn per absolute frame ([`super::noise`]), so
//!   with enough context a window reproduces the whole-utterance decode;
//! * frame 0 is never handed over, as upstream drops it, and once the model
//!   has stopped the last window takes every remaining frame.
//!
//! The same seed therefore yields the same samples whether a request
//! streams or not. Windows grow from a small first one (quick first audio)
//! to `max_frames`.

use super::frames::SnacCodes;
use super::noise::snac_noise;
use super::snac::SnacDecoder;
use crate::error::{Error, Result};

/// When a request's windows run and what each one decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maya1WindowPlan {
    /// Frames the first window hands over; each later window doubles the
    /// previous one up to `max_frames`.
    pub first_frames: usize,
    pub max_frames: usize,
    pub left_context: usize,
    pub right_context: usize,
}

impl Maya1WindowPlan {
    /// Frames window `index` (from 0) hands over.
    pub fn window_frames(&self, index: usize) -> usize {
        let doubled = self
            .first_frames
            .checked_shl(index.min(16) as u32)
            .unwrap_or(usize::MAX);
        doubled.min(self.max_frames).max(1)
    }
}

/// The default plan. Three frames of context either side cover SNAC's
/// receptive field (about ±10 latent steps, 2.5 frames), so a window matches
/// the whole-utterance decode (`reference_tests`); first audio needs six
/// frames (42 tokens).
pub const MAYA1_WINDOW_PLAN: Maya1WindowPlan = Maya1WindowPlan {
    first_frames: 2,
    max_frames: 16,
    left_context: 3,
    right_context: 3,
};

/// The frames one window decodes and the ones it hands over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maya1Window {
    /// First decoded frame (left context included).
    pub first: usize,
    /// Handed-over frames `start..end`.
    pub start: usize,
    pub end: usize,
    /// One past the last decoded frame (right context included).
    pub decoded_end: usize,
    /// Whether this window ends the utterance.
    pub last: bool,
}

impl Maya1Window {
    pub fn decoded_frames(&self) -> usize {
        self.decoded_end - self.first
    }
}

/// One request's decoding progress.
#[derive(Debug, Clone)]
pub struct Maya1AudioStream {
    plan: Maya1WindowPlan,
    seed: u64,
    /// Frames handed over, frame 0 (dropped) included.
    emitted: usize,
    windows: usize,
    samples: usize,
    done: bool,
}

impl Maya1AudioStream {
    pub fn new(plan: Maya1WindowPlan, seed: u64) -> Self {
        Self {
            plan,
            seed,
            emitted: 1,
            windows: 0,
            samples: 0,
            done: false,
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn windows(&self) -> usize {
        self.windows
    }

    /// Samples handed over so far.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// The next window, if one can run with `available` whole frames and
    /// the model `stopped` or not.
    pub fn next_window(&self, available: usize, stopped: bool) -> Option<Maya1Window> {
        if self.done {
            return None;
        }
        let start = self.emitted;
        let (end, decoded_end, last) = if stopped {
            (available.max(start), available, true)
        } else {
            let end = start + self.plan.window_frames(self.windows);
            let decoded_end = end + self.plan.right_context;
            if available < decoded_end {
                return None;
            }
            (end, decoded_end, false)
        };
        Some(Maya1Window {
            first: start.saturating_sub(self.plan.left_context),
            start,
            end,
            decoded_end,
            last,
        })
    }

    /// Decode `window` from `codes` (every whole frame so far) and hand over
    /// its frames' samples.
    pub fn decode(
        &mut self,
        snac: &SnacDecoder,
        codes: &SnacCodes,
        window: Maya1Window,
    ) -> Result<Vec<f32>> {
        let per_frame = snac.config().samples_per_frame();
        if window.end == window.start {
            return self.accept(window, &[], per_frame);
        }
        if window.decoded_end > codes.frames() {
            return Err(Error::InferenceError(
                "Maya1 audio window outran its frames".into(),
            ));
        }
        let rows = codes.slice(window.first, window.decoded_frames());
        let noise = snac_noise(snac.config(), &[(window.first, &rows, self.seed)]);
        let decoded = snac.decode(&[&rows], &noise)?;
        self.accept(window, &decoded[0], per_frame)
    }

    /// Record `window` as handed over and return its frames' samples from
    /// `decoded`, the window's whole decode.
    pub fn accept(
        &mut self,
        window: Maya1Window,
        decoded: &[f32],
        per_frame: usize,
    ) -> Result<Vec<f32>> {
        if window.start != self.emitted || self.done {
            return Err(Error::InferenceError(
                "Maya1 audio window out of order".into(),
            ));
        }
        let skip = (window.start - window.first) * per_frame;
        let keep = (window.end - window.start) * per_frame;
        let samples = decoded
            .get(skip..skip + keep)
            .ok_or_else(|| Error::InferenceError("Maya1 window decoded too few samples".into()))?
            .to_vec();
        self.emitted = window.end;
        self.windows += 1;
        self.samples += samples.len();
        self.done = window.last;
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_double_up_to_the_cap() {
        let sizes = (0..6)
            .map(|i| MAYA1_WINDOW_PLAN.window_frames(i))
            .collect::<Vec<_>>();
        assert_eq!(sizes, vec![2, 4, 8, 16, 16, 16]);
    }

    #[test]
    fn windows_wait_for_right_context_and_flush_at_stop() {
        let stream = Maya1AudioStream::new(MAYA1_WINDOW_PLAN, 1);
        // First window: frames 1..3, decoded 0..6.
        assert_eq!(stream.next_window(5, false), None);
        let first = stream.next_window(6, false).unwrap();
        assert_eq!(
            (first.first, first.start, first.end, first.decoded_end),
            (0, 1, 3, 6)
        );
        assert!(!first.last);
        let mut stream = stream;
        stream.accept(first, &[0.0; 6 * 4], 4).unwrap();
        // Stopped with 4 frames: the rest (frame 3) with what context exists.
        let last = stream.next_window(4, true).unwrap();
        assert_eq!(
            (last.first, last.start, last.end, last.decoded_end),
            (0, 3, 4, 4)
        );
        assert!(last.last);
        let tail = stream.accept(last, &[1.0; 4 * 4], 4).unwrap();
        assert_eq!(tail.len(), 4);
        assert!(stream.done() && stream.next_window(9, true).is_none());
        assert_eq!(stream.samples(), 2 * 4 + 4);
    }

    #[test]
    fn a_window_hands_over_only_its_own_frames() {
        let mut stream = Maya1AudioStream::new(MAYA1_WINDOW_PLAN, 1);
        let window = stream.next_window(6, false).unwrap();
        let decoded = (0..6 * 2).map(|i| i as f32).collect::<Vec<_>>();
        assert_eq!(
            stream.accept(window, &decoded, 2).unwrap(),
            vec![2.0, 3.0, 4.0, 5.0]
        );
        assert!(stream.accept(window, &decoded, 2).is_err());
    }
}
