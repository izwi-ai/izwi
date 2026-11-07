//! The pieces of a streamed spoken reply (SPK-F2, F13, F20):
//!
//! - [`SpeechSegmenter`] cuts the reply into speakable segments while the
//!   text model is still writing it;
//! - [`SegmentJoiner`] joins the segments' audio with the long-form pause and
//!   fades, so sentences neither click nor run together;
//! - [`SpokenLedger`] maps the reply's audio time back to its text, so an
//!   interrupted reply is remembered only up to what the user heard.

pub(crate) use izwi_core::text_segments::SpeechSegmenter;

use crate::api::tts_long_form::{
    fade_in_segment, fade_out_segment, segment_fade_samples, segment_pause_samples,
};

/// Joins a reply's segment audio as it streams: a later segment opens with
/// the long-form pause and fades in, and every segment fades out. The last
/// fade-length samples of the audio sent so far are held back until the
/// segment's end is known.
#[derive(Debug, Default)]
pub(crate) struct SegmentJoiner {
    held: Vec<f32>,
    segments: usize,
    segment_has_audio: bool,
}

impl SegmentJoiner {
    pub(crate) fn begin_segment(&mut self) {
        self.segments += 1;
        self.segment_has_audio = false;
    }

    /// The audio to send for `samples` of the current segment.
    pub(crate) fn push(&mut self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let mut out = std::mem::take(&mut self.held);
        if self.segment_has_audio || samples.is_empty() {
            out.extend_from_slice(samples);
        } else {
            self.segment_has_audio = true;
            if self.segments > 1 {
                out.resize(out.len() + segment_pause_samples(sample_rate), 0.0);
                let opening = out.len();
                out.extend_from_slice(samples);
                fade_in_segment(&mut out[opening..], sample_rate);
            } else {
                out.extend_from_slice(samples);
            }
        }
        let hold = segment_fade_samples(sample_rate).min(out.len());
        self.held = out.split_off(out.len() - hold);
        out
    }

    /// The current segment's held-back tail, faded out.
    pub(crate) fn end_segment(&mut self, sample_rate: u32) -> Vec<f32> {
        let mut tail = std::mem::take(&mut self.held);
        fade_out_segment(&mut tail, sample_rate);
        tail
    }
}

/// The reply's spoken segments on its audio timeline.
#[derive(Debug, Default, Clone)]
pub(crate) struct SpokenLedger {
    segments: Vec<SpokenSegment>,
    audio_ms: f64,
}

#[derive(Debug, Clone)]
struct SpokenSegment {
    text: String,
    start_ms: f64,
    end_ms: f64,
}

impl SpokenLedger {
    pub(crate) fn begin_segment(&mut self, text: &str) {
        self.segments.push(SpokenSegment {
            text: text.to_string(),
            start_ms: self.audio_ms,
            end_ms: self.audio_ms,
        });
    }

    /// Account for `duration_ms` of the current segment's audio sent.
    pub(crate) fn audio_sent(&mut self, duration_ms: f64) {
        self.audio_ms += duration_ms;
        if let Some(segment) = self.segments.last_mut() {
            segment.end_ms = self.audio_ms;
        }
    }

    /// Milliseconds of the reply's audio sent.
    pub(crate) fn audio_ms(&self) -> f64 {
        self.audio_ms
    }

    /// The text of every segment that has started sounding.
    #[cfg(test)]
    pub(crate) fn spoken_text(&self) -> String {
        self.heard_text(f64::INFINITY)
    }

    /// The text heard by `heard_ms` of the reply's audio: the segments that
    /// finished playing and, of the one playing, the share of its words its
    /// played audio covers.
    pub(crate) fn heard_text(&self, heard_ms: f64) -> String {
        let mut heard = Vec::new();
        for segment in &self.segments {
            if heard_ms >= segment.end_ms {
                if segment.end_ms > segment.start_ms {
                    heard.push(segment.text.clone());
                }
                continue;
            }
            if heard_ms > segment.start_ms {
                let words: Vec<&str> = segment.text.split_whitespace().collect();
                let share = (heard_ms - segment.start_ms) / (segment.end_ms - segment.start_ms);
                let count = (words.len() as f64 * share).floor() as usize;
                if count > 0 {
                    heard.push(words[..count].join(" "));
                }
            }
            break;
        }
        heard.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPK-F20: a later segment opens with the pause and fades in, every
    /// segment fades out, and no sample is lost or reordered.
    #[test]
    fn joined_segments_pause_and_fade_without_losing_audio() {
        let rate = 1_000;
        let fade = segment_fade_samples(rate);
        let pause = segment_pause_samples(rate);
        let mut joiner = SegmentJoiner::default();
        let mut out = Vec::new();
        joiner.begin_segment();
        out.extend(joiner.push(&[1.0; 50], rate));
        assert_eq!(out.len(), 50 - fade, "the fade tail is held back");
        out.extend(joiner.push(&[1.0; 50], rate));
        out.extend(joiner.end_segment(rate));
        joiner.begin_segment();
        out.extend(joiner.push(&[1.0; 30], rate));
        out.extend(joiner.end_segment(rate));

        assert_eq!(out.len(), 100 + pause + 30);
        assert_eq!(out[0], 1.0, "the first segment does not fade in");
        assert!(out[99] < 0.5, "the first segment fades out");
        assert!(out[100..100 + pause].iter().all(|sample| *sample == 0.0));
        assert!(out[100 + pause] < 0.5, "the second segment fades in");
        assert_eq!(out[100 + pause + fade], 1.0);
    }

    /// SPK-F13: what was heard maps back to the reply's text: whole
    /// finished segments, and the played share of the playing one.
    #[test]
    fn heard_audio_maps_back_to_the_reply_text() {
        let mut ledger = SpokenLedger::default();
        ledger.begin_segment("Sure, Dr. Lee.");
        ledger.audio_sent(1_000.0);
        ledger.begin_segment("The total is three units today.");
        ledger.audio_sent(1_200.0);
        ledger.audio_sent(800.0);
        ledger.begin_segment("Never synthesized.");

        assert_eq!(ledger.heard_text(0.0), "");
        assert_eq!(ledger.heard_text(999.0), "Sure, Dr.");
        assert_eq!(ledger.heard_text(1_000.0), "Sure, Dr. Lee.");
        assert_eq!(ledger.heard_text(2_000.0), "Sure, Dr. Lee. The total is");
        assert_eq!(
            ledger.heard_text(5_000.0),
            "Sure, Dr. Lee. The total is three units today."
        );
        assert_eq!(ledger.spoken_text(), ledger.heard_text(3_000.0));
    }
}
