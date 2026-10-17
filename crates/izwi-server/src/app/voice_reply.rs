//! The pieces of a streamed spoken reply (SPK-F2, F13, F20):
//!
//! - [`SpeechSegmenter`] cuts the reply into speakable segments while the
//!   text model is still writing it;
//! - [`SegmentJoiner`] joins the segments' audio with the long-form pause and
//!   fades, so sentences neither click nor run together;
//! - [`SpokenLedger`] maps the reply's audio time back to its text, so an
//!   interrupted reply is remembered only up to what the user heard.

use std::collections::VecDeque;

use crate::api::sentence_split::{pack_sentences, sentence_spans};
use crate::api::tts_long_form::{
    fade_in_segment, fade_out_segment, segment_fade_samples, segment_pause_samples,
};

/// Characters one segment carries after the reply's first sentence.
pub(crate) const SEGMENT_MAX_CHARS: usize = 240;

/// Cuts a reply into speakable segments as its text arrives.
///
/// Before the end of the text a segment holds only complete sentences: the
/// reply's first sentence alone, however short (the first audio waits for
/// nothing else), then whatever complete sentences have arrived, packed up
/// to [`SEGMENT_MAX_CHARS`]. A sentence that grows past the cap without ending
/// is spoken in clause- or word-sized pieces as it grows. Abbreviations and
/// decimals do not end a sentence (`sentence_spans`).
#[derive(Debug, Default)]
pub(crate) struct SpeechSegmenter {
    text: String,
    /// Bytes of `text` already handed out as segments.
    consumed: usize,
    first_emitted: bool,
    /// Pieces of an over-long complete sentence still to hand out.
    queued: VecDeque<String>,
}

impl SpeechSegmenter {
    pub(crate) fn push(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// The next segment ready to speak; `finished` once the whole reply has
    /// been pushed. Call until `None`. Text with nothing to say (a stray
    /// list marker or punctuation) is never a segment of its own.
    pub(crate) fn next_segment(&mut self, finished: bool) -> Option<String> {
        loop {
            let segment = self.next_piece(finished)?;
            if segment.chars().any(char::is_alphanumeric) {
                return Some(segment);
            }
        }
    }

    fn next_piece(&mut self, finished: bool) -> Option<String> {
        if let Some(piece) = self.queued.pop_front() {
            return Some(piece);
        }
        let pending = &self.text[self.consumed..];
        let spans = sentence_spans(pending);
        let complete = if finished {
            spans.len()
        } else {
            spans.len().saturating_sub(1)
        };
        if complete == 0 {
            let open = spans.first()?.clone();
            let cut = overflow_cut(&pending[open.clone()])?;
            let segment = pending[open.start..open.start + cut].trim().to_string();
            self.consumed += open.start + cut;
            self.first_emitted = true;
            return Some(segment);
        }

        let first = spans[0].clone();
        let mut end = first.end;
        if self.first_emitted {
            for span in &spans[1..complete] {
                if pending[first.start..span.end].chars().count() > SEGMENT_MAX_CHARS {
                    break;
                }
                end = span.end;
            }
        }
        let text = pending[first.start..end].to_string();
        self.consumed += end;
        self.first_emitted = true;
        if text.chars().count() <= SEGMENT_MAX_CHARS {
            return Some(text);
        }
        self.queued = pack_sentences(&text, SEGMENT_MAX_CHARS, &|_| true).into();
        self.queued.pop_front()
    }
}

/// Where to cut an unfinished sentence that has grown past the cap: after
/// the last clause break (`,` `;` `:` before a space) within the cap, else
/// at its last word break within the cap, else at the cap. `None` while the
/// sentence still fits.
fn overflow_cut(sentence: &str) -> Option<usize> {
    let mut chars = sentence.char_indices();
    let (limit, _) = chars.nth(SEGMENT_MAX_CHARS)?;
    let head = &sentence[..limit];
    let clause = head
        .char_indices()
        .filter(|(index, ch)| {
            matches!(ch, ',' | ';' | ':')
                && sentence[index + ch.len_utf8()..].starts_with(char::is_whitespace)
        })
        .map(|(index, ch)| index + ch.len_utf8())
        .next_back();
    let word = head
        .char_indices()
        .filter(|(_, ch)| ch.is_whitespace())
        .map(|(index, _)| index)
        .next_back()
        .filter(|index| *index > 0);
    Some(clause.or(word).unwrap_or(limit))
}

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

    fn segments_of(deltas: &[&str]) -> Vec<String> {
        let mut segmenter = SpeechSegmenter::default();
        let mut out = Vec::new();
        for delta in deltas {
            segmenter.push(delta);
            while let Some(segment) = segmenter.next_segment(false) {
                out.push(segment);
            }
        }
        while let Some(segment) = segmenter.next_segment(true) {
            out.push(segment);
        }
        out
    }

    /// SPK-F2: the first sentence is spoken as soon as it is complete, alone;
    /// later complete sentences are packed; the unfinished tail waits for
    /// the end. Titles and decimals do not end a sentence.
    #[test]
    fn segments_follow_the_streamed_text_first_sentence_first() {
        let mut segmenter = SpeechSegmenter::default();
        for delta in ["Sure,", " Dr.", " Lee", ". The", " total is 3", ".14 units"] {
            segmenter.push(delta);
        }
        assert_eq!(
            segmenter.next_segment(false).as_deref(),
            Some("Sure, Dr. Lee.")
        );
        assert_eq!(
            segmenter.next_segment(false),
            None,
            "the second sentence is open"
        );
        segmenter.push(". It ships today. Anything");
        assert_eq!(
            segmenter.next_segment(false).as_deref(),
            Some("The total is 3.14 units. It ships today.")
        );
        segmenter.push(" else?");
        assert_eq!(segmenter.next_segment(false), None);
        assert_eq!(
            segmenter.next_segment(true).as_deref(),
            Some("Anything else?")
        );
        assert_eq!(segmenter.next_segment(true), None);
    }

    /// A short first sentence is spoken alone, at once: Kokoro admits the
    /// audio a short text speaks (the 40-character segment minimum that
    /// worked around its budget is gone).
    #[test]
    fn a_short_first_sentence_is_spoken_alone() {
        let mut segmenter = SpeechSegmenter::default();
        segmenter.push("Sure. I can help with");
        assert_eq!(segmenter.next_segment(false).as_deref(), Some("Sure."));
        segmenter.push(" that. Okay. Bye.");
        assert_eq!(
            segmenter.next_segment(false).as_deref(),
            Some("I can help with that. Okay.")
        );
        assert_eq!(segmenter.next_segment(true).as_deref(), Some("Bye."));
    }

    /// Any split of the reply into deltas yields the reply's words, in
    /// order, once; one-sentence replies come out whole.
    #[test]
    fn every_delta_split_speaks_the_whole_reply_once() {
        let reply = "Sure, Dr. Lee. The total is 3.14 units. It ships today, and arrives \
            tomorrow. Anything else?";
        let words: Vec<&str> = reply.split_whitespace().collect();
        for step in 1..12 {
            let chars: Vec<char> = reply.chars().collect();
            let deltas: Vec<String> = chars.chunks(step).map(|c| c.iter().collect()).collect();
            let deltas: Vec<&str> = deltas.iter().map(String::as_str).collect();
            let segments = segments_of(&deltas);
            assert_eq!(segments[0], "Sure, Dr. Lee.", "step {step}");
            let spoken: Vec<String> = segments
                .iter()
                .flat_map(|segment| segment.split_whitespace().map(str::to_string))
                .collect();
            assert_eq!(spoken, words, "step {step}");
        }
        assert_eq!(segments_of(&["Hello"]), vec!["Hello"]);
        assert!(segments_of(&["  "]).is_empty());
        assert!(segments_of(&["…", " !"]).is_empty(), "nothing to say");
    }

    /// A sentence that runs past the cap without ending is spoken in
    /// clause-sized pieces while it streams, never waiting for its end.
    #[test]
    fn a_run_on_sentence_is_spoken_in_pieces_as_it_grows() {
        let clause = "and then the parade went down the main street past the bakery, ";
        let mut segmenter = SpeechSegmenter::default();
        let mut early = Vec::new();
        for _ in 0..8 {
            segmenter.push(clause);
            while let Some(segment) = segmenter.next_segment(false) {
                early.push(segment);
            }
        }
        assert!(
            !early.is_empty(),
            "pieces are spoken before the sentence ends"
        );
        for piece in &early {
            assert!(piece.chars().count() <= SEGMENT_MAX_CHARS, "{piece}");
            assert!(piece.ends_with(','), "cut at a clause: {piece}");
        }
    }

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
