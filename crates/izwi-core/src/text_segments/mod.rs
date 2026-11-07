//! Text cut into speakable pieces: sentences ([`sentence_spans`]), packed
//! segments ([`pack_sentences`]), and segments cut while the text still
//! streams in ([`SpeechSegmenter`]).

use std::collections::VecDeque;

mod sentences;

pub use sentences::{pack_sentences, sentence_spans};

/// Characters one segment carries after the reply's first sentence.
pub const SEGMENT_MAX_CHARS: usize = 240;

/// Cuts a reply into speakable segments as its text arrives.
///
/// Before the end of the text a segment holds only complete sentences: the
/// reply's first sentence alone, however short (the first audio waits for
/// nothing else), then whatever complete sentences have arrived, packed up
/// to [`SEGMENT_MAX_CHARS`]. A sentence that grows past the cap without ending
/// is spoken in clause- or word-sized pieces as it grows. Abbreviations and
/// decimals do not end a sentence (`sentence_spans`).
#[derive(Debug, Default)]
pub struct SpeechSegmenter {
    text: String,
    /// Bytes of `text` already handed out as segments.
    consumed: usize,
    first_emitted: bool,
    /// Pieces of an over-long complete sentence still to hand out.
    queued: VecDeque<String>,
}

impl SpeechSegmenter {
    pub fn push(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// The next segment ready to speak; `finished` once the whole reply has
    /// been pushed. Call until `None`. Text with nothing to say (a stray
    /// list marker or punctuation) is never a segment of its own.
    pub fn next_segment(&mut self, finished: bool) -> Option<String> {
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
}
