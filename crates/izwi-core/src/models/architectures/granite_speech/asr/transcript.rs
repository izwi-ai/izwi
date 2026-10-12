//! Parsers for Granite Speech rich transcript output: plus's speaker turns
//! (`[Speaker N]: …`) and word timestamps (`word [T:N]`, silences `_ [T:N]`).

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechSegment {
    pub speaker: Option<String>,
    pub text: String,
}

/// A word timed on its transcript's timeline: it starts where the previous
/// tag (word or silence) ended and ends at its own tag.
#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechTimestampWord {
    pub word: String,
    pub start_time_seconds: f32,
    pub end_time_seconds: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechParsedTranscript {
    /// The transcript without timestamp tags or silence markers.
    pub text: String,
    pub segments: Vec<GraniteSpeechSegment>,
    pub timestamp_words: Vec<GraniteSpeechTimestampWord>,
}

/// One chunk's words, timed relative to the chunk's start.
#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechTimestampChunk {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub words: Vec<GraniteSpeechTimestampWord>,
}

/// Silence entry the timestamp task emits between words.
const SILENCE_MARKER: &str = "_";
/// Tags carry the end time in centiseconds modulo 1000, so they roll over
/// every 10 s (plus model card).
const TIMESTAMP_ROLLOVER_CENTISECONDS: u64 = 1_000;

pub fn parse_granite_speech_output(raw: &str) -> GraniteSpeechParsedTranscript {
    let cleaned = cleanup_transcript_text(raw);
    let timestamp_words = parse_timestamp_words(&strip_speaker_markers(&cleaned));
    let tags = timestamp_tags(&cleaned);
    let text = if tags.is_empty() {
        cleaned
    } else {
        strip_timestamp_markup(&cleaned, &tags)
    };
    GraniteSpeechParsedTranscript {
        segments: parse_speaker_segments(&text),
        timestamp_words,
        text,
    }
}

/// Place each chunk's words on the input timeline and keep one copy of the
/// audio two neighbouring chunks both transcribed: the overlap is split at
/// its midpoint and a word belongs to the chunk whose side holds its end.
/// A chunk's truncated last word ends past the midpoint, so the next chunk's
/// complete copy is the one kept.
pub fn merge_granite_speech_timestamp_chunks(
    chunks: &[GraniteSpeechTimestampChunk],
) -> Vec<GraniteSpeechTimestampWord> {
    let mut merged = Vec::new();
    let mut previous_end = 0.0f32;
    for (index, chunk) in chunks.iter().enumerate() {
        let lower = index
            .checked_sub(1)
            .map(|previous| overlap_boundary(&chunks[previous], chunk));
        let upper = chunks
            .get(index + 1)
            .map(|next| overlap_boundary(chunk, next));
        for word in &chunk.words {
            let end = chunk.start_seconds + word.end_time_seconds;
            if lower.is_some_and(|boundary| end <= boundary)
                || upper.is_some_and(|boundary| end > boundary)
            {
                continue;
            }
            let start = (chunk.start_seconds + word.start_time_seconds)
                .max(previous_end)
                .min(end);
            previous_end = previous_end.max(end);
            merged.push(GraniteSpeechTimestampWord {
                word: word.word.clone(),
                start_time_seconds: start,
                end_time_seconds: end,
            });
        }
    }
    merged
}

fn overlap_boundary(
    earlier: &GraniteSpeechTimestampChunk,
    later: &GraniteSpeechTimestampChunk,
) -> f32 {
    if later.start_seconds < earlier.end_seconds {
        (later.start_seconds + earlier.end_seconds) / 2.0
    } else {
        earlier.end_seconds
    }
}

fn cleanup_transcript_text(raw: &str) -> String {
    raw.replace("<|end_of_text|>", "")
        .replace("<|endoftext|>", "")
        .replace("<|start_of_role|>assistant<|end_of_role|>", "")
        .trim()
        .to_string()
}

fn parse_speaker_segments(text: &str) -> Vec<GraniteSpeechSegment> {
    let mut segments = Vec::new();
    let mut cursor = 0usize;
    let mut current_speaker: Option<String> = None;

    while let Some(relative_start) = text[cursor..].find("[Speaker ") {
        let marker_start = cursor + relative_start;
        if marker_start > cursor {
            push_segment(
                &mut segments,
                current_speaker.clone(),
                &text[cursor..marker_start],
            );
        }

        let Some(relative_end) = text[marker_start..].find("]:") else {
            break;
        };
        let marker_end = marker_start + relative_end + 2;
        let label = &text[marker_start + 1..marker_end - 2];
        current_speaker = Some(label.trim().to_string());
        cursor = marker_end;
    }

    if cursor < text.len() {
        push_segment(&mut segments, current_speaker, &text[cursor..]);
    }

    segments
}

fn push_segment(segments: &mut Vec<GraniteSpeechSegment>, speaker: Option<String>, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    segments.push(GraniteSpeechSegment {
        speaker,
        text: text.to_string(),
    });
}

fn strip_speaker_markers(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(relative_start) = text[cursor..].find("[Speaker ") {
        let marker_start = cursor + relative_start;
        stripped.push_str(&text[cursor..marker_start]);
        let Some(relative_end) = text[marker_start..].find("]:") else {
            stripped.push_str(&text[marker_start..]);
            return stripped;
        };
        cursor = marker_start + relative_end + 2;
    }
    stripped.push_str(&text[cursor..]);
    stripped
}

/// A `[T:N]` tag and its byte span. IBM's parser accepts a lowercase `t` and
/// whitespace inside the brackets (`\[T:\s*(\d+)\s*\]`, case-insensitive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TimestampTag {
    start: usize,
    end: usize,
    centiseconds: u64,
}

fn timestamp_tags(text: &str) -> Vec<TimestampTag> {
    text.match_indices('[')
        .filter_map(|(start, _)| parse_timestamp_tag(text, start))
        .collect()
}

fn parse_timestamp_tag(text: &str, start: usize) -> Option<TimestampTag> {
    let inner = &text[start + 1..];
    let rest = inner
        .strip_prefix("T:")
        .or_else(|| inner.strip_prefix("t:"))?
        .trim_start();
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let centiseconds = rest[..digits].parse::<u64>().ok()?;
    let after_tag = rest[digits..].trim_start().strip_prefix(']')?;
    Some(TimestampTag {
        start,
        end: text.len() - after_tag.len(),
        centiseconds,
    })
}

/// The card's reference unwrap (plus model card, task 3) generalised to IBM's
/// tag pattern: each tag is a word's end time modulo 10 s, silence entries
/// included, and 10 s is added whenever time would otherwise run backwards.
fn parse_timestamp_words(text: &str) -> Vec<GraniteSpeechTimestampWord> {
    let mut words = Vec::new();
    let mut cursor = 0usize;
    let mut rollover = 0u64;
    let mut previous_end = 0u64;
    for tag in timestamp_tags(text) {
        let segment = text[cursor..tag.start]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        cursor = tag.end;
        let wrapped = tag.centiseconds.saturating_add(rollover);
        if wrapped < previous_end {
            let periods = (previous_end - wrapped).div_ceil(TIMESTAMP_ROLLOVER_CENTISECONDS);
            rollover =
                rollover.saturating_add(periods.saturating_mul(TIMESTAMP_ROLLOVER_CENTISECONDS));
        }
        let end = tag.centiseconds.saturating_add(rollover);
        let start = previous_end;
        previous_end = end;
        if segment.is_empty() || segment == SILENCE_MARKER {
            continue;
        }
        words.push(GraniteSpeechTimestampWord {
            word: segment,
            start_time_seconds: centiseconds_to_seconds(start),
            end_time_seconds: centiseconds_to_seconds(end),
        });
    }
    words
}

fn centiseconds_to_seconds(centiseconds: u64) -> f32 {
    (centiseconds as f64 / 100.0) as f32
}

/// The transcript the tags annotate: tags and silence entries removed.
fn strip_timestamp_markup(text: &str, tags: &[TimestampTag]) -> String {
    let mut kept = Vec::new();
    let mut cursor = 0usize;
    for tag in tags {
        let segment = text[cursor..tag.start].trim();
        if segment != SILENCE_MARKER {
            kept.push(segment);
        }
        cursor = tag.end;
    }
    kept.push(text[cursor..].trim());
    kept.join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(word: &str, start: f32, end: f32) -> GraniteSpeechTimestampWord {
        GraniteSpeechTimestampWord {
            word: word.to_string(),
            start_time_seconds: start,
            end_time_seconds: end,
        }
    }

    #[test]
    fn parser_extracts_speaker_segments() {
        let parsed =
            parse_granite_speech_output("[Speaker 1]: Hello how are you [Speaker 2]: I am fine");
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].speaker.as_deref(), Some("Speaker 1"));
        assert_eq!(parsed.segments[0].text, "Hello how are you");
        assert_eq!(parsed.segments[1].speaker.as_deref(), Some("Speaker 2"));
        assert_eq!(
            parsed.text,
            "[Speaker 1]: Hello how are you [Speaker 2]: I am fine"
        );
        assert!(parsed.timestamp_words.is_empty());
    }

    /// The card's prompt example and its reference loop's output.
    #[test]
    fn parser_matches_the_card_example_and_strips_tags_from_text() {
        let parsed = parse_granite_speech_output("hello [T:45] world [T:82]<|end_of_text|>");
        assert_eq!(
            parsed.timestamp_words,
            vec![word("hello", 0.0, 0.45), word("world", 0.45, 0.82)]
        );
        assert_eq!(parsed.text, "hello world");
    }

    #[test]
    fn parser_unwraps_timestamp_rollover() {
        let parsed = parse_granite_speech_output("hello [T:990] world [T:005]");
        assert_eq!(
            parsed.timestamp_words,
            vec![word("hello", 0.0, 9.9), word("world", 9.9, 10.05)]
        );
    }

    /// A silence tag must advance the rollover state: here the silence ends
    /// at 9.80 s, so the next word's `[T:020]` is 10.20 s, not 0.20 s.
    #[test]
    fn silence_tags_update_rollover_and_open_a_gap() {
        let parsed = parse_granite_speech_output("_ [T:450] hello [T:520] _ [T:980] world [T:020]");
        assert_eq!(
            parsed.timestamp_words,
            vec![word("hello", 4.5, 5.2), word("world", 9.8, 10.2)]
        );
        assert_eq!(parsed.text, "hello world");

        // A silence longer than the rollover period: it ends at 12.50 s
        // (`[T:250]` after 3.00 s), so `world [T:400]` is 14.00 s. Skipping the
        // silence before unwrapping dated it 4.00 s.
        let parsed = parse_granite_speech_output("hello [T:300] _ [T:250] world [T:400]");
        assert_eq!(
            parsed.timestamp_words,
            vec![word("hello", 0.0, 3.0), word("world", 12.5, 14.0)]
        );
    }

    #[test]
    fn parser_accepts_tags_without_spaces_and_ibm_tag_spacing() {
        let parsed = parse_granite_speech_output("hello[T:45] big[t: 60 ] world [T:82]");
        assert_eq!(
            parsed.timestamp_words,
            vec![
                word("hello", 0.0, 0.45),
                word("big", 0.45, 0.6),
                word("world", 0.6, 0.82),
            ]
        );
        assert_eq!(parsed.text, "hello big world");
        // Not timestamp tags: left in the text and not parsed as times.
        let parsed = parse_granite_speech_output("see [T:] and [Tx:4] and [T:12");
        assert!(parsed.timestamp_words.is_empty());
        assert_eq!(parsed.text, "see [T:] and [Tx:4] and [T:12");
    }

    #[test]
    fn parser_removes_speaker_labels_from_timestamp_words() {
        let parsed = parse_granite_speech_output("[Speaker 1]: hello [T:045]");
        assert_eq!(parsed.timestamp_words, vec![word("hello", 0.0, 0.45)]);
    }

    #[test]
    fn merge_offsets_chunks_and_keeps_one_copy_of_the_overlap() {
        // Chunk 0 covers 0-24 s, chunk 1 covers 21-45 s: the shared 21-24 s
        // splits at 22.5 s. "cut" is chunk 0's truncated copy of "cutoff".
        let chunks = vec![
            GraniteSpeechTimestampChunk {
                start_seconds: 0.0,
                end_seconds: 24.0,
                words: vec![
                    word("one", 0.0, 20.5),
                    word("two", 20.5, 22.0),
                    word("three", 22.0, 23.0),
                    word("cut", 23.0, 24.0),
                ],
            },
            GraniteSpeechTimestampChunk {
                start_seconds: 21.0,
                end_seconds: 45.0,
                words: vec![
                    word("two", 0.0, 1.0),
                    word("three", 1.0, 2.0),
                    word("cutoff", 2.0, 3.4),
                    word("four", 3.4, 5.0),
                ],
            },
        ];
        assert_eq!(
            merge_granite_speech_timestamp_chunks(&chunks),
            vec![
                word("one", 0.0, 20.5),
                word("two", 20.5, 22.0),
                word("three", 22.0, 23.0),
                word("cutoff", 23.0, 24.4),
                word("four", 24.4, 26.0),
            ]
        );
    }

    #[test]
    fn merge_drops_words_timed_past_a_chunk_that_has_no_overlap() {
        let chunks = vec![
            GraniteSpeechTimestampChunk {
                start_seconds: 0.0,
                end_seconds: 10.0,
                words: vec![word("inside", 0.0, 9.0), word("beyond", 9.0, 11.0)],
            },
            GraniteSpeechTimestampChunk {
                start_seconds: 10.0,
                end_seconds: 20.0,
                words: vec![word("next", 0.5, 1.0)],
            },
        ];
        assert_eq!(
            merge_granite_speech_timestamp_chunks(&chunks),
            vec![word("inside", 0.0, 9.0), word("next", 10.5, 11.0)]
        );
    }
}
