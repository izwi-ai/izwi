//! Parsers for Granite Speech rich transcript output.

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechSegment {
    pub speaker: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechTimestampWord {
    pub word: String,
    pub end_time_seconds: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechParsedTranscript {
    pub text: String,
    pub segments: Vec<GraniteSpeechSegment>,
    pub timestamp_words: Vec<GraniteSpeechTimestampWord>,
}

pub fn parse_granite_speech_output(raw: &str) -> GraniteSpeechParsedTranscript {
    let text = cleanup_transcript_text(raw);
    GraniteSpeechParsedTranscript {
        segments: parse_speaker_segments(&text),
        timestamp_words: parse_timestamp_words(&text),
        text,
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

fn parse_timestamp_words(text: &str) -> Vec<GraniteSpeechTimestampWord> {
    let mut words = Vec::new();
    let mut pending_word = String::new();
    let mut rollover = 0.0f32;
    let mut last_time = 0.0f32;

    for token in text.split_whitespace() {
        if let Some(centiseconds) = parse_timestamp_token(token) {
            if pending_word.trim().is_empty() {
                continue;
            }
            let mut seconds = centiseconds as f32 / 100.0 + rollover;
            while seconds < last_time {
                rollover += 10.0;
                seconds = centiseconds as f32 / 100.0 + rollover;
            }
            last_time = seconds;
            words.push(GraniteSpeechTimestampWord {
                word: pending_word.trim().to_string(),
                end_time_seconds: seconds,
            });
            pending_word.clear();
        } else {
            if !pending_word.is_empty() {
                pending_word.push(' ');
            }
            pending_word.push_str(token);
        }
    }

    words
}

fn parse_timestamp_token(token: &str) -> Option<u32> {
    let inner = token
        .strip_prefix("[T:")
        .and_then(|value| value.strip_suffix(']'))?;
    inner.parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_extracts_speaker_segments() {
        let parsed =
            parse_granite_speech_output("[Speaker 1]: Hello how are you [Speaker 2]: I am fine");
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].speaker.as_deref(), Some("Speaker 1"));
        assert_eq!(parsed.segments[0].text, "Hello how are you");
        assert_eq!(parsed.segments[1].speaker.as_deref(), Some("Speaker 2"));
    }

    #[test]
    fn parser_unwraps_timestamp_rollover() {
        let parsed = parse_granite_speech_output("hello [T:990] world [T:005]");
        assert_eq!(
            parsed.timestamp_words,
            vec![
                GraniteSpeechTimestampWord {
                    word: "hello".to_string(),
                    end_time_seconds: 9.9,
                },
                GraniteSpeechTimestampWord {
                    word: "world".to_string(),
                    end_time_seconds: 10.05,
                },
            ]
        );
    }
}
