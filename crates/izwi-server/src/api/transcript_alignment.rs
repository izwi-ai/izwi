//! Word timestamps for a finished transcript by windowed forced alignment.
//!
//! Long-form ASR reports each chunk's time range and the text it committed
//! (`chunking.chunk_transcriptions`). Those pairs are the alignment windows,
//! so each window's text is aligned only against the audio it was heard in.
//! Without chunk texts the whole transcript is one window over the audio.

use std::future::Future;

use izwi_core::{AlignmentSource, AlignmentWindow, WindowedAlignment};

/// Alignment windows for `text` heard in `duration_secs` of audio: one per
/// long-form chunk when the chunks' committed texts make up the transcript,
/// otherwise the whole transcript over the whole audio.
pub(crate) fn transcript_alignment_windows(
    text: &str,
    duration_secs: f32,
    diagnostics: Option<&serde_json::Value>,
) -> Vec<AlignmentWindow> {
    let chunk_windows = chunk_alignment_windows(diagnostics);
    let chunk_words = chunk_windows
        .iter()
        .flat_map(|window| comparable_words(&window.text))
        .collect::<Vec<_>>();
    if chunk_windows.len() > 1 && chunk_words == comparable_words(text) {
        return chunk_windows;
    }
    vec![AlignmentWindow {
        start_secs: 0.0,
        end_secs: f64::from(duration_secs.max(0.0)),
        text: text.to_string(),
    }]
}

/// Align `text` by windows, handing the windows to `align` (the runtime's
/// windowed aligner) in one call. When no window could be aligned the
/// timings would all be invented, so the first window error is returned.
pub(crate) async fn align_transcript<F, Fut>(
    text: &str,
    duration_secs: f32,
    diagnostics: Option<&serde_json::Value>,
    align: F,
) -> izwi_core::Result<WindowedAlignment>
where
    F: FnOnce(Vec<AlignmentWindow>) -> Fut,
    Fut: Future<Output = izwi_core::Result<WindowedAlignment>>,
{
    let alignment = align(transcript_alignment_windows(
        text,
        duration_secs,
        diagnostics,
    ))
    .await?;
    if alignment.source() != AlignmentSource::Failed {
        return Ok(alignment);
    }
    Err(alignment
        .windows
        .into_iter()
        .find_map(|window| window.error)
        .unwrap_or_else(|| izwi_core::Error::InferenceError("forced alignment failed".into())))
}

fn chunk_alignment_windows(diagnostics: Option<&serde_json::Value>) -> Vec<AlignmentWindow> {
    let Some(chunks) = diagnostics
        .and_then(|diagnostics| {
            diagnostics
                .get("chunking")
                .or_else(|| diagnostics.get("model_diagnostics")?.get("chunking"))
        })
        .and_then(|chunking| chunking.get("chunk_transcriptions"))
        .and_then(|chunks| chunks.as_array())
    else {
        return Vec::new();
    };
    let mut windows = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let start = chunk.get("start_seconds").and_then(|value| value.as_f64());
        let end = chunk.get("end_seconds").and_then(|value| value.as_f64());
        let text = chunk.get("committed_text").and_then(|value| value.as_str());
        let (Some(start_secs), Some(end_secs), Some(text)) = (start, end, text) else {
            // A chunk without its range or committed text breaks the
            // text-to-audio mapping; align the whole transcript instead.
            return Vec::new();
        };
        if !start_secs.is_finite() || !end_secs.is_finite() || end_secs < start_secs {
            return Vec::new();
        }
        if text.trim().is_empty() {
            continue;
        }
        windows.push(AlignmentWindow {
            start_secs: start_secs.max(0.0),
            end_secs,
            text: text.to_string(),
        });
    }
    windows
}

/// Words compared case- and punctuation-insensitively, so chunk texts that
/// differ from the transcript only in cleanup still map onto it.
fn comparable_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use izwi_core::AlignedWindow;

    pub(crate) fn chunked_diagnostics(chunks: &[(f64, f64, &str)]) -> serde_json::Value {
        serde_json::json!({
            "chunking": {
                "chunk_transcriptions": chunks
                    .iter()
                    .map(|(start, end, text)| serde_json::json!({
                        "start_seconds": start,
                        "end_seconds": end,
                        "committed_text": text,
                    }))
                    .collect::<Vec<_>>()
            }
        })
    }

    /// Stub of the runtime's windowed aligner: every word 400 ms apart from
    /// its window start, each window one call.
    pub(crate) fn stub_windowed_alignment(
        windows: &[AlignmentWindow],
        duration_secs: f64,
    ) -> WindowedAlignment {
        WindowedAlignment {
            windows: windows
                .iter()
                .map(|window| {
                    let start_ms = (window.start_secs * 1000.0) as u32;
                    AlignedWindow {
                        start_ms,
                        end_ms: (window.end_secs * 1000.0) as u32,
                        source: AlignmentSource::Model,
                        words: window
                            .text
                            .split_whitespace()
                            .enumerate()
                            .map(|(idx, word)| {
                                let start = start_ms + idx as u32 * 400;
                                (word.to_string(), start, start + 300)
                            })
                            .collect(),
                        detail: None,
                        error: None,
                    }
                })
                .collect(),
            duration_ms: (duration_secs * 1000.0) as u32,
        }
    }

    #[tokio::test]
    async fn long_transcripts_align_per_chunk_window() {
        let diagnostics = chunked_diagnostics(&[
            (0.0, 29.99, "Hello there, world."),
            (27.0, 56.0, "Second chunk words."),
            (53.0, 60.0, "Tail."),
        ]);
        let text = "Hello there, world. Second chunk words. Tail.";
        let mut calls = 0;
        let alignment = align_transcript(text, 60.0, Some(&diagnostics), |windows| {
            calls += 1;
            assert_eq!(windows.len(), 3, "{windows:?}");
            assert!(windows
                .iter()
                .all(|window| window.end_secs - window.start_secs <= 30.0));
            let alignment = stub_windowed_alignment(&windows, 60.0);
            async move { Ok(alignment) }
        })
        .await
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(alignment.windows.len(), 3);
        let words = alignment.words();
        assert_eq!(words.len(), 7);
        assert!(words.last().unwrap().1 >= 53_000, "{words:?}");
    }

    #[tokio::test]
    async fn an_alignment_with_no_aligned_window_is_an_error() {
        let result = align_transcript("only words", 40.0, None, |windows| {
            let mut alignment = stub_windowed_alignment(&windows, 40.0);
            for window in &mut alignment.windows {
                window.source = AlignmentSource::Failed;
                window.error = Some(izwi_core::Error::InvalidInput("too long".into()));
            }
            async move { Ok(alignment) }
        })
        .await;
        assert!(
            matches!(result, Err(izwi_core::Error::InvalidInput(message)) if message == "too long")
        );
    }

    #[test]
    fn short_or_mismatched_transcripts_align_as_one_window() {
        let one = transcript_alignment_windows("just this", 4.0, None);
        assert_eq!(
            one,
            vec![AlignmentWindow {
                start_secs: 0.0,
                end_secs: 4.0,
                text: "just this".into()
            }]
        );
        // Chunk texts that do not make up the transcript cannot be trusted.
        let diagnostics = chunked_diagnostics(&[(0.0, 20.0, "alpha"), (18.0, 40.0, "beta")]);
        let windows = transcript_alignment_windows("alpha gamma", 40.0, Some(&diagnostics));
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].text, "alpha gamma");
        // Cleanup differences in case and punctuation still match.
        let windows = transcript_alignment_windows("Alpha, beta!", 40.0, Some(&diagnostics));
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[1].start_secs, 18.0);
    }
}
