//! Windowed forced alignment.
//!
//! The aligner times at most one window of audio per forward (its
//! `max_audio_seconds_hint`). Longer audio is aligned as windows, each with
//! the text spoken in it, and every window's timings are rebased by the
//! window start. A window the aligner rejects or times degenerately keeps
//! interval timings and says so in its [`AlignmentSource`].

use serde::Serialize;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::asr::forced_alignment_words;
use crate::models::registry::ForcedAlignmentOutput;
use crate::runtime::diarization::{
    alignment_is_suspicious, fallback_word_timings_from_words, samples_to_ms,
};

/// The text spoken in `[start_secs, end_secs)` of the audio.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentWindow {
    pub start_secs: f64,
    pub end_secs: f64,
    pub text: String,
}

/// Where a window's word timings came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentSource {
    /// The aligner's own timings.
    Model,
    /// The aligner ran but its timings were degenerate; the words are spread
    /// evenly over the window.
    Heuristic,
    /// The aligner could not align the window; the words are spread evenly
    /// over the window.
    Failed,
}

/// One aligned window, in absolute milliseconds.
#[derive(Debug)]
pub struct AlignedWindow {
    pub start_ms: u32,
    pub end_ms: u32,
    pub source: AlignmentSource,
    /// `(word, start_ms, end_ms)` in absolute time.
    pub words: Vec<(String, u32, u32)>,
    /// Why the window is not [`AlignmentSource::Model`].
    pub detail: Option<String>,
    /// The aligner error of a [`AlignmentSource::Failed`] window.
    pub error: Option<Error>,
}

/// Every window of one alignment and the duration of the aligned audio.
#[derive(Debug)]
pub struct WindowedAlignment {
    pub windows: Vec<AlignedWindow>,
    pub duration_ms: u32,
}

impl WindowedAlignment {
    /// All words in window order. Windows may overlap (long-form ASR chunks
    /// do), so each start is clamped to the previous end and every time to
    /// the audio duration: the result is monotone and within the audio.
    pub fn words(&self) -> Vec<(String, u32, u32)> {
        let duration_ms = self.duration_ms.max(1);
        let mut previous_end = 0u32;
        let mut words = Vec::new();
        for (word, start, end) in self.windows.iter().flat_map(|window| &window.words) {
            let start = (*start).max(previous_end).min(duration_ms);
            let end = (*end).max(start).min(duration_ms);
            previous_end = end;
            words.push((word.clone(), start, end));
        }
        words
    }

    /// `Model` when every window was aligned by the model, `Failed` when no
    /// window was, otherwise `Heuristic` (some words carry interval timings).
    pub fn source(&self) -> AlignmentSource {
        let model = self
            .windows
            .iter()
            .filter(|window| window.source == AlignmentSource::Model)
            .count();
        if model == self.windows.len() {
            AlignmentSource::Model
        } else if model == 0
            && self
                .windows
                .iter()
                .all(|window| window.source == AlignmentSource::Failed)
        {
            AlignmentSource::Failed
        } else {
            AlignmentSource::Heuristic
        }
    }

    /// One message per window whose timings did not come from the model.
    pub fn warnings(&self) -> Vec<String> {
        self.windows
            .iter()
            .filter(|window| window.source != AlignmentSource::Model)
            .map(|window| {
                let kind = match window.source {
                    AlignmentSource::Failed => "alignment failed",
                    _ => "alignment was degenerate",
                };
                format!(
                    "Words in {:.2}-{:.2} s use evenly spaced timings: {kind}{}",
                    f64::from(window.start_ms) / 1000.0,
                    f64::from(window.end_ms) / 1000.0,
                    window
                        .detail
                        .as_deref()
                        .map(|detail| format!(" ({detail})"))
                        .unwrap_or_default()
                )
            })
            .collect()
    }

    /// The words, or the first window failure. For callers whose contract
    /// is one alignment that either succeeds or errors.
    pub(crate) fn into_words_or_error(mut self) -> Result<Vec<(String, u32, u32)>> {
        if let Some(error) = self
            .windows
            .iter_mut()
            .find_map(|window| window.error.take())
        {
            return Err(error);
        }
        Ok(self.words())
    }
}

/// An alignment window in samples of the decoded audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SampleWindow {
    pub(crate) start_sample: usize,
    pub(crate) end_sample: usize,
    pub(crate) text: String,
}

/// Convert caller windows to sample ranges of `total_samples` audio, clamped
/// to it. Times must be finite and ordered.
pub(crate) fn sample_windows(
    windows: Vec<AlignmentWindow>,
    sample_rate: u32,
    total_samples: usize,
) -> Result<Vec<SampleWindow>> {
    let rate = f64::from(sample_rate.max(1));
    windows
        .into_iter()
        .map(|window| {
            if !window.start_secs.is_finite()
                || !window.end_secs.is_finite()
                || window.start_secs < 0.0
                || window.end_secs < window.start_secs
            {
                return Err(Error::InvalidInput(format!(
                    "Invalid forced alignment window {}-{} s",
                    window.start_secs, window.end_secs
                )));
            }
            let start_sample = ((window.start_secs * rate).round() as usize).min(total_samples);
            let end_sample = ((window.end_secs * rate).round() as usize)
                .min(total_samples)
                .max(start_sample);
            Ok(SampleWindow {
                start_sample,
                end_sample,
                text: window.text,
            })
        })
        .collect()
}

/// Align one window with `align` (audio slice, text) and rebase the result
/// by the window start. Degenerate or failed alignments keep interval
/// timings over the window, flagged by their source.
pub(crate) fn align_window<F>(
    samples: &[f32],
    sample_rate: u32,
    window: &SampleWindow,
    align: F,
) -> AlignedWindow
where
    F: FnOnce(&[f32], &str) -> Result<ForcedAlignmentOutput>,
{
    let end_sample = window.end_sample.min(samples.len());
    let start_sample = window.start_sample.min(end_sample);
    let start_ms = samples_to_ms(start_sample, sample_rate);
    let end_ms = samples_to_ms(end_sample, sample_rate);
    let words = forced_alignment_words(&window.text);
    if words.is_empty() {
        return AlignedWindow {
            start_ms,
            end_ms,
            source: AlignmentSource::Model,
            words: Vec::new(),
            detail: None,
            error: None,
        };
    }

    let audio = &samples[start_sample..end_sample];
    let duration_secs = audio.len() as f32 / sample_rate.max(1) as f32;
    let result = if audio.is_empty() {
        Err(Error::InvalidInput(
            "the forced alignment window holds no audio".to_string(),
        ))
    } else {
        align(audio, &window.text)
    };
    let (source, relative, detail, error) = match result {
        Ok(output) if output.interval_fallback => (
            AlignmentSource::Heuristic,
            output.words,
            Some("the aligner's timings were degenerate".to_string()),
            None,
        ),
        Ok(output)
            if !output.words.is_empty()
                && !alignment_is_suspicious(&output.words, words.len(), duration_secs) =>
        {
            (AlignmentSource::Model, output.words, None, None)
        }
        Ok(_) => (
            AlignmentSource::Heuristic,
            fallback_word_timings_from_words(&words, duration_secs),
            Some("the aligner's timings were implausible".to_string()),
            None,
        ),
        Err(err) => (
            AlignmentSource::Failed,
            fallback_word_timings_from_words(&words, duration_secs),
            Some(err.to_string()),
            Some(err),
        ),
    };
    if source != AlignmentSource::Model {
        tracing::warn!(
            start_ms,
            end_ms,
            source = ?source,
            detail = detail.as_deref().unwrap_or_default(),
            "forced alignment window uses interval timings"
        );
    }
    let words = relative
        .into_iter()
        .map(|(word, start, end)| {
            (
                word,
                start_ms.saturating_add(start),
                start_ms.saturating_add(end),
            )
        })
        .collect();
    AlignedWindow {
        start_ms,
        end_ms,
        source,
        words,
        detail,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start_sample: usize, end_sample: usize, text: &str) -> SampleWindow {
        SampleWindow {
            start_sample,
            end_sample,
            text: text.to_string(),
        }
    }

    /// Stub aligner: the words of `text` at fixed relative times, 400 ms
    /// apart with 300 ms spans, regardless of the audio.
    fn fixed_aligner(audio: &[f32], text: &str) -> Result<ForcedAlignmentOutput> {
        assert!(!audio.is_empty());
        Ok(ForcedAlignmentOutput {
            words: forced_alignment_words(text)
                .into_iter()
                .enumerate()
                .map(|(idx, word)| (word, idx as u32 * 400, idx as u32 * 400 + 300))
                .collect(),
            interval_fallback: false,
        })
    }

    #[test]
    fn windows_rebase_by_window_start_and_stay_monotone() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 60 * rate as usize];
        let windows = [
            window(0, 30 * rate as usize, "one two three"),
            // Overlaps the first window by 3 s, as long-form chunks do.
            window(27 * rate as usize, 60 * rate as usize, "four five"),
        ];
        let aligned = WindowedAlignment {
            windows: windows
                .iter()
                .map(|window| align_window(&samples, rate, window, fixed_aligner))
                .collect(),
            duration_ms: 60_000,
        };

        assert_eq!(aligned.windows[0].start_ms, 0);
        assert_eq!(aligned.windows[1].start_ms, 27_000);
        assert_eq!(aligned.windows[1].end_ms, 60_000);
        assert_eq!(
            aligned.windows[1].words,
            vec![
                ("four".to_string(), 27_000, 27_300),
                ("five".to_string(), 27_400, 27_700),
            ]
        );
        assert!(aligned
            .windows
            .iter()
            .all(|window| window.source == AlignmentSource::Model));
        assert_eq!(aligned.source(), AlignmentSource::Model);
        assert!(aligned.warnings().is_empty());

        let words = aligned.words();
        assert_eq!(
            words
                .iter()
                .map(|(word, _, _)| word.as_str())
                .collect::<Vec<_>>(),
            ["one", "two", "three", "four", "five"]
        );
        assert!(words.windows(2).all(|pair| pair[1].1 >= pair[0].2));
        assert!(words
            .iter()
            .all(|(_, start, end)| start <= end && *end <= 60_000));
        assert!(words[3].1 >= 27_000, "second window rebased: {words:?}");
    }

    #[test]
    fn window_statuses_are_preserved_and_reported() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 20 * rate as usize];
        let ten = 10 * rate as usize;
        let failed = align_window(&samples, rate, &window(0, ten, "a b c"), |_, _| {
            Err(Error::InvalidInput("window too long".to_string()))
        });
        assert_eq!(failed.source, AlignmentSource::Failed);
        assert_eq!(
            failed.detail.as_deref(),
            Some("Invalid input: window too long")
        );
        assert_eq!(
            failed.words.len(),
            3,
            "failed windows keep interval timings"
        );
        assert!(failed.words.iter().all(|(_, _, end)| *end <= 10_000));

        let degenerate = align_window(&samples, rate, &window(ten, 2 * ten, "d e"), |_, text| {
            Ok(ForcedAlignmentOutput {
                interval_fallback: true,
                ..fixed_aligner(&[0.0], text)?
            })
        });
        assert_eq!(degenerate.source, AlignmentSource::Heuristic);
        assert_eq!(degenerate.words[0].1, 10_000);

        let failed_again = align_window(&samples, rate, &window(0, ten, "a b c"), |_, _| {
            Err(Error::InvalidInput("window too long".to_string()))
        });
        let alignment = WindowedAlignment {
            windows: vec![failed, degenerate],
            duration_ms: 20_000,
        };
        assert_eq!(alignment.source(), AlignmentSource::Heuristic);
        let warnings = alignment.warnings();
        assert_eq!(warnings.len(), 2);
        assert!(warnings[0].contains("0.00-10.00 s"), "{warnings:?}");
        assert!(warnings[0].contains("alignment failed"), "{warnings:?}");
        assert!(warnings[1].contains("degenerate"), "{warnings:?}");
        assert!(matches!(
            alignment.into_words_or_error(),
            Err(Error::InvalidInput(message)) if message == "window too long"
        ));

        let all_failed = WindowedAlignment {
            windows: vec![failed_again],
            duration_ms: 20_000,
        };
        assert_eq!(all_failed.source(), AlignmentSource::Failed);
    }

    #[test]
    fn collapsed_model_timings_are_flagged_heuristic() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 30 * rate as usize];
        let text = (0..20)
            .map(|idx| format!("w{idx}"))
            .collect::<Vec<_>>()
            .join(" ");
        let collapsed = align_window(
            &samples,
            rate,
            &window(0, samples.len(), &text),
            |_, text| {
                Ok(ForcedAlignmentOutput {
                    words: forced_alignment_words(text)
                        .into_iter()
                        .map(|word| (word, 29_900, 29_901))
                        .collect(),
                    interval_fallback: false,
                })
            },
        );
        assert_eq!(collapsed.source, AlignmentSource::Heuristic);
        assert!(collapsed.words[0].1 < 2_000, "{:?}", collapsed.words[0]);
    }

    #[test]
    fn sample_windows_clamp_to_the_audio_and_reject_invalid_times() {
        let windows = sample_windows(
            vec![
                AlignmentWindow {
                    start_secs: 0.0,
                    end_secs: 1.5,
                    text: "a".into(),
                },
                AlignmentWindow {
                    start_secs: 1.0,
                    end_secs: 9.0,
                    text: "b".into(),
                },
            ],
            16_000,
            32_000,
        )
        .unwrap();
        assert_eq!(windows[0], window(0, 24_000, "a"));
        assert_eq!(windows[1], window(16_000, 32_000, "b"));
        for (start_secs, end_secs) in [(2.0, 1.0), (f64::NAN, 1.0), (-1.0, 1.0)] {
            assert!(sample_windows(
                vec![AlignmentWindow {
                    start_secs,
                    end_secs,
                    text: "a".into()
                }],
                16_000,
                32_000
            )
            .is_err());
        }
    }

    /// `test.wav` twice (59.4 s) with its transcript twice, aligned as one
    /// window per copy. Before windowing, CPU/Metal encoded only 30 s and
    /// collapsed every second-copy word to about 29.9 s.
    #[test]
    #[ignore = "requires the local Qwen3-ForcedAligner-0.6B checkpoint (IZWI_ALIGNER_DEVICE=cpu|metal)"]
    fn forced_aligner_windows_cover_doubled_test_wav() {
        use crate::models::architectures::qwen3::asr::Qwen3AsrModel;
        use crate::{DeviceSelector, ModelVariant};

        let models = std::env::var("IZWI_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::data_local_dir()
                    .expect("data dir")
                    .join("izwi")
                    .join("models")
            });
        let model_dir = models.join("Qwen3-ForcedAligner-0.6B");
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let bytes = std::fs::read(data.join("test.wav")).expect("test.wav");
        let (once, rate) =
            crate::runtime::audio_io::decode_audio_bytes_canonical(&bytes).expect("decode");
        let text = std::fs::read_to_string(data.join("test.md")).expect("test.md");
        let text = text.trim();
        let samples = [once.as_slice(), once.as_slice()].concat();
        let duration_ms = samples_to_ms(samples.len(), rate);

        let device_name = std::env::var("IZWI_ALIGNER_DEVICE").unwrap_or_else(|_| "cpu".into());
        let device = DeviceSelector::detect_with_preference(Some(&device_name)).expect("device");
        let model = Qwen3AsrModel::load(&model_dir, ModelVariant::Qwen3ForcedAligner06B, device)
            .expect("load aligner");
        let windows = [
            window(0, once.len(), text),
            window(once.len(), samples.len(), text),
        ];
        let alignment = WindowedAlignment {
            windows: windows
                .iter()
                .map(|window| {
                    align_window(&samples, rate, window, |audio, text| {
                        model.force_align_detailed(audio, rate, text, None)
                    })
                })
                .collect(),
            duration_ms,
        };
        let words = alignment.words();
        let first_copy_words = forced_alignment_words(text).len();
        let second_start = words[first_copy_words].1;
        let last_end = words.last().expect("words").2;
        eprintln!(
            "{device_name}: {} words, duration {duration_ms} ms, second copy starts {second_start} ms, last word ends {last_end} ms, sources {:?}",
            words.len(),
            alignment.windows.iter().map(|window| window.source).collect::<Vec<_>>()
        );
        assert_eq!(words.len(), 2 * first_copy_words);
        assert_eq!(alignment.source(), AlignmentSource::Model);
        assert!(
            second_start >= 29_000,
            "second copy starts at {second_start} ms"
        );
        assert!(last_end >= 58_000, "last word ends at {last_end} ms");
        assert!(words.iter().all(|(_, _, end)| *end <= duration_ms));
        assert!(words.windows(2).all(|pair| pair[1].1 >= pair[0].2));
    }

    #[test]
    fn empty_window_text_skips_the_aligner() {
        let samples = vec![0.1f32; 16_000];
        let aligned = align_window(&samples, 16_000, &window(0, 16_000, " ,. "), |_, _| {
            panic!("no alignable words, no aligner call")
        });
        assert_eq!(aligned.source, AlignmentSource::Model);
        assert!(aligned.words.is_empty());
    }
}
