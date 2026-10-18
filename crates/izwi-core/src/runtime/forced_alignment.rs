//! Windowed forced alignment.
//!
//! The aligner times at most one window of audio per forward (its
//! `max_audio_seconds_hint`). Longer audio is aligned as windows, each with
//! the text spoken in it, and every window's timings are rebased by the
//! window start. The aligner's timings are the reference's, unchanged. A
//! window the aligner rejects, or whose timings fail the plausibility check
//! a caller asks for, keeps interval timings and says so in its
//! [`AlignmentSource`].
//!
//! A window longer than the aligner's limit with one text (a bare
//! `/v1/audio/align` request, or a transcript without chunk texts) is aligned
//! by [`LongTextPlanner`]: sequential sub-windows cut at low-energy points,
//! each given a generous share of the remaining words. The aligner pushes
//! words that are not in a sub-window's audio to its end, so only words that
//! end clearly inside it are kept and the next sub-window starts where the
//! last kept word ended.

use serde::Serialize;

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::asr::{alignment_units, qwen3_aligner_max_words};
use crate::models::registry::AlignmentUnit;
use crate::runtime::coordinator::{JobResourceObservation, StageWorkspaceLease};
use crate::runtime::diarization::{
    alignment_is_suspicious, fallback_word_timings_from_words, samples_to_ms,
};
use crate::runtime::lifecycle::asr_envelope::forced_alignment_envelope;
use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig};

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
    /// The aligner ran but its timings failed the plausibility check the
    /// caller asked for; the words are spread evenly over the window.
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
    /// `(word, start_ms, end_ms)` in absolute time, each word the source
    /// text of one aligner unit (punctuation included).
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

/// An alignment window in samples of the decoded audio, with its text and
/// the aligner's word units of that text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SampleWindow {
    pub(crate) start_sample: usize,
    pub(crate) end_sample: usize,
    pub(crate) text: String,
    pub(crate) units: Vec<AlignmentUnit>,
}

impl SampleWindow {
    /// A window whose text is split into word units for `language`.
    pub(crate) fn new(
        start_sample: usize,
        end_sample: usize,
        text: String,
        language: Option<&str>,
    ) -> Self {
        let units = alignment_units(&text, language);
        Self {
            start_sample,
            end_sample,
            text,
            units,
        }
    }

    /// The source text of each word unit, punctuation included.
    pub(crate) fn words(&self) -> Vec<String> {
        self.units
            .iter()
            .map(|unit| unit.surface(&self.text).to_string())
            .collect()
    }
}

/// The forward working set one aligner window reserves while it runs
/// (ADM-5): the forced-alignment envelope over the window's audio and its
/// words. Zero for a variant without one.
pub(crate) fn forced_alignment_window_workspace(
    variant: ModelVariant,
    backend: BackendKind,
    window: &SampleWindow,
    sample_rate: u32,
) -> Result<JobResourceObservation> {
    let samples = window.end_sample.saturating_sub(window.start_sample);
    let secs = samples as f32 / sample_rate.max(1) as f32;
    let words = window.units.len();
    Ok(forced_alignment_envelope(variant, secs, words, backend)?
        .map(|envelope| {
            JobResourceObservation::new(envelope.host_bytes, envelope.accelerator_bytes)
        })
        .unwrap_or_default())
}

/// Run one aligner forward under its stage lease: the working set counts
/// as allocated while `body` runs and as pending again before it is freed,
/// and the claim ends when the lease drops with the stage.
pub(crate) fn align_under_stage_lease<T>(
    lease: &StageWorkspaceLease,
    workspace: JobResourceObservation,
    body: impl FnOnce() -> T,
) -> Result<T> {
    lease.record_materialized_usage(workspace)?;
    let output = body();
    lease.prepare_materialized_release(JobResourceObservation::default())?;
    Ok(output)
}

/// Convert caller windows to sample ranges of `total_samples` audio, clamped
/// to it, with their text split into word units for `language`. Times must
/// be finite and ordered.
pub(crate) fn sample_windows(
    windows: Vec<AlignmentWindow>,
    sample_rate: u32,
    total_samples: usize,
    language: Option<&str>,
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
            Ok(SampleWindow::new(
                start_sample,
                end_sample,
                window.text,
                language,
            ))
        })
        .collect()
}

/// Align one window with `align` (audio slice, word units, giving one
/// `(start_ms, end_ms)` per unit) and rebase the result by the window start.
/// A failed alignment keeps interval timings over the window, flagged
/// [`AlignmentSource::Failed`]. With `check_plausibility`, collapsed timings
/// are replaced the same way and flagged [`AlignmentSource::Heuristic`];
/// otherwise the aligner's timings are kept as they are. [`LongTextPlanner`]
/// sub-windows skip the check, since their surplus words are expected to
/// pile up at the window end and the planner trims them itself.
pub(crate) fn align_window_checked<F>(
    samples: &[f32],
    sample_rate: u32,
    window: &SampleWindow,
    check_plausibility: bool,
    align: F,
) -> AlignedWindow
where
    F: FnOnce(&[f32], &[AlignmentUnit]) -> Result<Vec<(u32, u32)>>,
{
    let end_sample = window.end_sample.min(samples.len());
    let start_sample = window.start_sample.min(end_sample);
    let start_ms = samples_to_ms(start_sample, sample_rate);
    let end_ms = samples_to_ms(end_sample, sample_rate);
    let words = window.words();
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
        align(audio, &window.units).and_then(|timings| {
            if timings.len() != words.len() {
                return Err(Error::InferenceError(format!(
                    "the aligner timed {} of {} words",
                    timings.len(),
                    words.len()
                )));
            }
            Ok(words
                .iter()
                .cloned()
                .zip(timings)
                .map(|(word, (start, end))| (word, start, end))
                .collect::<Vec<_>>())
        })
    };
    let (source, relative, detail, error) = match result {
        Ok(timed)
            if check_plausibility
                && alignment_is_suspicious(&timed, words.len(), duration_secs) =>
        {
            (
                AlignmentSource::Heuristic,
                fallback_word_timings_from_words(&words, duration_secs),
                Some("the aligner's timings were implausible".to_string()),
                None,
            )
        }
        Ok(timed) => (AlignmentSource::Model, timed, None, None),
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

/// Words kept from a sub-window must end at least this long before its
/// end; later ones may be surplus words the aligner piled up there.
const LONG_TEXT_GUARD_MS: u32 = 1_000;
/// Words offered to a sub-window beyond its speech-proportional estimate.
/// Offering too many is safe (the surplus piles up and is trimmed); offering
/// too few would stretch the offered words over speech that is not theirs.
const LONG_TEXT_SURPLUS_RATIO: f64 = 1.3;
const LONG_TEXT_SURPLUS_WORDS: usize = 12;
/// Energy frame for the speech estimate.
const LONG_TEXT_FRAME_MS: usize = 20;
/// Frames at least this fraction of the loud (95th percentile) frame RMS
/// count as speech, above an absolute floor.
const LONG_TEXT_SPEECH_RELATIVE: f32 = 0.1;
const LONG_TEXT_SPEECH_FLOOR_RMS: f32 = 1e-3;

/// A sub-window to align next and what it was offered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedWindow {
    pub(crate) window: SampleWindow,
    /// Words the speech share of the sub-window predicts.
    estimate: usize,
    /// Whether the sub-window runs to the end of the long window.
    last: bool,
}

/// Plans the sub-windows of one window longer than the aligner's limit and
/// keeps the words each sub-window aligned inside it (see the module docs).
pub(crate) struct LongTextPlanner {
    /// The long window's text, its word units and their source texts.
    text: String,
    units: Vec<AlignmentUnit>,
    words: Vec<String>,
    next_word: usize,
    cursor: usize,
    end: usize,
    max_window: usize,
    sample_rate: u32,
    origin: usize,
    frame: usize,
    /// Prefix sums of speech frames from `origin`.
    speech_prefix: Vec<usize>,
}

impl LongTextPlanner {
    pub(crate) fn new(
        samples: &[f32],
        sample_rate: u32,
        window: &SampleWindow,
        max_window_samples: usize,
    ) -> Self {
        let end = window.end_sample.min(samples.len());
        let origin = window.start_sample.min(end);
        let frame = (sample_rate as usize * LONG_TEXT_FRAME_MS / 1000).max(1);
        let rms = samples[origin..end]
            .chunks(frame)
            .map(|chunk| {
                (chunk.iter().map(|sample| sample * sample).sum::<f32>() / chunk.len() as f32)
                    .sqrt()
            })
            .collect::<Vec<_>>();
        let mut sorted = rms.clone();
        sorted.sort_by(f32::total_cmp);
        let loud = sorted
            .get(sorted.len().saturating_mul(95) / 100)
            .or(sorted.last())
            .copied()
            .unwrap_or(0.0);
        let threshold = (loud * LONG_TEXT_SPEECH_RELATIVE).max(LONG_TEXT_SPEECH_FLOOR_RMS);
        let mut speech_prefix = Vec::with_capacity(rms.len() + 1);
        speech_prefix.push(0);
        for value in rms {
            let last = *speech_prefix.last().unwrap_or(&0);
            speech_prefix.push(last + usize::from(value >= threshold));
        }
        Self {
            text: window.text.clone(),
            units: window.units.clone(),
            words: window.words(),
            next_word: 0,
            cursor: origin,
            end,
            max_window: max_window_samples.max(1),
            sample_rate,
            origin,
            frame,
            speech_prefix,
        }
    }

    fn speech_frames(&self, from: usize, to: usize) -> usize {
        let index = |sample: usize| {
            ((sample.saturating_sub(self.origin)) / self.frame).min(self.speech_prefix.len() - 1)
        };
        self.speech_prefix[index(to)] - self.speech_prefix[index(from)]
    }

    /// The end of the next sub-window: a low-energy point at most the
    /// aligner's limit after the cursor.
    fn cut_after_cursor(&self, samples: &[f32]) -> usize {
        let max_secs = self.max_window as f32 / self.sample_rate.max(1) as f32;
        let search_secs = (max_secs / 6.0).min(4.0);
        let config = AsrLongFormConfig {
            target_chunk_secs: max_secs - search_secs,
            hard_max_chunk_secs: max_secs,
            overlap_secs: 0.0,
            silence_search_secs: search_secs,
            min_chunk_secs: max_secs / 2.0,
            ..AsrLongFormConfig::default()
        };
        let horizon = self.end.min(self.cursor + 2 * self.max_window);
        let cut = plan_audio_chunks(
            &samples[self.cursor..horizon],
            self.sample_rate,
            &config,
            Some(max_secs),
        )
        .first()
        .map(|chunk| self.cursor + chunk.end_sample)
        .unwrap_or(self.cursor + self.max_window);
        cut.clamp(self.cursor + 1, self.cursor + self.max_window)
    }

    /// The next sub-window, or `None` once every word or all audio is used.
    pub(crate) fn next_window(&mut self, samples: &[f32]) -> Option<PlannedWindow> {
        loop {
            let remaining_words = self.words.len() - self.next_word;
            if remaining_words == 0 || self.cursor >= self.end {
                return None;
            }
            let last = self.end - self.cursor <= self.max_window;
            let cut = if last {
                self.end
            } else {
                self.cut_after_cursor(samples)
            };
            let speech_here = self.speech_frames(self.cursor, cut);
            if !last && speech_here == 0 {
                // No speech: no words to place here.
                self.cursor = cut;
                continue;
            }
            let estimate = if last {
                remaining_words
            } else {
                let speech_rest = self.speech_frames(self.cursor, self.end);
                let (part, whole) = if speech_rest > 0 {
                    (speech_here, speech_rest)
                } else {
                    (cut - self.cursor, self.end - self.cursor)
                };
                ((remaining_words * part).div_ceil(whole.max(1))).clamp(1, remaining_words)
            };
            let window_secs = (cut - self.cursor) as f64 / f64::from(self.sample_rate.max(1));
            let offered = if last {
                remaining_words
            } else {
                ((estimate as f64 * LONG_TEXT_SURPLUS_RATIO).ceil() as usize
                    + LONG_TEXT_SURPLUS_WORDS)
                    .min(qwen3_aligner_max_words(window_secs))
                    .clamp(estimate, remaining_words)
            };
            // The offered units keep their own text: re-splitting a joined
            // text could split differently (Korean, Japanese).
            let units = &self.units[self.next_word..self.next_word + offered];
            let from = units[0].source.start;
            let to = units[units.len() - 1].source.end;
            return Some(PlannedWindow {
                window: SampleWindow {
                    start_sample: self.cursor,
                    end_sample: cut,
                    text: self.text[from..to].to_string(),
                    units: units
                        .iter()
                        .map(|unit| AlignmentUnit {
                            text: unit.text.clone(),
                            source: unit.source.start - from..unit.source.end - from,
                        })
                        .collect(),
                },
                estimate,
                last,
            });
        }
    }

    /// Keep the words `aligned` placed inside `planned` and advance past
    /// them. Words that end within the guard of the sub-window end, and every
    /// word after them, go to the next sub-window, which starts at the last
    /// kept word or, when the model placed nothing more in the sub-window,
    /// two guards before its end. A sub-window the model could not time
    /// takes its speech-estimated share of words with interval timings.
    pub(crate) fn absorb(
        &mut self,
        planned: &PlannedWindow,
        aligned: AlignedWindow,
    ) -> AlignedWindow {
        if planned.last {
            self.next_word = self.words.len();
            self.cursor = self.end;
            return aligned;
        }
        let cut = planned.window.end_sample;
        let cut_ms = samples_to_ms(cut, self.sample_rate);
        if aligned.source == AlignmentSource::Model {
            let mut words = aligned.words;
            let kept = words
                .iter()
                .take_while(|(_, _, end)| end.saturating_add(LONG_TEXT_GUARD_MS) <= cut_ms)
                .count();
            words.truncate(kept);
            let guard =
                (u64::from(LONG_TEXT_GUARD_MS) * u64::from(self.sample_rate) / 1000) as usize;
            let quiet_from = cut - (2 * guard).min((cut - self.cursor) / 2);
            let last_end = words.last().map_or(0, |word| {
                (u64::from(word.2) * u64::from(self.sample_rate) / 1000) as usize
            });
            self.next_word += kept;
            self.cursor = last_end.max(quiet_from).clamp(self.cursor + 1, cut);
            return AlignedWindow {
                start_ms: aligned.start_ms,
                end_ms: samples_to_ms(self.cursor, self.sample_rate),
                source: AlignmentSource::Model,
                words,
                detail: None,
                error: None,
            };
        }

        let count = planned.estimate.min(self.words.len() - self.next_word);
        let start_ms = samples_to_ms(planned.window.start_sample, self.sample_rate);
        let duration_secs = (planned.window.end_sample - planned.window.start_sample) as f32
            / self.sample_rate.max(1) as f32;
        let words = fallback_word_timings_from_words(
            &self.words[self.next_word..self.next_word + count],
            duration_secs,
        )
        .into_iter()
        .map(|(word, start, end)| {
            (
                word,
                start_ms.saturating_add(start),
                start_ms.saturating_add(end),
            )
        })
        .collect();
        self.next_word += count;
        self.cursor = planned.window.end_sample;
        AlignedWindow {
            start_ms,
            end_ms: cut_ms,
            source: aligned.source,
            words,
            detail: aligned.detail,
            error: aligned.error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn align_window<F>(
        samples: &[f32],
        sample_rate: u32,
        window: &SampleWindow,
        align: F,
    ) -> AlignedWindow
    where
        F: FnOnce(&[f32], &[AlignmentUnit]) -> Result<Vec<(u32, u32)>>,
    {
        align_window_checked(samples, sample_rate, window, true, align)
    }

    fn window(start_sample: usize, end_sample: usize, text: &str) -> SampleWindow {
        SampleWindow::new(start_sample, end_sample, text.to_string(), None)
    }

    /// Stub aligner: the units at fixed relative times, 400 ms apart with
    /// 300 ms spans, regardless of the audio.
    fn fixed_aligner(audio: &[f32], units: &[AlignmentUnit]) -> Result<Vec<(u32, u32)>> {
        assert!(!audio.is_empty());
        Ok((0..units.len() as u32)
            .map(|idx| (idx * 400, idx * 400 + 300))
            .collect())
    }

    /// ADM-5: an aligner window stage reserves its forward working set,
    /// which grows with the window's seconds and words, holds it as
    /// allocated while the forward runs, and releases it with the stage.
    #[tokio::test]
    async fn aligner_window_stage_reserves_its_forward_working_set() {
        use crate::engine::ResourceAmount;
        use crate::runtime::coordinator::{InferenceCoordinator, JobSpec};
        use crate::runtime::types::RuntimeRequestContext;
        use crate::runtime::CoordinatorLane;
        use std::sync::Arc;

        let rate = 16_000u32;
        let variant = ModelVariant::Qwen3ForcedAligner06B;
        let workspace = |secs: usize, text: &str| {
            let window = window(0, secs * rate as usize, text);
            let workspace =
                forced_alignment_window_workspace(variant, BackendKind::Metal, &window, rate)
                    .unwrap();
            workspace.host_bytes + workspace.accelerator_bytes
        };
        let words = |count: usize| vec!["word"; count].join(" ");
        let short = workspace(5, &words(12));
        assert!(short > 0);
        assert!(workspace(30, &words(75)) > short);
        assert!(workspace(180, &words(450)) > workspace(30, &words(75)));
        assert!(workspace(30, &words(300)) > workspace(30, &words(75)));
        assert_eq!(
            forced_alignment_window_workspace(
                ModelVariant::Qwen3Asr06BGguf,
                BackendKind::Metal,
                &window(0, 30 * rate as usize, "a b"),
                rate
            )
            .unwrap(),
            JobResourceObservation::default()
        );

        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Metal, 1, 4));
        let context = RuntimeRequestContext::default();
        let job = coordinator
            .admit(JobSpec {
                request_id: "aligner-stage-lease".to_string(),
                lane: CoordinatorLane::Atomic,
                priority: context.priority,
                workload_class: context.workload_class,
                deadline: None,
                resources: crate::runtime::service::transient_resources(BackendKind::Metal, 0),
            })
            .await
            .unwrap();
        let observation = forced_alignment_window_workspace(
            variant,
            BackendKind::Metal,
            &window(0, 180 * rate as usize, &words(450)),
            rate,
        )
        .unwrap();
        let bytes = observation.host_bytes + observation.accelerator_bytes;
        let authority = coordinator.resource_authority();
        let before = authority.snapshot();
        let plus = |amount: ResourceAmount| match amount {
            ResourceAmount::Known(value) => ResourceAmount::Known(value + bytes),
            ResourceAmount::Unknown => panic!("unknown authority amount"),
        };
        let lease = coordinator
            .reserve_stage_workspace(&job, "asr.forced_alignment", observation)
            .unwrap();
        assert_eq!(
            authority.snapshot().pending.unified_bytes,
            plus(before.pending.unified_bytes)
        );
        align_under_stage_lease(&lease, observation, || {
            let running = authority.snapshot();
            assert_eq!(
                running.materialized.unified_bytes,
                plus(before.materialized.unified_bytes)
            );
            assert_eq!(running.pending.unified_bytes, before.pending.unified_bytes);
        })
        .unwrap();
        drop(lease);
        let after = authority.snapshot();
        assert_eq!(after.reserved, before.reserved);
        assert_eq!(after.pending, before.pending);
        assert_eq!(after.reservations, before.reservations);
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

        let degenerate = align_window(
            &samples,
            rate,
            &window(ten, 2 * ten, "d e f g h i j k"),
            |_, units| Ok(vec![(9_900, 9_900); units.len()]),
        );
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
            |_, units| Ok(vec![(29_900, 29_901); units.len()]),
        );
        assert_eq!(collapsed.source, AlignmentSource::Heuristic);
        assert!(collapsed.words[0].1 < 2_000, "{:?}", collapsed.words[0]);
    }

    /// QFA-5: without the plausibility check the aligner's timings are kept
    /// exactly, zero-length words included; nothing is substituted silently.
    #[test]
    fn unchecked_windows_keep_the_aligner_timings() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 30 * rate as usize];
        let text = (0..20)
            .map(|idx| format!("w{idx}"))
            .collect::<Vec<_>>()
            .join(" ");
        let unchecked = align_window_checked(
            &samples,
            rate,
            &window(rate as usize, samples.len(), &text),
            false,
            |_, units| Ok(vec![(28_900, 28_900); units.len()]),
        );
        assert_eq!(unchecked.source, AlignmentSource::Model);
        assert!(unchecked
            .words
            .iter()
            .all(|(_, start, end)| (*start, *end) == (29_900, 29_900)));
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
            None,
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
                32_000,
                None
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
        #[cfg(feature = "metal")]
        let _lease = crate::backends::metal_test_device_or_skip(
            "forced_aligner_windows_cover_doubled_test_wav",
        );
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
        let doubled = format!("{text} {text}");
        let first_copy_words = alignment_units(text, None).len();
        let align = |window: &SampleWindow, check: bool| {
            align_window_checked(&samples, rate, window, check, |audio, units| {
                model.force_align_units(audio, rate, units)
            })
        };
        let report = |label: &str, alignment: &WindowedAlignment| {
            let words = alignment.words();
            let second_start = words[first_copy_words].1;
            let last_end = words.last().expect("words").2;
            eprintln!(
                "{device_name} {label}: {} words, duration {duration_ms} ms, second copy starts {second_start} ms, last word ends {last_end} ms, windows {:?}",
                words.len(),
                alignment
                    .windows
                    .iter()
                    .map(|window| (window.start_ms, window.end_ms, window.source, window.words.len()))
                    .collect::<Vec<_>>()
            );
            assert_eq!(words.len(), 2 * first_copy_words, "{label}");
            assert_eq!(alignment.source(), AlignmentSource::Model, "{label}");
            assert!(
                second_start >= 29_000,
                "{label}: second copy starts at {second_start} ms"
            );
            assert!(
                last_end >= 58_000,
                "{label}: last word ends at {last_end} ms"
            );
            assert!(
                words.iter().all(|(_, _, end)| *end <= duration_ms),
                "{label}"
            );
            assert!(
                words.windows(2).all(|pair| pair[1].1 >= pair[0].2),
                "{label}"
            );
            words
        };

        // Windows from known (range, text) pairs, as long-form ASR chunks give.
        let by_chunk = WindowedAlignment {
            windows: [
                window(0, once.len(), text),
                window(once.len(), samples.len(), text),
            ]
            .iter()
            .map(|window| align(window, true))
            .collect(),
            duration_ms,
        };
        let chunk_words = report("chunk windows", &by_chunk);

        // One text for the whole audio, as a bare /v1/audio/align request.
        let max_window = (f64::from(model.max_audio_seconds_hint().expect("aligner window"))
            * f64::from(rate))
        .floor() as usize;
        let whole = window(0, samples.len(), &doubled);
        let mut planner = LongTextPlanner::new(&samples, rate, &whole, max_window);
        let mut windows = Vec::new();
        while let Some(planned) = planner.next_window(&samples) {
            assert!(planned.window.end_sample - planned.window.start_sample <= max_window);
            let aligned = align(&planned.window, false);
            windows.push(planner.absorb(&planned, aligned));
        }
        let planned = WindowedAlignment {
            windows,
            duration_ms,
        };
        let planned_words = report("long-text planner", &planned);
        let drift = chunk_words
            .iter()
            .zip(&planned_words)
            .flat_map(|(a, b)| [a.1.abs_diff(b.1), a.2.abs_diff(b.2)])
            .collect::<Vec<_>>();
        let over_80 = drift.iter().filter(|value| **value > 80).count();
        let max_drift = drift.iter().max().copied().unwrap_or(0);
        let mean_drift =
            drift.iter().map(|value| u64::from(*value)).sum::<u64>() / drift.len().max(1) as u64;
        eprintln!(
            "{device_name} planner vs chunk windows: {over_80}/{} boundaries > 80 ms, max {max_drift} ms, mean {mean_drift} ms",
            drift.len()
        );
    }

    /// D1: one 180 s window against the old 30 s windows, on the real
    /// checkpoint (`IZWI_ALIGNER_DEVICE=cpu|metal`): `test.wav` twice
    /// (59.4 s), and optionally a long cut (`IZWI_ALIGNER_LONG_WAV` with its
    /// transcript in `IZWI_ALIGNER_LONG_TEXT`). `IZWI_ALIGNER_MODE` picks
    /// `single`, `windowed`, `both` (default) or `load` (for footprint
    /// baselines); `IZWI_ALIGNER_DUMP` names a directory for the words.
    #[test]
    #[ignore = "requires the local Qwen3-ForcedAligner-0.6B checkpoint (IZWI_ALIGNER_DEVICE=cpu|metal)"]
    fn forced_aligner_single_window_matches_30_s_windows() {
        #[cfg(feature = "metal")]
        let _lease = crate::backends::metal_test_device_or_skip(
            "forced_aligner_single_window_matches_30_s_windows",
        );
        use crate::models::architectures::qwen3::asr::Qwen3AsrModel;
        use crate::{DeviceSelector, ModelVariant};
        use std::io::Write;

        let models = std::env::var("IZWI_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::data_local_dir()
                    .expect("data dir")
                    .join("izwi")
                    .join("models")
            });
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let decode = |path: &std::path::Path| {
            let bytes = std::fs::read(path).expect("audio");
            crate::runtime::audio_io::decode_audio_bytes_canonical(&bytes).expect("decode")
        };
        let (once, rate) = decode(&data.join("test.wav"));
        let text = std::fs::read_to_string(data.join("test.md")).expect("test.md");
        let mut cases = vec![(
            "test.wav x2".to_string(),
            [once.as_slice(), once.as_slice()].concat(),
            format!("{} {}", text.trim(), text.trim()),
        )];
        if let (Ok(wav), Ok(transcript)) = (
            std::env::var("IZWI_ALIGNER_LONG_WAV"),
            std::env::var("IZWI_ALIGNER_LONG_TEXT"),
        ) {
            let (samples, long_rate) = decode(std::path::Path::new(&wav));
            assert_eq!(long_rate, rate);
            let transcript = std::fs::read_to_string(transcript).expect("long transcript");
            cases.push((wav, samples, transcript.trim().to_string()));
        }

        let device_name = std::env::var("IZWI_ALIGNER_DEVICE").unwrap_or_else(|_| "cpu".into());
        let mode = std::env::var("IZWI_ALIGNER_MODE").unwrap_or_else(|_| "both".into());
        let device = DeviceSelector::detect_with_preference(Some(&device_name)).expect("device");
        let model = Qwen3AsrModel::load(
            &models.join("Qwen3-ForcedAligner-0.6B"),
            ModelVariant::Qwen3ForcedAligner06B,
            device,
        )
        .expect("load aligner");
        let limit = model.max_audio_seconds_hint().expect("aligner window");
        assert_eq!(limit, 180.0, "window {limit} s");
        if mode == "load" {
            return;
        }
        // The pre-D1 CPU/Metal window: 3,000 mel frames.
        let window_30 = 480_000usize;
        for (label, samples, text) in cases {
            let duration_ms = samples_to_ms(samples.len(), rate);
            let whole = window(0, samples.len(), &text);
            let align = |window: &SampleWindow, check: bool| {
                align_window_checked(&samples, rate, window, check, |audio, units| {
                    model.force_align_units(audio, rate, units)
                })
            };
            let timed = |label: &str, run: &dyn Fn() -> WindowedAlignment| {
                let started = std::time::Instant::now();
                let alignment = run();
                let words = alignment.words();
                let _ = writeln!(
                    std::io::stderr(),
                    "{device_name} {label}: {} words in {:.1} s, sources {:?}",
                    words.len(),
                    started.elapsed().as_secs_f64(),
                    alignment
                        .windows
                        .iter()
                        .map(|window| (window.start_ms, window.end_ms, window.source))
                        .collect::<Vec<_>>()
                );
                assert_eq!(alignment.source(), AlignmentSource::Model, "{label}");
                assert!(words.windows(2).all(|pair| pair[1].1 >= pair[0].2));
                words
            };
            let single = (mode != "windowed").then(|| {
                timed(&format!("{label} single"), &|| WindowedAlignment {
                    windows: vec![align(&whole, true)],
                    duration_ms,
                })
            });
            let windowed = (mode != "single").then(|| {
                timed(&format!("{label} 30 s windows"), &|| {
                    let mut planner = LongTextPlanner::new(&samples, rate, &whole, window_30);
                    let mut windows = Vec::new();
                    while let Some(planned) = planner.next_window(&samples) {
                        let aligned = align(&planned.window, false);
                        windows.push(planner.absorb(&planned, aligned));
                    }
                    WindowedAlignment {
                        windows,
                        duration_ms,
                    }
                })
            });
            if let Ok(dir) = std::env::var("IZWI_ALIGNER_DUMP") {
                let name = label.rsplit('/').next().unwrap_or(&label).replace(' ', "_");
                for (kind, words) in [("single", &single), ("windowed", &windowed)] {
                    if let Some(words) = words {
                        let path = format!("{dir}/{device_name}-{name}-{kind}.json");
                        std::fs::write(path, serde_json::to_string(words).unwrap()).unwrap();
                    }
                }
            }
            if let (Some(single), Some(windowed)) = (single, windowed) {
                assert_eq!(single.len(), windowed.len(), "{label}");
                let drift = single
                    .iter()
                    .zip(&windowed)
                    .flat_map(|(a, b)| [a.1.abs_diff(b.1), a.2.abs_diff(b.2)])
                    .collect::<Vec<_>>();
                let over_80 = drift.iter().filter(|value| **value > 80).count();
                let mean = drift.iter().map(|value| u64::from(*value)).sum::<u64>() as f64
                    / drift.len().max(1) as f64;
                let _ = writeln!(
                    std::io::stderr(),
                    "{device_name} {label} single vs 30 s windows: mean {mean:.1} ms, max {} ms, \
                     {over_80}/{} boundaries > 80 ms",
                    drift.iter().max().copied().unwrap_or(0),
                    drift.len()
                );
            }
        }
    }

    /// Drive the planner the way the runtime does, one aligner call per
    /// planned sub-window.
    fn run_long_text<F>(
        samples: &[f32],
        rate: u32,
        window: &SampleWindow,
        max_window: usize,
        aligner: F,
    ) -> Vec<AlignedWindow>
    where
        F: Fn(&SampleWindow, &[AlignmentUnit]) -> Result<Vec<(u32, u32)>>,
    {
        let mut planner = LongTextPlanner::new(samples, rate, window, max_window);
        let mut out = Vec::new();
        while let Some(planned) = planner.next_window(samples) {
            assert!(planned.window.end_sample - planned.window.start_sample <= max_window);
            let aligned =
                align_window_checked(samples, rate, &planned.window, false, |_, units| {
                    aligner(&planned.window, units)
                });
            out.push(planner.absorb(&planned, aligned));
            assert!(out.len() < 100, "planner must make progress");
        }
        out
    }

    /// Word `w{k}` is spoken at 500k..500k+300 ms. Like the real aligner, a
    /// word whose speech is not in the window piles up at the window end.
    fn timeline_aligner(
        rate: u32,
    ) -> impl Fn(&SampleWindow, &[AlignmentUnit]) -> Result<Vec<(u32, u32)>> {
        move |window, units| {
            let start_ms = samples_to_ms(window.start_sample, rate);
            let length_ms = samples_to_ms(window.end_sample, rate) - start_ms;
            Ok(units
                .iter()
                .map(|unit| {
                    let k: u32 = unit.text[1..].parse().unwrap();
                    let (true_start, true_end) = (k * 500, k * 500 + 300);
                    if true_end <= start_ms + length_ms {
                        let relative = true_start.saturating_sub(start_ms);
                        (relative, true_end - start_ms)
                    } else {
                        (length_ms - 80, length_ms - 80)
                    }
                })
                .collect())
        }
    }

    #[test]
    fn long_text_is_aligned_in_sub_windows_within_the_limit() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 70 * rate as usize];
        let text = (0..140)
            .map(|k| format!("w{k}"))
            .collect::<Vec<_>>()
            .join(" ");
        let max_window = 30 * rate as usize - 160;
        let aligned = run_long_text(
            &samples,
            rate,
            &window(0, samples.len(), &text),
            max_window,
            timeline_aligner(rate),
        );

        assert!(aligned.len() >= 3, "{} windows", aligned.len());
        assert!(aligned
            .iter()
            .all(|window| window.source == AlignmentSource::Model));
        let alignment = WindowedAlignment {
            windows: aligned,
            duration_ms: 70_000,
        };
        let words = alignment.words();
        assert_eq!(words.len(), 140);
        for (k, (word, start, end)) in words.iter().enumerate() {
            let k = k as u32;
            assert_eq!(word, &format!("w{k}"));
            assert_eq!((*start, *end), (k * 500, k * 500 + 300), "{word}");
        }
    }

    #[test]
    fn long_text_windows_that_fail_keep_flagged_interval_timings() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 70 * rate as usize];
        let text = (0..140)
            .map(|k| format!("w{k}"))
            .collect::<Vec<_>>()
            .join(" ");
        let aligned = run_long_text(
            &samples,
            rate,
            &window(0, samples.len(), &text),
            30 * rate as usize,
            |_, _| Err(Error::InferenceError("aligner unavailable".into())),
        );
        assert!(aligned.len() >= 3);
        assert!(aligned
            .iter()
            .all(|window| window.source == AlignmentSource::Failed));
        let alignment = WindowedAlignment {
            windows: aligned,
            duration_ms: 70_000,
        };
        let words = alignment.words();
        assert_eq!(words.len(), 140);
        assert!(words.windows(2).all(|pair| pair[1].1 >= pair[0].2));
        assert!(words.last().unwrap().2 > 60_000);
        assert_eq!(alignment.source(), AlignmentSource::Failed);
    }

    #[test]
    fn long_text_skips_silent_stretches() {
        let rate = 16_000u32;
        // 10 s speech, 40 s silence, 20 s speech; words only where speech is.
        let mut samples = vec![0.1f32; 70 * rate as usize];
        samples[10 * rate as usize..50 * rate as usize].fill(0.0);
        let words = (0..20)
            .chain(100..140)
            .map(|k| format!("w{k}"))
            .collect::<Vec<_>>();
        let aligned = run_long_text(
            &samples,
            rate,
            &window(0, samples.len(), &words.join(" ")),
            30 * rate as usize,
            timeline_aligner(rate),
        );
        let alignment = WindowedAlignment {
            windows: aligned,
            duration_ms: 70_000,
        };
        let aligned_words = alignment.words();
        assert_eq!(aligned_words.len(), 60);
        assert_eq!(aligned_words[20].0, "w100");
        assert_eq!(aligned_words[20].1, 50_000, "{:?}", alignment.windows);
    }

    /// QFA-3: the aligner gets the language's units and the words keep the
    /// transcript's punctuation.
    #[test]
    fn window_words_are_language_units_with_their_punctuation() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 10 * rate as usize];
        let text = "We’re here, “AI—autonomous”!".to_string();
        let window = SampleWindow::new(0, samples.len(), text, Some("English"));
        let aligned = align_window(&samples, rate, &window, |audio, units| {
            let texts = units
                .iter()
                .map(|unit| unit.text.as_str())
                .collect::<Vec<_>>();
            assert_eq!(texts, ["Were", "here", "AIautonomous"]);
            fixed_aligner(audio, units)
        });
        assert_eq!(
            aligned.words,
            [
                ("We’re".to_string(), 0, 300),
                ("here,".to_string(), 400, 700),
                ("“AI—autonomous”!".to_string(), 800, 1_100),
            ]
        );

        let korean =
            SampleWindow::new(0, samples.len(), "학교에서는 공부합니다".into(), Some("ko"));
        assert_eq!(korean.words(), ["학교", "에서는", "공부", "합니다"]);
    }

    /// A sub-window takes its units from the long window instead of
    /// re-splitting its text: re-splitting the Korean `에서는` alone would
    /// cut it after the dictionary word `에서`.
    #[test]
    fn long_text_sub_windows_keep_the_long_window_units() {
        let rate = 16_000u32;
        let samples = vec![0.1f32; 70 * rate as usize];
        let text = vec!["학교에서는 공부합니다."; 70].join(" ");
        let long = SampleWindow::new(0, samples.len(), text, Some("Korean"));
        assert_eq!(long.units.len(), 280);
        assert_eq!(
            SampleWindow::new(0, 1, "에서는".into(), Some("Korean")).words(),
            ["에서", "는"]
        );
        let mut planner = LongTextPlanner::new(&samples, rate, &long, 30 * rate as usize);
        let planned = planner.next_window(&samples).expect("a sub-window");
        let offered = planned.window.units.len();
        assert!(offered < long.units.len());
        let first = long.units[0].source.start;
        let expected = long.units[..offered]
            .iter()
            .map(|unit| AlignmentUnit {
                text: unit.text.clone(),
                source: unit.source.start - first..unit.source.end - first,
            })
            .collect::<Vec<_>>();
        assert_eq!(planned.window.units, expected);
        assert_eq!(planned.window.words(), long.words()[..offered]);
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
