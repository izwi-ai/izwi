//! NeMo's `MultiSpeakerSOTWordTimestampAligner` on the host: from a
//! finished t-SOT transcript, the CTC head's log-probabilities and the
//! diarizer's speaker probabilities to word timestamps with speakers and
//! diarization segments, as vLLM's `ctc_timestamps` object reports them.
//!
//! The work is split as upstream splits it:
//! * [`prepare`] (upstream `_prepare`) needs only the CTC tokenizer: it
//!   parses the transcript into speaker streams, re-tokenises each stream,
//!   builds the blank-expanded CTC targets and lists the log-prob columns
//!   the alignment reads (blank first, then the sorted labels), so the
//!   device gathers only those columns.
//! * [`align`] (upstream `_align_banded_ctc` + vLLM's `_public_result`)
//!   resamples the 80 ms speaker sigmoids onto the 40 ms CTC grid, maps
//!   speaker tags to diarizer columns, runs the banded search per stream
//!   over the frames where its speaker is active, turns paths into word
//!   times, refines them against the 10 ms diarization segments and rounds
//!   every time to milliseconds.
//!
//! Defaults are NeMo's serving ones: speaker prior weight 1.0 (vLLM's
//! default; the checkpoint sets none), activity threshold 0.5, word cap
//! 1 s, refinement cap 12 encoder frames × the head's upsampling 2 = 24 CTC
//! frames, epsilon 1e-6. `config.json`'s `ctc_timestamps.alignment_config`
//! is not read by NeMo and is ignored here too.
//!
//! Numerics follow upstream: the DP in F32 with upstream's tie order, the
//! prior `w·ln(max(p, ε))` in F32, word and segment times in F64, the
//! speaker mapping in F64. Two platform-dependent roundings follow
//! upstream's arm64 builds (see `search` and `resample_speaker_probs`).
//!
//! Deviations from upstream:
//! * Thai keeps whitespace words: upstream segments Thai with pythainlp's
//!   `newmm`, which izwi does not ship.
//! * The banded search keeps checkpoints instead of a full backpointer
//!   matrix (identical paths, bounded memory).
//! * The speaker-mapping similarity sums sequentially where NumPy calls
//!   BLAS and pairwise sums: equal up to the last bit, which only matters
//!   for assignments that tie to within it.
//! * `prepare` does not strip the decoder's special tokens: callers pass a
//!   transcript cleaned with [`clean_sot_transcript`] (as vLLM does before
//!   preparing), or one that only carries speaker, event and style tags.
//!   An empty transcript prepares no streams and aligns to no words without
//!   an error, as vLLM reports it.
//!
//! [`clean_sot_transcript`]: super::clean_sot_transcript

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use super::assignment::speaker_mapping;
use super::search::{argmax, banded_ctc_search, SearchOutcome};
use super::text::{character_token_ids, parse_sot_words, timestamp_units, SotWord};
use super::tokenizer::CtcTokenizer;

/// Why a request has no word timestamps (vLLM's `error` strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CtcTimestampError {
    /// No capture is recorded for the request.
    NoCapture,
    /// The request has more than one audio item.
    MultipleAudio,
    /// The request's capture was not kept.
    CaptureUnavailable,
    /// The aligner cannot align the transcript.
    AlignmentFailed,
    /// Timestamps are not enabled.
    NotEnabled,
}

/// Aligner hyperparameters; `Default` is NeMo's serving configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcAlignmentConfig {
    /// Speaker activity at which a greedy token counts toward a diarizer
    /// column in the speaker mapping (0.5).
    pub speaker_activity_threshold: f32,
    /// Weight of the log speaker-activity prior on non-blank states (1.0,
    /// vLLM's default).
    pub speaker_logprob_weight: f32,
    /// Longest word, in seconds (1.0).
    pub maximum_token_len: f64,
    /// Refinement cap in CTC frames: 12 encoder frames × upsampling 2 (24).
    pub max_word_frames: usize,
    /// Floor of the speaker activity before its logarithm (1e-6).
    pub epsilon: f32,
    /// Duration of one diarization label frame, in seconds (0.01).
    pub diarization_frame_seconds: f64,
}

impl Default for CtcAlignmentConfig {
    fn default() -> Self {
        Self {
            speaker_activity_threshold: 0.5,
            speaker_logprob_weight: 1.0,
            maximum_token_len: 1.0,
            max_word_frames: 24,
            epsilon: 1.0e-6,
            diarization_frame_seconds: 0.01,
        }
    }
}

impl CtcAlignmentConfig {
    /// Upstream's constructor checks; an invalid configuration aligns
    /// nothing.
    fn is_valid(&self) -> bool {
        self.speaker_logprob_weight >= 0.0
            && (0.0..=1.0).contains(&self.speaker_activity_threshold)
            && self.maximum_token_len > 0.0
            && self.max_word_frames >= 1
            && self.diarization_frame_seconds > 0.0
    }
}

/// One speaker stream of a prepared transcript.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedStream {
    /// `str(tag)`; `None` for words before the first tag.
    pub speaker_tag: Option<String>,
    pub words: Vec<String>,
    /// Blank-expanded targets as compact columns (blank = 0).
    pub labels: Vec<u32>,
    /// Word of each target state; `None` for blanks.
    pub state_to_word: Vec<Option<usize>>,
}

/// A transcript turned into CTC targets: what [`align`] needs besides the
/// log-probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcPrepared {
    /// CTC classes the alignment reads, blank first, then every label the
    /// transcript uses in ascending order. Gather these columns, in this
    /// order, from the full-vocabulary log-softmax.
    pub columns: Vec<u32>,
    /// Valid CTC frames the transcript was checked against.
    pub ctc_frames: usize,
    pub(crate) streams: Vec<PreparedStream>,
}

impl CtcPrepared {
    /// Whether the transcript has no words to align.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Streams and targets from words that already carry their CTC ids
    /// (upstream `_prepare` after `_tokenize_words`).
    pub(crate) fn from_tokenized(
        words: Vec<(SotWord, Vec<u32>)>,
        ctc_frames: usize,
        blank: u32,
    ) -> Result<Self, CtcTimestampError> {
        let mut streams: Vec<PreparedStream> = Vec::new();
        for (word, ids) in words {
            let index = match streams
                .iter()
                .position(|s| s.speaker_tag == word.speaker_tag)
            {
                Some(index) => index,
                None => {
                    streams.push(PreparedStream {
                        speaker_tag: word.speaker_tag.clone(),
                        words: Vec::new(),
                        labels: vec![blank],
                        state_to_word: vec![None],
                    });
                    streams.len() - 1
                }
            };
            let stream = &mut streams[index];
            let word_index = stream.words.len();
            stream.words.push(word.word);
            for id in ids {
                stream.labels.extend([id, blank]);
                stream.state_to_word.extend([Some(word_index), None]);
            }
        }
        for stream in &streams {
            let tokens: Vec<u32> = stream.labels.iter().skip(1).step_by(2).copied().collect();
            let repeats = tokens.windows(2).filter(|pair| pair[0] == pair[1]).count();
            if tokens.len() + repeats > ctc_frames {
                // "Speaker transcript is too long for the CTC timeline."
                return Err(CtcTimestampError::AlignmentFailed);
            }
        }
        let mut columns: Vec<u32> = streams
            .iter()
            .flat_map(|stream| stream.labels.iter().copied())
            .filter(|&label| label != blank)
            .collect();
        columns.sort_unstable();
        columns.dedup();
        columns.insert(0, blank);
        let column_of: HashMap<u32, u32> = columns
            .iter()
            .enumerate()
            .map(|(column, &label)| (label, column as u32))
            .collect();
        for stream in &mut streams {
            for label in &mut stream.labels {
                *label = column_of[label];
            }
        }
        Ok(Self {
            columns,
            ctc_frames,
            streams,
        })
    }
}

/// Turns a transcript into CTC targets (upstream `_prepare` without the
/// log-prob read).
///
/// * `transcript`: the decoded t-SOT text with speaker, event and style
///   tags kept and other special tokens removed (`clean_sot_transcript`);
/// * `language`: the aligner's `timestamp_language`; `None` detects the
///   script per transcript, which is what NeMo serves with (zh/ja align
///   characters; every other language whitespace words);
/// * `ctc_frames`: valid CTC frames (40 ms each);
/// * `blank`: the CTC blank class (the last, 23,862 for the published head).
///
/// Fails with `AlignmentFailed` where upstream raises `ValueError`: no
/// valid frames, a token id not below `blank`, an empty word tokenization,
/// per-word and whole-stream tokenizations that disagree, a character the
/// vocabulary cannot represent, or a stream with more tokens than frames.
pub fn prepare(
    tokenizer: &CtcTokenizer,
    transcript: &str,
    language: Option<&str>,
    ctc_frames: usize,
    blank: u32,
    config: &CtcAlignmentConfig,
) -> Result<CtcPrepared, CtcTimestampError> {
    if ctc_frames == 0 || !config.is_valid() {
        return Err(CtcTimestampError::AlignmentFailed);
    }
    let units = timestamp_units(parse_sot_words(transcript), language);
    let ids = tokenize_words(tokenizer, &units, blank)?;
    CtcPrepared::from_tokenized(units.into_iter().zip(ids).collect(), ctc_frames, blank)
}

/// `_tokenize_words`: CTC ids per word, each speaker stream tokenized on
/// its own, in transcript order.
fn tokenize_words(
    tokenizer: &CtcTokenizer,
    words: &[SotWord],
    blank: u32,
) -> Result<Vec<Vec<u32>>, CtcTimestampError> {
    let mut tags: Vec<&Option<String>> = Vec::new();
    for word in words {
        if !tags.contains(&&word.speaker_tag) {
            tags.push(&word.speaker_tag);
        }
    }
    let mut ids = vec![Vec::new(); words.len()];
    for tag in tags {
        let indices: Vec<usize> = (0..words.len())
            .filter(|&i| &words[i].speaker_tag == tag)
            .collect();
        let stream: Vec<&SotWord> = indices.iter().map(|&i| &words[i]).collect();
        for (index, word_ids) in indices
            .into_iter()
            .zip(tokenize_word_stream(tokenizer, &stream, blank)?)
        {
            ids[index] = word_ids;
        }
    }
    Ok(ids)
}

/// `_tokenize_word_stream`: character units on their own; otherwise the
/// joined stream, which must equal the concatenated per-word ids.
fn tokenize_word_stream(
    tokenizer: &CtcTokenizer,
    words: &[&SotWord],
    blank: u32,
) -> Result<Vec<Vec<u32>>, CtcTimestampError> {
    const FAILED: CtcTimestampError = CtcTimestampError::AlignmentFailed;
    if words.iter().any(|word| word.character) {
        return words
            .iter()
            .map(|word| {
                if word.character {
                    character_token_ids(tokenizer, &word.word, blank).ok_or(FAILED)
                } else {
                    tokenize_word_stream(tokenizer, &[word], blank)
                        .and_then(|ids| ids.into_iter().next().ok_or(FAILED))
                }
            })
            .collect();
    }
    let text = words
        .iter()
        .map(|word| word.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let stream_ids = tokenizer.encode(&text);
    if stream_ids.is_empty() || stream_ids.iter().any(|&id| id >= blank) {
        return Err(FAILED);
    }
    let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
    let mut per_word = Vec::with_capacity(words.len());
    for word in words {
        let ids = cache
            .entry(word.word.as_str())
            .or_insert_with(|| tokenizer.encode(&word.word));
        if ids.is_empty() {
            return Err(FAILED);
        }
        per_word.push(ids.clone());
    }
    if per_word.iter().flatten().ne(stream_ids.iter()) {
        return Err(FAILED);
    }
    Ok(per_word)
}

/// Row-major `frames × speakers` activity (probabilities in `[0, 1]`).
#[derive(Debug, Clone, Copy)]
pub struct SpeakerActivity<'a> {
    pub values: &'a [f32],
    pub frames: usize,
    pub speakers: usize,
}

impl SpeakerActivity<'_> {
    fn is_valid(&self) -> bool {
        self.frames > 0 && self.values.len() == self.frames * self.speakers
    }
}

/// What [`align`] reads besides the prepared transcript.
#[derive(Debug, Clone, Copy)]
pub struct CtcAlignmentInputs<'a> {
    /// `ctc_frames × prepared.columns.len()`, row-major: the log-softmax
    /// over the full vocabulary, then the prepared columns gathered.
    pub log_probs: &'a [f32],
    /// The diarizer's continuous 80 ms sigmoids, valid frames only
    /// (upstream clamps them to the encoder length); `None` aligns without
    /// speaker information.
    pub speaker_probs: Option<SpeakerActivity<'a>>,
    /// The diarizer's native 10 ms probabilities, valid frames only;
    /// thresholded `> 0.5` into the diarization segments.
    pub diarization: Option<SpeakerActivity<'a>>,
    /// Audio duration in seconds; one CTC frame is `duration / ctc_frames`.
    pub audio_duration: f64,
}

/// One aligned word.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CtcWord {
    pub word: String,
    /// Seconds, rounded to milliseconds.
    pub start: f64,
    /// Seconds, rounded to milliseconds.
    pub end: f64,
    /// `str(tag)` of the word's `<spk:N>` tag: `"0"` throughout a transcript
    /// without tags, `"None"` for words before the first tag (as upstream).
    pub speaker: String,
}

/// One diarization segment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CtcSegment {
    /// Diarizer column.
    pub speaker: usize,
    /// Seconds, rounded to milliseconds.
    pub start: f64,
    /// Seconds, rounded to milliseconds.
    pub end: f64,
}

/// vLLM's public `ctc_timestamps` object.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CtcTimestamps {
    /// Words in start-time order (ties keep speaker-stream order).
    pub words: Vec<CtcWord>,
    /// 10 ms diarization segments in start-time order.
    pub diarization: Vec<CtcSegment>,
    /// `str(tag)` → diarizer column (`None` when unmapped).
    pub speaker_tag_to_diarization_speaker: BTreeMap<String, Option<usize>>,
    /// `None` when `words` is complete (an empty transcript included).
    pub error: Option<CtcTimestampError>,
}

impl CtcTimestamps {
    /// A result without words, diarization or mapping.
    pub fn failed(error: CtcTimestampError) -> Self {
        Self::without_words(Some(error), Vec::new())
    }

    /// A result without words that keeps the diarization (upstream returns
    /// it whenever the capture exists, even when alignment fails).
    pub fn without_words(error: Option<CtcTimestampError>, diarization: Vec<CtcSegment>) -> Self {
        Self {
            words: Vec::new(),
            diarization,
            speaker_tag_to_diarization_speaker: BTreeMap::new(),
            error,
        }
    }
}

/// `diarization_segments` + `_public_diarization`: each contiguous run of
/// frames above 0.5 becomes a segment, its end clipped to the audio;
/// segments are sorted by start, end, column and rounded to milliseconds.
/// An activity of the wrong shape yields no segments.
pub fn diarization_segments(
    activity: SpeakerActivity<'_>,
    audio_duration: f64,
    config: &CtcAlignmentConfig,
) -> Vec<CtcSegment> {
    raw_diarization_segments(activity, audio_duration, config.diarization_frame_seconds)
        .into_iter()
        .map(|(speaker, start, end)| CtcSegment {
            speaker,
            start: round_milliseconds(start),
            end: round_milliseconds(end),
        })
        .collect()
}

/// Unrounded `(column, start, end)` segments.
fn raw_diarization_segments(
    activity: SpeakerActivity<'_>,
    audio_duration: f64,
    frame_seconds: f64,
) -> Vec<(usize, f64, f64)> {
    if !activity.is_valid() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    for speaker in 0..activity.speakers {
        let active = |frame: usize| activity.values[frame * activity.speakers + speaker] > 0.5;
        let mut frame = 0;
        while frame < activity.frames {
            if !active(frame) {
                frame += 1;
                continue;
            }
            let first = frame;
            while frame < activity.frames && active(frame) {
                frame += 1;
            }
            let start = 0.0 + first as f64 * frame_seconds;
            let end = (0.0 + frame as f64 * frame_seconds).min(0.0 + audio_duration);
            if end > start {
                segments.push((speaker, start, end));
            }
        }
    }
    segments.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then(a.2.total_cmp(&b.2))
            .then(a.0.cmp(&b.0))
    });
    segments
}

/// Python's `round(value, 3)`: the decimal correctly rounded to three
/// places, ties to even, read back as the nearest double.
pub(crate) fn round_milliseconds(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.3}").parse().unwrap_or(value)
}

/// One aligned word before refinement.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WordRow {
    pub word: String,
    pub speaker_tag: Option<String>,
    pub start: f64,
    pub end: f64,
}

/// Aligns a prepared transcript (upstream `_align_banded_ctc` and vLLM's
/// `_public_result`). Streams with no legal path in their band lose their
/// words without an error, as upstream; the whole result fails
/// (`AlignmentFailed`, diarization kept) where upstream raises: inputs of
/// the wrong shape, no speaker frames, a non-positive or non-finite frame
/// duration, or an unsolvable speaker assignment.
pub fn align(
    prepared: &CtcPrepared,
    inputs: CtcAlignmentInputs<'_>,
    config: &CtcAlignmentConfig,
) -> CtcTimestamps {
    let diarization = inputs.diarization.map_or_else(Vec::new, |activity| {
        raw_diarization_segments(
            activity,
            inputs.audio_duration,
            config.diarization_frame_seconds,
        )
    });
    let public_diarization: Vec<CtcSegment> = diarization
        .iter()
        .map(|&(speaker, start, end)| CtcSegment {
            speaker,
            start: round_milliseconds(start),
            end: round_milliseconds(end),
        })
        .collect();
    let aligned = trace_alignment(prepared, &inputs, config).and_then(|trace| {
        let rows = refine_word_timestamps(
            trace.rows,
            &trace.mapping,
            trace.step,
            &diarization,
            config.max_word_frames,
        )?;
        Some((rows, trace.mapping))
    });
    match aligned {
        Some((rows, mapping)) => {
            let mut words: Vec<CtcWord> = rows
                .into_iter()
                .map(|row| CtcWord {
                    word: row.word,
                    start: round_milliseconds(row.start),
                    end: round_milliseconds(row.end),
                    speaker: python_str(&row.speaker_tag),
                })
                .collect();
            words.sort_by(|a, b| {
                a.start
                    .partial_cmp(&b.start)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(
                        a.end
                            .partial_cmp(&b.end)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
            });
            CtcTimestamps {
                words,
                diarization: public_diarization,
                speaker_tag_to_diarization_speaker: mapping
                    .into_iter()
                    .map(|(tag, column)| (python_str(&tag), column))
                    .collect(),
                error: None,
            }
        }
        None => CtcTimestamps::without_words(
            Some(CtcTimestampError::AlignmentFailed),
            public_diarization,
        ),
    }
}

/// `str(tag)` of an optional speaker tag.
fn python_str(tag: &Option<String>) -> String {
    tag.clone().unwrap_or_else(|| "None".to_string())
}

/// One stream's search, for tests and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StreamTrace {
    /// Mapped diarizer column.
    pub column: Option<usize>,
    /// Frames searched (speaker activity at least 0.01).
    pub retained: usize,
    pub outcome: SearchOutcome,
}

/// Everything `align` computes before refinement.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AlignmentTrace {
    /// `(tag, column)` per stream, in stream order.
    pub mapping: Vec<(Option<String>, Option<usize>)>,
    pub streams: Vec<StreamTrace>,
    /// Word rows of the aligned streams, in stream order.
    pub rows: Vec<WordRow>,
    /// Seconds per CTC frame.
    pub step: f64,
}

/// Speaker mapping, per-stream search and word rows (upstream
/// `_align_banded_ctc` up to `_refine_word_timestamps`); `None` where
/// upstream raises.
pub(crate) fn trace_alignment(
    prepared: &CtcPrepared,
    inputs: &CtcAlignmentInputs<'_>,
    config: &CtcAlignmentConfig,
) -> Option<AlignmentTrace> {
    let frames = prepared.ctc_frames;
    let columns = prepared.columns.len();
    if !config.is_valid() || frames == 0 || inputs.log_probs.len() != frames * columns {
        return None;
    }
    if inputs
        .diarization
        .is_some_and(|activity| !activity.is_valid())
    {
        return None;
    }
    let ctc = inputs.log_probs;
    let greedy: Vec<u32> = ctc
        .chunks_exact(columns)
        .map(|row| argmax(row) as u32)
        .collect();

    let speaker = match inputs.speaker_probs {
        None => None,
        Some(activity) if !activity.is_valid() => return None,
        Some(activity) => Some((
            resample_speaker_probs(activity.values, activity.frames, activity.speakers, frames),
            activity.speakers,
        )),
    };
    let mapping: Vec<Option<usize>> = match &speaker {
        None => vec![None; prepared.streams.len()],
        Some((probs, speakers)) => {
            let expected: Vec<Vec<u32>> = prepared
                .streams
                .iter()
                .map(|stream| stream.labels.iter().skip(1).step_by(2).copied().collect())
                .collect();
            speaker_mapping(
                &greedy,
                columns,
                &expected,
                probs,
                *speakers,
                config.speaker_activity_threshold,
            )?
        }
    };

    let step = inputs.audio_duration / frames as f64;
    let mut rows: Vec<WordRow> = Vec::new();
    let mut traces = Vec::with_capacity(prepared.streams.len());
    for (stream, &column) in prepared.streams.iter().zip(&mapping) {
        // Conservative support selection, separate from the > 0.5 labels.
        let (positions, prior): (Vec<usize>, Option<Vec<f32>>) = match (&speaker, column) {
            (None, _) => ((0..frames).collect(), None),
            (Some(_), None) => (Vec::new(), None),
            (Some((probs, speakers)), Some(column)) => {
                let activity: Vec<f32> =
                    (0..frames).map(|t| probs[t * speakers + column]).collect();
                let positions = (0..frames).filter(|&t| activity[t] >= 0.01f32).collect();
                let prior = activity
                    .iter()
                    .map(|&p| config.speaker_logprob_weight * nan_max(p, config.epsilon).ln())
                    .collect();
                (positions, Some(prior))
            }
        };
        let outcome = banded_ctc_search(
            ctc,
            columns,
            &stream.labels,
            &positions,
            prior.as_deref(),
            &greedy,
        );
        // A stream with no legal path is a failed stream upstream: its
        // words are dropped, the result is not an error.
        if let Some(path) = outcome.path.as_ref().filter(|_| outcome.score.is_finite()) {
            let mut word_frames: Vec<Option<(usize, usize)>> = vec![None; stream.words.len()];
            for (&frame, &state) in positions.iter().zip(path) {
                if let Some(word) = stream.state_to_word.get(state).copied().flatten() {
                    let span = word_frames[word].get_or_insert((frame, frame));
                    span.1 = frame;
                }
            }
            for (word, span) in stream.words.iter().zip(word_frames) {
                // "CTC path did not visit word": impossible for a legal path.
                let (first, last) = span?;
                let start = 0.0 + first as f64 * step;
                let end = (0.0 + (last + 1) as f64 * step).min(start + config.maximum_token_len);
                rows.push(WordRow {
                    word: word.clone(),
                    speaker_tag: stream.speaker_tag.clone(),
                    start,
                    end,
                });
            }
        }
        traces.push(StreamTrace {
            column,
            retained: positions.len(),
            outcome,
        });
    }

    Some(AlignmentTrace {
        mapping: prepared
            .streams
            .iter()
            .zip(&mapping)
            .map(|(stream, &column)| (stream.speaker_tag.clone(), column))
            .collect(),
        streams: traces,
        rows,
        step,
    })
}

/// `np.maximum`: NaN propagates.
fn nan_max(value: f32, floor: f32) -> f32 {
    if value.is_nan() || value >= floor {
        value
    } else {
        floor
    }
}

/// `_resample_speaker_probs` after `clamp(0, 1)`: linear interpolation of
/// `frames × speakers` probabilities onto `target` frames at the positions
/// `torch.linspace(0, frames - 1, target)` gives. torch's CPU kernel
/// computes the first half as `step·i` and the second as
/// `end - step·(target - 1 - i)`; its arm64 build fuses each into one
/// rounding, which `mul_add` reproduces (a build without FMA can differ by
/// one ulp of a position).
pub(crate) fn resample_speaker_probs(
    values: &[f32],
    frames: usize,
    speakers: usize,
    target: usize,
) -> Vec<f32> {
    let clamped: Vec<f32> = values.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
    if frames == target {
        return clamped;
    }
    if frames == 1 {
        return (0..target).flat_map(|_| clamped.iter().copied()).collect();
    }
    let start = 0.0f32;
    let end = (frames - 1) as f32;
    let step = if target > 1 {
        (end - start) / (target - 1) as f32
    } else {
        0.0
    };
    let halfway = target / 2;
    let mut output = Vec::with_capacity(target * speakers);
    for i in 0..target {
        let position = if target == 1 {
            start
        } else if i < halfway {
            step.mul_add(i as f32, start)
        } else {
            (-step).mul_add((target - i - 1) as f32, end)
        };
        let lower = (position.floor() as usize).min(frames - 1);
        let upper = (position.ceil() as usize).min(frames - 1);
        let fraction = position - lower as f32;
        let keep = 1.0 - fraction;
        for s in 0..speakers {
            let low = clamped[lower * speakers + s] * keep;
            let high = clamped[upper * speakers + s] * fraction;
            output.push(low + high);
        }
    }
    output
}

/// `_refine_word_timestamps`: within each speaker's diarization segments
/// (of its mapped column), the first word of a segment snaps its onset back
/// to the segment onset and every word's end extends to the next word in
/// the segment or the segment end; unmatched words end at the next word of
/// their speaker. Moves and durations are capped at `max_frames` frames of
/// `min(step, 0.08)` seconds. `None` for a non-positive or non-finite step.
fn refine_word_timestamps(
    rows: Vec<WordRow>,
    mapping: &[(Option<String>, Option<usize>)],
    step: f64,
    diarization: &[(usize, f64, f64)],
    max_frames: usize,
) -> Option<Vec<WordRow>> {
    if !step.is_finite() || step <= 0.0 || max_frames == 0 {
        return None;
    }
    let limit = max_frames as f64 * if 0.08 < step { 0.08 } else { step };
    let mut segments: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();
    for &(speaker, start, end) in diarization {
        if end > start {
            segments.entry(speaker).or_default().push((start, end));
        }
    }
    for intervals in segments.values_mut() {
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    }

    // Group by speaker in order of first appearance.
    let mut groups: Vec<(Option<String>, Vec<WordRow>)> = Vec::new();
    for row in rows {
        match groups.iter_mut().find(|(tag, _)| *tag == row.speaker_tag) {
            Some((_, words)) => words.push(row),
            None => groups.push((row.speaker_tag.clone(), vec![row])),
        }
    }
    let empty = Vec::new();
    let mut refined = Vec::new();
    for (tag, mut words) in groups {
        words.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let column = mapping
            .iter()
            .find(|(mapped, _)| *mapped == tag)
            .and_then(|&(_, column)| column);
        let intervals = column.and_then(|c| segments.get(&c)).unwrap_or(&empty);
        let memberships: Vec<Option<usize>> = words
            .iter()
            .map(|word| {
                let index = intervals.partition_point(|&(start, _)| start <= word.start);
                index
                    .checked_sub(1)
                    .filter(|&index| word.start < intervals[index].1)
            })
            .collect();
        let mut seen = Vec::new();
        for (word, &membership) in words.iter_mut().zip(&memberships) {
            if let Some(index) = membership {
                if !seen.contains(&index) {
                    let onset = intervals[index].0;
                    let moved = word.start - onset;
                    if 0.0 < moved && moved <= limit {
                        word.start = onset;
                    }
                    seen.push(index);
                }
            }
        }
        for i in 0..words.len() {
            let end = match memberships[i] {
                None => words.get(i + 1).map_or(words[i].end, |next| next.start),
                Some(index) if memberships.get(i + 1) == Some(&Some(index)) => words[i + 1].start,
                Some(index) => intervals[index].1,
            };
            let start = words[i].start;
            let capped = if start + limit < end {
                start + limit
            } else {
                end
            };
            words[i].end = if capped > start { capped } else { start };
        }
        refined.extend(words);
    }
    Some(refined)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(word: &str, tag: Option<&str>) -> SotWord {
        SotWord {
            word: word.to_string(),
            speaker_tag: tag.map(str::to_string),
            turn_index: None,
            character: false,
        }
    }

    #[test]
    fn targets_interleave_blanks_and_renumber_columns() {
        let words = vec![
            (unit("a", Some("1")), vec![7, 3]),
            (unit("b", Some("0")), vec![9]),
            (unit("c", Some("1")), vec![3]),
        ];
        let prepared = CtcPrepared::from_tokenized(words, 10, 50).unwrap();
        assert_eq!(prepared.columns, vec![50, 3, 7, 9]);
        assert_eq!(prepared.streams.len(), 2);
        let first = &prepared.streams[0];
        assert_eq!(first.speaker_tag.as_deref(), Some("1"));
        assert_eq!(first.words, vec!["a", "c"]);
        // [blank, 7, blank, 3, blank, 3, blank] as columns.
        assert_eq!(first.labels, vec![0, 2, 0, 1, 0, 1, 0]);
        assert_eq!(
            first.state_to_word,
            vec![None, Some(0), None, Some(0), None, Some(1), None]
        );
        assert_eq!(prepared.streams[1].labels, vec![0, 3, 0]);
        // The repeated 3 needs a blank frame: 3 tokens + 1 repeat > 3.
        let words = vec![
            (unit("a", Some("1")), vec![7, 3]),
            (unit("c", Some("1")), vec![3]),
        ];
        assert_eq!(
            CtcPrepared::from_tokenized(words.clone(), 3, 50),
            Err(CtcTimestampError::AlignmentFailed)
        );
        assert!(CtcPrepared::from_tokenized(words, 4, 50).is_ok());
        let empty = CtcPrepared::from_tokenized(Vec::new(), 4, 50).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.columns, vec![50]);
    }

    #[test]
    fn resampling_matches_torch_linspace() {
        // linspace(0, 2, 6) = [0, 0.4, 0.8, 1.2, 1.6, 2.0] (step 0.4f32).
        let probs = [0.0, 1.0, 0.5, 0.0, 1.0, 1.0];
        let out = resample_speaker_probs(&probs, 3, 2, 6);
        assert_eq!(out.len(), 12);
        assert_eq!(&out[0..2], &[0.0, 1.0]);
        assert_eq!(&out[10..12], &[1.0, 1.0]);
        let step = 2.0f32 / 5.0;
        let position = step * 2.0;
        let fraction = position - 0.0;
        assert_eq!(out[4], 0.0 * (1.0 - fraction) + 0.5 * fraction);
        // Clamped; the same length is a copy; one frame repeats.
        assert_eq!(
            resample_speaker_probs(&[1.5, -0.5], 1, 2, 3),
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(resample_speaker_probs(&[0.2, 0.3], 2, 1, 2), vec![0.2, 0.3]);
    }

    #[test]
    fn diarization_runs_become_sorted_clipped_segments() {
        // Two speakers, 10 ms frames; speaker 1 runs past the audio end.
        let values = [
            0.9, 0.1, //
            0.9, 0.6, //
            0.2, 0.6, //
            0.7, 0.6, //
            0.7, 0.9,
        ];
        let activity = SpeakerActivity {
            values: &values,
            frames: 5,
            speakers: 2,
        };
        let config = CtcAlignmentConfig::default();
        let segments = diarization_segments(activity, 0.045, &config);
        let flat: Vec<(usize, f64, f64)> = segments
            .iter()
            .map(|s| (s.speaker, s.start, s.end))
            .collect();
        assert_eq!(
            flat,
            vec![(0, 0.0, 0.02), (1, 0.01, 0.045), (0, 0.03, 0.045)]
        );
        // A wrong shape yields nothing.
        let bad = SpeakerActivity {
            values: &values[..3],
            frames: 5,
            speakers: 2,
        };
        assert!(diarization_segments(bad, 1.0, &config).is_empty());
    }

    #[test]
    fn rounding_is_pythons() {
        assert_eq!(round_milliseconds(0.0625), 0.062);
        assert_eq!(round_milliseconds(0.0635), 0.064);
        assert_eq!(round_milliseconds(1.0005), 1.0);
        assert_eq!(round_milliseconds(2.675), 2.675);
        assert_eq!(round_milliseconds(0.12345), 0.123);
        assert_eq!(round_milliseconds(12.3456), 12.346);
    }

    fn row(word: &str, tag: &str, start: f64, end: f64) -> WordRow {
        WordRow {
            word: word.to_string(),
            speaker_tag: Some(tag.to_string()),
            start,
            end,
        }
    }

    #[test]
    fn refinement_snaps_onsets_and_extends_ends_within_segments() {
        let rows = vec![
            row("a", "0", 1.10, 1.30),
            row("b", "0", 1.50, 1.60),
            row("c", "0", 3.00, 3.20),
            row("x", "1", 0.50, 0.70),
        ];
        let mapping = vec![
            (Some("0".to_string()), Some(1)),
            (Some("1".to_string()), None),
        ];
        let diarization = vec![(1, 1.0, 2.0), (1, 2.9, 3.05), (0, 0.0, 9.0)];
        let refined = refine_word_timestamps(rows, &mapping, 0.04, &diarization, 24).unwrap();
        let limit = 24.0 * 0.04;
        let flat: Vec<(&str, f64, f64)> = refined
            .iter()
            .map(|r| (r.word.as_str(), r.start, r.end))
            .collect();
        assert_eq!(
            flat,
            vec![
                // First word of [1.0, 2.0): onset snaps to 1.0; ends at "b".
                ("a", 1.0, 1.50),
                // Last word of the segment: ends at the segment end, capped.
                ("b", 1.50, (1.50f64 + limit).min(2.0)),
                // First word of [2.9, 3.05): 3.0 snaps to 2.9; ends at 3.05.
                ("c", 2.9, 3.05),
                // Unmapped speaker: last word keeps its end.
                ("x", 0.50, 0.70),
            ]
        );
        assert!(refine_word_timestamps(Vec::new(), &[], 0.0, &[], 24).is_none());
        assert!(refine_word_timestamps(Vec::new(), &[], f64::NAN, &[], 24).is_none());
    }
}
