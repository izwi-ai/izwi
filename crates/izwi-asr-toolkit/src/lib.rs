#![forbid(unsafe_code)]

use std::cmp;

use izwi_vad::{
    configured_snr_margin_db, detect_speech_regions_f32, SpeechEvidence, SpeechEvidenceConfig,
    VadRegionConfig, VadScorer, DEFAULT_EVIDENCE_WINDOW_MS, DEFAULT_EXIT_THRESHOLD,
    DEFAULT_MIN_EVIDENCE_MS, DEFAULT_SPEECH_THRESHOLD,
};

/// Tunables for long-form ASR chunking and transcript stitching.
#[derive(Debug, Clone)]
pub struct AsrLongFormConfig {
    /// Preferred chunk length before split-point search.
    pub target_chunk_secs: f32,
    /// Hard upper bound for a chunk. Chunker will not produce longer chunks.
    pub hard_max_chunk_secs: f32,
    /// Overlap between consecutive chunks to protect boundary words.
    pub overlap_secs: f32,
    /// Search window around the preferred boundary for silence/low-energy points.
    pub silence_search_secs: f32,
    /// Minimum chunk length.
    pub min_chunk_secs: f32,
    /// RMS probe frame length for split-point analysis.
    pub analysis_frame_ms: u32,
    /// Quantile used to estimate low-energy threshold in a search window.
    pub silence_energy_quantile: f32,
    /// Scale factor over quantile energy to decide candidate low-energy points.
    pub silence_energy_scale: f32,
    /// Minimum overlap tokens required before deduping chunk boundaries.
    pub min_word_overlap: usize,
    /// Maximum overlap tokens to compare while deduping.
    pub max_word_overlap: usize,
    /// Minimum replayed context words required before trimming a long repeated
    /// prefix from a later chunk.
    pub min_context_replay_words: usize,
    /// Maximum replayed context words to compare when trimming a later chunk.
    pub max_context_replay_words: usize,
    /// Maximum consecutive repeat count per character in chunk cleanup.
    pub max_repeated_chars: usize,
    /// Cut each chunk in the middle of the longest pause in the search window
    /// instead of at the quiet frame nearest the preferred boundary. Long
    /// pauses are where speakers usually change, and a cut there keeps a
    /// phrase whole on both sides of the seam.
    pub prefer_longest_pause: bool,
}

impl Default for AsrLongFormConfig {
    fn default() -> Self {
        Self {
            target_chunk_secs: 24.0,
            hard_max_chunk_secs: 30.0,
            overlap_secs: 3.0,
            silence_search_secs: 4.0,
            min_chunk_secs: 8.0,
            analysis_frame_ms: 20,
            silence_energy_quantile: 0.2,
            silence_energy_scale: 2.5,
            min_word_overlap: 3,
            max_word_overlap: 24,
            min_context_replay_words: 8,
            max_context_replay_words: 120,
            max_repeated_chars: 8,
            prefer_longest_pause: false,
        }
    }
}

/// A planned audio chunk in sample indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioChunk {
    pub start_sample: usize,
    pub end_sample: usize,
}

impl AudioChunk {
    pub fn len_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }
}

/// A detected speech region in original input sample indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechRegion {
    pub start_sample: usize,
    pub end_sample: usize,
}

impl SpeechRegion {
    pub fn len_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }
}

/// Tunables for speech-island planning before Whisper-style long-form ASR.
#[derive(Debug, Clone)]
pub struct AsrSpeechChunkConfig {
    /// Earshot score required to enter speech.
    pub start_threshold: f32,
    /// Earshot score below which active speech may exit.
    pub end_threshold: f32,
    /// Minimum continuous speech duration required to emit a region.
    pub min_speech_secs: f32,
    /// Minimum continuous silence duration required to close a region.
    pub min_silence_secs: f32,
    /// Padding applied to both sides of each detected speech region.
    pub speech_pad_secs: f32,
    /// Gaps at or below this length are merged into one speech region.
    pub merge_gap_secs: f32,
    /// Energy above the noise floor, in dB, a frame needs to start a region or
    /// to count as speech evidence. Earshot is level-blind, so without it
    /// steady noise scores as speech. Defaults to `IZWI_VAD_SNR_MARGIN_DB`.
    pub snr_margin_db: f32,
    /// Speech-qualified time one second of input must hold for
    /// [`scan_speech_presence`] to report speech.
    pub min_evidence_secs: f32,
}

impl Default for AsrSpeechChunkConfig {
    fn default() -> Self {
        Self {
            start_threshold: DEFAULT_SPEECH_THRESHOLD,
            end_threshold: DEFAULT_EXIT_THRESHOLD,
            min_speech_secs: 0.25,
            min_silence_secs: 0.45,
            speech_pad_secs: 0.2,
            merge_gap_secs: 0.35,
            snr_margin_db: configured_snr_margin_db(),
            min_evidence_secs: DEFAULT_MIN_EVIDENCE_MS as f32 / 1000.0,
        }
    }
}

/// Result from VAD-style speech chunk planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechChunkPlan {
    pub chunks: Vec<AudioChunk>,
    pub speech_regions: Vec<SpeechRegion>,
    pub speech_samples: usize,
    pub included_samples: usize,
    pub skipped_samples: usize,
    pub no_speech: bool,
}

impl SpeechChunkPlan {
    fn no_speech(total_samples: usize) -> Self {
        Self {
            chunks: Vec::new(),
            speech_regions: Vec::new(),
            speech_samples: 0,
            included_samples: 0,
            skipped_samples: total_samples,
            no_speech: true,
        }
    }
}

/// Build chunk boundaries for long-form ASR.
///
/// `model_max_chunk_secs` lets model backends provide a tighter limit (for example,
/// ASR frontends with strict max-frame settings).
pub fn plan_audio_chunks(
    samples: &[f32],
    sample_rate: u32,
    config: &AsrLongFormConfig,
    model_max_chunk_secs: Option<f32>,
) -> Vec<AudioChunk> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    let total = samples.len();
    let sr = sample_rate as f32;

    let mut hard_max_secs = config
        .hard_max_chunk_secs
        .max(config.min_chunk_secs.max(0.1));
    if let Some(limit) = model_max_chunk_secs.filter(|v| v.is_finite() && *v > 0.0) {
        hard_max_secs = hard_max_secs.min(limit.max(config.min_chunk_secs.max(0.1)));
    }

    let target_secs = config
        .target_chunk_secs
        .max(config.min_chunk_secs.max(0.1))
        .min(hard_max_secs);
    let overlap_secs = config.overlap_secs.clamp(0.0, target_secs * 0.45);
    let search_secs = config.silence_search_secs.clamp(0.0, target_secs * 0.5);

    let min_chunk_samples = secs_to_samples(config.min_chunk_secs.max(0.1), sr).max(1);
    let target_samples = secs_to_samples(target_secs, sr).max(min_chunk_samples);
    let hard_max_samples = secs_to_samples(hard_max_secs, sr).max(min_chunk_samples);
    let overlap_samples = secs_to_samples(overlap_secs, sr).min(hard_max_samples / 2);
    let search_samples = secs_to_samples(search_secs, sr);
    let frame_samples = ms_to_samples(config.analysis_frame_ms.max(5), sample_rate).max(1);

    if total <= hard_max_samples {
        return vec![AudioChunk {
            start_sample: 0,
            end_sample: total,
        }];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut safety = 0usize;

    while start < total {
        safety += 1;
        if safety > 10_000 {
            break;
        }

        let remaining = total.saturating_sub(start);
        if remaining <= hard_max_samples {
            chunks.push(AudioChunk {
                start_sample: start,
                end_sample: total,
            });
            break;
        }

        let min_end = (start + min_chunk_samples).min(total);
        let preferred_end = (start + target_samples).min(total);
        let max_end = (start + hard_max_samples).min(total);

        let search_low = preferred_end.saturating_sub(search_samples).max(min_end);
        let search_high = preferred_end.saturating_add(search_samples).min(max_end);

        let mut split = preferred_end.clamp(min_end, max_end);
        if search_high > search_low + frame_samples {
            let found = if config.prefer_longest_pause {
                choose_longest_pause_split_point(
                    samples,
                    search_low,
                    search_high,
                    preferred_end,
                    frame_samples,
                    config.silence_energy_scale,
                )
            } else {
                None
            }
            .or_else(|| {
                choose_split_point(
                    samples,
                    search_low,
                    search_high,
                    preferred_end,
                    frame_samples,
                    config.silence_energy_quantile,
                    config.silence_energy_scale,
                )
            });
            if let Some(found) = found {
                split = found;
            }
        }

        split = split.clamp(min_end, max_end);
        if split <= start {
            break;
        }

        chunks.push(AudioChunk {
            start_sample: start,
            end_sample: split,
        });

        if split >= total {
            break;
        }

        let mut next_start = split.saturating_sub(overlap_samples);
        // Open the next chunk at a quiet point near the nominal overlap start,
        // so it rarely begins on a word fragment that defeats boundary dedup.
        // Searching half the overlap either side keeps at least half of it.
        let snap_samples = overlap_samples / 2;
        if snap_samples > frame_samples {
            let snap_low = next_start.saturating_sub(snap_samples).max(start + 1);
            let snap_high = (next_start + snap_samples).min(split.saturating_sub(frame_samples));
            if snap_high > snap_low + frame_samples {
                if let Some(found) = choose_split_point(
                    samples,
                    snap_low,
                    snap_high,
                    next_start,
                    frame_samples,
                    config.silence_energy_quantile,
                    config.silence_energy_scale,
                ) {
                    next_start = found;
                }
            }
        }
        if next_start <= start {
            next_start = split;
        }

        if total.saturating_sub(next_start) < min_chunk_samples {
            // Fold a short tail into the last chunk only while that chunk
            // stays within the hard maximum; the model cannot see past it.
            if let Some(last) = chunks
                .last_mut()
                .filter(|last| total - last.start_sample <= hard_max_samples)
            {
                last.end_sample = total;
                break;
            }
        }

        start = next_start;
    }

    if chunks.is_empty() {
        chunks.push(AudioChunk {
            start_sample: 0,
            end_sample: total,
        });
    }

    chunks
}

/// Build chunk boundaries from detected speech islands.
///
/// This keeps chunks in the original waveform timeline instead of concatenating
/// speech-only audio. That makes the first version low-risk: transcript timing
/// can still be expressed as original sample offsets, and the existing
/// per-chunk Whisper path receives ordinary waveform slices.
pub fn plan_speech_audio_chunks(
    samples: &[f32],
    sample_rate: u32,
    long_form_config: &AsrLongFormConfig,
    speech_config: &AsrSpeechChunkConfig,
    model_max_chunk_secs: Option<f32>,
) -> SpeechChunkPlan {
    if samples.is_empty() || sample_rate == 0 {
        return SpeechChunkPlan::no_speech(samples.len());
    }

    let vad_config = speech_config_to_vad_region_config(speech_config);
    let regions = match detect_speech_regions_f32(samples, sample_rate, &vad_config) {
        Ok(regions) => regions
            .into_iter()
            .map(|region| SpeechRegion {
                start_sample: region.start_sample,
                end_sample: region.end_sample,
            })
            .collect::<Vec<_>>(),
        Err(_) => Vec::new(),
    };
    if regions.is_empty() {
        return SpeechChunkPlan::no_speech(samples.len());
    }

    let chunks = merge_regions_into_chunks(
        samples,
        sample_rate,
        long_form_config,
        &regions,
        model_max_chunk_secs,
    );
    let speech_samples = union_sample_len(
        &regions
            .iter()
            .map(|region| (region.start_sample, region.end_sample))
            .collect::<Vec<_>>(),
    );
    let included_samples = union_sample_len(
        &chunks
            .iter()
            .map(|chunk| (chunk.start_sample, chunk.end_sample))
            .collect::<Vec<_>>(),
    );

    SpeechChunkPlan {
        chunks,
        speech_regions: regions,
        speech_samples,
        included_samples,
        skipped_samples: samples.len().saturating_sub(included_samples),
        no_speech: false,
    }
}

/// Whole-input speech presence, scored by the same VAD as the speech planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechPresenceScan {
    /// No window held enough speech evidence. Input too short to score a
    /// single VAD frame is never reported as no-speech.
    pub no_speech: bool,
    /// The speech start threshold every frame was held to.
    pub threshold: f32,
    /// Energy above the noise floor every frame was held to, in dB.
    pub snr_margin_db: f32,
    /// Highest frame score seen before the scan stopped.
    pub peak_score: f32,
    /// Most speech-qualified time found in one evidence window.
    pub evidence_ms: f32,
    /// Speech-qualified time one window had to hold.
    pub required_ms: f32,
    /// Frames scored before the scan stopped.
    pub frames_scored: usize,
}

/// Decide whether a whole input holds any speech: some one-second window
/// must hold `min_evidence_secs` of frames that reach the start threshold
/// *and* rise `snr_margin_db` above the tracked noise floor. Earshot alone
/// scores steady noise as speech on scattered frames, so neither a single
/// frame nor a score without a level check is evidence. Short inputs need
/// proportionally less. The scan stops once the evidence is found, so speech
/// costs only the frames before it.
pub fn scan_speech_presence(
    samples: &[f32],
    sample_rate: u32,
    speech_config: &AsrSpeechChunkConfig,
) -> SpeechPresenceScan {
    let mut evidence = SpeechEvidence::new(SpeechEvidenceConfig {
        threshold: speech_config.start_threshold,
        snr_margin_db: speech_config.snr_margin_db,
        window_ms: DEFAULT_EVIDENCE_WINDOW_MS,
        min_evidence_ms: secs_to_ms(speech_config.min_evidence_secs),
    });
    let config = evidence.config().clone();
    let scan = |evidence: &SpeechEvidence, no_speech: bool| SpeechPresenceScan {
        no_speech,
        threshold: config.threshold,
        snr_margin_db: config.snr_margin_db,
        peak_score: evidence.peak_score(),
        evidence_ms: evidence.best_window_ms(),
        required_ms: if no_speech {
            evidence.required_ms()
        } else {
            config.min_evidence_ms as f32
        },
        frames_scored: evidence.frames(),
    };
    if sample_rate == 0 {
        return scan(&evidence, false);
    }
    let mut scorer = VadScorer::new();
    // About one second of input per push keeps the early exit cheap.
    for block in samples.chunks(sample_rate as usize) {
        let Ok(frames) = scorer.push_f32(block, sample_rate) else {
            return scan(&evidence, false);
        };
        for frame in frames {
            if evidence.push(&frame) {
                return scan(&evidence, false);
            }
        }
    }
    let no_speech = evidence.frames() > 0 && !evidence.is_sufficient();
    scan(&evidence, no_speech)
}

fn speech_config_to_vad_region_config(speech_config: &AsrSpeechChunkConfig) -> VadRegionConfig {
    VadRegionConfig {
        start_threshold: speech_config.start_threshold,
        end_threshold: speech_config.end_threshold,
        min_speech_ms: secs_to_ms(speech_config.min_speech_secs),
        min_silence_ms: secs_to_ms(speech_config.min_silence_secs),
        speech_pad_ms: secs_to_ms(speech_config.speech_pad_secs),
        merge_gap_ms: secs_to_ms(speech_config.merge_gap_secs),
        start_snr_margin_db: speech_config.snr_margin_db,
    }
}

fn merge_regions_into_chunks(
    samples: &[f32],
    sample_rate: u32,
    config: &AsrLongFormConfig,
    regions: &[SpeechRegion],
    model_max_chunk_secs: Option<f32>,
) -> Vec<AudioChunk> {
    let hard_max_samples = resolved_hard_max_samples(config, model_max_chunk_secs, sample_rate);
    let mut chunks = Vec::new();
    let mut current: Option<AudioChunk> = None;

    for region in regions {
        if region.len_samples() > hard_max_samples {
            if let Some(chunk) = current.take() {
                chunks.push(chunk);
            }
            let split_chunks = plan_audio_chunks(
                &samples[region.start_sample..region.end_sample],
                sample_rate,
                config,
                model_max_chunk_secs,
            );
            chunks.extend(split_chunks.into_iter().map(|chunk| AudioChunk {
                start_sample: region.start_sample + chunk.start_sample,
                end_sample: region.start_sample + chunk.end_sample,
            }));
            continue;
        }

        match current.as_mut() {
            Some(chunk)
                if region.end_sample.saturating_sub(chunk.start_sample) <= hard_max_samples =>
            {
                chunk.end_sample = region.end_sample;
            }
            Some(_) => {
                if let Some(chunk) = current.replace(AudioChunk {
                    start_sample: region.start_sample,
                    end_sample: region.end_sample,
                }) {
                    chunks.push(chunk);
                }
            }
            None => {
                current = Some(AudioChunk {
                    start_sample: region.start_sample,
                    end_sample: region.end_sample,
                });
            }
        }
    }

    if let Some(chunk) = current {
        chunks.push(chunk);
    }

    chunks
}

fn resolved_hard_max_samples(
    config: &AsrLongFormConfig,
    model_max_chunk_secs: Option<f32>,
    sample_rate: u32,
) -> usize {
    let mut hard_max_secs = config
        .hard_max_chunk_secs
        .max(config.min_chunk_secs.max(0.1));
    if let Some(limit) = model_max_chunk_secs.filter(|v| v.is_finite() && *v > 0.0) {
        hard_max_secs = hard_max_secs.min(limit.max(config.min_chunk_secs.max(0.1)));
    }
    secs_to_samples(hard_max_secs, sample_rate as f32).max(1)
}

fn union_sample_len(ranges: &[(usize, usize)]) -> usize {
    if ranges.is_empty() {
        return 0;
    }
    let mut ranges = ranges.to_vec();
    ranges.sort_unstable_by_key(|(start, end)| (*start, *end));
    let mut total = 0usize;
    let mut current = ranges[0];
    for range in ranges.into_iter().skip(1) {
        if range.0 <= current.1 {
            current.1 = current.1.max(range.1);
        } else {
            total = total.saturating_add(current.1.saturating_sub(current.0));
            current = range;
        }
    }
    total.saturating_add(current.1.saturating_sub(current.0))
}

/// Leading tokens of a chunk that boundary dedup may drop as fragments of a
/// word cut by the chunk start, ahead of the words it matches.
const MAX_FRAGMENT_TOKENS: usize = 2;
/// Upper bound on words spoken per second of overlapping audio.
const MAX_OVERLAP_WORDS_PER_SEC: f32 = 4.0;
/// Characters in an average word and the space after it.
const CHARS_PER_WORD: usize = 6;
/// Character-level dedup bounds, for scripts written without spaces.
const MIN_CHAR_OVERLAP: usize = 14;
const MAX_CHAR_OVERLAP: usize = 120;

/// Transcript assembler that merges chunk outputs and removes boundary duplicates.
#[derive(Debug, Clone)]
pub struct TranscriptAssembler {
    config: AsrLongFormConfig,
    merged: String,
    /// End of the last word committed by time.
    timed_end_secs: f32,
}

/// A transcribed word timed in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub text: String,
    pub start_secs: f32,
    pub end_secs: f32,
}

impl TimedWord {
    fn midpoint_secs(&self) -> f32 {
        (self.start_secs + self.end_secs) / 2.0
    }

    /// Whether the word belongs to the chunk between seams `after_secs` and
    /// `until_secs` (`None` = no neighbour on that side): its midpoint lies
    /// after the first and no later than the second.
    pub fn within_seams(&self, after_secs: Option<f32>, until_secs: Option<f32>) -> bool {
        let midpoint = self.midpoint_secs();
        !after_secs.is_some_and(|seam| midpoint <= seam)
            && !until_secs.is_some_and(|seam| midpoint > seam)
    }
}

/// Where two consecutive chunks hand over: the middle of the audio both
/// cover, or the earlier chunk's end when they do not overlap.
pub fn chunk_seam_secs(earlier_end_secs: f32, later_start_secs: f32) -> f32 {
    if later_start_secs < earlier_end_secs {
        (later_start_secs + earlier_end_secs) / 2.0
    } else {
        earlier_end_secs
    }
}

/// Overlap decision for a chunk whose text is still arriving, settled once the
/// text covers every word the overlap can hold. The commit reuses it, so text
/// streamed from the unfinished chunk is never contradicted by the merge.
#[derive(Debug, Clone, PartialEq)]
pub struct SettledChunkOverlap {
    overlap_secs: f32,
    /// Leading cleaned chunk text that duplicates the transcript tail.
    duplicate_prefix: String,
}

/// Bounds for matching a chunk start against the transcript tail.
#[derive(Debug, Clone, Copy)]
struct OverlapBounds {
    min_words: usize,
    max_words: usize,
    max_chars: usize,
}

impl OverlapBounds {
    /// A short overlap holds only a word or two, so the match bounds follow the
    /// overlap: at least one word per second of it (up to the configured
    /// minimum) and at most the words fast speech fits in it plus the word cut
    /// at its start. Skipped fragments count toward the maximum.
    fn new(config: &AsrLongFormConfig, overlap_secs: f32) -> Self {
        let min_words = (overlap_secs.round() as usize).clamp(1, config.min_word_overlap.max(1));
        let max_words = ((overlap_secs * MAX_OVERLAP_WORDS_PER_SEC).ceil() as usize + 1)
            .min(config.max_word_overlap.max(1));
        Self {
            min_words,
            max_words,
            max_chars: (max_words * CHARS_PER_WORD).min(MAX_CHAR_OVERLAP),
        }
    }
}

impl TranscriptAssembler {
    pub fn new(config: AsrLongFormConfig) -> Self {
        Self {
            config,
            merged: String::new(),
            timed_end_secs: 0.0,
        }
    }

    /// Commit a chunk by its word timings instead of its text. `words` are on
    /// the input timeline; a word belongs to this chunk when its midpoint lies
    /// after the seam with the previous chunk (`after_secs`) and no later than
    /// the seam with the next one (`until_secs`). Words of the shared audio
    /// are therefore kept exactly once, and a word a chunk boundary cut lands
    /// on the side holding most of it. Returns the kept words, each starting
    /// no earlier than the previous kept word ended.
    pub fn push_timed_chunk_words(
        &mut self,
        words: &[TimedWord],
        after_secs: Option<f32>,
        until_secs: Option<f32>,
    ) -> Vec<TimedWord> {
        let mut kept = Vec::new();
        for word in words {
            let text = word.text.trim();
            if text.is_empty() || !word.within_seams(after_secs, until_secs) {
                continue;
            }
            let end_secs = word.end_secs.max(word.start_secs);
            let start_secs = word.start_secs.max(self.timed_end_secs).min(end_secs);
            self.timed_end_secs = self.timed_end_secs.max(end_secs);
            append_with_spacing(&mut self.merged, text);
            kept.push(TimedWord {
                text: text.to_string(),
                start_secs,
                end_secs,
            });
        }
        kept
    }

    /// Push chunk text and get the incremental delta added to the merged output.
    /// Chunks are taken to overlap by the configured `overlap_secs`.
    pub fn push_chunk_text(&mut self, chunk_text: &str) -> String {
        self.push_chunk_text_with_overlap(chunk_text, self.config.overlap_secs)
    }

    /// Push the text of a chunk whose audio overlaps the previous chunk by
    /// `overlap_secs` and get the delta added to the merged output. Chunks
    /// that do not overlap are appended without dedup, so a phrase genuinely
    /// repeated across a pause survives.
    pub fn push_chunk_text_with_overlap(&mut self, chunk_text: &str, overlap_secs: f32) -> String {
        let cleaned = clean_chunk_text(chunk_text, self.config.max_repeated_chars);
        let delta_start = self.duplicate_prefix_len(&cleaned, overlap_secs);
        self.append_from(&cleaned, delta_start)
    }

    /// Settle the overlap for a chunk whose text so far is `partial_text`, or
    /// `None` while that text is too short to show which of its words repeat
    /// the transcript tail.
    pub fn settle_chunk_overlap(
        &self,
        partial_text: &str,
        overlap_secs: f32,
    ) -> Option<SettledChunkOverlap> {
        let cleaned = clean_chunk_text(partial_text, self.config.max_repeated_chars);
        if self.dedups(overlap_secs) {
            // Every word window the boundary search tries lies within the
            // first `max` tokens, so once a later token exists they are final.
            let bounds = OverlapBounds::new(&self.config, overlap_secs);
            if token_views(&cleaned).len() <= bounds.max_words {
                return None;
            }
        }
        let duplicate_len = self.duplicate_prefix_len(&cleaned, overlap_secs);
        Some(SettledChunkOverlap {
            overlap_secs,
            duplicate_prefix: cleaned[..duplicate_len].to_string(),
        })
    }

    /// Text the chunk adds to the transcript so far, spacing included. It only
    /// grows as `partial_text` grows, and the commit appends a continuation.
    pub fn provisional_chunk_text(
        &self,
        partial_text: &str,
        settled: &SettledChunkOverlap,
    ) -> String {
        let cleaned = clean_chunk_text(partial_text, self.config.max_repeated_chars);
        let Some(rest) = cleaned.strip_prefix(settled.duplicate_prefix.as_str()) else {
            return String::new();
        };
        let rest = rest.trim_start();
        if rest.is_empty() {
            return String::new();
        }
        format!("{}{rest}", spacing_between(&self.merged, rest))
    }

    /// Commit a chunk whose overlap was settled while it streamed. A final
    /// text that contradicts the settled prefix is merged from scratch.
    pub fn push_settled_chunk_text(
        &mut self,
        chunk_text: &str,
        settled: &SettledChunkOverlap,
    ) -> String {
        let cleaned = clean_chunk_text(chunk_text, self.config.max_repeated_chars);
        if !cleaned.starts_with(settled.duplicate_prefix.as_str()) {
            let delta_start = self.duplicate_prefix_len(&cleaned, settled.overlap_secs);
            return self.append_from(&cleaned, delta_start);
        }
        self.append_from(&cleaned, settled.duplicate_prefix.len())
    }

    pub fn text(&self) -> &str {
        &self.merged
    }

    pub fn finish(self) -> String {
        self.merged
    }

    fn dedups(&self, overlap_secs: f32) -> bool {
        !self.merged.is_empty() && overlap_secs > 0.0
    }

    /// Byte length of the leading part of `next` (cleaned chunk text) that
    /// repeats the end of the merged transcript. Only a bounded tail of the
    /// transcript is examined, so the cost does not grow with its length.
    fn duplicate_prefix_len(&self, next: &str, overlap_secs: f32) -> usize {
        if next.is_empty() || !self.dedups(overlap_secs) {
            return 0;
        }
        let bounds = OverlapBounds::new(&self.config, overlap_secs);
        let tail = tail_words(
            &self.merged,
            bounds.max_words.max(self.config.max_context_replay_words),
        );
        let prev = token_views(tail);
        let next_tokens = token_views(next);
        dedupe_overlap_word_boundary(
            &prev,
            &next_tokens,
            next.len(),
            bounds.min_words,
            bounds.max_words,
            MAX_FRAGMENT_TOKENS,
        )
        .or_else(|| {
            dedupe_overlap_char_boundary(
                tail_chars(&self.merged, bounds.max_chars),
                next,
                MIN_CHAR_OVERLAP,
                bounds.max_chars,
            )
        })
        .or_else(|| {
            // A decoder prompted with earlier text can replay a long stretch
            // of it at the chunk start; trim that as well.
            dedupe_overlap_word_boundary(
                &prev,
                &next_tokens,
                next.len(),
                self.config.min_context_replay_words,
                self.config.max_context_replay_words,
                0,
            )
        })
        .unwrap_or(0)
    }

    fn append_from(&mut self, cleaned: &str, delta_start: usize) -> String {
        if delta_start >= cleaned.len() {
            return String::new();
        }
        let delta_raw = cleaned[delta_start..].trim_start();
        if delta_raw.is_empty() {
            return String::new();
        }
        let before_len = self.merged.len();
        append_with_spacing(&mut self.merged, delta_raw);
        self.merged[before_len..].to_string()
    }
}

fn secs_to_samples(secs: f32, sample_rate: f32) -> usize {
    ((secs.max(0.0) * sample_rate).round() as usize).max(1)
}

fn secs_to_ms(secs: f32) -> u32 {
    ((secs.max(0.0) * 1000.0).round() as u32).max(1)
}

fn ms_to_samples(ms: u32, sample_rate: u32) -> usize {
    ((ms as u64 * sample_rate as u64) / 1000).max(1) as usize
}

fn choose_split_point(
    samples: &[f32],
    low: usize,
    high: usize,
    preferred: usize,
    frame_samples: usize,
    silence_quantile: f32,
    silence_scale: f32,
) -> Option<usize> {
    if high <= low || low >= samples.len() {
        return None;
    }

    let mut candidates = Vec::new();
    let mut pos = low;
    while pos <= high && pos < samples.len() {
        let e = local_energy(samples, pos, frame_samples);
        candidates.push((pos, e));
        match pos.checked_add(frame_samples) {
            Some(next) if next > pos => pos = next,
            _ => break,
        }
    }

    if candidates.is_empty() {
        return None;
    }

    let mut energies: Vec<f32> = candidates.iter().map(|(_, e)| *e).collect();
    let floor = percentile(&mut energies, silence_quantile.clamp(0.0, 1.0));
    let threshold = floor * silence_scale.max(1.0) + 1e-5;

    let low_energy = candidates
        .iter()
        .filter(|(_, e)| *e <= threshold)
        .min_by_key(|(p, _)| p.abs_diff(preferred))
        .map(|(p, _)| *p);

    if low_energy.is_some() {
        return low_energy;
    }

    candidates
        .iter()
        .min_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.abs_diff(preferred).cmp(&b.0.abs_diff(preferred)))
        })
        .map(|(p, _)| *p)
}

/// The middle of the longest pause in `low..=high` (ties go to the pause
/// nearest `preferred`), or `None` when no frame is quiet.
///
/// Pauses are a small share of conversational audio, so the quiet threshold
/// scales the window's 5th-percentile energy (the noise floor) and never
/// exceeds half the median, which is speech; a single louder frame inside a
/// pause (a click, a breath) does not split it.
fn choose_longest_pause_split_point(
    samples: &[f32],
    low: usize,
    high: usize,
    preferred: usize,
    frame_samples: usize,
    silence_scale: f32,
) -> Option<usize> {
    const NOISE_FLOOR_QUANTILE: f32 = 0.05;

    if high <= low || low >= samples.len() {
        return None;
    }

    let mut frames = Vec::new();
    let mut pos = low;
    while pos <= high && pos < samples.len() {
        frames.push((pos, local_energy(samples, pos, frame_samples)));
        match pos.checked_add(frame_samples) {
            Some(next) if next > pos => pos = next,
            _ => break,
        }
    }
    if frames.is_empty() {
        return None;
    }

    let mut energies: Vec<f32> = frames.iter().map(|(_, e)| *e).collect();
    let floor = percentile(&mut energies, NOISE_FLOOR_QUANTILE);
    let median = percentile(&mut energies, 0.5);
    let threshold = (floor * silence_scale.max(1.0) + 1e-5).min(median * 0.5 + 1e-5);

    let below: Vec<bool> = frames.iter().map(|(_, e)| *e <= threshold).collect();
    let quiet: Vec<bool> = below
        .iter()
        .enumerate()
        .map(|(index, &is_below)| {
            is_below
                || (index > 0
                    && below[index - 1]
                    && below.get(index + 1).copied().unwrap_or(false))
        })
        .collect();

    // (run length in frames, distance of its middle from `preferred`, middle)
    let mut best: Option<(usize, usize, usize)> = None;
    let mut run_start: Option<usize> = None;
    // A loud sentinel after the last frame closes a run that reaches the end.
    for (index, &is_quiet) in quiet.iter().chain(std::iter::once(&false)).enumerate() {
        match (is_quiet, run_start) {
            (true, None) => run_start = Some(index),
            (false, Some(first)) => {
                let length = index - first;
                let middle = frames[first + (length - 1) / 2].0;
                let distance = middle.abs_diff(preferred);
                let better = best.is_none_or(|(best_length, best_distance, _)| {
                    length > best_length || (length == best_length && distance < best_distance)
                });
                if better {
                    best = Some((length, distance, middle));
                }
                run_start = None;
            }
            _ => {}
        }
    }
    best.map(|(_, _, middle)| middle)
}

fn local_energy(samples: &[f32], center: usize, window: usize) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let half = window.max(1) / 2;
    let start = center
        .saturating_sub(half)
        .min(samples.len().saturating_sub(1));
    let end = (center + half).min(samples.len());
    if end <= start {
        return samples[start].abs();
    }

    let mut acc = 0.0f32;
    let mut n = 0usize;
    for sample in &samples[start..end] {
        acc += sample.abs();
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        acc / n as f32
    }
}

fn percentile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len().saturating_sub(1)) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    values[idx]
}

/// The cleanup every ASR transcript gets, whether it was decoded in one
/// window or merged from long-form chunks: whitespace collapsed within each
/// line, blank lines dropped, and runs of a repeated symbol cut to
/// `max_repeated_chars`. Letters and digits are never cut, so "1000000000"
/// and drawn-out words survive.
pub fn clean_transcript_text(text: &str, max_repeated_chars: usize) -> String {
    clean_chunk_text(text, max_repeated_chars)
}

fn clean_chunk_text(text: &str, max_repeated_chars: usize) -> String {
    let normalized = text
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if normalized.is_empty() {
        return String::new();
    }
    remove_repeated_chars(&normalized, max_repeated_chars.max(1))
}

/// Cut runs of a repeated non-alphanumeric character (a decoder stuck on
/// "........") to `max_run`.
fn remove_repeated_chars(text: &str, max_run: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last: Option<char> = None;
    let mut run = 0usize;

    for ch in text.chars() {
        if Some(ch) == last {
            run += 1;
        } else {
            last = Some(ch);
            run = 1;
        }

        if run <= max_run || ch.is_alphanumeric() {
            out.push(ch);
        }
    }

    out
}

fn append_with_spacing(dst: &mut String, delta: &str) {
    dst.push_str(spacing_between(dst, delta));
    dst.push_str(delta);
}

/// Separator to insert between merged text `prev` and an appended `delta`.
pub fn spacing_between(prev: &str, delta: &str) -> &'static str {
    let needs_space = match (prev.chars().next_back(), delta.chars().next()) {
        (Some(a), Some(b)) => {
            !a.is_whitespace()
                && !b.is_whitespace()
                && !is_closing_punctuation(b)
                && !is_opening_punctuation(a)
                && !written_without_spaces(a)
                && !written_without_spaces(b)
        }
        _ => false,
    };
    if needs_space {
        " "
    } else {
        ""
    }
}

/// Characters of scripts written without spaces between words (Chinese,
/// Japanese, Thai, Lao, Khmer, Myanmar), including CJK and full-width
/// punctuation. Chunks meeting at such a character join directly. Hangul is
/// excluded because Korean separates words with spaces.
fn written_without_spaces(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0E00..=0x0EFF // Thai, Lao
            | 0x1000..=0x109F // Myanmar
            | 0x1780..=0x17FF // Khmer
            | 0x3000..=0x30FF // CJK symbols and punctuation, Hiragana, Katakana
            | 0x31F0..=0x31FF // Katakana phonetic extensions
            | 0x3400..=0x4DBF // CJK extension A
            | 0x4E00..=0x9FFF // CJK unified ideographs
            | 0xF900..=0xFAFF // CJK compatibility ideographs
            | 0xFF00..=0xFFEF // half-width and full-width forms
            | 0x20000..=0x3134F // CJK extensions B and later
    )
}

fn is_opening_punctuation(ch: char) -> bool {
    matches!(ch, '(' | '[' | '{' | '"' | '\'' | '“' | '‘')
}

fn is_closing_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '.' | ',' | '!' | '?' | ':' | ';' | ')' | ']' | '}' | '"' | '\'' | '”' | '’'
    )
}

#[derive(Debug, Clone)]
struct TokenView {
    norm: String,
    start: usize,
}

/// Find where `next` stops repeating the end of `prev`: the last `k` tokens of
/// `prev` (`min_overlap..=max_overlap`, longest first) must match `k` tokens of
/// `next` after at most `max_skip` leading tokens that are a fragment of the
/// word the chunk start cut. Skipped and matched tokens together stay within
/// `max_overlap`. Returns the byte offset in `next` (of length `next_len`)
/// where new text starts.
fn dedupe_overlap_word_boundary(
    prev: &[TokenView],
    next: &[TokenView],
    next_len: usize,
    min_overlap: usize,
    max_overlap: usize,
    max_skip: usize,
) -> Option<usize> {
    if prev.is_empty() || next.is_empty() {
        return None;
    }

    let upper = cmp::min(max_overlap.max(1), cmp::min(prev.len(), next.len()));
    let lower = min_overlap.max(1);
    if upper < lower {
        return None;
    }

    for k in (lower..=upper).rev() {
        let tail = &prev[prev.len() - k..];
        let skip_limit = max_skip
            .min(next.len() - k)
            .min(max_overlap.saturating_sub(k));
        for skip in 0..=skip_limit {
            if skip > 0 && !is_cut_word_fragment(&prev[..prev.len() - k], &next[..skip]) {
                continue;
            }
            if !token_windows_match(tail, &next[skip..skip + k]) {
                continue;
            }
            let end = skip + k;
            return Some(if end == next.len() {
                next_len
            } else {
                next[end].start
            });
        }
    }

    None
}

/// A chunk that opens mid-word renders the cut word as a fragment of it, so the
/// skipped tokens must spell the end of the transcript words they overlap.
fn is_cut_word_fragment(prev_before_window: &[TokenView], skipped: &[TokenView]) -> bool {
    if prev_before_window.len() < skipped.len() {
        return false;
    }
    let counterpart = prev_before_window[prev_before_window.len() - skipped.len()..]
        .iter()
        .map(|token| token.norm.as_str())
        .collect::<String>();
    let fragment = skipped
        .iter()
        .map(|token| token.norm.as_str())
        .collect::<String>();
    counterpart.ends_with(&fragment)
}

/// Both ends must agree, which anchors the boundary; inside, 80% of the
/// tokens must agree, which tolerates one misheard word in five.
fn token_windows_match(left: &[TokenView], right: &[TokenView]) -> bool {
    let (Some(first), Some(last)) = (left.first(), left.last()) else {
        return false;
    };
    if first.norm != right[0].norm || last.norm != right[right.len() - 1].norm {
        return false;
    }
    let matched = left
        .iter()
        .zip(right)
        .filter(|(left, right)| left.norm == right.norm)
        .count();
    matched * 5 >= left.len() * 4
}

fn dedupe_overlap_char_boundary(
    prev: &str,
    next: &str,
    min_chars: usize,
    max_chars: usize,
) -> Option<usize> {
    let prev_chars: Vec<char> = prev.chars().collect();
    let next_chars: Vec<char> = next.chars().take(max_chars.max(1)).collect();
    if prev_chars.is_empty() || next_chars.is_empty() {
        return None;
    }

    let upper = cmp::min(
        max_chars.max(1),
        cmp::min(prev_chars.len(), next_chars.len()),
    );
    let lower = min_chars.max(1);
    if upper < lower {
        return None;
    }

    for k in (lower..=upper).rev() {
        let left = prev_chars[prev_chars.len() - k..]
            .iter()
            .flat_map(|c| c.to_lowercase())
            .collect::<String>();
        let right = next_chars[..k]
            .iter()
            .flat_map(|c| c.to_lowercase())
            .collect::<String>();
        if left == right {
            return Some(char_count_to_byte_index(next, k));
        }
    }

    None
}

/// The suffix of `text` holding its last `words` whitespace-separated words.
fn tail_words(text: &str, words: usize) -> &str {
    if words == 0 {
        return "";
    }
    let mut seen = 0usize;
    let mut in_word = false;
    for (idx, ch) in text.char_indices().rev() {
        if !ch.is_whitespace() {
            in_word = true;
            continue;
        }
        if in_word {
            seen += 1;
            if seen == words {
                return &text[idx + ch.len_utf8()..];
            }
            in_word = false;
        }
    }
    text
}

/// The suffix of `text` holding its last `chars` characters.
fn tail_chars(text: &str, chars: usize) -> &str {
    if chars == 0 {
        return "";
    }
    text.char_indices()
        .rev()
        .nth(chars - 1)
        .map_or(text, |(idx, _)| &text[idx..])
}

fn char_count_to_byte_index(text: &str, char_count: usize) -> usize {
    if char_count == 0 {
        return 0;
    }
    let mut count = 0usize;
    for (idx, ch) in text.char_indices() {
        count += 1;
        if count == char_count {
            return idx + ch.len_utf8();
        }
    }
    text.len()
}

fn token_views(text: &str) -> Vec<TokenView> {
    let mut views = Vec::new();
    let mut start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(st) = start.take() {
                let token = &text[st..idx];
                let norm = normalize_token(token);
                if !norm.is_empty() {
                    views.push(TokenView { norm, start: st });
                }
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }

    if let Some(st) = start {
        let token = &text[st..];
        let norm = normalize_token(token);
        if !norm.is_empty() {
            views.push(TokenView { norm, start: st });
        }
    }

    views
}

fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '\'' | '-'))
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_wav_mono_f32(name: &str) -> (Vec<f32>, u32) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(name);
        let mut reader = hound::WavReader::open(path).expect("test wav should open");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        let samples = reader
            .samples::<i16>()
            .map(|sample| sample.expect("test wav sample") as f32 / 32768.0)
            .collect::<Vec<_>>();
        (samples, spec.sample_rate)
    }

    #[test]
    fn short_audio_stays_single_chunk() {
        let cfg = AsrLongFormConfig::default();
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 5];
        let chunks = plan_audio_chunks(&samples, sr, &cfg, Some(30.0));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_sample, 0);
        assert_eq!(chunks[0].end_sample, samples.len());
    }

    #[test]
    fn long_audio_respects_model_limit() {
        let cfg = AsrLongFormConfig::default();
        let sr = 16_000u32;
        let samples = vec![0.01f32; (sr as usize) * 95];
        let chunks = plan_audio_chunks(&samples, sr, &cfg, Some(20.0));
        assert!(chunks.len() >= 4);

        let max_allowed = secs_to_samples(20.0, sr as f32);
        for chunk in &chunks {
            assert!(chunk.len_samples() <= max_allowed);
        }
    }

    /// A tail shorter than the minimum chunk is folded into the last chunk
    /// only when that keeps the chunk within the hard maximum.
    #[test]
    fn short_tail_never_stretches_a_chunk_past_the_hard_maximum() {
        let cfg = AsrLongFormConfig::default();
        let sr = 1_000u32;
        let max_allowed = secs_to_samples(28.5, sr as f32);
        for tenths in 286..=340 {
            let samples = vec![0.0f32; tenths * 100];
            let chunks = plan_audio_chunks(&samples, sr, &cfg, Some(28.5));
            assert!(chunks.len() > 1, "{tenths}");
            assert_eq!(chunks.last().unwrap().end_sample, samples.len());
            for chunk in &chunks {
                assert!(chunk.len_samples() <= max_allowed, "{tenths}: {chunk:?}");
            }
        }
    }

    #[test]
    fn speech_planner_returns_no_speech_for_silence() {
        let cfg = AsrLongFormConfig::default();
        let speech_cfg = AsrSpeechChunkConfig::default();
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 3];

        let plan = plan_speech_audio_chunks(&samples, sr, &cfg, &speech_cfg, Some(30.0));

        assert!(plan.no_speech);
        assert!(plan.chunks.is_empty());
        assert!(plan.speech_regions.is_empty());
        assert_eq!(plan.skipped_samples, samples.len());
    }

    #[test]
    fn speech_config_exposes_shared_vad_thresholds() {
        let speech_cfg = AsrSpeechChunkConfig {
            start_threshold: 0.61,
            end_threshold: 0.29,
            ..AsrSpeechChunkConfig::default()
        };

        let vad_cfg = speech_config_to_vad_region_config(&speech_cfg);

        assert_eq!(vad_cfg.start_threshold, 0.61);
        assert_eq!(vad_cfg.end_threshold, 0.29);
    }

    #[test]
    fn speech_planner_detects_fixture_speech_with_shared_vad() {
        let cfg = AsrLongFormConfig::default();
        let speech_cfg = AsrSpeechChunkConfig {
            min_speech_secs: 0.08,
            min_silence_secs: 0.2,
            ..AsrSpeechChunkConfig::default()
        };
        let (samples, sr) = load_wav_mono_f32("fox.wav");

        let plan = plan_speech_audio_chunks(&samples, sr, &cfg, &speech_cfg, Some(30.0));

        assert!(!plan.no_speech);
        assert!(!plan.speech_regions.is_empty());
        assert!(!plan.chunks.is_empty());
        assert!(plan.speech_samples > 0);
        assert!(plan.included_samples <= samples.len());
    }

    #[test]
    fn speech_presence_scan_reports_silence_as_no_speech() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 3];

        let scan = scan_speech_presence(&samples, sr, &AsrSpeechChunkConfig::default());

        assert!(scan.no_speech);
        assert_eq!(scan.threshold, DEFAULT_SPEECH_THRESHOLD);
        assert_eq!(scan.frames_scored, samples.len() / izwi_vad::VAD_FRAME_SAMPLES);
        assert!(scan.peak_score < scan.threshold);
    }

    #[test]
    fn speech_presence_scan_stops_once_speech_evidence_is_found() {
        let (samples, sr) = load_wav_mono_f32("fox.wav");

        let scan = scan_speech_presence(&samples, sr, &AsrSpeechChunkConfig::default());

        assert!(!scan.no_speech);
        assert!(scan.peak_score >= scan.threshold);
        assert!(scan.evidence_ms >= scan.required_ms);
        assert_eq!(scan.required_ms, 200.0);
        let total_frames = samples.len() * 16_000 / sr as usize / izwi_vad::VAD_FRAME_SAMPLES;
        assert!(scan.frames_scored < total_frames);
    }

    #[test]
    fn speech_presence_scan_reports_loud_steady_noise_as_no_speech() {
        use izwi_vad::test_signals::{fan_noise, mains_hum, pink_noise, white_noise};

        let sr = 16_000u32;
        let len = sr as usize * 10;
        for (name, samples) in [
            ("pink 0.05", pink_noise(len, 0.05, 4)),
            ("pink 0.3", pink_noise(len, 0.3, 4)),
            ("pink 0.8", pink_noise(len, 0.8, 4)),
            ("white 0.2", white_noise(len, 0.2, 4)),
            ("hum 0.3", mains_hum(len, 0.3)),
            ("fan 0.3", fan_noise(len, 0.3, 4)),
        ] {
            let scan = scan_speech_presence(&samples, sr, &AsrSpeechChunkConfig::default());
            assert!(scan.no_speech, "{name}: {scan:?}");
            assert_eq!(
                scan.frames_scored,
                len / izwi_vad::VAD_FRAME_SAMPLES,
                "{name}"
            );
        }
    }

    #[test]
    fn speech_planner_starts_no_region_on_steady_noise() {
        use izwi_vad::test_signals::fan_noise;

        let sr = 16_000u32;
        let samples = fan_noise(sr as usize * 40, 0.3, 4);
        let plan = plan_speech_audio_chunks(
            &samples,
            sr,
            &AsrLongFormConfig::default(),
            &AsrSpeechChunkConfig::default(),
            Some(30.0),
        );
        assert!(plan.no_speech, "{:?}", plan.speech_regions);
    }

    #[test]
    fn speech_presence_scan_never_gates_unscorable_input() {
        let config = AsrSpeechChunkConfig::default();

        assert!(!scan_speech_presence(&[], 16_000, &config).no_speech);
        assert!(!scan_speech_presence(&[0.0; 100], 16_000, &config).no_speech);
        assert!(!scan_speech_presence(&[0.0; 16_000], 0, &config).no_speech);
    }

    /// 60 s of "speech" at 1 kHz with a 0.2 s gap at 28 s (next to the
    /// preferred cut) and a 1.5 s pause at 22-23.5 s.
    fn speech_with_short_and_long_pause() -> Vec<f32> {
        let mut samples = vec![0.3f32; 60_000];
        samples[28_000..28_200].fill(0.0);
        samples[22_000..23_500].fill(0.0);
        samples
    }

    fn pause_test_config(prefer_longest_pause: bool) -> AsrLongFormConfig {
        AsrLongFormConfig {
            target_chunk_secs: 28.0,
            hard_max_chunk_secs: 35.0,
            silence_search_secs: 8.0,
            min_chunk_secs: 10.0,
            overlap_secs: 0.0,
            prefer_longest_pause,
            ..AsrLongFormConfig::default()
        }
    }

    #[test]
    fn longest_pause_mode_cuts_in_the_middle_of_the_longest_pause() {
        let samples = speech_with_short_and_long_pause();
        let chunks = plan_audio_chunks(&samples, 1_000, &pause_test_config(true), None);
        let cut = chunks[0].end_sample;
        assert!(
            (22_000..23_500).contains(&cut) && cut.abs_diff(22_750) <= 40,
            "cut at {cut}, expected the middle of the 22.0-23.5 s pause"
        );
    }

    #[test]
    fn default_mode_still_cuts_at_the_quiet_frame_nearest_the_target() {
        let samples = speech_with_short_and_long_pause();
        let chunks = plan_audio_chunks(&samples, 1_000, &pause_test_config(false), None);
        let cut = chunks[0].end_sample;
        assert!(
            (28_000..28_200).contains(&cut),
            "cut at {cut}, expected the short gap next to the 28 s target"
        );
    }

    #[test]
    fn longest_pause_mode_falls_back_without_quiet_frames() {
        let samples = vec![0.3f32; 60_000];
        let chunks = plan_audio_chunks(&samples, 1_000, &pause_test_config(true), None);
        assert!(chunks[0].end_sample <= 35_000 && chunks[0].end_sample >= 20_000);
    }

    #[test]
    fn region_chunker_combines_regions_under_model_limit() {
        let cfg = AsrLongFormConfig::default();
        let sr = 1_000u32;
        let samples = vec![0.0f32; 2_000];
        let regions = vec![
            SpeechRegion {
                start_sample: 200,
                end_sample: 800,
            },
            SpeechRegion {
                start_sample: 1_000,
                end_sample: 1_600,
            },
        ];

        let chunks = merge_regions_into_chunks(&samples, sr, &cfg, &regions, Some(30.0));

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_sample, 200);
        assert_eq!(chunks[0].end_sample, 1_600);
    }

    #[test]
    fn region_chunker_respects_model_limit_for_long_regions() {
        let cfg = AsrLongFormConfig {
            min_chunk_secs: 2.0,
            target_chunk_secs: 4.0,
            hard_max_chunk_secs: 5.0,
            overlap_secs: 0.5,
            ..AsrLongFormConfig::default()
        };
        let sr = 1_000u32;
        let samples = vec![0.2f32; 16_000];
        let regions = vec![SpeechRegion {
            start_sample: 0,
            end_sample: samples.len(),
        }];

        let chunks = merge_regions_into_chunks(&samples, sr, &cfg, &regions, Some(5.0));

        assert!(chunks.len() >= 3);
        let max_allowed = secs_to_samples(5.0, sr as f32);
        for chunk in &chunks {
            assert!(chunk.len_samples() <= max_allowed);
        }
    }

    #[test]
    fn next_chunk_starts_at_a_quiet_point_near_the_nominal_overlap() {
        let cfg = AsrLongFormConfig {
            target_chunk_secs: 10.0,
            hard_max_chunk_secs: 12.0,
            min_chunk_secs: 2.0,
            overlap_secs: 2.0,
            silence_search_secs: 1.0,
            ..AsrLongFormConfig::default()
        };
        let sr = 1_000u32;
        // Uniform energy splits the first chunk at its 10 s target, so the
        // nominal overlap start is 8 s. The only pause near it is 7.0-7.6 s.
        let mut samples = vec![0.5f32; 30_000];
        samples[7_000..7_600].fill(0.0);

        let chunks = plan_audio_chunks(&samples, sr, &cfg, None);

        assert_eq!(chunks[0].end_sample, 10_000);
        assert!(
            (7_000..7_600).contains(&chunks[1].start_sample),
            "next chunk should open in the pause, got {}",
            chunks[1].start_sample
        );
    }

    fn merged(chunks: &[(&str, f32)]) -> String {
        let mut assembler = TranscriptAssembler::new(AsrLongFormConfig::default());
        for (text, overlap_secs) in chunks {
            assembler.push_chunk_text_with_overlap(text, *overlap_secs);
        }
        assembler.finish()
    }

    fn timed(text: &str, start_secs: f32, end_secs: f32) -> TimedWord {
        TimedWord {
            text: text.to_string(),
            start_secs,
            end_secs,
        }
    }

    /// Chunk 0 covers 0-24 s and chunk 1 21-45 s, so they hand over at
    /// 22.5 s. Each word of the shared audio is kept once, from the chunk
    /// holding its midpoint: chunk 0's truncated "cut" yields to chunk 1's
    /// complete "cutoff", and chunk 1's copies of "two" and "three" drop.
    #[test]
    fn timed_chunks_keep_each_overlap_word_once() {
        let seam = chunk_seam_secs(24.0, 21.0);
        assert_eq!(seam, 22.5);
        let mut assembler = TranscriptAssembler::new(AsrLongFormConfig::default());
        let first = assembler.push_timed_chunk_words(
            &[
                timed("one", 0.0, 20.5),
                timed("two", 20.5, 22.0),
                timed("three", 22.0, 23.0),
                timed("cut", 23.0, 24.0),
            ],
            None,
            Some(seam),
        );
        assert_eq!(
            first,
            vec![
                timed("one", 0.0, 20.5),
                timed("two", 20.5, 22.0),
                timed("three", 22.0, 23.0),
            ]
        );
        let second = assembler.push_timed_chunk_words(
            &[
                timed("two", 21.0, 22.0),
                timed("three", 22.0, 23.0),
                timed("cutoff", 23.0, 24.4),
                timed("four", 24.4, 26.0),
            ],
            Some(seam),
            None,
        );
        assert_eq!(
            second,
            vec![timed("cutoff", 23.0, 24.4), timed("four", 24.4, 26.0)]
        );
        assert_eq!(assembler.text(), "one two three cutoff four");
    }

    /// Text dedup needs three matching words; by time a one-word overlap
    /// merges, and a genuine repeat in audio that is not shared survives.
    #[test]
    fn timed_chunks_merge_short_overlaps_and_keep_genuine_repeats() {
        let mut assembler = TranscriptAssembler::new(AsrLongFormConfig::default());
        let seam = chunk_seam_secs(8.0, 7.0);
        assembler.push_timed_chunk_words(
            &[timed("the", 5.0, 5.5), timed("fox.", 6.9, 7.6)],
            None,
            Some(seam),
        );
        assembler.push_timed_chunk_words(
            &[
                timed("fox.", 7.0, 7.6),
                timed("The", 7.8, 8.0),
                timed("fox.", 8.1, 8.6),
            ],
            Some(seam),
            None,
        );
        assert_eq!(assembler.text(), "the fox. The fox.");
    }

    /// Words a model times past its chunk's seam belong to the next chunk,
    /// and starts never run backwards across the seam.
    #[test]
    fn timed_chunks_drop_words_past_the_seam_and_keep_starts_monotonic() {
        let mut assembler = TranscriptAssembler::new(AsrLongFormConfig::default());
        let seam = chunk_seam_secs(10.0, 10.0);
        let first = assembler.push_timed_chunk_words(
            &[timed("inside", 0.0, 9.0), timed("beyond", 9.5, 11.0)],
            None,
            Some(seam),
        );
        assert_eq!(first, vec![timed("inside", 0.0, 9.0)]);
        let second = assembler.push_timed_chunk_words(
            &[timed("next", 8.8, 11.6), timed(" ", 11.6, 12.0)],
            Some(seam),
            None,
        );
        assert_eq!(second, vec![timed("next", 9.0, 11.6)]);
        assert_eq!(assembler.text(), "inside next");
    }

    #[test]
    fn short_overlaps_dedupe_one_and_two_word_boundary_repeats() {
        // Qwen3-ASR stream=true on a 116 s file, 1 s streaming overlaps.
        assert_eq!(
            merged(&[
                ("and the treatment is very expensive.", 0.0),
                ("Expensive, and there's no cure.", 1.0),
            ]),
            "and the treatment is very expensive. and there's no cure."
        );
        assert_eq!(
            merged(&[
                ("whether you live in the U.S. or not.", 0.0),
                ("Or not the reckoned cost is high.", 1.0),
            ]),
            "whether you live in the U.S. or not. the reckoned cost is high."
        );
        for overlap_secs in [0.45, 1.0] {
            assert_eq!(
                merged(&[
                    ("the quick brown fox", 0.0),
                    ("brown fox jumps", overlap_secs)
                ]),
                "the quick brown fox jumps",
                "overlap {overlap_secs}s"
            );
        }
    }

    #[test]
    fn chunk_opening_on_a_word_fragment_still_dedupes() {
        assert_eq!(
            merged(&[("the quick brown fox", 0.0), ("rown fox jumps", 1.0)]),
            "the quick brown fox jumps"
        );
    }

    #[test]
    fn boundary_match_tolerates_one_misheard_word_in_five() {
        assert_eq!(
            merged(&[
                ("count with me one two three four five", 0.0),
                ("one two tree four five six seven", 2.0),
            ]),
            "count with me one two three four five six seven"
        );
    }

    #[test]
    fn chunks_that_do_not_overlap_keep_a_genuine_repeat() {
        assert_eq!(
            merged(&[("I don't know.", 0.0), ("I don't know why.", 0.0)]),
            "I don't know. I don't know why."
        );
    }

    #[test]
    fn dedup_window_stays_within_the_words_an_overlap_can_hold() {
        // Half a second of overlap cannot repeat seven words, and "it was" is
        // not a fragment of the words before "over", so a chunk that opens
        // with a phrase the speaker said before keeps it.
        assert_eq!(
            merged(&[
                ("we said it was over and over and over", 0.0),
                ("it was over and over and over again", 0.45),
            ]),
            "we said it was over and over and over it was over and over and over again"
        );
    }

    #[test]
    fn settled_overlap_waits_for_the_partial_to_cover_the_overlap() {
        let mut assembler = TranscriptAssembler::new(AsrLongFormConfig::default());
        assembler.push_chunk_text("the quick brown fox jumps over");

        // One second of overlap can repeat up to five words.
        assert!(assembler
            .settle_chunk_overlap("fox jumps over the", 1.0)
            .is_none());
        let settled = assembler
            .settle_chunk_overlap("fox jumps over the lazy d", 1.0)
            .expect("six tokens cover the overlap");
        assert_eq!(
            assembler.provisional_chunk_text("fox jumps over the lazy d", &settled),
            " the lazy d"
        );
        assert_eq!(
            assembler.text(),
            "the quick brown fox jumps over",
            "settling must not advance the assembler"
        );

        let delta = assembler.push_settled_chunk_text("fox jumps over the lazy dog", &settled);

        assert_eq!(delta, " the lazy dog");
        assert_eq!(
            assembler.text(),
            "the quick brown fox jumps over the lazy dog"
        );
    }

    #[test]
    fn settled_overlap_is_recomputed_when_the_commit_contradicts_it() {
        let mut assembler = TranscriptAssembler::new(AsrLongFormConfig::default());
        assembler.push_chunk_text("alpha beta gamma");
        let settled = assembler
            .settle_chunk_overlap("gamma delta epsilon zeta eta theta", 1.0)
            .expect("settled");

        let delta = assembler.push_settled_chunk_text("beta gamma delta", &settled);

        assert_eq!(delta, " delta");
        assert_eq!(assembler.text(), "alpha beta gamma delta");
    }

    #[test]
    fn transcript_tails_are_bounded_by_words_and_chars() {
        assert_eq!(tail_words("one two  three four", 2), "three four");
        assert_eq!(tail_words("one two", 5), "one two");
        assert_eq!(tail_words("one two", 0), "");
        assert_eq!(tail_chars("héllo 世界", 4), "o 世界");
        assert_eq!(tail_chars("ab", 5), "ab");
    }

    #[test]
    fn assembler_removes_overlap_boundary() {
        let mut assembler = TranscriptAssembler::new(AsrLongFormConfig::default());
        let d1 = assembler.push_chunk_text("hello world this is a test");
        let d2 = assembler.push_chunk_text("this is a test of chunk stitching");

        assert_eq!(d1, "hello world this is a test");
        assert_eq!(d2, " of chunk stitching");
        assert_eq!(
            assembler.text(),
            "hello world this is a test of chunk stitching"
        );
    }

    #[test]
    fn assembler_trims_long_replayed_context_prefix() {
        let cfg = AsrLongFormConfig {
            max_word_overlap: 3,
            min_context_replay_words: 6,
            max_context_replay_words: 32,
            ..Default::default()
        };
        let mut assembler = TranscriptAssembler::new(cfg);

        let first = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
        let replayed = "eta theta iota kappa lambda mu new words arrive";
        let d1 = assembler.push_chunk_text(first);
        let d2 = assembler.push_chunk_text(replayed);

        assert_eq!(d1, first);
        assert_eq!(d2, " new words arrive");
        assert_eq!(
            assembler.text(),
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu new words arrive"
        );
    }

    #[test]
    fn assembler_does_not_trim_short_repeated_phrase_as_context_replay() {
        let cfg = AsrLongFormConfig {
            max_word_overlap: 2,
            min_context_replay_words: 6,
            ..Default::default()
        };
        let mut assembler = TranscriptAssembler::new(cfg);

        assembler.push_chunk_text("call and response happens here");
        let delta = assembler.push_chunk_text("happens here happens again");

        assert_eq!(delta, " happens here happens again");
        assert_eq!(
            assembler.text(),
            "call and response happens here happens here happens again"
        );
    }

    #[test]
    fn assembler_trims_excessive_symbol_repetition() {
        let cfg = AsrLongFormConfig {
            max_repeated_chars: 3,
            ..Default::default()
        };
        let mut assembler = TranscriptAssembler::new(cfg);
        let delta = assembler.push_chunk_text("wait.......... heyyyyyyyy there");
        assert_eq!(delta, "wait... heyyyyyyyy there");
    }

    #[test]
    fn cleanup_keeps_digit_and_letter_runs() {
        assert_eq!(
            clean_transcript_text("it cost 1000000000 dollars, sooooooooo much", 8),
            "it cost 1000000000 dollars, sooooooooo much"
        );
        assert_eq!(
            clean_transcript_text("and then!!!!!!!!!!!! ------------", 8),
            "and then!!!!!!!! --------"
        );
    }

    #[test]
    fn cleanup_collapses_spaces_but_keeps_line_breaks() {
        assert_eq!(
            clean_transcript_text("  Speaker 1:   hello \n\n\t Speaker 2:  hi  ", 8),
            "Speaker 1: hello\nSpeaker 2: hi"
        );
        assert_eq!(clean_transcript_text(" \n \t ", 8), "");
    }

    #[test]
    fn chunks_in_unspaced_scripts_join_without_a_space() {
        assert_eq!(
            merged(&[("今天天气很好。", 0.0), ("我们去公园吧。", 0.0)]),
            "今天天气很好。我们去公园吧。"
        );
        assert_eq!(
            merged(&[("我喜欢", 0.0), ("Python", 0.0), ("和", 0.0)]),
            "我喜欢Python和"
        );
        assert_eq!(
            merged(&[("東京に行きました", 0.0), ("「楽しかった」", 0.0)]),
            "東京に行きました「楽しかった」"
        );
        assert_eq!(merged(&[("สวัสดี", 0.0), ("ครับ", 0.0)]), "สวัสดีครับ");
        // Korean and Latin-script languages keep word spaces.
        assert_eq!(
            merged(&[("안녕하세요", 0.0), ("반갑습니다", 0.0)]),
            "안녕하세요 반갑습니다"
        );
        assert_eq!(merged(&[("hello", 0.0), ("world", 0.0)]), "hello world");
    }
}
