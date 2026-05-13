#![forbid(unsafe_code)]

use std::cmp;

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
    /// Maximum consecutive repeat count per character in chunk cleanup.
    pub max_repeated_chars: usize,
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
            max_repeated_chars: 8,
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
    /// RMS frame length for speech activity analysis.
    pub analysis_frame_ms: u32,
    /// Noise floor quantile used to build adaptive onset/offset thresholds.
    pub energy_floor_quantile: f32,
    /// Scale over the adaptive floor required to enter speech.
    pub onset_energy_scale: f32,
    /// Scale over the adaptive floor required to remain in speech.
    pub offset_energy_scale: f32,
    /// Absolute minimum onset threshold for quiet/silent clips.
    pub min_energy: f32,
    /// Minimum continuous speech duration required to emit a region.
    pub min_speech_secs: f32,
    /// Minimum continuous silence duration required to close a region.
    pub min_silence_secs: f32,
    /// Padding applied to both sides of each detected speech region.
    pub speech_pad_secs: f32,
    /// Gaps at or below this length are merged into one speech region.
    pub merge_gap_secs: f32,
}

impl Default for AsrSpeechChunkConfig {
    fn default() -> Self {
        Self {
            analysis_frame_ms: 20,
            energy_floor_quantile: 0.2,
            onset_energy_scale: 3.0,
            offset_energy_scale: 1.8,
            min_energy: 0.003,
            min_speech_secs: 0.25,
            min_silence_secs: 0.45,
            speech_pad_secs: 0.2,
            merge_gap_secs: 0.35,
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

#[derive(Debug, Clone, Copy)]
struct FrameEnergy {
    start_sample: usize,
    end_sample: usize,
    rms: f32,
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
            if let Some(found) = choose_split_point(
                samples,
                search_low,
                search_high,
                preferred_end,
                frame_samples,
                config.silence_energy_quantile,
                config.silence_energy_scale,
            ) {
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
        if next_start <= start {
            next_start = split;
        }

        if total.saturating_sub(next_start) < min_chunk_samples {
            if let Some(last) = chunks.last_mut() {
                last.end_sample = total;
            }
            break;
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

    let frame_samples = ms_to_samples(speech_config.analysis_frame_ms.max(5), sample_rate).max(1);
    let frames = speech_energy_frames(samples, frame_samples);
    if frames.is_empty() {
        return SpeechChunkPlan::no_speech(samples.len());
    }

    let mut energies: Vec<f32> = frames.iter().map(|frame| frame.rms).collect();
    let floor = percentile(
        &mut energies,
        speech_config.energy_floor_quantile.clamp(0.0, 1.0),
    );
    let peak = frames
        .iter()
        .fold(0.0f32, |current, frame| current.max(frame.rms));
    let mut onset_threshold =
        (floor * speech_config.onset_energy_scale.max(1.0)).max(speech_config.min_energy);
    if peak > speech_config.min_energy {
        onset_threshold = onset_threshold.min(peak * 0.5);
    }
    let mut offset_threshold =
        (floor * speech_config.offset_energy_scale.max(1.0)).max(speech_config.min_energy * 0.5);
    if offset_threshold >= onset_threshold {
        offset_threshold = onset_threshold * 0.75;
    }

    let mut regions = detect_speech_regions(
        &frames,
        samples.len(),
        sample_rate,
        onset_threshold,
        offset_threshold,
        speech_config.min_speech_secs,
        speech_config.min_silence_secs,
    );
    if regions.is_empty() {
        return SpeechChunkPlan::no_speech(samples.len());
    }

    apply_region_padding(
        &mut regions,
        samples.len(),
        secs_to_samples(speech_config.speech_pad_secs, sample_rate as f32),
    );
    regions = merge_close_regions(
        &regions,
        secs_to_samples(speech_config.merge_gap_secs, sample_rate as f32),
    );
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

fn speech_energy_frames(samples: &[f32], frame_samples: usize) -> Vec<FrameEnergy> {
    let frame_samples = frame_samples.max(1);
    let mut frames = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + frame_samples).min(samples.len());
        frames.push(FrameEnergy {
            start_sample: start,
            end_sample: end,
            rms: local_rms(&samples[start..end]),
        });
        start = end;
    }
    frames
}

fn local_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum_squares / samples.len() as f32).sqrt()
}

fn detect_speech_regions(
    frames: &[FrameEnergy],
    total_samples: usize,
    sample_rate: u32,
    onset_threshold: f32,
    offset_threshold: f32,
    min_speech_secs: f32,
    min_silence_secs: f32,
) -> Vec<SpeechRegion> {
    let min_speech_samples = secs_to_samples(min_speech_secs.max(0.0), sample_rate as f32);
    let min_silence_samples = secs_to_samples(min_silence_secs.max(0.0), sample_rate as f32);
    let mut regions = Vec::new();
    let mut in_speech = false;
    let mut active_start: Option<usize> = None;
    let mut speech_start = 0usize;
    let mut silence_start: Option<usize> = None;

    for frame in frames {
        if !in_speech {
            if frame.rms >= onset_threshold {
                let start = *active_start.get_or_insert(frame.start_sample);
                if frame.end_sample.saturating_sub(start) >= min_speech_samples {
                    in_speech = true;
                    speech_start = start;
                    silence_start = None;
                }
            } else {
                active_start = None;
            }
            continue;
        }

        if frame.rms < offset_threshold {
            let start = *silence_start.get_or_insert(frame.start_sample);
            if frame.end_sample.saturating_sub(start) >= min_silence_samples {
                if start > speech_start {
                    regions.push(SpeechRegion {
                        start_sample: speech_start,
                        end_sample: start.min(total_samples),
                    });
                }
                in_speech = false;
                active_start = None;
                silence_start = None;
            }
        } else {
            silence_start = None;
        }
    }

    if in_speech {
        if total_samples > speech_start {
            regions.push(SpeechRegion {
                start_sample: speech_start,
                end_sample: total_samples,
            });
        }
    } else if let Some(start) = active_start {
        if total_samples.saturating_sub(start) >= min_speech_samples {
            regions.push(SpeechRegion {
                start_sample: start,
                end_sample: total_samples,
            });
        }
    }

    regions
        .into_iter()
        .filter(|region| region.len_samples() >= min_speech_samples)
        .collect()
}

fn apply_region_padding(regions: &mut [SpeechRegion], total_samples: usize, pad_samples: usize) {
    for region in regions {
        region.start_sample = region.start_sample.saturating_sub(pad_samples);
        region.end_sample = region
            .end_sample
            .saturating_add(pad_samples)
            .min(total_samples);
    }
}

fn merge_close_regions(regions: &[SpeechRegion], merge_gap_samples: usize) -> Vec<SpeechRegion> {
    let mut merged: Vec<SpeechRegion> = Vec::new();
    for region in regions.iter().copied() {
        if region.end_sample <= region.start_sample {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            let gap = region.start_sample.saturating_sub(last.end_sample);
            if gap <= merge_gap_samples {
                last.end_sample = last.end_sample.max(region.end_sample);
                continue;
            }
        }
        merged.push(region);
    }
    merged
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

/// Transcript assembler that merges chunk outputs and removes boundary duplicates.
#[derive(Debug, Clone)]
pub struct TranscriptAssembler {
    config: AsrLongFormConfig,
    merged: String,
}

impl TranscriptAssembler {
    pub fn new(config: AsrLongFormConfig) -> Self {
        Self {
            config,
            merged: String::new(),
        }
    }

    /// Push chunk text and get the incremental delta added to the merged output.
    pub fn push_chunk_text(&mut self, chunk_text: &str) -> String {
        let cleaned = clean_chunk_text(chunk_text, self.config.max_repeated_chars);
        if cleaned.is_empty() {
            return String::new();
        }

        if self.merged.is_empty() {
            self.merged = cleaned.clone();
            return cleaned;
        }

        let mut delta_start = dedupe_overlap_word_boundary(
            &self.merged,
            &cleaned,
            self.config.min_word_overlap,
            self.config.max_word_overlap,
        )
        .unwrap_or(0);

        if delta_start == 0 {
            delta_start =
                dedupe_overlap_char_boundary(&self.merged, &cleaned, 14, 120).unwrap_or(0);
        }

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

    pub fn text(&self) -> &str {
        &self.merged
    }

    pub fn finish(self) -> String {
        self.merged
    }
}

fn secs_to_samples(secs: f32, sample_rate: f32) -> usize {
    ((secs.max(0.0) * sample_rate).round() as usize).max(1)
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

fn clean_chunk_text(text: &str, max_repeated_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    remove_repeated_chars(trimmed, max_repeated_chars.max(1))
}

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

        if run <= max_run {
            out.push(ch);
        }
    }

    out
}

fn append_with_spacing(dst: &mut String, delta: &str) {
    if dst.is_empty() {
        dst.push_str(delta);
        return;
    }

    let prev_last = dst.chars().next_back();
    let next_first = delta.chars().next();

    let needs_space = match (prev_last, next_first) {
        (Some(a), Some(b)) => {
            !a.is_whitespace()
                && !b.is_whitespace()
                && !is_closing_punctuation(b)
                && !is_opening_punctuation(a)
        }
        _ => false,
    };

    if needs_space {
        dst.push(' ');
    }
    dst.push_str(delta);
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

fn dedupe_overlap_word_boundary(
    prev: &str,
    next: &str,
    min_overlap: usize,
    max_overlap: usize,
) -> Option<usize> {
    let a = token_views(prev);
    let b = token_views(next);
    if a.is_empty() || b.is_empty() {
        return None;
    }

    let upper = cmp::min(max_overlap.max(1), cmp::min(a.len(), b.len()));
    let lower = min_overlap.max(1);
    if upper < lower {
        return None;
    }

    for k in (lower..=upper).rev() {
        let a_slice = &a[a.len() - k..];
        let b_slice = &b[..k];

        if a_slice
            .iter()
            .zip(b_slice.iter())
            .all(|(left, right)| !left.norm.is_empty() && left.norm == right.norm)
        {
            return if k == b.len() {
                Some(next.len())
            } else {
                Some(b[k].start)
            };
        }
    }

    None
}

fn dedupe_overlap_char_boundary(
    prev: &str,
    next: &str,
    min_chars: usize,
    max_chars: usize,
) -> Option<usize> {
    let prev_chars: Vec<char> = prev.chars().collect();
    let next_chars: Vec<char> = next.chars().collect();
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
    fn speech_planner_finds_separate_speech_islands() {
        let cfg = AsrLongFormConfig::default();
        let speech_cfg = AsrSpeechChunkConfig {
            speech_pad_secs: 0.0,
            merge_gap_secs: 0.2,
            min_speech_secs: 0.1,
            min_silence_secs: 0.15,
            ..AsrSpeechChunkConfig::default()
        };
        let sr = 1_000u32;
        let mut samples = vec![0.0f32; 4_000];
        samples[500..1_200].fill(0.2);
        samples[2_200..3_000].fill(0.25);

        let plan = plan_speech_audio_chunks(&samples, sr, &cfg, &speech_cfg, Some(30.0));

        assert!(!plan.no_speech);
        assert_eq!(plan.speech_regions.len(), 2);
        assert_eq!(plan.chunks.len(), 1);
        assert!(plan.speech_regions[0].start_sample <= 500);
        assert!(plan.speech_regions[0].end_sample >= 1_200);
        assert!(plan.speech_regions[1].start_sample <= 2_200);
        assert!(plan.speech_regions[1].end_sample >= 3_000);
        assert!(plan.skipped_samples > 0);
    }

    #[test]
    fn speech_planner_merges_short_gaps() {
        let cfg = AsrLongFormConfig::default();
        let speech_cfg = AsrSpeechChunkConfig {
            speech_pad_secs: 0.0,
            merge_gap_secs: 0.35,
            min_speech_secs: 0.1,
            min_silence_secs: 0.1,
            ..AsrSpeechChunkConfig::default()
        };
        let sr = 1_000u32;
        let mut samples = vec![0.0f32; 2_000];
        samples[200..800].fill(0.25);
        samples[1_000..1_600].fill(0.25);

        let plan = plan_speech_audio_chunks(&samples, sr, &cfg, &speech_cfg, Some(30.0));

        assert_eq!(plan.speech_regions.len(), 1);
        assert_eq!(plan.chunks.len(), 1);
        assert!(plan.chunks[0].start_sample <= 200);
        assert!(plan.chunks[0].end_sample >= 1_600);
    }

    #[test]
    fn speech_planner_respects_model_limit_for_long_regions() {
        let cfg = AsrLongFormConfig {
            min_chunk_secs: 2.0,
            target_chunk_secs: 4.0,
            hard_max_chunk_secs: 5.0,
            overlap_secs: 0.5,
            ..AsrLongFormConfig::default()
        };
        let speech_cfg = AsrSpeechChunkConfig {
            speech_pad_secs: 0.0,
            min_speech_secs: 0.1,
            min_silence_secs: 0.2,
            ..AsrSpeechChunkConfig::default()
        };
        let sr = 1_000u32;
        let samples = vec![0.2f32; 16_000];

        let plan = plan_speech_audio_chunks(&samples, sr, &cfg, &speech_cfg, Some(5.0));

        assert!(plan.chunks.len() >= 3);
        let max_allowed = secs_to_samples(5.0, sr as f32);
        for chunk in &plan.chunks {
            assert!(chunk.len_samples() <= max_allowed);
        }
    }

    #[test]
    fn speech_planner_uses_input_sample_rate_for_timing() {
        let cfg = AsrLongFormConfig::default();
        let speech_cfg = AsrSpeechChunkConfig {
            speech_pad_secs: 0.0,
            min_speech_secs: 0.1,
            min_silence_secs: 0.1,
            ..AsrSpeechChunkConfig::default()
        };
        let sr = 8_000u32;
        let mut samples = vec![0.0f32; 24_000];
        samples[8_000..12_000].fill(0.2);

        let plan = plan_speech_audio_chunks(&samples, sr, &cfg, &speech_cfg, Some(30.0));

        assert_eq!(plan.speech_regions.len(), 1);
        assert!(plan.speech_regions[0].start_sample <= 8_000);
        assert!(plan.speech_regions[0].end_sample >= 12_000);
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
    fn assembler_trims_excessive_char_repetition() {
        let mut cfg = AsrLongFormConfig::default();
        cfg.max_repeated_chars = 3;
        let mut assembler = TranscriptAssembler::new(cfg);
        let delta = assembler.push_chunk_text("heyyyyyyyy there");
        assert_eq!(delta, "heyyy there");
    }
}
