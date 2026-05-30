#![forbid(unsafe_code)]

use std::cmp;

use izwi_vad::{
    detect_speech_regions_f32, VadRegionConfig, DEFAULT_EXIT_THRESHOLD, DEFAULT_SPEECH_THRESHOLD,
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
    /// Retained for config compatibility. Shared VAD uses fixed 16 ms frames.
    pub analysis_frame_ms: u32,
    /// Deprecated compatibility field from the earlier adaptive-energy VAD; ignored by shared VAD.
    #[deprecated(note = "shared ASR VAD uses Earshot score thresholds; this field is ignored")]
    pub energy_floor_quantile: f32,
    /// Deprecated compatibility field from the earlier adaptive-energy VAD; ignored by shared VAD.
    #[deprecated(note = "shared ASR VAD uses Earshot score thresholds; this field is ignored")]
    pub onset_energy_scale: f32,
    /// Deprecated compatibility field from the earlier adaptive-energy VAD; ignored by shared VAD.
    #[deprecated(note = "shared ASR VAD uses Earshot score thresholds; this field is ignored")]
    pub offset_energy_scale: f32,
    /// Deprecated compatibility field from the earlier adaptive-energy VAD; ignored by shared VAD.
    #[deprecated(note = "shared ASR VAD uses Earshot score thresholds; this field is ignored")]
    pub min_energy: f32,
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
}

impl Default for AsrSpeechChunkConfig {
    #[allow(deprecated)]
    fn default() -> Self {
        Self {
            analysis_frame_ms: 20,
            energy_floor_quantile: 0.2,
            onset_energy_scale: 3.0,
            offset_energy_scale: 1.8,
            min_energy: 0.003,
            start_threshold: DEFAULT_SPEECH_THRESHOLD,
            end_threshold: DEFAULT_EXIT_THRESHOLD,
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

fn speech_config_to_vad_region_config(speech_config: &AsrSpeechChunkConfig) -> VadRegionConfig {
    VadRegionConfig {
        start_threshold: speech_config.start_threshold,
        end_threshold: speech_config.end_threshold,
        min_speech_ms: secs_to_ms(speech_config.min_speech_secs),
        min_silence_ms: secs_to_ms(speech_config.min_silence_secs),
        speech_pad_ms: secs_to_ms(speech_config.speech_pad_secs),
        merge_gap_ms: secs_to_ms(speech_config.merge_gap_secs),
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

        if delta_start == 0 {
            delta_start = dedupe_overlap_word_boundary(
                &self.merged,
                &cleaned,
                self.config.min_context_replay_words,
                self.config.max_context_replay_words,
            )
            .unwrap_or(0);
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
        let mut cfg = AsrLongFormConfig::default();
        cfg.max_word_overlap = 3;
        cfg.min_context_replay_words = 6;
        cfg.max_context_replay_words = 32;
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
        let mut cfg = AsrLongFormConfig::default();
        cfg.max_word_overlap = 2;
        cfg.min_context_replay_words = 6;
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
    fn assembler_trims_excessive_char_repetition() {
        let mut cfg = AsrLongFormConfig::default();
        cfg.max_repeated_chars = 3;
        let mut assembler = TranscriptAssembler::new(cfg);
        let delta = assembler.push_chunk_text("heyyyyyyyy there");
        assert_eq!(delta, "heyyy there");
    }
}
