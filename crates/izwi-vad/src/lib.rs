#![forbid(unsafe_code)]

use std::fmt;

pub const VAD_SAMPLE_RATE: u32 = 16_000;
pub const VAD_FRAME_SAMPLES: usize = 256;
pub const VAD_FRAME_MS: f32 = 16.0;
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.5;
pub const DEFAULT_EXIT_THRESHOLD: f32 = 0.35;
pub const DEFAULT_MIN_SPEECH_MS: u32 = 300;
pub const DEFAULT_SILENCE_MS: u32 = 900;
pub const DEFAULT_MAX_UTTERANCE_MS: u32 = 20_000;
pub const DEFAULT_PRE_ROLL_MS: u32 = 160;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadFrame {
    pub index: usize,
    pub score: f32,
    pub start_vad_sample: usize,
    pub end_vad_sample: usize,
    pub start_ms: f32,
    pub end_ms: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadError {
    InvalidSampleRate(u32),
}

impl fmt::Display for VadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleRate(sample_rate) => {
                write!(f, "invalid VAD input sample rate {sample_rate}")
            }
        }
    }
}

impl std::error::Error for VadError {}

pub struct VadScorer {
    detector: Box<earshot::Detector>,
    resampler: Option<StreamingLinearResampler>,
    vad_buffer: Vec<f32>,
    next_frame_index: usize,
    processed_vad_samples: usize,
}

impl Default for VadScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl VadScorer {
    pub fn new() -> Self {
        Self {
            detector: earshot::Detector::default_boxed(),
            resampler: None,
            vad_buffer: Vec::with_capacity(VAD_FRAME_SAMPLES * 2),
            next_frame_index: 0,
            processed_vad_samples: 0,
        }
    }

    pub fn reset(&mut self) {
        self.detector.reset();
        self.resampler = None;
        self.vad_buffer.clear();
        self.next_frame_index = 0;
        self.processed_vad_samples = 0;
    }

    pub fn push_i16(
        &mut self,
        samples: &[i16],
        sample_rate: u32,
    ) -> Result<Vec<VadFrame>, VadError> {
        let samples = pcm16_to_f32(samples);
        self.push_f32(&samples, sample_rate)
    }

    pub fn push_f32(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<VadFrame>, VadError> {
        validate_sample_rate(sample_rate)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        if sample_rate == VAD_SAMPLE_RATE {
            self.resampler = None;
            self.vad_buffer
                .extend(samples.iter().map(|s| sanitize_sample(*s)));
        } else {
            let resampler = self
                .resampler
                .get_or_insert_with(|| StreamingLinearResampler::new(sample_rate, VAD_SAMPLE_RATE));
            if resampler.src_rate != sample_rate {
                *resampler = StreamingLinearResampler::new(sample_rate, VAD_SAMPLE_RATE);
                self.detector.reset();
                self.vad_buffer.clear();
                self.next_frame_index = 0;
                self.processed_vad_samples = 0;
            }
            self.vad_buffer
                .extend(resampler.push(samples).into_iter().map(sanitize_sample));
        }

        Ok(self.drain_ready_frames())
    }

    pub fn score_complete_f32(
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<VadFrame>, VadError> {
        let mut scorer = Self::new();
        scorer.push_f32(samples, sample_rate)
    }

    pub fn score_complete_i16(
        samples: &[i16],
        sample_rate: u32,
    ) -> Result<Vec<VadFrame>, VadError> {
        let mut scorer = Self::new();
        scorer.push_i16(samples, sample_rate)
    }

    fn drain_ready_frames(&mut self) -> Vec<VadFrame> {
        let frame_count = self.vad_buffer.len() / VAD_FRAME_SAMPLES;
        if frame_count == 0 {
            return Vec::new();
        }

        let mut frames = Vec::with_capacity(frame_count);
        for frame in self.vad_buffer.chunks_exact(VAD_FRAME_SAMPLES) {
            let score = self.detector.predict_f32(frame).clamp(0.0, 1.0);
            let start_vad_sample = self.processed_vad_samples;
            let end_vad_sample = start_vad_sample + VAD_FRAME_SAMPLES;
            frames.push(VadFrame {
                index: self.next_frame_index,
                score,
                start_vad_sample,
                end_vad_sample,
                start_ms: vad_sample_to_ms(start_vad_sample),
                end_ms: vad_sample_to_ms(end_vad_sample),
            });
            self.next_frame_index += 1;
            self.processed_vad_samples = end_vad_sample;
        }

        let consumed = frame_count * VAD_FRAME_SAMPLES;
        self.vad_buffer.drain(0..consumed);
        frames
    }
}

#[derive(Debug, Clone)]
pub struct EndpointConfig {
    pub start_threshold: f32,
    pub end_threshold: f32,
    pub min_speech_ms: u32,
    pub silence_ms: u32,
    pub max_utterance_ms: u32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            start_threshold: DEFAULT_SPEECH_THRESHOLD,
            end_threshold: DEFAULT_EXIT_THRESHOLD,
            min_speech_ms: DEFAULT_MIN_SPEECH_MS,
            silence_ms: DEFAULT_SILENCE_MS,
            max_utterance_ms: DEFAULT_MAX_UTTERANCE_MS,
        }
    }
}

impl EndpointConfig {
    pub fn sanitized(mut self) -> Self {
        self.start_threshold = sanitize_score_threshold(self.start_threshold);
        self.end_threshold = sanitize_score_threshold(self.end_threshold).min(self.start_threshold);
        self.min_speech_ms = self.min_speech_ms.max(1);
        self.silence_ms = self.silence_ms.max(1);
        self.max_utterance_ms = self.max_utterance_ms.max(self.min_speech_ms);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointEndReason {
    Silence,
    MaxDuration,
    StreamStopped,
}

impl EndpointEndReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Silence => "silence",
            Self::MaxDuration => "max_duration",
            Self::StreamStopped => "stream_stopped",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointEvent {
    SpeechStart,
    SpeechEnd(EndpointEndReason),
    NoiseRejected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EndpointDecision {
    pub is_speech: bool,
    pub events: Vec<EndpointEvent>,
    pub speech_ms: f32,
    pub silence_ms: f32,
    pub total_ms: f32,
}

pub struct EndpointDetector {
    config: EndpointConfig,
    active: bool,
    speech_ms: f32,
    silence_ms: f32,
    total_ms: f32,
}

impl EndpointDetector {
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            config: config.sanitized(),
            active: false,
            speech_ms: 0.0,
            silence_ms: 0.0,
            total_ms: 0.0,
        }
    }

    pub fn config(&self) -> &EndpointConfig {
        &self.config
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn reset(&mut self) {
        self.active = false;
        self.speech_ms = 0.0;
        self.silence_ms = 0.0;
        self.total_ms = 0.0;
    }

    pub fn process_score(&mut self, score: f32, frame_ms: f32) -> EndpointDecision {
        let frame_ms = frame_ms.max(0.0);
        let threshold = if self.active {
            self.config.end_threshold
        } else {
            self.config.start_threshold
        };
        let is_speech = sanitize_score(score) >= threshold;
        let mut events = Vec::new();

        if !self.active {
            if is_speech {
                self.active = true;
                self.speech_ms = frame_ms;
                self.silence_ms = 0.0;
                self.total_ms = frame_ms;
                events.push(EndpointEvent::SpeechStart);
            }
            return self.decision(is_speech, events);
        }

        self.total_ms += frame_ms;
        if is_speech {
            self.speech_ms += frame_ms;
            self.silence_ms = 0.0;
        } else {
            self.silence_ms += frame_ms;
        }

        if self.total_ms >= self.config.max_utterance_ms as f32 {
            events.push(EndpointEvent::SpeechEnd(EndpointEndReason::MaxDuration));
            self.reset();
        } else if self.speech_ms < self.config.min_speech_ms as f32
            && self.silence_ms >= self.config.silence_ms as f32
        {
            events.push(EndpointEvent::NoiseRejected);
            self.reset();
        } else if self.speech_ms >= self.config.min_speech_ms as f32
            && self.silence_ms >= self.config.silence_ms as f32
        {
            events.push(EndpointEvent::SpeechEnd(EndpointEndReason::Silence));
            self.reset();
        }

        self.decision(is_speech, events)
    }

    pub fn finish(&mut self) -> Option<EndpointEvent> {
        if !self.active {
            return None;
        }
        let event = if self.speech_ms >= self.config.min_speech_ms as f32 {
            EndpointEvent::SpeechEnd(EndpointEndReason::StreamStopped)
        } else {
            EndpointEvent::NoiseRejected
        };
        self.reset();
        Some(event)
    }

    fn decision(&self, is_speech: bool, events: Vec<EndpointEvent>) -> EndpointDecision {
        EndpointDecision {
            is_speech,
            events,
            speech_ms: self.speech_ms,
            silence_ms: self.silence_ms,
            total_ms: self.total_ms,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechRegion {
    pub start_sample: usize,
    pub end_sample: usize,
}

impl SpeechRegion {
    pub fn len_samples(self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }
}

#[derive(Debug, Clone)]
pub struct VadRegionConfig {
    pub start_threshold: f32,
    pub end_threshold: f32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
    pub merge_gap_ms: u32,
}

impl Default for VadRegionConfig {
    fn default() -> Self {
        Self {
            start_threshold: DEFAULT_SPEECH_THRESHOLD,
            end_threshold: DEFAULT_EXIT_THRESHOLD,
            min_speech_ms: DEFAULT_MIN_SPEECH_MS,
            min_silence_ms: 450,
            speech_pad_ms: 200,
            merge_gap_ms: 350,
        }
    }
}

impl VadRegionConfig {
    pub fn sanitized(mut self) -> Self {
        self.start_threshold = sanitize_score_threshold(self.start_threshold);
        self.end_threshold = sanitize_score_threshold(self.end_threshold).min(self.start_threshold);
        self.min_speech_ms = self.min_speech_ms.max(1);
        self.min_silence_ms = self.min_silence_ms.max(1);
        self
    }
}

pub fn detect_speech_regions_f32(
    samples: &[f32],
    sample_rate: u32,
    config: &VadRegionConfig,
) -> Result<Vec<SpeechRegion>, VadError> {
    validate_sample_rate(sample_rate)?;
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let frames = VadScorer::score_complete_f32(samples, sample_rate)?;
    Ok(detect_speech_regions_from_frames(
        &frames,
        samples.len(),
        sample_rate,
        config,
    ))
}

pub fn detect_speech_regions_i16(
    samples: &[i16],
    sample_rate: u32,
    config: &VadRegionConfig,
) -> Result<Vec<SpeechRegion>, VadError> {
    let samples = pcm16_to_f32(samples);
    detect_speech_regions_f32(&samples, sample_rate, config)
}

pub fn detect_speech_regions_from_frames(
    frames: &[VadFrame],
    total_samples: usize,
    sample_rate: u32,
    config: &VadRegionConfig,
) -> Vec<SpeechRegion> {
    if frames.is_empty() || total_samples == 0 || sample_rate == 0 {
        return Vec::new();
    }

    let config = config.clone().sanitized();
    let total_ms = samples_to_ms(total_samples, sample_rate);
    let min_speech_samples = ms_to_samples(config.min_speech_ms, sample_rate);
    let min_silence_ms = config.min_silence_ms as f32;
    let mut regions_ms: Vec<(f32, f32)> = Vec::new();
    let mut in_speech = false;
    let mut speech_start_ms = 0.0f32;
    let mut silence_start_ms: Option<f32> = None;

    for frame in frames {
        if !in_speech {
            if frame.score >= config.start_threshold {
                in_speech = true;
                speech_start_ms = frame.start_ms;
                silence_start_ms = None;
            }
            continue;
        }

        if frame.score < config.end_threshold {
            let start = *silence_start_ms.get_or_insert(frame.start_ms);
            if frame.end_ms - start >= min_silence_ms {
                if start > speech_start_ms {
                    regions_ms.push((speech_start_ms, start.min(total_ms)));
                }
                in_speech = false;
                silence_start_ms = None;
            }
        } else {
            silence_start_ms = None;
        }
    }

    if in_speech && total_ms > speech_start_ms {
        regions_ms.push((speech_start_ms, total_ms));
    }

    let mut regions: Vec<SpeechRegion> = regions_ms
        .into_iter()
        .filter_map(|(start_ms, end_ms)| {
            let start_sample = ms_float_to_sample(start_ms, sample_rate).min(total_samples);
            let end_sample = ms_float_to_sample(end_ms, sample_rate).min(total_samples);
            (end_sample > start_sample).then_some(SpeechRegion {
                start_sample,
                end_sample,
            })
        })
        .filter(|region| region.len_samples() >= min_speech_samples)
        .collect();

    let pad_samples = ms_to_samples(config.speech_pad_ms, sample_rate);
    apply_padding(&mut regions, total_samples, pad_samples);
    merge_regions(&regions, ms_to_samples(config.merge_gap_ms, sample_rate))
}

pub fn speech_mask_for_frames_f32(
    samples: &[f32],
    sample_rate: u32,
    frame_count: usize,
    frame_stride_samples: usize,
    config: &VadRegionConfig,
) -> Result<Vec<bool>, VadError> {
    if frame_count == 0 || frame_stride_samples == 0 {
        return Ok(Vec::new());
    }

    let regions = detect_speech_regions_f32(samples, sample_rate, config)?;
    Ok(mask_from_regions(
        &regions,
        frame_count,
        frame_stride_samples,
        samples.len(),
    ))
}

pub fn mask_from_regions(
    regions: &[SpeechRegion],
    frame_count: usize,
    frame_stride_samples: usize,
    total_samples: usize,
) -> Vec<bool> {
    if frame_count == 0 || frame_stride_samples == 0 {
        return Vec::new();
    }

    let mut mask = vec![false; frame_count];
    for (frame_idx, active) in mask.iter_mut().enumerate() {
        let start = frame_idx * frame_stride_samples;
        if start >= total_samples {
            break;
        }
        let end = ((frame_idx + 1) * frame_stride_samples).min(total_samples);
        *active = regions
            .iter()
            .any(|region| region.start_sample < end && region.end_sample > start);
    }
    mask
}

pub fn pcm16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| sanitize_sample(*sample as f32 / 32768.0))
        .collect()
}

pub fn resample_linear_f32(
    samples: &[f32],
    source_sample_rate: u32,
    target_sample_rate: u32,
) -> Result<Vec<f32>, VadError> {
    validate_sample_rate(source_sample_rate)?;
    validate_sample_rate(target_sample_rate)?;
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if source_sample_rate == target_sample_rate {
        return Ok(samples.iter().map(|s| sanitize_sample(*s)).collect());
    }
    if samples.len() == 1 {
        return Ok(vec![sanitize_sample(samples[0])]);
    }

    let target_len = ((samples.len() as f64) * target_sample_rate as f64
        / source_sample_rate as f64)
        .ceil()
        .max(1.0) as usize;
    let step = source_sample_rate as f64 / target_sample_rate as f64;
    let mut out = Vec::with_capacity(target_len);
    for idx in 0..target_len {
        let src_pos = idx as f64 * step;
        let src_idx = src_pos.floor() as usize;
        if src_idx + 1 >= samples.len() {
            out.push(sanitize_sample(*samples.last().unwrap_or(&0.0)));
            continue;
        }
        let frac = (src_pos - src_idx as f64) as f32;
        out.push(sanitize_sample(
            samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac,
        ));
    }
    Ok(out)
}

pub fn sanitize_score_threshold(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.01, 0.99)
    } else {
        DEFAULT_SPEECH_THRESHOLD
    }
}

fn sanitize_score(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_sample(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn validate_sample_rate(sample_rate: u32) -> Result<(), VadError> {
    if sample_rate == 0 {
        Err(VadError::InvalidSampleRate(sample_rate))
    } else {
        Ok(())
    }
}

fn vad_sample_to_ms(sample: usize) -> f32 {
    samples_to_ms(sample, VAD_SAMPLE_RATE)
}

fn samples_to_ms(samples: usize, sample_rate: u32) -> f32 {
    (samples as f32 * 1000.0) / sample_rate as f32
}

fn ms_to_samples(ms: u32, sample_rate: u32) -> usize {
    ((sample_rate as u64 * ms as u64) / 1000) as usize
}

fn ms_float_to_sample(ms: f32, sample_rate: u32) -> usize {
    ((ms.max(0.0) * sample_rate as f32) / 1000.0).round() as usize
}

fn apply_padding(regions: &mut [SpeechRegion], total_samples: usize, pad_samples: usize) {
    for region in regions {
        region.start_sample = region.start_sample.saturating_sub(pad_samples);
        region.end_sample = region
            .end_sample
            .saturating_add(pad_samples)
            .min(total_samples);
    }
}

fn merge_regions(regions: &[SpeechRegion], merge_gap_samples: usize) -> Vec<SpeechRegion> {
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

struct StreamingLinearResampler {
    src_rate: u32,
    dst_rate: u32,
    input: Vec<f32>,
    next_src_pos: f64,
}

impl StreamingLinearResampler {
    fn new(src_rate: u32, dst_rate: u32) -> Self {
        Self {
            src_rate,
            dst_rate,
            input: Vec::new(),
            next_src_pos: 0.0,
        }
    }

    fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.input
            .extend(samples.iter().map(|s| sanitize_sample(*s)));
        if self.input.len() < 2 {
            return Vec::new();
        }

        let step = self.src_rate as f64 / self.dst_rate as f64;
        let mut out = Vec::new();
        while self.next_src_pos + 1.0 < self.input.len() as f64 {
            let src_idx = self.next_src_pos.floor() as usize;
            let frac = (self.next_src_pos - src_idx as f64) as f32;
            out.push(sanitize_sample(
                self.input[src_idx] * (1.0 - frac) + self.input[src_idx + 1] * frac,
            ));
            self.next_src_pos += step;
        }

        let drain = self.next_src_pos.floor() as usize;
        if drain > 0 {
            self.input.drain(0..drain);
            self.next_src_pos -= drain as f64;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(index: usize, score: f32) -> VadFrame {
        let start_vad_sample = index * VAD_FRAME_SAMPLES;
        let end_vad_sample = start_vad_sample + VAD_FRAME_SAMPLES;
        VadFrame {
            index,
            score,
            start_vad_sample,
            end_vad_sample,
            start_ms: vad_sample_to_ms(start_vad_sample),
            end_ms: vad_sample_to_ms(end_vad_sample),
        }
    }

    #[test]
    fn scorer_emits_finite_silence_scores() {
        let samples = vec![0.0f32; VAD_SAMPLE_RATE as usize];
        let frames = VadScorer::score_complete_f32(&samples, VAD_SAMPLE_RATE).unwrap();

        assert_eq!(frames.len(), samples.len() / VAD_FRAME_SAMPLES);
        assert!(frames
            .iter()
            .all(|frame| frame.score.is_finite() && (0.0..=1.0).contains(&frame.score)));
    }

    #[test]
    fn endpoint_detector_uses_hysteresis_and_silence_timeout() {
        let mut detector = EndpointDetector::new(EndpointConfig {
            start_threshold: 0.5,
            end_threshold: 0.35,
            min_speech_ms: 32,
            silence_ms: 32,
            max_utterance_ms: 10_000,
        });

        let first = detector.process_score(0.6, 16.0);
        assert_eq!(first.events, vec![EndpointEvent::SpeechStart]);
        assert!(first.is_speech);

        let held = detector.process_score(0.4, 16.0);
        assert!(held.is_speech);
        assert!(held.events.is_empty());

        let quiet = detector.process_score(0.1, 16.0);
        assert!(!quiet.is_speech);
        assert!(quiet.events.is_empty());

        let ended = detector.process_score(0.1, 16.0);
        assert_eq!(
            ended.events,
            vec![EndpointEvent::SpeechEnd(EndpointEndReason::Silence)]
        );
        assert!(!detector.is_active());
    }

    #[test]
    fn endpoint_detector_rejects_too_short_speech_on_finish() {
        let mut detector = EndpointDetector::new(EndpointConfig {
            min_speech_ms: 100,
            ..EndpointConfig::default()
        });

        detector.process_score(0.9, 16.0);
        assert_eq!(detector.finish(), Some(EndpointEvent::NoiseRejected));
        assert!(!detector.is_active());
    }

    #[test]
    fn endpoint_detector_rejects_too_short_speech_after_silence() {
        let mut detector = EndpointDetector::new(EndpointConfig {
            start_threshold: 0.5,
            end_threshold: 0.35,
            min_speech_ms: 100,
            silence_ms: 32,
            max_utterance_ms: 10_000,
        });

        detector.process_score(0.8, 16.0);
        let quiet = detector.process_score(0.0, 16.0);
        assert!(quiet.events.is_empty());
        let rejected = detector.process_score(0.0, 16.0);

        assert_eq!(rejected.events, vec![EndpointEvent::NoiseRejected]);
        assert!(!detector.is_active());
    }

    #[test]
    fn regions_from_scores_apply_min_speech_padding_and_merge() {
        let mut frames = Vec::new();
        for idx in 0..250 {
            let score = if (20..50).contains(&idx) || (65..90).contains(&idx) {
                0.8
            } else {
                0.0
            };
            frames.push(frame(idx, score));
        }
        let config = VadRegionConfig {
            min_speech_ms: 100,
            min_silence_ms: 80,
            speech_pad_ms: 20,
            merge_gap_ms: 400,
            ..VadRegionConfig::default()
        };

        let regions = detect_speech_regions_from_frames(&frames, 64_000, VAD_SAMPLE_RATE, &config);

        assert_eq!(regions.len(), 1);
        assert!(regions[0].start_sample < 20 * VAD_FRAME_SAMPLES);
        assert!(regions[0].end_sample > 90 * VAD_FRAME_SAMPLES);
    }

    #[test]
    fn mask_from_regions_marks_overlapping_frames() {
        let regions = vec![SpeechRegion {
            start_sample: 100,
            end_sample: 260,
        }];

        let mask = mask_from_regions(&regions, 4, 100, 400);

        assert_eq!(mask, vec![false, true, true, false]);
    }
}
