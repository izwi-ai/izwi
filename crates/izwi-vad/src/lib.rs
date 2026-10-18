#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::fmt;

mod gate;
#[cfg(any(test, feature = "test-signals"))]
pub mod test_signals;

pub use gate::{NoiseGate, NoiseGateConfig, DEFAULT_NOISE_GATE_LOOKAHEAD_MS};

pub const VAD_SAMPLE_RATE: u32 = 16_000;
pub const VAD_FRAME_SAMPLES: usize = 256;
pub const VAD_FRAME_MS: f32 = 16.0;
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.5;
pub const DEFAULT_EXIT_THRESHOLD: f32 = 0.35;
pub const DEFAULT_MIN_SPEECH_MS: u32 = 300;
pub const DEFAULT_SILENCE_MS: u32 = 900;
pub const DEFAULT_MAX_UTTERANCE_MS: u32 = 20_000;
pub const DEFAULT_PRE_ROLL_MS: u32 = 160;
/// How far a frame's energy must rise above the tracked noise floor to count
/// as speech. Earshot normalizes every frame by its own RMS, so it cannot tell
/// steady background noise from a voice by level; this margin can.
pub const DEFAULT_SNR_MARGIN_DB: f32 = 6.0;
/// SNR margin that keeps an active utterance alive.
pub const DEFAULT_CONTINUE_SNR_MARGIN_DB: f32 = 3.0;
/// The noise floor is the quietest smoothed frame energy over this window.
pub const NOISE_FLOOR_WINDOW_MS: u32 = 3_000;
/// The no-speech gate looks for speech evidence inside windows this long.
pub const DEFAULT_EVIDENCE_WINDOW_MS: u32 = 1_000;
/// Speech-qualified frames one evidence window must hold for an input to
/// count as speech.
pub const DEFAULT_MIN_EVIDENCE_MS: u32 = 200;
/// Frame energy of digital silence; the floor of every energy reading.
pub const MIN_ENERGY_DBFS: f32 = -120.0;
const NOISE_FLOOR_SMOOTHING: f32 = 0.2;
const MAX_SNR_MARGIN_DB: f32 = 40.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadFrame {
    pub index: usize,
    pub score: f32,
    pub start_vad_sample: usize,
    pub end_vad_sample: usize,
    pub start_ms: f32,
    pub end_ms: f32,
    /// Mean frame energy in dBFS, never below [`MIN_ENERGY_DBFS`].
    pub energy_dbfs: f32,
    /// Frame energy above the tracked noise floor, in dB. Reacts within one
    /// frame, so it decides where speech starts.
    pub snr_db: f32,
    /// Smoothed (about 80 ms) energy above the noise floor, in dB. Steadier
    /// than [`Self::snr_db`] on noise, so it decides whether speech goes on.
    pub smoothed_snr_db: f32,
}

impl VadFrame {
    pub fn duration_ms(&self) -> f32 {
        self.end_ms - self.start_ms
    }

    /// Whether this frame clears both the model score threshold and the SNR
    /// margin above the noise floor.
    pub fn is_speech(&self, threshold: f32, snr_margin_db: f32) -> bool {
        sanitize_score(self.score) >= threshold && self.snr_db >= snr_margin_db
    }
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
    noise_floor: NoiseFloorTracker,
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
            noise_floor: NoiseFloorTracker::new(),
            vad_buffer: Vec::with_capacity(VAD_FRAME_SAMPLES * 2),
            next_frame_index: 0,
            processed_vad_samples: 0,
        }
    }

    pub fn reset(&mut self) {
        self.detector.reset();
        self.resampler = None;
        self.noise_floor.reset();
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
                self.noise_floor.reset();
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
        for frame in self.vad_buffer.as_chunks::<VAD_FRAME_SAMPLES>().0 {
            let score = self.detector.predict_f32(frame).clamp(0.0, 1.0);
            let energy_dbfs = frame_energy_dbfs(frame);
            let (smoothed_dbfs, floor_dbfs) = self.noise_floor.push(energy_dbfs);
            let snr_db = energy_dbfs - floor_dbfs;
            let smoothed_snr_db = smoothed_dbfs - floor_dbfs;
            let start_vad_sample = self.processed_vad_samples;
            let end_vad_sample = start_vad_sample + VAD_FRAME_SAMPLES;
            frames.push(VadFrame {
                index: self.next_frame_index,
                score,
                start_vad_sample,
                end_vad_sample,
                start_ms: vad_sample_to_ms(start_vad_sample),
                end_ms: vad_sample_to_ms(end_vad_sample),
                energy_dbfs,
                snr_db,
                smoothed_snr_db,
            });
            self.next_frame_index += 1;
            self.processed_vad_samples = end_vad_sample;
        }

        let consumed = frame_count * VAD_FRAME_SAMPLES;
        self.vad_buffer.drain(0..consumed);
        frames
    }
}

/// Causal minimum-statistics noise floor: the quietest EMA-smoothed frame
/// energy over the last [`NOISE_FLOOR_WINDOW_MS`]. Steady noise sits on its
/// own floor (SNR near 0 dB) while speech rises well above it at any input
/// level. Before the window fills, the floor is the quietest frame so far.
struct NoiseFloorTracker {
    smoothed_dbfs: Option<f32>,
    /// Monotonic deque of `(frame index, smoothed energy)` holding the
    /// window minimum at its front.
    window: VecDeque<(usize, f32)>,
    frames: usize,
}

impl NoiseFloorTracker {
    const WINDOW_FRAMES: usize =
        (NOISE_FLOOR_WINDOW_MS as usize * VAD_SAMPLE_RATE as usize) / (1000 * VAD_FRAME_SAMPLES);

    fn new() -> Self {
        Self {
            smoothed_dbfs: None,
            window: VecDeque::with_capacity(Self::WINDOW_FRAMES),
            frames: 0,
        }
    }

    fn reset(&mut self) {
        self.smoothed_dbfs = None;
        self.window.clear();
        self.frames = 0;
    }

    /// Track one frame's energy and return `(smoothed energy, floor)`.
    fn push(&mut self, energy_dbfs: f32) -> (f32, f32) {
        let smoothed = match self.smoothed_dbfs {
            Some(previous) => previous + NOISE_FLOOR_SMOOTHING * (energy_dbfs - previous),
            None => energy_dbfs,
        };
        self.smoothed_dbfs = Some(smoothed);
        let index = self.frames;
        self.frames += 1;

        while self
            .window
            .back()
            .is_some_and(|(_, value)| *value >= smoothed)
        {
            self.window.pop_back();
        }
        self.window.push_back((index, smoothed));
        while self
            .window
            .front()
            .is_some_and(|(start, _)| start + Self::WINDOW_FRAMES <= index)
        {
            self.window.pop_front();
        }
        let floor = self.window.front().map_or(smoothed, |(_, floor)| *floor);
        (smoothed, floor)
    }
}

fn frame_energy_dbfs(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return MIN_ENERGY_DBFS;
    }
    let power = frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32;
    (10.0 * (power + 1e-12).log10()).max(MIN_ENERGY_DBFS)
}

#[derive(Debug, Clone)]
pub struct EndpointConfig {
    pub start_threshold: f32,
    pub end_threshold: f32,
    pub min_speech_ms: u32,
    pub silence_ms: u32,
    pub max_utterance_ms: u32,
    /// A frame can start an utterance only when its energy is at least this
    /// far above the noise floor. `0` disables the check.
    pub snr_margin_db: f32,
    /// A frame keeps an active utterance alive only when its smoothed energy
    /// is at least this far above the noise floor. Lower than the start
    /// margin, so the softer tail of speech under noise is not cut off.
    pub continue_snr_margin_db: f32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            start_threshold: DEFAULT_SPEECH_THRESHOLD,
            end_threshold: DEFAULT_EXIT_THRESHOLD,
            min_speech_ms: DEFAULT_MIN_SPEECH_MS,
            silence_ms: DEFAULT_SILENCE_MS,
            max_utterance_ms: DEFAULT_MAX_UTTERANCE_MS,
            snr_margin_db: DEFAULT_SNR_MARGIN_DB,
            continue_snr_margin_db: DEFAULT_CONTINUE_SNR_MARGIN_DB,
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
        self.snr_margin_db = sanitize_snr_margin(self.snr_margin_db);
        self.continue_snr_margin_db =
            sanitize_snr_margin(self.continue_snr_margin_db).min(self.snr_margin_db);
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

    /// Advance on one scored frame, holding it to the score threshold and the
    /// SNR margin.
    pub fn process_frame(&mut self, frame: &VadFrame) -> EndpointDecision {
        let snr_db = if self.active {
            frame.smoothed_snr_db
        } else {
            frame.snr_db
        };
        self.process(frame.score, snr_db, frame.duration_ms())
    }

    /// Advance on a bare model score, with no SNR check. For callers and
    /// tests that have a score but no audio.
    pub fn process_score(&mut self, score: f32, frame_ms: f32) -> EndpointDecision {
        self.process(score, f32::INFINITY, frame_ms)
    }

    fn process(&mut self, score: f32, snr_db: f32, frame_ms: f32) -> EndpointDecision {
        let frame_ms = frame_ms.max(0.0);
        let (threshold, snr_margin_db) = if self.active {
            (
                self.config.end_threshold,
                self.config.continue_snr_margin_db,
            )
        } else {
            (self.config.start_threshold, self.config.snr_margin_db)
        };
        let is_speech = sanitize_score(score) >= threshold && snr_db >= snr_margin_db;
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
    /// A region may only start on a frame at least this far above the noise
    /// floor. Continuation is held to the score alone, so speech under noise
    /// is not cut mid-region. `0` (the default) disables the check.
    pub start_snr_margin_db: f32,
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
            start_snr_margin_db: 0.0,
        }
    }
}

impl VadRegionConfig {
    pub fn sanitized(mut self) -> Self {
        self.start_threshold = sanitize_score_threshold(self.start_threshold);
        self.end_threshold = sanitize_score_threshold(self.end_threshold).min(self.start_threshold);
        self.min_speech_ms = self.min_speech_ms.max(1);
        self.min_silence_ms = self.min_silence_ms.max(1);
        self.start_snr_margin_db = sanitize_snr_margin(self.start_snr_margin_db);
        self
    }
}

/// Settings for [`SpeechEvidence`].
#[derive(Debug, Clone)]
pub struct SpeechEvidenceConfig {
    /// Model score a frame must reach.
    pub threshold: f32,
    /// Energy above the noise floor a frame must reach, in dB.
    pub snr_margin_db: f32,
    /// Length of the sliding window evidence is counted in.
    pub window_ms: u32,
    /// Speech-qualified frames one window must hold.
    pub min_evidence_ms: u32,
}

impl Default for SpeechEvidenceConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_SPEECH_THRESHOLD,
            snr_margin_db: configured_snr_margin_db(),
            window_ms: DEFAULT_EVIDENCE_WINDOW_MS,
            min_evidence_ms: DEFAULT_MIN_EVIDENCE_MS,
        }
    }
}

impl SpeechEvidenceConfig {
    pub fn sanitized(mut self) -> Self {
        self.threshold = sanitize_score_threshold(self.threshold);
        self.snr_margin_db = sanitize_snr_margin(self.snr_margin_db);
        self.window_ms = self.window_ms.max(VAD_FRAME_MS as u32);
        self.min_evidence_ms = self.min_evidence_ms.clamp(1, self.window_ms);
        self
    }
}

/// Whether a stream of scored frames holds speech: some sliding window must
/// contain at least `min_evidence_ms` of frames that clear both the score
/// threshold and the SNR margin. A single speech-like frame, which a
/// level-blind model gives steady noise now and then, is not enough.
#[derive(Debug, Clone)]
pub struct SpeechEvidence {
    config: SpeechEvidenceConfig,
    /// `(start_ms, speech-qualified duration)` of the frames in the window.
    window: VecDeque<(f32, f32)>,
    window_speech_ms: f32,
    best_window_ms: f32,
    total_ms: f32,
    peak_score: f32,
    frames: usize,
}

impl SpeechEvidence {
    pub fn new(config: SpeechEvidenceConfig) -> Self {
        Self {
            config: config.sanitized(),
            window: VecDeque::new(),
            window_speech_ms: 0.0,
            best_window_ms: 0.0,
            total_ms: 0.0,
            peak_score: 0.0,
            frames: 0,
        }
    }

    pub fn config(&self) -> &SpeechEvidenceConfig {
        &self.config
    }

    /// Count one frame. Returns whether one window already holds the full
    /// `min_evidence_ms`, which settles the question however long the input
    /// turns out to be.
    pub fn push(&mut self, frame: &VadFrame) -> bool {
        let duration_ms = frame.duration_ms().max(0.0);
        let speech_ms = if frame.is_speech(self.config.threshold, self.config.snr_margin_db) {
            duration_ms
        } else {
            0.0
        };
        self.window.push_back((frame.start_ms, speech_ms));
        self.window_speech_ms += speech_ms;
        let window_start_ms = frame.end_ms - self.config.window_ms as f32;
        while self
            .window
            .front()
            .is_some_and(|(start_ms, _)| *start_ms < window_start_ms)
        {
            if let Some((_, speech_ms)) = self.window.pop_front() {
                self.window_speech_ms -= speech_ms;
            }
        }
        self.best_window_ms = self.best_window_ms.max(self.window_speech_ms);
        self.total_ms += duration_ms;
        self.peak_score = self.peak_score.max(sanitize_score(frame.score));
        self.frames += 1;
        self.best_window_ms >= self.config.min_evidence_ms as f32
    }

    /// Evidence one window must hold, judged on everything pushed so far.
    /// An input shorter than one window needs proportionally less, so a short
    /// clip is never held to evidence it cannot contain. Only meaningful once
    /// the whole input is pushed: mid-stream, use [`Self::push`]'s result.
    pub fn required_ms(&self) -> f32 {
        let window_ms = self.config.window_ms as f32;
        let coverage = (self.total_ms / window_ms).clamp(0.0, 1.0);
        self.config.min_evidence_ms as f32 * coverage
    }

    /// Final verdict once the whole input is pushed.
    pub fn is_sufficient(&self) -> bool {
        self.best_window_ms > 0.0 && self.best_window_ms >= self.required_ms()
    }

    /// Most speech-qualified time seen in any one window.
    pub fn best_window_ms(&self) -> f32 {
        self.best_window_ms
    }

    pub fn peak_score(&self) -> f32 {
        self.peak_score
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn total_ms(&self) -> f32 {
        self.total_ms
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
            if frame.is_speech(config.start_threshold, config.start_snr_margin_db) {
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

/// The deployment's SNR margin: `IZWI_VAD_SNR_MARGIN_DB` when set to a
/// number, otherwise [`DEFAULT_SNR_MARGIN_DB`]. `0` turns the noise-floor
/// check off everywhere, leaving the model score alone to decide. Read once.
pub fn configured_snr_margin_db() -> f32 {
    static MARGIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *MARGIN.get_or_init(|| {
        std::env::var("IZWI_VAD_SNR_MARGIN_DB")
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .map_or(DEFAULT_SNR_MARGIN_DB, sanitize_snr_margin)
    })
}

/// SNR margins are clamped to `0..=40` dB; a non-finite margin falls back to
/// the default.
pub fn sanitize_snr_margin(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, MAX_SNR_MARGIN_DB)
    } else {
        DEFAULT_SNR_MARGIN_DB
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

        let drain = (self.next_src_pos.floor() as usize).min(self.input.len());
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
            energy_dbfs: -30.0,
            snr_db: 20.0,
            smoothed_snr_db: 20.0,
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
    fn scorer_accepts_browser_sized_48khz_chunks() {
        let mut scorer = VadScorer::new();
        let samples = vec![0i16; 2048];

        let first = scorer.push_i16(&samples, 48_000).unwrap();
        let second = scorer.push_i16(&samples, 48_000).unwrap();

        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 3);
        assert!(first
            .iter()
            .chain(second.iter())
            .all(|frame| frame.score.is_finite() && (0.0..=1.0).contains(&frame.score)));
    }

    #[test]
    fn streaming_resampler_caps_drain_for_exact_ratio_chunks() {
        let mut resampler = StreamingLinearResampler::new(48_000, VAD_SAMPLE_RATE);
        let samples = vec![0.0f32; 2048];

        let first = resampler.push(&samples);
        let second = resampler.push(&samples);

        assert_eq!(first.len(), 683);
        assert_eq!(second.len(), 682);
    }

    #[test]
    fn endpoint_detector_uses_hysteresis_and_silence_timeout() {
        let mut detector = EndpointDetector::new(EndpointConfig {
            start_threshold: 0.5,
            end_threshold: 0.35,
            min_speech_ms: 32,
            silence_ms: 32,
            max_utterance_ms: 10_000,
            ..EndpointConfig::default()
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
            ..EndpointConfig::default()
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

    mod noise {
        use super::super::test_signals::*;
        use super::super::*;
        use std::path::Path;

        const SR: usize = VAD_SAMPLE_RATE as usize;

        /// A repository speech fixture as 16 kHz mono.
        fn fixture(name: &str) -> Vec<f32> {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../data")
                .join(name);
            let mut reader = hound::WavReader::open(&path).expect("speech fixture");
            let rate = reader.spec().sample_rate;
            let samples: Vec<f32> = reader
                .samples::<i16>()
                .map(|sample| sample.expect("16-bit sample") as f32 / 32768.0)
                .collect();
            resample_linear_f32(&samples, rate, VAD_SAMPLE_RATE).expect("resample")
        }

        fn frames(samples: &[f32]) -> Vec<VadFrame> {
            VadScorer::score_complete_f32(samples, VAD_SAMPLE_RATE).expect("score")
        }

        fn has_speech(samples: &[f32]) -> bool {
            let mut evidence = SpeechEvidence::new(SpeechEvidenceConfig::default());
            for frame in frames(samples) {
                evidence.push(&frame);
            }
            evidence.is_sufficient()
        }

        fn rms(samples: &[f32]) -> f32 {
            (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32)
                .sqrt()
        }

        /// Noise at speech-like levels, which earshot alone scores as speech
        /// on some or most frames.
        fn loud_noises() -> Vec<(String, Vec<f32>)> {
            let len = SR * 10;
            let mut noises = Vec::new();
            for peak in [0.05, 0.1, 0.3, 0.8] {
                noises.push((format!("pink {peak}"), pink_noise(len, peak, 4)));
            }
            for peak in [0.05, 0.2] {
                noises.push((format!("white {peak}"), white_noise(len, peak, 4)));
            }
            noises.push(("brown 0.3".into(), brown_noise(len, 0.3, 4)));
            noises.push(("hum 0.3".into(), mains_hum(len, 0.3)));
            noises.push(("fan 0.3".into(), fan_noise(len, 0.3, 4)));
            noises.push(("keyboard 0.5".into(), keyboard_clicks(len, 0.5, 4)));
            noises
        }

        #[test]
        fn steady_noise_sits_on_its_own_floor() {
            for (name, samples) in loud_noises() {
                // Clicks and brown noise's drifting level rise over any floor
                // by design; the evidence rule handles them.
                if name.starts_with("keyboard") || name.starts_with("brown") {
                    continue;
                }
                let frames = frames(&samples);
                let settled = &frames[NOISE_FLOOR_WINDOW_MS as usize / 16..];
                // Single frames fluctuate over the start margin now and then;
                // the evidence rule needs 13 of them inside one second.
                let raw = settled
                    .iter()
                    .filter(|frame| frame.snr_db >= DEFAULT_SNR_MARGIN_DB)
                    .count();
                assert!(
                    raw * 10 < settled.len(),
                    "{name}: {raw}/{} frames 6 dB over the floor",
                    settled.len()
                );
                // Typical noise frames sit within the continuation margin of
                // their own floor.
                let mut smoothed: Vec<f32> =
                    settled.iter().map(|frame| frame.smoothed_snr_db).collect();
                smoothed.sort_by(f32::total_cmp);
                let median = smoothed[smoothed.len() / 2];
                assert!(
                    median < DEFAULT_CONTINUE_SNR_MARGIN_DB,
                    "{name}: median smoothed SNR {median} dB"
                );
            }
        }

        #[test]
        fn digital_silence_has_finite_energy_and_no_speech() {
            let silence = vec![0.0; SR * 2];
            let frames = frames(&silence);
            assert!(frames
                .iter()
                .all(|frame| frame.energy_dbfs == MIN_ENERGY_DBFS && frame.snr_db == 0.0));
            assert!(!has_speech(&silence));
        }

        #[test]
        fn loud_noise_holds_no_speech_evidence() {
            for (name, samples) in loud_noises() {
                assert!(!has_speech(&samples), "{name} must hold no speech evidence");
            }
        }

        #[test]
        fn speech_holds_evidence_at_any_level_and_down_to_zero_db_snr() {
            for name in ["fox.wav", "test.wav"] {
                let speech = fixture(name);
                for db in [0.0, -20.0, -40.0] {
                    assert!(has_speech(&attenuated(&speech, db)), "{name} at {db} dB");
                }
                for (noise_name, noise) in [
                    ("pink", pink_noise(speech.len(), 0.5, 9)),
                    ("white", white_noise(speech.len(), 0.5, 9)),
                    ("fan", fan_noise(speech.len(), 0.5, 9)),
                    ("hum", mains_hum(speech.len(), 0.5)),
                ] {
                    for snr in [10.0, 0.0] {
                        assert!(
                            has_speech(&mix_at_snr(&speech, &noise, snr)),
                            "{name} + {noise_name} at {snr} dB SNR"
                        );
                    }
                }
            }
        }

        #[test]
        fn one_speech_like_frame_is_not_evidence() {
            let mut evidence = SpeechEvidence::new(SpeechEvidenceConfig::default());
            for index in 0..625 {
                let score = if index == 300 { 0.99 } else { 0.0 };
                assert!(!evidence.push(&super::frame(index, score)));
            }
            assert!(!evidence.is_sufficient());
            assert_eq!(evidence.best_window_ms(), VAD_FRAME_MS);
        }

        #[test]
        fn short_input_needs_proportional_evidence() {
            let mut evidence = SpeechEvidence::new(SpeechEvidenceConfig::default());
            // 320 ms of input with 80 ms of speech: 64 ms required.
            for index in 0..20 {
                evidence.push(&super::frame(index, if index < 5 { 0.9 } else { 0.0 }));
            }
            assert_eq!(evidence.required_ms(), 64.0);
            assert!(evidence.is_sufficient());
        }

        #[test]
        fn evidence_window_slides() {
            let mut evidence = SpeechEvidence::new(SpeechEvidenceConfig::default());
            // 7 speech frames every 2 s: never 13 frames inside one second.
            for index in 0..625 {
                let score = if index % 125 < 7 { 0.9 } else { 0.0 };
                assert!(!evidence.push(&super::frame(index, score)));
            }
            assert_eq!(evidence.best_window_ms(), 7.0 * VAD_FRAME_MS);
            // 13 consecutive frames settle it mid-stream.
            for index in 625..638 {
                evidence.push(&super::frame(index, 0.9));
            }
            assert!(evidence.is_sufficient());
        }

        /// Fox spoken 3 s into 12 s of noise at 10 dB SNR: endpointing yields
        /// one turn around the speech and none on the noise.
        #[test]
        fn endpointing_ignores_steady_noise_and_keeps_speech_under_it() {
            let fox = fixture("fox.wav");
            let len = SR * 12;
            for (name, noise) in [
                ("hum", mains_hum(len, 0.5)),
                ("fan", fan_noise(len, 0.5, 7)),
                ("white", white_noise(len, 0.5, 7)),
                ("pink", pink_noise(len, 0.5, 7)),
            ] {
                let gain = rms(&fox) / rms(&noise) * 10f32.powf(-10.0 / 20.0);
                let mut samples: Vec<f32> = noise.iter().map(|sample| sample * gain).collect();
                for (idx, sample) in fox.iter().enumerate() {
                    samples[SR * 3 + idx] += sample;
                }

                let mut detector = EndpointDetector::new(EndpointConfig::default());
                let mut starts = Vec::new();
                let mut ends = Vec::new();
                for frame in frames(&samples) {
                    for event in detector.process_frame(&frame).events {
                        match event {
                            EndpointEvent::SpeechStart => starts.push(frame.start_ms),
                            EndpointEvent::SpeechEnd(reason) => ends.push((frame.end_ms, reason)),
                            EndpointEvent::NoiseRejected => {}
                        }
                    }
                }
                assert_eq!(
                    ends.len(),
                    1,
                    "{name}: one turn, got {ends:?} from {starts:?}"
                );
                assert_eq!(ends[0].1, EndpointEndReason::Silence, "{name}");
                let turn_start = starts
                    .iter()
                    .rev()
                    .copied()
                    .find(|start| *start < ends[0].0)
                    .expect("turn start");
                // Fox speech spans about 3240-6330 ms here.
                assert!(
                    (3_000.0..3_400.0).contains(&turn_start),
                    "{name}: start {turn_start}"
                );
                assert!(
                    (6_300.0..8_000.0).contains(&ends[0].0),
                    "{name}: end {:?}",
                    ends[0]
                );
            }

            // The noise alone never produces a turn.
            let mut detector = EndpointDetector::new(EndpointConfig::default());
            for frame in frames(&fan_noise(len, 0.3, 7)) {
                let events = detector.process_frame(&frame).events;
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, EndpointEvent::SpeechEnd(_))),
                    "fan noise produced a turn"
                );
            }
        }

        #[test]
        fn region_start_margin_skips_noise_but_keeps_speech() {
            let config = VadRegionConfig {
                start_snr_margin_db: DEFAULT_SNR_MARGIN_DB,
                ..VadRegionConfig::default()
            };
            let fan = fan_noise(SR * 10, 0.3, 4);
            let regions = detect_speech_regions_f32(&fan, VAD_SAMPLE_RATE, &config).unwrap();
            assert!(regions.is_empty(), "fan regions {regions:?}");
            let unchecked =
                detect_speech_regions_f32(&fan, VAD_SAMPLE_RATE, &VadRegionConfig::default())
                    .unwrap();
            assert!(
                !unchecked.is_empty(),
                "earshot alone hears speech in fan noise"
            );

            let fox = fixture("fox.wav");
            let mixed = mix_at_snr(&fox, &fan_noise(fox.len(), 0.5, 9), 10.0);
            let regions = detect_speech_regions_f32(&mixed, VAD_SAMPLE_RATE, &config).unwrap();
            let covered: usize = regions.iter().map(|region| region.len_samples()).sum();
            assert!(covered * 10 > fox.len() * 7, "{covered} of {}", fox.len());
        }

        #[test]
        fn snr_margins_are_sanitized() {
            let config = EndpointConfig {
                snr_margin_db: 4.0,
                continue_snr_margin_db: 9.0,
                ..EndpointConfig::default()
            }
            .sanitized();
            assert_eq!(config.continue_snr_margin_db, 4.0);
            assert_eq!(sanitize_snr_margin(f32::NAN), DEFAULT_SNR_MARGIN_DB);
            assert_eq!(sanitize_snr_margin(-3.0), 0.0);
            assert_eq!(sanitize_snr_margin(90.0), 40.0);
        }
    }
}
